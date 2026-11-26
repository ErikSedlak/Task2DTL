"""Detector tracks: cached Grounding DINO + SAM 2 detections -> 3D object instances (CHANGE 3 Part E).

Used by `PerceptionSource` when `perception.detector.enabled`. Per processed frame that has cached
detections (`detect_cache.DetectionCache`, every `frame_every`-th processed frame):

1. every detection with score >= `min_score`: mask eroded by `erode_px`, valid depth <= `max_depth`,
   depth outliers dropped (|d - median| > max(`depth_mad_k` * 1.4826 * MAD, 5 cm): background seen
   through or past the object's silhouette), back-projected with the frame's aligned pose, points
   near reference structure dropped (`SurfaceFilter.keep_struct`: wall bleed), voxelised (`voxel`);
2. **fusion**: detections in score order join the track holding >= `join_frac` of their voxels (one
   detection per track and frame), else they start a new track `d<k>`. A track keeps its voxels
   (mean point each), per-frame pixel masks, decorrelated views (the `Decorrelator` rule, per track)
   and the summed per-label detection scores.

`instances()` (both `finish` and `peek`): tracks whose voxel sets overlap by >= `merge_frac` of the smaller
one are merged (the older id stays); tracks with >= `min_views` decorrelated views and >= `min_voxels`
voxels outside door swing regions become `ObsInstance(kind="detector")`:
- `label_topk`: the summed label scores, normalised (or CLIP on the crops, `labels: clip`, in the source);
- `newness` = share of the track's voxels farther than `surf_dist` from the reference mesh: a re-detected
  unchanged object lies on the reference surface (~0), a new or moved one does not (~1).
  `change.baselines.emits_new` turns only instances with newness >= `p_new_min` into `new` nodes.
- `confidence` stays None: a matched / new node's presence comes from its views (as in oracle mode).

Method side: rescan colour (via the cache) and depth frames, poses, the reference mesh. Never imports
`oracle` / `eval`, never reads rescan annotations (hard rules 1-2).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lsg.geom.transforms import apply
from lsg.obs.base import PackedMasks
from lsg.perception.cluster import Candidate, drop_door_clusters
from lsg.perception.voxels import pack_keys


@dataclass
class Track:
    """One fused detector track."""

    tid: int
    keys: list[np.ndarray] = field(default_factory=list)  # voxel keys per detection
    pts: list[np.ndarray] = field(default_factory=list)  # points per detection
    keyset: set = field(default_factory=set)
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> bool (H, W)
    views: int = 0
    n_det: int = 0
    label_sum: np.ndarray | None = None


def depth_inliers(z: np.ndarray, k: float, floor: float = 0.05) -> np.ndarray:
    """Pixels whose depth is within max(k * 1.4826 * MAD, floor) of the median."""
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med))) * 1.4826
    return np.abs(z - med) <= max(k * mad, floor)


def backproject(pix: np.ndarray, z: np.ndarray, K: np.ndarray, pose: np.ndarray, W: int) -> np.ndarray:
    """Pixel indices (row-major, width W) with depth z -> points in the pose's world frame."""
    v, u = np.divmod(pix, W)
    return apply(pose, np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1))


class DetectorTracks:
    """Fuses per-frame detections into 3D tracks (module docstring)."""

    def __init__(self, cache, filt, cfg: Mapping[str, Any], K: np.ndarray, wh: tuple[int, int], doors=()):
        """
        Args:
            cache: `detect_cache.DetectionCache` of the rescan (or any object with `.frame(fidx)` and `.prompts`).
            filt: `residual.SurfaceFilter` of the reference (structure test, reference mesh distance).
            cfg: Full config (`perception`, `perception.detector`, `decorrelation`).
            K: Depth intrinsics; wh: depth (W, H).
            doors: `cluster.door_regions` of the reference.
        """
        from lsg.change.visibility import Decorrelator

        self.cache, self.filt, self.cfg = cache, filt, cfg
        self.pc = cfg["perception"]
        self.d = self.pc["detector"]
        self.K = np.asarray(K, float)
        self.W, self.H = wh
        self.doors = list(doors)
        self.voxel = float(self.pc["voxel"])
        self.dec = Decorrelator.from_config(cfg)
        self.tracks: dict[int, Track] = {}
        self.owner: dict[int, set] = {}  # voxel key -> track ids
        self._next = 0
        self.stats = {"det_frames": 0, "det_detections": 0, "det_used": 0, "det_tracks": 0}

    # ---- per frame ---------------------------------------------------------------------------

    def observe(self, fidx: int, depth: np.ndarray, pose: np.ndarray) -> None:
        """Add the cached detections of frame `fidx` (no-op if the frame has none cached)."""
        det = self.cache.frame(int(fidx))
        if det is None:
            return
        self.stats["det_frames"] += 1
        self.stats["det_detections"] += int(len(det["scores"]))
        items = self._lift(det, np.asarray(depth, np.float64), np.asarray(pose, float))
        used: set[int] = set()
        for it in sorted(items, key=lambda x: -x["score"]):
            cnt = Counter(t for k in it["uniq"] for t in self.owner.get(int(k), ()))
            best, best_n = None, float(self.d.get("join_frac", 0.3)) * len(it["uniq"])
            for t, n in cnt.most_common():
                if t in used:
                    continue
                if n >= best_n:
                    best = t
                break
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = Track(best)
            used.add(best)
            self._add(self.tracks[best], it, int(fidx), pose)
        self.stats["det_used"] += len(items)

    def _lift(self, det: Mapping[str, np.ndarray], depth: np.ndarray, pose: np.ndarray) -> list[dict[str, Any]]:
        from scipy.ndimage import binary_erosion

        out = []
        min_score = float(self.d.get("min_score", 0.3))
        er = int(self.d.get("erode_px", 1))
        max_d = float(self.pc["max_depth"])
        min_px = int(self.d.get("min_px", 15))
        for i, sc in enumerate(det["scores"]):
            if float(sc) < min_score:
                continue
            m = np.asarray(det["masks"][i], bool)
            if er > 0:
                m = binary_erosion(m, iterations=er)
            m &= (depth > 0) & (depth <= max_d)
            pix = np.flatnonzero(m)
            if pix.size < min_px:
                continue
            z = depth.ravel()[pix]
            ok = depth_inliers(z, float(self.d.get("depth_mad_k", 3.0)))
            pix, z = pix[ok], z[ok]
            pts = backproject(pix, z, self.K, pose, self.W)
            ok = self.filt.keep_struct(pts)
            pix, pts = pix[ok], pts[ok]
            if pix.size < min_px:
                continue
            keys = pack_keys(np.floor(pts / self.voxel).astype(np.int64))
            mk = np.zeros(self.H * self.W, bool)
            mk[pix] = True
            out.append({"score": float(sc), "keys": keys, "uniq": np.unique(keys), "pts": pts,
                        "mask": mk.reshape(self.H, self.W), "label_scores": np.asarray(det["label_scores"][i], float)})  # fmt: skip
        return out

    def _add(self, tr: Track, it: Mapping[str, Any], fidx: int, pose: np.ndarray) -> None:
        tr.keys.append(it["keys"])
        tr.pts.append(it["pts"])
        for k in it["uniq"].tolist():
            if k not in tr.keyset:
                tr.keyset.add(k)
                self.owner.setdefault(k, set()).add(tr.tid)
        tr.masks[fidx] = tr.masks[fidx] | it["mask"] if fidx in tr.masks else it["mask"]
        tr.views += int(self.dec.counts(("det", tr.tid), pose))
        tr.n_det += 1
        ls = it["label_scores"] * it["score"]
        tr.label_sum = ls if tr.label_sum is None else tr.label_sum + ls

    # ---- instances ------------------------------------------------------------------------------

    def _merged(self) -> list[list[int]]:
        """Groups of track ids whose voxel sets overlap by >= `merge_frac` of the smaller one (union-find, oldest first)."""
        frac = float(self.d.get("merge_frac", 0.5))
        parent = {t: t for t in self.tracks}

        def find(t: int) -> int:
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        for t in sorted(self.tracks):
            ks = self.tracks[t].keyset
            cnt = Counter(u for k in ks for u in self.owner.get(k, ()) if u != t)
            for u, n in cnt.items():
                if n >= frac * min(len(ks), len(self.tracks[u].keyset)):
                    a, b = find(t), find(u)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        groups: dict[int, list[int]] = {}
        for t in sorted(self.tracks):
            groups.setdefault(find(t), []).append(t)
        return [groups[g] for g in sorted(groups)]

    def instances(self, record: bool = True) -> list[dict[str, Any]]:
        """Fused instances (dicts: token, points, voxel_coords, masks, views, n_det, label_sum, newness), by token."""
        out = []
        min_views, min_vox = int(self.d.get("min_views", 1)), int(self.d.get("min_voxels", 5))
        surf = float(self.pc["surf_dist"])
        for grp in self._merged():
            trs = [self.tracks[t] for t in grp]
            keys = np.concatenate([k for tr in trs for k in tr.keys])
            pts = np.concatenate([p for tr in trs for p in tr.pts])
            uk, inv = np.unique(keys, return_inverse=True)
            if len(uk) < min_vox:
                continue
            n = np.bincount(inv, minlength=len(uk)).astype(float)
            P = np.stack([np.bincount(inv, weights=pts[:, j], minlength=len(uk)) / n for j in range(3)], axis=1)
            views = sum(tr.views for tr in trs)
            if views < min_views:
                continue
            kept, _ = drop_door_clusters([Candidate(np.arange(len(P)), P)], self.doors, float(self.pc["door_frac"]))
            if not kept:
                continue
            masks = PackedMasks((self.H, self.W))
            for f in sorted({f for tr in trs for f in tr.masks}):
                m = np.zeros((self.H, self.W), bool)
                for tr in trs:
                    if f in tr.masks:
                        m |= tr.masks[f]
                masks.add(f, m)
            dist = np.asarray(self.filt.ref.distance(P))
            ls = sum(tr.label_sum for tr in trs if tr.label_sum is not None)
            out.append({"token": f"d{grp[0]}", "points": P, "voxel_coords": np.floor(P / self.voxel).astype(np.int64),
                        "masks": masks, "views": int(views), "n_det": int(sum(tr.n_det for tr in trs)),
                        "label_sum": np.asarray(ls, float), "newness": float(np.mean(dist > surf))})  # fmt: skip
        if record:
            self.stats["det_tracks"] = len(self.tracks)
            self.stats["det_instances"] = len(out)
        return out

    def label_topk(self, label_sum: np.ndarray, k: int) -> list[tuple[str, float]]:
        """The `k` best prompt labels of a summed score vector, normalised to sum 1 over all labels."""
        s = np.asarray(label_sum, float)
        if s.size == 0 or s.sum() <= 0:
            return []
        p = s / s.sum()
        idx = np.argsort(-p, kind="stable")[:k]
        return [(str(self.cache.prompts[i]), round(float(p[i]), 6)) for i in idx]


def overlaps(points: np.ndarray, others: Sequence[np.ndarray], dist: float, frac: float) -> bool:
    """Whether >= `frac` of `points` lie within `dist` of any point of `others`."""
    if not len(others) or not len(points):
        return False
    from scipy.spatial import cKDTree

    tree = cKDTree(np.concatenate(others))
    d, _ = tree.query(points, k=1)
    return bool(np.mean(d <= dist) >= frac)
