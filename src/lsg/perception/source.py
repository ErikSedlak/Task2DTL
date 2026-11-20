"""`PerceptionSource`: rescan objects from depth residuals + CLIP labels (DESIGN §3, §6.6).

Per processed frame (`observe`, called by `change.online.run_online` with the reference render
at the frame's aligned pose): nearer residual points (steps 1-2, `residual`) go into the
decorrelated voxel grid (steps 3-4, `voxels`), and the (pixel, voxel) pairs of the frame are
kept. At the end (`finish`): kept voxels -> connected components (step 5, `cluster`) -> door
swing-region filter -> one `ObsInstance` per candidate:

- `points`: mean residual point of each of its voxels (reference frame);
- `masks`: per frame, the pixels whose residual point fell into one of its voxels;
- `crops`: upright colour crops of the `crop_k` frames with the largest masks;
- `label_topk` from the labeller (CLIP), `label` = top-1, `nyu40` from the vocabulary table;
- token `c<k>` (k = rank by size).

Relaxed candidate rules (CHANGE 1, `perception.relaxed`, docs/CHANGE_candidate_detection.md): size
decides how much evidence a candidate needs. *Strict tier* = the steps above with
`relaxed.min_views_small` (2) views at `tau_new`. *Lenient tier* = a second voxel grid filled at
`relaxed.tau_new_large` (0.05 m) with `relaxed.min_views_large` (1) view; only its clusters with
>= `large_min_voxels` voxels (and `large_min_extent`) are kept, and they replace the strict clusters
inside them. `relaxed.enabled: false` = the M7 method (`min_views` views at `tau_new`, no lenient tier).

Candidate confidence (CHANGE 2 Part B, `confidence.candidates`, docs/CHANGE_confidence_unified.md):
both depth grids carry per-voxel log-odds (`voxels`): hits raise a voxel, see-through votes of
later frames lower it (`VoxelEvidence.observe_free`, every frame on every tracked voxel). The
view-count rules become thresholds: strict voxels need `lv >= l_keep_small` (relaxed off:
`L_hit · min_views`), lenient voxels `lv >= l_keep_large`. Each depth candidate gets
`confidence = sigmoid(median lv)`; the update step drops candidates below `p_new_min` from the
new nodes (`change.baselines.emits_new`). `live_candidates()` gives the current candidates for
the snapshot sidecars. Appearance candidates are unchanged (view counts, confidence None).

Thin-object candidates (M6, DESIGN §6.6 output ii; `appearance.thin_candidates`): when the
online loop passes an `AppearanceMap` in `RefRender`, mismatching pixels whose depth agrees with
the reference (|d_obs - d_exp| <= tau_near) and whose first hit is a support surface (a non-thin
anchor / thing or the floor, face |normal z| >= `support_min_normal_z`) are back-projected into
a **separate** voxel grid (structure filter only: such objects lie within `surf_dist` of the
reference surface). The same >= `min_views` / clustering / door rules apply; clusters touching a
depth candidate are dropped (depth wins). They become `ObsInstance(kind="appearance")`, with
tokens continuing after the depth candidates.

Method side: inputs are the reference context and the rescan's RGB-D frames + poses +
refined `T_align` only. Never imports `oracle` / `eval`, never reads rescan annotations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.geom.transforms import apply, invert
from lsg.io.scan3r import Frame, open_sequence
from lsg.obs.base import ObsInstance, PackedMasks, RefContext, RefRender, SeqInfo
from lsg.perception.cluster import Candidate, cluster, door_regions, drop_door_clusters, is_large
from lsg.perception.labeler import Labeler
from lsg.perception.residual import SurfaceFilter, nearer_mask, nearer_points
from lsg.perception.voxels import VoxelEvidence, window_min_depth

UNLABELLED = "object"  # label of a candidate without usable crops


class PerceptionSource:
    """Implements `lsg.obs.ObsSource` from depth residuals against the rendered reference."""

    def __init__(
        self,
        labeler: Labeler | None,
        label_nyu: Mapping[str, int],
        cfg: Mapping[str, Any],
        with_crops: bool = True,
    ):
        """
        Args:
            labeler: Turns crops into `label_topk` (None = every candidate is `UNLABELLED`).
            label_nyu: Vocabulary label -> nyu40 (`train_meta.perception_vocab`).
            cfg: Full config (`perception`, `decorrelation`).
            with_crops: Read colour frames for crops (needs `seq_info.root` with the sequence).
        """
        self.labeler = labeler
        self.label_nyu = dict(label_nyu)
        self.cfg = cfg
        self.pc = cfg["perception"]
        self.with_crops = bool(with_crops)
        self.candidates: list[Candidate] = []
        self.dropped_door: list[Candidate] = []
        self.stats: dict[str, Any] = {}
        ac = cfg.get("appearance", {}) or {}
        self.wants_appearance = bool(ac.get("thin_candidates", False))  # read by `run_online`

    # ---- ObsSource -----------------------------------------------------------------------

    def begin(self, ref_ctx: RefContext, seq_info: SeqInfo) -> None:
        """Build the surface / structure filter, door regions and an empty voxel grid."""
        self.ctx, self.info = ref_ctx, seq_info
        self.filt = SurfaceFilter(ref_ctx, self.cfg)
        self.doors = door_regions(ref_ctx, float(self.pc["door_margin"])) if self.pc.get("door_filter", True) else []
        cc = (self.cfg.get("confidence", {}) or {}).get("candidates") or {}
        self.candconf = bool(cc.get("enabled", False))  # CHANGE 2: per-voxel log-odds
        self.free_votes = self.candconf and bool(cc.get("free_votes", True))
        self.vox = VoxelEvidence.from_config(self.cfg, logodds=self.candconf)
        rc = self.pc.get("relaxed") or {}
        self.relaxed = bool(rc.get("enabled", False))  # CHANGE 1: size-dependent candidate rules
        self.min_views_small = int(rc["min_views_small"]) if self.relaxed else int(self.pc["min_views"])
        if self.candconf:  # voxel thresholds in log-odds; without the relaxed rules: M7's min_views views
            L_hit = float(cc["L_vox_hit"])  # voxel weight (CHANGE 3: not the node weight confidence.L_hit)
            self.l_keep_small = float(cc["l_keep_small"]) if self.relaxed else L_hit * int(self.pc["min_views"])
            self.l_keep_large = float(cc["l_keep_large"])
        self.vox_large = VoxelEvidence.from_config(self.cfg, logodds=self.candconf) if self.relaxed else None
        self._frames_large: list[tuple[int, np.ndarray, np.ndarray]] = []
        W, H = seq_info.intrinsics.depth_wh
        self._wh = (W, H)
        self._frames: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.candidates, self.dropped_door = [], []
        self.stats = {"frames": 0, "nearer_px": 0, "residual_pts": 0, "voxels": 0, "kept_voxels": 0,
                      "clusters": 0, "door_dropped": 0, "doors": len(self.doors), "labelled": 0}  # fmt: skip
        if self.relaxed:  # extra keys only with the relaxed rules, so the strict path keeps its stats
            self.stats.update(large_pts=0, large_voxels=0, large_clusters=0, large_door_dropped=0, large_replaced=0)
        if self.candconf:
            self.stats.update(free_votes=0, seen_through_voxels=0, below_p_new_min=0)
            if self.relaxed:
                self.stats.update(large_free_votes=0, large_seen_through_voxels=0)
        # thin-object candidates (output ii)
        self.vox_app = VoxelEvidence.from_config(self.cfg)
        self._frames_app: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._support = np.zeros(1, bool)
        if self.wants_appearance:
            ac = self.cfg["appearance"]
            thin = ref_ctx.thin_oids(float(ac["thin_max_extent"]))
            oids = [int(o) for o in ref_ctx.scan.instances]
            self._support = np.zeros(max(oids, default=0) + 1, bool)
            for o in oids:
                n = ref_ctx.object_nyu40(o)
                if n is None:
                    continue
                if (node_level(n) in ("anchor", "thing") and o not in thin) or n in (2, 20):  # floor, floor mat
                    self._support[o] = True
        self.stats.update(app_px=0, app_pts=0, app_voxels=0, app_kept_voxels=0, app_clusters=0,
                          app_dropped_depth=0, app_door_dropped=0)  # fmt: skip

    def observe(self, frame: Frame, ref_render: RefRender | None = None) -> None:
        """Accumulate one frame's residual points (renders the reference if `ref_render` is None)."""
        K = self.info.intrinsics.K_depth
        W, H = self._wh
        pose = np.asarray(self.info.T_align, float) @ frame.pose_c2w
        d_exp = ref_render.depth if ref_render is not None else self.filt.ref.render(K, invert(pose), W, H)[0]
        self.stats["frames"] += 1
        self.stats["nearer_px"] += int(nearer_mask(frame.depth, d_exp, self.cfg).sum())
        if self.wants_appearance and ref_render is not None and ref_render.appearance is not None:
            self._observe_appearance(frame, ref_render, K, pose)
        tau = float(self.pc["tau_new"])
        vl = None
        if not self.relaxed:
            pts, pix = nearer_points(frame.depth, d_exp, K, pose, self.filt, self.cfg)
        else:
            # one pass at the lenient threshold; the strict tier gets exactly the pixels nearer than tau_new
            tau_low = min(tau, float(self.pc["relaxed"]["tau_new_large"]))
            pts, pix = nearer_points(frame.depth, d_exp, K, pose, self.filt, self.cfg, tau=tau_low)
            if len(pts) and self.vox_large is not None:
                self.stats["large_pts"] += len(pts)
                vl = self.vox_large.add(pts, pose)
                self._frames_large.append((int(frame.idx), pix.astype(np.int32), vl.astype(np.int32)))
            strict = np.asarray(frame.depth, np.float64).ravel()[pix] < np.asarray(d_exp, np.float64).ravel()[pix] - tau
            pts, pix = pts[strict], pix[strict]
        vidx = None
        if len(pts):
            self.stats["residual_pts"] += len(pts)
            vidx = self.vox.add(pts, pose)
            self._frames.append((int(frame.idx), pix.astype(np.int32), vidx.astype(np.int32)))
        if self.free_votes:
            self._observe_free(frame.depth, K, pose, vidx, vl)

    def _observe_free(self, depth: np.ndarray, K: np.ndarray, pose: np.ndarray, vidx, vl) -> None:
        """CHANGE 2: see-through votes of this frame on both depth grids (voxels hit now are skipped)."""
        cc = self.cfg["confidence"]["candidates"]
        dmin = window_min_depth(depth, int(cc["free_window"]), float(self.pc["max_depth"]))
        self.stats["free_votes"] += self.vox.observe_free(dmin, K, pose, None if vidx is None else np.unique(vidx))
        if self.vox_large is not None:
            self.stats["large_free_votes"] += self.vox_large.observe_free(dmin, K, pose,
                                                                          None if vl is None else np.unique(vl))  # fmt: skip

    def _observe_appearance(self, frame: Frame, rr: RefRender, K: np.ndarray, pose: np.ndarray) -> None:
        """Output (ii): mismatching support-surface pixels -> the appearance voxel grid."""
        ac = self.cfg["appearance"]
        d = np.asarray(frame.depth, np.float64)
        de = np.asarray(rr.depth, np.float64)
        lab = np.asarray(rr.labels, np.int64)
        sup = np.zeros(lab.shape, bool)
        inside = (lab > 0) & (lab < len(self._support))
        sup[inside] = self._support[lab[inside]]
        up = rr.up if rr.up is not None else np.zeros(lab.shape, np.float32)
        m = (rr.appearance.mismatch_px & sup & (d > 0) & (de > 0) & (d <= float(self.pc["max_depth"]))
             & (np.abs(d - de) <= float(self.cfg["visibility"]["tau_near"]))
             & (up >= float(ac["support_min_normal_z"])))  # fmt: skip
        pix = np.flatnonzero(m)
        self.stats["app_px"] += int(pix.size)
        if pix.size == 0:
            return
        W = d.shape[1]
        v, u = np.divmod(pix, W)
        z = d.ravel()[pix]
        pts = apply(pose, np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1))
        ok = self.filt.keep_struct(pts)
        pts, pix = pts[ok], pix[ok]
        if len(pts) == 0:
            return
        self.stats["app_pts"] += len(pts)
        vidx = self.vox_app.add(pts, pose)
        self._frames_app.append((int(frame.idx), pix.astype(np.int32), vidx.astype(np.int32)))

    def finish(self) -> list[ObsInstance]:
        """Cluster kept voxels into candidates and turn them into `ObsInstance`s."""
        cands, grid, dropped = self._depth_clusters(record=True)
        conf = [self._confidence(c, g) for c, g in zip(cands, grid)]
        if self.candconf:
            p_min = float(self.cfg["confidence"]["candidates"]["p_new_min"])
            self.stats["seen_through_voxels"] = self.vox.seen_through(self.l_keep_small)
            if self.vox_large is not None:
                self.stats["large_seen_through_voxels"] = self.vox_large.seen_through(self.l_keep_large)
            self.stats["below_p_new_min"] = int(sum(bool(p is not None and p < p_min) for p in conf))
        masks = self._tier_masks(cands, grid)
        kinds = ["depth"] * len(cands)
        app = self._appearance_candidates([(self.vox_large if g else self.vox).coords(c.voxels) for c, g in zip(cands, grid)])
        if app:
            masks += self._masks(app, self._frames_app, len(self.vox_app))
            kinds += ["appearance"] * len(app)
        all_c = cands + app
        self.candidates, self.dropped_door = all_c, dropped
        crops = self._crops(masks) if self.with_crops else [[] for _ in all_c]
        k_top = int(self.pc["clip"]["topk"])
        out = []
        conf += [None] * len(app)
        for k, (c, pm, cr, kind, p) in enumerate(zip(all_c, masks, crops, kinds, conf)):
            topk = self.labeler.topk(cr, k_top) if (self.labeler is not None and cr) else []
            self.stats["labelled"] += bool(topk)
            label = topk[0][0] if topk else UNLABELLED
            out.append(ObsInstance(
                token=f"c{k}", label=label, label_topk=[(a, float(b)) for a, b in topk],
                nyu40=self.label_nyu.get(label), points=c.points.copy(), masks=pm, crops=cr, kind=kind,
                confidence=p,
            ))  # fmt: skip
        return out

    def live_candidates(self) -> list[dict[str, Any]]:
        """Depth candidates as they stand now (mid-walk; snapshot sidecar, CHANGE 2): no labels, no masks.

        Rows `{"min", "max", "confidence", "n_voxels", "tier"}` (confidence None without
        `confidence.candidates`); `stats` is not changed.
        """
        if not hasattr(self, "vox"):
            return []
        cands, grid, _ = self._depth_clusters(record=False)
        return [{"min": [round(float(v), 4) for v in c.points.min(axis=0)],
                 "max": [round(float(v), 4) for v in c.points.max(axis=0)],
                 "confidence": self._confidence(c, g), "n_voxels": int(len(c.voxels)),
                 "tier": "large" if g else "small"} for c, g in zip(cands, grid)]  # fmt: skip

    def _confidence(self, c: Candidate, grid: int) -> float | None:
        """sigmoid(median voxel log-odds) of a depth candidate (CHANGE 2), None when off."""
        if not self.candconf:
            return None
        vox = self.vox_large if grid else self.vox
        m = float(np.median(vox.logodds(c.voxels)))
        return round(float(1.0 / (1.0 + np.exp(-m))), 6)

    def _depth_clusters(self, record: bool = True) -> tuple[list[Candidate], list[int], list[Candidate]]:
        """Kept voxels -> clusters -> door filter (-> lenient tier merge with CHANGE 1).

        Kept: `lv >= l_keep_small` with candidate log-odds (CHANGE 2), else `>= min_views_small`
        views. `record`: write the counts into `stats` (False for mid-walk snapshots).

        Returns:
            (candidates, grid of each (0 strict `vox` / 1 lenient `vox_large`), door-dropped clusters).
        """
        kept = self.vox.kept_by_logodds(self.l_keep_small) if self.candconf else self.vox.kept(self.min_views_small)
        cands = cluster(kept, self.vox.coords(kept), self.vox.mean_points(kept), self.cfg)
        cands, dropped = drop_door_clusters(cands, self.doors, float(self.pc["door_frac"]))
        st: dict[str, Any] = {"voxels": len(self.vox), "kept_voxels": len(kept), "clusters": len(cands),
                              "door_dropped": len(dropped)}  # fmt: skip
        grid = [0] * len(cands)
        if self.relaxed:
            cands, grid, big_dropped, st_large = self._merge_large(cands)
            dropped = dropped + big_dropped
            st.update(st_large)
        if record:
            self.stats.update(st)
        return cands, grid, dropped

    def _merge_large(
        self, strict: list[Candidate]
    ) -> tuple[list[Candidate], list[int], list[Candidate], dict[str, int]]:
        """Add the large clusters of the lenient grid (CHANGE 1).

        Lenient tier: voxels hit at `tau_new_large` with >= `min_views_large` views, clusters with
        >= `large_min_voxels` voxels that pass `is_large` and the door filter. Every strict voxel is
        also a lenient voxel (same lattice, superset of hits), so a strict cluster lies inside one
        lenient cluster. `relaxed.merge = replace`: an accepted large cluster replaces the strict
        clusters inside it; `add`: a large cluster containing a strict cluster is dropped (strict
        candidates are never changed, large clusters are only added where no strict cluster is).
        Other strict clusters stay. Order: size descending, then centroid (deterministic tokens).

        With candidate log-odds (CHANGE 2) the lenient voxels are those with `lv >= l_keep_large`.

        Returns:
            (candidates, grid of each (0 strict / 1 lenient), door-dropped lenient clusters, stats).
        """
        rc = self.pc["relaxed"]
        vl = self.vox_large
        kept = vl.kept_by_logodds(self.l_keep_large) if self.candconf else vl.kept(int(rc["min_views_large"]))
        big = cluster(kept, vl.coords(kept), vl.mean_points(kept), self.cfg, min_voxels=int(rc["large_min_voxels"]),
                      link=int(rc["link_voxels_large"]))  # fmt: skip
        big = [c for c in big if is_large(c, self.cfg)]
        big, big_dropped = drop_door_clusters(big, self.doors, float(self.pc["door_frac"]))
        replaced = np.zeros(len(strict), bool)
        accepted = []
        for b in big:
            bkeys = vl.keys[b.voxels]
            inside = [i for i, c in enumerate(strict) if np.isin(self.vox.keys[c.voxels], bkeys).any()]
            if inside and rc["merge"] == "add":  # strict candidates stay as they are; only uncovered large clusters are added
                continue
            replaced[inside] = True
            accepted.append(b)
        big = accepted
        entries = [(c, 0) for c, r in zip(strict, replaced) if not r] + [(b, 1) for b in big]
        entries.sort(key=lambda e: (-len(e[0].voxels), *np.round(e[0].centroid, 6).tolist()))
        st = {"large_voxels": len(vl), "large_clusters": len(big), "large_door_dropped": len(big_dropped),
              "large_replaced": int(replaced.sum()), "clusters": len(entries)}  # fmt: skip
        return [e[0] for e in entries], [e[1] for e in entries], big_dropped, st

    def _tier_masks(self, cands: list[Candidate], grid: list[int]) -> list[PackedMasks]:
        """Masks per candidate from the (pixel, voxel) pairs of the grid the candidate lives in."""
        out: list[PackedMasks | None] = [None] * len(cands)
        for g, (frames, vox) in enumerate(((self._frames, self.vox), (self._frames_large, self.vox_large))):
            idx = [i for i, x in enumerate(grid) if x == g]
            if idx:
                for i, m in zip(idx, self._masks([cands[i] for i in idx], frames, len(vox))):
                    out[i] = m
        return out  # type: ignore[return-value]

    def _appearance_candidates(self, depth_coords: list[np.ndarray]) -> list[Candidate]:
        """Clusters of the appearance grid that do not touch a depth candidate (output ii).

        `depth_coords`: integer voxel coordinates of every depth candidate (either grid).
        """
        if not self.wants_appearance or len(self.vox_app) == 0:
            return []
        kept = self.vox_app.kept(int(self.pc["min_views"]))
        ca = cluster(kept, self.vox_app.coords(kept), self.vox_app.mean_points(kept), self.cfg)
        ca, dropped = drop_door_clusters(ca, self.doors, float(self.pc["door_frac"]))
        self.stats.update(app_voxels=len(self.vox_app), app_kept_voxels=len(kept), app_door_dropped=len(dropped))
        if ca and depth_coords:
            from scipy.spatial import cKDTree

            dc = np.concatenate(depth_coords).astype(np.float64)
            tree = cKDTree(dc)
            link = float(self.pc["link_voxels"])
            keep = []
            for c in ca:
                dist, _ = tree.query(self.vox_app.coords(c.voxels).astype(np.float64), k=1, p=np.inf)
                keep.append(bool(np.min(dist) > link))
            self.stats["app_dropped_depth"] = int(len(ca) - sum(keep))
            ca = [c for c, k in zip(ca, keep) if k]
        self.stats["app_clusters"] = len(ca)
        return ca

    # ---- helpers ---------------------------------------------------------------------------

    def _masks(self, cands: list[Candidate], frames, n_vox: int) -> list[PackedMasks]:
        """Per candidate and frame: pixels whose residual point landed in one of its voxels."""
        W, H = self._wh
        owner = np.full(n_vox, -1, np.int64)
        for k, c in enumerate(cands):
            owner[c.voxels] = k
        out = [PackedMasks((H, W)) for _ in cands]
        for fidx, pix, vidx in frames:
            o = owner[vidx]
            sel = o >= 0
            if not sel.any():
                continue
            for k in np.unique(o[sel]).tolist():
                m = np.zeros(H * W, bool)
                m[pix[o == k]] = True
                out[k].add(fidx, m.reshape(H, W))
        return out

    def _crops(self, masks: list[PackedMasks]) -> list[list[np.ndarray]]:
        """Upright colour crops of the `crop_k` largest masks of every candidate.

        Depth -> colour pixels by the per-axis resolution scale (same FOV, as `OracleSource`);
        the box is enlarged by `crop_margin` per side. Masks below `min_crop_px` give no crop.
        """
        k = int(self.pc["crop_k"])
        min_px = int(self.pc["min_crop_px"])
        margin = float(self.pc["crop_margin"])
        W, H = self._wh
        CW, CH = self.info.intrinsics.color_wh
        sx, sy = CW / W, CH / H
        best = [[f for f in sorted(pm, key=lambda f: (-pm.area(f), f))[:k] if pm.area(f) >= min_px] for pm in masks]
        out: list[list[np.ndarray]] = [[] for _ in masks]
        if not any(best):
            return out
        cache: dict[int, np.ndarray] = {}
        with open_sequence(self.info.root, self.info.rescan_id) as seq:
            for i, (pm, frames) in enumerate(zip(masks, best)):
                for f in frames:
                    if f not in cache:
                        cache[f] = seq.color(f)
                    rows, cols = np.nonzero(pm[f])
                    r0, r1 = rows.min(), rows.max() + 1
                    c0, c1 = cols.min(), cols.max() + 1
                    dr, dc = margin * (r1 - r0), margin * (c1 - c0)
                    R0, R1 = int(max(0, (r0 - dr) * sy)), int(min(CH, np.ceil((r1 + dr) * sy)))
                    C0, C1 = int(max(0, (c0 - dc) * sx)), int(min(CW, np.ceil((c1 + dc) * sx)))
                    if R1 > R0 and C1 > C0:
                        out[i].append(np.rot90(cache[f][R0:R1, C0:C1], k=-1).copy())
        return out
