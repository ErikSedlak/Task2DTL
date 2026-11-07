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

Method side: inputs are the reference context and the rescan's RGB-D frames + poses +
refined `T_align` only. Never imports `oracle` / `eval`, never reads rescan annotations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lsg.geom.transforms import invert
from lsg.io.scan3r import Frame, open_sequence
from lsg.obs.base import ObsInstance, PackedMasks, RefContext, RefRender, SeqInfo
from lsg.perception.cluster import Candidate, cluster, door_regions, drop_door_clusters
from lsg.perception.labeler import Labeler
from lsg.perception.residual import SurfaceFilter, nearer_mask, nearer_points
from lsg.perception.voxels import VoxelEvidence

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

    # ---- ObsSource -----------------------------------------------------------------------

    def begin(self, ref_ctx: RefContext, seq_info: SeqInfo) -> None:
        """Build the surface / structure filter, door regions and an empty voxel grid."""
        self.ctx, self.info = ref_ctx, seq_info
        self.filt = SurfaceFilter(ref_ctx, self.cfg)
        self.doors = door_regions(ref_ctx, float(self.pc["door_margin"])) if self.pc.get("door_filter", True) else []
        self.vox = VoxelEvidence.from_config(self.cfg)
        W, H = seq_info.intrinsics.depth_wh
        self._wh = (W, H)
        self._frames: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.candidates, self.dropped_door = [], []
        self.stats = {"frames": 0, "nearer_px": 0, "residual_pts": 0, "voxels": 0, "kept_voxels": 0,
                      "clusters": 0, "door_dropped": 0, "doors": len(self.doors), "labelled": 0}  # fmt: skip

    def observe(self, frame: Frame, ref_render: RefRender | None = None) -> None:
        """Accumulate one frame's residual points (renders the reference if `ref_render` is None)."""
        K = self.info.intrinsics.K_depth
        W, H = self._wh
        pose = np.asarray(self.info.T_align, float) @ frame.pose_c2w
        d_exp = ref_render.depth if ref_render is not None else self.filt.ref.render(K, invert(pose), W, H)[0]
        self.stats["frames"] += 1
        self.stats["nearer_px"] += int(nearer_mask(frame.depth, d_exp, self.cfg).sum())
        pts, pix = nearer_points(frame.depth, d_exp, K, pose, self.filt, self.cfg)
        if len(pts) == 0:
            return
        self.stats["residual_pts"] += len(pts)
        vidx = self.vox.add(pts, pose)
        self._frames.append((int(frame.idx), pix.astype(np.int32), vidx.astype(np.int32)))

    def finish(self) -> list[ObsInstance]:
        """Cluster kept voxels into candidates and turn them into `ObsInstance`s."""
        kept = self.vox.kept(int(self.pc["min_views"]))
        cands = cluster(kept, self.vox.coords(kept), self.vox.mean_points(kept), self.cfg)
        cands, dropped = drop_door_clusters(cands, self.doors, float(self.pc["door_frac"]))
        self.candidates, self.dropped_door = cands, dropped
        self.stats.update(voxels=len(self.vox), kept_voxels=len(kept), clusters=len(cands), door_dropped=len(dropped))
        masks = self._masks(cands)
        crops = self._crops(masks) if self.with_crops else [[] for _ in cands]
        k_top = int(self.pc["clip"]["topk"])
        out = []
        for k, (c, pm, cr) in enumerate(zip(cands, masks, crops)):
            topk = self.labeler.topk(cr, k_top) if (self.labeler is not None and cr) else []
            self.stats["labelled"] += bool(topk)
            label = topk[0][0] if topk else UNLABELLED
            out.append(ObsInstance(
                token=f"c{k}", label=label, label_topk=[(a, float(b)) for a, b in topk],
                nyu40=self.label_nyu.get(label), points=c.points.copy(), masks=pm, crops=cr,
            ))  # fmt: skip
        return out

    # ---- helpers ---------------------------------------------------------------------------

    def _masks(self, cands: list[Candidate]) -> list[PackedMasks]:
        """Per candidate and frame: pixels whose residual point landed in one of its voxels."""
        W, H = self._wh
        owner = np.full(len(self.vox), -1, np.int64)
        for k, c in enumerate(cands):
            owner[c.voxels] = k
        out = [PackedMasks((H, W)) for _ in cands]
        for fidx, pix, vidx in self._frames:
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
