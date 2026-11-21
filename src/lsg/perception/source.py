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

Co-segmentation (CHANGE 3 Part N, `perception.segment`): with method `geometric` / `sam`
every frame's residual pixels get image-segment ids (depth-edge components or cached SAM masks), and
both depth tiers cluster their kept voxels with `segment.cosegment_cluster` (adjacent voxels link only
if the frames seeing both mostly put them in one segment) instead of `cluster.cluster`.

Thin-object candidates (M6, DESIGN §6.6 output ii; `appearance.thin_candidates`): when the
online loop passes an `AppearanceMap` in `RefRender`, mismatching pixels whose depth agrees with
the reference (|d_obs - d_exp| <= tau_near) and whose first hit is a support surface (a non-thin
anchor / thing or the floor, face |normal z| >= `support_min_normal_z`) are back-projected into
a **separate** voxel grid (structure filter only: such objects lie within `surf_dist` of the
reference surface). The same >= `min_views` / clustering / door rules apply; clusters touching a
depth candidate are dropped (depth wins). They become `ObsInstance(kind="appearance")`, with
tokens continuing after the depth candidates.

Detector (CHANGE 3 Part E, `perception.detector`): cached Grounding DINO + SAM 2 detections are fused
into 3D tracks (`detector.DetectorTracks`) and become `ObsInstance(kind="detector")` with tokens `d<k>`
(stable across snapshots), `label_topk` from the detection scores (or CLIP, `labels: clip`) and a
`newness` score. `mode: replace` drops the residual depth candidates (thin-object candidates stay);
`union` keeps the residual candidates that overlap no detector instance.

Method side: inputs are the reference context and the rescan's RGB-D frames + poses +
refined `T_align` only. Never imports `oracle` / `eval`, never reads rescan annotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.geom.transforms import apply, invert
from lsg.io.scan3r import Frame, open_sequence
from lsg.obs.base import ObsInstance, PackedMasks, RefContext, RefRender, SeqInfo
from lsg.perception.cluster import Candidate, cluster, door_regions, drop_door_clusters, is_large
from lsg.perception.labeler import Labeler
from lsg.perception.residual import SurfaceFilter, nearer_mask, nearer_points
from lsg.perception.segment import SamSegments, cosegment_cluster, seg_method, segment_frame, voxel_segments
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
        det_prompts: Sequence[str] | None = None,
    ):
        """
        Args:
            labeler: Turns crops into `label_topk` (None = every candidate is `UNLABELLED`).
            label_nyu: Vocabulary label -> nyu40 (`train_meta.perception_vocab`).
            cfg: Full config (`perception`, `decorrelation`).
            with_crops: Read colour frames for crops (needs `seq_info.root` with the sequence).
            det_prompts: The detector's ordered prompt list (`detect_cache.ordered_prompts`; CHANGE 3 Part E).
        """
        self.det_prompts = list(det_prompts or [])
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
        # CHANGE 3 Part C: candidate confidence from cluster-level votes over the cluster's voxel history
        tc = cc.get("track") or {}
        self.track = self.candconf and self.free_votes and bool(tc.get("enabled", False))
        self.app_conf = bool(tc.get("appearance", False))  # CHANGE 3 Part N: cluster votes for thin-object candidates
        self._poses: dict[int, np.ndarray] = {}
        self._hist_trees: dict[tuple[int, int], Any] = {}
        for v in (self.vox, self.vox_large):
            if v is not None:
                v.log_free = self.track
        self._frames_large: list[tuple[int, np.ndarray, np.ndarray]] = []
        W, H = seq_info.intrinsics.depth_wh
        self._wh = (W, H)
        self._frames: list[tuple[int, np.ndarray, np.ndarray]] = []
        dc = self.pc.get("detector") or {}  # CHANGE 3 Part E
        self.det = None
        self.det_mode = str(dc.get("mode", "replace"))
        if dc.get("enabled", False):
            from lsg.perception.detect_cache import DetectionCache
            from lsg.perception.detector import DetectorTracks

            cache = DetectionCache(self.cfg, seq_info.rescan_id, self.det_prompts)
            if cache.n_frames() == 0:
                raise FileNotFoundError(f"no detections cached in {cache.dir} (run `lsg detect cache`)")
            self.det = DetectorTracks(cache, self.filt, self.cfg, seq_info.intrinsics.K_depth, (W, H), self.doors)
        # CHANGE 3 Part N: image segments of every frame's residual pixels (frame -> (sorted pix, seg id))
        self.segm = seg_method(self.cfg)
        self._segs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sam = None
        if self.segm == "sam":
            from lsg.perception.sam_cache import rescan_dir

            self._sam = SamSegments(rescan_dir(self.cfg, seq_info.rescan_id))
        self.candidates, self.dropped_door = [], []
        # CHANGE 3 Part B: snapshot tracks (`peek`)
        self._tracks: dict[str, tuple[int, np.ndarray]] = {}
        self._track_label: dict[str, tuple[int, list]] = {}
        self._next_track = 0
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
        if self.track:
            self._poses[int(frame.idx)] = pose
        self.stats["nearer_px"] += int(nearer_mask(frame.depth, d_exp, self.cfg).sum())
        if self.wants_appearance and ref_render is not None and ref_render.appearance is not None:
            self._observe_appearance(frame, ref_render, K, pose)
        if self.det is not None:
            self.det.observe(int(frame.idx), frame.depth, pose)
            if self.det_mode == "replace":  # no residual depth candidates
                return
        tau = float(self.pc["tau_new"])
        vl = None
        if not self.relaxed:
            pts, pix = nearer_points(frame.depth, d_exp, K, pose, self.filt, self.cfg)
            self._store_segments(frame, pix)
        else:
            # one pass at the lenient threshold; the strict tier gets exactly the pixels nearer than tau_new
            tau_low = min(tau, float(self.pc["relaxed"]["tau_new_large"]))
            pts, pix = nearer_points(frame.depth, d_exp, K, pose, self.filt, self.cfg, tau=tau_low)
            self._store_segments(frame, pix)
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
            self._observe_free(frame.depth, K, pose, vidx, vl, int(frame.idx))

    def _store_segments(self, frame: Frame, pix: np.ndarray) -> None:
        """CHANGE 3 Part N: segment ids of this frame's residual pixels (lenient pass = superset of both tiers)."""
        if self.segm == "none" or not len(pix):
            return
        pix = np.sort(np.asarray(pix, np.int64))
        if self.segm == "geometric":
            seg = segment_frame(frame.depth, pix, self.cfg, K=self.info.intrinsics.K_depth)
        else:
            assert self._sam is not None
            seg = self._sam.segment(int(frame.idx), pix)
        self._segs[int(frame.idx)] = (pix, seg)

    def _seg_of(self, fidx: int, pix: np.ndarray) -> np.ndarray:
        """Segment id of each pixel of a stored frame entry (-1 if unknown)."""
        sp = self._segs.get(int(fidx))
        if sp is None or not len(sp[0]):
            return np.full(len(pix), -1, np.int64)
        P, S = sp
        pix = np.asarray(pix, np.int64)
        i = np.clip(np.searchsorted(P, pix), 0, len(P) - 1)
        return np.where(P[i] == pix, S[i], -1)

    def _cluster(self, vox: VoxelEvidence, kept: np.ndarray, frames, min_voxels: int | None = None,
                 link: int | None = None, record: dict[str, Any] | None = None, prefix: str = "") -> list[Candidate]:  # fmt: skip
        """Kept voxels -> clusters: `cluster.cluster`, or co-segmentation with `perception.segment` (Part N)."""
        if self.segm == "none":
            return cluster(kept, vox.coords(kept), vox.mean_points(kept), self.cfg, min_voxels=min_voxels, link=link)
        mv = int(self.pc["min_voxels"] if min_voxels is None else min_voxels)
        lk = int(self.pc.get("link_voxels", 1) if link is None else link)
        fs = [voxel_segments(vidx, self._seg_of(f, pix)) for f, pix, vidx in frames]
        sc = self.pc["segment"]
        bu = sc.get("bridge_unseen")
        cands, st = cosegment_cluster(kept, vox.coords(kept), vox.mean_points(kept), fs, len(vox), mv, lk,
                                      cut_frac=float(sc.get("cut_frac", 0.5)), bridge_unseen=None if bu is None else int(bu))
        if record is not None:
            record[f"{prefix}seg_edges"] = st["edges"]
            record[f"{prefix}seg_cut_edges"] = st["cut_edges"]
        return cands

    def _observe_free(self, depth: np.ndarray, K: np.ndarray, pose: np.ndarray, vidx, vl, fidx: int) -> None:
        """CHANGE 2: see-through votes of this frame on both depth grids (voxels hit now are skipped)."""
        cc = self.cfg["confidence"]["candidates"]
        dmin = window_min_depth(depth, int(cc["free_window"]), float(self.pc["max_depth"]))
        self.stats["free_votes"] += self.vox.observe_free(dmin, K, pose, None if vidx is None else np.unique(vidx),
                                                          frame_idx=fidx)  # fmt: skip
        if self.vox_large is not None:
            self.stats["large_free_votes"] += self.vox_large.observe_free(dmin, K, pose,
                                                                          None if vl is None else np.unique(vl),
                                                                          frame_idx=fidx)  # fmt: skip

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
        return self._build(record=True)

    def peek(self) -> list[ObsInstance]:
        """The current candidates without ending the walk (CHANGE 3 Part B snapshots); no state the final
        result depends on changes. Tokens are track ids `t<k>` that persist across snapshots (`_track_ids`);
        CLIP labels are reused per track unless it grew by more than `snapshots.relabel_growth`."""
        return self._build(record=False)

    def _build(self, record: bool) -> list[ObsInstance]:
        """`finish` (record: stats, `candidates`, tokens `c<k>` by size) or `peek` (tracks, cached labels)."""
        dets = self.det.instances(record) if self.det is not None else []
        if self.det is not None and self.det_mode == "replace":
            cands, grid, dropped = [], [], []
        else:
            cands, grid, dropped = self._depth_clusters(record=record)
            if dets:  # union: residual candidates overlapping a detector instance are dropped
                from lsg.perception.detector import overlaps

                d_pts = [d["points"] for d in dets]
                keep = [not overlaps(c.points, d_pts, 1.5 * float(self.pc["voxel"]), 0.3) for c in cands]
                if record:
                    self.stats["det_union_dropped"] = int(len(keep) - sum(keep))
                cands = [c for c, k in zip(cands, keep) if k]
                grid = [g for g, k in zip(grid, keep) if k]
        votes = [self.cluster_votes(c, g) if self.track else None for c, g in zip(cands, grid)]
        conf = [self._confidence(c, g, v) for c, g, v in zip(cands, grid, votes)]
        if self.candconf and record:
            p_min = float(self.cfg["confidence"]["candidates"]["p_new_min"])
            self.stats["seen_through_voxels"] = self.vox.seen_through(self.l_keep_small)
            if self.vox_large is not None:
                self.stats["large_seen_through_voxels"] = self.vox_large.seen_through(self.l_keep_large)
            self.stats["below_p_new_min"] = int(sum(bool(p is not None and p < p_min) for p in conf))
        masks = self._tier_masks(cands, grid)
        kinds = ["depth"] * len(cands)
        app = self._appearance_candidates([(self.vox_large if g else self.vox).coords(c.voxels) for c, g in zip(cands, grid)]
                                          + [d["voxel_coords"] for d in dets])  # fmt: skip
        if app:
            masks += self._masks(app, self._frames_app, len(self.vox_app))
            kinds += ["appearance"] * len(app)
        all_c = cands + app
        grids = list(grid) + [2] * len(app)
        if record:
            self.candidates, self.dropped_door = all_c, dropped
        if self.track and self.app_conf:  # CHANGE 3 Part N: thin-object candidates face the same threshold
            app_votes = [self.cluster_votes(c, 2) for c in app]
            conf += [self._confidence(c, 2, v) for c, v in zip(app, app_votes)]
            votes += app_votes
        else:
            conf += [None] * len(app)
            votes += [None] * len(app)
        k_top = int(self.pc["clip"]["topk"])
        if record:
            tokens = [f"c{k}" for k in range(len(all_c))]
            crops = self._crops(masks) if self.with_crops else [[] for _ in all_c]
            topks = []
            for cr in crops:
                topk = self.labeler.topk(cr, k_top) if (self.labeler is not None and cr) else []
                self.stats["labelled"] += bool(topk)
                topks.append(topk)
        else:
            tokens = self._track_ids(all_c, grids)
            grow = float((self.cfg.get("online") or {}).get("relabel_growth", 0.25))
            need = [i for i, (t, c) in enumerate(zip(tokens, all_c))
                    if t not in self._track_label or len(c.voxels) > (1.0 + grow) * self._track_label[t][0]]  # fmt: skip
            crops = [[] for _ in all_c]
            if self.with_crops and need:
                for i, cr in zip(need, self._crops([masks[i] for i in need])):
                    crops[i] = cr
            for i in need:
                topk = self.labeler.topk(crops[i], k_top) if (self.labeler is not None and crops[i]) else []
                self._track_label[tokens[i]] = (len(all_c[i].voxels), topk)
            topks = [self._track_label[t][1] for t in tokens]
        out = []
        for tok, c, pm, cr, kind, p, cv, topk in zip(tokens, all_c, masks, crops, kinds, conf, votes, topks):
            label = topk[0][0] if topk else UNLABELLED
            out.append(ObsInstance(
                token=tok, label=label, label_topk=[(a, float(b)) for a, b in topk],
                nyu40=self.label_nyu.get(label), points=c.points.copy(), masks=pm, crops=cr, kind=kind,
                confidence=p, cand_votes=cv,
            ))  # fmt: skip
        out += self._detector_instances(dets, record, k_top)
        return out if record else sorted(out, key=lambda o: (o.token[0], int(o.token[1:])))

    def _detector_instances(self, dets: list[dict[str, Any]], record: bool, k_top: int) -> list[ObsInstance]:
        """CHANGE 3 Part E: detector tracks -> `ObsInstance(kind="detector")` (labels: summed detection scores or CLIP)."""
        if self.det is None:
            return []
        if record:
            self.stats.update(self.det.stats)
        if not dets:
            return []
        use_clip = str(self.pc["detector"].get("labels", "gdino")) == "clip" and self.labeler is not None
        crops: list[list[np.ndarray]] = [[] for _ in dets]
        if use_clip:
            grow = float((self.cfg.get("online") or {}).get("relabel_growth", 0.25))
            need = [i for i, d in enumerate(dets) if record or d["token"] not in self._track_label
                    or len(d["points"]) > (1.0 + grow) * self._track_label[d["token"]][0]]  # fmt: skip
            if self.with_crops and need:
                for i, cr in zip(need, self._crops([dets[i]["masks"] for i in need])):
                    crops[i] = cr
            for i in need:
                topk = self.labeler.topk(crops[i], k_top) if crops[i] else []
                self._track_label[dets[i]["token"]] = (len(dets[i]["points"]), topk)
            topks = [self._track_label[d["token"]][1] for d in dets]
        else:
            topks = [self.det.label_topk(d["label_sum"], k_top) for d in dets]
        out = []
        for d, cr, topk in zip(dets, crops, topks):
            label = topk[0][0] if topk else UNLABELLED
            if record:
                self.stats["labelled"] += bool(topk)
            out.append(ObsInstance(
                token=d["token"], label=label, label_topk=[(a, float(b)) for a, b in topk],
                nyu40=self.label_nyu.get(label), points=d["points"].copy(), masks=d["masks"], crops=cr,
                kind="detector", confidence=None, newness=round(float(d["newness"]), 6),
            ))  # fmt: skip
        return out

    def _track_ids(self, cands: list[Candidate], grids: list[int]) -> list[str]:
        """Stable ids across snapshots: the previous snapshot's tracks, oldest first, each take the unassigned
        cluster of the same grid with the highest voxel IoU >= `online.track_iou` (a merge keeps the older id,
        a split gives the smaller part a new id); unmatched clusters get new ids `t<k>`."""
        thr = float((self.cfg.get("online") or {}).get("track_iou", 0.3))
        ids: list[str | None] = [None] * len(cands)
        vsets = [np.asarray(c.voxels, np.int64) for c in cands]
        for tid in sorted(self._tracks, key=lambda t: int(t[1:])):
            g0, v0 = self._tracks[tid]
            best, best_iou = None, thr
            for i, (c, g) in enumerate(zip(vsets, grids)):
                if ids[i] is not None or g != g0:
                    continue
                inter = len(np.intersect1d(v0, c, assume_unique=True))
                iou = inter / (len(v0) + len(c) - inter) if inter else 0.0
                if iou >= best_iou and (best is None or iou > best_iou):
                    best, best_iou = i, iou
            if best is not None:
                ids[best] = tid
        for i in range(len(cands)):
            if ids[i] is None:
                ids[i] = f"t{self._next_track}"
                self._next_track += 1
        self._tracks = {t: (g, v) for t, g, v in zip(ids, grids, vsets)}  # type: ignore[misc]
        return ids  # type: ignore[return-value]

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

    def _confidence(self, c: Candidate, grid: int, votes: tuple[int, int] | None = None) -> float | None:
        """Confidence of a depth candidate, None when candidate log-odds are off.

        CHANGE 3 Part C (`confidence.candidates.track`): sigmoid(clip(L_seen·seen − L_free·free)) of the
        cluster votes (`cluster_votes`); otherwise CHANGE 2's sigmoid(median voxel log-odds).
        """
        if not self.candconf:
            return None
        if grid == 2 and votes is None:
            return None
        if self.track:
            tc = self.cfg["confidence"]["candidates"]["track"]
            s, f = votes if votes is not None else self.cluster_votes(c, grid)
            clip = float(self.cfg["confidence"]["clip"])
            m = max(-clip, min(clip, float(tc["L_seen"]) * s - float(tc["L_free"]) * f))
        else:
            vox = self.vox_large if grid else self.vox
            m = float(np.median(vox.logodds(c.voxels)))
        return round(float(1.0 / (1.0 + np.exp(-m))), 6)

    def _grid_vox(self, grid: int) -> VoxelEvidence:
        """Voxel grid by index: 0 strict depth, 1 lenient depth, 2 appearance (thin objects, CHANGE 3 Part N)."""
        return (self.vox, self.vox_large, self.vox_app)[grid]

    def _history(self, c: Candidate, grid: int) -> np.ndarray:
        """Bool mask over the grid's voxels: the cluster's voxels plus every tracked voxel of that grid
        within the linking distance of them (also those below `l_keep`, e.g. seen through)."""
        from scipy.spatial import cKDTree

        vox = self._grid_vox(grid)
        n = len(vox)
        link = int(self.pc["relaxed"]["link_voxels_large"]) if grid == 1 else int(self.pc["link_voxels"])
        key = (grid, n)
        if key not in self._hist_trees:
            self._hist_trees = {k: v for k, v in self._hist_trees.items() if k[0] != grid}
            self._hist_trees[key] = cKDTree(vox.coords(np.arange(n)).astype(float))
        tree = self._hist_trees[key]
        near = tree.query_ball_point(vox.coords(c.voxels).astype(float), r=float(link) + 1e-9, p=np.inf)
        mask = np.zeros(n, bool)
        mask[np.asarray(c.voxels, np.int64)] = True
        for lst in near:
            mask[lst] = True
        return mask

    def cluster_votes(self, c: Candidate, grid: int) -> tuple[int, int]:
        """Counted (decorrelated) seen / free votes of a depth cluster (CHANGE 3 Part C).

        Per processed frame, h = history voxels hit and f = history voxels seen through. The frame votes
        `seen` if h >= `min_vox` and h >= `vote_frac`·(h + f), `free` if f >= `min_vox` and
        f >= `vote_frac`·(h + f): the node rule of DESIGN §3 applied to voxels. Votes count per kind
        only from a new viewpoint (the `decorrelation` rule: moved >= d_dec or turned >= theta_dec).
        """
        tc = self.cfg["confidence"]["candidates"]["track"]
        vox = self._grid_vox(grid)
        frames = (self._frames, self._frames_large, self._frames_app)[grid]
        H = self._history(c, grid)
        hf: dict[int, list[int]] = {}
        for fidx, _pix, vidx in frames:
            hf.setdefault(int(fidx), [0, 0])[0] += int(H[np.unique(vidx)].sum())
        for fidx, fi in vox.free_log:
            hf.setdefault(int(fidx), [0, 0])[1] += int(H[fi].sum())
        dc = self.cfg["decorrelation"]
        on, d_dec, cos_dec = bool(dc.get("enabled", True)), float(dc["d_dec"]), float(np.cos(np.radians(float(dc["theta_dec_deg"]))))
        mv, frac = int(tc["min_vox"]), float(tc["vote_frac"])
        last: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        counted = {"seen": 0, "free": 0}
        for fidx in sorted(hf):
            h, f = hf[fidx]
            kind = "seen" if (h >= mv and h >= frac * (h + f)) else "free" if (f >= mv and f >= frac * (h + f)) else None
            if kind is None or fidx not in self._poses:
                continue
            P = self._poses[fidx]
            cpos, z = P[:3, 3], P[:3, 2] / (np.linalg.norm(P[:3, 2]) or 1.0)
            prev = last.get(kind)
            if (not on or prev is None or np.linalg.norm(prev[0] - cpos) >= d_dec or float(prev[1] @ z) <= cos_dec):
                counted[kind] += 1
                last[kind] = (cpos, z)
        return counted["seen"], counted["free"]

    def _depth_clusters(self, record: bool = True) -> tuple[list[Candidate], list[int], list[Candidate]]:
        """Kept voxels -> clusters -> door filter (-> lenient tier merge with CHANGE 1).

        Kept: `lv >= l_keep_small` with candidate log-odds (CHANGE 2), else `>= min_views_small`
        views. `record`: write the counts into `stats` (False for mid-walk snapshots).

        Returns:
            (candidates, grid of each (0 strict `vox` / 1 lenient `vox_large`), door-dropped clusters).
        """
        kept = self.vox.kept_by_logodds(self.l_keep_small) if self.candconf else self.vox.kept(self.min_views_small)
        seg_st: dict[str, Any] = {}
        cands = self._cluster(self.vox, kept, self._frames, record=seg_st)
        cands, dropped = drop_door_clusters(cands, self.doors, float(self.pc["door_frac"]))
        st: dict[str, Any] = {"voxels": len(self.vox), "kept_voxels": len(kept), "clusters": len(cands),
                              "door_dropped": len(dropped), **seg_st}  # fmt: skip
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
        seg_st: dict[str, Any] = {}
        big = self._cluster(vl, kept, self._frames_large, min_voxels=int(rc["large_min_voxels"]),
                            link=int(rc["link_voxels_large"]), record=seg_st, prefix="large_")  # fmt: skip
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
              "large_replaced": int(replaced.sum()), "clusters": len(entries), **seg_st}  # fmt: skip
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
