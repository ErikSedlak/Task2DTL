"""`OracleSource`: rescan GT instances as observations (DESIGN §6.5, upper bound).

Renders the rescan's own labelled mesh at each processed frame (its own frame, its own pose),
so it reports exactly the instances, pixels and mesh vertices that these frames see. A pixel
counts only where the sensor has valid depth (DECISIONS M2), i.e. where perception could see
something too. Instance ids are hidden behind tokens from a seeded permutation; the mapping
(`token_map`) is for `lsg.eval` only and is never handed to the method.

Reads rescan annotations, which hard rule 1 allows only here and in `lsg.eval`.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from lsg.geom.render import Renderer
from lsg.geom.transforms import apply, invert
from lsg.io.scan3r import Frame, Scan, open_sequence
from lsg.obs.base import ObsInstance, PackedMasks, RefContext, RefRender, SeqInfo


def shuffle_seed(seed: int, rescan_id: str) -> list[int]:
    """Seed sequence of the token permutation: config seed + a stable hash of the rescan id."""
    return [int(seed), zlib.crc32(rescan_id.encode("utf-8"))]


class OracleSource:
    """Implements `lsg.obs.ObsSource` from the rescan's GT instance segmentation."""

    def __init__(
        self,
        rescan: Scan,
        objects_S: Mapping[int, Mapping[str, Any]],
        cfg: Mapping[str, Any],
        seed: int | None = None,
    ):
        """
        Args:
            rescan: The rescan's labelled mesh (`load_scan`, loaded by the eval harness).
            objects_S: The rescan's 3DSSG objects (for nyu40); may be empty.
            cfg: Full config; uses `cfg["oracle"]` (`min_points`, `seed`, `crops`, `crop_k`).
            seed: Overrides `cfg["oracle"]["seed"]`.
        """
        self.S = rescan
        self.objects = objects_S
        self.ocfg = dict(cfg["oracle"])
        self.seed = int(self.ocfg["seed"] if seed is None else seed)
        self._renderer: Renderer | None = None
        self._token_of: dict[int, str] = {}

    # ---- ObsSource -----------------------------------------------------------------------

    def begin(self, ref_ctx: RefContext | None, seq_info: SeqInfo) -> None:
        """Build the rescan renderer and reset the per-sequence accumulators."""
        self.info = seq_info
        if self._renderer is None:
            self._renderer = Renderer(self.S.vertices, self.S.faces, self.S.face_labels)
        W, H = seq_info.intrinsics.depth_wh
        self._wh = (W, H)
        self._face_seen = np.zeros(len(self.S.faces), dtype=bool)
        self._masks: dict[int, PackedMasks] = {}
        self._token_of = {}
        self._peek_token_of: dict[int, str] | None = None
        self._peek_seen: set[str] = set()

    def observe(self, frame: Frame, ref_render: RefRender | None = None) -> None:
        """Render the rescan mesh at this frame; record per-instance masks and hit faces."""
        assert self._renderer is not None, "call begin() first"
        W, H = self._wh
        r = self._renderer
        _, prim = r.cast(r.camera_rays(self.info.intrinsics.K_depth, invert(frame.pose_c2w), W, H))
        valid = (prim >= 0) & (frame.depth > 0)
        if not valid.any():
            return
        self._face_seen[prim[valid]] = True
        lab = np.where(valid, self.S.face_labels[np.maximum(prim, 0)], 0)
        for oid in np.unique(lab[lab > 0]):
            pm = self._masks.get(int(oid))
            if pm is None:
                pm = self._masks[int(oid)] = PackedMasks((H, W))
            pm.add(frame.idx, lab == oid)

    def _kept(self) -> dict[int, np.ndarray]:
        """Instances with >= `min_points` observed vertices so far: {objectId: vertex indices}."""
        min_pts = int(self.ocfg["min_points"])
        seen_faces = self.S.faces[self._face_seen]
        seen_labels = self.S.face_labels[self._face_seen]
        kept: dict[int, np.ndarray] = {}
        for oid in sorted(self._masks):
            vidx = np.unique(seen_faces[seen_labels == oid].ravel())
            if len(vidx) >= min_pts:
                kept[oid] = vidx
        return kept

    def _instances(self, kept: Mapping[int, np.ndarray], token_of: Mapping[int, str]) -> list[ObsInstance]:
        out = []
        for oid in sorted(kept):
            label = self.S.semseg_labels.get(oid) or (self.S.instances[oid].label if oid in self.S.instances else "")
            label = label or str(self.objects.get(oid, {}).get("label", ""))
            o = self.objects.get(oid)
            if o is not None and "nyu40" in o:
                nyu: int | None = int(o["nyu40"])
            else:
                nyu = int(self.S.instances[oid].nyu40) if oid in self.S.instances else None
            out.append(
                ObsInstance(
                    token=token_of[oid],
                    label=label,
                    label_topk=[(label, 1.0)],
                    nyu40=nyu,
                    points=apply(self.info.T_align, self.S.vertices[kept[oid]]),
                    masks=self._masks[oid],
                )
            )
        return out

    def finish(self) -> list[ObsInstance]:
        """Instances with >= `min_points` observed vertices, tokens shuffled, sorted by token."""
        kept = self._kept()
        ids = sorted(kept)
        perm = np.random.default_rng(shuffle_seed(self.seed, self.info.rescan_id)).permutation(len(ids))
        self._token_of = {oid: f"o{int(k)}" for oid, k in zip(ids, perm)}
        out = self._instances(kept, self._token_of)
        if self.ocfg.get("crops"):
            self._add_crops(out, int(self.ocfg.get("crop_k", 5)))
        return sorted(out, key=lambda x: x.token)

    def peek(self) -> list[ObsInstance]:
        """The instances observed so far, without ending the walk (CHANGE 3 Part B snapshots).

        Tokens `p<k>` come from a seeded permutation over *all* rescan instance ids, fixed per
        sequence, so a token keeps its object across snapshots (`finish` keeps its own `o<k>`).
        No crops.
        """
        if self._peek_token_of is None:
            ids = sorted({int(i) for i in self.S.instances} | {int(i) for i in np.unique(self.S.face_labels) if i > 0})
            perm = np.random.default_rng([*shuffle_seed(self.seed, self.info.rescan_id), 1]).permutation(len(ids))
            self._peek_token_of = {oid: f"p{int(k)}" for oid, k in zip(ids, perm)}
        out = self._instances(self._kept(), self._peek_token_of)
        self._peek_seen = {o.token for o in out}
        return sorted(out, key=lambda x: x.token)

    # ---- eval only -------------------------------------------------------------------------

    def token_map(self) -> dict[str, int]:
        """`{token: rescan objectId}`. For `lsg.eval` only; never give it to the method."""
        return {t: oid for oid, t in self._token_of.items()}

    def peek_token_map(self) -> dict[str, int]:
        """`{peek token: rescan objectId}` of the last `peek()`. For `lsg.eval` only."""
        seen = getattr(self, "_peek_seen", set())
        return {t: oid for oid, t in (self._peek_token_of or {}).items() if t in seen}

    # ---- helpers ---------------------------------------------------------------------------

    def _add_crops(self, insts: list[ObsInstance], k: int) -> None:
        """Upright colour crops from the `k` frames where each instance's mask is largest."""
        W, H = self._wh
        CW, CH = self.info.intrinsics.color_wh
        sx, sy = CW / W, CH / H
        cache: dict[int, np.ndarray] = {}
        with open_sequence(self.info.root, self.info.rescan_id) as seq:
            for inst in insts:
                pm = inst.masks
                assert isinstance(pm, PackedMasks)
                best = sorted(pm, key=lambda f: (-pm.area(f), f))[:k]
                for f in best:
                    if f not in cache:
                        cache[f] = seq.color(f)
                    rows, cols = np.nonzero(pm[f])
                    r0, r1 = int(rows.min() * sy), int(np.ceil((rows.max() + 1) * sy))
                    c0, c1 = int(cols.min() * sx), int(np.ceil((cols.max() + 1) * sx))
                    inst.crops.append(np.rot90(cache[f][r0:r1, c0:c1], k=-1).copy())
