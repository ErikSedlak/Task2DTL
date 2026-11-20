"""Identity check for association (CHANGE 3 Part N, N3): does an observation look like a reference node?

A node with free evidence at its old place may move, and association used to give it any same-label
observation (oracle dev_tune: 64 GT new objects taken by a same-label reference node; audit: mostly a
*different* object of the class). `IdentityScorer.similarity(node, obs)` compares their appearance:

- **observation**: the `k_obs` frames with the largest mask; per frame the rescan colour image and the
  mask (depth resolution, upsampled) are turned upright, cropped to a square around the mask
  (+ `crop_margin`), resized to `crop_px`;
- **node**: the reference texture rendered from 8 virtual cameras around the node (25° elevation,
  distance clamp(1.2 · diag + 0.8, 1, 3) m, looking at its centroid); the node's mask is where the
  reference render's first hit is the node's face; the `k_node` views with the largest mask are cropped
  the same way;
- each crop -> frozen patch features (DINOv2 by default, `appearance.make_extractor`), mean-pooled over
  the patches that are >= half inside the mask, L2-normalised; fewer than `min_patches` such patches ->
  no descriptor for that crop;
- similarity = the largest cosine over (node view, observation crop) pairs; None if either side has
  no descriptor (no information -> the gate allows the pair).

Descriptors are memoised per node / observation; every scored pair is logged (`log`) for the
`lsg chg3 idstats` statistics. Method side: reference texture + mesh labels (reference annotations
are method input), rescan colour frames + masks; no rescan annotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.geom.transforms import invert
from lsg.graph.model import Node, object_id_of
from lsg.obs.base import ObsInstance
from lsg.perception.appearance import PATCH, resize


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 camera->world pose (x right, y down, z forward; world +Z up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def square_crop(img: np.ndarray, mask: np.ndarray, margin: float, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Square crop around `mask` (+ margin, clipped to the image), resized to `size`; None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    H, W = mask.shape
    cy, cx = 0.5 * (ys.min() + ys.max()), 0.5 * (xs.min() + xs.max())
    half = 0.5 * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1.0 + 2.0 * margin)
    y0, y1 = int(max(0, np.floor(cy - half))), int(min(H, np.ceil(cy + half)))
    x0, x1 = int(max(0, np.floor(cx - half))), int(min(W, np.ceil(cx + half)))
    rgb = resize(np.ascontiguousarray(img[y0:y1, x0:x1]), (size, size))
    m = mask[y0:y1, x0:x1]
    vi = np.minimum((np.arange(size) * m.shape[0] / size).astype(int), m.shape[0] - 1)
    ui = np.minimum((np.arange(size) * m.shape[1] / size).astype(int), m.shape[1] - 1)
    return rgb, m[np.ix_(vi, ui)]


def patch_fraction(mask: np.ndarray) -> np.ndarray:
    """(S/14, S/14) share of mask pixels per 14 x 14 patch of a square mask of side S."""
    S = mask.shape[0]
    g = S // PATCH
    return mask[: g * PATCH, : g * PATCH].reshape(g, PATCH, g, PATCH).mean(axis=(1, 3))


class IdentityScorer:
    """Appearance similarity of (reference node, observation) pairs; one instance per pair of scans."""

    def __init__(self, seq: Any, seq_info: Any, tex: Any, renderer: Any, extractor: Any, cfg: Mapping[str, Any]):
        """
        Args:
            seq: The rescan's open sequence (`color(idx)` = stored-orientation RGB).
            seq_info: `SeqInfo` (colour / depth sizes in `intrinsics`).
            tex: `geom.render.TextureRenderer` of the reference.
            renderer: `geom.render.Renderer` of the reference mesh with objectId face labels.
            extractor: Patch feature extractor (`perception.appearance.make_extractor`).
            cfg: Full config (`association.identity`).
        """
        self.seq, self.info, self.tex, self.renderer, self.ex = seq, seq_info, tex, renderer, extractor
        ic = cfg["association"]["identity"]
        self.k_obs, self.k_node = int(ic.get("k_obs", 3)), int(ic.get("k_node", 2))
        self.min_patches = int(ic.get("min_patches", 4))
        self.margin = float(ic.get("crop_margin", 0.15))
        self.size = PATCH * max(1, round(int(ic.get("crop_px", 224)) / PATCH))
        self.view_px = int(ic.get("view_px", 448))
        self._obs: dict[str, list[np.ndarray]] = {}
        self._node: dict[str, list[np.ndarray]] = {}
        self.log: list[tuple[str, str, float | None]] = []

    # ---- descriptors -------------------------------------------------------------------------

    def _pool(self, crops: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        if not crops:
            return []
        F = self.ex.features([c[0] for c in crops])
        out = []
        for f, (_, m) in zip(F, crops):
            w = patch_fraction(m.astype(np.float32)) >= 0.5
            if w.shape != f.shape[:2] or w.sum() < self.min_patches:
                continue
            v = f[w].mean(axis=0)
            nv = float(np.linalg.norm(v))
            if nv > 0:
                out.append(v / nv)
        return out

    def obs_descriptor(self, o: ObsInstance) -> list[np.ndarray]:
        """Pooled feature vectors of the observation's `k_obs` largest-mask frames (memoised)."""
        if o.token in self._obs:
            return self._obs[o.token]
        masks = o.masks
        area = {f: (masks.area(f) if hasattr(masks, "area") else int(np.asarray(masks[f]).sum())) for f in masks}
        frames = sorted(area, key=lambda f: (-area[f], f))[: self.k_obs]
        crops = []
        for f in frames:
            if area[f] == 0:
                continue
            rgb = np.asarray(self.seq.color(int(f)))
            m = np.asarray(masks[f], bool)
            Hc, Wc = rgb.shape[:2]
            vi = np.minimum((np.arange(Hc) * m.shape[0] / Hc).astype(int), m.shape[0] - 1)
            ui = np.minimum((np.arange(Wc) * m.shape[1] / Wc).astype(int), m.shape[1] - 1)
            mc = m[np.ix_(vi, ui)]
            c = square_crop(np.rot90(rgb, k=-1), np.rot90(mc, k=-1), self.margin, self.size)
            if c is not None:
                crops.append(c)
        self._obs[o.token] = self._pool(crops)
        return self._obs[o.token]

    def node_descriptor(self, n: Node) -> list[np.ndarray]:
        """Pooled feature vectors of the node's `k_node` best virtual reference views (memoised)."""
        if n.id in self._node:
            return self._node[n.id]
        oid = object_id_of(n.id)
        out: list[np.ndarray] = []
        if oid is not None and n.bbox_3d is not None:
            lo, hi = np.asarray(n.bbox_3d["min"], float), np.asarray(n.bbox_3d["max"], float)
            c = 0.5 * (lo + hi)
            r = float(np.clip(1.2 * np.linalg.norm(hi - lo) + 0.8, 1.0, 3.0))
            S = self.view_px
            f = 0.5 * S / np.tan(np.radians(30.0))
            K = np.array([[f, 0, 0.5 * S], [0, f, 0.5 * S], [0, 0, 1.0]])
            el = np.radians(25.0)
            views = []
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                eye = c + r * np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
                Tw2c = invert(look_at(eye, c))
                _, lab = self.renderer.render(K, Tw2c, S, S)
                m = np.asarray(lab) == oid
                if m.sum() >= self.min_patches * PATCH * PATCH // 4:
                    views.append((int(m.sum()), Tw2c, m))
            views.sort(key=lambda v: -v[0])
            crops = []
            for _, Tw2c, m in views[: self.k_node]:
                rgb, _ = self.tex.render_color(K, Tw2c, S, S)
                cr = square_crop(rgb, m, self.margin, self.size)
                if cr is not None:
                    crops.append(cr)
            out = self._pool(crops)
        self._node[n.id] = out
        return out

    def similarity(self, n: Node, o: ObsInstance) -> float | None:
        """Largest cosine between node views and observation crops; None without descriptors. Logged."""
        a, b = self.node_descriptor(n), self.obs_descriptor(o)
        s = None if not a or not b else round(float(max(float(x @ y) for x in a for y in b)), 4)
        self.log.append((n.id, o.token, s))
        return s
