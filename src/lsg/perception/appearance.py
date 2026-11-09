"""Appearance residual (DESIGN §3, §6.6; M6): rendered reference texture vs observed colour.

Per processed frame the reference texture is rendered at the rescan camera's aligned pose
(`geom.render.TextureRenderer`, colour intrinsics scaled to `render_w`), and both images are
compared patch by patch with exposure-robust features:

- `DinoV2Features` (default): frozen DINOv2 ViT-S/14 patch tokens (timm), upright input;
- `ColorFeatures` (fallback, fast tests): per-patch mean/std-normalised colour cells.

`patch_similarity` gives one cosine similarity per 14 x 14 patch, in the stored (rotated)
orientation. `mismatch` turns it into depth-resolution maps:

- a depth pixel is *comparable* if depth neither shows an occluder nor sees through the
  reference: valid and -tau_near <= d_obs - d_exp <= tau_free (M6, Erik: thin nodes up to
  tau_free thick are compared too);
- a patch is compared if >= `min_comparable_frac` of its depth pixels are comparable;
- a comparable pixel of a compared patch *mismatches* if its patch distance 1 - sim, minus the
  frame's median distance over compared patches (`relative`, M6: rescans are lit differently
  from the visit the texture was baked from; own sequences have a median of ~0.25, rescans
  ~0.42), exceeds `tau_app`. Frames with < `min_frame_patches` compared patches compare nothing.

`rel_px` holds every comparable pixel's patch distance relative to the frame median. The maps
feed (i) thin-node votes (`change.visibility.thin_votes`: the node's *mean* relative distance
over its comparable pixels against `tau_node`, M6 dev finding: a per-patch threshold at the
same false rate almost never fires on a whole node) and (ii) thin-object voxels in
`PerceptionSource` (per-pixel `mismatch_px`, threshold `tau_app`). Similarities are cached per (rescan, frame, aligned
pose, reference, features, render size) in `cache/appearance/`, rounded to 1e-3 (§7.5).

Method side: reads only the reference texture and the rescan's colour frames + poses.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lsg.geom.transforms import invert

PATCH = 14  # DINOv2 ViT-S/14 patch size; the colour fallback uses the same grid


class FeatureExtractor(Protocol):
    """Patch features of upright images whose sides are multiples of `PATCH`."""

    name: str

    def features(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        """(B, H/14, W/14, D) float32 features of B upright (H, W, 3) uint8 images."""


class ColorFeatures:
    """Exposure-normalised colour per patch (fallback, DESIGN §3).

    Each image is normalised per channel by its own mean / std (removes global exposure and
    white-balance differences); a patch's feature is the mean of each 7 x 7 cell (2 x 2 cells x
    3 channels) plus a constant 1, so a flat patch never becomes a zero vector.
    """

    name = "color"
    device = "cpu"

    def features(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for im in imgs:
            a = np.asarray(im, np.float32)
            a = (a - a.reshape(-1, 3).mean(0)) / (a.reshape(-1, 3).std(0) + 1e-3)
            H, W = a.shape[:2]
            c = PATCH // 2
            cells = a.reshape(H // c, c, W // c, c, 3).mean(axis=(1, 3))  # (H/7, W/7, 3)
            g = cells.reshape(H // PATCH, 2, W // PATCH, 2, 3).transpose(0, 2, 1, 3, 4)
            g = g.reshape(H // PATCH, W // PATCH, 12)
            out.append(np.concatenate([g, np.ones((*g.shape[:2], 1), np.float32)], axis=-1))
        return np.stack(out)


class DinoV2Features:
    """Frozen DINOv2 patch tokens via timm (lazy torch import, CUDA -> CPU fallback)."""

    _MODELS: dict[tuple[str, str], Any] = {}

    def __init__(self, model: str = "vit_small_patch14_dinov2.lvd142m", device: str = "cuda"):
        self.model_name = model
        self.name = f"dinov2:{model}"
        self.req_device = device
        self.device: str | None = None
        self._m = None
        self.n_images = 0

    def _load(self):
        if self._m is not None:
            return
        import timm
        import torch

        dev = self.req_device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        key = (self.model_name, dev)
        if key not in self._MODELS:
            torch.manual_seed(0)
            m = timm.create_model(self.model_name, pretrained=True, dynamic_img_size=True).eval().to(dev)
            self._MODELS[key] = m
        self._m = self._MODELS[key]
        cfg = self._m.pretrained_cfg
        self._mean = np.asarray(cfg.get("mean", (0.485, 0.456, 0.406)), np.float32)
        self._std = np.asarray(cfg.get("std", (0.229, 0.224, 0.225)), np.float32)

    def features(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        self._load()
        x = np.stack([(np.asarray(i, np.float32) / 255.0 - self._mean) / self._std for i in imgs])
        H, W = x.shape[1:3]
        t = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)
        with torch.no_grad():
            f = self._m.forward_features(t)[:, self._m.num_prefix_tokens :].float()
        self.n_images += len(imgs)
        return f.cpu().numpy().reshape(len(imgs), H // PATCH, W // PATCH, -1)


def _per_feature(cfg: Mapping[str, Any], key: str) -> float:
    ac = cfg["appearance"]
    t = ac[key]
    return float(t[ac.get("features", "dinov2")] if isinstance(t, Mapping) else t)


def tau_app(cfg: Mapping[str, Any]) -> float:
    """Per-pixel (patch) mismatch threshold of the configured feature type (`appearance.tau_app`)."""
    return _per_feature(cfg, "tau_app")


def tau_node(cfg: Mapping[str, Any]) -> float:
    """Thin-node threshold on the mean relative distance (`appearance.tau_node`)."""
    return _per_feature(cfg, "tau_node")


def make_extractor(cfg: Mapping[str, Any]) -> FeatureExtractor:
    """Extractor from `cfg["appearance"]` (`features: dinov2 | color`)."""
    ac = cfg["appearance"]
    if ac.get("features", "dinov2") == "color":
        return ColorFeatures()
    return DinoV2Features(ac["model"], ac.get("device", "cuda"))


def render_size(color_wh: tuple[int, int], render_w: int) -> tuple[int, int]:
    """(W, H) of the stored-orientation render: width `render_w`, colour aspect, both multiples of 14."""
    CW, CH = color_wh
    w = PATCH * max(1, round(render_w / PATCH))
    h = PATCH * max(1, round(w * CH / CW / PATCH))
    return w, h


def resize(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Area-filtered resize of a uint8 RGB image to (W, H)."""
    from PIL import Image

    if (img.shape[1], img.shape[0]) == tuple(wh):
        return img
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize(tuple(wh), Image.BOX))


def patch_similarity(rgb_obs: np.ndarray, rgb_exp: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    """Per-patch cosine similarity of two stored-orientation images of equal size.

    Both are rotated upright (`rot90 k=-1`) for the feature extractor; the (gh, gw) result is
    rotated back to the stored orientation (exact, because both sides are multiples of 14).
    """
    a, b = np.rot90(rgb_obs, k=-1), np.rot90(rgb_exp, k=-1)
    f = extractor.features([np.ascontiguousarray(a), np.ascontiguousarray(b)])
    fa, fb = f[0], f[1]
    num = (fa * fb).sum(-1)
    den = np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1)
    sim = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return np.rot90(sim, k=1).astype(np.float32)


@dataclass
class AppearanceMap:
    """One frame's appearance comparison (stored orientation)."""

    sim: np.ndarray  # (gh, gw) per-patch cosine similarity
    patch_ok: np.ndarray  # (gh, gw) bool, compared patches
    comparable_px: np.ndarray  # (H, W) bool at depth resolution
    mismatch_px: np.ndarray  # (H, W) bool at depth resolution
    rel_px: np.ndarray | None = None  # (H, W) float32 frame-relative patch distance, 0 off comparable

    @property
    def match_px(self) -> np.ndarray:
        """Comparable pixels whose patch agrees."""
        return self.comparable_px & ~self.mismatch_px


def comparable_pixels(d_obs: np.ndarray, d_exp: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Depth pixels where appearance may be compared: valid, not nearer, not seen through."""
    vc = cfg["visibility"]
    diff = d_obs - d_exp
    return (d_obs > 0) & (d_exp > 0) & (diff >= -float(vc["tau_near"])) & (diff <= float(vc["tau_free"]))


def patch_index(depth_wh: tuple[int, int], grid_hw: tuple[int, int], render_wh: tuple[int, int]) -> np.ndarray:
    """(H, W) flat patch index of every depth pixel (pixel centres, per-axis resolution scale)."""
    W, H = depth_wh
    gh, gw = grid_hw
    RW, RH = render_wh
    r = np.minimum(((np.arange(H) + 0.5) * RH / H // PATCH).astype(np.int64), gh - 1)
    c = np.minimum(((np.arange(W) + 0.5) * RW / W // PATCH).astype(np.int64), gw - 1)
    return r[:, None] * gw + c[None, :]


def mismatch(
    sim: np.ndarray,
    d_obs: np.ndarray,
    d_exp: np.ndarray,
    cfg: Mapping[str, Any],
    render_wh: tuple[int, int],
) -> AppearanceMap:
    """Depth-resolution comparable / mismatch maps from a per-patch similarity (module docstring)."""
    ac = cfg["appearance"]
    H, W = d_obs.shape
    gh, gw = sim.shape
    pidx = patch_index((W, H), (gh, gw), render_wh)
    comp = comparable_pixels(np.asarray(d_obs, np.float64), np.asarray(d_exp, np.float64), cfg)
    n_all = np.bincount(pidx.ravel(), minlength=gh * gw)
    n_comp = np.bincount(pidx.ravel(), weights=comp.ravel(), minlength=gh * gw)
    ok = (n_all > 0) & (n_comp >= float(ac["min_comparable_frac"]) * np.maximum(n_all, 1))
    if ok.sum() < int(ac.get("min_frame_patches", 1)):
        ok[:] = False  # too few compared patches for a frame reference level
    dist = 1.0 - sim.ravel().astype(np.float64)
    if ac.get("relative", True) and ok.any():
        dist = dist - float(np.median(dist[ok]))  # frame-relative: removes global lighting / exposure shifts
    dist = np.round(dist, 6)
    bad = ok & (dist > tau_app(cfg))
    comp_px = comp & ok[pidx]
    rel_px = np.where(comp_px, dist[pidx], 0.0).astype(np.float32)
    return AppearanceMap(sim, ok.reshape(gh, gw), comp_px, comp_px & bad[pidx], rel_px)


class SimilarityCache:
    """Per-frame patch similarities on disk (`cache/appearance/<key>.json`), rounded to 1e-3."""

    def __init__(self, folder: str | Path | None):
        self.folder = Path(folder) if folder is not None else None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(parts: Sequence[Any], pose: np.ndarray) -> str:
        h = hashlib.sha256(json.dumps([str(p) for p in parts]).encode("utf-8"))
        h.update(np.round(np.asarray(pose, np.float64), 4).tobytes())
        return h.hexdigest()[:24]

    def get(self, key: str) -> np.ndarray | None:
        if self.folder is None:
            return None
        p = self.folder / f"{key}.json"
        if not p.is_file():
            return None
        self.hits += 1
        return np.asarray(json.loads(p.read_text(encoding="utf-8")), np.float32)

    def put(self, key: str, sim: np.ndarray) -> np.ndarray:
        """Round, store, and return the rounded similarity (identical on cold and cached runs)."""
        s = np.round(np.asarray(sim, np.float64), 3).astype(np.float32)
        self.misses += 1
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / f"{key}.json").write_text(json.dumps(np.round(s.astype(np.float64), 3).tolist()),
                                                     encoding="utf-8")  # fmt: skip
        return s


class AppearanceRunner:
    """Per-frame appearance maps for `change.online.run_online` (DESIGN §6.7 step 1).

    Holds the reference `TextureRenderer`, the feature extractor and the similarity cache.
    `begin(seq_info, seq)` binds a rescan; `frame(frame, pose_ref, d_exp)` returns the frame's
    `AppearanceMap` (colour read from the sequence, texture rendered at `pose_ref`).
    """

    def __init__(self, tex_renderer, extractor: FeatureExtractor, cfg: Mapping[str, Any],
                 ref_id: str = "", cache: SimilarityCache | None = None):  # fmt: skip
        self.tr = tex_renderer
        self.ext = extractor
        self.cfg = cfg
        self.ref_id = ref_id
        self.cache = cache or SimilarityCache(None)
        self.n_frames = 0
        self.time_s = 0.0

    def begin(self, seq_info, seq) -> None:
        """Bind a rescan (intrinsics, id) and its open sequence (colour frames)."""
        self.info, self.seq = seq_info, seq
        intr = seq_info.intrinsics
        self.wh = render_size(intr.color_wh, int(self.cfg["appearance"]["render_w"]))
        K = np.asarray(intr.K_color, float).copy()
        K[0] *= self.wh[0] / intr.color_wh[0]
        K[1] *= self.wh[1] / intr.color_wh[1]
        self.K = K

    def similarity(self, frame_idx: int, pose_ref: np.ndarray) -> np.ndarray:
        """Cached per-patch similarity of the observed vs the rendered colour at one frame."""
        parts = [self.info.rescan_id, int(frame_idx), self.ref_id, self.ext.name, *self.wh]
        k = self.cache.key(parts, pose_ref)
        s = self.cache.get(k)
        if s is not None:
            return s
        obs = resize(self.seq.color(frame_idx), self.wh)
        exp, _ = self.tr.render_color(self.K, invert(pose_ref), *self.wh)
        return self.cache.put(k, patch_similarity(obs, exp, self.ext))

    def frame(self, frame, pose_ref: np.ndarray, d_exp: np.ndarray) -> AppearanceMap:
        """`AppearanceMap` of one processed frame."""
        import time

        t = time.time()
        sim = self.similarity(int(frame.idx), pose_ref)
        out = mismatch(sim, frame.depth, d_exp, self.cfg, self.wh)
        self.n_frames += 1
        self.time_s += time.time() - t
        return out
