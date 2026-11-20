"""SAM segment cache for co-segmentation (CHANGE 3 Part N, N1; `uv run lsg sam cache`).

Per processed rescan frame: the upright colour image -> SAM ViT-B automatic masks
(`SamAutomaticMaskGenerator`, `perception.segment.sam`) -> one label map (masks painted largest
first, so smaller masks win where they overlap; -1 = no mask) -> rotated back to the stored
orientation and sampled at the depth pixel centres (colour and depth share the field of view) ->
`<cache_root>/sam/<key>/<rescan>/<frame>.npz` (`labels`, int16 at depth resolution).

`key` hashes the model and the generator parameters, so a parameter change gets its own folder.
Method side: reads only the rescan's colour frames (hard rules 1-2). The model is loaded lazily;
`segment_anything` is needed only to build the cache, not to read it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lsg.config import cache_root

WEIGHTS = {"vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")}


def sam_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The `perception.segment.sam` block."""
    return (cfg["perception"].get("segment") or {}).get("sam") or {}


def gen_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """`SamAutomaticMaskGenerator` keyword arguments from the config."""
    s = sam_cfg(cfg)
    return {"points_per_side": int(s.get("points_per_side", 16)),
            "pred_iou_thresh": float(s.get("pred_iou_thresh", 0.86)),
            "stability_score_thresh": float(s.get("stability_score_thresh", 0.92)),
            "min_mask_region_area": int(s.get("min_mask_region_area", 0))}  # fmt: skip


def cache_key(cfg: Mapping[str, Any]) -> str:
    """Hash of the model and generator parameters (12 hex chars)."""
    blob = json.dumps({"model": str(sam_cfg(cfg).get("model", "vit_b")), **gen_params(cfg)}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def rescan_dir(cfg: Mapping[str, Any], rescan_id: str) -> Path:
    """Cache folder of one rescan's label maps."""
    return cache_root(dict(cfg)) / "sam" / cache_key(cfg) / rescan_id


def paint(masks: Sequence[Mapping[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    """One label map from SAM mask records (`segmentation`, `area`): largest first, -1 = none."""
    lab = np.full(shape, -1, np.int32)
    for k, m in enumerate(sorted(masks, key=lambda r: -int(r["area"]))):
        lab[np.asarray(m["segmentation"], bool)] = k
    return lab


def to_depth_grid(lab_upright: np.ndarray, depth_wh: tuple[int, int]) -> np.ndarray:
    """Upright colour-resolution label map -> stored orientation, sampled at depth pixel centres."""
    stored = np.rot90(lab_upright, k=1)  # inverse of the viewing rotation rot90(k=-1)
    Hc, Wc = stored.shape
    Wd, Hd = depth_wh
    v = np.minimum(((np.arange(Hd) + 0.5) * Hc / Hd).astype(int), Hc - 1)
    u = np.minimum(((np.arange(Wd) + 0.5) * Wc / Wd).astype(int), Wc - 1)
    return stored[np.ix_(v, u)]


class SamRunner:
    """Lazily loaded SAM automatic mask generator."""

    def __init__(self, cfg: Mapping[str, Any], device: str = "cuda"):
        self.cfg, self.device = cfg, device
        self._gen = None

    def _load(self):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model = str(sam_cfg(self.cfg).get("model", "vit_b"))
        fname, url = WEIGHTS[model]
        path = cache_root(dict(self.cfg)) / "models" / fname
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path))
        dev = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = dev
        sam = sam_model_registry[model](checkpoint=str(path)).to(dev)
        self._gen = SamAutomaticMaskGenerator(sam, **gen_params(self.cfg))

    def labels(self, rgb_stored: np.ndarray, depth_wh: tuple[int, int]) -> np.ndarray:
        """Label map at depth resolution (stored orientation) of one stored-orientation colour image."""
        if self._gen is None:
            self._load()
        up = np.ascontiguousarray(np.rot90(rgb_stored, k=-1))
        return to_depth_grid(paint(self._gen.generate(up), up.shape[:2]), depth_wh)


def cache_rescan(cfg: Mapping[str, Any], root: str | Path, rescan_id: str, frame_ids: Sequence[int],
                 runner: SamRunner) -> dict[str, Any]:  # fmt: skip
    """Cache the label maps of `frame_ids` (skips frames already cached). Returns {"done", "skipped", "s"}."""
    from lsg.io.scan3r import open_sequence

    out = rescan_dir(cfg, rescan_id)
    out.mkdir(parents=True, exist_ok=True)
    t0, done, skipped = time.time(), 0, 0
    with open_sequence(root, rescan_id) as seq:
        wh = seq.info.depth_wh
        for f in frame_ids:
            p = out / f"{int(f):06d}.npz"
            if p.exists():
                skipped += 1
                continue
            lab = runner.labels(seq.color(int(f)), wh)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, labels=lab.astype(np.int16))
            tmp.replace(p)
            done += 1
    return {"done": done, "skipped": skipped, "s": round(time.time() - t0, 1)}
