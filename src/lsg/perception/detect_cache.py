"""Open-vocabulary detection cache (CHANGE 3 Part E; `uv run lsg detect cache`).

Per detected rescan frame (every `perception.detector.frame_every`-th processed frame, by position
in the strided sequence, so observed-fraction chunks still hit the cache): the upright colour image
-> **Grounding DINO** (one pass; text prompt = the pair's perception vocabulary, reference labels
first, then train labels by frequency, cut to the text-token limit) -> per query a score for every
prompt label (mean sigmoid logit over the label's tokens), box score = the best label's score ->
class-agnostic NMS -> **SAM 2** with the boxes as prompts -> one mask per box -> rotated back to the
stored orientation and sampled at the depth pixel centres (`sam_cache.to_depth_grid`) ->
`<cache_root>/detect/<key>/<rescan>_<prompts>/<frame>.npz`: `boxes` (N, 4, upright colour pixels),
`scores` (N,), `labels` (N, best label index into `prompts.json`), `label_scores` (N, n_prompts,
float16), `masks` (N, H_d, W_d) bit-packed.

Detections are stored down to `cache_box_threshold`, so `perception.detector.min_score` can be
tuned without recomputing. `key` hashes the models and the cache-time parameters; the folder name
also carries a hash of the prompt list, since the vocabulary depends on the reference.

Method side: reads only the rescan's colour frames (hard rules 1-2). The models (Hugging Face
`transformers`) are loaded lazily; reading the cache needs no model.
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
from lsg.perception.sam_cache import to_depth_grid

MAX_TEXT_TOKENS = 256  # Grounding DINO's text length, incl. [CLS] / [SEP]


def det_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The `perception.detector` block."""
    return cfg["perception"].get("detector") or {}


def cache_key(cfg: Mapping[str, Any]) -> str:
    """Hash of the models and the cache-time parameters (12 hex chars)."""
    d = det_cfg(cfg)
    keys = ("gdino_model", "sam_model", "cache_box_threshold", "image_short_side", "nms_iou", "max_detections")
    return hashlib.sha1(json.dumps({k: d.get(k) for k in keys}, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def prompts_key(prompts: Sequence[str]) -> str:
    """Hash of a prompt list (8 hex chars)."""
    return hashlib.sha1("\n".join(prompts).encode("utf-8")).hexdigest()[:8]


def rescan_dir(cfg: Mapping[str, Any], rescan_id: str, prompts: Sequence[str]) -> Path:
    """Cache folder of one rescan's detections for one (ordered, untruncated) prompt list."""
    return cache_root(dict(cfg)) / "detect" / cache_key(cfg) / f"{rescan_id}_{prompts_key(prompts)}"


def detect_frames(frame_ids: Sequence[int], stride: int, every: int) -> list[int]:
    """Frames that get detections: every `every`-th frame of the strided sequence (`select_frames` at frac 1)."""
    return list(frame_ids)[:: int(stride)][:: max(int(every), 1)]


def ordered_prompts(ref_labels: Sequence[str], table_labels: Sequence[str], vocab: Sequence[str]) -> list[str]:
    """The vocabulary in prompt priority: reference labels first, then train labels by frequency, then the rest."""
    v = set(vocab)
    out: list[str] = []
    for lab in [*sorted(set(ref_labels)), *table_labels, *sorted(v)]:
        if lab in v and lab not in out:
            out.append(lab)
    return out


def fit_prompts(prompts: Sequence[str], token_ids, max_tokens: int = MAX_TEXT_TOKENS) -> tuple[list[str], list[tuple[int, int]]]:
    """Leading prompts whose "a . b . c ." text fits `max_tokens` (incl. [CLS] / [SEP]), and each one's token span.

    `token_ids(text)` gives the tokens of a text without special tokens. Spans are [start, end) positions in
    the model's `input_ids` (position 0 = [CLS]); each label is followed by one "." token.
    """
    used, pos = [], 1
    spans: list[tuple[int, int]] = []
    for p in prompts:
        n = len(token_ids(p.lower()))
        if pos + n + 1 + 1 > max_tokens:  # label, ".", then [SEP]
            break
        used.append(p)
        spans.append((pos, pos + n))
        pos += n + 1
    return used, spans


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by class-agnostic non-maximum suppression (descending score)."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    for i in order:
        ok = True
        for j in keep:
            x0, y0 = np.maximum(boxes[i, :2], boxes[j, :2])
            x1, y1 = np.minimum(boxes[i, 2:], boxes[j, 2:])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if inter / max(area[i] + area[j] - inter, 1e-9) > iou:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return np.asarray(keep, int)


def label_scores(probs: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """(Q, n_labels): mean token probability of each label's span, per query."""
    return np.stack([probs[:, a:b].mean(axis=1) for a, b in spans], axis=1)


class DetectorRunner:
    """Lazily loaded Grounding DINO + SAM 2 (Hugging Face `transformers`)."""

    def __init__(self, cfg: Mapping[str, Any], device: str = "cuda"):
        self.cfg, self.device = cfg, device
        self._gd = self._sam = None
        self.sam_kind: str | None = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        d = det_cfg(self.cfg)
        dev = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = dev
        self._gd_proc = AutoProcessor.from_pretrained(d["gdino_model"])
        self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(d["gdino_model"]).to(dev).eval()
        try:
            from transformers import Sam2Model, Sam2Processor

            self._sam_proc = Sam2Processor.from_pretrained(d["sam_model"])
            self._sam = Sam2Model.from_pretrained(d["sam_model"]).to(dev).eval()
            self.sam_kind = "sam2"
        except Exception as e:  # pragma: no cover - SAM 1 fallback (Erik, 2026-11-21)
            print(f"[detect] SAM 2 unavailable ({e}); falling back to SAM 1", flush=True)
            from segment_anything import SamPredictor, sam_model_registry

            from lsg.perception.sam_cache import WEIGHTS

            fname, url = WEIGHTS["vit_b"]
            path = cache_root(dict(self.cfg)) / "models" / fname
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(url, str(path))
            self._sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(path)).to(dev))
            self.sam_kind = "sam1"

    def token_ids(self, text: str) -> list[int]:
        """Grounding DINO tokens of a text, without special tokens."""
        if self._gd is None:
            self._load()
        return self._gd_proc.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detect(self, rgb_stored: np.ndarray, prompts: Sequence[str], depth_wh: tuple[int, int]) -> dict[str, np.ndarray]:
        """Detections of one stored-orientation colour frame (labels index `prompts`, which must fit one pass)."""
        import torch
        from PIL import Image

        if self._gd is None:
            self._load()
        d = det_cfg(self.cfg)
        used, spans = fit_prompts(prompts, self.token_ids)
        assert len(used) == len(prompts), "prompts must fit one pass (use fit_prompts first)"
        up = np.ascontiguousarray(np.rot90(rgb_stored, k=-1))
        H, W = up.shape[:2]
        img = Image.fromarray(up)
        short = int(d.get("image_short_side", 540))
        size = {"shortest_edge": short, "longest_edge": int(round(short * max(H, W) / min(H, W)))}
        text = " . ".join(p.lower() for p in prompts) + " ."
        inp = self._gd_proc(images=img, text=text, return_tensors="pt", size=size).to(self.device)
        with torch.no_grad():
            out = self._gd(**inp)
        probs = out.logits[0].sigmoid().cpu().numpy()  # (Q, 256)
        LS = label_scores(probs, spans)  # (Q, n)
        sc = LS.max(axis=1)
        keep = np.nonzero(sc >= float(d["cache_box_threshold"]))[0]
        Wd, Hd = depth_wh
        empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
                 "labels": np.zeros(0, np.int16), "label_scores": np.zeros((0, len(prompts)), np.float16),
                 "masks": np.zeros((0, Hd, Wd), bool)}  # fmt: skip
        if len(keep) == 0:
            return empty
        cxcywh = out.pred_boxes[0].cpu().numpy()[keep]
        B = np.stack([(cxcywh[:, 0] - cxcywh[:, 2] / 2) * W, (cxcywh[:, 1] - cxcywh[:, 3] / 2) * H,
                      (cxcywh[:, 0] + cxcywh[:, 2] / 2) * W, (cxcywh[:, 1] + cxcywh[:, 3] / 2) * H], axis=1)  # fmt: skip
        B = np.clip(B, 0, [W, H, W, H]).astype(np.float32)
        S, LSk = sc[keep].astype(np.float32), LS[keep]
        k2 = nms(B, S, float(d.get("nms_iou", 0.7)))[: int(d.get("max_detections", 60))]
        B, S, LSk = B[k2], S[k2], LSk[k2]
        ok = (B[:, 2] - B[:, 0] >= 2) & (B[:, 3] - B[:, 1] >= 2)
        B, S, LSk = B[ok], S[ok], LSk[ok]
        if len(S) == 0:
            return empty
        masks_up = self._masks(up, img, B)
        masks = np.stack([to_depth_grid(m.astype(np.int32), depth_wh) > 0 for m in masks_up])
        return {"boxes": B, "scores": S, "labels": LSk.argmax(axis=1).astype(np.int16),
                "label_scores": LSk.astype(np.float16), "masks": masks}  # fmt: skip

    def _masks(self, up: np.ndarray, img, B: np.ndarray) -> np.ndarray:
        """(N, H, W) bool masks of the boxes on the upright image."""
        import torch

        if self.sam_kind == "sam2":
            inp = self._sam_proc(images=img, input_boxes=[B.tolist()], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._sam(**inp, multimask_output=False)
            m = self._sam_proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
            return m[:, 0].numpy().astype(bool)
        self._sam.set_image(up)  # pragma: no cover - SAM 1 fallback
        tb = self._sam.transform.apply_boxes_torch(torch.as_tensor(B, device=self.device), up.shape[:2])
        with torch.no_grad():
            m, _, _ = self._sam.predict_torch(None, None, boxes=tb, multimask_output=False)
        return m[:, 0].cpu().numpy().astype(bool)


def save_frame(path: Path, det: Mapping[str, np.ndarray]) -> None:
    """Write one frame's detections (masks bit-packed), atomically."""
    m = np.asarray(det["masks"], bool)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, boxes=det["boxes"], scores=det["scores"], labels=det["labels"],
                        label_scores=det["label_scores"], masks=np.packbits(m.reshape(len(m), -1), axis=1),
                        shape=np.asarray(m.shape[1:]))  # fmt: skip
    tmp.replace(path)


def load_frame(path: Path) -> dict[str, np.ndarray]:
    """Read one frame's detections (masks unpacked to bool (N, H, W))."""
    z = np.load(path)
    H, W = (int(v) for v in z["shape"])
    n = len(z["scores"])
    masks = (np.unpackbits(z["masks"], axis=1, count=H * W).reshape(n, H, W).astype(bool) if n
             else np.zeros((0, H, W), bool))  # fmt: skip
    return {"boxes": z["boxes"], "scores": z["scores"], "labels": z["labels"],
            "label_scores": z["label_scores"].astype(np.float32), "masks": masks}  # fmt: skip


class DetectionCache:
    """Read access to one rescan's cached detections (method side)."""

    def __init__(self, cfg: Mapping[str, Any], rescan_id: str, prompts: Sequence[str]):
        self.dir = rescan_dir(cfg, rescan_id, prompts)
        p = self.dir / "prompts.json"
        self.prompts: list[str] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []

    def frame(self, fidx: int) -> dict[str, np.ndarray] | None:
        """Detections of frame `fidx`, or None if the frame was not detected (not in the cache)."""
        p = self.dir / f"{int(fidx):06d}.npz"
        return load_frame(p) if p.is_file() else None

    def n_frames(self) -> int:
        """Number of cached frames."""
        return len(list(self.dir.glob("*.npz"))) if self.dir.is_dir() else 0


def cache_rescan(cfg: Mapping[str, Any], root: str | Path, rescan_id: str, prompts: Sequence[str],
                 frame_ids: Sequence[int], runner: DetectorRunner) -> dict[str, Any]:  # fmt: skip
    """Cache the detections of `frame_ids` (skips cached frames). `prompts` = the ordered vocabulary; the
    leading part that fits one pass is used and stored in `prompts.json`. Returns {"done", "skipped", "s", "n_det"}."""
    from lsg.io.scan3r import open_sequence

    out = rescan_dir(cfg, rescan_id, prompts)
    out.mkdir(parents=True, exist_ok=True)
    used, _ = fit_prompts(prompts, runner.token_ids)
    (out / "prompts.json").write_text(json.dumps(used), encoding="utf-8")
    t0, done, skipped, n_det = time.time(), 0, 0, 0
    with open_sequence(root, rescan_id) as seq:
        wh = seq.info.depth_wh
        for f in frame_ids:
            p = out / f"{int(f):06d}.npz"
            if p.exists():
                skipped += 1
                continue
            det = runner.detect(seq.color(int(f)), used, wh)
            save_frame(p, det)
            done += 1
            n_det += len(det["scores"])
    return {"done": done, "skipped": skipped, "s": round(time.time() - t0, 1), "n_det": n_det, "n_prompts": len(used)}
