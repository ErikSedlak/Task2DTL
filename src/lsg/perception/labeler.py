"""Zero-shot candidate labelling with CLIP (frozen ViT-B/32, `open_clip_torch`).

`CLIPLabeler.topk(crops)` encodes the upright colour crops of one candidate (up to K = 5),
averages their normalised image embeddings, and scores the average against the text
embeddings of the vocabulary (`prompt.format(label)`): softmax over the vocabulary, top-k
`(label, score)`. Scores are rounded to 1e-4 and ties broken by label, and results are cached
on disk per (crops, model, vocabulary), so reruns read identical labels.

torch / open_clip are imported lazily: importing `lsg.perception` stays cheap, and tests use
`StubLabeler`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Labeler(Protocol):
    """Anything that turns a candidate's crops into `label_topk`."""

    def topk(self, crops: Sequence[np.ndarray], k: int) -> list[tuple[str, float]]:
        """Top-k `(label, score)`, best first; empty if there are no crops."""


class StubLabeler:
    """Fixed answer, for tests (no model)."""

    def __init__(self, topk: Sequence[tuple[str, float]] = (("object", 1.0),)):
        self._topk = [(str(a), float(b)) for a, b in topk]
        self.calls: list[int] = []

    def topk(self, crops: Sequence[np.ndarray], k: int) -> list[tuple[str, float]]:
        """The fixed list (cut to k); records the number of crops of each call."""
        self.calls.append(len(crops))
        return self._topk[:k]


def _vocab_hash(vocab: Sequence[str], prompt: str) -> str:
    """Short hash of the vocabulary + prompt."""
    return hashlib.sha256(json.dumps([prompt, list(vocab)]).encode("utf-8")).hexdigest()[:16]


class CLIPLabeler:
    """open_clip model + cached text features of one vocabulary."""

    def __init__(
        self,
        vocab: Sequence[str],
        model: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        prompt: str = "a photo of a {}",
        cache_dir: str | Path | None = None,
        threads: int | None = None,
    ):
        """
        Args:
            vocab: Candidate labels (duplicates removed, sorted).
            model, pretrained: open_clip model name and weights tag.
            device: "cuda" or "cpu"; "cuda" falls back to CPU if unavailable.
            prompt: Text template with one `{}` for the label.
            cache_dir: Folder for cached results (None = no cache).
            threads: torch CPU threads (None = torch default).
        """
        self.vocab = sorted(set(map(str, vocab)))
        self.model_name, self.pretrained, self.prompt = model, pretrained, prompt
        self.req_device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.threads = threads
        self.vhash = _vocab_hash(self.vocab, prompt)
        self.device: str | None = None
        self._model = None
        self._pre = None
        self._text = None
        self.n_encoded = 0
        self.n_cached = 0

    # -- shared model --------------------------------------------------------------------------

    _MODELS: dict[tuple[str, str, str], tuple[object, object]] = {}

    def _load(self) -> None:
        """Load the model (once per process and device) and the vocabulary's text features."""
        if self._text is not None:
            return
        import open_clip
        import torch

        if self.threads:
            torch.set_num_threads(int(self.threads))
        dev = self.req_device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        key = (self.model_name, self.pretrained, dev)
        if key not in self._MODELS:
            m, _, pre = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=dev)
            m.eval()
            self._MODELS[key] = (m, pre)
        self._model, self._pre = self._MODELS[key]
        tok = open_clip.get_tokenizer(self.model_name)
        with torch.no_grad():
            t = self._model.encode_text(tok([self.prompt.format(v) for v in self.vocab]).to(dev)).float()
            self._text = t / t.norm(dim=-1, keepdim=True)
            self._scale = float(self._model.logit_scale.exp())

    # -- labelling -----------------------------------------------------------------------------

    def _key(self, crops: Sequence[np.ndarray]) -> str:
        """Cache key: model, vocabulary and the exact crop pixels."""
        h = hashlib.sha256(f"{self.model_name}|{self.pretrained}|{self.vhash}".encode())
        for c in crops:
            a = np.ascontiguousarray(c, np.uint8)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()[:24]

    def probs(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Softmax over the vocabulary for the mean image embedding of `crops`."""
        import torch
        from PIL import Image

        self._load()
        imgs = torch.stack([self._pre(Image.fromarray(np.ascontiguousarray(c, np.uint8))) for c in crops])
        with torch.no_grad():
            f = self._model.encode_image(imgs.to(self.device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            m = f.mean(dim=0)
            m = m / m.norm()
            p = torch.softmax(self._scale * (self._text @ m), dim=0)
        self.n_encoded += len(crops)
        return p.cpu().numpy().astype(np.float64)

    def topk(self, crops: Sequence[np.ndarray], k: int) -> list[tuple[str, float]]:
        """Top-k `(label, score)` (scores rounded to 1e-4, ties by label); [] without crops."""
        crops = [c for c in crops if c is not None and c.size and min(c.shape[:2]) > 0]
        if not crops:
            return []
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{self._key(crops)}.json"
            if path.is_file():
                self.n_cached += 1
                return [(a, float(b)) for a, b in json.loads(path.read_text(encoding="utf-8"))][:k]
        p = self.probs(crops)
        rows = sorted(((round(float(s), 4), lab) for lab, s in zip(self.vocab, p)), key=lambda r: (-r[0], r[1]))
        top = [(lab, s) for s, lab in rows[: max(int(k), 5)]]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(top), encoding="utf-8")
        return top[:k]
