"""Train-split metadata for the method: the CLIP vocabulary (M4) and, from M6, class priors.

DESIGN §2.3: both come from train-split metadata (3RScan.json + 3DSSG `objects.json`),
**excluding** the scenes of `dev_train.txt` and `eval_relchange_train.txt` (reference and all
its rescans). Only scans with `split == "train"` are read, so validation / test scans never
contribute. This is training data of *other* rooms, not the rescan under evaluation.

Method side (outside `perception/`, which may not read 3DSSG at all): no `oracle` / `eval`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from lsg.classes import node_level
from lsg.config import REPO_ROOT
from lsg.graph.model import Graph
from lsg.io.dssg import load_objects
from lsg.io.scan3r import Meta, load_meta, read_pairs

VOCAB_LEVELS = ("thing", "anchor")


def excluded_references(split_files: Sequence[str | Path]) -> set[str]:
    """Reference ids of the pair lists that must not feed priors / vocabulary."""
    out: set[str] = set()
    for f in split_files:
        p = Path(f)
        out |= {r for r, _ in read_pairs(p if p.is_absolute() else REPO_ROOT / p)}
    return out


def train_scans(meta: Meta, exclude_refs: set[str]) -> list[str]:
    """Train scans (references + rescans) whose room is not excluded, sorted."""
    return sorted(
        s.scan_id for s in meta.scans.values()
        if s.split == "train" and s.reference_id not in exclude_refs
    )  # fmt: skip


def label_table(objects_by_scan: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> list[tuple[str, int, int]]:
    """Thing / anchor labels with their majority nyu40 and object count.

    Returns:
        `[(label, nyu40, count)]` sorted by count (descending), then label. nyu40 ties go to
        the smaller id.
    """
    counts: Counter = Counter()
    nyu: dict[str, Counter] = {}
    for objs in objects_by_scan.values():
        for o in objs.values():
            if "nyu40" not in o or not o.get("label"):
                continue
            n = int(o["nyu40"])
            if node_level(n) not in VOCAB_LEVELS:
                continue
            lab = str(o["label"])
            counts[lab] += 1
            nyu.setdefault(lab, Counter())[n] += 1
    rows = [(lab, min(nyu[lab].items(), key=lambda kv: (-kv[1], kv[0]))[0], c) for lab, c in counts.items()]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


@lru_cache(maxsize=2)
def _vocab_table_cached(root: str, exclude: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    """`clip_vocab_table` body, cached per process."""
    meta = load_meta(root)
    ids = train_scans(meta, excluded_references(exclude))
    return tuple(label_table({s: load_objects(root, s) for s in ids}))


def clip_vocab_table(root: str | Path, cfg: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """Train thing/anchor label table (`label_table`) for the scenes allowed by `train_meta`."""
    return list(_vocab_table_cached(str(root), tuple(cfg["train_meta"]["exclude_splits"])))


def perception_vocab(
    G_ref: Graph, table: Sequence[tuple[str, int, int]], top_n: int
) -> tuple[list[str], dict[str, int]]:
    """CLIP vocabulary = reference-graph labels ∪ the `top_n` most frequent train labels (§6.6).

    Returns:
        (sorted labels, label -> nyu40). nyu40 comes from the reference graph node with that
        label if there is one, else from the train table.
    """
    nyu: dict[str, int] = {}
    for lab, n, _ in table[: int(top_n)]:
        nyu[lab] = int(n)
    for nd in G_ref.nodes:
        if nd.level in VOCAB_LEVELS and nd.label and nd.nyu40 is not None:
            nyu[nd.label] = int(nd.nyu40)
    return sorted(nyu), nyu
