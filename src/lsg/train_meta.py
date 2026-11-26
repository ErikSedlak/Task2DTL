"""Train-split metadata for the method: the CLIP vocabulary (M4) and class priors (M6).

Both come from train-split metadata (3RScan.json + 3DSSG `objects.json`),
**excluding** the scenes of `dev_train.txt` and `eval_relchange_train.txt` (reference and all
its rescans). Only scans with `split == "train"` are read, so validation / test scans never
contribute. This is training data of *other* rooms, not the rescan under evaluation.

Method side (outside `perception/`, which may not read 3DSSG at all): no `oracle` / `eval`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    """CLIP vocabulary = reference-graph labels ∪ the `top_n` most frequent train labels.

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


# --------------------------------------------------------------------------------------------
# Class priors (M6)
# --------------------------------------------------------------------------------------------


@dataclass
class PriorTable:
    """Per-class persistence counts over train (reference object, rescan) pairs.

    `present`: the object is in the rescan's 3DSSG objects (unchanged or moved); `removed`:
    listed in the rescan's `removed`. Absent-unlisted objects are not counted (their fate is
    unknown, as in the eval universe). Rates are Beta-smoothed toward the parent
    level: label -> nyu40 -> global (`p_persist`).
    """

    by_label: dict[str, tuple[int, int]] = field(default_factory=dict)  # label -> (present, removed)
    by_nyu: dict[int, tuple[int, int]] = field(default_factory=dict)
    total: tuple[int, int] = (0, 0)
    kappa: float = 20.0

    @property
    def p_global(self) -> float:
        p, r = self.total
        return p / (p + r) if p + r else 0.9

    def p_nyu(self, nyu40: int | None) -> float:
        """Smoothed persistence of a nyu40 class (global rate if unknown)."""
        pg = self.p_global
        if nyu40 is None or int(nyu40) not in self.by_nyu:
            return pg
        p, r = self.by_nyu[int(nyu40)]
        return (p + self.kappa * pg) / (p + r + self.kappa)

    def p_persist(self, label: str | None, nyu40: int | None) -> float:
        """Smoothed persistence of a label, backing off to its nyu40 class, then the global rate."""
        pn = self.p_nyu(nyu40)
        if not label or label not in self.by_label:
            return pn
        p, r = self.by_label[label]
        return (p + self.kappa * pn) / (p + r + self.kappa)


def prior_counts(meta: Meta, scans: Sequence[str], objects_of) -> PriorTable:
    """Count present / removed thing+anchor objects over the train rescans in `scans`.

    Args:
        meta: 3RScan metadata (train rescans' `removed` lists: training data of other rooms).
        scans: Allowed train scans (`train_scans`); references are skipped, rescans counted.
        objects_of: scan id -> {objectId: 3DSSG object dict} (e.g. `load_objects`).
    """
    lab: dict[str, list[int]] = {}
    nyu: dict[int, list[int]] = {}
    tot = [0, 0]
    for s in scans:
        m = meta[s]
        if m.is_reference or m.reference_id == s:
            continue
        try:
            oR, oS = objects_of(m.reference_id), objects_of(s)
        except KeyError:
            continue
        if not oR or not oS:
            continue
        removed = {int(x) for x in m.removed}
        for oid, o in oR.items():
            if "nyu40" not in o or not o.get("label"):
                continue
            n = int(o["nyu40"])
            if node_level(n) not in VOCAB_LEVELS:
                continue
            if int(oid) in removed:
                k = 1
            elif oid in oS:
                k = 0
            else:
                continue  # absent-unlisted: unknown fate
            for d, key in ((lab, str(o["label"])), (nyu, n)):
                d.setdefault(key, [0, 0])[k] += 1
            tot[k] += 1
    return PriorTable({k: (v[0], v[1]) for k, v in lab.items()}, {k: (v[0], v[1]) for k, v in nyu.items()},
                      (tot[0], tot[1]))  # fmt: skip


@lru_cache(maxsize=2)
def _priors_cached(root: str, exclude: tuple[str, ...]) -> PriorTable:
    """`class_priors` body, cached per process (kappa set by the caller)."""
    meta = load_meta(root)
    ids = train_scans(meta, excluded_references(exclude))
    return prior_counts(meta, ids, lambda s: load_objects(root, s))


def class_priors(root: str | Path, cfg: Mapping[str, Any]) -> PriorTable:
    """Train class priors for the scenes allowed by `train_meta` (same exclusions as the vocabulary)."""
    t = _priors_cached(str(root), tuple(cfg["train_meta"]["exclude_splits"]))
    return PriorTable(t.by_label, t.by_nyu, t.total, float(cfg["confidence"].get("prior_kappa", 20)))


def logit(p: float) -> float:
    """log(p / (1 - p)), with p clipped to [1e-6, 1 - 1e-6]."""
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
