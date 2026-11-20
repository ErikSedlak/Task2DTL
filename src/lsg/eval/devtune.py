"""Tuning set `splits/dev_tune.txt` (CHANGE 3 Part 0, docs/CHANGE_online_graph.md §2).

`dev_train` (3 pairs, 1 GT-removed, 3 GT-new objects) cannot tune anything. `dev_tune` adds train
rooms chosen from metadata only (3RScan.json + 3DSSG `objects.json`, no downloads):

- `n_heavy` rooms with the most changes (score = w_removed · removed + w_new · new + w_rigid · rigid
  of the room's best rescan), because removals are rare (158 of 718 eligible train pairs have one);
- `n_random` further rooms drawn with a seed, one rescan each, so ordinary rescans keep their share
  (the false-deletion side is tuned on them too);
- then the `include` pairs (`dev_train`), which stay the 3-pair regression set.

Lives in `eval/` because it reads train rescans' `removed` / `rigid` lists (hard rule 1). Counts are
metadata estimates of the §5 GT: things and anchors only; a listed removal still present in the
rescan is dont_care and not counted; `rigid` includes moves below τ_eval_move (centroids need the PLY).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.io.scan3r import Meta

SCORED = ("thing", "anchor")


@dataclass(frozen=True)
class PairCounts:
    """Metadata change counts of one train (reference, rescan) pair."""

    ref: str
    rescan: str
    removed: int
    rigid: int
    new: int

    def score(self, w: Mapping[str, float]) -> float:
        """Weighted change count used to rank rooms."""
        return float(w["removed"]) * self.removed + float(w["new"]) * self.new + float(w["rigid"]) * self.rigid


def _scored(o: Mapping[str, Any]) -> bool:
    """Thing or anchor (3DSSG nyu40); objects without a class are ignored."""
    try:
        return node_level(int(o["nyu40"])) in SCORED
    except (KeyError, TypeError, ValueError):
        return False


def pair_counts(
    meta: Meta, objects_of: Callable[[str], Mapping[int, Mapping[str, Any]]], exclude_refs: set[str]
) -> list[PairCounts]:
    """Counts for every train rescan whose room is not excluded and whose two scans are in 3DSSG.

    Args:
        meta: 3RScan metadata.
        objects_of: scan id -> {objectId: 3DSSG object} (`dssg.load_objects`); empty = not in 3DSSG.
        exclude_refs: reference ids whose rooms are never selected (dev, relchange).

    Returns:
        One `PairCounts` per eligible pair, sorted by (reference, rescan).
    """
    out: list[PairCounts] = []
    for s in sorted(meta.scans.values(), key=lambda m: (m.reference_id, m.scan_id)):
        if s.is_reference or s.split != "train" or s.reference_id in exclude_refs:
            continue
        oR, oS = objects_of(s.reference_id), objects_of(s.scan_id)
        if not oR or not oS:
            continue
        removed = {int(i) for i in s.removed if int(i) in oR and _scored(oR[int(i)]) and int(i) not in oS}
        rigid = {int(e.id) for e in s.rigid if int(e.id) in oR and _scored(oR[int(e.id)])}
        new = {int(i) for i, o in oS.items() if int(i) not in oR and _scored(o)}
        out.append(PairCounts(s.reference_id, s.scan_id, len(removed), len(rigid), len(new)))
    return out


def select_dev_tune(
    rows: Sequence[PairCounts], n_heavy: int, n_random: int, seed: int, weights: Mapping[str, float]
) -> tuple[list[PairCounts], list[PairCounts]]:
    """Pick the change-heavy and the random rooms (one rescan each).

    Best rescan per room = highest score, ties by rescan id. Heavy rooms = the top `n_heavy` by
    (-score, reference id). Random rooms = `n_random` drawn without replacement from the sorted
    remaining rooms with `numpy.random.default_rng(seed)`, then one of their rescans with the same
    generator (as `io.splits.subset_pairs`).

    Returns:
        (heavy, random), each in selection order.
    """
    by_ref: dict[str, list[PairCounts]] = {}
    for r in rows:
        by_ref.setdefault(r.ref, []).append(r)
    best = {ref: min(rs, key=lambda r: (-r.score(weights), r.rescan)) for ref, rs in by_ref.items()}
    ranked = sorted(best.values(), key=lambda r: (-r.score(weights), r.ref))
    heavy = ranked[: int(n_heavy)]
    taken = {r.ref for r in heavy}
    rest = sorted(ref for ref in by_ref if ref not in taken)
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(len(rest), size=min(int(n_random), len(rest)), replace=False).tolist() if rest else []
    rand: list[PairCounts] = []
    for i in pick:
        rs = sorted(by_ref[rest[int(i)]], key=lambda r: r.rescan)
        rand.append(rs[int(rng.integers(len(rs)))])
    return heavy, rand


def download_ids(pairs: Iterable[tuple[str, str]], already: set[str]) -> list[str]:
    """Reference and rescan ids of `pairs` that are not downloaded yet (`already`), sorted."""
    ids = {x for p in pairs for x in p}
    return sorted(ids - set(already))


def totals(rows: Iterable[PairCounts]) -> dict[str, int]:
    """Summed counts of the given pairs."""
    rows = list(rows)
    return {"pairs": len(rows), "removed": sum(r.removed for r in rows), "rigid": sum(r.rigid for r in rows),
            "new": sum(r.new for r in rows)}  # fmt: skip
