"""Tuning-set selection `lsg.eval.devtune` (CHANGE 3 Part 0)."""

import numpy as np

from lsg.eval.devtune import PairCounts, download_ids, pair_counts, select_dev_tune, totals
from lsg.io.scan3r import Meta, RigidEntry, ScanMeta

W = {"removed": 2.0, "new": 1.0, "rigid": 0.5}


def obj(nyu40):
    return {"label": "x", "nyu40": nyu40}


def test_pair_counts_follow_the_gt_rules():
    """Listed-but-present removals, structure and non-train scans do not count; rigid ids count once."""
    I = np.eye(4)
    meta = Meta({
        "R": ScanMeta("R", "R", True, "train", I),
        "S": ScanMeta("S", "R", False, "train", I, rigid=[RigidEntry(3, I), RigidEntry(3, I), RigidEntry(4, None),
                                                           RigidEntry(1, I)],
                      removed=[2, 5, 1]),
        "X": ScanMeta("X", "R", False, "validation", I, removed=[2]),
        "Q": ScanMeta("Q", "Q", True, "train", I),
        "T": ScanMeta("T", "Q", False, "train", I, removed=[1]),
    })  # fmt: skip
    objs = {
        # 1 wall (structure), 2 chair (removed), 3 table (rigid), 4 lamp (rigid, bare int), 5 pillow (listed, still present)
        "R": {1: obj(1), 2: obj(5), 3: obj(7), 4: obj(35), 5: obj(18)},
        "S": {1: obj(1), 3: obj(7), 4: obj(35), 5: obj(18), 9: obj(5), 10: obj(2)},  # 9 new chair, 10 new floor
        "X": {1: obj(1)},
        "Q": {1: obj(5)},
        "T": {},  # not in 3DSSG -> skipped
    }
    rows = pair_counts(meta, lambda s: objs.get(s, {}), exclude_refs=set())
    assert rows == [PairCounts("R", "S", removed=1, rigid=2, new=1)]
    assert pair_counts(meta, lambda s: objs.get(s, {}), exclude_refs={"R"}) == []


def rows_fixture():
    """Room A: two rescans; B, C, D, E, F: one each."""
    return [
        PairCounts("A", "a1", 1, 0, 0), PairCounts("A", "a2", 3, 0, 0),
        PairCounts("B", "b1", 0, 0, 6),  # score 6 = A's best (a2), tie broken by ref id
        PairCounts("C", "c1", 0, 2, 0), PairCounts("D", "d1", 0, 0, 0),
        PairCounts("E", "e1", 0, 0, 1), PairCounts("F", "f1", 0, 0, 0),
    ]  # fmt: skip


def test_heavy_rooms_take_the_best_rescan_and_break_ties_by_id():
    heavy, rand = select_dev_tune(rows_fixture(), n_heavy=2, n_random=0, seed=0, weights=W)
    assert [(r.ref, r.rescan) for r in heavy] == [("A", "a2"), ("B", "b1")]
    assert rand == []


def test_random_rooms_are_disjoint_seeded_and_one_per_room():
    rows = rows_fixture()
    h1, r1 = select_dev_tune(rows, 2, 3, seed=0, weights=W)
    h2, r2 = select_dev_tune(rows, 2, 3, seed=0, weights=W)
    assert (h1, r1) == (h2, r2)
    refs = [r.ref for r in r1]
    assert len(refs) == len(set(refs)) == 3 and not set(refs) & {"A", "B"}
    _, r3 = select_dev_tune(rows, 2, 10, seed=0, weights=W)  # more than available -> all remaining rooms
    assert sorted(r.ref for r in r3) == ["C", "D", "E", "F"]


def test_download_ids_and_totals():
    assert download_ids([("A", "a1"), ("B", "b1"), ("A", "a2")], already={"A", "b1"}) == sorted(["B", "a1", "a2"])
    t = totals(rows_fixture()[:3])
    assert t == {"pairs": 3, "removed": 4, "rigid": 0, "new": 6}
