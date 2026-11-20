"""`lsg.eval.movediag` (CHANGE 3 Part A diagnostic)."""

import json

from lsg.config import load_config
from lsg.eval.movediag import assoc_class, distance_bucket, evidence_class, format_breakdown, move_breakdown

CFG = load_config()
CFG["confidence"].update(L_hit=2.5, L_miss=2.5, L_free_like=2.5)  # the class tests below use equal vote weights


def test_evidence_classes():
    assert evidence_class(None, CFG) == "unobserved"
    assert evidence_class({"seen": 0, "free": 0, "free_like": 0}, CFG) == "unobserved"
    assert evidence_class({"seen": 2, "free": 0, "free_like": 0}, CFG) == "net_seen"
    assert evidence_class({"seen": 1, "free": 2, "free_like": 0}, CFG) == "free_ev"  # free evidence wins
    assert evidence_class({"seen": 3, "free": 2, "free_like": 0}, CFG) == "net_seen"  # net seen first (removal veto)
    assert evidence_class({"seen": 1, "free": 1, "free_like": 0}, CFG) == "mixed"


def test_assoc_and_distance():
    assert assoc_class({"oid": 5, "obs_gt": [5, "chair"]}) == "own"
    assert assoc_class({"oid": 5, "obs_gt": [9, "chair"]}) == "other"
    assert assoc_class({"oid": 5, "obs_gt": None}) == "none"
    e = [0.1, 0.3, 1.0]
    assert distance_bucket({"centroid": [0.2, 0, 0], "centroid_ref": [0, 0, 0]}, e) == "[0.1,0.3)"
    assert distance_bucket({"centroid": [2, 0, 0], "centroid_ref": [0, 0, 0]}, e) == ">=1"
    assert distance_bucket({"centroid": [0.05, 0, 0], "centroid_ref": [0, 0, 0]}, e) == "<0.1"
    assert distance_bucket({"centroid": [0, 0, 0], "centroid_ref": None}, e) == "?"


def test_breakdown_counts_only_predicted_moves(tmp_path):
    rows = [
        {"oid": 1, "gt_status": "absent_unlisted", "pred_state": "moved", "votes": {"seen": 2, "free": 0, "free_like": 0},
         "obs_gt": [7, "chair"], "centroid": [2, 0, 0], "centroid_ref": [0, 0, 0]},
        {"oid": 2, "gt_status": "moved", "pred_state": "moved", "votes": {"seen": 0, "free": 3, "free_like": 0},
         "obs_gt": [2, "chair"], "centroid": [0.5, 0, 0], "centroid_ref": [0, 0, 0]},
        {"oid": 3, "gt_status": "unchanged", "pred_state": "unknown", "votes": None, "obs_gt": None,
         "centroid": [0, 0, 0], "centroid_ref": [0, 0, 0]},
    ]  # fmt: skip
    d = tmp_path / "details" / "oracle"
    d.mkdir(parents=True)
    (d / "R_S.json").write_text(json.dumps({"objects": rows}), encoding="utf-8")
    res = move_breakdown(tmp_path, CFG)["oracle"]
    assert res["n_moved"] == 2 and res["n_pairs"] == 1
    assert res["by_evidence"] == {"absent_unlisted|net_seen": 1, "moved|free_ev": 1}
    assert res["by_assoc"] == {"absent_unlisted|other": 1, "moved|own": 1}
    assert res["by_distance"] == {"absent_unlisted|>=1": 1, "moved|[0.3,1)": 1}
    assert "net_seen" in format_breakdown({"oracle": res})


def test_part_a_selection_rule():
    """CHANGE 3 Part A selection (chg3_checks.select): both modes must pass; best mean F1 wins, ties to the first."""
    from lsg.eval.chg3_checks import select

    def row(fm, f1):
        return {"oracle": {"false_move_rate": fm, "moved_F1": f1}, "perception": {"false_move_rate": fm, "moved_F1": f1}}

    crit = CFG["chg3"]["part_a"]["criterion"]
    rows = {"off": row(0.20, 0.50), "a": row(0.10, 0.49), "b": row(0.08, 0.47), "c": row(0.11, 0.60), "d": row(0.10, 0.49)}
    s = select(rows, crit)
    assert s["verdict"]["a"]["pass"] and not s["verdict"]["b"]["pass"] and not s["verdict"]["c"]["pass"]
    assert s["winner"] == "a"  # a and d tie: the first listed wins
    assert select({"off": row(0.2, 0.5), "x": row(0.15, 0.5)}, crit)["winner"] is None
