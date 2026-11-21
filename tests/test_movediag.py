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


def test_p_del_selection_rule():
    """CHANGE 3 Part C: false deletion <= ref + 1 point in both modes, best mean removed F1, ties nearest 0.2."""
    from lsg.eval.chg3_checks import select_p_del

    def row(fd, f1):
        return {m: {"false_deletion_rate": fd, "removed_F1": f1} for m in ("oracle", "perception")}

    ref = row(0.03, 0.30)
    crit = CFG["chg3"]["part_c"]["p_del"]["criterion"]
    rows = {"p0.1": row(0.035, 0.40), "p0.3": row(0.035, 0.45), "p0.4": row(0.05, 0.60)}
    vals = {"p0.1": 0.1, "p0.3": 0.3, "p0.4": 0.4}
    s = select_p_del(rows, vals, ref, crit)
    assert not s["verdict"]["p0.4"]["pass"] and s["winner"] == "p0.3"  # p0.4 deletes too much
    tie = select_p_del({"p0.05": row(0.03, 0.4), "p0.3": row(0.03, 0.4)}, {"p0.05": 0.05, "p0.3": 0.3}, ref, crit)
    assert tie["winner"] == "p0.3"  # equal F1: nearer 0.2
    assert select_p_del({"p0.4": row(0.05, 0.6)}, {"p0.4": 0.4}, ref, crit)["winner"] is None


def test_part_n_segment_rule():
    """CHANGE 3 Part N (N1) pre-written rule: most class-agnostic hits among variants passing the F1 guards;
    ties within 2 objects go to the simpler variant."""
    from lsg.eval.chg3_checks import select_segment

    def row(hn, hm, mf1, af1):
        p = {"diag_cand_new_hit": hn, "diag_cand_moved_hit": hm, "moved_F1": mf1, "new_ag_F1": af1}
        return {"perception": p, "oracle": {}}

    rows = {"none": row(73, 59, 0.47, 0.05), "geometric": row(100, 60, 0.465, 0.06), "sam": row(101, 60, 0.47, 0.07),
            "sam_add": row(120, 60, 0.40, 0.08)}  # sam_add fails the moved-F1 guard
    crit = {"f1_drop": 0.01, "tie_objects": 2}
    order = ["none", "geometric", "sam", "sam_add"]
    sel = select_segment(rows, crit, order)
    assert sel["verdict"]["sam_add"]["pass"] is False
    assert sel["winner"] == "geometric"  # 160 vs 161 hits: within the tie band -> the simpler one
    rows["sam"] = row(110, 60, 0.47, 0.07)
    assert select_segment(rows, crit, order)["winner"] == "sam"


def test_part_n_explain_rule():
    """CHANGE 3 Part N (N2): best agnostic new F1 with agnostic new R >= off - 1 pt; ties -> d10, then off."""
    from lsg.eval.chg3_checks import select_explain

    def row(r, f1):
        return {"perception": {"new_ag_R": r, "new_ag_F1": f1}, "oracle": {}}

    crit = {"r_drop": 0.01, "tie_f1": 0.005}
    rows = {"ref_off": row(0.49, 0.08), "d10": row(0.485, 0.10), "d15": row(0.47, 0.12)}
    sel = select_explain(rows, crit, ["d10", "ref_off", "d15"])
    assert sel["verdict"]["d15"]["pass"] is False and sel["winner"] == "d10"
    rows["d10"] = row(0.485, 0.083)  # within 0.5 pt of off -> d10 first in the tie order
    assert select_explain(rows, crit, ["d10", "ref_off", "d15"])["winner"] == "d10"


def test_part_n_noise_rule():
    """CHANGE 3 Part N (N5): best curve F1 among variants passing the change-metric and hit guards."""
    from lsg.eval.chg3_checks import select_noise

    def row(mf1, rf1, fd, hn, hm):
        return {"perception": {"moved_F1": mf1, "removed_F1": rf1, "false_deletion_rate": fd,
                               "diag_cand_new_hit": hn, "diag_cand_moved_hit": hm}, "oracle": {}}  # fmt: skip

    def curve(f1, t):
        return {"best": {"f1": f1, "p_new_min": t}}

    rows = {"base": row(0.47, 0.40, 0.04, 120, 58), "app": row(0.47, 0.40, 0.04, 120, 58),
            "edge": row(0.47, 0.40, 0.04, 110, 58), "all": row(0.47, 0.38, 0.04, 119, 58)}  # fmt: skip
    curves = {"base": curve(0.072, 0.5175), "app": curve(0.080, 0.5175), "edge": curve(0.09, 0.51),
              "all": curve(0.10, 0.52)}  # fmt: skip
    crit = {"f1_drop": 0.01, "fd_up": 0.01, "hits_drop": 5, "tie_f1": 0.005}
    sel = select_noise(rows, curves, crit, ["base", "app", "edge", "all"])
    assert not sel["verdict"]["edge"]["pass"]  # loses 10 hits
    assert not sel["verdict"]["all"]["pass"]  # removed F1 -2 pt
    assert sel["winner"] == "app" and sel["p_new_min"] == 0.5175
