"""§7.4 audit (M8): per-object details, FP pools, sampling, veto set, verdict summary."""

import json

import pytest

from lsg.eval.audit import (
    compare_metrics,
    pools,
    read_csv,
    sample_rows,
    summarize_rows,
    veto_pool,
    wilson,
    write_csv,
)
from lsg.eval.metrics import score_pair

from test_metrics import CFG, TRUTH, gt, good_prediction, graph, node  # noqa: F401 (gt is a fixture)


def _details(G, gt_, mode="oracle"):
    sc = score_pair(G, gt_, mode, CFG, obs_truth=TRUTH, details=True)
    return {"ref": "R", "rescan": "S", "mode": mode, "T_align": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "frame_ids": [0], "objects": json.loads(json.dumps(sc["objects"], default=float)), "new": sc["new"]}


def test_details_do_not_change_counts(gt):
    a = score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH)
    b = score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH, details=True)
    assert a["counts"] == b["counts"] and a["loc_errors"] == b["loc_errors"]
    rows = {r["node"]: r for r in b["objects"]}
    assert rows["r1"]["gt_status"] == "moved" and rows["r1"]["obs_gt"] == [1, "chair"]
    assert rows["r4"]["pred_state"] == "removed" and rows["r4"]["gt_status"] == "absent_unlisted"
    assert {n["id"]: n["outcome"] for n in b["new"]} == {"n0": "tp", "n1": "ignored", "n2": "fp"}


def test_pools(gt):
    G = graph([
        node("r1", "chair", "removed", (0, 0, 0)),  # GT moved -> removed pool (scored FP)
        node("r2", "cup", "removed", (0, 1, 0)),  # GT removed -> TP, not in a pool
        node("r3", "table", "moved", (1.5, 1, 0), level="anchor"),  # GT unchanged -> moved pool
        node("r4", "book", "removed", (2, 2, 0)),  # absent-unlisted -> removed pool
        node("r5", "pillow", "moved", (3.5, 3, 0)),  # dont_care -> no pool
        node("n2", "plant", "new", (5, 5, 0), token="o5"),  # FP new
    ])  # fmt: skip
    p = pools([_details(G, gt)])
    assert [(r["node"], r["gt_status"]) for r in p["removed"]] == [("r1", "moved"), ("r4", "absent_unlisted")]
    assert [(r["node"], r["gt_status"]) for r in p["moved"]] == [("r3", "unchanged")]
    assert [r["node"] for r in p["new"]] == ["n2"]


def test_veto_pool(gt, tmp_path):
    full = graph([node("r2", "cup", "unknown", (0, 1, 0)), node("r4", "book", "unknown", (2, 2, 0))])
    b2 = graph([node("r2", "cup", "removed", (0, 1, 0)), node("r4", "book", "removed", (2, 2, 0))])
    b2.save(tmp_path / "oracle" / "R_S.json")
    v = veto_pool([_details(full, gt)], tmp_path)
    assert [r["node"] for r in v] == ["r2"]  # r4 is absent-unlisted, not GT removed


def test_sample_rows_seeded_and_ordered():
    pool = [{"k": i} for i in range(50)]
    a, b = sample_rows(pool, 10, (0, 1, 2)), sample_rows(pool, 10, (0, 1, 2))
    assert a == b and len(a) == 10
    assert [r["k"] for r in a] == sorted(r["k"] for r in a)
    assert sample_rows(pool[:3], 10, (0,)) == pool[:3]


def test_wilson():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson(0, 0) == (None, None)


def test_csv_roundtrip_and_summary(tmp_path):
    rows = [{"id": f"oracle_removed_{k:02d}", "mode": "oracle", "category": "removed", "ref": "R", "rescan": "S",
             "node": f"r{k}", "label": "x", "gt_status": "absent_unlisted" if k < 3 else "unchanged",
             "pred_state": "removed", "confidence": 0.1} for k in range(4)]  # fmt: skip
    p = write_csv(rows, tmp_path / "a.csv")
    back = read_csv(p)
    assert [r["verdict"] for r in back] == [""] * 4
    with pytest.raises(ValueError):
        summarize_rows(back)
    for r, v in zip(back, ("gt_missing", "gt_missing", "unclear", "method_error")):
        r["verdict"] = v
    s = summarize_rows(back)
    assert s["oracle/removed"]["n"] == 4 and s["oracle/removed"]["gt_missing_share"] == 0.5
    assert s["oracle/removed/absent_unlisted"]["gt_missing"] == 2
    assert s["oracle/removed/unchanged"]["method_error"] == 1
    assert s["all"]["n"] == 4


def test_compare_metrics():
    m = {"summary": {"oracle": {"a": 1}}, "per_pair": [{"ref": "R", "rescan": "S", "mode": "oracle",
                                                         "counts": {"x": 1}, "loc_errors": []}]}  # fmt: skip
    assert compare_metrics(m, json.loads(json.dumps(m))) == []
    m2 = json.loads(json.dumps(m))
    m2["per_pair"][0]["counts"]["x"] = 2
    assert compare_metrics(m, m2) == ["per_pair ('R', 'S', 'oracle') differs"]


def test_audit_runs_the_strict_candidate_rules_unless_relaxed(monkeypatch):
    """The audit must reproduce m7_final/val/FULL, i.e. the M7 method (CHANGE 1 and 2 off)."""
    import copy

    import lsg.eval.harness as harness
    from lsg.config import load_config
    from lsg.eval.audit import run_audit_sample

    seen = []

    def fake_run_eval(cfg, *a, **k):
        seen.append(cfg["perception"]["relaxed"]["enabled"])
        c = cfg["confidence"]
        assert (c["candidates"]["enabled"], c["class_priors"], c["p_del"]) == ((True, False, 0.2) if seen[-1]
                                                                              else (False, True, 0.2))  # fmt: skip
        raise RuntimeError("stop")

    monkeypatch.setattr(harness, "run_eval", fake_run_eval)
    cfg = copy.deepcopy(load_config())
    assert cfg["perception"]["relaxed"]["enabled"] is True  # the default method is relaxed
    for relaxed in (False, True):
        with pytest.raises(RuntimeError):
            run_audit_sample(cfg, "x", relaxed=relaxed, max_pairs=1)
    assert seen == [False, True]
