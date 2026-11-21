"""`lsg newcurve` (CHANGE 2): new P / R over the candidate-confidence threshold from saved graphs."""

import copy

import lsg.eval.newcurve as nc
from lsg.config import load_config
from lsg.graph.model import Node


def test_thresholds_include_the_default_sorted_unique():
    cfg = copy.deepcopy(load_config())
    assert nc.thresholds_of(cfg) == sorted({float(cfg["confidence"]["candidates"]["p_new_min"]), 0.95, 0.99, 0.999})
    assert nc.thresholds_of(cfg, [0.99, 0.92]) == sorted({float(cfg["confidence"]["candidates"]["p_new_min"]), 0.92, 0.99})


def test_curve_filters_new_nodes_by_confidence(monkeypatch):
    def fake_match(nodes, gt, mode, cfg, label_nyu=None, use_label=True):  # node "t*" are true positives; GT has 2 new objects
        tp = {n.id: 1 for n in nodes if n.id.startswith("t")}
        return {"tp": tp, "fp": [n.id for n in nodes if n.id not in tp], "ignored": [], "fn": [0] * (2 - len(tp))}

    monkeypatch.setattr(nc, "match_new", fake_match)
    nodes = [Node("t0", "thing", "a", confidence=0.999), Node("t1", "thing", "b", confidence=0.95),
             Node("f0", "thing", "c", confidence=0.93), Node("f1", "thing", "d", confidence=None)]  # fmt: skip
    rows = nc.curve_counts({"p": nodes}, {"p": None}, {}, [0.92, 0.96, 0.9995])
    assert [(r["tp"], r["fp"], r["fn"]) for r in rows] == [(2, 2, 0), (1, 1, 1), (0, 1, 2)]
    assert rows[0]["precision"] == 0.5 and rows[1]["recall"] == 0.5
    assert "p_new_min" in nc.format_curve({"n_pairs": 1, "run": "x", "curve": rows})
    # CHANGE 3 Part E: explicit per-node scores (detector newness) instead of the confidence
    rows = nc.curve_counts({"p": nodes}, {"p": None}, {}, [0.5], scores={"p": [0.9, 0.1, 0.9, None]})
    assert (rows[0]["tp"], rows[0]["fp"]) == (1, 2) and rows[0]["ag_tp"] == 1 and rows[0]["f1"] > 0
