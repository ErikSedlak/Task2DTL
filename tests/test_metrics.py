"""Metric unit tests on a 5-object toy pair (DESIGN §9 M2), incl. the perception matching rule."""

import numpy as np
import pytest

from lsg.config import load_config
from lsg.eval.gt import GTLabels, GTNew, GTObject
from lsg.eval.metrics import (
    aabb_iou,
    aggregate,
    bucket_of,
    match_new,
    score_pair,
)
from lsg.graph.model import ROOM_ID, Graph, Node, bbox_dict

CFG = load_config()
A = np.array


def box(c, h):
    c = A(c, float)
    return c - h, c + h


def gt_obj(i, label, nyu, lv, size, status, reason=None, disp=None, c_ref=(0, 0, 0), c_new=None, h=0.1):
    o = GTObject(i, label, nyu, lv, size, status, reason, in_rigid=disp is not None, displacement=disp)
    o.centroid_ref = A(c_ref, float)
    if c_new is not None:
        o.centroid_new = A(c_new, float)
        o.aabb_new_min, o.aabb_new_max = box(c_new, h)
    return o


@pytest.fixture
def gt():
    objs = {
        1: gt_obj(1, "chair", 5, "thing", "large", "moved", disp=1.2, c_new=(1.2, 0, 0)),
        2: gt_obj(2, "cup", 40, "thing", "small", "removed", c_ref=(0, 1, 0)),
        3: gt_obj(3, "table", 7, "anchor", "large", "unchanged", c_ref=(1, 1, 0), c_new=(1, 1, 0)),
        4: gt_obj(4, "book", 23, "thing", "small", "absent_unlisted", c_ref=(2, 2, 0)),
        5: gt_obj(5, "pillow", 18, "thing", "small", "dont_care", "nonrigid", c_ref=(3, 3, 0), c_new=(3, 3, 0)),
    }
    new = {
        20: GTNew(20, "box", 29, "thing", "small", A([2.0, 0, 0]), *box((2, 0, 0), 0.1)),
        21: GTNew(21, "lamp", 35, "thing", "large", A([0, 2.0, 0]), *box((0, 2, 0), 0.3)),
    }
    return GTLabels("R", "S", objs, new, 1, 1, 1, 2)


def node(nid, label, state, c, h=0.1, token=None, nyu=None, level="thing"):
    lo, hi = box(c, h)
    return Node(nid, level, label, nyu40=nyu, bbox_3d=bbox_dict(lo, hi), centroid=list(map(float, c)),
                state=state, obs_token=token)  # fmt: skip


def graph(nodes):
    return Graph(Graph.new_meta("R", "S", "oracle"), [Node(ROOM_ID, "room", "room"), *nodes], [])


TRUTH = {"o1": (1, "chair"), "o2": (3, "table"), "o3": (20, "box"), "o4": (5, "pillow"),
         "o5": (99, "plant"), "o6": (21, "lamp")}  # fmt: skip


def good_prediction():
    return graph([
        node("r1", "chair", "moved", (1.25, 0, 0), token="o1"),
        node("r2", "cup", "removed", (0, 1, 0)),
        node("r3", "table", "unchanged", (1, 1, 0), token="o2", level="anchor"),
        node("r4", "book", "removed", (2, 2, 0)),
        node("r5", "pillow", "unchanged", (3, 3, 0)),
        node("n0", "box", "new", (2, 0, 0), token="o3"),
        node("n1", "pillow", "new", (3, 3, 0), token="o4"),  # = dont_care pillow -> ignored
        node("n2", "plant", "new", (5, 5, 0), token="o5"),  # not a GT new -> FP
    ])  # fmt: skip


def test_toy_oracle_counts(gt):
    sc = score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH)
    c = sc["counts"]
    assert (c["moved_tp"], c["moved_fp"], c["moved_fn"]) == (1, 0, 0)
    assert (c["removed_tp"], c["removed_fp"], c["removed_fn"]) == (1, 0, 0)
    assert (c["new_tp"], c["new_fp"], c["new_fn"], c["new_ignored"]) == (1, 1, 1, 1)
    assert (c["unch_n"], c["unch_strict"], c["unch_kept"]) == (1, 1, 1)
    assert (c["absent_n"], c["absent_del"], c["absent_move"]) == (1, 1, 0)
    assert (c["ref_n"], c["ref_unknown"]) == (5, 0)
    assert (c["assoc_n"], c["assoc_strict"], c["assoc_label"]) == (2, 2, 2)  # chair + table
    assert (c["moved_n_d2"], c["moved_tp_d2"], c["moved_n_d0"]) == (1, 1, 0)  # 1.2 m -> >= 1 m
    assert (c["moved_n_large"], c["moved_tp_large"]) == (1, 1)
    assert (c["new_n_small"], c["new_tp_small"], c["new_n_large"], c["new_tp_large"]) == (1, 1, 1, 0)
    assert sc["loc_errors"] == [pytest.approx(0.05)]


def test_toy_metrics_values(gt):
    agg = aggregate([score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH)], CFG)
    M = agg["metrics"]
    for k in ("moved_P", "moved_R", "moved_F1", "removed_P", "removed_R", "unchanged_acc"):
        assert M[k]["value"] == pytest.approx(1.0), k
    assert M["new_P"]["value"] == pytest.approx(0.5) and M["new_R"]["value"] == pytest.approx(0.5)
    assert M["new_F1"]["value"] == pytest.approx(0.5) and M["new_F1"]["n"] == 2
    assert M["false_deletion_rate"]["value"] == 1.0 and M["false_deletion_rate"]["n"] == 1
    assert M["false_move_rate"]["value"] == 0.0
    assert M["new_R_small"]["value"] == 1.0 and M["new_R_large"]["value"] == 0.0
    assert M["moved_R_disp>=1"]["value"] == 1.0
    assert M["moved_R_disp[0.1,0.3)"]["value"] is None and M["moved_R_disp[0.1,0.3)"]["n"] == 0
    assert agg["loc_error"]["n"] == 1 and agg["loc_error"]["median"] == pytest.approx(0.05)


def test_unknown_kept_and_false_move(gt):
    G = graph([
        node("r1", "chair", "unchanged", (0, 0, 0), token="o2"),  # missed move, wrong association
        node("r2", "cup", "unknown", (0, 1, 0)),  # unknown != removed
        node("r3", "table", "unknown", (1, 1, 0), level="anchor"),
        node("r4", "book", "moved", (2.5, 2, 0)),  # absent_unlisted -> false move, not an FP
        node("r5", "pillow", "moved", (3, 3, 0)),  # dont_care -> ignored
    ])  # fmt: skip
    c = score_pair(G, gt, "oracle", CFG, obs_truth=TRUTH)["counts"]
    assert (c["moved_tp"], c["moved_fp"], c["moved_fn"]) == (0, 0, 1)
    assert (c["removed_tp"], c["removed_fp"], c["removed_fn"]) == (0, 0, 1)
    assert (c["unch_strict"], c["unch_kept"]) == (0, 1)
    assert (c["absent_del"], c["absent_move"]) == (0, 1)
    assert c["ref_unknown"] == 2
    assert (c["assoc_n"], c["assoc_strict"], c["assoc_label"]) == (2, 0, 0)  # chair -> table obs
    assert (c["new_tp"], c["new_fn"]) == (0, 2)


def test_same_label_tolerant_association(gt):
    truth = {**TRUTH, "o9": (77, "chair")}  # another chair of the rescan
    G = graph([node("r1", "chair", "moved", (1.2, 0, 0), token="o9"),
               node("r3", "table", "unchanged", (1, 1, 0), token="o2", level="anchor")])  # fmt: skip
    c = score_pair(G, gt, "oracle", CFG, obs_truth=truth)["counts"]
    assert (c["assoc_n"], c["assoc_strict"], c["assoc_label"]) == (2, 1, 2)


def test_gt_object_without_node_is_counted_not_scored(gt):
    c = score_pair(graph([node("r1", "chair", "moved", (1.2, 0, 0))]), gt, "oracle", CFG)["counts"]
    assert c["ref_no_node"] == 4 and c["ref_n"] == 1


# ---- perception matching rule (§5) ------------------------------------------------------------


def test_perception_rule(gt):
    preds = [
        node("A", "box", "new", (2.2, 0, 0)),  # 0.2 m, IoU 0: distance gate passes
        node("B", "carton", "new", (2.05, 0, 0), nyu=29),  # same nyu40, IoU 0.6: wins the box
        node("C", "lamp", "new", (0, 2.5, 0), h=0.3),  # 0.5 m and IoU 0.09: no match
        node("D", "lamp", "new", (0, 2.1, 0), h=0.3),  # 0.1 m: matches the lamp
        node("E", "chair", "new", (2, 0, 0)),  # right place, wrong class (label_nyu chair -> 5)
        node("F", "pillow", "new", (3, 3, 0)),  # = dont_care pillow's rescan instance -> ignored
    ]  # fmt: skip
    m = match_new(preds, gt, "perception", CFG, label_nyu={"chair": 5, "box": 29, "lamp": 35})
    assert m["tp"] == {"B": 20, "D": 21}
    assert sorted(m["fp"]) == ["A", "C", "E"]
    assert m["ignored"] == ["F"] and m["fn"] == []


def test_perception_rule_label_to_nyu40_fallback(gt):
    p = [node("G", "carton", "new", (2.1, 0, 0))]  # no nyu40 on the node
    assert match_new(p, gt, "perception", CFG, label_nyu={"carton": 29})["tp"] == {"G": 20}
    assert match_new(p, gt, "perception", CFG, label_nyu={})["tp"] == {}


def test_perception_iou_gate_alone(gt):
    # centroid 0.35 m away (> 0.30) but a big box overlapping the lamp: IoU >= 0.25 passes
    p = [node("H", "lamp", "new", (0, 2.35, 0), h=0.3)]
    m = match_new(p, gt, "perception", CFG)
    lo, hi = box((0, 2.35, 0), 0.3)
    assert aabb_iou(lo, hi, gt.new[21].aabb_min, gt.new[21].aabb_max) >= 0.25
    assert m["tp"] == {"H": 21}


def test_score_pair_perception_mode(gt):
    G = graph([node("r1", "chair", "moved", (1.2, 0, 0)), node("n0", "box", "new", (2.0, 0, 0))])
    c = score_pair(G, gt, "perception", CFG)["counts"]
    assert (c["new_tp"], c["new_fp"], c["new_fn"]) == (1, 0, 1)


# ---- buckets, bootstrap -------------------------------------------------------------------------


def test_bucket_of():
    e = [0.1, 0.3, 1.0]
    assert [bucket_of(d, e) for d in (0.05, 0.1, 0.29, 0.3, 0.99, 1.0, 7.0, None)] == [None, 0, 0, 1, 1, 2, 2, None]


def test_bootstrap_deterministic_and_brackets_value(gt):
    per = [score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH)]
    bad = score_pair(graph([node("r1", "chair", "unchanged", (0, 0, 0))]), gt, "oracle", CFG)
    per += [bad] * 3
    a1, a2 = aggregate(per, CFG), aggregate(per, CFG)
    assert a1 == a2
    m = a1["metrics"]["moved_R"]
    assert m["value"] == pytest.approx(0.25) and m["n"] == 4
    assert m["ci"][0] <= m["value"] <= m["ci"][1]
    assert 0.0 <= m["ci"][0] and m["ci"][1] <= 1.0


def test_vote_evidence_diagnostics(gt):
    from lsg.change.visibility import NodeVotes

    votes = {
        "r2": NodeVotes(raw_free=3, free=2),  # removed, with free evidence
        "r3": NodeVotes(raw_free=2, free=2, seen=4, raw_seen=4),  # unchanged, false free evidence
        "r4": NodeVotes(seen=1, raw_seen=1, free=1, raw_free=1),  # absent: seen, 1 free < k_free
    }
    c = score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH, votes=votes)["counts"]
    assert (c["diag_votes"], c["diag_free_removed"], c["diag_free_unch"]) == (1, 1, 1)
    assert (c["diag_free_absent"], c["diag_seen_absent"]) == (0, 1)
    M = aggregate([{"counts": c, "loc_errors": []}], CFG)["metrics"]
    assert M["free_ev_unchanged"]["value"] == 1.0 and M["free_ev_removed"]["value"] == 1.0
    assert M["free_ev_absent"]["value"] == 0.0 and M["seen_ev_absent"]["value"] == 1.0
    c0 = score_pair(good_prediction(), gt, "oracle", CFG, obs_truth=TRUTH)["counts"]
    assert c0["diag_votes"] == 0 and c0["diag_free_unch"] == 0


def _cand(token, label, c, h=0.1, nyu=None):
    from lsg.obs import ObsInstance

    lo, hi = box(c, h)
    return ObsInstance(token, label, [(label, 1.0)], nyu, np.stack([lo, hi]))


def test_perception_obs_truth_and_candidate_counts(gt):
    from lsg.eval.metrics import candidate_counts, perception_obs_truth

    cands = [
        _cand("c0", "chair", (1.25, 0, 0)),  # moved-to of chair 1, right label
        _cand("c1", "bag", (2.05, 0, 0)),  # GT new box 20, wrong label
        _cand("c2", "table", (1.0, 1.0, 0)),  # on the unchanged table's rescan instance
        _cand("c3", "lamp", (9, 9, 9)),  # unexplained (person, mesh hole, ...)
    ]
    truth = perception_obs_truth(cands, gt, CFG)
    assert truth == {"c0": (1, "chair"), "c2": (3, "table")}  # c1's label does not match box
    c = candidate_counts(cands, gt, CFG)
    assert c["diag_cand"] == 1 and c["diag_cand_n"] == 4
    assert (c["diag_cand_new_n"], c["diag_cand_new_hit"], c["diag_cand_new_lab"]) == (2, 1, 0)
    assert (c["diag_cand_moved_n"], c["diag_cand_moved_hit"], c["diag_cand_moved_lab"]) == (1, 1, 1)
    assert c["diag_cand_explained"] == 3
    # through score_pair + aggregate (association uses the perception truth)
    G = graph([node("r1", "chair", "moved", (1.25, 0, 0), token="c0"), node("n0", "bag", "new", (2.05, 0, 0))])
    sc = score_pair(G, gt, "perception", CFG, obs_truth=truth, visible={1, 3}, candidates=cands)
    assert sc["counts"]["assoc_n"] == 1 and sc["counts"]["assoc_strict"] == 1  # r3 has no node here
    M = aggregate([sc], CFG)["metrics"]
    assert M["cand_recall_new"]["value"] == 0.5 and M["cand_recall_moved_labelled"]["value"] == 1.0
    assert M["cand_per_pair"]["value"] == 4.0 and M["cand_explained"]["value"] == 0.75
    # oracle pairs carry no candidate diagnostics
    so = score_pair(G, gt, "oracle", CFG)
    assert so["counts"]["diag_cand"] == 0 and aggregate([so], CFG)["metrics"]["cand_recall_new"]["value"] is None
