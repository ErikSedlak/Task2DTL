"""Relation scoring (M5) on a hand-made pair."""

from lsg.config import load_config
from lsg.eval.gt import GTLabels, GTNew, GTObject
from lsg.eval.metrics import aggregate, count_keys
from lsg.eval.relations import relation_count_keys, score_relations
from lsg.graph.model import ROOM_ID, Edge, Graph, Node
from lsg.graph.relations import FLOOR, GTSupport

CFG = load_config()


def gto(i, status, reason=None, level="thing"):
    return GTObject(i, f"obj{i}", 40, level, "small", status, reason)


def pair():
    objects = {
        3: gto(3, "unchanged", level="anchor"),
        6: gto(6, "unchanged", level="anchor"),
        10: gto(10, "unchanged"),  # cup on table, correct
        11: gto(11, "moved"),  # book table -> shelf (swapped), predicted still on table: wrong
        12: gto(12, "moved"),  # pillow gained a floor parent, predicted floor: correct
        13: gto(13, "unchanged"),  # box in table (GT), predicted unknown with "on": parent ok, type wrong
        14: gto(14, "dont_care", "nonrigid"),  # excluded
        15: gto(15, "unchanged"),  # only structure parents -> excluded
        16: gto(16, "unchanged"),  # no GT parent -> skipped
        17: gto(17, "dont_care", "small_move"),  # plate, two GT parents, predicted the second one
    }
    new = {50: GTNew(50, "bag", 40, "thing", "small", None, None, None),
           51: GTNew(51, "box", 40, "thing", "small", None, None, None)}  # fmt: skip
    gt = GTLabels("R", "S", objects, new, 0, 0, 0, 0)
    gs = GTSupport(parents={10: {3: "on"}, 11: {6: "on"}, 12: {FLOOR: "on"}, 13: {3: "in"}, 14: {3: "on"},
                            17: {3: "on", 6: "on"}, 50: {51: "on"}}, excluded={15})  # fmt: skip
    kinds = {11: "swapped", 12: "gained"}
    st = {"r10": "unchanged", "r11": "moved", "r12": "moved", "r13": "unknown", "r14": "moved", "r15": "unchanged",
          "r16": "unchanged", "r17": "unchanged"}  # fmt: skip
    nodes = [Node(ROOM_ID, "room", "room"), Node("r3", "anchor", "table"), Node("r6", "anchor", "shelf")]
    nodes += [Node(k, "thing", "x", state=v) for k, v in st.items()]
    nodes += [Node("n0", "thing", "bag", state="new", obs_token="t0"), Node("n1", "thing", "box", state="new", obs_token="t1")]
    edges = [Edge("r10", "r3", "on"), Edge("r11", "r3", "on"), Edge("r12", "floor", "on"), Edge("r13", "r3", "on"),
             Edge("r14", "r3", "on"), Edge("r15", "r3", "on"), Edge("r17", "r6", "on"), Edge("n0", "n1", "on")]  # fmt: skip
    G = Graph(Graph.new_meta("R"), nodes, edges)
    truth = {"t0": (50, "bag"), "t1": (51, "box")}
    new_tp = {"n0": 50}  # n1 is not a TP here (e.g. unmatched); only matched new things are scored
    return G, gt, gs, kinds, new_tp, truth


def test_relation_counts():
    c = score_relations(*pair(), prefix="upd")
    # (a) GT universe: r10 ok, r11 wrong, r12 ok (floor), r13 parent ok/type wrong, r17 ok, n0 ok -> 5/6
    assert (c["rel_upd_a_gt_all_n"], c["rel_upd_a_gt_all_par"], c["rel_upd_a_gt_all_typ"]) == (6, 5, 4)
    # without floor parents: r12 drops out
    assert (c["rel_upd_a_gt_nf_n"], c["rel_upd_a_gt_nf_par"]) == (5, 4)
    # associated only: r13 (unknown) drops out
    assert (c["rel_upd_a_as_all_n"], c["rel_upd_a_as_all_par"], c["rel_upd_a_as_all_typ"]) == (5, 4, 4)
    # (b) GT moved: r11 wrong, r12 ok
    assert (c["rel_upd_b_gt_all_n"], c["rel_upd_b_gt_all_par"]) == (2, 1)
    # (c) swapped: r11 only; (cb) swapped + gained: r11, r12
    assert (c["rel_upd_c_gt_all_n"], c["rel_upd_c_gt_all_par"]) == (1, 0)
    assert (c["rel_upd_cb_gt_all_n"], c["rel_upd_cb_gt_all_par"]) == (2, 1)
    assert set(c) == {k for k in relation_count_keys() if k.startswith("rel_upd_")}


def test_relation_metrics_aggregate():
    c = score_relations(*pair(), prefix="R0")
    counts = dict.fromkeys(count_keys(CFG), 0) | c
    agg = aggregate([{"counts": counts, "loc_errors": []}], CFG)["metrics"]
    assert abs(agg["rel_R0_a_gt_all_parent_acc"]["value"] - 5 / 6) < 1e-12
    assert abs(agg["rel_R0_a_gt_all_type_acc"]["value"] - 4 / 5) < 1e-12
    assert agg["rel_R1_a_gt_all_parent_acc"]["value"] is None  # not scored -> n = 0
