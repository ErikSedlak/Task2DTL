"""Graph update (DESIGN §6.12, M5): support relations under the update / stale / geometric policies."""

import copy

import numpy as np
import pytest

from conftest import make_scan
from lsg.change.baselines import translation
from lsg.change.update import update_graph
from lsg.config import load_config
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, ref_node_id, validate
from lsg.obs import ObsInstance, RefContext
from test_b2 import shell

CFG = load_config()

OBJ = {
    3: (shell((0, 3, 0), (1.2, 3.8, 0.75), 0.02), 7, "table"),
    6: (shell((2.0, 3.0, 0), (2.6, 3.4, 1.2), 0.02), 15, "shelves"),
    4: (shell((0.4, 3.3, 0.75), (0.5, 3.4, 0.85), 0.02), 40, "cup"),
    8: (shell((0.7, 3.2, 0.75), (1.0, 3.5, 0.78), 0.02), 40, "tray"),
    1: (shell((4, 0, 0), (4.5, 0.5, 0.9), 0.02), 5, "chair"),
}
LEVEL = {7: "anchor", 15: "anchor"}


def graph():
    scan = make_scan(OBJ)
    nodes = [Node(ref_node_id(o), LEVEL.get(n, "thing"), lab, n, bbox_3d=bbox_dict(P.min(0), P.max(0)),
                  centroid=list(P.mean(0)), last_seen="R") for o, (P, n, lab) in OBJ.items()]  # fmt: skip
    edges = [Edge(ROOM_ID, n.id, "hierarchy") for n in nodes]
    edges += [Edge("r4", "r3", "on"), Edge("r8", "r3", "on"), Edge("r1", "floor", "on")]
    G = Graph(Graph.new_meta("R"), [Node(ROOM_ID, "room", "room"), *nodes], edges)
    for n in G.nodes:
        if n.id in ("r4", "r8"):
            n.anchor = "r3"
    return G, RefContext("R", scan, G, 0.0, {o: v[1] for o, v in OBJ.items()})


def move(G, nid, t):
    n = G.node(nid)
    t = np.asarray(t, float)
    n.state = "moved"
    n.T_move = translation(t)
    n.centroid = list(np.asarray(n.centroid) + t)
    n.bbox_3d = bbox_dict(np.asarray(n.bbox_3d["min"]) + t, np.asarray(n.bbox_3d["max"]) + t)


def test_moved_cup_gets_new_parent_shelf():
    G, ctx = graph()
    move(G, "r4", (2.25 - 0.45, 3.2 - 3.35, 1.2 - 0.75))  # onto the shelf top
    U = update_graph(G, ctx, [], CFG, "update")
    sp = U.support_parent()
    assert sp["r4"] == ("r6", "on") and U.node("r4").anchor == "r6"
    assert sp["r8"] == ("r3", "on") and sp["r1"] == ("floor", "on")  # untouched
    assert U.meta["relation_policy"] == "update" and validate(U) == []
    # stale (R0) keeps the reference parent
    assert update_graph(G, ctx, [], CFG, "stale").support_parent()["r4"] == ("r3", "on")


def test_child_of_moved_parent_is_recomputed():
    G, ctx = graph()
    move(G, "r3", (0, -2.5, 0))  # the table moves away; the unobserved tray stays in the air
    U = update_graph(G, ctx, [], CFG, "update")
    sp = U.support_parent()
    assert "r8" not in sp and "r4" not in sp  # nothing below them any more (no floor either)
    assert U.node("r8").anchor is None


def test_child_of_removed_parent_never_keeps_it():
    G, ctx = graph()
    G.node("r3").state = "removed"
    U = update_graph(G, ctx, [], CFG, "update")
    assert "r4" not in U.support_parent()
    # the removed node itself stays in the file (view filter)
    assert U.node("r3").state == "removed" and validate(U) == []


def test_new_node_gets_parent():
    G, ctx = graph()
    P = shell((0.2, 3.5, 0.75), (0.3, 3.6, 0.9), 0.02)
    G.nodes.append(Node("n0", "thing", "bottle", 40, bbox_3d=bbox_dict(P.min(0), P.max(0)), centroid=list(P.mean(0)),
                        state="new", obs_token="o7"))  # fmt: skip
    G.edges.append(Edge(ROOM_ID, "n0", "hierarchy"))
    o = ObsInstance("o7", "bottle", [("bottle", 1.0)], 40, P)
    U = update_graph(G, ctx, [o], CFG, "update")
    assert U.support_parent()["n0"] == ("r3", "on") and U.node("n0").anchor == "r3"
    assert "n0" not in update_graph(G, ctx, [o], CFG, "stale").support_parent()  # R0: no relation for new
    assert ("n0", "r3") in {(e.source, e.target) for e in U.edges if e.type == "next_to"} or \
        ("r3", "n0") in {(e.source, e.target) for e in U.edges if e.type == "next_to"}


def test_geometric_policy_recomputes_everything():
    G, ctx = graph()
    G2 = copy.deepcopy(G)
    G2.edges = [e for e in G2.edges if e.type == "hierarchy"]  # wipe reference relations
    U = update_graph(G2, ctx, [], CFG, "geometric")
    sp = U.support_parent()
    assert sp["r4"] == ("r3", "on") and sp["r8"] == ("r3", "on") and sp["r1"] == ("floor", "on")
    assert validate(U) == []


def test_unknown_policy():
    G, ctx = graph()
    with pytest.raises(ValueError):
        update_graph(G, ctx, [], CFG, "magic")


def test_moved_thing_keeps_parent_that_still_supports_it():
    G, ctx = graph()
    move(G, "r8", (-0.4, 0.1, 0))  # tray slides along the table top
    move(G, "r1", (1.0, 0.5, 0))  # chair moves along the floor
    U = update_graph(G, ctx, [], CFG, "update")
    sp = U.support_parent()
    assert sp["r8"] == ("r3", "on") and sp["r1"] == ("floor", "on")
    # with the rule off, infer_support re-picks (same answer here, but via the full search)
    cfg = copy.deepcopy(CFG)
    cfg["update"]["keep_supported_parent"] = False
    assert update_graph(G, ctx, [], cfg, "update").support_parent()["r8"] == ("r3", "on")


def test_still_supports_rules():
    from lsg.graph.relations import FLOOR, SupportCandidate, still_supports

    rc = CFG["relations"]
    table = SupportCandidate("r3", "anchor", OBJ[3][0])
    cup = OBJ[4][0]
    assert still_supports(cup, table, "on", 0.0, rc) == "on"
    assert still_supports(cup + [3, 0, 0], table, "on", 0.0, rc) is None  # moved off the table
    assert still_supports(OBJ[1][0], FLOOR, "on", 0.0, rc) == "on"
    assert still_supports(cup, FLOOR, "on", 0.0, rc) is None  # 0.75 m above the floor
