"""Baseline B0 (DESIGN §8) on a synthetic reference graph + observations."""

import numpy as np

from conftest import box_points
from lsg.change.baselines import b0
from lsg.config import load_config
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, validate
from lsg.obs import ObsInstance

CFG = load_config()


def ref_node(nid, label, c, h=0.2, level="thing", nyu=5):
    c = np.asarray(c, float)
    return Node(nid, level, label, nyu, bbox_3d=bbox_dict(c - h, c + h), centroid=list(c), last_seen="R")


def obs(token, label, c, nyu=5, h=0.2):
    c = np.asarray(c, float)
    return ObsInstance(token, label, [(label, 1.0)], nyu, box_points(c - h, c + h, 0.1))


def g_ref():
    nodes = [
        ref_node("r1", "chair", (0, 0, 0)),
        ref_node("r2", "chair", (3, 0, 0)),
        ref_node("r3", "table", (0, 3, 0), level="anchor", nyu=7),
        ref_node("r4", "cup", (5, 5, 0), nyu=40),
    ]
    edges = [Edge(ROOM_ID, n.id, "hierarchy") for n in nodes] + [Edge("r4", "r3", "on")]
    return Graph(Graph.new_meta("R"), [Node(ROOM_ID, "room", "room"), *nodes], edges)


def test_b0_states_and_new_nodes():
    G = g_ref()
    O = [
        obs("o0", "chair", (0.1, 0, 0)),  # r1 unchanged (0.1 m)
        obs("o1", "chair", (3, 1.5, 0)),  # r2 moved (1.5 m)
        obs("o2", "table", (0, 3.3, 0), nyu=7),  # r3 unchanged (0.3 m)
        obs("o3", "chair", (8, 8, 0)),  # third chair: nobody left -> new
        obs("o4", "wall", (9, 0, 0), nyu=1),  # structure: ignored
        obs("o5", "lamp", (6, 0, 0), nyu=35),  # new
    ]  # fmt: skip
    U = b0(G, O, CFG, rescan_id="S", obs_mode="oracle")
    st = {n.id: n.state for n in U.nodes if n.level != "room"}
    assert st == {"r1": "unchanged", "r2": "moved", "r3": "unchanged", "r4": "removed", "n0": "new", "n1": "new"}
    nm = U.node_map()
    assert {nm["n0"].obs_token, nm["n1"].obs_token} == {"o3", "o5"}
    assert nm["r1"].obs_token == "o0" and nm["r4"].obs_token is None
    r2 = nm["r2"]
    np.testing.assert_allclose(r2.centroid, [3, 1.5, 0], atol=1e-9)
    np.testing.assert_allclose(r2.bbox_3d["min"], [2.8, 1.3, -0.2], atol=1e-9)
    np.testing.assert_allclose(np.array(r2.T_move)[:3, 3], [0, 1.5, 0], atol=1e-9)
    assert r2.last_seen == "S" and nm["r4"].last_seen == "R"
    assert U.meta["obs_mode"] == "oracle" and U.meta["scan_obs"] == "S" and U.meta["method"] == "B0"
    assert validate(U) == []
    # G_ref untouched
    assert all(n.state == "unchanged" for n in G.nodes) and len(G.nodes) == 5


def test_b0_one_to_one_nearest_first():
    # two reference chairs, one observed chair near r2: r2 takes it, r1 is removed
    U = b0(g_ref(), [obs("o0", "chair", (2.9, 0, 0))], CFG)
    st = {n.id: n.state for n in U.nodes}
    assert st["r2"] == "unchanged" and st["r1"] == "removed"
    assert not any(n.state == "new" for n in U.nodes)


def test_b0_label_must_match_exactly():
    U = b0(g_ref(), [obs("o0", "armchair", (0, 0, 0))], CFG)
    st = {n.id: n.state for n in U.nodes}
    assert st["r1"] == "removed" and st["n0"] == "new"


def test_b0_threshold_from_config():
    import copy

    cfg = copy.deepcopy(CFG)
    cfg["baselines"]["b0"]["near_dist"] = 2.0
    U = b0(g_ref(), [obs("o1", "chair", (3, 1.5, 0))], cfg)
    assert U.node("r2").state == "unchanged"
