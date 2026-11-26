"""B2 (M5): confirmed-in-place skip + absorption, Hungarian, ICP moved test."""

import numpy as np

from conftest import box_points, make_scan
from lsg.change.baselines import METHODS, b1, b2
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, ref_node_id, validate
from lsg.obs import ObsInstance, RefContext

CFG = load_config()
FLOOR_Z = 0.0


def shell(lo, hi, step=0.04):
    """Box surface points (all faces); the grid always includes both ends of every axis."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.linspace(a, b, max(2, int(np.ceil((b - a) / step - 1e-9)) + 1)) for a, b in zip(lo, hi)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return P[(np.isclose(P, lo, atol=1e-9) | np.isclose(P, hi, atol=1e-9)).any(axis=1)]


# reference objects: {oid: (points, nyu40, label)}
OBJ = {
    1: (shell((0, 0, 0), (0.5, 0.5, 0.9)), 5, "chair"),  # confirmed in place
    2: (shell((3, 0, 0), (3.5, 0.5, 0.9)), 5, "chair"),  # moved +0.6 m in y
    3: (shell((0, 3, 0), (1.2, 3.8, 0.75)), 7, "table"),  # anchor
    4: (shell((0.4, 3.3, 0.75), (0.5, 3.4, 0.85), 0.02), 40, "cup"),  # removed (free votes)
    5: (shell((6, 0, 0), (6.5, 0.5, 0.9)), 5, "chair"),  # unseen (no votes, no obs)
}


def scene():
    scan = make_scan(OBJ)
    nodes = []
    for oid, (P, nyu, lab) in OBJ.items():
        lv = "anchor" if nyu == 7 else "thing"
        nodes.append(Node(ref_node_id(oid), lv, lab, nyu, bbox_3d=bbox_dict(P.min(0), P.max(0)),
                          centroid=list(P.mean(0)), last_seen="R"))  # fmt: skip
    edges = [Edge(ROOM_ID, n.id, "hierarchy") for n in nodes]
    edges += [Edge("r1", "floor", "on"), Edge("r2", "floor", "on"), Edge("r4", "r3", "on"), Edge("r5", "floor", "on")]
    G = Graph(Graph.new_meta("R"), [Node(ROOM_ID, "room", "room"), *nodes], edges)
    G.meta["floor_z"] = FLOOR_Z
    return G, RefContext("R", scan, G, FLOOR_Z, {o: v[1] for o, v in OBJ.items()})


def oobs(token, oid, shift=(0, 0, 0), frac=1.0, label=None, nyu=None):
    P, n, lab = OBJ[oid]
    if frac < 1.0:  # partial view: the faces a camera at +x / -y / above sees (frac < 0.5: +x and top only)
        lo, hi = P.min(0), P.max(0)
        vis = np.isclose(P[:, 0], hi[0]) | np.isclose(P[:, 2], hi[2])
        if frac >= 0.5:
            vis |= np.isclose(P[:, 1], lo[1])
        P = P[vis]
    P = P + np.asarray(shift, float)
    return ObsInstance(token, label or lab, [(label or lab, 1.0)], nyu or n, P)


VOTES = {
    "r1": NodeVotes(raw_seen=10, seen=4),
    "r3": NodeVotes(raw_seen=8, seen=3),
    "r2": NodeVotes(raw_free=4, free=2),  # old place seen empty, but it is matched at the new one
    "r4": NodeVotes(raw_free=5, free=3),
}


def test_b2_states_absorption_and_icp():
    G, ctx = scene()
    lamp = ObsInstance("o9", "lamp", [("lamp", 1.0)], 35, box_points((8, 8, 0), (8.2, 8.2, 1.2), 0.05))
    O = [oobs("o0", 1, (0.01, 0, 0), frac=0.5), oobs("o1", 2, (0, 0.6, 0), frac=0.6), oobs("o2", 3), lamp]
    U = b2(G, O, CFG, rescan_id="S", obs_mode="oracle", votes=VOTES, ref_ctx=ctx)
    st = {n.id: n.state for n in U.nodes if n.level != "room"}
    assert st == {"r1": "unchanged", "r2": "moved", "r3": "unchanged", "r4": "removed", "r5": "unknown",
                  "n0": "new"}  # fmt: skip
    nm = U.node_map()
    # confirmed nodes absorbed their own observations: no false new node, obs_token set
    assert nm["r1"].obs_token == "o0" and nm["r3"].obs_token == "o2" and nm["n0"].obs_token == "o9"
    # moved: ICP centroid of the full reference points, not the partial-view centroid
    np.testing.assert_allclose(nm["r2"].centroid, OBJ[2][0].mean(0) + [0, 0.6, 0], atol=0.02)
    assert np.array(nm["r2"].T_move).shape == (4, 4)
    assert nm["r2"].last_seen == "S" and nm["r5"].last_seen == "R"
    assert U.meta["method"] == "B2" and U.meta["assoc"]["confirmed"] == 2 and U.meta["assoc"]["absorbed"] == 2
    assert validate(U) == []
    assert all(n.state == "unchanged" for n in G.nodes)  # G_ref untouched


def test_b2_unseen_node_does_not_steal_far_same_label_obs_from_confirmed():
    # r1 confirmed in place; its observation must not become r5's (unseen chair 6 m away) "move"
    G, ctx = scene()
    U = b2(G, [oobs("o0", 1)], CFG, obs_mode="oracle", votes=VOTES, ref_ctx=ctx)
    assert U.node("r5").state == "unknown" and U.node("r1").obs_token == "o0"
    # an observation exactly where r5 stands goes to r5 (unchanged), not to the confirmed r1
    U2 = b2(G, [oobs("oX", 1, (6.0, 0, 0))], CFG, obs_mode="oracle", votes=VOTES, ref_ctx=ctx)
    assert U2.node("r5").state == "unchanged" and U2.node("r1").state == "unchanged"


def test_b2_small_partial_offset_is_unchanged_not_moved():
    # half view of an unchanged chair: centroid offset > tau_move, ICP displacement ~0
    G, ctx = scene()
    votes = {"r2": NodeVotes(raw_seen=2, seen=2)}  # not confirmed (k_seen = 3)
    O = [oobs("o1", 2, frac=0.4)]
    off = np.linalg.norm(O[0].centroid - OBJ[2][0].mean(0))
    assert off > CFG["classify"]["tau_move"]
    assert b2(G, O, CFG, obs_mode="oracle", votes=votes, ref_ctx=ctx).node("r2").state == "unchanged"
    # B1 ignores partial views too (near_dist 0.5 m), so both agree on this one
    assert b1(G, O, CFG, obs_mode="oracle", votes=votes).node("r2").state == "unchanged"


def test_b2_registered_and_perception_mode():
    assert METHODS["B2"] is b2
    G, ctx = scene()
    cand = ObsInstance("c0", "armchair", [("armchair", 0.5), ("chair", 0.3)], 5, OBJ[2][0] + [0, 0.6, 0])
    U = b2(G, [cand], CFG, obs_mode="perception", votes=VOTES, ref_ctx=ctx)
    assert U.node("r2").state == "moved" and U.node("r2").obs_token == "c0"
