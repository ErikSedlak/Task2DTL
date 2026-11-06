"""Baseline B1 (DESIGN §8): B0 matching + visibility removal rule, on synthetic votes."""

import copy

from lsg.change.baselines import METHODS, b0, b1
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.graph.model import validate
from test_b0 import g_ref, obs

CFG = load_config()


def test_unmatched_needs_k_free_counted_free_votes():
    O = [obs("o0", "chair", (0.1, 0, 0)), obs("o1", "chair", (3, 1.5, 0))]  # r1 unchanged, r2 moved
    votes = {
        "r3": NodeVotes(raw_free=5, free=2),  # table: 2 counted free -> removed
        "r4": NodeVotes(raw_free=9, free=1, raw_seen=2, seen=1),  # cup: 1 counted -> unknown
        "r1": NodeVotes(raw_free=4, free=4),  # matched: free votes do not matter in B1
    }
    U = b1(g_ref(), O, CFG, rescan_id="S", obs_mode="oracle", votes=votes)
    st = {n.id: n.state for n in U.nodes if n.level != "room"}
    assert st == {"r1": "unchanged", "r2": "moved", "r3": "removed", "r4": "unknown"}
    nm = U.node_map()
    assert nm["r4"].last_seen == "S" and nm["r3"].last_seen == "R"  # r4 had a raw seen vote
    assert U.meta["method"] == "B1" and U.meta["tentative"] is False
    assert validate(U) == []
    # matched nodes are exactly as in B0
    B = b0(g_ref(), O, CFG, rescan_id="S")
    for nid in ("r1", "r2"):
        assert (B.node(nid).state, B.node(nid).centroid, B.node(nid).obs_token) == (
            nm[nid].state, nm[nid].centroid, nm[nid].obs_token)


def test_no_votes_means_unknown_and_new_nodes_kept():
    U = b1(g_ref(), [obs("o5", "lamp", (6, 0, 0), nyu=35)], CFG)
    st = {n.id: n.state for n in U.nodes if n.level != "room"}
    assert st == {"r1": "unknown", "r2": "unknown", "r3": "unknown", "r4": "unknown", "n0": "new"}


def test_k_free_from_config_and_registry():
    cfg = copy.deepcopy(CFG)
    cfg["states"]["k_free"] = 1
    U = b1(g_ref(), [], cfg, votes={"r4": NodeVotes(raw_free=1, free=1)})
    assert U.node("r4").state == "removed"
    assert METHODS["B1"] is b1 and METHODS["B0"] is b0
    # B0 accepts (and ignores) votes
    assert b0(g_ref(), [], CFG, votes={"r4": NodeVotes()}).node("r4").state == "removed"
