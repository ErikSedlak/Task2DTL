"""CHANGE 3 Part A: a move needs consistent evidence at the old place (FULL only).

Scene from test_b2: r1 chair confirmed, r2 chair at x 3..3.5, r3 table confirmed, r4 cup, r5 chair at
x 6..6.5 (no votes unless set).
"""

import copy

import numpy as np

from lsg.change.association import old_place_gate
from lsg.change.baselines import b2, full
from lsg.change.visibility import NodeVotes, old_place_class
from test_b2 import CFG, OBJ, VOTES, oobs, scene


def cfg_with(**old_place):
    c = copy.deepcopy(CFG)
    c["association"]["old_place"].update(old_place)
    return c


def states(G):
    return {n.id: n.state for n in G.nodes if n.level != "room"}


def test_old_place_classes():
    assert old_place_class(None, CFG) == "unobserved"
    assert old_place_class(NodeVotes(seen=2), CFG) == "net_seen"
    assert old_place_class(NodeVotes(seen=3, free=2), CFG) == "net_seen"  # net seen first (removal veto)
    assert old_place_class(NodeVotes(free=2), CFG) == "free_ev"
    assert old_place_class(NodeVotes(seen=1, free=1), CFG) == "mixed"
    assert old_place_gate(VOTES, cfg_with(enabled=False)) is None


def test_net_seen_node_does_not_move_to_a_far_same_label_obs():
    G, ctx = scene()
    votes = {**VOTES, "r2": NodeVotes(raw_seen=2, seen=2), "r5": NodeVotes(raw_seen=1, seen=1)}
    O = [oobs("o1", 2, (0, 2.4, 0))]  # a chair 2.4 m from r2: in reality another chair
    U = full(G, O, CFG, "S", "oracle", votes=votes, ref_ctx=ctx)
    st = states(U)
    assert st["r2"] == "unknown" and st["r5"] == "unknown" and st.get("n0") == "new"
    off = full(G, O, cfg_with(enabled=False), "S", "oracle", votes=votes, ref_ctx=ctx)
    assert states(off)["r2"] == "moved" and "n0" not in states(off)
    # B2 is not affected by the switch (the rule is FULL only)
    assert states(b2(G, O, CFG, "S", "oracle", votes=votes, ref_ctx=ctx))["r2"] == "moved"


def test_free_evidence_node_still_moves():
    G, ctx = scene()
    U = full(G, [oobs("o1", 2, (0, 0.6, 0), frac=0.6)], CFG, "S", "oracle", votes=VOTES, ref_ctx=ctx)
    assert states(U)["r2"] == "moved"  # r2 has 2 counted free votes in VOTES


def test_net_seen_node_keeps_its_own_instance_inside_its_place():
    G, ctx = scene()
    votes = {**VOTES, "r2": NodeVotes(raw_seen=2, seen=2)}
    U = full(G, [oobs("o1", 2, frac=0.4)], CFG, "S", "oracle", votes=votes, ref_ctx=ctx)
    st = states(U)
    assert st["r2"] == "unchanged" and U.node("r2").obs_token == "o1" and "n0" not in st


def test_unobserved_node_distance_limit_and_inside_only():
    G, ctx = scene()
    votes = {**VOTES, "r2": NodeVotes(raw_seen=2, seen=2)}  # r2 gated; r5 has no votes
    O = [oobs("o5", 5, (0, 3.0, 0))]
    free_unseen = cfg_with(unobserved_inside_only=False)
    assert states(full(G, O, free_unseen, "S", "oracle", votes=votes, ref_ctx=ctx))["r5"] == "moved"
    for c in (cfg_with(unobserved_inside_only=False, max_dist_unseen=1.5), CFG):  # CFG default: inside-only
        st = states(full(G, O, c, "S", "oracle", votes=votes, ref_ctx=ctx))
        assert st["r5"] == "unknown" and st.get("n0") == "new"
    near = [oobs("o5", 5, (0, 1.0, 0))]
    assert states(full(G, near, cfg_with(unobserved_inside_only=False, max_dist_unseen=1.5), "S", "oracle", votes=votes,
                       ref_ctx=ctx))["r5"] == "moved"  # fmt: skip


def test_tau_move_seen_raises_the_move_threshold_for_net_seen_nodes():
    G, ctx = scene()
    votes = {**VOTES, "r2": NodeVotes(raw_seen=2, seen=2)}
    O = [oobs("o1", 2, (0, 0.15, 0))]
    U = full(G, O, cfg_with(tau_move_seen=None), "S", "oracle", votes=votes, ref_ctx=ctx)
    assert states(U)["r2"] == "moved"
    np.testing.assert_allclose(U.node("r2").centroid, OBJ[2][0].mean(0) + [0, 0.15, 0], atol=0.02)
    assert states(full(G, O, CFG, "S", "oracle", votes=votes, ref_ctx=ctx))["r2"] == "unchanged"  # default 0.2
