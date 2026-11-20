"""M7 method variants (ablations): immediate commitment and "children follow parent"."""

import copy
from pathlib import Path

import numpy as np

from lsg.change.baselines import b2, full
from lsg.change.online import run_online
from lsg.change.update import update_graph
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.geom.transforms import invert
from lsg.graph.model import validate
from lsg.obs import RefContext, SeqInfo
from synth import INTR, T_ALIGN, FakeSeq, make_scan
from test_b2 import OBJ, VOTES, oobs, scene
from test_online import RecordingSource, ref_graph, room_parts, walk

CFG = load_config()


def immediate(cfg=CFG):
    c = copy.deepcopy(cfg)
    c["states"]["commitment"] = "immediate"
    return c


def committed_votes():
    """r2's old place was seen empty mid-walk (committed), but r2 is observed 0.6 m away."""
    v = copy.deepcopy(VOTES)
    v["r2"] = NodeVotes(raw_free=4, free=2, commit_removed_frame=30)
    v["r4"] = NodeVotes(raw_free=5, free=3, commit_removed_frame=12)
    return v


def test_deferred_moves_what_immediate_commits_removed():
    G, ctx = scene()
    O = [oobs("o0", 1), oobs("o1", 2, (0, 0.6, 0), frac=0.6), oobs("o2", 3)]
    for method in (b2, full):
        D = method(G, O, CFG, "S", "oracle", votes=committed_votes(), ref_ctx=ctx)
        assert D.node("r2").state == "moved" and D.node("r4").state == "removed"
        assert not any(n.state == "new" for n in D.nodes)
        I = method(G, O, immediate(), "S", "oracle", votes=committed_votes(), ref_ctx=ctx)
        assert I.node("r2").state == "removed" and I.node("r4").state == "removed"
        # its observation is mis-assigned: the unseen chair r5 "moves" 3 m to take it
        assert I.node("r2").obs_token is None
        assert I.node("r5").state == "moved" and I.node("r5").obs_token == "o1"
        assert I.meta["assoc"]["committed_removed"] == 2 and validate(I) == []
        # without another chair to take it, it becomes a spurious new node
        J = method(G, O, immediate(), "S", "oracle", votes={**committed_votes(), "r5": NodeVotes(free=2,
                   commit_removed_frame=3)}, ref_ctx=ctx)  # fmt: skip
        new = [n for n in J.nodes if n.state == "new"]
        assert len(new) == 1 and new[0].obs_token == "o1"


def test_immediate_commitment_is_not_vetoed_by_confidence():
    G, ctx = scene()
    v = {"r5": NodeVotes(raw_seen=9, seen=3, raw_free=2, free=2, commit_removed_frame=6)}  # seen later
    assert full(G, [], CFG, votes=v, ref_ctx=ctx).node("r5").state == "unknown"  # deferred: vetoed
    assert full(G, [], immediate(), votes=v, ref_ctx=ctx).node("r5").state == "removed"


def test_no_commitment_frame_means_deferred_behaviour():
    G, ctx = scene()
    O = [oobs("o0", 1), oobs("o1", 2, (0, 0.6, 0), frac=0.6), oobs("o2", 3)]
    a = b2(G, O, CFG, votes=VOTES, ref_ctx=ctx)
    b = b2(G, O, immediate(), votes=VOTES, ref_ctx=ctx)  # VOTES carry no commitment frames
    assert {n.id: n.state for n in a.nodes} == {n.id: n.state for n in b.nodes}


class _Priors:
    """Stub prior table: every class persists with p (a bed-like 0.999 needs 4 free votes)."""

    def __init__(self, p):
        self.p = p

    def p_persist(self, label, nyu40):
        return self.p


def _walk_votes(priors=None):
    R = make_scan(room_parts(), "REF")
    G = ref_graph(R)
    S = make_scan(room_parts(with_box=False), "S", T=invert(T_ALIGN))
    seq = FakeSeq(S, walk())
    info = SeqInfo("S", Path("."), INTR, T_ALIGN, list(seq.frame_ids))
    cfg = copy.deepcopy(CFG)
    cfg["confidence"].update(class_priors=True, p_del=0.2)  # the class-prior machinery (FULL+classprior, M7)
    res = run_online(RefContext("REF", R, G, 0.0), info, seq, RecordingSource([]), cfg, priors=priors)
    return res.votes["r10"], res.votes["r11"]


def test_online_records_first_tentative_removed_frame():
    v10, v11 = _walk_votes()
    assert v10.free >= 2 and v10.commit_removed_frame is not None and v11.commit_removed_frame is None
    # with a strong prior the tentative confidence must also drop below p_del: later or never
    w10, _ = _walk_votes(_Priors(0.999))
    assert w10.commit_removed_frame is None or w10.commit_removed_frame > v10.commit_removed_frame


def test_children_follow_moved_anchor():
    G, ctx = scene()
    votes = {"r1": VOTES["r1"]}  # table r3 and cup r4: no counted votes
    O = [oobs("o0", 1), oobs("o2", 3, (0, 0.6, 0))]  # the table moved +0.6 m in y, the cup was not seen
    U = full(G, O, CFG, votes=votes, ref_ctx=ctx)
    assert U.node("r3").state == "moved" and U.node("r4").state == "unknown"
    cfg = copy.deepcopy(CFG)
    cfg["update"]["children_follow_parent"] = True
    F = full(G, O, cfg, votes=votes, ref_ctx=ctx)
    assert F.node("r4").state == "moved" and F.meta["assoc"]["followed_parent"] == 1
    np.testing.assert_allclose(F.node("r4").centroid, OBJ[4][0].mean(0) + [0, 0.6, 0], atol=0.02)
    np.testing.assert_allclose(F.node("r4").T_move, F.node("r3").T_move)
    # a cup with counted votes (e.g. seen in place) does not follow; things on the floor never do
    votes2 = {**votes, "r4": NodeVotes(raw_seen=2, seen=1)}
    assert full(G, O, cfg, votes=votes2, ref_ctx=ctx).node("r4").state == "unknown"
    assert F.node("r5").state == "unknown"
    # the relation update keeps the cup on the moved table
    Gu = update_graph(F, ctx, O, cfg, "update")
    assert Gu.support_parent()["r4"][0] == "r3" and validate(Gu) == []
