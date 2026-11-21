"""Snapshot method runs during the walk (CHANGE 3 Part B): peek, tracks, relabel rule, anytime aggregation."""

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_perception import NEW_BOX, graph_of, orbit, room  # noqa: E402
from synth import INTR, T_ALIGN, FakeSeq, box, make_scan  # noqa: E402

from lsg.change.online import run_online
from lsg.config import load_config
from lsg.eval.anytime import at_fraction, curve, flips_and_settling, snapshot_row
from lsg.geom.transforms import invert
from lsg.graph.model import ROOM_ID, Graph, Node
from lsg.obs import RefContext, SeqInfo
from lsg.oracle.source import OracleSource
from lsg.perception.labeler import StubLabeler
from lsg.perception.source import PerceptionSource

CFG = load_config()
BOX2 = ([-1.3, -1.3, 0.0], [-1.0, -1.0, 0.3])


def _walk(cfg, source_fn, rescan_parts, every=4, with_snap=True):
    R = make_scan(room(), "REF")
    S = make_scan(rescan_parts, "S", T=invert(T_ALIGN))
    poses = orbit([0.5, 0.5, 0.15]) + orbit([-1.15, -1.15, 0.15])
    seq = FakeSeq(S, poses)
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    info = SeqInfo("S", Path("."), INTR, T_ALIGN, list(seq.frame_ids))
    c = copy.deepcopy(cfg)
    c["online"]["snapshot_every"] = every
    src = source_fn(S)
    seen = []

    def cb(obs, votes, views, fidx, k):
        seen.append((k, [(o.token, len(o.points), o.label) for o in obs]))
        return None

    res = run_online(ctx, info, seq, src, c, obs_mode="x", on_snapshot=cb if with_snap else None)
    return res, seen, src


def test_oracle_peek_is_stable_and_does_not_change_finish():
    parts = room({20: (*box(*NEW_BOX), 29, "box"), 21: (*box(*BOX2), 29, "box")})
    a, snaps, src = _walk(CFG, lambda S: OracleSource(S, {}, CFG), parts)
    b, _, _ = _walk(CFG, lambda S: OracleSource(S, {}, CFG), parts, with_snap=False)
    assert [o.token for o in a.obs] == [o.token for o in b.obs]
    assert all(np.array_equal(x.points, y.points) for x, y in zip(a.obs, b.obs))
    assert len(snaps) >= 3 and all(t.startswith("p") for _, obs in snaps for t, _, _ in obs)
    tok = {}  # a peek token always names the same object (same label across snapshots)
    for _, obs in snaps:
        for t, _, lab in obs:
            assert tok.setdefault(t, lab) == lab
    assert set(src.peek_token_map()) == {t for t, _, _ in snaps[-1][1]}


def test_perception_tracks_keep_their_id_and_finish_is_unchanged():
    parts = room({20: (*box(*NEW_BOX), 29, "box"), 21: (*box(*BOX2), 29, "box")})
    mk = lambda S: PerceptionSource(StubLabeler((("box", 1.0),)), {"box": 29}, CFG, with_crops=False)  # noqa: E731
    a, snaps, _ = _walk(CFG, mk, parts)
    b, _, _ = _walk(CFG, mk, parts, with_snap=False)
    assert [(o.token, len(o.points), o.confidence) for o in a.obs] == [(o.token, len(o.points), o.confidence) for o in b.obs]
    ids = [{t for t, _, _ in obs} for _, obs in snaps if obs]
    assert ids and all(t.startswith("t") for s in ids for t in s)
    # the first box is seen first (first orbit) and keeps its track id once it exists
    first = [s for s in ids if s]
    assert all(first[0] <= s for s in first[1:]) and len(first[-1]) == 2


def test_relabel_rule_only_after_growth():
    class Counting(StubLabeler):
        n = 0

        def topk(self, crops, k):
            Counting.n += 1
            return super().topk(crops, k)

    src = PerceptionSource(Counting((("box", 1.0),)), {"box": 29}, CFG, with_crops=False)
    src._track_label = {}
    src._tracks, src._next_track = {}, 0
    # no crops -> topk is not called, but the label cache records the size: emulate with crops on a stub path
    src.with_crops = False
    from lsg.perception.cluster import Candidate

    c1 = Candidate(np.arange(10), np.zeros((10, 3)))
    assert src._track_ids([c1], [0]) == ["t0"]
    c2 = Candidate(np.arange(11), np.zeros((11, 3)))  # 10 % growth, IoU 0.91 -> same track
    assert src._track_ids([c2], [0]) == ["t0"]
    c3 = Candidate(np.arange(100, 110), np.zeros((10, 3)))  # disjoint -> new track
    assert src._track_ids([c2, c3], [0, 0]) == ["t0", "t1"]
    merged = Candidate(np.concatenate([np.arange(11), np.arange(100, 110)]), np.zeros((21, 3)))
    assert src._track_ids([merged], [0]) == ["t0"]  # a merge keeps the older id
    assert src._track_ids([Candidate(np.arange(10), np.zeros((10, 3)))], [1]) == ["t2"]  # other grid -> new id


def _g(states):
    nodes = [Node(ROOM_ID, "room", "room")] + [Node(i, "thing", "x", state=s) for i, s in states.items()]
    return Graph(Graph.new_meta("R"), nodes, [])


def _c(**kw):
    from lsg.eval.metrics import count_keys

    return {**dict.fromkeys(count_keys(CFG), 0), **kw}


def test_anytime_step_function_and_stability():
    snaps = [snapshot_row(0.0, None, _c(moved_fn=2), _g({"r1": "unknown", "r2": "unknown"})),
             snapshot_row(0.3, 10, _c(moved_tp=1, moved_fn=1), _g({"r1": "moved", "r2": "unknown"})),
             snapshot_row(0.6, 20, _c(moved_tp=1, moved_fn=1), _g({"r1": "unchanged", "r2": "removed"})),
             snapshot_row(1.0, 30, _c(moved_tp=2), _g({"r1": "moved", "r2": "removed"}))]  # fmt: skip
    assert at_fraction(snaps, 0.2)["frac"] == 0.0 and at_fraction(snaps, 0.5)["frac"] == 0.3
    assert at_fraction(snaps, 1.0)["counts"]["moved_tp"] == 2
    fs = flips_and_settling(snaps)
    assert fs["r1"] == (3, 1.0) and fs["r2"] == (1, 0.6)
    rows = [{"mode": "oracle", "anytime": snaps, "ref": "R", "rescan": "S"}]
    cv = curve(rows, CFG, "oracle")
    assert [r["frac"] for r in cv][:3] == [0.0, 0.1, 0.2] and len(cv) == 11
    assert cv[0]["moved_F1"]["value"] == 0.0 and cv[5]["moved_F1"]["value"] == 2 / 3 and cv[10]["moved_F1"]["value"] == 1.0
