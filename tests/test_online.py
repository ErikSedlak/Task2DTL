"""Online loop on a synthetic room (DESIGN §9 M3 acceptance): render -> votes -> decorrelation
-> deferred B1 decision.

Reference room: floor (z = 0), back wall (y = 4), front wall (y = -4), box r10 and basket r11
on the floor ~2 m in front of the cameras. The rescan is the same room in its own frame
(`T_align` != I), with r10 removed and optionally an occluder. Rescan depth is rendered from
the rescan mesh, so unchanged surfaces agree exactly with the reference render.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lsg.change.baselines import b0, b1
from lsg.change.online import run_online, tentative_graph
from lsg.config import load_config
from lsg.geom.transforms import invert
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, validate
from lsg.io.scan3r import Scan
from lsg.obs import ObsInstance, RefContext, RefRender, SeqInfo
from lsg.oracle.source import OracleSource
from synth import INTR, T_ALIGN, FakeSeq, H, W, box, look_at, make_scan, quad

BOX = ([-0.3, 1.8, 0.0], [0.3, 2.2, 0.5])  # r10
BASKET = ([0.6, 1.8, 0.0], [1.0, 2.2, 0.4])  # r11
OCCLUDER = ([-1.5, 1.0, 0.0], [0.8, 1.05, 2.0])  # thin panel between all cameras and r10


def room_parts(with_box=True, occluder=False):
    p = {
        1: (*quad([-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]), 2, "floor"),
        2: (*quad([-6, 4, -1], [6, 4, -1], [6, 4, 4], [-6, 4, 4]), 1, "wall"),
        3: (*quad([-6, -4, -1], [6, -4, -1], [6, -4, 4], [-6, -4, 4]), 1, "wall"),
        11: (*box(*BASKET), 40, "basket"),
    }
    if with_box:
        p[10] = (*box(*BOX), 29, "box")
    if occluder:
        p[20] = (*box(*OCCLUDER), 39, "panel")
    return p


def walk(n=10, away=False):
    """Reference-frame camera poses on a 2 m line at y = 0, height 1.2 m, looking at r10
    (or straight away from it, at the front wall)."""
    poses = []
    for x in np.linspace(-1.0, 1.0, n):
        eye = [x, 0.0, 1.2]
        poses.append(look_at(eye, [x, -3.0, 0.6] if away else [0.0, 2.0, 0.25]))
    return poses


class RecordingSource:
    """ObsSource returning fixed observations; records what `observe` received."""

    def __init__(self, obs):
        self.obs, self.calls, self.begun = obs, [], False

    def begin(self, ref_ctx, seq_info):
        self.begun = True

    def observe(self, frame, ref_render):
        self.calls.append((frame.idx, ref_render))

    def finish(self):
        return list(self.obs)


def ref_graph(R: Scan) -> Graph:
    nodes = [Node(ROOM_ID, "room", "room", last_seen=R.scan_id)]
    for oid, nyu in ((10, 29), (11, 40)):
        i = R.instances[oid]
        nodes.append(Node(f"r{oid}", "thing", i.label, nyu, bbox_3d=bbox_dict(i.aabb_min, i.aabb_max),
                          centroid=list(i.centroid), last_seen=R.scan_id))  # fmt: skip
    edges = [Edge(ROOM_ID, "r10", "hierarchy"), Edge(ROOM_ID, "r11", "hierarchy"),
             Edge("r10", "floor", "on"), Edge("r11", "floor", "on")]  # fmt: skip
    meta = Graph.new_meta(R.scan_id)
    meta["floor_z"] = 0.0
    return Graph(meta, nodes, edges)


def basket_obs():
    lo, hi = np.array(BASKET[0]), np.array(BASKET[1])
    pts = np.stack(np.meshgrid(*[np.linspace(a, b, 5) for a, b in zip(lo, hi)]), -1).reshape(-1, 3)
    return ObsInstance("o0", "basket", [("basket", 1.0)], 40, pts)


@pytest.fixture(scope="module")
def ref():
    R = make_scan(room_parts(), "REF")
    return R, ref_graph(R)


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def run(ref, cfg, rescan_parts, poses, src=None, snapshot_dir=None):
    R, G = ref
    S = make_scan(rescan_parts, "S", T=invert(T_ALIGN))
    seq = FakeSeq(S, poses)
    ctx = RefContext("REF", R, G, 0.0)
    info = SeqInfo("S", Path("."), INTR, T_ALIGN, list(seq.frame_ids))
    src = src or RecordingSource([basket_obs()])
    res = run_online(ctx, info, seq, src, cfg, snapshot_dir=snapshot_dir, obs_mode="oracle")
    return res, src, S


def states(G):
    return {n.id: n.state for n in G.nodes if n.level != "room"}


def test_removed_box_gets_free_votes_and_is_removed(ref, cfg):
    res, src, _ = run(ref, cfg, room_parts(with_box=False), walk())
    v10, v11 = res.votes["r10"], res.votes["r11"]
    assert v10.raw_free == 10 and v10.free >= 2 and v10.seen == 0
    assert v10.free < v10.raw_free  # 0.22 m steps: not every frame counts
    assert v11.raw_seen >= 8 and v11.free == 0
    G1 = b1(ref[1], res.obs, cfg, "S", "oracle", votes=res.votes)
    assert states(G1) == {"r10": "removed", "r11": "unchanged"}
    assert validate(G1) == [] and G1.meta["method"] == "B1"
    # the loop handed the reference render to the source, every frame, in time order
    assert src.begun and [f for f, _ in src.calls] == [3 * i for i in range(10)]
    rr = src.calls[0][1]
    assert isinstance(rr, RefRender) and rr.depth.shape == (H, W) and 10 in np.unique(rr.labels)


def test_occluded_removed_box_is_unknown(ref, cfg):
    res, _, _ = run(ref, cfg, room_parts(with_box=False, occluder=True), walk())
    v10 = res.votes["r10"]
    assert v10.free == 0 and v10.noinfo >= 8  # nearer pixels: occlusion, no information
    assert states(b1(ref[1], res.obs, cfg, votes=res.votes))["r10"] == "unknown"
    assert states(b0(ref[1], res.obs, cfg))["r10"] == "removed"  # B0 deletes it anyway


def test_out_of_frustum_box_is_unknown(ref, cfg):
    res, _, _ = run(ref, cfg, room_parts(with_box=False), walk(away=True), src=RecordingSource([]))
    v10 = res.votes["r10"]
    assert v10.n_observed_frames == 0 and v10.free == 0
    assert states(b1(ref[1], res.obs, cfg, votes=res.votes)) == {"r10": "unknown", "r11": "unknown"}
    assert states(b0(ref[1], res.obs, cfg))["r10"] == "removed"


def test_one_viewpoint_repeated_is_one_vote(ref, cfg):
    poses = [walk()[0]] * 30
    res, _, _ = run(ref, cfg, room_parts(with_box=False), poses)
    assert res.votes["r10"].raw_free == 30 and res.votes["r10"].free == 1
    assert states(b1(ref[1], res.obs, cfg, votes=res.votes))["r10"] == "unknown"  # 1 < k_free
    cfg["decorrelation"]["enabled"] = False  # ablation: every frame counts
    res, _, _ = run(ref, cfg, room_parts(with_box=False), poses)
    assert res.votes["r10"].free == 30
    assert states(b1(ref[1], res.obs, cfg, votes=res.votes))["r10"] == "removed"


def test_unchanged_room_is_confirmed_in_place(ref, cfg):
    res, _, _ = run(ref, cfg, room_parts(), walk())
    for nid in ("r10", "r11"):
        assert res.votes[nid].free == 0 and res.votes[nid].seen >= 3
    G = tentative_graph(ref[1], res.votes, 27, cfg)
    assert states(G) == {"r10": "unchanged", "r11": "unchanged"}


def test_snapshots_are_tentative_and_commitment_is_deferred(ref, cfg, tmp_path):
    cfg["online"]["snapshot_every"] = 1
    res, _, _ = run(ref, cfg, room_parts(with_box=False), walk(), snapshot_dir=tmp_path / "snap")
    assert len(res.snapshots) == 10
    snaps = [Graph.load(p) for p in res.snapshots]
    assert [g.meta["frame_idx"] for g in snaps] == [3 * i for i in range(10)]
    assert all(g.meta["tentative"] is True and validate(g) == [] for g in snaps)
    seq_r10 = [g.node("r10").state for g in snaps]
    assert seq_r10[0] == "unknown" and seq_r10[-1] == "removed"  # 1 counted free vote, then >= 2
    k = seq_r10.index("removed")
    assert all(s == "unknown" for s in seq_r10[:k]) and all(s == "removed" for s in seq_r10[k:])
    raw = json.loads(res.snapshots[0].read_text())
    assert raw["meta"]["obs_mode"] == "oracle" and raw["meta"]["scan_obs"] == "S"
    # the final graph is not tentative
    G1 = b1(ref[1], res.obs, cfg, "S", "oracle", votes=res.votes)
    assert G1.meta["tentative"] is False and G1.meta["frame_idx"] is None


def test_with_oracle_source(ref, cfg):
    cfg["oracle"]["min_points"] = 1
    S = make_scan(room_parts(with_box=False), "S", T=invert(T_ALIGN))
    src = OracleSource(S, {11: {"nyu40": 40}}, cfg)
    res, _, _ = run(ref, cfg, room_parts(with_box=False), walk(), src=src)
    labels = {o.label for o in res.obs}
    assert "basket" in labels and "box" not in labels
    basket = next(o for o in res.obs if o.label == "basket")
    np.testing.assert_allclose(basket.centroid, np.mean(BASKET, axis=0), atol=0.15)  # reference frame
    U = b1(ref[1], res.obs, cfg, "S", "oracle", votes=res.votes)
    st = states(U)
    assert st["r10"] == "removed" and st["r11"] == "unchanged"
    assert not any(n.state == "new" and n.label == "basket" for n in U.nodes)
