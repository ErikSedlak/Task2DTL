"""Appearance residual (DESIGN §3 / §6.6, M6): patch similarity, mismatch maps, thin-node votes,
the synthetic book acceptance check, and thin-object candidates.

Scene: floor, two walls, a table (anchor r20, top at z = 0.75) and a flat striped book (thing
r30, 2 cm thick) on it. The rescan is the same room in its own frame (`T_align` != I); depth and
colour are rendered from the rescan's textured mesh.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from lsg.change.baselines import full
from lsg.change.online import run_online
from lsg.change.visibility import (
    FREE,
    FREE_LIKE,
    NOINFO,
    SEEN,
    UNOBSERVED,
    Decorrelator,
    FrameVote,
    NodeVotes,
    confirmed_in_place,
    has_free_evidence,
    thin_votes,
)
from lsg.config import load_config
from lsg.geom.render import TextureRenderer
from lsg.geom.transforms import invert
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, validate
from lsg.obs import RefContext, SeqInfo
from lsg.perception.appearance import (
    AppearanceRunner,
    ColorFeatures,
    SimilarityCache,
    comparable_pixels,
    mismatch,
    patch_index,
    patch_similarity,
    render_size,
)
from lsg.perception.labeler import StubLabeler
from lsg.perception.source import PerceptionSource
from synth import T_ALIGN, TexFakeSeq, box, look_at, make_scan, quad, solid, stripes, textured

TABLE = ([-0.5, 1.6, 0.0], [0.5, 2.4, 0.75])
BOOK = ([-0.16, 1.88, 0.75], [0.16, 2.12, 0.77])  # 32 x 24 x 2 cm
CELLS = {1: solid((90, 90, 90)), 2: solid((200, 200, 190)), 3: solid((200, 200, 190)),
         20: solid((140, 100, 60)), 30: stripes((200, 30, 30), (240, 220, 40))}  # fmt: skip


def parts(book=True):
    p = {
        1: (*quad([-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]), 2, "floor"),
        2: (*quad([-6, 4, -1], [6, 4, -1], [6, 4, 4], [-6, 4, 4]), 1, "wall"),
        3: (*quad([-6, -4, -1], [6, -4, -1], [6, -4, 4], [-6, -4, 4]), 1, "wall"),
        20: (*box(*TABLE), 7, "table"),
    }
    if book:
        p[30] = (*box(*BOOK), 23, "book")
    return p


def walk(n=10):
    """Reference-frame poses on a 2.4 m line in front of the table, looking down at the book."""
    return [look_at([x, 0.4, 2.0], [0.0, 2.0, 0.76]) for x in np.linspace(-1.2, 1.2, n)]


def graph_of(R):
    nodes = [Node(ROOM_ID, "room", "room", last_seen=R.scan_id)]
    edges = []
    for oid, lvl, nyu in ((20, "anchor", 7), (30, "thing", 23)):
        if oid not in R.instances:
            continue
        i = R.instances[oid]
        nodes.append(Node(f"r{oid}", lvl, i.label, nyu, bbox_3d=bbox_dict(i.aabb_min, i.aabb_max),
                          centroid=list(i.centroid), last_seen=R.scan_id))  # fmt: skip
        edges.append(Edge(ROOM_ID, f"r{oid}", "hierarchy"))
    meta = Graph.new_meta(R.scan_id)
    meta["floor_z"] = 0.0
    return Graph(meta, nodes, edges)


@pytest.fixture
def cfg():
    """Colour-feature config. The synthetic scene's flat colours have their own distance scale
    (book vs table top ~0.3), so the colour thresholds are set for it here; the slow DINOv2
    variant uses the configured (calibrated) values."""
    c = copy.deepcopy(load_config())
    c["appearance"]["features"] = "color"
    c["appearance"]["tau_app"]["color"] = 0.15
    c["appearance"]["tau_node"]["color"] = 0.15
    return c


class NoObs:
    """ObsSource without observations (the table is confirmed in place, the book unmatched)."""

    def begin(self, ref_ctx, seq_info):
        pass

    def observe(self, frame, ref_render):
        pass

    def finish(self):
        return []


def setup(cfg, ref_book=True, rescan_book=False, extractor=None):
    R = make_scan(parts(ref_book), "REF")
    S = make_scan(parts(rescan_book), "S", T=invert(T_ALIGN))
    seq = TexFakeSeq(S, textured(S, CELLS), walk())
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    info = SeqInfo("S", Path("."), seq.info, T_ALIGN, list(seq.frame_ids))
    runner = AppearanceRunner(TextureRenderer(textured(R, CELLS)), extractor or ColorFeatures(), cfg, "REF",
                              SimilarityCache(None))  # fmt: skip
    return ctx, info, seq, runner


# ---- units -------------------------------------------------------------------------------------


def test_color_features_similarity():
    a = np.zeros((28, 42, 3), np.uint8)
    a[:, :21] = (200, 30, 30)
    a[:, 21:] = (30, 200, 30)
    b = a.copy()
    s_same = patch_similarity(a, b, ColorFeatures())
    assert s_same.shape == (2, 3) and np.allclose(s_same, 1.0, atol=1e-5)  # stored orientation grid
    b[:14, :14] = (30, 30, 200)  # one changed patch (stored orientation, top-left)
    s = patch_similarity(a, b, ColorFeatures())
    # per-image normalisation shifts the other patches a little (absorbed by the frame median)
    assert s[0, 0] < 0.5 and (np.delete(s.ravel(), 0) > 0.9).all()


def test_render_size_and_patch_index():
    assert render_size((960, 540), 448) == (448, 252)
    pidx = patch_index((224, 172), (18, 32), (448, 252))
    assert pidx.shape == (172, 224) and pidx[0, 0] == 0 and pidx[-1, -1] == 18 * 32 - 1
    assert pidx[0, 7] == 1 and pidx[0, 6] == 0  # depth col 7 -> render col 14.x -> patch 1


def test_comparable_rule_and_mismatch_map(cfg):
    cfg["appearance"]["min_frame_patches"] = 1
    d_exp = np.full((172, 224), 2.0)
    d_obs = d_exp.copy()
    d_obs[:, :50] = 1.5  # occluder in front: not comparable
    d_obs[:, 50:60] = 2.3  # seen through: not comparable
    d_obs[:, 60:70] = 2.12  # thin-object band (tau_near < diff <= tau_free): comparable
    comp = comparable_pixels(d_obs, d_exp, cfg)
    assert not comp[:, :60].any() and comp[:, 60:].all()
    sim = np.ones((18, 32), np.float32)
    sim[:, 10] = 0.0  # one bad patch column: render cols 140-153 = depth cols 70-76
    am = mismatch(sim, d_obs, d_exp, cfg, (448, 252))
    assert not am.patch_ok[:, :3].any() and am.patch_ok[:, 10:].all()
    cols = np.flatnonzero(am.mismatch_px.any(axis=0))
    assert cols.min() >= 70 and cols.max() <= 77 and not (am.mismatch_px & ~am.comparable_px).any()


def test_relative_distance_removes_global_shift(cfg):
    d = np.full((172, 224), 2.0)
    sim = np.full((18, 32), 0.55, np.float32)  # globally lower similarity (other lighting)
    sim[5, 5] = 0.0
    am = mismatch(sim, d, d, cfg, (448, 252))
    assert am.patch_ok.all()
    assert am.mismatch_px.sum() == (patch_index((224, 172), (18, 32), (448, 252)) == 5 * 32 + 5).sum()
    cfg["appearance"]["relative"] = False
    assert mismatch(sim, d, d, cfg, (448, 252)).mismatch_px.all()  # absolute: everything "changed"


def test_thin_vote_replacement_table(cfg):
    lab = np.zeros((172, 224), np.int64)
    lab[:, :20] = 5  # thin node
    lab[:, 20:40] = 6  # thin node, seen through by depth
    lab[:, 40:60] = 7  # non-thin
    comp = np.ones_like(lab, bool)
    rel = np.zeros(lab.shape, np.float32)
    rel[:, :20] = 0.5
    rel[:, 40:60] = 0.5
    fv = {5: FrameVote(SEEN, 3440, 3440, 3440, 0), 6: FrameVote(FREE, 3440, 3440, 0, 3440),
          7: FrameVote(SEEN, 3440, 3440, 3440, 0), 8: FrameVote(UNOBSERVED, 0, 0, 0, 0)}  # fmt: skip
    out = thin_votes(fv, {5, 6, 8}, lab, comp, rel, cfg, 0.3)
    assert out[5].kind == FREE_LIKE and out[5].n_mismatch == 3440 and out[5].n_comparable == 3440
    assert out[6].kind == FREE and out[7].kind == SEEN and out[8].kind == UNOBSERVED
    assert thin_votes(fv, {5}, lab, comp, np.zeros_like(rel), cfg, 0.3)[5].kind == SEEN  # appearance agrees
    half = rel.copy()
    half[:, :12] = 0.0  # mean over the node = 0.2 <= 0.3: a few bad patches do not decide
    assert thin_votes(fv, {5}, lab, comp, half, cfg, 0.3)[5].kind == SEEN
    few = np.zeros_like(comp)
    few[:2, :5] = True  # 10 comparable pixels < min_mask_px
    assert thin_votes(fv, {5}, lab, few, rel, cfg, 0.3)[5].kind == NOINFO


def test_free_like_evidence_rules(cfg):
    def v(free=0, fl=0, seen=0):
        return NodeVotes(free=free, free_like=fl, seen=seen)

    assert has_free_evidence(v(fl=3), cfg) and not has_free_evidence(v(fl=2), cfg)
    assert has_free_evidence(v(free=1, fl=1), cfg) and has_free_evidence(v(free=2), cfg)
    assert not has_free_evidence(v(free=1), cfg)
    assert not confirmed_in_place(v(seen=5, fl=1), cfg) and confirmed_in_place(v(seen=5), cfg)
    nv = NodeVotes()
    dec = Decorrelator(0.3, 20)
    P = np.eye(4)
    for f in range(30):  # one viewpoint repeated: one counted free-like vote
        nv.add(FREE_LIKE, dec.counts(("r1", FREE_LIKE), P), f)
    assert nv.raw_free_like == 30 and nv.free_like == 1 and nv.first_free_like_frame == 0


def test_similarity_cache_roundtrip(tmp_path):
    c = SimilarityCache(tmp_path)
    k = c.key(["S", 3, "REF", "color", 448, 252], np.eye(4))
    assert c.get(k) is None
    s = c.put(k, np.full((2, 3), 0.123456))
    assert np.allclose(s, 0.123) and np.array_equal(c.get(k), s) and c.hits == 1 and c.misses == 1
    assert c.key(["S", 3, "REF", "color", 448, 252], np.eye(4) + 1e-3) != k


# ---- synthetic book: the M6 acceptance check --------------------------------------------------------


def run_book(cfg, appearance: bool, extractor=None):
    ctx, info, seq, runner = setup(cfg, ref_book=True, rescan_book=False, extractor=extractor)
    res = run_online(ctx, info, seq, NoObs(), cfg, obs_mode="oracle", appearance=runner if appearance else None)
    G = full(ctx.graph, res.obs, cfg, "S", "oracle", votes=res.votes, ref_ctx=ctx, obs_views=res.obs_views)
    return res, G


def test_book_removed_missed_by_depth_caught_by_appearance(cfg):
    res0, G0 = run_book(cfg, appearance=False)
    v0 = res0.votes["r30"]
    assert v0.free == 0 and v0.seen >= 3  # 2 cm: the table behind is within tau_near
    assert G0.node("r30").state == "unchanged" and G0.node("r20").state == "unchanged"  # missed

    res1, G1 = run_book(cfg, appearance=True)
    v1 = res1.votes["r30"]
    assert v1.free_like >= 3 and v1.seen == 0 and res1.stats["thin_nodes"] == 1
    assert G1.node("r30").state == "removed" and G1.node("r30").confidence < 0.2  # caught
    assert G1.node("r20").state == "unchanged" and G1.node("r20").confidence > 0.9
    assert validate(G1) == []


def test_book_unchanged_stays_with_appearance(cfg):
    ctx, info, seq, runner = setup(cfg, ref_book=True, rescan_book=True)
    res = run_online(ctx, info, seq, NoObs(), cfg, obs_mode="oracle", appearance=runner)
    v = res.votes["r30"]
    assert v.free_like == 0 and v.seen >= 3
    G = full(ctx.graph, res.obs, cfg, "S", "oracle", votes=res.votes, ref_ctx=ctx, obs_views=res.obs_views)
    assert G.node("r30").state == "unchanged"


def test_book_added_gives_one_appearance_candidate(cfg):
    ctx, info, seq, runner = setup(cfg, ref_book=False, rescan_book=True)
    src = PerceptionSource(StubLabeler((("book", 1.0),)), {"book": 23}, cfg, with_crops=False)
    run_online(ctx, info, seq, src, cfg, obs_mode="perception")  # depth only
    assert src.finish() == []  # 2 cm above the table: no depth residual
    ctx, info, seq, runner = setup(cfg, ref_book=False, rescan_book=True)
    src = PerceptionSource(StubLabeler((("book", 1.0),)), {"book": 23}, cfg, with_crops=False)
    res = run_online(ctx, info, seq, src, cfg, obs_mode="perception", appearance=runner)
    assert len(res.obs) == 1 and res.obs[0].kind == "appearance" and res.obs[0].token == "c0"
    top = np.array([0.0, 2.0, 0.77])  # observable (top) surface centre of the book
    assert np.linalg.norm(res.obs[0].centroid - top) < 0.05
    assert res.obs_views["c0"] >= 3


def test_thin_candidates_switch_off(cfg):
    cfg["appearance"]["thin_candidates"] = False
    ctx, info, seq, runner = setup(cfg, ref_book=False, rescan_book=True)
    src = PerceptionSource(None, {}, cfg, with_crops=False)
    res = run_online(ctx, info, seq, src, cfg, obs_mode="perception", appearance=runner)
    assert res.obs == [] and res.stats["appearance_frames"] == 0  # no thin node, no candidates wanted


@pytest.mark.slow
def test_book_removed_with_dinov2(cfg):
    from lsg.perception.appearance import DinoV2Features

    cfg["appearance"]["features"] = "dinov2"
    res, G = run_book(cfg, appearance=True, extractor=DinoV2Features(cfg["appearance"]["model"], "cuda"))
    assert res.votes["r30"].free_like >= 3
    assert G.node("r30").state == "removed"
