"""Seen votes need contrast (M7): second-surface depth, the frame-vote
rule, and the synthetic acceptance scenes (pillow against a backrest removed: old rule `seen`,
new rule `noinfo`; an object 1 m in front of a wall: `seen` under both)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from lsg.change.online import run_online
from lsg.change.visibility import FREE, NOINFO, SEEN, frame_votes
from lsg.config import load_config
from lsg.geom.render import Renderer
from lsg.geom.transforms import invert
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict
from lsg.obs import RefContext, SeqInfo
from synth import INTR, K, T_ALIGN, FakeSeq, H, W, box, look_at, make_scan, quad

CFG = load_config()


# ---- Renderer.depth_behind -------------------------------------------------------------------


def _scene(with_wall=True):
    """Camera frame = world (identity pose): box (label 2) at z 1.0-1.5, wall (label 1) at z 3."""
    Vb, Fb = box([-0.3, -0.3, 1.0], [0.3, 0.3, 1.5])
    parts = [(Vb, Fb, 2)]
    if with_wall:
        Vw = np.array([[-50, -50, 3.0], [50, -50, 3.0], [50, 50, 3.0], [-50, 50, 3.0]])
        parts.append((Vw, np.array([[0, 1, 2], [0, 2, 3]]), 1))
    V, F, L, off = [], [], [], 0
    for v, f, l in parts:
        V.append(v)
        F.append(f + off)
        L.append(np.full(len(f), l))
        off += len(v)
    return Renderer(np.concatenate(V), np.concatenate(F), np.concatenate(L))


def test_depth_behind_skips_own_faces_and_finds_next_surface():
    r = _scene()
    d, lab = r.render(K, np.eye(4), W, H)
    on_box = lab == 2
    assert on_box.sum() > 100 and np.allclose(d[on_box], 1.0, atol=1e-4)
    db = r.depth_behind(K, np.eye(4), W, H, d, lab, on_box)
    # the ray passes the box's front (1.0) and back (1.5) faces and stops at the wall
    np.testing.assert_allclose(db[on_box], 3.0, atol=1e-3)
    assert np.all(np.isinf(db[~on_box]))  # inactive pixels


def test_depth_behind_no_surface_is_inf():
    r = _scene(with_wall=False)
    d, lab = r.render(K, np.eye(4), W, H)
    on_box = lab == 2
    assert np.all(np.isinf(r.depth_behind(K, np.eye(4), W, H, d, lab, on_box)[on_box]))


def test_depth_behind_finds_surface_coincident_with_back_face():
    """A 'pillow' box whose back face lies on a 'backrest' plane: the backrest is found."""
    Vb, Fb = box([-0.3, -0.3, 1.0], [0.3, 0.3, 1.2])
    Vw = np.array([[-50, -50, 1.2], [50, -50, 1.2], [50, 50, 1.2], [-50, 50, 1.2]])
    r = Renderer(np.concatenate([Vb, Vw]), np.concatenate([Fb, np.array([[0, 1, 2], [0, 2, 3]]) + 8]),
                 np.concatenate([np.full(12, 2), np.full(2, 1)]))  # fmt: skip
    d, lab = r.render(K, np.eye(4), W, H)
    on_box = lab == 2
    db = r.depth_behind(K, np.eye(4), W, H, d, lab, on_box)
    np.testing.assert_allclose(db[on_box], 1.2, atol=1e-3)


# ---- frame_votes(d_behind=) ------------------------------------------------------------------


def _maps():
    d_exp = np.full((H, W), 2.0, np.float32)
    lab = np.zeros((H, W), np.int64)
    lab[20:40, 50:70] = 10
    return d_exp, lab


def test_seen_needs_contrastive_pixels():
    d_exp, lab = _maps()
    near = np.full((H, W), 2.05)  # background 5 cm behind: a gone node would look the same
    far = np.full((H, W), 3.0)
    assert frame_votes([10], d_exp, lab, d_exp, CFG)[10].kind == SEEN  # M3 rule
    assert frame_votes([10], d_exp, lab, d_exp, CFG, d_behind=far)[10].kind == SEEN
    v = frame_votes([10], d_exp, lab, d_exp, CFG, d_behind=near)[10]
    assert v.kind == NOINFO and v.n_present == 400 and v.n_contrast == 0
    # free is decided over all valid pixels, unchanged by the contrast rule
    assert frame_votes([10], d_exp, lab, d_exp + 1.0, CFG, d_behind=near)[10].kind == FREE


def test_seen_counts_only_over_contrastive_pixels():
    d_exp, lab = _maps()
    db = np.full((H, W), 2.05)
    db[20:22, 50:70] = 3.0  # 40 contrastive pixels, all present
    v = frame_votes([10], d_exp, lab, d_exp, CFG, d_behind=db)[10]
    assert v.kind == SEEN and v.n_contrast == 40 and v.n_present_contrast == 40
    db = np.full((H, W), 2.05)
    db[20, 50:60] = 3.0  # 10 < min_mask_px contrastive pixels
    assert frame_votes([10], d_exp, lab, d_exp, CFG, d_behind=db)[10].kind == NOINFO
    obs = d_exp.copy()
    db = np.full((H, W), 3.0)
    obs[20:30, 50:70] = 1.0  # half the node occluded: 200 present of 400 contrastive
    assert frame_votes([10], d_exp, lab, obs, CFG, d_behind=db)[10].kind == SEEN
    obs[30, 50:70] = 1.0  # 180 present < 50% of 400
    assert frame_votes([10], d_exp, lab, obs, CFG, d_behind=db)[10].kind == NOINFO


# ---- synthetic online scenes -----------------------------------------------------------------

BACKREST = ([-1.0, 2.2, 0.0], [1.0, 2.6, 1.0])  # r12, anchor (sofa backrest)
PILLOW = ([-0.25, 2.12, 0.4], [0.25, 2.2, 0.8])  # r13, 8 cm against the backrest
CRATE = ([-1.6, 1.8, 0.0], [-1.2, 2.2, 0.5])  # r14, on the floor, 1.8 m in front of the wall


def parts(with_pillow=True):
    p = {
        1: (*quad([-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]), 2, "floor"),
        2: (*quad([-6, 4, -1], [6, 4, -1], [6, 4, 4], [-6, 4, 4]), 1, "wall"),
        12: (*box(*BACKREST), 6, "sofa"),
        14: (*box(*CRATE), 29, "box"),
    }
    if with_pillow:
        p[13] = (*box(*PILLOW), 18, "pillow")
    return p


def graph(R):
    nodes = [Node(ROOM_ID, "room", "room", last_seen=R.scan_id)]
    for oid, lev in ((12, "anchor"), (13, "thing"), (14, "thing")):
        i = R.instances[oid]
        nodes.append(Node(f"r{oid}", lev, i.label, i.nyu40, bbox_3d=bbox_dict(i.aabb_min, i.aabb_max),
                          centroid=list(i.centroid), last_seen=R.scan_id))  # fmt: skip
    edges = [Edge(ROOM_ID, f"r{o}", "hierarchy") for o in (12, 13, 14)] + [Edge("r13", "r12", "on")]
    meta = Graph.new_meta(R.scan_id)
    meta["floor_z"] = 0.0
    return Graph(meta, nodes, edges)


class _Src:
    def begin(self, *a):
        pass

    def observe(self, *a):
        pass

    def finish(self):
        return []


@pytest.fixture(scope="module")
def ref():
    R = make_scan(parts(), "REF")
    return R, graph(R)


def walk(n=10):
    """Frontal views from 2.2 m, eye height 0.6 m, along x in [-0.6, 0.6]."""
    return [look_at([x, 0.0, 0.6], [x * 0.5, 2.2, 0.6]) for x in np.linspace(-0.6, 0.6, n)]


def votes(ref, with_pillow, contrast):
    R, G = ref
    S = make_scan(parts(with_pillow), "S", T=invert(T_ALIGN))
    seq = FakeSeq(S, walk())
    info = SeqInfo("S", Path("."), INTR, T_ALIGN, list(seq.frame_ids))
    res = run_online(RefContext("REF", R, G, 0.0), info, seq, _Src(), copy.deepcopy(CFG), contrast=contrast)
    return res.votes


def test_removed_pillow_against_backrest_seen_by_old_rule_noinfo_by_new(ref):
    old = votes(ref, with_pillow=False, contrast=False)["r13"]
    new = votes(ref, with_pillow=False, contrast=True)["r13"]
    assert old.raw_seen >= 8 and old.seen >= 3 and old.free == 0  # 8 cm < tau_near: "seen" though gone
    assert new.raw_seen == 0 and new.seen == 0 and new.noinfo >= 8 and new.free == 0


def test_object_in_front_of_wall_seen_under_both_rules(ref):
    for contrast in (False, True):
        v = votes(ref, with_pillow=True, contrast=contrast)
        assert v["r14"].raw_seen >= 8 and v["r14"].seen >= 3, contrast
        assert v["r12"].free == 0 and v["r13"].free == 0
