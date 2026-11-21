"""Residual noise rejection and thin-object candidate confidence (CHANGE 3 Part N, N5)."""

import copy
import sys
from pathlib import Path

import numpy as np

from lsg.config import load_config
from lsg.obs import RefContext
from lsg.perception.residual import SurfaceFilter, depth_edges, fit_plane, nearer_mask, on_planes

sys.path.insert(0, str(Path(__file__).parent))
from synth import box, make_scan, quad  # noqa: E402

CFG = load_config()


def test_depth_edges_mark_both_sides_of_a_jump_only():
    d = np.ones((5, 6))
    d[:, 3:] = 2.0
    e = depth_edges(d, CFG)
    assert e[:, 2].all() and e[:, 3].all() and not e[:, [0, 1, 4, 5]].any()
    ramp = 1.0 + 0.01 * np.tile(np.arange(6), (5, 1))
    assert not depth_edges(ramp, CFG).any()
    hole = np.ones((5, 6))
    hole[2, 2] = 0.0  # invalid pixels never make an edge
    assert not depth_edges(hole, CFG).any()


def test_edge_switch_removes_flying_pixels_from_the_nearer_mask():
    d_obs = np.full((5, 6), 1.0)
    d_obs[:, 3:] = 2.5
    d_exp = np.full((5, 6), 3.0)
    c = copy.deepcopy(CFG)
    assert nearer_mask(d_obs, d_exp, c).all()
    c["perception"]["noise"]["edge"]["enabled"] = True
    m = nearer_mask(d_obs, d_exp, c)
    assert not m[:, 2:4].any() and m[:, [0, 1, 4, 5]].all()


def _ctx(parts):
    R = make_scan(parts, "REF")
    return RefContext("REF", R, None, 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})


def _wall_with_hole():
    """A wall at x = 4 that covers only y <= 0 (the reference mesh has a hole above y = 0), plus a floor."""
    return {1: (*quad([-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]), 2, "floor"),
            2: (*quad([4, -6, 0], [4, 0, 0], [4, 0, 3], [4, -6, 3]), 1, "wall"),
            11: (*box([0, 0, 0], [0.4, 0.4, 0.4]), 40, "basket")}  # fmt: skip


def test_fit_plane_and_extent():
    ctx = _ctx(_wall_with_hole())
    p = fit_plane(ctx, 2, min_vertices=3, max_rms=0.03, margin=0.3)
    assert p is not None and abs(abs(p[1][0]) - 1.0) < 1e-9  # wall normal along x
    assert fit_plane(ctx, 11, min_vertices=3, max_rms=0.03, margin=0.3) is None  # a box is not planar
    assert fit_plane(ctx, 2, min_vertices=100, max_rms=0.03, margin=0.3) is None  # too few vertices
    planes = [(*p, 0.10)]
    pts = np.array([[3.95, -1.0, 1.0], [3.95, 0.2, 1.0], [3.95, 1.0, 1.0], [3.5, -1.0, 1.0]])
    assert on_planes(pts, planes).tolist() == [True, True, False, False]  # 0.2 is inside the 0.3 m margin


def test_planes_and_dscale_in_the_surface_filter():
    parts = _wall_with_hole()
    c = copy.deepcopy(CFG)
    c["perception"]["noise"]["planes"].update(enabled=True, min_vertices=3, margin=0.3)
    off, on = SurfaceFilter(_ctx(parts), CFG), SurfaceFilter(_ctx(parts), c)
    hole_pt = np.array([[3.95, 0.25, 1.5]])  # 5 cm in front of the wall plane, where the mesh has a hole
    assert off.keep(hole_pt).tolist() == [True] and on.keep(hole_pt).tolist() == [False]
    # dscale: a point 7 cm above the floor mesh survives at 1 m depth, not at 3 m (5 + 1.5 * 3 = 9.5 cm)
    d = copy.deepcopy(CFG)
    d["perception"]["noise"]["dscale"]["enabled"] = True
    ds = SurfaceFilter(_ctx(parts), d)
    p = np.array([[-2.0, -2.0, 0.07]])
    assert ds.keep(p, np.array([1.0])).tolist() == [True] and ds.keep(p, np.array([3.0])).tolist() == [False]
    assert ds.keep(p).tolist() == [True]  # without depths: the plain surf_dist


def test_thin_object_candidates_get_a_confidence_when_switched_on():
    import test_appearance as ta
    from lsg.change.online import run_online
    from lsg.perception.labeler import StubLabeler
    from lsg.perception.source import PerceptionSource

    c = copy.deepcopy(load_config())  # the colour-feature setup of test_appearance's cfg fixture
    c["appearance"]["features"] = "color"
    c["appearance"]["tau_app"]["color"] = 0.15
    c["appearance"]["tau_node"]["color"] = 0.15
    out = {}
    for flag in (False, True):
        cc = copy.deepcopy(c)
        cc["confidence"]["candidates"]["track"]["appearance"] = flag
        ctx, info, seq, runner = ta.setup(cc, ref_book=False, rescan_book=True)
        src = PerceptionSource(StubLabeler((("book", 1.0),)), {"book": 23}, cc, with_crops=False)
        res = run_online(ctx, info, seq, src, cc, obs_mode="perception", appearance=runner)
        assert len(res.obs) == 1 and res.obs[0].kind == "appearance"
        out[flag] = res.obs[0]
    assert out[False].confidence is None and out[False].cand_votes is None
    s, f = out[True].cand_votes
    assert s >= 3 and f == 0 and out[True].confidence is not None and out[True].confidence > 0.5
