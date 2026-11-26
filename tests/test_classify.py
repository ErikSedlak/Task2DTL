"""ICP moved test and final states (M5)."""

import numpy as np

from conftest import box_points
from lsg.change.classify import classify, displacement, icp_move
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.geom.transforms import apply
from lsg.obs import ObsInstance
from test_b0 import ref_node

CFG = load_config()


def surface(lo, hi, step=0.02):
    """Points on the 5 visible faces of a box (no bottom), like a reconstructed object."""
    P = box_points(lo, hi, step)
    lo, hi = np.asarray(lo), np.asarray(hi)
    on = np.isclose(P, lo, atol=1e-9) | np.isclose(P, hi, atol=1e-9)
    keep = on.any(axis=1) & ~np.isclose(P[:, 2], lo[2], atol=1e-9)
    return P[keep]


def yaw(deg, t):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


REF = surface((0, 0, 0), (0.8, 0.5, 0.4))  # an asymmetric box


def test_icp_recovers_a_move():
    T_true = yaw(10, (0.25, 0.05, 0))
    obs_pts = apply(T_true, REF)
    T, info = icp_move(REF, obs_pts, CFG)
    assert not info["fallback"] and info["fitness"] > 0.9
    c = REF.mean(axis=0)
    assert abs(displacement(T, c) - displacement(T_true, c)) < 0.01


def test_partial_view_of_unchanged_object_is_not_moved():
    # only the +x half is observed: the centroid offset is 0.2 m, but ICP finds no motion
    part = REF[REF[:, 0] > 0.45]
    off = np.linalg.norm(part.mean(axis=0) - REF.mean(axis=0))
    assert off > CFG["classify"]["tau_move"]
    T, info = icp_move(REF, part, CFG)
    assert displacement(T, REF.mean(axis=0)) < 0.05 and not info["fallback"]  # well below tau_move


def test_low_fitness_falls_back_to_centroid_offset():
    far = REF + np.array([3.0, 0, 0]) + np.random.default_rng(0).normal(0, 0.5, REF.shape)
    T, info = icp_move(REF, far, CFG)
    assert info["fallback"]
    np.testing.assert_allclose(T[:3, 3], far.mean(axis=0) - REF.mean(axis=0), atol=1e-9)


def test_classify_states():
    cfg = CFG
    n_moved = ref_node("r1", "box", REF.mean(axis=0))
    n_still = ref_node("r2", "box", REF.mean(axis=0) + [5, 0, 0])
    n_gone = ref_node("r3", "cup", (9, 9, 0))
    n_unseen = ref_node("r4", "cup", (8, 8, 0))
    n_conf = ref_node("r5", "lamp", (7, 7, 0))
    o1 = ObsInstance("o1", "box", [("box", 1.0)], 40, apply(yaw(0, (0.3, 0, 0)), REF))
    o2 = ObsInstance("o2", "box", [("box", 1.0)], 40, REF + [5.02, 0, 0])
    ref_pts = {"r1": REF, "r2": REF + [5, 0, 0]}
    votes = {"r3": NodeVotes(raw_free=3, free=2), "r4": NodeVotes(raw_free=1, free=1)}
    d = classify([(n_moved, o1, 0.1), (n_still, o2, 0.1)], [n_gone, n_unseen], votes, [n_conf], ref_pts, cfg)
    assert {k: v.state for k, v in d.items()} == {"r1": "moved", "r2": "unchanged", "r3": "removed",
                                                   "r4": "unknown", "r5": "unchanged"}  # fmt: skip
    assert abs(d["r1"].displacement - 0.3) < 0.01 and d["r2"].displacement < 0.05
