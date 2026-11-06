"""Frame votes (DESIGN §3) on hand-built depth / label maps, and the viewpoint decorrelator."""

import copy

import numpy as np

from lsg.change.visibility import (
    FREE,
    NOINFO,
    SEEN,
    UNOBSERVED,
    Decorrelator,
    NodeVotes,
    confirmed_in_place,
    erode_labels,
    frame_votes,
    has_free_evidence,
)
from lsg.config import load_config

CFG = load_config()
H, W = 172, 224


def maps():
    """Rendered depth 2 m everywhere; node 10 on a 20x20 patch, node 11 on another."""
    d_exp = np.full((H, W), 2.0, np.float32)
    lab = np.zeros((H, W), np.int64)
    lab[20:40, 50:70] = 10
    lab[100:120, 150:170] = 11
    lab[0:3, 0:4] = 12  # 12 px < min_mask_px
    return d_exp, lab


def test_seen_free_noinfo_unobserved():
    d_exp, lab = maps()
    d_obs = d_exp.copy()
    d_obs[20:40, 50:70] = 3.0  # node 10 seen through (+1 m)
    d_obs[100:120, 150:170] = 1.0  # node 11 occluded (something nearer)
    v = frame_votes([10, 11, 12, 13], d_exp, lab, d_obs, CFG)
    assert v[10].kind == FREE and v[10].n_px == 400 and v[10].n_free == 400
    assert v[11].kind == NOINFO  # occlusion is no information
    assert v[12].kind == UNOBSERVED and v[12].n_px == 12
    assert v[13].kind == UNOBSERVED and v[13].n_px == 0  # not in the frame at all
    v = frame_votes([10], d_exp, lab, d_exp + 0.05, CFG)  # within tau_near
    assert v[10].kind == SEEN and v[10].n_present == 400


def test_thresholds_and_majority():
    d_exp, lab = maps()
    d_obs = d_exp.copy()
    d_obs[20:40, 50:70] = 2.0 + 0.12  # between tau_near and tau_free: neither present nor free
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == NOINFO
    d_obs = d_exp.copy()
    d_obs[20:29, 50:70] = 3.0  # 180 of 400 free (45%), rest present (55%)
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == SEEN
    d_obs[20:31, 50:70] = 3.0  # 220 of 400 free (55%)
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == FREE


def test_invalid_depth_is_no_information():
    d_exp, lab = maps()
    d_obs = np.full_like(d_exp, 3.0)
    d_obs[:, :] = 0.0  # sensor invalid everywhere
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == NOINFO
    d_obs[20:40, 50:70] = 3.0
    d_obs[20:40, 50:69] = 0.0  # only 20 valid pixels -> enough
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == FREE
    d_obs[20:40, 50:70] = 0.0
    d_obs[20:34, 69] = 3.0  # 14 valid pixels < min_mask_px -> noinfo, not free
    v = frame_votes([10], d_exp, lab, d_obs, CFG)[10]
    assert v.kind == NOINFO and v.n_valid == 14


def test_erosion_removes_silhouette_pixels():
    d_exp, lab = maps()
    e = erode_labels(lab, 1)
    assert (e == 10).sum() == 18 * 18 and (e == 12).sum() == 2 * 3  # 3x4 patch; image border does not erode
    cfg = copy.deepcopy(CFG)
    cfg["visibility"]["erode_px"] = 3
    d_obs = d_exp.copy()
    d_obs[20:40, 50:70] = 3.0
    d_obs[23:37, 53:67] = 2.0  # interior present, 3-px rim seen through (silhouette error)
    assert frame_votes([10], d_exp, lab, d_obs, CFG)[10].kind == FREE  # 196/400 present: false free
    v = frame_votes([10], d_exp, lab, d_obs, cfg)[10]
    assert v.kind == SEEN and v.n_px == 14 * 14


def pose(c, yaw_deg=0.0):
    """Camera->world pose at centre c, looking along +y rotated by yaw about +z."""
    a = np.radians(yaw_deg)
    z = np.array([-np.sin(a), np.cos(a), 0.0])
    x = np.array([np.cos(a), np.sin(a), 0.0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = c
    return T


def test_same_viewpoint_30x_counts_once():
    dec = Decorrelator(0.30, 20.0)
    P = pose([0, 0, 1])
    assert sum(dec.counts(("r1", FREE), P) for _ in range(30)) == 1


def test_decorrelation_distance_and_angle():
    dec = Decorrelator(0.30, 20.0)
    assert dec.counts("k", pose([0, 0, 1]))
    assert not dec.counts("k", pose([0.29, 0, 1]))
    assert dec.counts("k", pose([0.31, 0, 1]))
    assert not dec.counts("k", pose([0.31, 0, 1], yaw_deg=19))
    assert dec.counts("k", pose([0.31, 0, 1], yaw_deg=21))
    # the reference viewpoint is the last *counted* one: small steps never accumulate
    dec2 = Decorrelator(0.30, 20.0)
    counted = [0.25 * i for i in range(10) if dec2.counts("k", pose([0.25 * i, 0, 1]))]
    assert counted == [0.0, 0.5, 1.0, 1.5, 2.0]
    # keys are independent
    dec3 = Decorrelator(0.30, 20.0)
    assert dec3.counts(("r1", SEEN), pose([0, 0, 1])) and dec3.counts(("r1", FREE), pose([0, 0, 1]))
    assert dec3.counts(("r2", SEEN), pose([0, 0, 1]))


def test_decorrelation_off_counts_everything():
    dec = Decorrelator(0.30, 20.0, enabled=False)
    assert sum(dec.counts("k", pose([0, 0, 1])) for _ in range(30)) == 30
    assert Decorrelator.from_config(CFG).enabled is True


def test_node_votes_and_state_rules():
    v = NodeVotes()
    for i, (kind, counted) in enumerate([(SEEN, True), (SEEN, False), (UNOBSERVED, False), (NOINFO, False)]):
        v.add(kind, counted, i)
    assert (v.raw_seen, v.seen, v.noinfo, v.n_observed_frames, v.last_seen_frame) == (2, 1, 1, 3, 1)
    assert not confirmed_in_place(v, CFG) and not has_free_evidence(v, CFG)
    v.add(SEEN, True, 5)
    v.add(SEEN, True, 6)
    assert confirmed_in_place(v, CFG)
    v.add(FREE, True, 7)
    v.add(FREE, False, 8)
    assert v.free == 1 and v.raw_free == 2 and v.first_free_frame == 7
    assert not confirmed_in_place(v, CFG) and not has_free_evidence(v, CFG)
    v.add(FREE, True, 9)
    assert has_free_evidence(v, CFG) and not has_free_evidence(None, CFG)
