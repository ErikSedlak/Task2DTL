"""OracleSource on a synthetic rescan: masks, points in the reference frame, min_points, tokens."""

import copy
from pathlib import Path

import numpy as np
import pytest

from lsg.config import load_config
from lsg.geom.transforms import apply
from lsg.io.scan3r import Frame, Instance, Intrinsics, Scan
from lsg.obs import ObsSource, PackedMasks, SeqInfo
from lsg.oracle.source import OracleSource, shuffle_seed

K = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]])
W, H = 224, 172
INTR = Intrinsics(K * 4, (960, 540), K, (W, H), 1000.0, 1)


def box(cmin, cmax):
    x0, y0, z0 = cmin
    x1, y1, z1 = cmax
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)  # fmt: skip
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])  # fmt: skip
    return V, F


def make_rescan():
    """Camera at the origin looking along +z: wall (1) at z = 3, box A (10) left, box B (11)
    right, box C (12) hidden behind the wall."""
    parts = {
        1: (np.array([[-20, -20, 3], [20, -20, 3], [20, 20, 3], [-20, 20, 3]], float), np.array([[0, 1, 2], [0, 2, 3]]), 1),
        10: (*box([-0.5, -0.2, 1.5], [-0.1, 0.2, 1.9]), 5),
        11: (*box([0.1, -0.2, 1.5], [0.5, 0.2, 1.9]), 5),
        12: (*box([-0.2, -0.2, 4.0], [0.2, 0.2, 4.4]), 40),
    }  # fmt: skip
    V, F, vo, vn, fl = [], [], [], [], []
    off = 0
    for oid, (v, f, nyu) in parts.items():
        V.append(v)
        F.append(f + off)
        vo.append(np.full(len(v), oid))
        vn.append(np.full(len(v), nyu))
        fl.append(np.full(len(f), oid))
        off += len(v)
    V, F = np.concatenate(V), np.concatenate(F)
    vo, vn, fl = np.concatenate(vo), np.concatenate(vn), np.concatenate(fl)
    inst = {i: Instance(i, "", n, V[vo == i].mean(0), V[vo == i].min(0), V[vo == i].max(0), int((vo == i).sum()))
            for i, (_, _, n) in parts.items()}  # fmt: skip
    labels = {1: "wall", 10: "chair", 11: "chair", 12: "cup"}
    return Scan("S1", V, F, vo, vn, fl, vn[F[:, 0]], inst, INTR, labels)


def depth_frame(idx=0, invalid_from_col=120):
    """Valid depth (2 m) on the left part only: box B (columns >= ~123) is never valid."""
    d = np.full((H, W), 2.0, np.float32)
    d[:, invalid_from_col:] = 0.0
    return Frame(idx, np.eye(4), d, f"frame-{idx:06d}.color.jpg")


T_ALIGN = np.eye(4)
T_ALIGN[:3, 3] = [0.5, -1.0, 0.25]


def run(S, cfg, rescan_id="S1", frames=(0,), **kw):
    src = OracleSource(S, {10: {"nyu40": 5}}, cfg, **kw)
    assert isinstance(src, ObsSource)
    src.begin(None, SeqInfo(rescan_id, Path("."), INTR, T_ALIGN, list(frames)))
    for i in frames:
        src.observe(depth_frame(i), None)
    return src, src.finish()


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["oracle"]["min_points"] = 1
    return c


def test_visible_instances_only_where_depth_is_valid(cfg):
    S = make_rescan()
    src, obs = run(S, cfg)
    tm = src.token_map()
    assert set(tm.values()) == {1, 10}  # B: no valid depth; C: hidden behind the wall
    by_id = {tm[o.token]: o for o in obs}
    a = by_id[10]
    assert a.label == "chair" and a.nyu40 == 5 and a.label_topk == [("chair", 1.0)]
    m = a.masks[0]
    assert m.shape == (H, W) and m.sum() > 100
    assert not m[:, 120:].any()  # invalid sensor pixels are not observed
    assert not (m & by_id[1].masks[0]).any()


def test_points_are_hit_vertices_in_reference_frame(cfg):
    S = make_rescan()
    src, obs = run(S, cfg)
    a = next(o for o in obs if src.token_map()[o.token] == 10)
    ref_vertices = apply(T_ALIGN, S.instance_points(10))
    d = np.linalg.norm(a.points[:, None] - ref_vertices[None], axis=-1).min(axis=1)
    np.testing.assert_allclose(d, 0, atol=1e-9)
    # front face (z = 1.5) and inner side (x = -0.1) are visible: 6 of 8 vertices
    assert len(a.points) == 6


def test_min_points_drops_small_instances(cfg):
    cfg["oracle"]["min_points"] = 5  # wall quad has 4 vertices
    src, obs = run(make_rescan(), cfg)
    assert set(src.token_map().values()) == {10}


def test_masks_accumulate_over_frames(cfg):
    src, obs = run(make_rescan(), cfg, frames=(0, 3, 6))
    for o in obs:
        assert list(o.masks) == [0, 3, 6]


def test_tokens_shuffled_and_deterministic(cfg):
    _, a = run(make_rescan(), cfg)
    _, b = run(make_rescan(), cfg)
    assert [o.token for o in a] == [o.token for o in b]
    assert all(o.token.startswith("o") for o in a)
    assert [o.token for o in a] == sorted(o.token for o in a)
    p1 = np.random.default_rng(shuffle_seed(0, "S1")).permutation(30)
    p2 = np.random.default_rng(shuffle_seed(0, "S2")).permutation(30)
    p3 = np.random.default_rng(shuffle_seed(1, "S1")).permutation(30)
    assert not np.array_equal(p1, p2) and not np.array_equal(p1, p3)
    assert not np.array_equal(p1, np.arange(30))


def test_packed_masks_roundtrip(rng):
    pm = PackedMasks((H, W))
    m1 = rng.random((H, W)) > 0.7
    m2 = rng.random((H, W)) > 0.9
    pm.add(5, m1)
    pm.add(2, m2)
    assert list(pm) == [5, 2] and len(pm) == 2
    np.testing.assert_array_equal(pm[5], m1)
    assert pm.area(2) == int(m2.sum())
    pm.add(5, m2)  # OR into an existing mask
    np.testing.assert_array_equal(pm[5], m1 | m2)
    with pytest.raises(ValueError):
        pm.add(1, np.zeros((10, 10), bool))
