import numpy as np

from lsg.config import load_config
from lsg.geom.align import mesh_target, refine_alignment
from lsg.geom.transforms import apply, invert, rotation_angle_deg


def grid_room(n: int = 40):
    """Floor + two walls + a box, as a triangulated grid mesh (non-degenerate for ICP)."""
    verts, faces = [], []

    def plane(origin, du, dv):
        o = len(verts)
        for i in range(n + 1):
            for j in range(n + 1):
                verts.append(np.asarray(origin) + du * i / n + dv * j / n)
        for i in range(n):
            for j in range(n):
                a = o + i * (n + 1) + j
                b, c, d = a + 1, a + n + 1, a + n + 2
                faces.extend([[a, c, b], [b, c, d]])

    X, Y, Z = np.eye(3)
    plane([0, 0, 0], 4 * X, 3 * Y)  # floor
    plane([0, 0, 0], 4 * X, 2.5 * Z)  # wall y = 0
    plane([0, 0, 0], 3 * Y, 2.5 * Z)  # wall x = 0
    plane([1.5, 1.0, 0.8], 0.8 * X, 0.6 * Y)  # table top
    return np.array(verts), np.array(faces)


def small_rigid(t, deg):
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def test_icp_recovers_small_misalignment(rng):
    cfg = load_config()["alignment"]
    V, F = grid_room()
    tgt = mesh_target(V, F)
    T_true = small_rigid([0.3, -0.2, 0.05], 20.0)  # rescan -> reference
    pts_ref = V + rng.normal(scale=0.005, size=V.shape)
    pts_rescan = apply(invert(T_true), pts_ref)
    T_init = small_rigid([0.04, -0.03, 0.0], 1.0) @ T_true  # ~5 cm / 1 deg off
    T_hat, info = refine_alignment(pts_rescan, tgt, T_init, cfg)
    err = T_hat @ invert(T_true)
    assert info["accepted"]
    assert np.linalg.norm(err[:3, 3]) < 0.01
    assert rotation_angle_deg(err) < 0.3


def test_icp_rejects_large_correction(rng):
    cfg = dict(load_config()["alignment"], max_delta_t=0.01)
    V, F = grid_room()
    T_init = small_rigid([0.05, 0.0, 0.0], 0.0)
    T_hat, info = refine_alignment(V, mesh_target(V, F), T_init, cfg)
    assert not info["accepted"]
    np.testing.assert_array_equal(T_hat, T_init)
