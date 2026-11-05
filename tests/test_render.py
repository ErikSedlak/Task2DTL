import numpy as np
import pytest

from lsg.geom.render import Renderer
from lsg.geom.transforms import apply, backproject, invert

K = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]])
W, H = 224, 172


def quad(z: float, half: float = 50.0):
    """Big square in the plane z = const (camera frame), two triangles."""
    V = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return V, F


def box(cmin, cmax):
    x0, y0, z0 = cmin
    x1, y1, z1 = cmax
    V = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    )  # fmt: skip
    F = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
         [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    )  # fmt: skip
    return V, F


def test_plane_depth_is_z_everywhere():
    V, F = quad(2.0)
    depth, labels = Renderer(V, F).render(K, np.eye(4), W, H)
    # t_hit with unnormalised rays = camera-z depth, also at off-centre pixels
    np.testing.assert_allclose(depth, 2.0, atol=1e-5)
    assert np.all(labels == 1)


def test_box_in_front_of_plane_labels_and_depth():
    Vp, Fp = quad(3.0)
    Vb, Fb = box([-0.2, -0.2, 1.0], [0.2, 0.2, 1.4])
    V = np.vstack([Vp, Vb])
    F = np.vstack([Fp, Fb + len(Vp)])
    labels_f = np.array([1] * len(Fp) + [7] * len(Fb))
    depth, labels = Renderer(V, F, labels_f).render(K, np.eye(4), W, H)
    u0, v0 = int(round(K[0, 2])), int(round(K[1, 2]))
    assert labels[v0, u0] == 7 and depth[v0, u0] == pytest.approx(1.0, abs=1e-5)
    assert labels[0, 0] == 1 and depth[0, 0] == pytest.approx(3.0, abs=1e-5)
    # box footprint: |x| <= 0.2 at z = 1 -> |u - cx| <= 0.2 * fx
    on_box = labels == 7
    v, u = np.nonzero(on_box)
    assert np.all(np.abs(u - K[0, 2]) <= 0.2 * K[0, 0] + 1)
    assert set(np.unique(labels)) == {1, 7}


def test_no_hit_is_zero():
    V, F = quad(2.0, half=0.1)
    depth, labels = Renderer(V, F, np.array([5, 5])).render(K, np.eye(4), W, H)
    assert depth[0, 0] == 0 and labels[0, 0] == 0
    assert labels[int(K[1, 2]), int(K[0, 2])] == 5


def test_posed_render_matches_backprojection():
    """Render from an arbitrary pose; back-projected depth must lie on the mesh."""
    Vp, Fp = quad(0.0, half=20.0)  # floor plane z = 0 in world
    Vb, Fb = box([-0.5, -0.5, 0.0], [0.5, 0.5, 0.8])
    V = np.vstack([Vp, Vb])
    F = np.vstack([Fp, Fb + len(Vp)])
    # camera 1.5 m above the floor, 3 m away, looking at the box
    eye, target = np.array([3.0, 0.5, 1.5]), np.array([0.0, 0.0, 0.4])
    zc = (target - eye) / np.linalg.norm(target - eye)
    xc = np.cross(zc, [0, 0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = np.stack([xc, yc, zc], axis=1)
    T_c2w[:3, 3] = eye
    r = Renderer(V, F)
    depth, _ = r.render(K, invert(T_c2w), W, H)
    assert np.count_nonzero(depth) > 0.9 * depth.size
    pw = backproject(depth, K, T_c2w)
    assert np.median(r.distance(pw)) < 1e-4
    # and the inverse pose convention would not fit
    pw_wrong = apply(invert(T_c2w), backproject(depth, K))
    assert np.median(r.distance(pw_wrong)) > 0.1


def test_distance():
    V, F = quad(0.0)
    d = Renderer(V, F).distance(np.array([[0, 0, 0.3], [1, 2, -0.05]]))
    np.testing.assert_allclose(d, [0.3, 0.05], atol=1e-6)
