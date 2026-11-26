import numpy as np

from conftest import random_rigid
from lsg.geom.transforms import apply, backproject, invert, mat4_from_colmajor


def test_colmajor():
    T = np.arange(16, dtype=float).reshape(4, 4)
    flat_colmajor = T.T.reshape(-1).tolist()  # column-major serialisation
    np.testing.assert_array_equal(mat4_from_colmajor(flat_colmajor), T)


def test_invert(rng):
    T = random_rigid(rng)
    np.testing.assert_allclose(invert(T) @ T, np.eye(4), atol=1e-12)


def test_backproject_project_roundtrip(rng):
    K = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]])
    depth = rng.uniform(0.5, 4.0, size=(172, 224)).astype(np.float32)
    depth[::7, ::5] = 0  # invalid pixels are skipped
    T_c2w = random_rigid(rng)
    pw = backproject(depth, K, T_c2w)
    assert len(pw) == np.count_nonzero(depth)
    pc = apply(invert(T_c2w), pw)
    np.testing.assert_allclose(pc[:, 2], depth[depth > 0], rtol=1e-6)
    v, u = np.nonzero(depth > 0)  # pinhole projection gives the pixel grid back
    np.testing.assert_allclose(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], u, atol=1e-6)
    np.testing.assert_allclose(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2], v, atol=1e-6)
