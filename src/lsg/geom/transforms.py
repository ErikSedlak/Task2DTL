"""Rigid-transform helpers. All matrices are 4x4 homogeneous, units metres."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mat4_from_colmajor(t: Sequence[float]) -> np.ndarray:
    """3RScan.json stores 16 floats column-major (DESIGN §2.1)."""
    a = np.asarray(t, dtype=np.float64)
    if a.size != 16:
        raise ValueError(f"expected 16 floats, got {a.size}")
    return a.reshape(4, 4).T.copy()


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (R orthonormal)."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def pixel_grid(W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer pixel coordinates (u = column, v = row), each (H, W).

    Convention used everywhere: pixel (u, v) <-> x = (u - cx) * z / fx, no +0.5 offset
    (logged in DECISIONS 2026-11-05/M0).
    """
    v, u = np.mgrid[0:H, 0:W]
    return u.astype(np.float64), v.astype(np.float64)


def backproject(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray | None = None) -> np.ndarray:
    """Valid depth pixels (> 0) -> (N, 3) points, in world frame if T_c2w is given."""
    H, W = depth.shape
    u, v = pixel_grid(W, H)
    m = depth > 0
    z = depth[m]
    x = (u[m] - K[0, 2]) * z / K[0, 0]
    y = (v[m] - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    return apply(T_c2w, pts) if T_c2w is not None else pts


def project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixel coords (u, v)."""
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def rotation_angle_deg(T: np.ndarray) -> float:
    c = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
