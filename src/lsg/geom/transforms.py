"""Rigid-transform helpers. All matrices are 4x4 homogeneous, units metres."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mat4_from_colmajor(t: Sequence[float]) -> np.ndarray:
    """Turn a 3RScan.json transform (16 floats, column-major) into a 4x4 matrix (DESIGN §2.1).

    Args:
        t: The 16 floats as stored in the JSON.

    Returns:
        4x4 float64 matrix, translation in the last column, in metres.
    """
    a = np.asarray(t, dtype=np.float64)
    if a.size != 16:
        raise ValueError(f"expected 16 floats, got {a.size}")
    return a.reshape(4, 4).T.copy()


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, computed as [R^T | -R^T t] (exact and cheaper than inv).

    Only valid for rotation + translation (R orthonormal), which all transforms here are.
    """
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points.

    Args:
        T: 4x4 rigid transform, e.g. camera->world or rescan->reference.
        pts: (N, 3) points.

    Returns:
        (N, 3) float64 transformed points.
    """
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
    """Turn a depth image into 3D points (one point per valid pixel).

    Args:
        depth: (H, W) depth in metres; 0 marks invalid pixels, which are skipped.
        K: 3x3 intrinsics matching the depth resolution (`Intrinsics.K_depth`).
        T_c2w: Optional camera->world pose. Without it, points stay in the camera frame.

    Returns:
        (N, 3) points, N = number of valid pixels, in row-major pixel order.
    """
    H, W = depth.shape
    u, v = pixel_grid(W, H)
    m = depth > 0
    z = depth[m]
    x = (u[m] - K[0, 2]) * z / K[0, 0]
    y = (v[m] - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    return apply(T_c2w, pts) if T_c2w is not None else pts


def rotation_angle_deg(T: np.ndarray) -> float:
    """Size of the rotation part of T in degrees (angle about its axis), 0..180."""
    c = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
