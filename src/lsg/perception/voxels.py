"""Voxel evidence with viewpoint decorrelation (DESIGN §3 candidate steps 3-4).

Residual points are accumulated in a voxel grid. A voxel's hit in a frame **counts** as a new
view only if the camera centre moved >= `d_dec` or the optical axis turned >= `theta_dec`
since the voxel's last counted view: the `change.visibility.Decorrelator` rule, vectorised
over all voxels of a frame. Kept voxels have >= `min_views` counted views; this filters
people walking through, sensor noise and one-off artefacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_BITS = 21
_OFF = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    """(N, 3) integer voxel coordinates -> (N,) int64 keys (21 bits per axis)."""
    a = np.asarray(ijk, np.int64) + _OFF
    return (a[:, 0] << (2 * _BITS)) | (a[:, 1] << _BITS) | a[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    """Inverse of `pack_keys`."""
    k = np.asarray(keys, np.int64)
    return np.stack([(k >> (2 * _BITS)) & _MASK, (k >> _BITS) & _MASK, k & _MASK], axis=1) - _OFF


class VoxelEvidence:
    """Per-voxel decorrelated view counts and point means."""

    def __init__(self, voxel: float, d_dec: float, theta_dec_deg: float, enabled: bool = True):
        """
        Args:
            voxel: Voxel edge length [m].
            d_dec, theta_dec_deg: Decorrelation thresholds (config `decorrelation`).
            enabled: False = every frame hit counts (decorrelation-off ablation).
        """
        self.voxel = float(voxel)
        self.d_dec = float(d_dec)
        self.cos_dec = float(np.cos(np.radians(theta_dec_deg)))
        self.enabled = bool(enabled)
        self._index: dict[int, int] = {}
        self._n = 0
        cap = 1024
        self.keys = np.zeros(cap, np.int64)
        self.views = np.zeros(cap, np.int64)
        self.hits = np.zeros(cap, np.int64)  # raw (frame) hits
        self.last_c = np.zeros((cap, 3))
        self.last_z = np.zeros((cap, 3))
        self.psum = np.zeros((cap, 3))
        self.pcount = np.zeros(cap, np.int64)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> VoxelEvidence:
        """Voxel size from `perception.voxel`, thresholds from `decorrelation`."""
        d = cfg["decorrelation"]
        return cls(float(cfg["perception"]["voxel"]), float(d["d_dec"]), float(d["theta_dec_deg"]),
                   bool(d.get("enabled", True)))  # fmt: skip

    def __len__(self) -> int:
        """Number of voxels hit so far."""
        return self._n

    def _grow(self, need: int) -> None:
        """Make room for `need` voxels."""
        cap = len(self.keys)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in ("keys", "views", "hits", "last_c", "last_z", "psum", "pcount"):
            a = getattr(self, name)
            b = np.zeros((new, *a.shape[1:]), a.dtype)
            b[:cap] = a
            setattr(self, name, b)

    def voxel_keys(self, points: np.ndarray) -> np.ndarray:
        """Packed keys of the voxels containing `points`."""
        return pack_keys(np.floor(np.asarray(points, float) / self.voxel).astype(np.int64))

    def add(self, points: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
        """Add one frame's residual points seen from `cam_pose` (4x4 camera -> reference).

        Returns:
            (N,) voxel index of every point (indices are stable across frames).
        """
        pts = np.asarray(points, float).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, np.int64)
        keys = self.voxel_keys(pts)
        uk, inv = np.unique(keys, return_inverse=True)
        uidx = np.empty(len(uk), np.int64)
        new_keys = []
        for j, k in enumerate(uk.tolist()):
            i = self._index.get(k)
            if i is None:
                i = self._n + len(new_keys)
                self._index[k] = i
                new_keys.append(k)
            uidx[j] = i
        if new_keys:
            self._grow(self._n + len(new_keys))
            self.keys[self._n : self._n + len(new_keys)] = new_keys
            self._n += len(new_keys)
        c = np.asarray(cam_pose[:3, 3], float)
        z = np.asarray(cam_pose[:3, 2], float)
        z = z / (np.linalg.norm(z) or 1.0)
        first = self.views[uidx] == 0
        if self.enabled:
            far = np.linalg.norm(self.last_c[uidx] - c, axis=1) >= self.d_dec
            turned = self.last_z[uidx] @ z <= self.cos_dec
            count = first | far | turned
        else:
            count = np.ones(len(uidx), bool)
        ci = uidx[count]
        self.views[ci] += 1
        self.last_c[ci] = c
        self.last_z[ci] = z
        self.hits[uidx] += 1
        idx = uidx[inv]
        np.add.at(self.psum, idx, pts)
        np.add.at(self.pcount, idx, 1)
        return idx

    def kept(self, min_views: int) -> np.ndarray:
        """Indices of voxels with >= `min_views` counted views, in key order."""
        idx = np.flatnonzero(self.views[: self._n] >= int(min_views))
        return idx[np.argsort(self.keys[idx], kind="stable")]

    def coords(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) integer voxel coordinates of voxel indices."""
        return unpack_keys(self.keys[np.asarray(idx, np.int64)])

    def mean_points(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) mean residual point of each voxel."""
        idx = np.asarray(idx, np.int64)
        return self.psum[idx] / np.maximum(self.pcount[idx], 1)[:, None]
