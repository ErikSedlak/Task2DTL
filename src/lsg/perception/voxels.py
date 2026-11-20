"""Voxel evidence with viewpoint decorrelation (DESIGN §3 candidate steps 3-4; CHANGE 2).

Residual points are accumulated in a voxel grid. A voxel's hit in a frame **counts** as a new
view only if the camera centre moved >= `d_dec` or the optical axis turned >= `theta_dec`
since the voxel's last counted view: the `change.visibility.Decorrelator` rule, vectorised
over all voxels of a frame. Kept voxels have >= `min_views` counted views; this filters
people walking through, sensor noise and one-off artefacts.

Per-voxel log-odds (CHANGE 2 Part B, `confidence.candidates`, docs/CHANGE_confidence_unified.md):
every voxel also carries `lv = clip(hit_sum - free_sum, ±clip)`.
- hit: a decorrelated hit adds `L_hit`; a correlated one adds `L_corr`, the correlated additions
  since the voxel's last decorrelated hit capped at `corr_cap`;
- free (`observe_free`): the voxel's mean point projects into the image and the minimum depth of
  a (2w+1)² window around its pixel lies more than `free_margin` behind it (seen through). Any
  invalid pixel in the window, out of view, behind the camera or a hit in the same frame = no
  information. Decorrelated free votes subtract `L_free`, correlated ones `L_corr` (own cap);
  hits and free votes have separate decorrelation states.
With `L_corr = 0` and no free votes, `lv = L_hit · views`, so `kept_by_logodds(L_hit · m)` equals
`kept(m)` (below the clip).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_BITS = 21
_OFF = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1
_EPS = 1e-9


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    """(N, 3) integer voxel coordinates -> (N,) int64 keys (21 bits per axis)."""
    a = np.asarray(ijk, np.int64) + _OFF
    return (a[:, 0] << (2 * _BITS)) | (a[:, 1] << _BITS) | a[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    """Inverse of `pack_keys`."""
    k = np.asarray(keys, np.int64)
    return np.stack([(k >> (2 * _BITS)) & _MASK, (k >> _BITS) & _MASK, k & _MASK], axis=1) - _OFF


def window_min_depth(depth: np.ndarray, window: int, max_depth: float = np.inf) -> np.ndarray:
    """Minimum depth over a (2w+1)² window; invalid (0, non-finite, > `max_depth`) pixels count as 0.

    A window touching an invalid pixel or the image border therefore has minimum 0 (no see-through).
    """
    from scipy.ndimage import minimum_filter

    d = np.asarray(depth, np.float64).copy()
    d[~np.isfinite(d) | (d <= 0) | (d > max_depth)] = 0.0
    if window <= 0:
        return d
    return minimum_filter(d, size=2 * int(window) + 1, mode="constant", cval=0.0)


class VoxelEvidence:
    """Per-voxel decorrelated view counts, point means and log-odds."""

    _ARRAYS = ("keys", "views", "hits", "last_c", "last_z", "psum", "pcount",
               "lpos", "lneg", "corr_hit", "corr_free", "nfree", "last_c_free", "last_z_free")  # fmt: skip

    def __init__(
        self,
        voxel: float,
        d_dec: float,
        theta_dec_deg: float,
        enabled: bool = True,
        L_hit: float = 2.5,
        L_free: float = 2.5,
        L_corr: float = 0.0,
        corr_cap: float = 0.0,
        clip: float = np.inf,
        free_margin: float = 0.15,
    ):
        """
        Args:
            voxel: Voxel edge length [m].
            d_dec, theta_dec_deg: Decorrelation thresholds (config `decorrelation`).
            enabled: False = every frame hit counts (decorrelation-off ablation).
            L_hit, L_free: Log-odds of a decorrelated hit / see-through vote.
            L_corr, corr_cap: Correlated vote and the cap of correlated additions (0 = none).
            clip: |lv| clip.
            free_margin: [m] see-through needs the window depth > z + free_margin.
        """
        self.voxel = float(voxel)
        self.d_dec = float(d_dec)
        self.cos_dec = float(np.cos(np.radians(theta_dec_deg)))
        self.enabled = bool(enabled)
        self.L_hit, self.L_free = float(L_hit), float(L_free)
        self.L_corr, self.corr_cap = float(L_corr), float(corr_cap)
        self.clip = float(clip)
        self.free_margin = float(free_margin)
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
        self.lpos = np.zeros(cap)  # hit log-odds sum
        self.lneg = np.zeros(cap)  # see-through log-odds sum
        self.corr_hit = np.zeros(cap)  # correlated hit additions since the last decorrelated hit
        self.corr_free = np.zeros(cap)
        self.nfree = np.zeros(cap, np.int64)  # counted (decorrelated) free votes
        self.last_c_free = np.zeros((cap, 3))
        self.last_z_free = np.zeros((cap, 3))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], logodds: bool = False) -> VoxelEvidence:
        """Voxel size from `perception.voxel`, thresholds from `decorrelation`.

        `logodds=True`: vote weights from `confidence` (`L_hit`, `L_miss`, `clip`) and
        `confidence.candidates` (`L_corr`, `corr_cap`, `free_margin`); else neutral weights.
        """
        d = cfg["decorrelation"]
        kw: dict[str, float] = {}
        if logodds:
            c = cfg["confidence"]
            cc = c["candidates"]
            kw = {"L_hit": float(c["L_hit"]), "L_free": float(c["L_miss"]), "L_corr": float(cc["L_corr"]),
                  "corr_cap": float(cc["corr_cap"]), "clip": float(c["clip"]), "free_margin": float(cc["free_margin"])}  # fmt: skip
        return cls(float(cfg["perception"]["voxel"]), float(d["d_dec"]), float(d["theta_dec_deg"]),
                   bool(d.get("enabled", True)), **kw)  # fmt: skip

    def __len__(self) -> int:
        """Number of voxels hit so far."""
        return self._n

    def _grow(self, need: int) -> None:
        """Make room for `need` voxels."""
        cap = len(self.keys)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in self._ARRAYS:
            a = getattr(self, name)
            b = np.zeros((new, *a.shape[1:]), a.dtype)
            b[:cap] = a
            setattr(self, name, b)

    def voxel_keys(self, points: np.ndarray) -> np.ndarray:
        """Packed keys of the voxels containing `points`."""
        return pack_keys(np.floor(np.asarray(points, float) / self.voxel).astype(np.int64))

    def _decorrelated(self, idx: np.ndarray, first: np.ndarray, last_c: np.ndarray, last_z: np.ndarray,
                      c: np.ndarray, z: np.ndarray) -> np.ndarray:  # fmt: skip
        """Which of `idx` count as a new viewpoint (first vote, moved or turned)."""
        if not self.enabled:
            return np.ones(len(idx), bool)
        far = np.linalg.norm(last_c[idx] - c, axis=1) >= self.d_dec
        turned = last_z[idx] @ z <= self.cos_dec
        return first | far | turned

    def _correlated_add(self, idx: np.ndarray, acc: np.ndarray, target: np.ndarray) -> None:
        """`target[idx] += min(L_corr, corr_cap - acc[idx])` (capped correlated votes)."""
        if self.L_corr <= 0 or len(idx) == 0:
            return
        add = np.minimum(self.L_corr, np.maximum(self.corr_cap - acc[idx], 0.0))
        target[idx] += add
        acc[idx] += add

    @staticmethod
    def _axis(cam_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(cam_pose[:3, 3], float)
        z = np.asarray(cam_pose[:3, 2], float)
        return c, z / (np.linalg.norm(z) or 1.0)

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
        c, z = self._axis(cam_pose)
        count = self._decorrelated(uidx, self.views[uidx] == 0, self.last_c, self.last_z, c, z)
        ci = uidx[count]
        self.views[ci] += 1
        self.last_c[ci] = c
        self.last_z[ci] = z
        self.lpos[ci] += self.L_hit
        self.corr_hit[ci] = 0.0
        self._correlated_add(uidx[~count], self.corr_hit, self.lpos)
        self.hits[uidx] += 1
        idx = uidx[inv]
        np.add.at(self.psum, idx, pts)
        np.add.at(self.pcount, idx, 1)
        return idx

    def observe_free(
        self, dmin: np.ndarray, K: np.ndarray, cam_pose: np.ndarray, skip: np.ndarray | None = None
    ) -> int:
        """See-through votes of one frame on every tracked voxel (module docstring).

        Args:
            dmin: Window-minimum depth image of the frame (`window_min_depth`), [m].
            K: 3x3 intrinsics of that image.
            cam_pose: 4x4 camera -> reference.
            skip: Voxel indices hit in this frame (they get no free vote).

        Returns:
            Number of decorrelated free votes cast.
        """
        n = self._n
        if n == 0:
            return 0
        H, W = dmin.shape
        X = self.psum[:n] / np.maximum(self.pcount[:n], 1)[:, None]
        R, t = np.asarray(cam_pose[:3, :3], float), np.asarray(cam_pose[:3, 3], float)
        Xc = (X - t) @ R  # reference -> camera (R orthonormal)
        zc = Xc[:, 2]
        ok = zc > 1e-3
        u = np.full(n, -1, np.int64)
        v = np.full(n, -1, np.int64)
        u[ok] = np.round(K[0, 0] * Xc[ok, 0] / zc[ok] + K[0, 2]).astype(np.int64)
        v[ok] = np.round(K[1, 1] * Xc[ok, 1] / zc[ok] + K[1, 2]).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        free = np.zeros(n, bool)
        free[ok] = dmin[v[ok], u[ok]] > zc[ok] + self.free_margin
        if skip is not None and len(skip):
            free[np.asarray(skip, np.int64)] = False
        fi = np.flatnonzero(free)
        if len(fi) == 0:
            return 0
        c, z = self._axis(cam_pose)
        count = self._decorrelated(fi, self.nfree[fi] == 0, self.last_c_free, self.last_z_free, c, z)
        ci = fi[count]
        self.nfree[ci] += 1
        self.last_c_free[ci] = c
        self.last_z_free[ci] = z
        self.lneg[ci] += self.L_free
        self.corr_free[ci] = 0.0
        self._correlated_add(fi[~count], self.corr_free, self.lneg)
        return int(len(ci))

    def logodds(self, idx: np.ndarray | None = None) -> np.ndarray:
        """lv = clip(hit_sum - free_sum, ±clip) of voxel indices (all voxels if None)."""
        i = np.arange(self._n) if idx is None else np.asarray(idx, np.int64)
        return np.clip(self.lpos[i] - self.lneg[i], -self.clip, self.clip)

    def kept(self, min_views: int) -> np.ndarray:
        """Indices of voxels with >= `min_views` counted views, in key order."""
        idx = np.flatnonzero(self.views[: self._n] >= int(min_views))
        return idx[np.argsort(self.keys[idx], kind="stable")]

    def kept_by_logodds(self, l_keep: float) -> np.ndarray:
        """Indices of voxels with lv >= `l_keep`, in key order (CHANGE 2)."""
        idx = np.flatnonzero(self.logodds() >= float(l_keep) - _EPS)
        return idx[np.argsort(self.keys[idx], kind="stable")]

    def seen_through(self, l_keep: float) -> int:
        """Voxels whose hits alone pass `l_keep` but whose see-through votes pull them below it."""
        n = self._n
        hit_ok = np.clip(self.lpos[:n], -self.clip, self.clip) >= float(l_keep) - _EPS
        return int(np.sum(hit_ok & (self.logodds() < float(l_keep) - _EPS)))

    def coords(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) integer voxel coordinates of voxel indices."""
        return unpack_keys(self.keys[np.asarray(idx, np.int64)])

    def mean_points(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) mean residual point of each voxel."""
        idx = np.asarray(idx, np.int64)
        return self.psum[idx] / np.maximum(self.pcount[idx], 1)[:, None]
