"""Clustering kept voxels into candidates, and the door swing-region filter (DESIGN §3 step 5).

Kept voxels are grouped by connected components (M4, Erik: no scikit-learn): two voxels are
linked if their Chebyshev distance is <= `link_voxels` (1 = 26-connectivity; the default 2
bridges the one-voxel gaps that sparse, grazing-angle sampling leaves inside one surface).
Clusters below `min_voxels` are dropped. Candidates are ordered
deterministically (size descending, then centroid), which fixes their tokens `c<k>`.

Door swings (DECISIONS M4): the reference mesh has each door in its reference position, so a
swung door leaf leaves "nearer" residuals up to a door width away from the closed-door
surface. A cluster is dropped if >= `door_frac` of its voxels lie inside a reference door's
swing region: a vertical cylinder of radius door width + `door_margin` around each of the
door's two possible hinge edges, over the door's height.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree

from lsg.obs.base import RefContext

DOOR_NYU40 = 8


@dataclass
class Candidate:
    """One cluster of kept voxels (reference frame)."""

    voxels: np.ndarray  # (K,) voxel indices into the VoxelEvidence
    points: np.ndarray  # (K, 3) mean residual point per voxel

    @property
    def centroid(self) -> np.ndarray:
        """Mean of the voxel points."""
        return self.points.mean(axis=0)

    @property
    def aabb_min(self) -> np.ndarray:
        """Lower AABB corner."""
        return self.points.min(axis=0)

    @property
    def aabb_max(self) -> np.ndarray:
        """Upper AABB corner."""
        return self.points.max(axis=0)


def connected_components(coords: np.ndarray, link: int = 1, connectivity: int = 26) -> np.ndarray:
    """Component label (0..n-1) of each integer voxel coordinate.

    Two voxels are linked if their Chebyshev distance is <= `link` (link 1 = 26-connectivity;
    link 2 also bridges one-voxel gaps). `connectivity=6` links face neighbours only.
    Components are numbered by their first voxel in input order (deterministic).

    Args:
        coords: (N, 3) integer voxel coordinates (unique).
        link: Linking distance in voxels.
        connectivity: 26 (Chebyshev metric) or 6 (Manhattan distance 1; `link` ignored).
    """
    coords = np.asarray(coords, np.int64)
    n = len(coords)
    if n == 0:
        return np.zeros(0, np.int64)
    tree = cKDTree(coords.astype(float))
    if connectivity == 6:
        pairs = tree.query_pairs(r=1.0, p=1, output_type="ndarray")
    else:
        pairs = tree.query_pairs(r=float(link) + 1e-9, p=np.inf, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = csgraph.connected_components(g, directed=False)
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(np.argsort(first))  # component -> rank of its first voxel
    return order[lab].astype(np.int64)


def cluster(
    voxel_idx: np.ndarray,
    coords: np.ndarray,
    points: np.ndarray,
    cfg: Mapping[str, Any],
    min_voxels: int | None = None,
) -> list[Candidate]:
    """Connected components of kept voxels with >= `min_voxels` voxels.

    Args:
        voxel_idx: (N,) indices of the kept voxels.
        coords: (N, 3) their integer coordinates.
        points: (N, 3) their mean residual points.
        cfg: Full config.
        min_voxels: Cluster size floor (default `perception.min_voxels`; the lenient tier of the
            relaxed rules passes `perception.relaxed.large_min_voxels`).

    Returns:
        Candidates sorted by size (descending), then centroid (x, y, z).
    """
    lab = connected_components(coords, link=int(cfg["perception"].get("link_voxels", 1)))
    min_vox = int(cfg["perception"]["min_voxels"] if min_voxels is None else min_voxels)
    out = []
    for c in range(int(lab.max()) + 1 if len(lab) else 0):
        m = lab == c
        if m.sum() >= min_vox:
            out.append(Candidate(np.asarray(voxel_idx)[m], np.asarray(points)[m]))
    out.sort(key=lambda k: (-len(k.voxels), *np.round(k.centroid, 6).tolist()))
    return out


def is_large(c: Candidate, cfg: Mapping[str, Any]) -> bool:
    """Size rule of the relaxed candidate rules (CHANGE 1): enough voxels and, if set, a long enough AABB.

    Uses `perception.relaxed.large_min_voxels` and `large_min_extent` (0 = no extent rule).
    """
    r = cfg["perception"]["relaxed"]
    if len(c.voxels) < int(r["large_min_voxels"]):
        return False
    return float(np.max(c.aabb_max - c.aabb_min)) >= float(r["large_min_extent"])


@dataclass
class DoorRegion:
    """Swing region of one reference door: two vertical cylinders (one per possible hinge)."""

    oid: int
    hinges: np.ndarray  # (2, 2) xy of the two vertical edges
    radius: float
    zmin: float
    zmax: float

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """Bool mask of points inside either cylinder."""
        p = np.asarray(pts, float)
        inz = (p[:, 2] >= self.zmin) & (p[:, 2] <= self.zmax)
        d = np.linalg.norm(p[:, None, :2] - self.hinges[None], axis=2).min(axis=1)
        return inz & (d <= self.radius)


def door_regions(ref_ctx: RefContext, margin: float) -> list[DoorRegion]:
    """Swing regions of the reference doors (nyu40 8), from their AABBs.

    The door's longer horizontal AABB side is its width w; its two ends (at the middle of the
    short side) are the possible hinge edges; radius = w + `margin`.
    """
    out = []
    for oid, inst in sorted(ref_ctx.scan.instances.items()):
        if ref_ctx.object_nyu40(oid) != DOOR_NYU40:
            continue
        lo, hi = np.asarray(inst.aabb_min, float), np.asarray(inst.aabb_max, float)
        ext = hi - lo
        a = 0 if ext[0] >= ext[1] else 1  # major horizontal axis
        b = 1 - a
        mid = 0.5 * (lo[b] + hi[b])
        h = np.zeros((2, 2))
        h[0, a], h[1, a] = lo[a], hi[a]
        h[:, b] = mid
        out.append(DoorRegion(int(oid), h, float(ext[a] + margin), float(lo[2] - margin), float(hi[2] + margin)))
    return out


def drop_door_clusters(
    cands: Sequence[Candidate], regions: Sequence[DoorRegion], frac: float
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into (kept, dropped): dropped if >= `frac` of its voxels are in a region."""
    if not regions:
        return list(cands), []
    kept, dropped = [], []
    for c in cands:
        inside = np.zeros(len(c.points), bool)
        for r in regions:
            inside |= r.contains(c.points)
        (dropped if inside.mean() >= frac else kept).append(c)
    return kept, dropped
