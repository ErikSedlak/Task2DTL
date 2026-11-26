"""Per-frame image segments and co-segmentation clustering of candidate voxels (CHANGE 3 Part N, N1).

Connected components over spatially adjacent voxels (`cluster.cluster`) chain touching objects into
one "blob" (dev_tune: 24 candidates held 2-10 GT new objects each). Here two adjacent kept voxels are
linked only if the frames that see both put them in the same **image segment** at least as often as
in different ones; pairs never seen together keep today's spatial link.

Segment sources (`perception.segment.method`):
- `geometric`: per frame, the residual pixels form a 4-neighbour graph whose edges are cut where the
  depth jumps by more than `depth_edge_abs + depth_edge_rel * z` (`segment_frame`);
- `none`: no co-segmentation (today's `cluster.cluster`).

(A third source, cached SAM masks, lost the dev_tune selection and was removed.)
Segment id -1 = unknown (crease pixel); unknown pixels give no vote. Method side: uses only the
rescan's depth frames (hard rules 1-2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree

from lsg.perception.cluster import Candidate

METHODS = ("none", "geometric")


def seg_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The `perception.segment` block (empty = method none)."""
    return cfg["perception"].get("segment") or {}


def seg_method(cfg: Mapping[str, Any]) -> str:
    """Configured segment source; raises on an unknown name."""
    m = str(seg_cfg(cfg).get("method", "none"))
    if m not in METHODS:
        raise ValueError(f"perception.segment.method {m!r} not in {METHODS}")
    return m


def normals_from_depth(depth: np.ndarray, K: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points and unit normals (towards the camera) from central differences `step` px apart.

    Returns:
        (P (H, W, 3), N (H, W, 3)); N is NaN where a neighbour is invalid or the depth jumps.
    """
    d = np.asarray(depth, np.float64)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], axis=-1)
    N = np.full((H, W, 3), np.nan)
    s = int(step)
    c = (slice(s, H - s), slice(s, W - s))
    dx = P[s:H - s, 2 * s:] - P[s:H - s, : W - 2 * s]
    dy = P[2 * s:, s:W - s] - P[: H - 2 * s, s:W - s]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    dc = d[c]
    ok = (dc > 0) & (d[s:H - s, 2 * s:] > 0) & (d[s:H - s, : W - 2 * s] > 0) & (d[2 * s:, s:W - s] > 0) & (d[: H - 2 * s, s:W - s] > 0)
    ok &= (np.abs(dx[..., 2]) < 0.1 * dc) & (np.abs(dy[..., 2]) < 0.1 * dc) & (ln[..., 0] > 0)
    n = np.where(ok[..., None], n / np.maximum(ln, 1e-12), np.nan)
    flip = np.sum(n * P[c], axis=-1) > 0  # orient towards the camera
    n[flip] *= -1
    N[c] = n
    return P, N


def concave_band(P: np.ndarray, N: np.ndarray, deg: float, step: int = 2) -> np.ndarray:
    """(H, W) bool: pixels on a concave crease (normals of a neighbour `step` px away differ by > `deg`
    and the surfaces meet concavely, (n_p - n_q) . (x_p - x_q) < 0; local-convexity rule)."""
    H, W = N.shape[:2]
    band = np.zeros((H, W), bool)
    cos_t = np.cos(np.radians(float(deg)))
    s = int(step)
    for dv, du in ((0, s), (s, 0), (s, s), (s, -s)):
        u0, u1 = max(0, -du), W - max(0, du)  # pixel p = (v, u), neighbour q = (v + dv, u + du)
        a = (slice(0, H - dv), slice(u0, u1))
        b = (slice(dv, H), slice(u0 + du, u1 + du))
        na, nb = N[a], N[b]
        cosang = np.sum(na * nb, axis=-1)
        conc = np.sum((na - nb) * (P[a] - P[b]), axis=-1) < 0
        hit = np.nan_to_num(cosang, nan=1.0) < cos_t
        hit &= conc
        band[a] |= hit
        band[b] |= hit
    return band


def segment_frame(depth: np.ndarray, pix: np.ndarray, cfg: Mapping[str, Any], K: np.ndarray | None = None) -> np.ndarray:
    """Geometric segments of one frame's residual pixels.

    Edges of the 4-neighbour graph over the residual pixels are cut at depth jumps
    (> `depth_edge_abs` + `depth_edge_rel` · z). With intrinsics `K` and `concave_deg` set, pixels on a
    concave crease (`concave_band`: e.g. where a box's top meets the side of a taller box behind it,
    which has no depth jump) leave the graph and get segment -1 (unknown).

    Args:
        depth: (H, W) observed depth [m] (stored orientation).
        pix: (N,) flat pixel indices of the residual pixels.
        cfg: Full config (`perception.segment`).
        K: 3x3 depth intrinsics (None = depth jumps only).

    Returns:
        (N,) int64 segment id per pixel (connected components, numbered from 0; -1 = crease pixel).
    """
    pix = np.asarray(pix, np.int64)
    n = len(pix)
    if n == 0:
        return np.zeros(0, np.int64)
    sc = seg_cfg(cfg)
    a_abs, a_rel = float(sc.get("depth_edge_abs", 0.03)), float(sc.get("depth_edge_rel", 0.02))
    H, W = depth.shape
    d = np.asarray(depth, np.float64).ravel()
    crease = np.zeros(n, bool)
    deg = sc.get("concave_deg")
    if K is not None and deg is not None:
        P, Nrm = normals_from_depth(depth, K, int(sc.get("normal_step", 2)))
        crease = concave_band(P, Nrm, float(deg), int(sc.get("normal_step", 2))).ravel()[pix]
    pos = np.full(H * W, -1, np.int64)
    pos[pix[~crease]] = np.flatnonzero(~crease)
    rows, cols = [], []
    for step, ok in ((1, ((pix % W) < W - 1) & ~crease), (W, (pix < (H - 1) * W) & ~crease)):
        q = pix[ok] + step
        j = pos[q]
        has = j >= 0
        i_src, j_dst = np.flatnonzero(ok)[has], j[has]
        z1, z2 = d[pix[i_src]], d[pix[j_dst]]
        keep = np.abs(z1 - z2) <= a_abs + a_rel * np.minimum(z1, z2)
        rows.append(i_src[keep])
        cols.append(j_dst[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    seg = csgraph.connected_components(g, directed=False)[1].astype(np.int64)
    seg[crease] = -1
    return seg


def voxel_segments(vidx: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority segment per voxel of one frame (ties -> lowest segment id); unknown pixels ignored.

    Returns:
        (voxel ids, their segment ids), voxel ids sorted ascending.
    """
    vidx, seg = np.asarray(vidx, np.int64), np.asarray(seg, np.int64)
    ok = seg >= 0
    vidx, seg = vidx[ok], seg[ok]
    if not len(vidx):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs, cnt = np.unique(np.stack([vidx, seg], axis=1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -cnt, pairs[:, 0]))  # by voxel, then count desc, then seg asc
    p = pairs[order]
    first = np.ones(len(p), bool)
    first[1:] = p[1:, 0] != p[:-1, 0]
    return p[first, 0], p[first, 1]


def cosegment_cluster(
    voxel_idx: np.ndarray,
    coords: np.ndarray,
    points: np.ndarray,
    frame_segs: Sequence[tuple[np.ndarray, np.ndarray]],
    n_grid: int,
    min_voxels: int,
    link: int,
    cut_frac: float = 0.5,
    bridge_unseen: int | None = None,
) -> tuple[list[Candidate], dict[str, int]]:
    """Clusters of kept voxels linked by space and agreeing image segments.

    Args:
        voxel_idx: (N,) kept voxel indices into the grid.
        coords: (N, 3) their integer coordinates.
        points: (N, 3) their mean residual points.
        frame_segs: Per processed frame, `voxel_segments` output (voxel ids, segment ids).
        n_grid: Number of voxels in the grid (index range).
        min_voxels: Cluster size floor.
        link: Spatial linking distance in voxels (Chebyshev), as `cluster.cluster`.
        cut_frac: A co-observed pair is cut if more than this share of the frames seeing both put
            them in different segments (0 = any disagreement cuts; 0.5 = majority).
        bridge_unseen: Pairs never seen together link only up to this Chebyshev distance (None =
            `link`; 0 = never). Touching objects share voxels at the contact, and with
            `link` 2 such pairs bridge them (synthetic two-box scene: 0 + any-disagreement splits).

    Returns:
        (candidates sorted by size desc, then centroid - as `cluster.cluster`; stats
        {"edges", "cut_edges"}).
    """
    vi = np.asarray(voxel_idx, np.int64)
    N = len(vi)
    if N == 0:
        return [], {"edges": 0, "cut_edges": 0}
    E = cKDTree(np.asarray(coords, float)).query_pairs(r=float(link) + 1e-9, p=np.inf, output_type="ndarray")
    E = np.asarray(E, np.int64).reshape(-1, 2)
    pos = np.full(int(n_grid), -1, np.int64)
    pos[vi] = np.arange(N)
    same = np.zeros(len(E), np.int32)
    diff = np.zeros(len(E), np.int32)
    seg_at = np.full(N, -1, np.int64)
    for vox, seg in frame_segs:
        if not len(vox):
            continue
        p = pos[np.asarray(vox, np.int64)]
        keep = p >= 0
        if not keep.any():
            continue
        seg_at.fill(-1)
        seg_at[p[keep]] = np.asarray(seg, np.int64)[keep]
        a, b = seg_at[E[:, 0]], seg_at[E[:, 1]]
        both = (a >= 0) & (b >= 0)
        same += both & (a == b)
        diff += both & (a != b)
    tot = same + diff
    bu = int(link) if bridge_unseen is None else int(bridge_unseen)
    cheb = np.abs(np.asarray(coords)[E[:, 0]] - np.asarray(coords)[E[:, 1]]).max(axis=1) if len(E) else np.zeros(0)
    linked = np.where(tot > 0, diff <= float(cut_frac) * tot, cheb <= bu)
    Ek = E[linked]
    g = coo_matrix((np.ones(len(Ek), np.int8), (Ek[:, 0], Ek[:, 1])), shape=(N, N))
    lab = csgraph.connected_components(g, directed=False)[1]
    out = []
    P = np.asarray(points)
    for c in range(int(lab.max()) + 1):
        m = lab == c
        if m.sum() >= int(min_voxels):
            out.append(Candidate(vi[m], P[m]))
    out.sort(key=lambda k: (-len(k.voxels), *np.round(k.centroid, 6).tolist()))
    return out, {"edges": int(len(E)), "cut_edges": int((~linked).sum())}
