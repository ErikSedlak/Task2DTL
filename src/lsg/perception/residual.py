"""Depth residual points (DESIGN §3 candidate steps 1-2).

Step 1: nearer pixels = valid observed depth in front of the rendered reference surface by
more than `tau_new`. Step 2: back-project them into the reference frame, drop points within
`surf_dist` of the reference mesh and points within `struct_dist` of reference structure
surfaces (walls, ceiling, doors, windows, ...). Floor and floor mat are not in the structure
set, so objects standing on the floor survive; their bottom slice is already within
`surf_dist` of the floor.

Noise rejection (CHANGE 3 Part N, N5; `perception.noise`, each switch off by default until the dev_tune
sweep decides): `edge` drops residual pixels next to a depth jump (flying pixels between foreground and
background); `dscale` grows the `surf_dist` test with depth (sensor noise grows with range); `planes`
extends reference walls / ceiling / doors (`struct_dist`) and the floor (`surf_dist`) as fitted planes
over their extent + `margin`, so holes in the reference mesh stop producing residuals.

Uses only reference data (mesh, reference labels via `RefContext`) and the rescan's depth +
pose. Method side: no `oracle` / `eval` imports, no rescan annotations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.geom.render import Renderer
from lsg.geom.transforms import apply
from lsg.obs.base import RefContext


class SurfaceFilter:
    """Distance tests against the reference mesh and its structure surfaces (step 2)."""

    def __init__(self, ref_ctx: RefContext, cfg: Mapping[str, Any], renderer: Renderer | None = None):
        """
        Args:
            ref_ctx: Reference scan + per-object nyu40.
            cfg: Full config; uses `cfg["perception"]` (`surf_dist`, `struct_dist`,
                `struct_exclude_nyu40`).
            renderer: Prebuilt renderer of the full reference mesh (built here if None).
        """
        pc = cfg["perception"]
        self.surf_dist = float(pc["surf_dist"])
        self.struct_dist = float(pc["struct_dist"])
        scan = ref_ctx.scan
        self.ref = renderer or Renderer(scan.vertices, scan.faces)
        excl = {int(x) for x in pc.get("struct_exclude_nyu40", [])}
        self.struct_oids = sorted(
            oid for oid in scan.instances
            if (n := ref_ctx.object_nyu40(oid)) is not None and node_level(n) == "structure" and n not in excl
        )  # fmt: skip
        V, F = ref_ctx.faces_of(self.struct_oids)
        self.struct = Renderer(V, F) if len(F) else None
        nc = pc.get("noise") or {}
        ds = nc.get("dscale") or {}
        self.surf_rel = float(ds.get("surf_rel", 0.0)) if ds.get("enabled", False) else 0.0
        pl = nc.get("planes") or {}
        self.planes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        if pl.get("enabled", False):
            floors = sorted(oid for oid in scan.instances if ref_ctx.object_nyu40(oid) == 2)
            for oids, thr in ((self.struct_oids, self.struct_dist), (floors, self.surf_dist)):
                for oid in oids:
                    p = fit_plane(ref_ctx, oid, int(pl.get("min_vertices", 200)), float(pl.get("max_rms", 0.03)),
                                  float(pl.get("margin", 0.30)))  # fmt: skip
                    if p is not None:
                        self.planes.append((*p, thr))

    def keep_struct(self, pts: np.ndarray) -> np.ndarray:
        """Bool mask: points farther than `struct_dist` from structure (the surface test is skipped;
        appearance thin-object points lie on the reference surface by construction, M6)."""
        ok = np.ones(len(pts), bool)
        if self.struct is not None and len(pts):
            ok = self.struct.distance(pts) > self.struct_dist
        return ok

    def keep(self, pts: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        """Bool mask: points farther than `surf_dist` (+ `surf_rel` · z with `noise.dscale`) from the mesh,
        `struct_dist` from structure, and (with `noise.planes`) off the extended structure / floor planes."""
        if len(pts) == 0:
            return np.zeros(0, bool)
        thr = self.surf_dist + (self.surf_rel * np.asarray(z, float) if (self.surf_rel and z is not None) else 0.0)
        ok = self.ref.distance(pts) > thr
        if self.struct is not None and ok.any():
            ok[ok] = self.struct.distance(pts[ok]) > self.struct_dist
        if self.planes and ok.any():
            ok[ok] = ~on_planes(pts[ok], self.planes)
        return ok


def fit_plane(ref_ctx: RefContext, oid: int, min_vertices: int, max_rms: float, margin: float):
    """Plane of one reference instance (PCA of its mesh vertices) with its in-plane extent + `margin`.

    Returns:
        (centroid, normal, in-plane axes (2, 3), lower (2,), upper (2,)) or None if the instance has
        fewer than `min_vertices` vertices or is not planar (rms distance to the plane > `max_rms`).
    """
    V, F = ref_ctx.faces_of([oid])
    if not len(F):
        return None
    P = V[np.unique(F.ravel())]
    if len(P) < min_vertices:
        return None
    c = P.mean(axis=0)
    w, U = np.linalg.eigh(np.cov((P - c).T))
    if np.sqrt(max(w[0], 0.0)) > max_rms:
        return None
    n, A = U[:, 0], U[:, 1:].T
    uv = (P - c) @ A.T
    return c, n, A, uv.min(axis=0) - margin, uv.max(axis=0) + margin


def on_planes(pts: np.ndarray, planes) -> np.ndarray:
    """Bool mask: points within a plane's distance threshold and inside its extended extent."""
    hit = np.zeros(len(pts), bool)
    for c, n, A, lo, hi, thr in planes:
        d = np.abs((pts - c) @ n)
        uv = (pts - c) @ A.T
        hit |= (d <= thr) & np.all((uv >= lo) & (uv <= hi), axis=1)
    return hit


def depth_edges(depth: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(H, W) bool: valid pixels with a valid 4-neighbour more than `edge.abs` + `edge.rel` · z away in depth."""
    ec = ((cfg["perception"].get("noise") or {}).get("edge")) or {}
    d = np.asarray(depth, np.float64)
    a, r = float(ec.get("abs", 0.05)), float(ec.get("rel", 0.03))
    out = np.zeros(d.shape, bool)
    for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):  # fmt: skip
        x, y = d[sl_a], d[sl_b]
        jump = (x > 0) & (y > 0) & (np.abs(x - y) > a + r * np.minimum(x, y))
        out[sl_a] |= jump
        out[sl_b] |= jump
    return out


def nearer_mask(depth: np.ndarray, d_exp: np.ndarray, cfg: Mapping[str, Any], tau: float | None = None) -> np.ndarray:
    """Step 1: valid observed depth (<= `max_depth`) nearer than the rendered surface by > `tau`.

    `tau` defaults to `perception.tau_new`; the lenient tier of the relaxed candidate rules passes
    `perception.relaxed.tau_new_large`. Pixels without a reference hit (`d_exp == 0`) are excluded:
    there is no expected surface to be in front of.
    """
    pc = cfg["perception"]
    tau = float(pc["tau_new"] if tau is None else tau)
    d_obs = np.asarray(depth, np.float64)
    d_e = np.asarray(d_exp, np.float64)
    m = (d_obs > 0) & (d_obs <= float(pc["max_depth"])) & (d_e > 0) & (d_obs < d_e - tau)
    if ((pc.get("noise") or {}).get("edge") or {}).get("enabled", False):
        m &= ~depth_edges(d_obs, cfg)  # CHANGE 3 Part N: flying pixels at depth jumps
    return m


def nearer_points(
    depth: np.ndarray,
    d_exp: np.ndarray,
    K: np.ndarray,
    pose_ref: np.ndarray,
    filt: SurfaceFilter | None,
    cfg: Mapping[str, Any],
    tau: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1-2 for one frame.

    Args:
        depth: (H, W) observed depth [m], 0 = invalid (stored orientation).
        d_exp: (H, W) reference depth rendered at the aligned pose [m], 0 = no hit.
        K: Depth intrinsics.
        pose_ref: Camera -> reference frame (`T_align @ pose_c2w`).
        filt: Surface / structure filter (None = step 1 only).
        cfg: Full config.
        tau: Nearer threshold [m] (default `perception.tau_new`).

    Returns:
        (points (N, 3) in the reference frame, flat pixel indices (N,) into the (H, W) image).
    """
    m = nearer_mask(depth, d_exp, cfg, tau)
    pix = np.flatnonzero(m)
    if pix.size == 0:
        return np.zeros((0, 3)), pix
    W = depth.shape[1]
    v, u = np.divmod(pix, W)
    z = np.asarray(depth, np.float64).ravel()[pix]
    pts_c = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts = apply(pose_ref, pts_c)
    if filt is not None:
        ok = filt.keep(pts, z)
        pts, pix = pts[ok], pix[ok]
    return pts, pix
