"""Depth residual points (DESIGN §3 candidate steps 1-2).

Step 1: nearer pixels = valid observed depth in front of the rendered reference surface by
more than `tau_new`. Step 2: back-project them into the reference frame, drop points within
`surf_dist` of the reference mesh and points within `struct_dist` of reference structure
surfaces (walls, ceiling, doors, windows, ...). Floor and floor mat are not in the structure
set, so objects standing on the floor survive; their bottom slice is already within
`surf_dist` of the floor.

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

    def keep_struct(self, pts: np.ndarray) -> np.ndarray:
        """Bool mask: points farther than `struct_dist` from structure (the surface test is skipped;
        appearance thin-object points lie on the reference surface by construction, M6)."""
        ok = np.ones(len(pts), bool)
        if self.struct is not None and len(pts):
            ok = self.struct.distance(pts) > self.struct_dist
        return ok

    def keep(self, pts: np.ndarray) -> np.ndarray:
        """Bool mask: points farther than `surf_dist` from the mesh and `struct_dist` from structure."""
        if len(pts) == 0:
            return np.zeros(0, bool)
        ok = self.ref.distance(pts) > self.surf_dist
        if self.struct is not None and ok.any():
            ok[ok] = self.struct.distance(pts[ok]) > self.struct_dist
        return ok


def nearer_mask(depth: np.ndarray, d_exp: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Step 1: valid observed depth (<= `max_depth`) nearer than the rendered surface by > `tau_new`.

    Pixels without a reference hit (`d_exp == 0`) are excluded: there is no expected surface
    to be in front of.
    """
    pc = cfg["perception"]
    d_obs = np.asarray(depth, np.float64)
    d_e = np.asarray(d_exp, np.float64)
    return (d_obs > 0) & (d_obs <= float(pc["max_depth"])) & (d_e > 0) & (d_obs < d_e - float(pc["tau_new"]))


def nearer_points(
    depth: np.ndarray,
    d_exp: np.ndarray,
    K: np.ndarray,
    pose_ref: np.ndarray,
    filt: SurfaceFilter | None,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1-2 for one frame.

    Args:
        depth: (H, W) observed depth [m], 0 = invalid (stored orientation).
        d_exp: (H, W) reference depth rendered at the aligned pose [m], 0 = no hit.
        K: Depth intrinsics.
        pose_ref: Camera -> reference frame (`T_align @ pose_c2w`).
        filt: Surface / structure filter (None = step 1 only).
        cfg: Full config.

    Returns:
        (points (N, 3) in the reference frame, flat pixel indices (N,) into the (H, W) image).
    """
    m = nearer_mask(depth, d_exp, cfg)
    pix = np.flatnonzero(m)
    if pix.size == 0:
        return np.zeros((0, 3)), pix
    W = depth.shape[1]
    v, u = np.divmod(pix, W)
    z = np.asarray(depth, np.float64).ravel()[pix]
    pts_c = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts = apply(pose_ref, pts_c)
    if filt is not None:
        ok = filt.keep(pts)
        pts, pix = pts[ok], pix[ok]
    return pts, pix
