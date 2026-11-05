"""Depth + label rendering with Open3D RaycastingScene (CPU; DECISIONS 2026-11-05)."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from lsg.geom.transforms import pixel_grid


class Renderer:
    """Ray-casts a triangle mesh. `face_labels[k]` is reported for pixels whose first hit is face k."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_labels: np.ndarray | None = None):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        n = len(self.faces)
        self.face_labels = (
            np.asarray(face_labels, dtype=np.int64) if face_labels is not None else np.ones(n, np.int64)
        )
        if len(self.face_labels) != n:
            raise ValueError("face_labels must have one entry per face")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.faces))

    @staticmethod
    def camera_rays(K: np.ndarray, T_w2c: np.ndarray, W: int, H: int) -> np.ndarray:
        """(H, W, 6) float32 rays [origin, direction] in world frame.

        The camera-frame direction is [(u-cx)/fx, (v-cy)/fy, 1] (not normalised), so `t_hit`
        is camera-z depth, the quantity stored in the depth images.
        """
        u, v = pixel_grid(W, H)
        d_cam = np.stack(
            [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
        )
        R_c2w = T_w2c[:3, :3].T
        origin = -R_c2w @ T_w2c[:3, 3]
        d_world = d_cam @ R_c2w.T
        o = np.broadcast_to(origin, d_world.shape)
        return np.concatenate([o, d_world], axis=-1).astype(np.float32)

    def cast(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rays (..., 6) -> (t_hit (...,) float32 with 0 = no hit, face index (...,) int64 with -1)."""
        ans = self.scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))
        t = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        miss = ~np.isfinite(t) | (prim == int(o3d.t.geometry.RaycastingScene.INVALID_ID))
        t = np.where(miss, 0.0, t).astype(np.float32)
        prim = np.where(miss, -1, prim)
        return t, prim

    def render(
        self, K: np.ndarray, T_w2c: np.ndarray, W: int, H: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """-> depth (H, W) float32 metres (0 = no hit), label_map (H, W) int64 (0 = no hit)."""
        t, prim = self.cast(self.camera_rays(K, T_w2c, W, H))
        labels = np.where(prim >= 0, self.face_labels[np.maximum(prim, 0)], 0)
        return t, labels

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Unsigned point-to-surface distance (N,) for (N, 3) points."""
        q = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        return self.scene.compute_distance(q).numpy()
