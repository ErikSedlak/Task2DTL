"""Depth + label rendering with Open3D RaycastingScene (CPU; DECISIONS 2026-11-05)."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from lsg.geom.transforms import pixel_grid


class Renderer:
    """Draws a triangle mesh as seen from a camera: a depth image and an object-id image.

    Uses ray casting on the CPU (one ray per pixel, first triangle hit wins). Building the
    acceleration structure happens once in the constructor; each `render` call is ~5 ms at
    224x172. `face_labels[k]` is reported for pixels whose first hit is face k.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_labels: np.ndarray | None = None):
        """
        Args:
            vertices: (N, 3) mesh vertices in world coordinates (metres).
            faces: (M, 3) vertex indices per triangle.
            face_labels: (M,) integer label per face (usually objectId). Defaults to all 1.
        """
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
        """One ray per pixel, in the world frame.

        The camera-frame direction is [(u-cx)/fx, (v-cy)/fy, 1] (not normalised), so `t_hit`
        is camera-z depth, the quantity stored in the depth images.

        Args:
            K: 3x3 intrinsics for a W x H image.
            T_w2c: 4x4 world->camera transform (the inverse of a camera pose).
            W, H: Image size in pixels.

        Returns:
            (H, W, 6) float32 array [origin xyz, direction xyz].
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
        """Cast arbitrary rays against the mesh.

        Args:
            rays: (..., 6) array [origin xyz, direction xyz].

        Returns:
            t_hit: (...,) float32 distance along the ray in units of the direction length;
                0 where nothing was hit.
            face: (...,) int64 index of the first face hit; -1 where nothing was hit.
        """
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
        """Render what a camera at pose `T_w2c` would see of the mesh.

        Args:
            K: 3x3 intrinsics for a W x H image (depth or color resolution).
            T_w2c: 4x4 world->camera transform, i.e. `invert(pose_c2w)`.
            W, H: Image size in pixels.

        Returns:
            depth: (H, W) float32 camera-z depth in metres, comparable to the sensor's depth;
                0 = no surface hit.
            label_map: (H, W) int64 face label of the first hit (objectId); 0 = no hit.
        """
        t, prim = self.cast(self.camera_rays(K, T_w2c, W, H))
        labels = np.where(prim >= 0, self.face_labels[np.maximum(prim, 0)], 0)
        return t, labels

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point to the closest spot on the mesh surface.

        Args:
            points: (N, 3) query points in world coordinates.

        Returns:
            (N,) float32 unsigned distances in metres.
        """
        q = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        return self.scene.compute_distance(q).numpy()
