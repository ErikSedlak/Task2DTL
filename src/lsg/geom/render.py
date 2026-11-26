"""Depth + label rendering, and reference-texture colour rendering, with Open3D RaycastingScene
(CPU; M6)."""

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
        self, K: np.ndarray, T_w2c: np.ndarray, W: int, H: int, return_prim: bool = False
    ) -> tuple[np.ndarray, ...]:
        """Render what a camera at pose `T_w2c` would see of the mesh.

        Args:
            K: 3x3 intrinsics for a W x H image (depth or color resolution).
            T_w2c: 4x4 world->camera transform, i.e. `invert(pose_c2w)`.
            W, H: Image size in pixels.

        Returns:
            depth: (H, W) float32 camera-z depth in metres, comparable to the sensor's depth;
                0 = no surface hit.
            label_map: (H, W) int64 face label of the first hit (objectId); 0 = no hit.
            prim: (H, W) int64 index of the first face hit, -1 = none (only if `return_prim`).
        """
        t, prim = self.cast(self.camera_rays(K, T_w2c, W, H))
        labels = np.where(prim >= 0, self.face_labels[np.maximum(prim, 0)], 0)
        if return_prim:
            return t, labels, prim
        return t, labels

    def depth_behind(
        self,
        K: np.ndarray,
        T_w2c: np.ndarray,
        W: int,
        H: int,
        depth: np.ndarray,
        labels: np.ndarray,
        active: np.ndarray,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """Depth of the nearest surface along each pixel's ray whose label differs from the first hit's.

        For "seen votes need contrast". All intersections of the active
        pixels' rays are listed (`RaycastingScene.list_intersections`); hits with the first
        hit's label are skipped (the node's own front / back / side faces), the nearest other
        hit at t >= first-hit depth - `eps` wins. A surface coincident with the node's back face
        (pillow against a backrest) is therefore found, which a restart-behind-the-hit cast
        would skip. Values are camera-z depth, like `render`.

        Args:
            K, T_w2c, W, H: The camera of the `render` call that produced `depth` / `labels`.
            depth: (H, W) first-hit depth from `render` (0 = no hit).
            labels: (H, W) first-hit labels from `render`.
            active: (H, W) bool, pixels to resolve (e.g. first hit on a voting node).
            eps: Tolerance [m] for other-label hits level with the first hit.

        Returns:
            (H, W) float64: the depth behind; `inf` where nothing else is hit, and for inactive
            or no-hit pixels.
        """
        out = np.full(H * W, np.inf)
        idx = np.flatnonzero(np.asarray(active, bool).ravel() & (np.asarray(depth).ravel() > 0))
        if idx.size == 0:
            return out.reshape(H, W)
        rays = self.camera_rays(K, T_w2c, W, H).reshape(-1, 6)[idx]
        ans = self.scene.list_intersections(o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))
        ray_ids = ans["ray_ids"].numpy().astype(np.int64)
        t = ans["t_hit"].numpy().astype(np.float64)
        lab = self.face_labels[ans["primitive_ids"].numpy().astype(np.int64)]
        lab0 = np.asarray(labels).ravel()[idx]
        t0 = np.asarray(depth, np.float64).ravel()[idx]
        ok = (lab != lab0[ray_ids]) & (t >= t0[ray_ids] - eps)
        best = np.full(idx.size, np.inf)
        np.minimum.at(best, ray_ids[ok], t[ok])
        out[idx] = best
        return out.reshape(H, W)

    def face_normal_z(self) -> np.ndarray:
        """(M,) |z| component of each face's unit normal (1 = horizontal surface, e.g. a table top)."""
        if getattr(self, "_nz", None) is None:
            v = self.vertices.astype(np.float64)
            f = self.faces.astype(np.int64)
            n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            ln = np.linalg.norm(n, axis=1)
            self._nz = np.where(ln > 0, np.abs(n[:, 2]) / np.maximum(ln, 1e-12), 0.0).astype(np.float32)
        return self._nz

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point to the closest spot on the mesh surface.

        Args:
            points: (N, 3) query points in world coordinates.

        Returns:
            (N,) float32 unsigned distances in metres.
        """
        q = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        return self.scene.compute_distance(q).numpy()


class TextureRenderer:
    """Renders the reference texture as seen from a camera (M6).

    Own `RaycastingScene` on the OBJ geometry (so it is correct even if OBJ and PLY differed;
    they are checked to be identical at M6). Per pixel: first face hit (`primitive_ids`),
    barycentric coordinates (`primitive_uvs`: hit = (1 - a - b)·c0 + a·c1 + b·c2), the
    interpolated corner UVs, then a bilinear texture lookup with v flipped (OBJ v = 0 is the
    image bottom).
    """

    def __init__(self, textured_mesh):
        """
        Args:
            textured_mesh: `lsg.io.scan3r.TexturedMesh` (vertices, faces, corner_uv, texture).
        """
        tm = textured_mesh
        self.r = Renderer(tm.vertices, tm.faces)
        self.corner_uv = np.asarray(tm.corner_uv, np.float32)
        self.texture = np.asarray(tm.texture, np.uint8)

    def render_color(
        self, K: np.ndarray, T_w2c: np.ndarray, W: int, H: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected colour image at a camera.

        Args:
            K: 3x3 intrinsics for a W x H image (e.g. colour intrinsics scaled to W x H).
            T_w2c: 4x4 world->camera transform.
            W, H: Image size.

        Returns:
            rgb: (H, W, 3) uint8, black where nothing is hit.
            hit: (H, W) bool.
        """
        rays = self.r.camera_rays(K, T_w2c, W, H)
        ans = self.r.scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))
        t = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        bary = ans["primitive_uvs"].numpy().astype(np.float32)
        hit = np.isfinite(t) & (prim != int(o3d.t.geometry.RaycastingScene.INVALID_ID))
        out = np.zeros((H, W, 3), np.uint8)
        if not hit.any():
            return out, hit
        p = prim[hit]
        a, b = bary[hit, 0:1], bary[hit, 1:2]
        c = self.corner_uv[p]  # (n, 3, 2)
        uv = (1.0 - a - b) * c[:, 0] + a * c[:, 1] + b * c[:, 2]
        out[hit] = sample_bilinear(self.texture, uv)
        return out, hit


def sample_bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup. `uv` (n, 2) in [0, 1], v = 0 at the bottom row; returns (n, 3) uint8."""
    Ht, Wt = tex.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (Wt - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (Ht - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, Wt - 1), np.minimum(y0 + 1, Ht - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    def g(yy, xx):
        return tex[yy, xx].astype(np.float32)  # gather first, convert only the samples

    top = g(y0, x0) * (1 - fx) + g(y0, x1) * fx
    bot = g(y1, x0) * (1 - fx) + g(y1, x1) * fx
    return np.clip(np.rint(top * (1 - fy) + bot * fy), 0, 255).astype(np.uint8)
