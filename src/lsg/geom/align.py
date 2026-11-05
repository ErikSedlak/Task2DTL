"""Refinement of 3RScan's rescan->reference alignment (DECISIONS 2026-11-05/M0, DESIGN risk #4).

Method side: uses only rescan depth + poses and the reference mesh. No rescan labels.
Robust point-to-plane ICP (Tukey loss, coarse-to-fine) from `T_align`. Moved objects are outliers;
the static room dominates. A correction larger than the configured limits is rejected and the
raw `T_align` is kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import open3d as o3d

from lsg.geom.transforms import backproject, invert, rotation_angle_deg
from lsg.io.scan3r import SequenceReader


def depth_points(
    seq: SequenceReader, frame_ids: Iterable[int], voxel: float, max_depth: float
) -> np.ndarray:
    """Back-projected rescan depth (rescan mesh frame), voxel-downsampled. (N, 3)."""
    K = seq.info.K_depth
    chunks = []
    for i in frame_ids:
        T = seq.pose(i)
        if not np.all(np.isfinite(T)):
            continue
        d = seq.depth(i)
        d = np.where(d <= max_depth, d, 0.0)
        chunks.append(backproject(d, K, T))
    pts = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pc.voxel_down_sample(voxel).points)


def mesh_target(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.PointCloud:
    """Reference mesh vertices with vertex normals (deterministic ICP target)."""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    tgt = o3d.geometry.PointCloud(mesh.vertices)
    tgt.normals = mesh.vertex_normals
    return tgt


def refine_alignment(
    points_rescan: np.ndarray,
    target: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """-> (T_refined rescan->reference, info). cfg = the `alignment` config block."""
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_rescan))
    T = np.asarray(T_init, dtype=np.float64).copy()
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane
    fitness = rmse = None
    for mcd in cfg["icp_schedule"]:
        res = o3d.pipelines.registration.registration_icp(
            src,
            target,
            float(mcd),
            T,
            est(o3d.pipelines.registration.TukeyLoss(k=float(mcd))),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg["max_iter"])),
        )
        T = np.asarray(res.transformation)
        fitness, rmse = float(res.fitness), float(res.inlier_rmse)
    dT = T @ invert(T_init)
    info = {
        "n_source": int(len(points_rescan)),
        "delta_t_m": float(np.linalg.norm(dT[:3, 3])),
        "delta_rot_deg": rotation_angle_deg(dT),
        "fitness": fitness,
        "inlier_rmse": rmse,
    }
    ok = info["delta_t_m"] <= cfg["max_delta_t"] and info["delta_rot_deg"] <= cfg["max_delta_rot_deg"]
    info["accepted"] = bool(ok)
    return (T if ok else np.asarray(T_init, dtype=np.float64).copy()), info


def refine_from_sequence(
    seq: SequenceReader,
    ref_vertices: np.ndarray,
    ref_faces: np.ndarray,
    T_init: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience: build source from every `frame_step`-th frame and refine."""
    ids = seq.frame_ids[:: int(cfg["frame_step"])]
    pts = depth_points(seq, ids, float(cfg["voxel"]), float(cfg["max_depth"]))
    return refine_alignment(pts, mesh_target(ref_vertices, ref_faces), T_init, cfg)
