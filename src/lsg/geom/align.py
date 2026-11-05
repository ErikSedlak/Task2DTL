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
    """Point cloud of a rescan built from its depth frames (no labels involved).

    Args:
        seq: The rescan's sequence.
        frame_ids: Frames to use; frames with a non-finite pose are skipped.
        voxel: Voxel size in metres; keeps one point per voxel to thin out the cloud.
        max_depth: Depth beyond this (metres) is ignored, because far depth is noisy.

    Returns:
        (N, 3) points in the rescan's own mesh frame.
    """
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
    """ICP target: the reference mesh's vertices with their normals.

    Using the vertices (instead of random surface samples) keeps runs deterministic.
    Normals are needed for point-to-plane ICP.
    """
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
    """Nudge an approximate rescan->reference transform so the two scans overlap better.

    Runs robust point-to-plane ICP in stages with shrinking correspondence distance
    (`cfg["icp_schedule"]`). The Tukey loss ignores points far from the reference surface,
    so moved objects do not pull the result; the unchanged room dominates.

    Args:
        points_rescan: (N, 3) rescan points in the rescan frame (see `depth_points`).
        target: Reference point cloud with normals (see `mesh_target`).
        T_init: Starting 4x4 rescan->reference transform (3RScan's `T_align`).
        cfg: The `alignment` block of the config.

    Returns:
        T: Refined 4x4 transform, or `T_init` unchanged if the correction exceeds
            `max_delta_t` / `max_delta_rot_deg` (treated as ICP failure).
        info: `{"n_source", "delta_t_m", "delta_rot_deg", "fitness", "inlier_rmse",
            "accepted"}` describing the correction.
    """
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
    """One-call alignment refinement for a rescan: build its point cloud, then run ICP.

    Args:
        seq: The rescan's sequence (depth + poses only).
        ref_vertices, ref_faces: Reference mesh.
        T_init: 3RScan's `T_align` for this rescan.
        cfg: The `alignment` config block (uses `frame_step`, `voxel`, `max_depth` and the
            ICP settings).

    Returns:
        Same as `refine_alignment`.
    """
    ids = seq.frame_ids[:: int(cfg["frame_step"])]
    pts = depth_points(seq, ids, float(cfg["voxel"]), float(cfg["max_depth"]))
    return refine_alignment(pts, mesh_target(ref_vertices, ref_faces), T_init, cfg)
