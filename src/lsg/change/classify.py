"""Final per-node states at the end of the sequence (DESIGN §6.10): the ICP moved test.

- confirmed in place (§3) -> `unchanged` (no match needed);
- matched -> ICP gives T_move (motion of the object in the reference frame); the ICP-refined
  centroid displacement ‖T_move·c_R − c_R‖ > `tau_move` -> `moved`, else `unchanged`;
- unmatched -> `removed` with >= k_free counted free votes (§3), else `unknown`.

ICP direction: the observation is usually a *partial* view (oracle: vertices of the faces
hit in the processed frames; perception: residual voxels), so the partial observed points
(source) are registered onto the complete reference points (target). Every source point then
has a true correspondence. Init = the centroid offset; T_move is the inverse of the result.
Below `min_fitness` (too few correspondences) the plain centroid offset is used.

Method side: imports only `change.visibility`, `geom`, `graph`, `obs`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lsg.change.visibility import NodeVotes, has_free_evidence, old_place_class
from lsg.geom.transforms import apply
from lsg.graph.model import Node
from lsg.obs.base import ObsInstance


@dataclass
class Decision:
    """Final state of one reference node."""

    state: str
    T_move: np.ndarray | None = None  # 4x4, reference-frame motion (matched nodes)
    displacement: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


def _subsample(P: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(P) <= k:
        return P
    rng = np.random.default_rng(seed)
    return P[np.sort(rng.choice(len(P), size=k, replace=False))]


def icp_move(ref_pts: np.ndarray, obs_pts: np.ndarray, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigid motion T_move (reference pose -> observed pose) of one object.

    Args:
        ref_pts: (N, 3) reference points of the node (reference frame).
        obs_pts: (M, 3) observed points (reference frame), possibly partial.
        cfg: Full config (`classify`).

    Returns:
        (T_move 4x4, info with `fitness`, `rmse`, `fallback`).
    """
    import open3d as o3d

    c = cfg["classify"]
    R = _subsample(np.asarray(ref_pts, float), int(c["max_points"]), 0)
    O = _subsample(np.asarray(obs_pts, float), int(c["max_points"]), 1)
    t0 = O.mean(axis=0) - R.mean(axis=0)
    T_init_move = np.eye(4)
    T_init_move[:3, 3] = t0
    if len(R) < 3 or len(O) < 3:
        return T_init_move, {"fitness": 0.0, "rmse": None, "fallback": True}
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(O))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(R))
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(c["icp_max_iter"]))

    def run(T: np.ndarray) -> tuple[np.ndarray, float, float]:
        """ICP schedule from one init (obs -> reference pose); (T, fitness, rmse)."""
        res = None
        for mcd in c["icp_schedule"]:
            res = o3d.pipelines.registration.registration_icp(src, tgt, float(mcd), T, est, crit)
            T = np.asarray(res.transformation)
        assert res is not None
        return T, float(res.fitness), float(res.inlier_rmse)

    # two inits: the centroid offset (§6.10) and "no motion". A partial view biases the
    # centroid offset, and ICP from it can slide along planar faces into a local minimum;
    # for an unchanged object the identity init converges to zero motion. Best fitness wins,
    # near ties (< fit_tie) go to the lower inlier RMSE.
    hyps = [(*run(np.linalg.inv(T_init_move)), "centroid"), (*run(np.eye(4)), "identity")]
    tie = float(c.get("fit_tie", 0.02))
    best = hyps[0]
    for h in hyps[1:]:
        if h[1] > best[1] + tie or (abs(h[1] - best[1]) <= tie and h[2] < best[2]):
            best = h
    T, fit, rmse, init = best
    info = {"fitness": fit, "rmse": rmse, "init": init}
    if fit < float(c["min_fitness"]):
        return T_init_move, {**info, "fallback": True}
    return np.linalg.inv(T), {**info, "fallback": False}


def displacement(T_move: np.ndarray, c_ref: np.ndarray) -> float:
    """‖T_move·c_R − c_R‖."""
    c = np.asarray(c_ref, float)
    return float(np.linalg.norm(apply(T_move, c[None])[0] - c))


def classify(
    pairs: Sequence[tuple[Node, ObsInstance, float]],
    unmatched_ref: Sequence[Node],
    votes: Mapping[str, NodeVotes] | None,
    confirmed: Sequence[Node],
    ref_points: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    old_place: bool = False,
) -> dict[str, Decision]:
    """Final states of all scored reference nodes (module docstring).

    Args:
        pairs: Matched (node, observation, cost) from `association.associate`.
        unmatched_ref: Nodes without a match (not confirmed).
        votes: Online vote tallies (node id -> `NodeVotes`).
        confirmed: Nodes confirmed in place (skipped the matching).
        ref_points: Node id -> reference points (reference frame). A node without points
            falls back to its stored centroid (pure translation).
        cfg: Full config (`classify`, `states`).
        old_place: FULL with CHANGE 3 Part A on: a net-seen node needs displacement >
            `association.old_place.tau_move_seen` (if set) instead of `tau_move`.
    """
    tau = float(cfg["classify"]["tau_move"])
    t_seen = cfg["association"].get("old_place", {}).get("tau_move_seen") if old_place else None
    vs = votes or {}
    out: dict[str, Decision] = {}
    for n in confirmed:
        out[n.id] = Decision("unchanged")
    for n, o, cost in pairs:
        c_ref = np.asarray(n.centroid, float)
        P = ref_points.get(n.id)
        if P is not None and len(P):
            T, info = icp_move(P, o.points, cfg)
            c_ref = P.mean(axis=0)
        else:
            T = np.eye(4)
            T[:3, 3] = o.centroid - c_ref
            info = {"fallback": True, "fitness": None, "rmse": None}
        d = displacement(T, c_ref)
        info["cost"] = cost
        t = tau
        if t_seen is not None and old_place_class(vs.get(n.id), cfg) == "net_seen":
            t = max(tau, float(t_seen))
        out[n.id] = Decision("moved" if d > t else "unchanged", T, d, info)
    for n in unmatched_ref:
        out[n.id] = Decision("removed" if has_free_evidence(vs.get(n.id), cfg) else "unknown")
    return out
