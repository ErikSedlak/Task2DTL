"""Association of reference nodes with observed instances (DESIGN §6.9, B2).

Two stages, both at the end of the sequence (deferred commitment):

0. **Absorption** (M5, Erik): a node confirmed in place (§3) does not enter the matching, but
   it claims the observations that belong to it, so that its oracle instance (or a perception
   candidate from a mesh hole / moving foliage inside it) does not become a false `new` node or
   a false `moved` match for another same-label node. An observation is absorbed by a confirmed
   node if its label is compatible and >= `absorb.min_frac` of its points lie inside the
   node's AABB + `absorb.margin`. Many-to-one: each such observation goes to its best node
   (highest point share, then centroid distance, then node id); each node keeps the best of its
   absorbed observations as `obs_token`.
1. **Hungarian matching** of the remaining reference nodes and observations on
   cost = label term + w_g * size term + w_p * position term. Pairs with cost >= `tau_assoc`
   (per mode) stay unmatched.

Label term: oracle = hard gate (0 if labels are equal, `nyu_penalty` if only nyu40 agrees and
the class is not a catch-all in `nyu_fallback_exclude` (39 otherfurniture, 40 otherprop; M5),
else +inf); perception = soft, -log max(p(label_ref), p_min) from the candidate's `label_topk`.

Old-place gate (CHANGE 3 Part A, FULL only, `association.old_place`): a move claims that the object
left its reference place, so it needs evidence there, as a removal does. A node whose place is
net-seen (counted seen > free + free-like) may only take an observation inside its own place
(the absorption geometry), so its own instance can still match (-> ICP -> usually unchanged) but a
distant same-label observation cannot. Optionally the same for nodes without any counted vote
(`unobserved_inside_only`), or a distance limit for them (`max_dist_unseen`).

Method side: imports only `change.visibility`, `graph`, `obs` (no-leak test).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from lsg.change.visibility import NodeVotes, old_place_class
from lsg.graph.model import Node
from lsg.obs.base import ObsInstance

INF = float("inf")


def _acfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return cfg["association"]


def label_p(node: Node, o: ObsInstance) -> float:
    """p(node label) in the observation's `label_topk` (0 if absent)."""
    return max((float(s) for lab, s in o.label_topk if lab == node.label), default=0.0)


def label_cost(node: Node, o: ObsInstance, mode: str, cfg: Mapping[str, Any]) -> float:
    """§6.9 label term. Oracle: hard gate with nyu40 fallback; perception: soft top-k term."""
    a = _acfg(cfg)
    if mode == "oracle":
        if o.label == node.label:
            return 0.0
        if (o.nyu40 is not None and node.nyu40 is not None and int(o.nyu40) == int(node.nyu40)
                and int(node.nyu40) not in {int(x) for x in a.get("nyu_fallback_exclude", [])}):
            return float(a["nyu_penalty"])
        return INF
    if mode == "perception":
        return -math.log(max(label_p(node, o), float(a["p_min"])))
    raise ValueError(f"unknown mode {mode!r}")


def label_compatible(node: Node, o: ObsInstance, mode: str, cfg: Mapping[str, Any]) -> bool:
    """Label condition of the absorption rule: oracle same label or nyu40; perception the
    node's label is in `label_topk` with p >= p_min."""
    if mode == "oracle":
        return label_cost(node, o, mode, cfg) < INF
    return label_p(node, o) >= float(_acfg(cfg)["p_min"])


def _extents(lo: Sequence[float], hi: Sequence[float], floor: float) -> np.ndarray:
    return np.sort(np.maximum(np.asarray(hi, float) - np.asarray(lo, float), floor))


def size_cost(node: Node, o: ObsInstance, cfg: Mapping[str, Any]) -> float:
    """Mean |log ratio| of the sorted AABB extents of node and observation."""
    f = float(_acfg(cfg)["min_extent"])
    assert node.bbox_3d is not None
    a = _extents(node.bbox_3d["min"], node.bbox_3d["max"], f)
    b = _extents(o.aabb_min, o.aabb_max, f)
    return float(np.mean(np.abs(np.log(a / b))))


def pos_cost(node: Node, o: ObsInstance, cfg: Mapping[str, Any]) -> float:
    """Weak position term min(centroid distance / pos_scale, 1)."""
    d = float(np.linalg.norm(o.centroid - np.asarray(node.centroid, float)))
    return min(d / float(_acfg(cfg)["pos_scale"]), 1.0)


def pair_cost(node: Node, o: ObsInstance, mode: str, cfg: Mapping[str, Any]) -> float:
    """Full §6.9 cost of one (node, observation) pair (+inf if the label gate fails)."""
    lc = label_cost(node, o, mode, cfg)
    if lc == INF:
        return INF
    a = _acfg(cfg)
    return lc + float(a["w_g"]) * size_cost(node, o, cfg) + float(a["w_p"]) * pos_cost(node, o, cfg)


def tau_assoc(mode: str, cfg: Mapping[str, Any]) -> float:
    """Gate of the given mode."""
    t = _acfg(cfg)["tau_assoc"]
    return float(t[mode] if isinstance(t, Mapping) else t)


def inside_frac(node: Node, o: ObsInstance, margin: float) -> float:
    """Share of the observation's points inside the node's AABB grown by `margin`."""
    assert node.bbox_3d is not None
    lo = np.asarray(node.bbox_3d["min"], float) - margin
    hi = np.asarray(node.bbox_3d["max"], float) + margin
    P = np.asarray(o.points, float)
    if not len(P):
        return 0.0
    return float(np.all((P >= lo) & (P <= hi), axis=1).mean())


@dataclass
class Absorption:
    """Result of stage 0."""

    token_of: dict[str, str]  # confirmed node id -> best absorbed token
    absorbed: dict[str, str]  # token -> confirmed node id that absorbed it


def absorb_confirmed(
    confirmed: Sequence[Node], obs: Sequence[ObsInstance], mode: str, cfg: Mapping[str, Any]
) -> Absorption:
    """Stage 0: confirmed-in-place nodes claim compatible observations (module docstring)."""
    ab = _acfg(cfg)["absorb"]
    min_frac, margin = float(ab["min_frac"]), float(ab["margin"])
    absorbed: dict[str, str] = {}
    best: dict[str, tuple[float, float, str]] = {}  # node -> (-frac, dist, token)
    for o in sorted(obs, key=lambda x: x.token):
        if not len(o.points):
            continue
        opts = []
        for n in confirmed:
            if not label_compatible(n, o, mode, cfg):
                continue
            f = inside_frac(n, o, margin)
            if f >= min_frac:
                d = float(np.linalg.norm(o.centroid - np.asarray(n.centroid, float)))
                opts.append((-f, d, n.id))
        if not opts:
            continue
        nf, d, nid = min(opts)
        absorbed[o.token] = nid
        key = (nf, d, o.token)
        if nid not in best or key < best[nid]:
            best[nid] = key
    return Absorption({nid: k[2] for nid, k in best.items()}, absorbed)


Gate = Callable[[Node, ObsInstance], bool]


def old_place_gate(votes: Mapping[str, NodeVotes] | None, cfg: Mapping[str, Any]) -> Gate | None:
    """The Part A pair gate (module docstring); None if `association.old_place.enabled` is false."""
    oc = _acfg(cfg).get("old_place") or {}
    if not oc.get("enabled", False):
        return None
    vs = votes or {}
    frac, margin = float(oc["inside_frac"]), float(oc["margin"])
    unseen_inside = bool(oc.get("unobserved_inside_only", False))
    dmax = oc.get("max_dist_unseen")

    def gate(n: Node, o: ObsInstance) -> bool:
        c = old_place_class(vs.get(n.id), cfg)
        if c == "net_seen" or (c == "unobserved" and unseen_inside):
            return inside_frac(n, o, margin) >= frac
        if c == "unobserved" and dmax is not None:
            d = float(np.linalg.norm(o.centroid - np.asarray(n.centroid, float)))
            return d <= float(dmax) or inside_frac(n, o, margin) >= frac
        return True

    return gate


def associate(
    nodes: Sequence[Node], obs: Sequence[ObsInstance], mode: str, cfg: Mapping[str, Any], gate: Gate | None = None
) -> tuple[list[tuple[Node, ObsInstance, float]], list[Node], list[ObsInstance]]:
    """Stage 1: Hungarian assignment with gate `tau_assoc` (DESIGN §6.9).

    Costs are clipped at tau (= "leave unmatched" costs tau), so the assignment is optimal
    among gated pairs; assigned pairs with cost >= tau are dropped. Inputs are sorted by node
    id / token first, so the result is deterministic.
    `gate(node, obs)` False -> the pair costs +inf (the Part A old-place gate, `old_place_gate`).

    Returns:
        (pairs (node, obs, cost), unmatched nodes, unmatched observations).
    """
    N = sorted(nodes, key=lambda n: n.id)
    O = sorted((o for o in obs if len(o.points)), key=lambda o: o.token)
    tau = tau_assoc(mode, cfg)
    if not N or not O:
        return [], N, O
    C = np.array([[pair_cost(n, o, mode, cfg) if gate is None or gate(n, o) else INF for o in O] for n in N], float)
    rows, cols = linear_sum_assignment(np.minimum(C, tau))
    pairs = [(N[i], O[j], float(C[i, j])) for i, j in zip(rows, cols) if C[i, j] < tau]
    mn = {p[0].id for p in pairs}
    mo = {p[1].token for p in pairs}
    return pairs, [n for n in N if n.id not in mn], [o for o in O if o.token not in mo]
