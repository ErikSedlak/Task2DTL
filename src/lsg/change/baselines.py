"""Baselines (DESIGN §8). M2: B0 naive; M3: B1 + visibility; M5: B2 + Hungarian.

B0: nearest same-label observation within `near_dist` -> unchanged; same label farther ->
moved; unmatched reference node -> removed; unmatched observation -> new. No visibility, no
confidence. Matching is greedy one-to-one by centroid distance (DECISIONS M2), so one observed
chair cannot explain two reference chairs.

B1: B0's matching; an unmatched reference node is `removed` only with >= k_free counted
(decorrelated) free votes from the online loop, else `unknown` (kept). Decided once, at the
end of the sequence (deferred commitment).

B2: confirmed-in-place nodes are `unchanged` and absorb their observations; the rest are
matched by Hungarian assignment on label + size + position (`change.association`); matched
nodes get the ICP moved test, unmatched ones B1's removal rule (`change.classify`).

Every method has the signature `method(G_ref, obs, cfg, rescan_id=None, obs_mode=None,
votes=None, ref_ctx=None) -> G_upd`; `votes` = `OnlineResult.votes` (node id -> `NodeVotes`),
`ref_ctx` = the reference context (B2 reads the reference node points from it). Relations
are not touched here; `change.update.update_graph` recomputes them.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.change.association import absorb_confirmed, associate
from lsg.change.classify import classify
from lsg.change.visibility import NodeVotes, confirmed_in_place, has_free_evidence
from lsg.classes import node_level
from lsg.geom.transforms import apply
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, object_id_of
from lsg.obs.base import ObsInstance, RefContext

SCORED_LEVELS = ("anchor", "thing")


def obs_level(o: ObsInstance) -> str:
    """Graph level of an observation from its nyu40 (unknown class -> thing)."""
    return node_level(int(o.nyu40)) if o.nyu40 is not None else "thing"


def greedy_label_match(
    nodes: Sequence[Node], obs: Sequence[ObsInstance]
) -> list[tuple[Node, ObsInstance, float]]:
    """One-to-one matching of same-label (node, obs) pairs, shortest centroid distance first.

    Ties are broken by node id and token, so the result is deterministic.
    """
    cand = []
    for n in nodes:
        c = np.asarray(n.centroid, float)
        for o in obs:
            if o.label == n.label:
                cand.append((float(np.linalg.norm(o.centroid - c)), n.id, o.token, n, o))
    cand.sort(key=lambda x: x[:3])
    used_n: set[str] = set()
    used_o: set[str] = set()
    out = []
    for d, nid, tok, n, o in cand:
        if nid in used_n or tok in used_o:
            continue
        used_n.add(nid)
        used_o.add(tok)
        out.append((n, o, d))
    return out


def translation(t: np.ndarray) -> list[list[float]]:
    """4x4 translation matrix as nested lists (for `Node.T_move`)."""
    T = np.eye(4)
    T[:3, 3] = t
    return T.tolist()


def new_node(k: int, o: ObsInstance, rescan_id: str | None) -> Node:
    """Graph node `n<k>` for an unmatched observation."""
    return Node(
        id=f"n{k}",
        level=obs_level(o),
        label=o.label,
        nyu40=o.nyu40,
        label_topk=[[lab, float(s)] for lab, s in o.label_topk],
        bbox_3d=bbox_dict(o.aabb_min, o.aabb_max),
        centroid=[float(v) for v in o.centroid],
        last_seen=rescan_id,
        state="new",
        obs_token=o.token,
    )


def add_new_nodes(G: Graph, obs: Sequence[ObsInstance], rescan_id: str | None) -> list[Node]:
    """Append new nodes (+ room hierarchy edges) for `obs`, in token order."""
    added = []
    for k, o in enumerate(sorted(obs, key=lambda x: x.token)):
        n = new_node(k, o, rescan_id)
        G.nodes.append(n)
        G.edges.append(Edge(ROOM_ID, n.id, "hierarchy"))
        added.append(n)
    return added


def _match_b0(
    G_ref: Graph,
    obs: Sequence[ObsInstance],
    cfg: Mapping[str, Any],
    rescan_id: str | None,
    obs_mode: str | None,
    method: str,
) -> tuple[Graph, list[Node], set[str], list[ObsInstance]]:
    """B0 matching shared by B0 and B1.

    Returns:
        (G copy with matched nodes set to unchanged / moved, scored nodes, ids of matched
        nodes, unmatched non-structure observations). Unmatched nodes are left for the caller.
    """
    near = float(cfg["baselines"]["b0"]["near_dist"])
    G = copy.deepcopy(G_ref)
    G.meta.update(scan_obs=rescan_id, obs_mode=obs_mode, method=method, tentative=False, frame_idx=None)
    nodes = [n for n in G.nodes if n.level in SCORED_LEVELS]
    usable = [o for o in obs if obs_level(o) != "structure" and len(o.points)]

    matched_tokens: set[str] = set()
    matched_nodes: set[str] = set()
    for n, o, d in greedy_label_match(nodes, usable):
        matched_tokens.add(o.token)
        matched_nodes.add(n.id)
        n.obs_token = o.token
        n.last_seen = rescan_id
        if d <= near:
            n.state = "unchanged"
            continue
        t = o.centroid - np.asarray(n.centroid, float)
        n.state = "moved"
        n.centroid = [float(v) for v in o.centroid]
        assert n.bbox_3d is not None
        n.bbox_3d = bbox_dict(np.asarray(n.bbox_3d["min"]) + t, np.asarray(n.bbox_3d["max"]) + t)
        n.T_move = translation(t)
    return G, nodes, matched_nodes, [o for o in usable if o.token not in matched_tokens]


def b0(
    G_ref: Graph,
    obs: Sequence[ObsInstance],
    cfg: Mapping[str, Any],
    rescan_id: str | None = None,
    obs_mode: str | None = None,
    votes: Mapping[str, NodeVotes] | None = None,
    ref_ctx: RefContext | None = None,
) -> Graph:
    """Naive baseline B0 (DESIGN §8). Returns the updated graph G_upd.

    Args:
        G_ref: Reference graph (not modified).
        obs: Observed instances (`ObsSource.finish()`); structure-level ones are ignored.
        cfg: Full config; uses `cfg["baselines"]["b0"]["near_dist"]`.
        rescan_id, obs_mode: Written to `meta.scan_obs` / `meta.obs_mode`.
        votes: Ignored (B0 has no visibility reasoning).

    Returns:
        G_upd: reference nodes with `state` unchanged / moved / removed (moved nodes carry the
        observed centroid, a shifted bbox and `T_move` = the translation), `obs_token` of the
        matched observation, plus one `new` node per unmatched non-structure observation.
        Relations are not recomputed (M5).
    """
    G, nodes, matched, rest = _match_b0(G_ref, obs, cfg, rescan_id, obs_mode, "B0")
    for n in nodes:
        if n.id not in matched:
            n.state = "removed"
    add_new_nodes(G, rest, rescan_id)
    return G


def b1(
    G_ref: Graph,
    obs: Sequence[ObsInstance],
    cfg: Mapping[str, Any],
    rescan_id: str | None = None,
    obs_mode: str | None = None,
    votes: Mapping[str, NodeVotes] | None = None,
    ref_ctx: RefContext | None = None,
) -> Graph:
    """Baseline B1 = B0 matching + visibility removal rule (DESIGN §8). Returns G_upd.

    Matched nodes are unchanged / moved exactly as in B0. An unmatched node is `removed` only
    if it has >= `states.k_free` counted free votes, else `unknown` (kept, not deleted).
    Without `votes` (None) every unmatched node is `unknown`. Unmatched nodes with a raw seen
    vote get `last_seen` = the rescan.
    """
    G, nodes, matched, rest = _match_b0(G_ref, obs, cfg, rescan_id, obs_mode, "B1")
    vs = votes or {}
    for n in nodes:
        if n.id in matched:
            continue
        v = vs.get(n.id)
        n.state = "removed" if has_free_evidence(v, cfg) else "unknown"
        if v is not None and v.raw_seen > 0:
            n.last_seen = rescan_id
    add_new_nodes(G, rest, rescan_id)
    return G


def ref_node_points(G: Graph, ref_ctx: RefContext | None) -> dict[str, np.ndarray]:
    """Reference points (mesh vertices) of every reference node, from the reference scan."""
    if ref_ctx is None:
        return {}
    out = {}
    for n in G.nodes:
        oid = object_id_of(n.id)
        if oid is not None and n.level in SCORED_LEVELS:
            out[n.id] = ref_ctx.scan.instance_points(oid)
    return out


def b2(
    G_ref: Graph,
    obs: Sequence[ObsInstance],
    cfg: Mapping[str, Any],
    rescan_id: str | None = None,
    obs_mode: str | None = None,
    votes: Mapping[str, NodeVotes] | None = None,
    ref_ctx: RefContext | None = None,
) -> Graph:
    """Baseline B2 = confirmed-in-place skip + Hungarian + ICP moved test (DESIGN §8, §6.9-10).

    1. Nodes confirmed in place (>= k_seen counted seen, no counted free) are `unchanged` and
       absorb compatible observations lying inside their AABB (stage 0, M5).
    2. The other scored nodes are matched to the remaining non-structure observations
       (Hungarian, mode-dependent label term, gate `association.tau_assoc`).
    3. Matched: ICP-refined displacement > `classify.tau_move` -> `moved` (centroid and bbox
       moved with T_move), else `unchanged`. Unmatched: B1's rule (`removed` / `unknown`).
    4. Unmatched, unabsorbed observations become `new` nodes.

    Without `ref_ctx`, ICP falls back to centroid offsets. `obs_mode` None is treated as oracle.
    """
    mode = obs_mode or "oracle"
    G = copy.deepcopy(G_ref)
    G.meta.update(scan_obs=rescan_id, obs_mode=obs_mode, method="B2", tentative=False, frame_idx=None)
    vs = votes or {}
    nodes = [n for n in G.nodes if n.level in SCORED_LEVELS]
    usable = [o for o in obs if obs_level(o) != "structure" and len(o.points)]
    conf = [n for n in nodes if confirmed_in_place(vs.get(n.id), cfg)]
    ab = absorb_confirmed(conf, usable, mode, cfg)
    rest_obs = [o for o in usable if o.token not in ab.absorbed]
    conf_ids = {n.id for n in conf}
    pairs, unmatched, unmatched_obs = associate([n for n in nodes if n.id not in conf_ids], rest_obs, mode, cfg)
    P = ref_node_points(G, ref_ctx)
    dec = classify(pairs, unmatched, vs, conf, P, cfg)
    tok = {n.id: o.token for n, o, _ in pairs} | ab.token_of
    for n in nodes:
        d = dec[n.id]
        n.state = d.state
        n.obs_token = tok.get(n.id)
        v = vs.get(n.id)
        if n.obs_token is not None or n.id in conf_ids or (v is not None and v.raw_seen > 0):
            n.last_seen = rescan_id
        if d.state != "moved":
            continue
        assert d.T_move is not None
        pts = P.get(n.id)
        if pts is not None and len(pts):
            Q = apply(d.T_move, pts)
            n.centroid = [float(x) for x in Q.mean(axis=0)]
            n.bbox_3d = bbox_dict(Q.min(axis=0), Q.max(axis=0))
        else:
            assert n.bbox_3d is not None
            n.centroid = [float(x) for x in apply(d.T_move, np.asarray(n.centroid, float)[None])[0]]
            t = d.T_move[:3, 3]
            n.bbox_3d = bbox_dict(np.asarray(n.bbox_3d["min"]) + t, np.asarray(n.bbox_3d["max"]) + t)
        n.T_move = np.asarray(d.T_move, float).tolist()
    add_new_nodes(G, unmatched_obs, rescan_id)
    G.meta["assoc"] = {"confirmed": len(conf), "absorbed": len(ab.absorbed), "matched": len(pairs),
                       "icp_fallback": sum(bool(dec[n.id].info.get("fallback")) for n, _, _ in pairs)}  # fmt: skip
    return G


METHODS = {"B0": b0, "B1": b1, "B2": b2}
