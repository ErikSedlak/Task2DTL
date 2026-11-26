"""Graph update after the change states are decided: support relations.

`update_graph(G_states, ref_ctx, obs, cfg, policy)` takes a method's output (node states,
moved geometry, new nodes; reference relations still in place) and recomputes the support
edges according to `policy`:

- `update` (the method): moved things, new things and things whose parent moved or was
  removed are re-checked; every other thing keeps its reference edge. A re-checked thing
  keeps its old parent if that parent still carries it at the new pose (`still_supports`,
  config `update.keep_supported_parent`, M5); otherwise `infer_support` picks a new one;
- `stale` (baseline R0): reference edges only; new nodes get no parent;
- `geometric` (baseline R1): `infer_support` for every thing.

Support candidates are the anchor/thing nodes that are not removed: reference nodes with their
reference mesh points (moved nodes: `T_move` applied), new nodes with the points of their
observation (`obs_token`). Removed nodes keep their old edge in the file (deletion is a view
filter) but never act as a parent. After the edges, the `anchor` field is recomputed up
the chain, and `next_to` edges touching moved / new nodes are recomputed (AABB gap).

Method side: imports only `geom`, `graph`, `obs`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.geom.transforms import apply
from lsg.graph.build import anchor_of
from lsg.graph.model import FLOOR_ID, SUPPORT_TYPES, Edge, Graph, object_id_of
from lsg.graph.relations import FLOOR, SupportCandidate, aabb_gap, infer_support, still_supports
from lsg.obs.base import ObsInstance, RefContext

POLICIES = ("update", "stale", "geometric")
NODE_LEVELS = ("anchor", "thing")


def node_points(G: Graph, ref_ctx: RefContext, obs: Sequence[ObsInstance]) -> dict[str, np.ndarray]:
    """Current points of every anchor/thing node in the reference frame.

    Reference nodes: reference mesh vertices, moved with `T_move` if the node moved. New
    nodes: the observation's points. Nodes without points are left out.
    """
    by_tok = {o.token: o for o in obs}
    out: dict[str, np.ndarray] = {}
    for n in G.nodes:
        if n.level not in NODE_LEVELS:
            continue
        oid = object_id_of(n.id)
        if oid is not None:
            P = ref_ctx.scan.instance_points(oid)
            if n.state == "moved" and n.T_move is not None and len(P):
                P = apply(np.asarray(n.T_move, float), P)
        elif n.obs_token in by_tok:
            P = np.asarray(by_tok[n.obs_token].points, float)
        else:
            continue
        if len(P):
            out[n.id] = P
    return out


def update_graph(
    G_states: Graph,
    ref_ctx: RefContext,
    obs: Sequence[ObsInstance],
    cfg: Mapping[str, Any],
    policy: str = "update",
) -> Graph:
    """Recompute support relations of a method's output graph.

    Args:
        G_states: Output of a change method (not modified).
        ref_ctx: Reference context (scan points, floor height).
        obs: The observations the method saw (points of new nodes).
        cfg: Full config (`relations`).
        policy: "update" | "stale" | "geometric".

    Returns:
        A new graph with `meta.relation_policy` = policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown relation policy {policy!r} (choose from {POLICIES})")
    rcfg = cfg["relations"]
    G = copy.deepcopy(G_states)
    G.meta["relation_policy"] = policy
    nodes = {n.id: n for n in G.nodes}
    level = {n.id: n.level for n in G.nodes}
    parent: dict[str, tuple[str, str]] = {e.source: (e.target, e.type) for e in G.edges if e.type in SUPPORT_TYPES}
    things = [n for n in G.nodes if n.level == "thing"]

    if policy == "stale":
        recompute: list[str] = []
    elif policy == "geometric":
        recompute = [n.id for n in things if n.state != "removed"]
    else:
        changed = {n.id for n in G.nodes if n.state in ("moved", "removed")}
        recompute = [
            n.id for n in things
            if n.state != "removed"
            and (n.state in ("moved", "new") or (n.id in parent and parent[n.id][0] in changed))
        ]  # fmt: skip

    pts = node_points(G, ref_ctx, obs) if recompute else {}
    cands = [SupportCandidate(nid, level[nid], P) for nid, P in pts.items() if nodes[nid].state != "removed"]
    by_id = {c.id: c for c in cands}
    keep = policy == "update" and bool(cfg.get("update", {}).get("keep_supported_parent", True))
    fz = float(ref_ctx.floor_z)
    for nid in recompute:
        old = parent.pop(nid, None)
        if nid not in pts:
            continue
        if keep and old is not None:  # the old parent still carries it at the new pose -> keep (M5)
            p_old = FLOOR if old[0] == FLOOR_ID else by_id.get(old[0])
            t = still_supports(pts[nid], p_old, old[1], fz, rcfg) if p_old is not None else None
            if t is not None:
                parent[nid] = (old[0], t)
                continue
        p, t, _ = infer_support(pts[nid], cands, fz, rcfg, child_id=nid)
        if p is not None:
            parent[nid] = (str(p), str(t))

    up = {c: p for c, (p, _) in parent.items()}
    for n in G.nodes:
        if n.level == "thing":
            n.anchor = anchor_of(n.id, up, level)

    # next_to: drop and recompute edges touching moved / new nodes (exported only)
    moving = {n.id for n in G.nodes if n.state in ("moved", "new")} if policy != "stale" else set()
    nt = {(e.source, e.target) for e in G.edges if e.type == "next_to" and not ({e.source, e.target} & moving)}
    if moving:
        boxes = {n.id: (np.asarray(n.bbox_3d["min"]), np.asarray(n.bbox_3d["max"]))
                 for n in G.nodes if n.level in NODE_LEVELS and n.bbox_3d is not None and n.state != "removed"}  # fmt: skip
        gap = float(rcfg["next_to_gap"])
        for a in sorted(moving & set(boxes)):
            for b in sorted(boxes):
                if a != b and aabb_gap(*boxes[a], *boxes[b]) <= gap:
                    nt.add((min(a, b), max(a, b)))

    G.edges = (
        [e for e in G.edges if e.type == "hierarchy"]
        + [Edge(c, p, t) for c, (p, t) in sorted(parent.items()) if c in nodes and (p in nodes or p == FLOOR_ID)]
        + [Edge(a, b, "next_to") for a, b in sorted(nt)]
    )
    return G
