"""Build the reference scene graph G_ref from a reference scan (DESIGN §6.3, §4).

Two relation modes:
- `gt`: support parents from the reference's 3DSSG relations (§5 predicate mapping);
- `geometric`: support parents from `relations.infer_support` on the reference mesh (R1).

Only reference-scan data is used (mesh, semseg/PLY labels, 3DSSG of the reference), which is
the method's allowed input (CLAUDE.md hard rule 1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.graph.model import (
    FLOOR_ID,
    ROOM_ID,
    Edge,
    Graph,
    Node,
    bbox_dict,
    ref_node_id,
)
from lsg.graph.relations import (
    FLOOR,
    SupportCandidate,
    aabb_gap,
    estimate_floor_z,
    gt_support,
    infer_support,
    pick_parent,
)
from lsg.io.scan3r import Scan


def object_nyu40(oid: int, scan: Scan, objects: Mapping[int, Mapping[str, Any]]) -> int:
    """nyu40 class of an object: 3DSSG `objects.json` if present, else the PLY vertex majority."""
    o = objects.get(oid)
    if o is not None and "nyu40" in o:
        return int(o["nyu40"])
    return int(scan.instances[oid].nyu40)


def anchor_of(node_id: str, parent: Mapping[str, str], level: Mapping[str, str]) -> str | None:
    """First anchor up the support chain of `node_id` (cycle-safe); None if there is none."""
    seen = {node_id}
    p = parent.get(node_id)
    while p is not None and p != FLOOR_ID and p not in seen:
        if level.get(p) == "anchor":
            return p
        seen.add(p)
        p = parent.get(p)
    return None


def build_reference_graph(
    scan: Scan,
    objects: Mapping[int, Mapping[str, Any]],
    relations: Iterable[tuple[int, int, str]] | None = None,
    mode: str = "gt",
    cfg: Mapping[str, Any] | None = None,
) -> Graph:
    """Reference scene graph of one reference scan.

    Nodes: one `room` node plus one node per annotated object whose level (from nyu40,
    `lsg.classes.node_level`) is `anchor` or `thing`; structure gets no node. Node fields:
    id `r<objectId>`, label (semseg, else 3DSSG), nyu40, AABB + centroid of the object's mesh
    vertices, `last_seen` = the reference scan, `state` = unchanged, `confidence` None (set at
    M6), `anchor` = first anchor up the support chain (None for anchors themselves).

    Edges: `hierarchy` room -> every anchor/thing; at most one support edge (`on`/`in`) per
    **thing** (anchors carry none, DECISIONS M1), target an anchor, a thing or `floor`;
    undirected `next_to` (GT: `close by`; geometric: AABB gap <= `next_to_gap`), exported only.

    Args:
        scan: Reference scan from `lsg.io.scan3r.load_scan`.
        objects: 3DSSG objects of the same scan (`load_objects`); may be empty.
        relations: 3DSSG relations of the same scan; required for `mode="gt"`.
        mode: "gt" or "geometric".
        cfg: Full config (uses `cfg["relations"]`); required for `mode="geometric"`.

    Returns:
        The graph, `meta.relations` = mode, `meta.floor_z` = estimated floor height.
    """
    if mode not in ("gt", "geometric"):
        raise ValueError(f"mode must be 'gt' or 'geometric', not {mode!r}")
    if mode == "gt" and relations is None:
        raise ValueError("mode='gt' needs the reference's 3DSSG relations")
    rcfg = (cfg or {}).get("relations")
    if mode == "geometric" and rcfg is None:
        raise ValueError("mode='geometric' needs cfg['relations']")

    # ---- nodes
    level: dict[str, str] = {}
    nodes: list[Node] = []
    oids: dict[str, int] = {}
    for oid in sorted(scan.instances):
        lv = node_level(object_nyu40(oid, scan, objects))
        if lv == "structure":
            continue
        inst = scan.instances[oid]
        nid = ref_node_id(oid)
        label = inst.label or str(objects.get(oid, {}).get("label", ""))
        nodes.append(
            Node(
                id=nid,
                level=lv,
                label=label,
                nyu40=object_nyu40(oid, scan, objects),
                bbox_3d=bbox_dict(inst.aabb_min, inst.aabb_max),
                centroid=[float(v) for v in inst.centroid],
                last_seen=scan.scan_id,
                state="unchanged",
            )
        )
        level[nid] = lv
        oids[nid] = oid

    floor_z = estimate_floor_z(scan.vertices, scan.vertex_nyu40, rcfg or {"floor_fallback_percentile": 1.0})

    # ---- support parents (things only)
    parent: dict[str, str] = {}
    ptype: dict[str, str] = {}
    next_to: set[tuple[str, str]] = set()
    if mode == "gt":
        gs = gt_support(relations or [], objects)
        for nid, oid in oids.items():
            if level[nid] != "thing":
                continue
            ps = {
                (FLOOR_ID if p == FLOOR else ref_node_id(p)): t
                for p, t in gs.parents.get(oid, {}).items()
                if p == FLOOR or ref_node_id(p) in level
            }
            ps.pop(nid, None)
            if ps:
                p, t = pick_parent(ps, lambda q: level[q])
                parent[nid], ptype[nid] = p, t
        for a, b in gs.next_to:
            na, nb = ref_node_id(a), ref_node_id(b)
            if na in level and nb in level:
                next_to.add((min(na, nb), max(na, nb)))
    else:
        assert rcfg is not None
        pts = {nid: scan.instance_points(oid) for nid, oid in oids.items()}
        cands = [SupportCandidate(nid, level[nid], pts[nid]) for nid in oids if len(pts[nid])]
        for nid in oids:
            if level[nid] != "thing" or not len(pts[nid]):
                continue
            p, t, _ = infer_support(pts[nid], cands, floor_z, rcfg, child_id=nid)
            if p is not None:
                parent[nid], ptype[nid] = str(p), str(t)
        boxes = [(c.id, c.aabb_min, c.aabb_max) for c in cands]
        for i, (ia, amin, amax) in enumerate(boxes):
            for ib, bmin, bmax in boxes[i + 1 :]:
                if aabb_gap(amin, amax, bmin, bmax) <= rcfg["next_to_gap"]:
                    next_to.add((min(ia, ib), max(ia, ib)))

    # ---- anchors up the chain
    for n in nodes:
        if n.level == "thing":
            n.anchor = anchor_of(n.id, parent, level)

    # ---- room node + edges
    vmin, vmax = scan.vertices.min(axis=0), scan.vertices.max(axis=0)
    room = Node(
        id=ROOM_ID,
        level="room",
        label="room",
        bbox_3d=bbox_dict(vmin, vmax),
        centroid=[float(v) for v in (vmin + vmax) / 2],
        last_seen=scan.scan_id,
    )
    edges = [Edge(ROOM_ID, n.id, "hierarchy") for n in nodes]
    edges += [Edge(c, parent[c], ptype[c]) for c in sorted(parent)]
    edges += [Edge(a, b, "next_to") for a, b in sorted(next_to)]

    meta = Graph.new_meta(scan.scan_id)
    meta["relations"] = mode
    meta["floor_z"] = float(np.round(floor_z, 4))
    return Graph(meta, [room, *nodes], edges)
