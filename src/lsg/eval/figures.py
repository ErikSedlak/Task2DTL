"""Figures that need GT (M8): the top-down GT panel for `lsg viz topdown --gt`.

Lives in `eval/` because it reads rescan annotations (`gt_changes`, hard rule 1); the drawing
itself is in `lsg.viz.topdown`, which only receives the finished GT dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsg.config import data_root
from lsg.eval.gt import GTLabels, gt_changes
from lsg.graph.build import build_reference_graph
from lsg.graph.model import Graph, ref_node_id
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan
from lsg.viz.topdown import room_outline, topdown_png


def gt_panel(gt: GTLabels) -> dict[str, Any]:
    """GT dict for `topdown_png(gt=...)`: reference-object statuses, GT new boxes, GT new footprints."""
    states = {ref_node_id(i): o.status for i, o in gt.objects.items()}
    moved_to = {ref_node_id(i): (o.aabb_new_min, o.aabb_new_max) for i, o in gt.objects.items()
                if o.status == "moved" and o.aabb_new_min is not None}  # fmt: skip
    new = [(g.label, g.aabb_min, g.aabb_max) for g in gt.new.values() if g.aabb_min is not None]
    return {"states": states, "new": new, "moved_to": moved_to}


def topdown_for_pair(cfg: dict[str, Any], graph_path: str | Path, ref: str, rescan: str, out: str | Path,
                     with_gt: bool = False, title: str | None = None, edges: bool = False) -> Path:  # fmt: skip
    """Load the reference, its GT graph and a saved G_upd, then write the top-down PNG."""
    root = data_root(cfg)
    R = load_scan(root, ref, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    G_upd = Graph.load(graph_path)
    gt = None
    if with_gt:
        S = load_scan(root, rescan, with_intrinsics=False)
        gt = gt_panel(gt_changes(load_meta(root)[rescan], R, S, objs_R, load_objects(root, rescan), cfg))
    outline = room_outline(R.vertices, R.vertex_nyu40)
    return topdown_png(G_ref, G_upd, out, outline=outline, gt=gt, edges=edges,
                       title=title or f"{ref[:8]} → {rescan[:8]}")  # fmt: skip
