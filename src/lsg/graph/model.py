"""Graph data model and its JSON form.

Node ids: reference nodes `"r<objectId>"`, new nodes `"n<k>"`, one `"room"` node per scan.
`"floor"` is a pseudo-parent id for support edges and is **not** a node.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOM_ID = "room"
FLOOR_ID = "floor"
SCHEMA_VERSION = 2
LEVELS = ("room", "place", "anchor", "thing")
STATES = ("unchanged", "moved", "removed", "unknown", "new")
EDGE_TYPES = ("hierarchy", "on", "in", "next_to")
SUPPORT_TYPES = ("on", "in")


def ref_node_id(object_id: int) -> str:
    """Graph id of a reference object: `r<objectId>`."""
    return f"r{int(object_id)}"


def object_id_of(node_id: str) -> int | None:
    """Inverse of `ref_node_id`; None for `room`, `floor`, new nodes (`n<k>`)."""
    return int(node_id[1:]) if node_id.startswith("r") and node_id[1:].isdigit() else None


def _vec(x: Any) -> list[float] | None:
    """numpy / sequence -> list of plain floats (None stays None)."""
    return None if x is None else [float(v) for v in np.asarray(x, dtype=float).ravel()]


@dataclass
class Node:
    """One graph node. Geometry in the reference frame, metres."""

    id: str
    level: str
    label: str
    nyu40: int | None = None
    label_topk: list[list[Any]] | None = None
    bbox_3d: dict[str, list[float]] | None = None  # {"min": [x,y,z], "max": [x,y,z]}, AABB
    centroid: list[float] | None = None
    embedding: list[float] | None = None
    confidence: float | None = None
    last_seen: str | None = None
    state: str = "unchanged"
    anchor: str | None = None  # first anchor up the support chain
    T_move: list[list[float]] | None = None
    obs_token: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Plain-JSON dict in the schema's field order."""
        return asdict(self)


@dataclass(frozen=True)
class Edge:
    """Directed edge. Support edges (`on`/`in`) point child -> parent; `hierarchy` room -> node."""

    source: str
    target: str
    type: str

    def to_json(self) -> dict[str, str]:
        """`{"source", "target", "type"}`."""
        return {"source": self.source, "target": self.target, "type": self.type}


@dataclass
class Graph:
    """A scene graph: `meta` block, nodes (by id) and edges."""

    meta: dict[str, Any]
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    @staticmethod
    def new_meta(scan_ref: str, scan_obs: str | None = None, obs_mode: str | None = None,
                 frame_idx: int | None = None, tentative: bool = False) -> dict[str, Any]:  # fmt: skip
        """The `meta` block."""
        return {
            "scan_ref": scan_ref,
            "scan_obs": scan_obs,
            "frame": "ref",
            "units": "m",
            "version": SCHEMA_VERSION,
            "obs_mode": obs_mode,
            "frame_idx": frame_idx,
            "tentative": tentative,
        }

    def node(self, node_id: str) -> Node:
        """Node by id (KeyError if missing)."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def node_map(self) -> dict[str, Node]:
        """`{id: Node}`."""
        return {n.id: n for n in self.nodes}

    def support_parent(self) -> dict[str, tuple[str, str]]:
        """`{child_id: (parent_id, "on"|"in")}` from the support edges."""
        return {e.source: (e.target, e.type) for e in self.edges if e.type in SUPPORT_TYPES}

    def to_json(self) -> dict[str, Any]:
        """The full JSON object."""
        return {
            "meta": dict(self.meta),
            "nodes": [n.to_json() for n in self.nodes],
            "edges": [e.to_json() for e in self.edges],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Graph:
        """Inverse of `to_json` (unknown node keys are ignored)."""
        keys = Node.__dataclass_fields__.keys()
        nodes = [Node(**{k: v for k, v in n.items() if k in keys}) for n in d["nodes"]]
        edges = [Edge(e["source"], e["target"], e["type"]) for e in d["edges"]]
        return cls(dict(d["meta"]), nodes, edges)

    def save(self, path: str | Path) -> Path:
        """Write the JSON to `path` (parent folders created)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """Read a graph JSON written by `save`."""
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def bbox_dict(aabb_min: Any, aabb_max: Any) -> dict[str, list[float]]:
    """`{"min": [...], "max": [...]}` from two 3-vectors."""
    return {"min": _vec(aabb_min), "max": _vec(aabb_max)}  # type: ignore[dict-item]


def validate(g: Graph) -> list[str]:
    """Light schema check of a graph. Returns a list of problems (empty = valid).

    Checks: meta keys and version; node ids unique, level / state values legal, bbox and
    centroid shapes; edge types legal and endpoints exist (`floor` allowed as support target);
    at most one support edge per node and only things carry one; `anchor` refers to an anchor.
    """
    probs: list[str] = []
    for k in ("scan_ref", "scan_obs", "frame", "units", "version", "obs_mode", "frame_idx",
              "tentative"):  # fmt: skip
        if k not in g.meta:
            probs.append(f"meta missing '{k}'")
    if g.meta.get("version") != SCHEMA_VERSION:
        probs.append(f"meta.version {g.meta.get('version')} != {SCHEMA_VERSION}")
    if g.meta.get("frame") != "ref" or g.meta.get("units") != "m":
        probs.append("meta.frame must be 'ref' and units 'm'")

    ids: dict[str, Node] = {}
    for n in g.nodes:
        if n.id in ids:
            probs.append(f"duplicate node id {n.id}")
        ids[n.id] = n
        if n.level not in LEVELS:
            probs.append(f"{n.id}: bad level {n.level}")
        if n.state not in STATES:
            probs.append(f"{n.id}: bad state {n.state}")
        if n.level in ("anchor", "thing"):
            if n.bbox_3d is None or len(n.bbox_3d.get("min", [])) != 3 or len(n.bbox_3d.get("max", [])) != 3:
                probs.append(f"{n.id}: bbox_3d must be {{min: [3], max: [3]}}")
            if n.centroid is None or len(n.centroid) != 3:
                probs.append(f"{n.id}: centroid must have 3 values")
        if n.T_move is not None and np.asarray(n.T_move, dtype=float).shape != (4, 4):
            probs.append(f"{n.id}: T_move must be 4x4")
        if n.confidence is not None and not (0.0 <= float(n.confidence) <= 1.0):
            probs.append(f"{n.id}: confidence must be in [0, 1]")
        if n.label_topk is not None and not all(
            isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in n.label_topk
        ):
            probs.append(f"{n.id}: label_topk must be [[label, score], ...]")
        if n.id.startswith("n") and n.state != "new":
            probs.append(f"{n.id}: new-node id with state {n.state}")
        if n.state == "new" and not n.id.startswith("n"):
            probs.append(f"{n.id}: state new needs an n<k> id")
    if ROOM_ID not in ids:
        probs.append("no room node")

    n_support: dict[str, int] = {}
    for e in g.edges:
        if e.type not in EDGE_TYPES:
            probs.append(f"edge {e.source}->{e.target}: bad type {e.type}")
        if e.source not in ids:
            probs.append(f"edge source {e.source} is not a node")
        if e.target not in ids and not (e.type in SUPPORT_TYPES and e.target == FLOOR_ID):
            probs.append(f"edge target {e.target} is not a node")
        if e.type in SUPPORT_TYPES:
            n_support[e.source] = n_support.get(e.source, 0) + 1
            if e.source in ids and ids[e.source].level != "thing":
                probs.append(f"support edge from non-thing {e.source}")
    probs += [f"{c}: {k} support edges (max 1)" for c, k in n_support.items() if k > 1]
    for n in g.nodes:
        if n.anchor is not None and (n.anchor not in ids or ids[n.anchor].level != "anchor"):
            probs.append(f"{n.id}: anchor {n.anchor} is not an anchor node")
    return probs
