"""Support relations: 3DSSG predicate mapping (DESIGN §5) and geometric rules (DESIGN §6.3).

One support parent per thing: an anchor, a thing, or the pseudo-parent `"floor"`; relation
type `on` or `in`. Thresholds come from `cfg["relations"]` (tune on dev reference scans only).

Ids in this module are plain values chosen by the caller (3DSSG objectIds for `gt_support`,
graph node ids for `infer_support`); `FLOOR` marks the floor.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from lsg.classes import node_level
from lsg.graph.model import FLOOR_ID

FLOOR = FLOOR_ID

# DESIGN §5 - spec definitions, not tunables
ON_PREDICATES = frozenset({"standing on", "lying on", "supported by", "hanging on"})
IN_PREDICATES = frozenset({"standing in", "lying in", "hanging in", "inside", "build in"})
NEXT_TO_PREDICATES = frozenset({"close by"})
FLOOR_NYU40 = 2


def support_type(predicate: str) -> str | None:
    """3DSSG predicate -> "on" / "in" / None (not a support relation)."""
    if predicate in ON_PREDICATES:
        return "on"
    if predicate in IN_PREDICATES:
        return "in"
    return None


# --------------------------------------------------------------------------------------------
# Ground-truth support from 3DSSG
# --------------------------------------------------------------------------------------------


@dataclass
class GTSupport:
    """Support parents of one scan according to its 3DSSG relations.

    `parents[child] = {parent: "on"|"in"}` with parent = objectId or `FLOOR`. `excluded` holds
    children whose only support parents are structure other than the floor (wall, ceiling,
    window, ...): such a thing has no representable parent and is excluded from relation
    metrics (§5). `next_to` holds undirected `close by` pairs (sorted tuples).
    """

    parents: dict[int, dict[Any, str]] = field(default_factory=dict)
    excluded: set[int] = field(default_factory=set)
    next_to: set[tuple[int, int]] = field(default_factory=set)


def gt_support(
    relations: Iterable[tuple[int, int, str]], objects: Mapping[int, Mapping[str, Any]]
) -> GTSupport:
    """Map a scan's 3DSSG relationships to support parents (DESIGN §5).

    Args:
        relations: `(child_id, parent_id, predicate)` from `lsg.io.dssg.load_relations`.
        objects: `{objectId: {"nyu40", ...}}` from `lsg.io.dssg.load_objects` (same scan).
            Relations whose parent is not in `objects` are ignored.

    Returns:
        `GTSupport`. If one child has both `on` and `in` to the same parent, `on` is kept.
    """
    out = GTSupport()
    struct_only: set[int] = set()
    for c, p, name in relations:
        if name in NEXT_TO_PREDICATES:
            if c != p:
                out.next_to.add((min(c, p), max(c, p)))
            continue
        t = support_type(name)
        if t is None or p not in objects:
            continue
        nyu = int(objects[p].get("nyu40", 0))
        if nyu == FLOOR_NYU40:
            key: Any = FLOOR
        elif node_level(nyu) == "structure":
            struct_only.add(c)
            continue
        else:
            key = p
        d = out.parents.setdefault(c, {})
        if d.get(key) != "on":
            d[key] = t
    out.excluded = {c for c in struct_only if c not in out.parents}
    return out


_LEVEL_RANK = {"anchor": 0, "thing": 1}


def pick_parent(parents: Mapping[Any, str], level_of: Callable[[Any], str]) -> tuple[Any, str]:
    """Choose one parent when the GT gives several (deterministic, DECISIONS M1).

    Order: anchor > thing > floor, then `on` before `in`, then the smallest id.

    Args:
        parents: `{parent: "on"|"in"}`, non-empty; parent = id or `FLOOR`.
        level_of: id -> "anchor" | "thing" (not called for `FLOOR`).

    Returns:
        `(parent, type)`.
    """

    def key(item: tuple[Any, str]) -> tuple[int, int, str]:
        """Sort key implementing the order above."""
        p, t = item
        rank = 2 if p == FLOOR else _LEVEL_RANK.get(level_of(p), 3)
        return (rank, 0 if t == "on" else 1, f"{p!s:>12}")

    return min(parents.items(), key=key)


# --------------------------------------------------------------------------------------------
# Geometric rules
# --------------------------------------------------------------------------------------------


def estimate_floor_z(vertices: np.ndarray, vertex_nyu40: np.ndarray, cfg: Mapping[str, Any]) -> float:
    """Floor height of a reference scan (+Z is up; the floor is not at z = 0).

    Median z of the vertices labelled floor (nyu40 = 2); with fewer than 100 such vertices,
    the `floor_fallback_percentile` of all vertex z. Uses reference annotations only.
    """
    fz = vertices[vertex_nyu40 == FLOOR_NYU40, 2]
    if fz.size >= 100:
        return float(np.median(fz))
    return float(np.percentile(vertices[:, 2], cfg["floor_fallback_percentile"]))


@dataclass
class SupportCandidate:
    """A possible support parent: an anchor or thing with its points (reference frame)."""

    id: Hashable
    level: str
    points: np.ndarray  # (N, 3)
    aabb_min: np.ndarray = field(init=False)
    aabb_max: np.ndarray = field(init=False)
    _tree: cKDTree | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the AABB."""
        self.points = np.asarray(self.points, dtype=np.float64)
        self.aabb_min = self.points.min(axis=0)
        self.aabb_max = self.points.max(axis=0)

    @property
    def tree_xy(self) -> cKDTree:
        """2-D KD-tree over the candidate's (x, y), built on first use."""
        if self._tree is None:
            self._tree = cKDTree(self.points[:, :2])
        return self._tree


def contact_fraction(
    slice_pts: np.ndarray, cand: SupportCandidate, cfg: Mapping[str, Any], floor_z: float | None = None
) -> float:
    """Share of the child's bottom-slice points that rest on `cand`.

    A slice point is supported if some candidate point lies within `contact_xy` horizontally
    and 0..`contact_dz` below it. With `floor_z`, candidate points lower than
    `floor_z + support_min_height` do not count: at floor level the floor carries the child,
    not a neighbour's bottom edge (DECISIONS M1).
    """
    nb = cand.tree_xy.query_ball_point(slice_pts[:, :2], r=cfg["contact_xy"])
    lens = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(nb))
    if lens.sum() == 0:
        return 0.0
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in nb if x])
    owner = np.repeat(np.arange(len(nb)), lens)
    dz = slice_pts[owner, 2] - cand.points[flat, 2]
    ok = (dz >= 0.0) & (dz <= cfg["contact_dz"])
    if floor_z is not None:
        ok &= cand.points[flat, 2] >= floor_z + cfg.get("support_min_height", 0.0)
    supported = np.zeros(len(nb), dtype=bool)
    supported[owner[ok]] = True
    return float(supported.mean())


def containment_fraction(child_pts: np.ndarray, cand: SupportCandidate) -> float:
    """Share of the child's points inside the candidate's AABB."""
    inside = np.all((child_pts >= cand.aabb_min) & (child_pts <= cand.aabb_max), axis=1)
    return float(inside.mean())


def infer_support(
    child_pts: np.ndarray,
    candidates: Iterable[SupportCandidate],
    floor_z: float,
    cfg: Mapping[str, Any],
    child_id: Hashable | None = None,
) -> tuple[Hashable | None, str | None, float]:
    """Geometric support parent of one thing (DESIGN §6.3).

    Rules: contact (>= `contact_min_frac` of the bottom-slice points rest on a candidate, whose
    supporting points lie >= `support_min_height` above the floor) ->
    `on`; containment (>= `contain_min_frac` of the points inside an **anchor's** AABB) -> `in`;
    the best score wins, an anchor beats a thing on ties; otherwise `floor` (`on`) if the
    child's bottom is < `floor_max_gap` above `floor_z`.

    Args:
        child_pts: (N, 3) points of the child, reference frame.
        candidates: Possible parents (anchors and things); `child_id` itself is skipped.
        floor_z: Floor height (`estimate_floor_z`).
        cfg: `cfg["relations"]`.
        child_id: Id of the child, to skip it among the candidates.

    Returns:
        `(parent id | FLOOR | None, "on" | "in" | None, score)`; score is the winning fraction
        (1.0 for the floor, 0.0 for no parent).
    """
    child_pts = np.asarray(child_pts, dtype=np.float64)
    zmin = float(child_pts[:, 2].min())
    top = zmin + cfg["bottom_slice"]
    slice_pts = child_pts[child_pts[:, 2] <= top]
    cmin, cmax = child_pts.min(axis=0), child_pts.max(axis=0)
    m = cfg["candidate_margin"]

    best: tuple[float, int, Hashable, str] | None = None  # (score, anchor_bonus, id, type)
    for c in candidates:
        if child_id is not None and c.id == child_id:
            continue
        if np.any(c.aabb_min[:2] - m > cmax[:2]) or np.any(c.aabb_max[:2] + m < cmin[:2]):
            continue
        if c.aabb_min[2] > top:  # entirely above the child's bottom: cannot support it
            continue
        is_anchor = int(c.level == "anchor")
        options = []
        if c.aabb_max[2] >= zmin - cfg["contact_dz"]:
            f = contact_fraction(slice_pts, c, cfg, floor_z)
            if f >= cfg["contact_min_frac"]:
                options.append((f, "on"))
        if is_anchor:
            f = containment_fraction(child_pts, c)
            if f >= cfg["contain_min_frac"]:
                options.append((f, "in"))
        for f, t in options:
            cand = (f, is_anchor, c.id, t)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
    if best is not None:
        return best[2], best[3], best[0]
    if zmin - floor_z < cfg["floor_max_gap"]:
        return FLOOR, "on", 1.0
    return None, None, 0.0


def still_supports(
    child_pts: np.ndarray,
    parent: SupportCandidate | str,
    old_type: str,
    floor_z: float,
    cfg: Mapping[str, Any],
) -> str | None:
    """Does a known parent still carry the child (same rules as `infer_support`)? (M5)

    Used by `update_graph` to keep a re-checked thing's old parent when the geometry still
    agrees, instead of re-picking among all candidates.

    Args:
        child_pts: (N, 3) child points at their current pose.
        parent: The old parent's `SupportCandidate` (current pose), or `FLOOR`.
        old_type: The old relation type ("on" / "in").
        floor_z, cfg: As in `infer_support` (`cfg["relations"]`).

    Returns:
        The relation type that holds ("on" / "in"; the old type is preferred), or None.
    """
    child_pts = np.asarray(child_pts, dtype=np.float64)
    zmin = float(child_pts[:, 2].min())
    if isinstance(parent, str):
        return "on" if parent == FLOOR and zmin - floor_z < cfg["floor_max_gap"] else None
    slice_pts = child_pts[child_pts[:, 2] <= zmin + cfg["bottom_slice"]]
    ok = []
    if parent.aabb_max[2] >= zmin - cfg["contact_dz"] and contact_fraction(slice_pts, parent, cfg, floor_z) >= cfg["contact_min_frac"]:
        ok.append("on")
    if parent.level == "anchor" and containment_fraction(child_pts, parent) >= cfg["contain_min_frac"]:
        ok.append("in")
    if not ok:
        return None
    return old_type if old_type in ok else ok[0]


def aabb_gap(amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Euclidean gap between two AABBs (0 if they overlap)."""
    d = np.maximum(0.0, np.maximum(amin - bmax, bmin - amax))
    return float(np.linalg.norm(d))
