"""Change ground truth per (reference R, rescan S) pair.

Lives in `eval/`: reads rescan annotations (semseg / PLY labels / 3DSSG) and the
`rigid / removed / nonrigid` lists, which hard rule 1 allows only here and in `oracle/`.

Status precedence per reference object (things + anchors only):
1. listed `removed` -> `removed`, or `dont_care` ("removed_but_present") if it is still in O_S;
2. `nonrigid` -> `dont_care` ("nonrigid");
3. in `rigid` -> `moved`; `dont_care` ("small_move") if its displacement < τ_eval_move;
   bare-int entries have no transform -> `moved` with displacement None (kept, but excluded
   from the displacement buckets);
4. in O_S -> `unchanged`;
5. otherwise `absent_unlisted` (not in the rescan annotation and not listed removed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.geom.transforms import apply
from lsg.graph.relations import GTSupport
from lsg.io.scan3r import Scan, ScanMeta

STATUSES = ("moved", "removed", "unchanged", "absent_unlisted", "dont_care")


@dataclass
class GTObject:
    """GT of one reference object (things and anchors only)."""

    id: int
    label: str
    nyu40: int
    level: str
    size: str  # "small" | "large"
    status: str
    reason: str | None = None  # why dont_care
    in_rigid: bool = False
    displacement: float | None = None  # [m] ‖M_ref·c_R − c_R‖, raw T_align
    M_ref: np.ndarray | None = None  # T_align @ T_rigid: motion in the reference frame
    centroid_ref: np.ndarray | None = None
    centroid_new: np.ndarray | None = None  # rescan instance centroid, mapped with raw T_align
    aabb_new_min: np.ndarray | None = None  # AABB of the rescan instance points (raw T_align)
    aabb_new_max: np.ndarray | None = None
    min_extent_ref: float | None = None  # smallest reference AABB extent [m] (thin objects, M7)


@dataclass
class GTNew:
    """A GT new object: rescan objectId not in the reference (things + anchors only)."""

    id: int
    label: str
    nyu40: int
    level: str
    size: str
    centroid: np.ndarray | None  # reference frame (raw T_align); None if no vertices
    aabb_min: np.ndarray | None
    aabb_max: np.ndarray | None


@dataclass
class GTLabels:
    """All change GT of one pair."""

    ref: str
    rescan: str
    objects: dict[int, GTObject]
    new: dict[int, GTNew]
    n_rigid_raw: int  # len(rigid list)
    n_removed_raw: int
    n_nonrigid_raw: int
    n_new_raw: int  # |O_S \ O_R| incl. structure
    n_rigid_bare: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def by_status(self, status: str) -> list[int]:
        """Reference object ids with a given status."""
        return sorted(i for i, o in self.objects.items() if o.status == status)

    def counts(self) -> dict[str, int]:
        """Per-status counts plus the raw counts."""
        out = {s: len(self.by_status(s)) for s in STATUSES}
        for r in ("removed_but_present", "nonrigid", "small_move"):
            out[f"dont_care_{r}"] = sum(o.reason == r for o in self.objects.values())
        out.update(
            new=len(self.new),
            rigid_raw=self.n_rigid_raw,
            rigid_bare=self.n_rigid_bare,
            removed_raw=self.n_removed_raw,
            nonrigid_raw=self.n_nonrigid_raw,
            new_raw=self.n_new_raw,
            moved_things_raw=sum(o.in_rigid and o.level == "thing" for o in self.objects.values()),
        )
        return out


def size_class(level: str, aabb_min: np.ndarray, aabb_max: np.ndarray, small_max_extent: float) -> str:
    """`small` = thing whose largest AABB extent < `small_max_extent`; else `large`."""
    if level == "thing" and float(np.max(aabb_max - aabb_min)) < small_max_extent:
        return "small"
    return "large"


def _nyu(oid: int, scan: Scan, objects: Mapping[int, Mapping[str, Any]]) -> int | None:
    """nyu40 of an object: 3DSSG first, then PLY majority; None if unknown."""
    o = objects.get(oid)
    if o is not None and "nyu40" in o:
        return int(o["nyu40"])
    inst = scan.instances.get(oid)
    return int(inst.nyu40) if inst is not None else None


def _ids(scan: Scan) -> set[int]:
    """O_R / O_S: objectIds of the semseg file (falls back to the PLY instances)."""
    return set(scan.semseg_labels) or set(scan.instances)


def gt_changes(
    sm: ScanMeta,
    R: Scan,
    S: Scan,
    objs_R: Mapping[int, Mapping[str, Any]],
    objs_S: Mapping[int, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> GTLabels:
    """Change GT of one pair (precedence in the module docstring).

    Args:
        sm: The rescan's metadata (`meta[S]`): raw `T_align` and the change lists.
        R: Reference scan (`load_scan`).
        S: Rescan (`load_scan`; annotation reader, allowed in `eval/`).
        objs_R: 3DSSG objects of R (for nyu40); may be empty.
        objs_S: 3DSSG objects of S.
        cfg: Full config; uses `cfg["eval"]` (`tau_eval_move`, `small_max_extent`).

    Returns:
        `GTLabels`. Geometry is in the reference frame (raw `T_align`, never the refined one).
    """
    ecfg = cfg["eval"]
    tau, small = float(ecfg["tau_eval_move"]), float(ecfg["small_max_extent"])
    T = sm.T_align
    OR, OS = _ids(R), _ids(S)
    rigid = {e.id: e for e in sm.rigid}
    removed, nonrigid = set(sm.removed), set(sm.nonrigid)

    objects: dict[int, GTObject] = {}
    for oid in sorted(OR):
        nyu = _nyu(oid, R, objs_R)
        if nyu is None:
            continue
        lv = node_level(nyu)
        if lv == "structure":
            continue
        inst = R.instances.get(oid)
        label = R.semseg_labels.get(oid) or (inst.label if inst else "") or str(objs_R.get(oid, {}).get("label", ""))
        size = size_class(lv, inst.aabb_min, inst.aabb_max, small) if inst else "large"
        o = GTObject(oid, label, nyu, lv, size, "unchanged", in_rigid=oid in rigid)
        o.centroid_ref = inst.centroid if inst else None
        o.min_extent_ref = float(np.min(inst.aabb_max - inst.aabb_min)) if inst else None
        s_inst = S.instances.get(oid)
        if s_inst is not None:
            o.centroid_new = apply(T, s_inst.centroid[None])[0]
            P_new = apply(T, S.instance_points(oid))
            if len(P_new):
                o.aabb_new_min, o.aabb_new_max = P_new.min(axis=0), P_new.max(axis=0)
        if oid in rigid and rigid[oid].T is not None:
            o.M_ref = T @ rigid[oid].T
            if inst is not None:
                c = inst.centroid
                o.displacement = float(np.linalg.norm(apply(o.M_ref, c[None])[0] - c))

        if oid in removed:
            o.status, o.reason = ("dont_care", "removed_but_present") if oid in OS else ("removed", None)
        elif oid in nonrigid:
            o.status, o.reason = "dont_care", "nonrigid"
        elif oid in rigid:
            if o.displacement is not None and o.displacement < tau:
                o.status, o.reason = "dont_care", "small_move"
            else:
                o.status = "moved"
        elif oid in OS:
            o.status = "unchanged"
        else:
            o.status = "absent_unlisted"
        objects[oid] = o

    new_ids = OS - OR
    new: dict[int, GTNew] = {}
    for oid in sorted(new_ids):
        nyu = _nyu(oid, S, objs_S)
        if nyu is None or node_level(nyu) == "structure":
            continue
        lv = node_level(nyu)
        P = S.instance_points(oid)
        if len(P):
            P = apply(T, P)
            c, lo, hi = P.mean(axis=0), P.min(axis=0), P.max(axis=0)
            size = size_class(lv, lo, hi, small)
        else:
            c = lo = hi = None
            size = "large"
        label = S.semseg_labels.get(oid) or str(objs_S.get(oid, {}).get("label", ""))
        new[oid] = GTNew(oid, label, nyu, lv, size, c, lo, hi)

    return GTLabels(
        ref=R.scan_id,
        rescan=S.scan_id,
        objects=objects,
        new=new,
        n_rigid_raw=len(sm.rigid),
        n_removed_raw=len(sm.removed),
        n_nonrigid_raw=len(sm.nonrigid),
        n_new_raw=len(new_ids),
        n_rigid_bare=sum(e.T is None for e in sm.rigid),
    )


PARENT_CHANGE_KINDS = ("same", "swapped", "gained", "lost", "none", "excluded")


def parent_change_kind(oid: int, gs_R: GTSupport, gs_S: GTSupport) -> str:
    """How a thing's GT support parent differs between reference and rescan 3DSSG.

    Returns:
        "excluded" (only structure parents in either scan), "none" (no parent in either),
        "gained" (no reference parent, rescan has one), "lost" (the reverse),
        "same" (parent sets intersect), or "swapped" (both present, disjoint).
    """
    if oid in gs_R.excluded or oid in gs_S.excluded:
        return "excluded"
    a, b = set(gs_R.parents.get(oid, {})), set(gs_S.parents.get(oid, {}))
    if not a and not b:
        return "none"
    if not a:
        return "gained"
    if not b:
        return "lost"
    return "same" if a & b else "swapped"


def gt_parent_changes(
    sm: ScanMeta, objs_R: Mapping[int, Mapping[str, Any]], gs_R: GTSupport, gs_S: GTSupport
) -> dict[int, str]:
    """Parent-change kind for every moved **thing** (all `rigid` ids, before τ_eval_move).

    This reproduces the expected count of "parent-changed moved things" (= gained + lost + swapped)
    and breaks it down; the parent-changed definition is fixed at M5.
    """
    out = {}
    for e in sm.rigid:
        o = objs_R.get(e.id)
        if o is None or node_level(int(o.get("nyu40", 0))) != "thing":
            continue
        out[e.id] = parent_change_kind(e.id, gs_R, gs_S)
    return out
