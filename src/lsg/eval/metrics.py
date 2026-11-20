"""Change-detection scoring (DESIGN §5 matching, §7.1 metrics) for both observation modes.

Per pair, `score_pair` turns a predicted graph G_upd and the pair's `GTLabels` into **counts**
(TP / FP / FN, numerators / denominators) plus a list of localisation errors. `aggregate`
micro-averages the counts over a split and adds 95% bootstrap CIs over pairs.

Scoring definitions (Erik, DECISIONS 2026-11-06 M2):
- reference objects are scored if they are things/anchors with GT status moved / removed /
  unchanged; `dont_care` and `absent_unlisted` are excluded from moved P/R, removed P/R and
  unchanged accuracy;
- `absent_unlisted`: false-deletion rate (predicted removed) and false-move rate (predicted
  moved);
- unchanged: strict accuracy (predicted unchanged) and kept rate (unchanged or unknown);
- a predicted `unknown` counts as not removed and not moved;
- predicted new: TP if matched to a scored GT new; ignored if it corresponds to a `dont_care`
  reference object; FP otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lsg.change.visibility import has_free_evidence
from lsg.eval.gt import GTLabels
from lsg.eval.relations import relation_count_keys, relation_metric_defs
from lsg.graph.model import Graph, Node, bbox_dict, ref_node_id

SIZES = ("small", "large")


# --------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------


def aabb_iou(amin, amax, bmin, bmax) -> float:
    """3D IoU of two axis-aligned boxes (0 if disjoint or degenerate)."""
    amin, amax, bmin, bmax = (np.asarray(x, float) for x in (amin, amax, bmin, bmax))
    inter = np.prod(np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None))
    va, vb = np.prod(np.clip(amax - amin, 0, None)), np.prod(np.clip(bmax - bmin, 0, None))
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def bucket_names(edges: Sequence[float]) -> list[str]:
    """§7.1 displacement buckets from the config edges, e.g. `[0.1,0.3)`, `[0.3,1)`, `>=1`."""
    e = [float(x) for x in edges]
    fmt = lambda x: f"{x:g}"  # noqa: E731
    return [f"[{fmt(a)},{fmt(b)})" for a, b in zip(e, e[1:])] + [f">={fmt(e[-1])}"]


def bucket_of(d: float | None, edges: Sequence[float]) -> int | None:
    """Index of the bucket containing displacement `d`; None below the first edge or if unknown."""
    if d is None or d < edges[0]:
        return None
    k = int(np.searchsorted(np.asarray(edges, float), d, side="right")) - 1
    return min(k, len(edges) - 1)


# --------------------------------------------------------------------------------------------
# new-node matching (§5)
# --------------------------------------------------------------------------------------------


@dataclass
class GTBox:
    """A GT instance in the reference frame, as the perception matching rule sees it."""

    id: int
    label: str
    nyu40: int | None
    centroid: np.ndarray | None
    aabb_min: np.ndarray | None
    aabb_max: np.ndarray | None


def _pred_nyu(n: Node, label_nyu: Mapping[str, int] | None) -> int | None:
    """nyu40 of a predicted node: its own field, else label -> nyu40 from 3DSSG."""
    if n.nyu40 is not None:
        return int(n.nyu40)
    return (label_nyu or {}).get(n.label)


def perception_compatible(
    n: Node, g: GTBox, cfg: Mapping[str, Any], label_nyu=None, use_label: bool = True
) -> tuple[bool, float, float]:
    """§5 perception rule for one (prediction, GT instance) pair.

    Compatible if (same label or same nyu40) and (centroid distance <= `match_new_dist` or
    AABB IoU >= `match_new_iou`). `use_label=False` drops the class condition (class-agnostic
    candidate recall, M4).

    Returns:
        (compatible, IoU, centroid distance).
    """
    if g.centroid is None or n.centroid is None:
        return False, 0.0, float("inf")
    pn = _pred_nyu(n, label_nyu)
    same_class = n.label == g.label or (pn is not None and g.nyu40 is not None and pn == g.nyu40)
    d = float(np.linalg.norm(np.asarray(n.centroid, float) - g.centroid))
    iou = 0.0
    if n.bbox_3d is not None and g.aabb_min is not None:
        iou = aabb_iou(n.bbox_3d["min"], n.bbox_3d["max"], g.aabb_min, g.aabb_max)
    ecfg = cfg["eval"]
    near = d <= float(ecfg["match_new_dist"]) or iou >= float(ecfg["match_new_iou"])
    ok = (same_class or not use_label) and near
    return ok, iou, d


def greedy_perception_match(
    preds: Sequence[Node], gts: Sequence[GTBox], cfg: Mapping[str, Any], label_nyu=None, use_label: bool = True
) -> dict[str, int]:
    """One-to-one matching of predictions to GT instances, greedy by IoU (then distance).

    Returns:
        `{pred node id: GT id}`.
    """
    cand = []
    for n in preds:
        for g in gts:
            ok, iou, d = perception_compatible(n, g, cfg, label_nyu, use_label)
            if ok:
                cand.append((-iou, d, n.id, g.id))
    cand.sort()
    used_p: set[str] = set()
    used_g: set[int] = set()
    out: dict[str, int] = {}
    for _, _, pid, gid in cand:
        if pid in used_p or gid in used_g:
            continue
        used_p.add(pid)
        used_g.add(gid)
        out[pid] = gid
    return out


def _gt_new_boxes(gt: GTLabels) -> list[GTBox]:
    """Scored GT new objects as `GTBox`es."""
    return [GTBox(i, n.label, n.nyu40, n.centroid, n.aabb_min, n.aabb_max) for i, n in sorted(gt.new.items())]


def _dont_care_boxes(gt: GTLabels) -> list[GTBox]:
    """Rescan instances of `dont_care` reference objects (raw `T_align`)."""
    return [
        GTBox(i, o.label, o.nyu40, o.centroid_new, o.aabb_new_min, o.aabb_new_max)
        for i, o in sorted(gt.objects.items())
        if o.status == "dont_care" and o.centroid_new is not None
    ]


def match_new(
    pred_new: Sequence[Node],
    gt: GTLabels,
    mode: str,
    cfg: Mapping[str, Any],
    obs_truth: Mapping[str, tuple[int, str]] | None = None,
    label_nyu: Mapping[str, int] | None = None,
    use_label: bool = True,
) -> dict[str, Any]:
    """Match predicted new nodes to GT new objects with the mode's rule (§5).

    Args:
        pred_new: Predicted nodes with `state == "new"`.
        gt: The pair's GT.
        mode: "oracle" (by `obs_token` -> rescan objectId) or "perception" (label/nyu40 +
            centroid distance or IoU, greedy by IoU, one-to-one).
        cfg: Full config (`cfg["eval"]` thresholds).
        obs_truth: Oracle: `{token: (rescan objectId, label)}` (unshuffled, eval only).
        label_nyu: Perception: label -> nyu40 fallback.
        use_label: Perception: False drops the class condition (class-agnostic new P/R, CHANGE 3
            Part N); oracle matching is by instance and does not use labels.

    Returns:
        `{"tp": {pred id: GT id}, "fp": [pred ids], "ignored": [pred ids], "fn": [GT ids]}`.
    """
    tp: dict[str, int] = {}
    ignored: list[str] = []
    if mode == "oracle":
        truth = obs_truth or {}
        for n in pred_new:
            oid = truth.get(n.obs_token or "", (None, ""))[0]
            if oid is not None and oid in gt.new and oid not in tp.values():
                tp[n.id] = oid
            elif oid is not None and oid in gt.objects and gt.objects[oid].status == "dont_care":
                ignored.append(n.id)
    elif mode == "perception":
        tp = greedy_perception_match(pred_new, _gt_new_boxes(gt), cfg, label_nyu, use_label)
        rest = [n for n in pred_new if n.id not in tp]
        dc = _dont_care_boxes(gt)
        ignored = [n.id for n in rest if any(perception_compatible(n, g, cfg, label_nyu, use_label)[0] for g in dc)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    ign = set(ignored)
    fp = [n.id for n in pred_new if n.id not in tp and n.id not in ign]
    fn = sorted(set(gt.new) - set(tp.values()))
    return {"tp": tp, "fp": fp, "ignored": sorted(ign), "fn": fn}


def obs_node(o: Any) -> Node:
    """An `ObsInstance` as a pseudo graph node (for the §5 matching rule)."""
    return Node(id=o.token, level="thing", label=o.label, nyu40=o.nyu40,
                bbox_3d=bbox_dict(o.aabb_min, o.aabb_max), centroid=[float(v) for v in o.centroid])  # fmt: skip


def _box(i: int, o: Any) -> GTBox:
    """`GTBox` of a reference object's rescan instance."""
    return GTBox(i, o.label, o.nyu40, o.centroid_new, o.aabb_new_min, o.aabb_new_max)


def gt_rescan_boxes(gt: GTLabels) -> list[GTBox]:
    """Every scored GT rescan instance in the reference frame: GT new objects plus the rescan
    instances of reference things/anchors (any status). Ids are rescan objectIds."""
    return _gt_new_boxes(gt) + [_box(i, o) for i, o in sorted(gt.objects.items()) if o.centroid_new is not None]


def perception_obs_truth(
    obs: Sequence[Any], gt: GTLabels, cfg: Mapping[str, Any], label_nyu: Mapping[str, int] | None = None
) -> dict[str, tuple[int, str]]:
    """Perception `obs_truth`: candidate token -> (GT rescan objectId, GT label) by the §5 rule.

    One-to-one, greedy by IoU, against GT new objects and the rescan instances of reference
    objects. Feeds the association metric (a reference node's `obs_token` is correct if its
    candidate matches the node's own rescan instance).
    """
    boxes = gt_rescan_boxes(gt)
    lab = {b.id: b.label for b in boxes}
    m = greedy_perception_match([obs_node(o) for o in obs if len(o.points)], boxes, cfg, label_nyu)
    return {t: (gid, lab[gid]) for t, gid in m.items()}


CAND_KEYS = ("diag_cand", "diag_cand_n", "diag_cand_new_n", "diag_cand_new_hit", "diag_cand_new_lab",
             "diag_cand_moved_n", "diag_cand_moved_hit", "diag_cand_moved_lab", "diag_cand_explained",
             "diag_cand_app_n", "diag_cand_app_explained", "diag_cand_multi", "diag_cand_multi_n")  # fmt: skip


def candidate_counts(
    obs: Sequence[Any], gt: GTLabels, cfg: Mapping[str, Any], label_nyu: Mapping[str, int] | None = None
) -> dict[str, int]:
    """Candidate diagnostics of one pair (perception, M4; method-independent).

    Targets = GT new objects and the rescan instances of GT-moved objects ("moved-to").
    - `*_hit`: targets matched by a candidate, class-agnostic (centroid <= `match_new_dist` or
      IoU >= `match_new_iou`), one-to-one greedy over all targets jointly;
    - `*_lab`: the same with the §5 label condition;
    - `diag_cand_explained`: candidates within the geometric gate of *any* GT rescan instance
      (many-to-one); the rest are unexplained (people, mesh holes, structure, unannotated);
    - `diag_cand_multi` (CHANGE 3 Part N): GT new objects whose centroid lies inside a candidate
      AABB (+ `MULTI_MARGIN`) that holds >= 2 GT-new centroids (merged "blobs");
      `diag_cand_multi_n`: such candidates.
    """
    nodes = [obs_node(o) for o in obs if len(o.points)]
    new = _gt_new_boxes(gt)
    moved = [_box(i, o) for i, o in sorted(gt.objects.items()) if o.status == "moved" and o.centroid_new is not None]
    new_ids = {b.id for b in new}
    c = {"diag_cand": 1, "diag_cand_n": len(nodes), "diag_cand_new_n": len(new), "diag_cand_moved_n": len(moved)}
    for suffix, use_label in (("hit", False), ("lab", True)):
        hit = set(greedy_perception_match(nodes, new + moved, cfg, label_nyu, use_label).values())
        c[f"diag_cand_new_{suffix}"] = len(hit & new_ids)
        c[f"diag_cand_moved_{suffix}"] = len(hit - new_ids)
    allb = gt_rescan_boxes(gt)
    expl = [any(perception_compatible(n, b, cfg, None, use_label=False)[0] for b in allb) for n in nodes]
    c["diag_cand_explained"] = sum(expl)
    kinds = [getattr(o, "kind", "depth") for o in obs if len(o.points)]
    c["diag_cand_app_n"] = sum(k == "appearance" for k in kinds)
    c["diag_cand_app_explained"] = sum(e for e, k in zip(expl, kinds) if k == "appearance")
    c["diag_cand_multi"], c["diag_cand_multi_n"] = multi_object_candidates(nodes, new)
    return c


MULTI_MARGIN = 0.05  # [m] AABB margin of the blob diagnostic (eval only)


def multi_object_candidates(nodes: Sequence[Node], new: Sequence[GTBox]) -> tuple[int, int]:
    """(GT new objects inside a candidate that holds >= 2 GT-new centroids, number of such candidates)."""
    C = np.array([b.centroid for b in new if b.centroid is not None], float).reshape(-1, 3)
    if not len(C):
        return 0, 0
    inside_any = np.zeros(len(C), bool)
    n_multi = 0
    for n in nodes:
        assert n.bbox_3d is not None
        lo = np.asarray(n.bbox_3d["min"], float) - MULTI_MARGIN
        hi = np.asarray(n.bbox_3d["max"], float) + MULTI_MARGIN
        m = np.all((C >= lo) & (C <= hi), axis=1)
        if m.sum() >= 2:
            n_multi += 1
            inside_any |= m
    return int(inside_any.sum()), n_multi


# --------------------------------------------------------------------------------------------
# per-pair counts
# --------------------------------------------------------------------------------------------


# M6 diagnostics: appearance free-like evidence, thin nodes, confidence means (p x 1000 sums over
# 1000 x n, so the integer bootstrap machinery gives the mean p)
THIN_CONF_KEYS = ("diag_freelike_removed", "diag_freelike_unch", "diag_thin_removed_n", "diag_thin_removed_tp",
                  "diag_thin_removed_free", "diag_thin_unch_n", "diag_thin_unch_kept",
                  *(f"diag_conf_{g}_{x}" for g in ("unch", "removed", "absent") for x in ("sum", "den", "n")))  # fmt: skip


def count_keys(cfg: Mapping[str, Any]) -> list[str]:
    """All count names `score_pair` emits (fixed order; used for the bootstrap matrix)."""
    nb = len(cfg["eval"]["disp_buckets"])
    keys = [f"{k}_{x}" for k in ("moved", "removed", "new", "new_ag") for x in ("tp", "fp", "fn")]
    keys += ["new_ignored", "new_ag_ignored", "unch_n", "unch_strict", "unch_kept", "absent_n", "absent_del", "absent_move",
             "ref_n", "ref_unknown", "ref_no_node", "assoc_n", "assoc_strict", "assoc_label",
             "moved_no_disp", "diag_free_unch", "diag_free_removed", "diag_free_absent",
             "diag_seen_absent", "diag_votes", "diag_free_unch_removed", "graph_invalid", *CAND_KEYS,
             *THIN_CONF_KEYS]  # fmt: skip
    keys += [f"moved_{x}_d{b}" for b in range(nb) for x in ("n", "tp")]
    keys += [f"{k}_{x}_{s}" for k in ("moved", "new") for s in SIZES for x in ("n", "tp")]
    keys += relation_count_keys()  # §7.2 (M5); 0 when relations were not scored
    return keys


def score_pair(
    G_upd: Graph,
    gt: GTLabels,
    mode: str,
    cfg: Mapping[str, Any],
    obs_truth: Mapping[str, tuple[int, str]] | None = None,
    visible: set[int] | None = None,
    label_nyu: Mapping[str, int] | None = None,
    votes: Mapping[str, Any] | None = None,
    candidates: Sequence[Any] | None = None,
    details: bool = False,
) -> dict[str, Any]:
    """Counts of one pair (module docstring for the definitions).

    Args:
        G_upd: Predicted graph; reference nodes `r<id>` carry `state` (and `obs_token` when
            associated), new nodes have `state == "new"`.
        gt: The pair's GT.
        mode: "oracle" or "perception".
        cfg: Full config.
        obs_truth: `{token: (rescan objectId, label)}` for tokens that correspond to a GT
            rescan instance. Oracle: the unshuffled token map; perception (M4): from the §5 rule.
        visible: Rescan objectIds visible in the processed frames (GT-matchable for the
            association metric). Default: the ids in `obs_truth`.
        label_nyu: Perception: label -> nyu40 fallback for the matching rule.
        votes: Online-loop vote tallies `{node id: NodeVotes}` (M3). Gives the method-
            independent evidence diagnostics: GT-unchanged / removed / absent-unlisted
            objects with >= `states.k_free` counted free votes, and absent-unlisted objects
            with any counted seen vote (`diag_votes` = 1 if votes were given).
        candidates: Perception `ObsInstance`s; adds the `candidate_counts` diagnostics.
        details: Also return per-object rows (`"objects"`: GT status, predicted state, confidence,
            counted votes, associated token's GT instance; `"new"`: per predicted new node its
            outcome). For the §7.4 audit (M8); the counts do not depend on it.

    Returns:
        `{"counts": {name: int}, "loc_errors": [m], "new_match": match_new(...)}` (+ `objects`, `new`).
    """
    ecfg = cfg["eval"]
    edges = [float(b) for b in ecfg["disp_buckets"]]
    truth = obs_truth or {}
    vis = set(visible) if visible is not None else {oid for oid, _ in truth.values()}
    c = dict.fromkeys(count_keys(cfg), 0)
    nodes = G_upd.node_map()
    loc: list[float] = []
    vts = votes or {}
    c["diag_votes"] = int(votes is not None)

    def free_ev(oid: int) -> bool:
        """§3 free evidence of a reference object (counted free + free-like votes; M6)."""
        return has_free_evidence(vts.get(ref_node_id(oid)), cfg)

    def free_like_ev(oid: int) -> bool:
        """>= k_free_like counted appearance free-like votes (thin nodes, M6)."""
        v = vts.get(ref_node_id(oid))
        return v is not None and v.free_like >= int(cfg.get("states", {}).get("k_free_like", 3))

    thin_max = float(cfg.get("appearance", {}).get("thin_max_extent", 0.15))

    def is_thin(n: Node, o: Any) -> bool:
        """Thin thing by its *reference* AABB (M7: the same objects in both modes, also when moved)."""
        if n.level != "thing":
            return False
        ext = getattr(o, "min_extent_ref", None)
        if ext is None:
            if n.bbox_3d is None:
                return False
            ext = float((np.asarray(n.bbox_3d["max"], float) - np.asarray(n.bbox_3d["min"], float)).min())
        return float(ext) < thin_max

    def conf(group: str, n: Node) -> None:
        if n.confidence is not None:
            c[f"diag_conf_{group}_sum"] += int(round(1000 * float(n.confidence)))
            c[f"diag_conf_{group}_den"] += 1000
            c[f"diag_conf_{group}_n"] += 1

    for oid, o in sorted(gt.objects.items()):
        n = nodes.get(ref_node_id(oid))
        if n is None:
            c["ref_no_node"] += 1
            continue
        st = n.state
        c["ref_n"] += 1
        c["ref_unknown"] += st == "unknown"
        s = o.status
        if s == "absent_unlisted":
            c["absent_n"] += 1
            c["absent_del"] += st == "removed"
            c["absent_move"] += st == "moved"
            c["diag_free_absent"] += free_ev(oid)
            v = vts.get(ref_node_id(oid))
            c["diag_seen_absent"] += v is not None and v.seen > 0
            conf("absent", n)
            continue
        if s == "dont_care":
            continue
        # moved / removed P/R over {moved, removed, unchanged}
        if s == "moved":
            hit = st == "moved"
            c["moved_tp" if hit else "moved_fn"] += 1
            c[f"moved_n_{o.size}"] += 1
            c[f"moved_tp_{o.size}"] += hit
            b = bucket_of(o.displacement, edges)
            if b is None:
                c["moved_no_disp"] += 1
            else:
                c[f"moved_n_d{b}"] += 1
                c[f"moved_tp_d{b}"] += hit
            if hit and o.centroid_new is not None and n.centroid is not None:
                # rounded to 1 µm: fresh multi-threaded ICP differs at ~1e-15 m between runs (§7.5)
                loc.append(round(float(np.linalg.norm(np.asarray(n.centroid, float) - o.centroid_new)), 6))
        elif st == "moved":
            c["moved_fp"] += 1
        if s == "removed":
            c["removed_tp" if st == "removed" else "removed_fn"] += 1
            c["diag_free_removed"] += free_ev(oid)
            c["diag_freelike_removed"] += free_like_ev(oid)
            conf("removed", n)
            if is_thin(n, o):
                c["diag_thin_removed_n"] += 1
                c["diag_thin_removed_tp"] += st == "removed"
                c["diag_thin_removed_free"] += free_ev(oid)
        elif st == "removed":
            c["removed_fp"] += 1
        if s == "unchanged":
            c["unch_n"] += 1
            c["unch_strict"] += st == "unchanged"
            c["unch_kept"] += st in ("unchanged", "unknown")
            c["diag_free_unch"] += free_ev(oid)
            c["diag_free_unch_removed"] += free_ev(oid) and st == "removed"
            c["diag_freelike_unch"] += free_like_ev(oid)
            conf("unch", n)
            if is_thin(n, o):
                c["diag_thin_unch_n"] += 1
                c["diag_thin_unch_kept"] += st in ("unchanged", "unknown")
        # association: GT-matchable = moved/unchanged objects visible in the processed frames
        if s in ("moved", "unchanged") and oid in vis:
            c["assoc_n"] += 1
            t = truth.get(n.obs_token or "") if n.obs_token else None
            if t is not None:
                c["assoc_strict"] += t[0] == oid
                c["assoc_label"] += t[1] == o.label
    pred_new = [n for n in G_upd.nodes if n.state == "new"]
    m = match_new(pred_new, gt, mode, cfg, truth, label_nyu)
    c["new_tp"], c["new_fp"], c["new_fn"] = len(m["tp"]), len(m["fp"]), len(m["fn"])
    c["new_ignored"] = len(m["ignored"])
    # class-agnostic new P/R (CHANGE 3 Part N): perception without the label condition; oracle = labelled
    m_ag = match_new(pred_new, gt, mode, cfg, truth, label_nyu, use_label=False) if mode == "perception" else m
    c["new_ag_tp"], c["new_ag_fp"], c["new_ag_fn"] = len(m_ag["tp"]), len(m_ag["fp"]), len(m_ag["fn"])
    c["new_ag_ignored"] = len(m_ag["ignored"])
    matched = set(m["tp"].values())
    for gid, g in gt.new.items():
        c[f"new_n_{g.size}"] += 1
        c[f"new_tp_{g.size}"] += gid in matched
    if candidates is not None:
        c.update(candidate_counts(candidates, gt, cfg, label_nyu))
    out: dict[str, Any] = {"counts": {k: int(v) for k, v in c.items()}, "loc_errors": loc, "new_match": m}
    if details:
        out["objects"] = object_rows(G_upd, gt, truth, vts)
        if candidates is not None:
            out["candidates"] = candidate_detail_rows(candidates, gt, cfg)
        outcome = {**{i: "tp" for i in m["tp"]}, **{i: "fp" for i in m["fp"]}, **{i: "ignored" for i in m["ignored"]}}
        out["new"] = [{"id": n.id, "label": n.label, "outcome": outcome.get(n.id), "gt_id": m["tp"].get(n.id),
                       "confidence": n.confidence, "obs_token": n.obs_token,
                       "gt_token": list(truth[n.obs_token]) if n.obs_token in truth else None,
                       "bbox_3d": n.bbox_3d, "centroid": n.centroid} for n in pred_new]  # fmt: skip
    return out


def candidate_detail_rows(obs: Sequence[Any], gt: GTLabels, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per perception candidate: explained by any GT rescan instance (the `candidate_counts` gate), its
    cluster votes and confidence (CHANGE 3 Part C, `lsg chg3 votestats`), and its class `cls`
    (CHANGE 3 Part N): "new" = near a GT new object, "ref" = near only rescan instances of reference
    objects, "none" = unexplained."""
    allb = gt_rescan_boxes(gt)
    newb = _gt_new_boxes(gt)
    new_ids = {b.id for b in newb}
    rows = []
    for o in obs:
        if not len(o.points):
            continue
        n = obs_node(o)
        near = [b for b in allb if perception_compatible(n, b, cfg, None, use_label=False)[0]]
        cls = "new" if any(b.id in new_ids for b in near) else ("ref" if near else "none")
        rows.append({"token": o.token, "kind": getattr(o, "kind", "depth"), "confidence": o.confidence,
                     "cand_votes": None if getattr(o, "cand_votes", None) is None else [int(x) for x in o.cand_votes],
                     "explained": bool(near), "cls": cls})  # fmt: skip
    return rows


def object_rows(G_upd: Graph, gt: GTLabels, truth: Mapping[str, tuple[int, str]],
                votes: Mapping[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """Per reference object: GT status vs prediction, with the evidence behind it (audit, M8)."""
    nodes = G_upd.node_map()
    rows = []
    for oid, o in sorted(gt.objects.items()):
        n = nodes.get(ref_node_id(oid))
        if n is None:
            continue
        v = votes.get(n.id)
        t = truth.get(n.obs_token) if n.obs_token else None
        rows.append({
            "oid": int(oid), "node": n.id, "label": o.label, "level": o.level, "size": o.size,
            "gt_status": o.status, "gt_reason": o.reason, "displacement": o.displacement,
            "pred_state": n.state, "confidence": n.confidence, "obs_token": n.obs_token,
            "obs_gt": list(t) if t is not None else None,
            "votes": None if v is None else {"seen": v.seen, "free": v.free, "free_like": v.free_like,
                                             "raw_seen": v.raw_seen, "raw_free": v.raw_free, "noinfo": v.noinfo},
            "bbox_3d": n.bbox_3d, "centroid": n.centroid,
            "centroid_ref": None if o.centroid_ref is None else [float(x) for x in o.centroid_ref],
            "gt_aabb_new": None if o.aabb_new_min is None else [list(map(float, o.aabb_new_min)),
                                                               list(map(float, o.aabb_new_max))],
        })  # fmt: skip
    return rows


# --------------------------------------------------------------------------------------------
# aggregation + bootstrap
# --------------------------------------------------------------------------------------------


def metric_defs(cfg: Mapping[str, Any]) -> dict[str, tuple[list[str], list[str], list[str]]]:
    """`{metric: (numerator keys, denominator keys, support keys)}`; value = Σnum / Σden."""
    d: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for k in ("moved", "removed", "new", "new_ag"):
        tp, fp, fn = f"{k}_tp", f"{k}_fp", f"{k}_fn"
        d[f"{k}_P"] = ([tp], [tp, fp], [tp, fp])
        d[f"{k}_R"] = ([tp], [tp, fn], [tp, fn])
        d[f"{k}_F1"] = ([tp, tp], [tp, tp, fp, fn], [tp, fn])
    d["unchanged_acc"] = (["unch_strict"], ["unch_n"], ["unch_n"])
    d["unchanged_kept"] = (["unch_kept"], ["unch_n"], ["unch_n"])
    d["false_deletion_rate"] = (["absent_del"], ["absent_n"], ["absent_n"])
    d["false_move_rate"] = (["absent_move"], ["absent_n"], ["absent_n"])
    d["unknown_rate"] = (["ref_unknown"], ["ref_n"], ["ref_n"])
    d["assoc_acc_strict"] = (["assoc_strict"], ["assoc_n"], ["assoc_n"])
    d["assoc_acc_same_label"] = (["assoc_label"], ["assoc_n"], ["assoc_n"])
    # online-loop evidence diagnostics (M3; independent of the method's decisions)
    d["free_ev_unchanged"] = (["diag_free_unch"], ["unch_n"], ["unch_n"])
    d["free_ev_removed"] = (["diag_free_removed"], ["removed_tp", "removed_fn"], ["removed_tp", "removed_fn"])
    d["free_ev_unchanged_removed"] = (["diag_free_unch_removed"], ["diag_free_unch"], ["diag_free_unch"])
    d["free_ev_absent"] = (["diag_free_absent"], ["absent_n"], ["absent_n"])
    d["seen_ev_absent"] = (["diag_seen_absent"], ["absent_n"], ["absent_n"])
    # appearance / thin nodes / confidence (M6)
    rem = ["removed_tp", "removed_fn"]
    d["free_like_ev_removed"] = (["diag_freelike_removed"], rem, rem)
    d["free_like_ev_unchanged"] = (["diag_freelike_unch"], ["unch_n"], ["unch_n"])
    d["removed_R_thin"] = (["diag_thin_removed_tp"], ["diag_thin_removed_n"], ["diag_thin_removed_n"])
    d["free_ev_removed_thin"] = (["diag_thin_removed_free"], ["diag_thin_removed_n"], ["diag_thin_removed_n"])
    d["unchanged_kept_thin"] = (["diag_thin_unch_kept"], ["diag_thin_unch_n"], ["diag_thin_unch_n"])
    for g, name in (("unch", "unchanged"), ("removed", "removed"), ("absent", "absent")):
        d[f"conf_mean_{name}"] = ([f"diag_conf_{g}_sum"], [f"diag_conf_{g}_den"], [f"diag_conf_{g}_n"])
    d["cand_appearance_per_pair"] = (["diag_cand_app_n"], ["diag_cand"], ["diag_cand_app_n"])
    d["cand_appearance_explained"] = (["diag_cand_app_explained"], ["diag_cand_app_n"], ["diag_cand_app_n"])
    # perception candidate diagnostics (M4)
    d["cand_per_pair"] = (["diag_cand_n"], ["diag_cand"], ["diag_cand_n"])
    d["cand_recall_new"] = (["diag_cand_new_hit"], ["diag_cand_new_n"], ["diag_cand_new_n"])
    d["cand_recall_moved"] = (["diag_cand_moved_hit"], ["diag_cand_moved_n"], ["diag_cand_moved_n"])
    d["cand_recall_new_labelled"] = (["diag_cand_new_lab"], ["diag_cand_new_n"], ["diag_cand_new_n"])
    d["cand_recall_moved_labelled"] = (["diag_cand_moved_lab"], ["diag_cand_moved_n"], ["diag_cand_moved_n"])
    d["cand_explained"] = (["diag_cand_explained"], ["diag_cand_n"], ["diag_cand_n"])
    d["cand_new_in_blob"] = (["diag_cand_multi"], ["diag_cand_new_n"], ["diag_cand_new_n"])
    d["cand_blobs_per_pair"] = (["diag_cand_multi_n"], ["diag_cand"], ["diag_cand_multi_n"])
    for b, name in enumerate(bucket_names(cfg["eval"]["disp_buckets"])):
        d[f"moved_R_disp{name}"] = ([f"moved_tp_d{b}"], [f"moved_n_d{b}"], [f"moved_n_d{b}"])
    for k in ("moved", "new"):
        for s in SIZES:
            d[f"{k}_R_{s}"] = ([f"{k}_tp_{s}"], [f"{k}_n_{s}"], [f"{k}_n_{s}"])
    d.update(relation_metric_defs())
    return d


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den with NaN where den == 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def aggregate(per_pair: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Micro-averaged metrics over pairs with 95% bootstrap CIs (resampling pairs).

    Args:
        per_pair: `score_pair` outputs (need `counts` and `loc_errors`).
        cfg: Full config (`eval.bootstrap_n`, `eval.bootstrap_seed`).

    Returns:
        `{"n_pairs", "counts": summed counts, "metrics": {name: {"value", "ci": [lo, hi],
        "n"}}, "loc_error": {"n", "median", "p90"}}`. Values are None when undefined (n = 0).
    """
    keys = count_keys(cfg)
    C = np.array([[p["counts"][k] for k in keys] for p in per_pair], float).reshape(len(per_pair), len(keys))
    col = {k: i for i, k in enumerate(keys)}
    tot = C.sum(axis=0)
    ecfg = cfg["eval"]
    nb = int(ecfg["bootstrap_n"])
    rng = np.random.default_rng(int(ecfg["bootstrap_seed"]))
    P = len(per_pair)
    boot = C[rng.integers(0, P, size=(nb, P))].sum(axis=1) if P else np.zeros((0, len(keys)))

    def val(S: np.ndarray, num: list[str], den: list[str]) -> np.ndarray:
        """Metric value for summed counts S (..., K)."""
        return _ratio(sum(S[..., col[k]] for k in num), sum(S[..., col[k]] for k in den))

    def nn(x: float) -> float | None:
        """NaN -> None, so metrics.json stays valid JSON (`null` = undefined, n = 0)."""
        return None if x != x else float(x)

    out: dict[str, Any] = {}
    for name, (num, den, sup) in metric_defs(cfg).items():
        v = float(val(tot, num, den))
        if P and nb:
            bs = val(boot, num, den)
            bs = bs[np.isfinite(bs)]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs.size else [float("nan")] * 2
        else:
            ci = [float("nan")] * 2
        out[name] = {"value": nn(v), "ci": [nn(ci[0]), nn(ci[1])], "n": int(sum(tot[col[k]] for k in sup))}
    errs = np.array([e for p in per_pair for e in p["loc_errors"]], float)
    loc = {"n": int(errs.size),
           "median": float(np.median(errs)) if errs.size else None,
           "p90": float(np.percentile(errs, 90)) if errs.size else None}  # fmt: skip
    return {"n_pairs": P, "counts": {k: int(tot[col[k]]) for k in keys}, "metrics": out, "loc_error": loc}
