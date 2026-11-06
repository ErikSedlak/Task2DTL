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

from lsg.eval.gt import GTLabels
from lsg.graph.model import Graph, Node, ref_node_id

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


def perception_compatible(n: Node, g: GTBox, cfg: Mapping[str, Any], label_nyu=None) -> tuple[bool, float, float]:
    """§5 perception rule for one (prediction, GT instance) pair.

    Compatible if (same label or same nyu40) and (centroid distance <= `match_new_dist` or
    AABB IoU >= `match_new_iou`).

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
    ok = same_class and (d <= float(ecfg["match_new_dist"]) or iou >= float(ecfg["match_new_iou"]))
    return ok, iou, d


def greedy_perception_match(
    preds: Sequence[Node], gts: Sequence[GTBox], cfg: Mapping[str, Any], label_nyu=None
) -> dict[str, int]:
    """One-to-one matching of predictions to GT instances, greedy by IoU (then distance).

    Returns:
        `{pred node id: GT id}`.
    """
    cand = []
    for n in preds:
        for g in gts:
            ok, iou, d = perception_compatible(n, g, cfg, label_nyu)
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
        tp = greedy_perception_match(pred_new, _gt_new_boxes(gt), cfg, label_nyu)
        rest = [n for n in pred_new if n.id not in tp]
        dc = _dont_care_boxes(gt)
        ignored = [n.id for n in rest if any(perception_compatible(n, g, cfg, label_nyu)[0] for g in dc)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    ign = set(ignored)
    fp = [n.id for n in pred_new if n.id not in tp and n.id not in ign]
    fn = sorted(set(gt.new) - set(tp.values()))
    return {"tp": tp, "fp": fp, "ignored": sorted(ign), "fn": fn}


# --------------------------------------------------------------------------------------------
# per-pair counts
# --------------------------------------------------------------------------------------------


def count_keys(cfg: Mapping[str, Any]) -> list[str]:
    """All count names `score_pair` emits (fixed order; used for the bootstrap matrix)."""
    nb = len(cfg["eval"]["disp_buckets"])
    keys = [f"{k}_{x}" for k in ("moved", "removed", "new") for x in ("tp", "fp", "fn")]
    keys += ["new_ignored", "unch_n", "unch_strict", "unch_kept", "absent_n", "absent_del", "absent_move",
             "ref_n", "ref_unknown", "ref_no_node", "assoc_n", "assoc_strict", "assoc_label",
             "moved_no_disp"]  # fmt: skip
    keys += [f"moved_{x}_d{b}" for b in range(nb) for x in ("n", "tp")]
    keys += [f"{k}_{x}_{s}" for k in ("moved", "new") for s in SIZES for x in ("n", "tp")]
    return keys


def score_pair(
    G_upd: Graph,
    gt: GTLabels,
    mode: str,
    cfg: Mapping[str, Any],
    obs_truth: Mapping[str, tuple[int, str]] | None = None,
    visible: set[int] | None = None,
    label_nyu: Mapping[str, int] | None = None,
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

    Returns:
        `{"counts": {name: int}, "loc_errors": [m], "new_match": match_new(...)}`.
    """
    ecfg = cfg["eval"]
    edges = [float(b) for b in ecfg["disp_buckets"]]
    truth = obs_truth or {}
    vis = set(visible) if visible is not None else {oid for oid, _ in truth.values()}
    c = dict.fromkeys(count_keys(cfg), 0)
    nodes = G_upd.node_map()
    loc: list[float] = []

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
        elif st == "removed":
            c["removed_fp"] += 1
        if s == "unchanged":
            c["unch_n"] += 1
            c["unch_strict"] += st == "unchanged"
            c["unch_kept"] += st in ("unchanged", "unknown")
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
    matched = set(m["tp"].values())
    for gid, g in gt.new.items():
        c[f"new_n_{g.size}"] += 1
        c[f"new_tp_{g.size}"] += gid in matched
    return {"counts": {k: int(v) for k, v in c.items()}, "loc_errors": loc, "new_match": m}


# --------------------------------------------------------------------------------------------
# aggregation + bootstrap
# --------------------------------------------------------------------------------------------


def metric_defs(cfg: Mapping[str, Any]) -> dict[str, tuple[list[str], list[str], list[str]]]:
    """`{metric: (numerator keys, denominator keys, support keys)}`; value = Σnum / Σden."""
    d: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for k in ("moved", "removed", "new"):
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
    for b, name in enumerate(bucket_names(cfg["eval"]["disp_buckets"])):
        d[f"moved_R_disp{name}"] = ([f"moved_tp_d{b}"], [f"moved_n_d{b}"], [f"moved_n_d{b}"])
    for k in ("moved", "new"):
        for s in SIZES:
            d[f"{k}_R_{s}"] = ([f"{k}_tp_{s}"], [f"{k}_n_{s}"], [f"{k}_n_{s}"])
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
