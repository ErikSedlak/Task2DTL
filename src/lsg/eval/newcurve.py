"""New-object precision / recall over the candidate-confidence threshold (CHANGE 2, `lsg newcurve`).

A finished perception run saves its graphs (`graphs/perception/<ref>_<rescan>.json`); with
candidate confidence on, every `new` node carries the confidence of its candidate. Raising
`p_new_min` only removes `new` nodes, so the curve is scored from the saved graphs without
rerunning perception: for each threshold, the new nodes with confidence >= threshold are
matched to the pair's GT new objects with the §5 perception rule (`metrics.match_new`) and the
TP / FP / FN are summed over pairs (micro P / R, as the `new` row of the results tables).

Caveat: the association step is not rerun. A candidate that was emitted at the run's threshold
stays out of association either way (only unmatched candidates become `new` nodes), so the
curve is exact for thresholds >= the run's `p_new_min`; lower thresholds would need a rerun.

Writes `outputs/<run>/newcurve/metrics.json` (hard rule 5). Lives in `eval/` (reads GT).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lsg.config import data_root
from lsg.eval.gt import gt_changes
from lsg.eval.metrics import match_new
from lsg.graph.model import Graph
from lsg.io.dssg import label_to_nyu40, load_objects
from lsg.io.scan3r import load_meta, load_scan
from lsg.runinfo import write_metrics


def thresholds_of(cfg: Mapping[str, Any], extra: Sequence[float] | None = None) -> list[float]:
    """The run's default `p_new_min` plus `m7.p_new_sweep` (or `extra`), sorted, unique."""
    base = float(cfg["confidence"]["candidates"]["p_new_min"])
    sweep = list(extra) if extra is not None else list(cfg.get("m7", {}).get("p_new_sweep", []))
    return sorted({round(base, 6), *(round(float(t), 6) for t in sweep)})


def curve_counts(pred_new_by_pair: Mapping[str, Sequence[Any]], gts: Mapping[str, Any], cfg: Mapping[str, Any],
                 thresholds: Sequence[float], label_nyu: Mapping[str, int] | None = None,
                 scores: Mapping[str, Sequence[float | None]] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """TP / FP / FN / P / R / F1 per threshold over pairs, labelled and class-agnostic (`ag_*`).

    A node's score is its `confidence`, or `scores[pair][i]` when given (CHANGE 3 Part E: detector newness);
    nodes without a score always count.
    """
    rows = []
    for t in thresholds:
        c = {"tp": 0, "fp": 0, "fn": 0, "ag_tp": 0, "ag_fp": 0, "ag_fn": 0}
        for key, nodes in pred_new_by_pair.items():
            sc = scores[key] if scores is not None else [n.confidence for n in nodes]
            keep = [n for n, s in zip(nodes, sc) if s is None or float(s) >= t]
            for pre, lab in (("", True), ("ag_", False)):
                m = match_new(keep, gts[key], "perception", cfg, label_nyu=label_nyu, use_label=lab)
                for k in ("tp", "fp", "fn"):
                    c[pre + k] += len(m[k])
        row = {"p_new_min": float(t), **c, "n_new": c["tp"] + c["fp"]}
        for pre in ("", "ag_"):
            tp, fp, fn = c[pre + "tp"], c[pre + "fp"], c[pre + "fn"]
            P = tp / (tp + fp) if tp + fp else None
            R = tp / (tp + fn) if tp + fn else None
            row[pre + "precision"], row[pre + "recall"] = P, R
            row[pre + "f1"] = 2 * P * R / (P + R) if P and R else 0.0
        rows.append(row)
    return rows


def new_curve(cfg: dict[str, Any], run_dir: str | Path, thresholds: Sequence[float] | None = None,
              score: str = "confidence") -> dict[str, Any]:  # fmt: skip
    """Score the saved perception graphs of `run_dir` at every threshold; write `newcurve/metrics.json`.

    `score`: `confidence` (the new node's) or `newness` (CHANGE 3 Part E: the detector instance's newness, read
    from `candidates/<pair>.json` by the node's `obs_token`; the run must have used `p_new_min` 0).
    """
    import json
    run_dir = Path(run_dir)
    files = sorted((run_dir / "graphs" / "perception").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no perception graphs in {run_dir / 'graphs' / 'perception'}")
    root = data_root(cfg)
    meta = load_meta(root)
    label_nyu = label_to_nyu40(root)
    ths = thresholds_of(cfg, thresholds)
    preds: dict[str, list[Any]] = {}
    scores: dict[str, list[float | None]] | None = {} if score == "newness" else None
    gts: dict[str, Any] = {}
    cache_R: dict[str, Any] = {}
    for f in files:
        ref, rescan = f.stem.split("_", 1)
        if ref not in cache_R:
            cache_R = {ref: (load_scan(root, ref, with_intrinsics=False), load_objects(root, ref))}
        R, objs_R = cache_R[ref]
        S = load_scan(root, rescan, with_intrinsics=False)
        gts[f.stem] = gt_changes(meta[rescan], R, S, objs_R, load_objects(root, rescan), cfg)
        preds[f.stem] = [n for n in Graph.load(f).nodes if n.state == "new"]
        if scores is not None:
            cand = json.loads((run_dir / "candidates" / f"{f.stem}.json").read_text(encoding="utf-8"))["candidates"]
            nw = {c["token"]: c.get("newness") for c in cand}
            scores[f.stem] = [nw.get(n.obs_token) for n in preds[f.stem]]
    rows = curve_counts(preds, gts, cfg, ths, label_nyu, scores)
    out = {"run": str(run_dir), "n_pairs": len(files), "score": score, "curve": rows}
    write_metrics(run_dir / "newcurve", out, cfg)
    return out


def format_curve(res: Mapping[str, Any]) -> str:
    """Plain-text table of `new_curve`'s result."""
    L = [f"new-object P / R over p_new_min ({res['n_pairs']} pairs, {res['run']})",
         "  p_new_min   n_new    TP    FP    FN       P       R"]  # fmt: skip
    for r in res["curve"]:
        p = "   -  " if r["precision"] is None else f"{100 * r['precision']:6.1f}"
        rc = "   -  " if r["recall"] is None else f"{100 * r['recall']:6.1f}"
        L.append(f"  {r['p_new_min']:9.4f} {r['n_new']:7d} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}  {p}  {rc}")
    return "\n".join(L)
