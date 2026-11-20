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
                 thresholds: Sequence[float], label_nyu: Mapping[str, int] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """TP / FP / FN / P / R per threshold over pairs (nodes without confidence always count)."""
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for key, nodes in pred_new_by_pair.items():
            keep = [n for n in nodes if n.confidence is None or float(n.confidence) >= t]
            m = match_new(keep, gts[key], "perception", cfg, label_nyu=label_nyu)
            tp += len(m["tp"])
            fp += len(m["fp"])
            fn += len(m["fn"])
        rows.append({"p_new_min": float(t), "tp": tp, "fp": fp, "fn": fn, "n_new": tp + fp,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None})  # fmt: skip
    return rows


def new_curve(cfg: dict[str, Any], run_dir: str | Path, thresholds: Sequence[float] | None = None) -> dict[str, Any]:
    """Score the saved perception graphs of `run_dir` at every threshold; write `newcurve/metrics.json`."""
    run_dir = Path(run_dir)
    files = sorted((run_dir / "graphs" / "perception").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no perception graphs in {run_dir / 'graphs' / 'perception'}")
    root = data_root(cfg)
    meta = load_meta(root)
    label_nyu = label_to_nyu40(root)
    ths = thresholds_of(cfg, thresholds)
    preds: dict[str, list[Any]] = {}
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
    rows = curve_counts(preds, gts, cfg, ths, label_nyu)
    out = {"run": str(run_dir), "n_pairs": len(files), "curve": rows}
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
