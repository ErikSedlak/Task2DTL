"""CHANGE 3 dev_tune sweeps (`lsg chg3 parta`): variants of a rule, both modes, a pre-written selection rule.

Part A: every variant in `chg3.part_a.variants`
is one `run_eval` of FULL on `splits/dev_tune.txt` (both modes) into `outputs/<run>/<variant>`. The
variant `off` (rule disabled) is the baseline of the selection rule (`chg3.part_a.criterion`):

- a variant passes if, in **both** modes, its false-move rate is at most `false_move_frac` x the
  baseline's and its moved F1 is at least the baseline's minus `f1_drop`;
- among passing variants the highest mean moved F1 over the two modes wins; ties go to the variant
  listed first (the list runs from simple to complex);
- if none passes, `winner` is None (report and ask Erik).

Tuning set only (hard rule 4); writes `outputs/<run>/metrics.json` (hard rule 5).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.config import outputs_root
from lsg.eval.ablations import apply_overrides
from lsg.eval.harness import MODES, run_split
from lsg.runinfo import new_run_id, write_metrics

KEYS = ("moved_P", "moved_R", "moved_F1", "false_move_rate", "false_deletion_rate", "unchanged_acc", "unchanged_kept",
        "removed_P", "removed_R", "removed_F1", "new_P", "new_R", "unknown_rate", "moved_R_disp[0.1,0.3)", "moved_R_disp[0.3,1)",
        "moved_R_disp>=1", "new_F1", "new_ag_P", "new_ag_R", "new_ag_F1", "cand_per_pair", "cand_recall_new",
        "cand_new_in_blob")  # fmt: skip
CAND_COUNTS = ("diag_cand_n", "diag_cand_new_hit", "diag_cand_moved_hit", "diag_cand_multi", "diag_cand_multi_n", "new_tp",
               "new_fp", "new_ag_tp", "new_ag_fp")  # fmt: skip


def _pick(summary: Mapping[str, Any]) -> dict[str, Any]:
    M = summary["metrics"]
    C = summary["counts"]
    return ({k: M[k]["value"] for k in KEYS if k in M} | {"moved_fp": C["moved_fp"], "moved_tp": C["moved_tp"]}
            | {k: C[k] for k in CAND_COUNTS if k in C})  # fmt: skip


def select(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any], base: str = "off") -> dict[str, Any]:
    """Apply the selection rule to `{variant: {mode: picked metrics}}`."""
    b = rows[base]
    frac, drop = float(crit["false_move_frac"]), float(crit["f1_drop"])
    verdict: dict[str, Any] = {}
    for name, r in rows.items():
        if name == base:
            continue
        ok = all(r[m]["false_move_rate"] is not None and r[m]["moved_F1"] is not None
                 and r[m]["false_move_rate"] <= frac * b[m]["false_move_rate"] + 1e-12
                 and r[m]["moved_F1"] >= b[m]["moved_F1"] - drop - 1e-12 for m in MODES)  # fmt: skip
        verdict[name] = {"pass": bool(ok), "mean_moved_F1": sum(r[m]["moved_F1"] or 0.0 for m in MODES) / len(MODES)}
    passing = [n for n in rows if n in verdict and verdict[n]["pass"]]  # list order = simplicity order
    winner = max(passing, key=lambda n: (verdict[n]["mean_moved_F1"], -passing.index(n))) if passing else None
    return {"verdict": verdict, "winner": winner}


def run_variants(cfg: dict[str, Any], variants: Mapping[str, Mapping[str, Any]], rid: str, split: str,
                 workers: int | None = None, details: bool = False, resume: bool = False,
                 only: list[str] | None = None, modes: Sequence[str] = MODES) -> tuple[dict[str, Any], dict[str, str]]:  # fmt: skip
    """FULL on `split` (`modes`, default both) once per variant (dotted overrides) into `outputs/<rid>/<variant>`.

    `resume`: a variant whose `metrics.json` exists is read back instead of rerun. `only`: run just these variants (others are read back if present).
    """
    import json

    rows: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for name, ov in variants.items():
        mp = outputs_root(cfg) / rid / name / "metrics.json"
        if (resume and mp.exists()) or (only is not None and name not in only):
            if mp.exists():
                summ = json.loads(mp.read_text(encoding="utf-8"))["metrics"]["summary"]
                rows[name] = {m: _pick(summ[m]) for m in modes}
                paths[name] = str(mp)
            continue
        c = apply_overrides(cfg, ov)
        r = run_split(c, split, obs_modes=list(modes), method="FULL", run_id=f"{rid}/{name}", workers=workers,
                      details=details)  # fmt: skip
        rows[name] = {m: _pick(r["summary"][m]) for m in modes}
        paths[name] = r["metrics_path"]
    return rows, paths


def run_part_a(cfg: dict[str, Any], run_id: str | None = None, split: str = "dev_tune",
               workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """All Part A variants on the tuning set + the selection; returns the result dict."""
    t0 = time.time()
    pc = cfg["chg3"]["part_a"]
    rid = run_id or new_run_id("chg3_partA")
    rows, paths = run_variants(cfg, pc["variants"], rid, split, workers)
    sel = select(rows, pc["criterion"])
    res = {"split": split, "variants": {n: dict(v) for n, v in pc["variants"].items()}, "rows": rows,
           "criterion": dict(pc["criterion"]), **sel, "metrics_paths": paths}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / rid, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_a(res: Mapping[str, Any]) -> str:
    """Table: one line per variant and mode, then the verdicts and the winner."""
    cols = ("moved_F1", "moved_P", "moved_R", "false_move_rate", "unchanged_kept", "false_deletion_rate", "removed_R",
            "new_P", "new_R")  # fmt: skip
    head = f"{'variant':<10} {'mode':<11}" + "".join(f"{c[:14]:>15}" for c in cols) + f"{'mv tp/fp':>10}"
    lines = [head]
    for name, r in res["rows"].items():
        for m in MODES:
            v = r[m]
            lines.append(f"{name:<10} {m:<11}" + "".join(f"{'n/a' if v.get(c) is None else f'{100 * v[c]:.1f}':>15}"
                                                       for c in cols) + f"{v['moved_tp']:>5}/{v['moved_fp']:<4}")  # fmt: skip
    for n, v in res["verdict"].items():
        lines.append(f"{n:<10} pass={v['pass']}  mean moved F1 {100 * v['mean_moved_F1']:.1f}")
    lines.append(f"winner: {res['winner']}")
    return "\n".join(lines)


# ---- Part C (C1): p_del at the learned vote weights ---------------------------------------------


def select_p_del(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], values: Mapping[str, float],
                 ref: Mapping[str, Mapping[str, Any]], crit: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Pre-written rule: false deletion <= ref + `fd_margin` in both modes; then the
    highest mean removed F1 (oracle, perception); ties to the value nearest `tie_target`; None if none qualifies."""
    margin, target = float(crit["fd_margin"]), float(crit["tie_target"])
    verdict: dict[str, Any] = {}
    for name, r in rows.items():
        ok = all(r[m]["false_deletion_rate"] is not None
                 and r[m]["false_deletion_rate"] <= ref[m]["false_deletion_rate"] + margin + 1e-12 for m in MODES)  # fmt: skip
        verdict[name] = {"pass": bool(ok), "p_del": float(values[name]),
                         "mean_removed_F1": sum(r[m]["removed_F1"] or 0.0 for m in MODES) / len(MODES)}  # fmt: skip
    ok = [n for n in rows if verdict[n]["pass"]]
    winner = (max(ok, key=lambda n: (round(verdict[n]["mean_removed_F1"], 9), -abs(verdict[n]["p_del"] - target)))
              if ok else None)  # fmt: skip
    return {"verdict": verdict, "winner": winner}


def run_part_c_pdel(cfg: dict[str, Any], run_id: str | None = None, split: str = "dev_tune",
                    workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """The p_del sweep at the configured (learned) weights + the selection against the Part A reference run."""
    import json

    t0 = time.time()
    pc = cfg["chg3"]["part_c"]["p_del"]
    rid = run_id or new_run_id("chg3_partC_pdel")
    values = {f"p{v:g}": float(v) for v in pc["values"]}
    rows, paths = run_variants(cfg, {n: {"confidence.p_del": v} for n, v in values.items()}, rid, split, workers)
    ref_m = json.loads((outputs_root(cfg) / pc["ref_run"] / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    ref = {m: _pick(ref_m["summary"][m]) for m in MODES}
    sel = select_p_del(rows, values, ref, pc["criterion"])
    res = {"split": split, "values": values, "rows": rows, "ref_run": pc["ref_run"], "ref_rows": ref,
           "criterion": dict(pc["criterion"]), **sel, "metrics_paths": paths}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / rid, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_c_pdel(res: Mapping[str, Any]) -> str:
    """Table of the p_del sweep with the Part A reference row, the verdicts and the winner."""
    cols = ("removed_F1", "removed_P", "removed_R", "false_deletion_rate", "unchanged_kept", "moved_F1", "false_move_rate")
    lines = [f"{'variant':<12} {'mode':<11}" + "".join(f"{c[:14]:>15}" for c in cols)]
    for name, r in {"ref(PartA)": res["ref_rows"], **res["rows"]}.items():
        for m in MODES:
            v = r[m]
            lines.append(f"{name:<12} {m:<11}" + "".join(f"{'n/a' if v.get(c) is None else f'{100 * v[c]:.1f}':>15}"
                                                        for c in cols))  # fmt: skip
    for n, v in res["verdict"].items():
        lines.append(f"{n:<12} pass={v['pass']}  mean removed F1 {100 * v['mean_removed_F1']:.1f}")
    lines.append(f"winner: {res['winner']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CHANGE 3 Part N (N1): blob splitting variants
# --------------------------------------------------------------------------------------------


def select_segment(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any],
                   order: list[str], base: str = "none") -> dict[str, Any]:  # fmt: skip
    """Pre-written N1 rule (CHANGE 3 Part N): primary = perception class-agnostic candidate
    hits on change targets (`diag_cand_new_hit + diag_cand_moved_hit`); guards (perception): moved F1 and
    class-agnostic new F1 (at `p_new_min` 0) >= base - `f1_drop`; ties within `tie_objects` -> earlier in `order`."""
    b = rows[base]["perception"]
    drop, tie = float(crit["f1_drop"]), int(crit["tie_objects"])
    verdict: dict[str, Any] = {}
    for name in order:
        if name not in rows:
            continue
        p = rows[name]["perception"]
        ok = name == base or ((p["moved_F1"] or 0) >= (b["moved_F1"] or 0) - drop - 1e-12
                              and (p["new_ag_F1"] or 0) >= (b["new_ag_F1"] or 0) - drop - 1e-12)  # fmt: skip
        verdict[name] = {"pass": bool(ok), "hits": int(p["diag_cand_new_hit"] + p["diag_cand_moved_hit"])}
    passing = [n for n in order if n in verdict and verdict[n]["pass"]]
    best = max(verdict[n]["hits"] for n in passing)
    winner = next(n for n in passing if verdict[n]["hits"] >= best - tie)  # simplest within the tie band
    return {"verdict": verdict, "winner": winner, "missing": [n for n in order if n not in rows]}


def run_part_n_segment(cfg: dict[str, Any], run_id: str = "chg3_partN_seg", split: str = "dev_tune",
                       workers: int | None = None, only: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
    """The N1 variants (`chg3.part_n.segment`) on the tuning set, resumable, + the selection once all ran."""
    t0 = time.time()
    pc = cfg["chg3"]["part_n"]["segment"]
    common = dict(pc.get("common") or {})
    variants = {n: {**common, **(ov or {})} for n, ov in pc["variants"].items()}
    rows, paths = run_variants(cfg, variants, run_id, split, workers, details=True, resume=True, only=only)
    order = list(pc["variants"])
    sel = select_segment(rows, pc["criterion"], order) if "none" in rows else {"verdict": {}, "winner": None}
    if sel.get("missing"):
        sel["winner"] = None  # not all variants ran yet
    res = {"split": split, "variants": variants, "rows": rows, "criterion": dict(pc["criterion"]), **sel,
           "metrics_paths": paths}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / run_id, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_n_segment(res: Mapping[str, Any]) -> str:
    """Table of the N1 variants (perception candidates and new objects, both modes' moved), verdicts, winner."""
    cols = [("cands", "diag_cand_n", 0), ("new hit", "diag_cand_new_hit", 0), ("mv hit", "diag_cand_moved_hit", 0),
            ("in blob", "diag_cand_multi", 0), ("new P", "new_P", 1), ("new R", "new_R", 1), ("ag P", "new_ag_P", 1),
            ("ag R", "new_ag_R", 1), ("ag F1", "new_ag_F1", 1), ("moved F1", "moved_F1", 1),
            ("f.move", "false_move_rate", 1)]  # fmt: skip
    head = f"{'variant':<14} {'mode':<11}" + "".join(f"{c[0]:>9}" for c in cols)
    lines = [head]
    for name, r in res["rows"].items():
        for m in MODES:
            v = r[m]
            cells = []
            for _, k, pct in cols:
                x = v.get(k)
                cells.append(f"{'-' if x is None else (f'{100 * x:.1f}' if pct else str(int(x))):>9}")
            lines.append(f"{name:<14} {m:<11}" + "".join(cells))
    for n, v in res.get("verdict", {}).items():
        score = f"hits(new+moved) {v['hits']}" if "hits" in v else f"agnostic new F1 {100 * v['new_ag_F1']:.1f}"
        lines.append(f"{n:<14} pass={v['pass']}  {score}")
    lines.append(f"winner: {res.get('winner')}" + (f"  (missing: {res['missing']})" if res.get("missing") else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CHANGE 3 Part N (N5): noise rejection + thin-object confidence, each variant at its best threshold
# --------------------------------------------------------------------------------------------


def best_threshold(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """Labelled perception new F1 over every distinct new-node confidence of a run (`p_new_min` 0, exact);
    the best (ties -> lower threshold), as the Part C / N4 rule."""
    import json
    from pathlib import Path

    from lsg.eval.newcurve import new_curve

    vals = {0.0}
    for f in Path(run_dir, "graphs", "perception").glob("*.json"):
        for n in json.loads(f.read_text(encoding="utf-8"))["nodes"]:
            if n.get("state") == "new" and n.get("confidence") is not None:
                vals.add(round(float(n["confidence"]), 6))
    rows = new_curve(cfg, run_dir, sorted(vals))["curve"]
    for r in rows:
        r["f1"] = 2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"]) if (2 * r["tp"] + r["fp"] + r["fn"]) else 0.0
    best = max(rows, key=lambda r: (round(r["f1"], 9), -r["p_new_min"]))
    return {"best": best, "n_thresholds": len(rows)}


def select_noise(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], curves: Mapping[str, Mapping[str, Any]],
                 crit: Mapping[str, Any], order: list[str], base: str = "base") -> dict[str, Any]:  # fmt: skip
    """Pre-written N5 rule: guards (perception) moved F1 and removed F1 >= base - `f1_drop`,
    false deletion <= base + `fd_up`, class-agnostic candidate hits (new + moved) >= base - `hits_drop`; primary =
    best labelled new F1 on the variant's own curve; ties within `tie_f1` -> earlier in `order`."""
    b = rows[base]["perception"]
    verdict: dict[str, Any] = {}
    for name in order:
        if name not in rows:
            continue
        p = rows[name]["perception"]
        hits, bh = p["diag_cand_new_hit"] + p["diag_cand_moved_hit"], b["diag_cand_new_hit"] + b["diag_cand_moved_hit"]
        ok = name == base or ((p["moved_F1"] or 0) >= (b["moved_F1"] or 0) - crit["f1_drop"] - 1e-12
                              and (p["removed_F1"] or 0) >= (b["removed_F1"] or 0) - crit["f1_drop"] - 1e-12
                              and (p["false_deletion_rate"] or 0) <= (b["false_deletion_rate"] or 0) + crit["fd_up"] + 1e-12
                              and hits >= bh - crit["hits_drop"])  # fmt: skip
        bt = curves[name]["best"]
        verdict[name] = {"pass": bool(ok), "best_new_F1": bt["f1"], "p_new_min": bt["p_new_min"], "hits": int(hits)}
    passing = [n for n in order if verdict.get(n, {}).get("pass")]
    top = max(verdict[n]["best_new_F1"] for n in passing)
    winner = next(n for n in passing if verdict[n]["best_new_F1"] >= top - crit["tie_f1"] - 1e-12)
    return {"verdict": verdict, "winner": winner, "p_new_min": verdict[winner]["p_new_min"]}


def run_part_n_noise(cfg: dict[str, Any], run_id: str = "chg3_partN_noise", split: str = "dev_tune",
                     workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """The N5 variants (`chg3.part_n.noise`) at `p_new_min` 0, each one's curve, + the selection (resumable)."""
    t0 = time.time()
    pc = cfg["chg3"]["part_n"]["noise"]
    common = dict(pc.get("common") or {})
    variants = {n: {**common, **(ov or {})} for n, ov in pc["variants"].items()}
    rows, paths = run_variants(cfg, variants, run_id, split, workers, details=True, resume=True)
    curves = {}
    for n in rows:
        c = apply_overrides(cfg, variants[n])
        curves[n] = best_threshold(c, outputs_root(cfg) / run_id / n)
    sel = select_noise(rows, curves, pc["criterion"], list(pc["variants"]))
    res = {"split": split, "variants": variants, "rows": rows, "curves": curves, "criterion": dict(pc["criterion"]),
           **sel, "metrics_paths": paths}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / run_id, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_n_noise(res: Mapping[str, Any]) -> str:
    """Variant table (perception at p_new_min 0) + each variant's best threshold, verdicts, winner."""
    lines = [format_part_n_segment({"rows": res["rows"], "verdict": {}, "winner": None})]
    lines[0] = "\n".join(lines[0].split("\n")[:-1])
    for n, v in res["verdict"].items():
        b = res["curves"][n]["best"]
        lines.append(f"{n:<8} pass={v['pass']}  hits {v['hits']}  best p_new_min {b['p_new_min']:.6f}: new P / R "
                     f"{100 * (b['precision'] or 0):.1f} / {100 * (b['recall'] or 0):.1f}, F1 {100 * b['f1']:.2f} ({b['n_new']} nodes)")  # fmt: skip
    lines.append(f"winner: {res['winner']} at p_new_min {res['p_new_min']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CHANGE 3 Part B (B3): online alignment variants
# --------------------------------------------------------------------------------------------

ALIGN_PRIMARY = ("moved_F1", "removed_F1", "new_F1")


def select_align(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any], order: list[str],
                 ref: str = "offline") -> dict[str, Any]:  # fmt: skip
    """Pre-written B3 rule: among the online variants, the highest mean of moved / removed /
    new F1 over both modes wins (ties within `tie` -> earlier in `order`); it is flagged `ask` (Erik decides) if it
    loses more than `f1_drop` moved or removed F1, or more than `rate_drop` on false deletion / unchanged kept,
    against `ref` in either mode."""
    def score(r):
        return sum((r[m].get(k) or 0.0) for m in MODES for k in ALIGN_PRIMARY) / (len(MODES) * len(ALIGN_PRIMARY))

    cand = [n for n in order if n in rows and n != ref]
    sc = {n: score(rows[n]) for n in cand}
    top = max(sc.values())
    winner = next(n for n in cand if sc[n] >= top - float(crit["tie"]) - 1e-12)
    w, b = rows[winner], rows[ref]
    losses = {}
    for m in MODES:
        for k in ("moved_F1", "removed_F1"):
            losses[f"{m}.{k}"] = (b[m].get(k) or 0) - (w[m].get(k) or 0)
        losses[f"{m}.false_deletion_rate"] = (w[m].get("false_deletion_rate") or 0) - (b[m].get("false_deletion_rate") or 0)
        losses[f"{m}.unchanged_kept"] = (b[m].get("unchanged_kept") or 0) - (w[m].get("unchanged_kept") or 0)
    ask = any(v > float(crit["f1_drop"]) + 1e-12 for k, v in losses.items() if k.endswith("F1")) or \
        any(v > float(crit["rate_drop"]) + 1e-12 for k, v in losses.items() if not k.endswith("F1"))
    return {"scores": sc, "ref_score": score(b), "winner": winner, "losses": losses, "ask": bool(ask)}


def align_stats(metrics_path) -> dict[str, Any]:
    """Online-alignment outcome over a run's pairs (from metrics.json `run.per_pair[].online.align_online`)."""
    import json
    from collections import Counter

    d = json.loads(open(metrics_path, encoding="utf-8").read())
    status, first, final_dt = Counter(), [], []
    for r in d["run"]["per_pair"]:
        ups = (r.get("online") or {}).get("align_online") or []
        if r["mode"] != "oracle" or not ups:
            continue
        for u in ups:
            status[u["status"]] += 1
        acc = [i for i, u in enumerate(ups) if u["status"] == "accepted"]
        first.append(acc[0] if acc else None)
        last = [u for u in ups if u["status"] == "accepted"]
        if last:
            final_dt.append(last[-1]["delta_t_m"])
    return {"updates": dict(status), "pairs_never_accepted": sum(f is None for f in first),
            "first_accepted_update_median": (float(np.median([f for f in first if f is not None]))
                                             if any(f is not None for f in first) else None),
            "final_correction_vs_raw_cm_median": float(100 * np.median(final_dt)) if final_dt else None}  # fmt: skip


def run_part_b_align(cfg: dict[str, Any], run_id: str = "chg3_partB_align", split: str = "dev_tune",
                     workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """The B3 variants (`chg3.part_b.align`) on the tuning set (resumable) + the selection."""
    t0 = time.time()
    pc = cfg["chg3"]["part_b"]["align"]
    variants = {n: dict(ov or {}) for n, ov in pc["variants"].items()}
    rows, paths = run_variants(cfg, variants, run_id, split, workers, resume=True)
    sel = select_align(rows, pc["criterion"], list(pc["variants"]))
    stats = {n: align_stats(p) for n, p in paths.items() if n != "offline"}
    res = {"split": split, "variants": variants, "rows": rows, "criterion": dict(pc["criterion"]), **sel,
           "align_stats": stats, "metrics_paths": paths}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / run_id, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_b_align(res: Mapping[str, Any]) -> str:
    """Variant table (both modes), primary scores, losses of the winner vs offline, alignment stats."""
    cols = ("moved_F1", "removed_F1", "new_F1", "false_deletion_rate", "unchanged_kept", "false_move_rate")
    lines = [f"{'variant':<14} {'mode':<11}" + "".join(f"{c[:14]:>15}" for c in cols)]
    for n, r in res["rows"].items():
        for m in MODES:
            lines.append(f"{n:<14} {m:<11}" + "".join(f"{'-' if r[m].get(c) is None else f'{100 * r[m][c]:.1f}':>15}"
                                                       for c in cols))  # fmt: skip
    lines.append(f"primary (mean moved / removed / new F1, both modes): offline {100 * res['ref_score']:.2f}; "
                 + ", ".join(f"{n} {100 * v:.2f}" for n, v in res["scores"].items()))  # fmt: skip
    lines.append(f"winner: {res['winner']}; losses vs offline (pt): "
                 + ", ".join(f"{k} {100 * v:+.1f}" for k, v in res["losses"].items()))  # fmt: skip
    lines.append("STOP: loss beyond the limit -> Erik decides" if res["ask"] else "within the limit -> new default")
    for n, st in res["align_stats"].items():
        lines.append(f"{n}: {st}")
    return "\n".join(lines)


# ---- Part E: open-vocabulary detector (rule written before the runs) ----------------------


def part_e_grid(pe: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Step 1 variants: every (min_score, min_views, labels) of `chg3.part_e.step1`, detector `replace`, p_new_min 0."""
    s1 = pe["step1"]
    out = {}
    for lab in s1["labels"]:
        for ms in s1["min_score"]:
            for mv in s1["min_views"]:
                out[f"{lab}_s{ms:g}_v{mv}"] = {**pe["common"], "perception.detector.min_score": float(ms),
                                               "perception.detector.min_views": int(mv),
                                               "perception.detector.labels": lab, "perception.detector.p_new_min": 0.0}
    return out


def select_part_e_step1(curves: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Best (variant, newness threshold) by labelled new F1 (ties: earlier variant, then lower threshold)."""
    best = None
    for name, cv in curves.items():
        for r in cv["curve"]:
            if best is None or r["f1"] > best["f1"] + 1e-12:
                best = {"variant": name, "p_new_min": r["p_new_min"], "f1": r["f1"], "precision": r["precision"],
                        "recall": r["recall"], "ag_f1": r["ag_f1"]}  # fmt: skip
    return best or {}


def select_part_e(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any], order: list[str],
                  ref: str = "residual") -> dict[str, Any]:  # fmt: skip
    """Step 2 rule: highest perception new F1 among the variants that pass the guards against `ref` (perception:
    moved / removed F1 not lower by more than `f1_drop`, false deletion not higher and unchanged kept not lower by
    more than `rate_drop`); ties within `tie` -> earlier in `order`. `ref` itself always passes. Oracle rows must be
    identical (`oracle_differs` lists the variants whose oracle metrics differ from `ref`)."""
    b = rows[ref]["perception"]
    verdict = {}
    for n in order:
        r = rows[n]["perception"]
        losses = {k: (b.get(k) or 0) - (r.get(k) or 0) for k in ("moved_F1", "removed_F1", "unchanged_kept")}
        losses["false_deletion_rate"] = (r.get("false_deletion_rate") or 0) - (b.get("false_deletion_rate") or 0)
        ok = n == ref or (losses["moved_F1"] <= float(crit["f1_drop"]) + 1e-12
                          and losses["removed_F1"] <= float(crit["f1_drop"]) + 1e-12
                          and losses["unchanged_kept"] <= float(crit["rate_drop"]) + 1e-12
                          and losses["false_deletion_rate"] <= float(crit["rate_drop"]) + 1e-12)  # fmt: skip
        verdict[n] = {"pass": bool(ok), "new_F1": float(r.get("new_F1") or 0.0), "losses": losses}
    passing = [n for n in order if verdict[n]["pass"]]
    top = max(verdict[n]["new_F1"] for n in passing)
    winner = next(n for n in passing if verdict[n]["new_F1"] >= top - float(crit["tie"]) - 1e-12)
    oracle_differs = [n for n in order if "oracle" in rows[n] and "oracle" in rows[ref]
                      and rows[n]["oracle"] != rows[ref]["oracle"]]  # fmt: skip
    return {"verdict": verdict, "winner": winner, "oracle_differs": oracle_differs}


def run_part_e(cfg: dict[str, Any], run_id: str = "chg3_partE", split: str = "dev_tune",
               workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """Part E on the tuning set (resumable): step 1 grid (perception) + newness curves, step 2 variants + rule."""
    from pathlib import Path

    from lsg.eval.newcurve import new_curve
    from lsg.eval.newfunnel import new_funnel

    t0 = time.time()
    pe = cfg["chg3"]["part_e"]
    grid = part_e_grid(pe)
    rows1, paths1 = run_variants(cfg, grid, f"{run_id}/step1", split, workers, resume=True, modes=("perception",))
    ths = [float(t) for t in pe["thresholds"]]
    curves = {n: new_curve(apply_overrides(cfg, ov), Path(paths1[n]).parent, ths, score="newness")
              for n, ov in grid.items()}  # fmt: skip
    pick = select_part_e_step1(curves)
    best = {**grid[pick["variant"]], "perception.detector.p_new_min": float(pick["p_new_min"])}
    variants = {"residual": {},
                "det": {**best, "perception.detector.mode": "replace", "perception.detector.p_new_min": 0.0},
                "det_new": {**best, "perception.detector.mode": "replace"},
                "det_union": {**best, "perception.detector.mode": "union"}}  # fmt: skip
    rows2, paths2 = run_variants(cfg, variants, f"{run_id}/step2", split, workers, details=True, resume=True)
    sel = select_part_e(rows2, pe["criterion"], list(variants))
    funnels = {n: new_funnel(Path(p).parent) for n, p in paths2.items()}
    res = {"split": split, "step1": {"variants": grid, "rows": rows1, "curves": {n: c["curve"] for n, c in curves.items()},
                                     "pick": pick},
           "step2": {"variants": variants, "rows": rows2, **sel, "funnels": funnels},
           "criterion": dict(pe["criterion"]), "metrics_paths": {**paths1, **{f"step2/{n}": p for n, p in paths2.items()}}}  # fmt: skip
    path = write_metrics(outputs_root(cfg) / run_id, res, cfg, extra={"wall_time_s": time.time() - t0})
    return {**res, "metrics_path": str(path)}


def format_part_e(res: Mapping[str, Any]) -> str:
    """Step 1 best thresholds per variant, the pick; step 2 table, verdicts, winner, funnels."""
    L = ["step 1 (perception, detector replace): best labelled new F1 over the newness threshold"]
    for n, cv in res["step1"]["curves"].items():
        b = max(cv, key=lambda r: r["f1"])
        L.append(f"  {n:<22} p_new_min {b['p_new_min']:.2f}: P {100 * (b['precision'] or 0):5.1f} R "
                 f"{100 * (b['recall'] or 0):5.1f} F1 {100 * b['f1']:5.1f}  (agnostic F1 {100 * b['ag_f1']:5.1f}, "
                 f"{b['n_new']} new)")  # fmt: skip
    p = res["step1"]["pick"]
    L.append(f"  pick: {p['variant']} at newness >= {p['p_new_min']:.2f} (F1 {100 * p['f1']:.1f})")
    cols = ("new_P", "new_R", "new_F1", "new_ag_P", "new_ag_R", "moved_F1", "removed_F1", "false_deletion_rate",
            "unchanged_kept", "false_move_rate", "cand_per_pair")  # fmt: skip
    L.append(f"step 2 (FULL, dev_tune): {'variant':<10} {'mode':<11}" + "".join(f"{c[:12]:>13}" for c in cols))
    s2 = res["step2"]
    for n, r in s2["rows"].items():
        for m in MODES:
            L.append(f"  {n:<10} {m:<11}" + "".join(
                f"{'-' if r[m].get(c) is None else (f'{r[m][c]:.1f}' if c == 'cand_per_pair' else f'{100 * r[m][c]:.1f}'):>13}"
                for c in cols))  # fmt: skip
    for n, v in s2["verdict"].items():
        L.append(f"  {n:<10} {'pass' if v['pass'] else 'FAIL'}  new F1 {100 * v['new_F1']:.1f}  losses vs residual (pt): "
                 + ", ".join(f"{k} {100 * x:+.1f}" for k, x in v["losses"].items()))  # fmt: skip
    L.append(f"  winner: {s2['winner']}; oracle differs from residual in: {s2['oracle_differs'] or 'none'}")
    for n, fu in s2["funnels"].items():
        L.append(f"  funnel {n}: {fu['summary']}")
    return "\n".join(L)
