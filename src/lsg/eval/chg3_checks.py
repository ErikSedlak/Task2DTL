"""CHANGE 3 dev_tune sweeps (`lsg chg3 parta`): variants of a rule, both modes, a pre-written selection rule.

Part A (docs/CHANGE_online_graph.md §3, DECISIONS 2026-11-20): every variant in `chg3.part_a.variants`
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
from collections.abc import Mapping
from typing import Any

from lsg.config import outputs_root
from lsg.eval.ablations import apply_overrides
from lsg.eval.harness import MODES, run_split
from lsg.runinfo import new_run_id, write_metrics

KEYS = ("moved_P", "moved_R", "moved_F1", "false_move_rate", "false_deletion_rate", "unchanged_acc", "unchanged_kept",
        "removed_P", "removed_R", "removed_F1", "new_P", "new_R", "unknown_rate", "moved_R_disp[0.1,0.3)", "moved_R_disp[0.3,1)",
        "moved_R_disp>=1")  # fmt: skip


def _pick(summary: Mapping[str, Any]) -> dict[str, Any]:
    M = summary["metrics"]
    return {k: M[k]["value"] for k in KEYS if k in M} | {"moved_fp": summary["counts"]["moved_fp"],
                                                       "moved_tp": summary["counts"]["moved_tp"]}  # fmt: skip


def select(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any], base: str = "off") -> dict[str, Any]:
    """Apply the selection rule (module docstring) to `{variant: {mode: picked metrics}}`."""
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
                 workers: int | None = None) -> tuple[dict[str, Any], dict[str, str]]:  # fmt: skip
    """FULL on `split` (both modes) once per variant (dotted overrides) into `outputs/<rid>/<variant>`."""
    rows: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for name, ov in variants.items():
        c = apply_overrides(cfg, ov)
        r = run_split(c, split, obs_modes=list(MODES), method="FULL", run_id=f"{rid}/{name}", workers=workers)
        rows[name] = {m: _pick(r["summary"][m]) for m in MODES}
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
    """Pre-written rule (DECISIONS 2026-11-20): false deletion <= ref + `fd_margin` in both modes; then the
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
