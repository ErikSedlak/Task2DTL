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
        "removed_P", "removed_R", "new_P", "new_R", "unknown_rate", "moved_R_disp[0.1,0.3)", "moved_R_disp[0.3,1)",
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


def run_part_a(cfg: dict[str, Any], run_id: str | None = None, split: str = "dev_tune",
               workers: int | None = None) -> dict[str, Any]:  # fmt: skip
    """All Part A variants on the tuning set + the selection; returns the result dict."""
    t0 = time.time()
    pc = cfg["chg3"]["part_a"]
    rid = run_id or new_run_id("chg3_partA")
    rows: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for name, ov in pc["variants"].items():
        c = apply_overrides(cfg, ov)
        r = run_split(c, split, obs_modes=list(MODES), method="FULL", run_id=f"{rid}/{name}", workers=workers)
        rows[name] = {m: _pick(r["summary"][m]) for m in MODES}
        paths[name] = r["metrics_path"]
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
