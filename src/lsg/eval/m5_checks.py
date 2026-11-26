"""M5 acceptance checks (`uv run lsg m5`): association, ICP moved test, relations.

1. Dev (`splits/dev_train.txt`): B0, B1, B2 in perception + oracle mode. Acceptance per mode:
   B2 association accuracy (strict) >= B0's, B2 moved F1 >= B1's.
2. Every saved G_upd validates (`graph_invalid` count = 0 over all runs).
3. Dev diagnostics of B2: confirmed / absorbed / matched counts; GT-unchanged objects with
   free evidence that B2 turns into `removed`.
4. Optional dev sweep (B2, both modes): `association.tau_assoc` x `association.w_p`.
5. Val (reported once, nothing tuned): B2 in both modes with the relation table R0
   (stale) / R1 (geometric) / update. Default: the milestone val subset
   (`splits/eval_val_small.txt`); `full_val=True` for all of `eval_val`.

The unit / synthetic checks are pytest tests (`test_association.py`, `test_classify.py`,
`test_b2.py`, `test_update.py`, `test_relations_eval.py`). Lives in `eval/` (reads GT).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from lsg.config import REPO_ROOT, milestone_val_split, outputs_root
from lsg.eval.harness import format_summary as eval_summary
from lsg.eval.harness import run_eval
from lsg.eval.relations import REL_POLICIES, REL_SUBSETS
from lsg.io.scan3r import read_pairs
from lsg.runinfo import new_run_id, write_metrics

MODES = ("perception", "oracle")
REPORT_KEYS = ("moved_P", "moved_R", "moved_F1", "removed_P", "removed_R", "new_P", "new_R", "new_F1",
               "unchanged_acc", "unchanged_kept", "false_deletion_rate", "false_move_rate", "unknown_rate",
               "assoc_acc_strict", "assoc_acc_same_label", "free_ev_unchanged", "free_ev_unchanged_removed")  # fmt: skip


def _pick(summary: dict[str, Any], keys=REPORT_KEYS) -> dict[str, Any]:
    """value + n of selected metrics of one mode, plus localisation error and invalid graphs."""
    M = summary["metrics"]
    out = {k: {"value": M[k]["value"], "n": M[k]["n"]} for k in keys}
    out["loc_error_median"] = summary["loc_error"]["median"]
    out["graph_invalid"] = summary["counts"]["graph_invalid"]
    return out


def _rel_table(summary: dict[str, Any]) -> dict[str, Any]:
    """Relation rows of one mode: {subset_universe_floor: {policy: {parent, type, n}}}."""
    M = summary["metrics"]
    out: dict[str, Any] = {}
    for s in REL_SUBSETS:
        for u in ("gt", "as"):
            for f in ("all", "nf"):
                row = {}
                for pol, pre in REL_POLICIES.items():
                    k = f"rel_{pre}_{s}_{u}_{f}"
                    row[pol] = {"parent_acc": M[f"{k}_parent_acc"]["value"], "n": M[f"{k}_parent_acc"]["n"],
                                "type_acc": M[f"{k}_type_acc"]["value"]}  # fmt: skip
                out[f"{s}_{u}_{f}"] = row
    return out


def _assoc_stats(metrics_path: str) -> dict[str, dict[str, int]]:
    """Summed B2 association stats (confirmed / absorbed / matched / ICP fallbacks) per mode."""
    d = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for p in d.get("run", {}).get("per_pair", []):
        a = p.get("assoc")
        if a:
            t = out.setdefault(p["mode"], {})
            for k, v in a.items():
                t[k] = t.get(k, 0) + int(v)
    return out


def run_m5(cfg: dict[str, Any], run_id: str | None = None, sweep: bool = True, val: bool = True,
           full_val: bool = False) -> dict[str, Any]:  # fmt: skip
    """All M5 checks; writes `outputs/<run_id>/metrics.json`."""
    t0 = time.time()
    rid = run_id or new_run_id("m5")
    run_dir = outputs_root(cfg) / rid
    pairs = read_pairs(REPO_ROOT / "splits" / "dev_train.txt")
    res: dict[str, Any] = {"split": "dev_train", "n_pairs": len(pairs), "dev": {}, "sweep": [], "val": None}
    summaries: dict[str, Any] = {}
    for method in ("B0", "B1", "B2"):
        r = run_eval(cfg, pairs, list(MODES), method=method, run_id=f"{rid}/dev_{method}",
                     split_name="dev_train", workers=len(pairs))  # fmt: skip
        summaries[f"dev_{method}"] = r
        res["dev"][method] = {m: _pick(r["summary"][m]) for m in MODES}
        res["dev"][method]["metrics_path"] = r["metrics_path"]
    res["dev"]["B2"]["assoc_stats"] = _assoc_stats(res["dev"]["B2"]["metrics_path"])
    res["dev"]["B2"]["relations"] = {m: _rel_table(summaries["dev_B2"]["summary"][m]) for m in MODES}

    if sweep:
        a = cfg["association"]
        for mode in MODES:
            base = float(a["tau_assoc"][mode])
            for tau in sorted({round(base * f, 2) for f in (0.6, 0.8, 1.0, 1.2)}):
                for w_p in (0.5, 1.0, 2.0):
                    c2 = copy.deepcopy(cfg)
                    c2["association"]["tau_assoc"][mode] = tau
                    c2["association"]["w_p"] = w_p
                    c2["eval"]["save_graphs"] = False
                    r = run_eval(c2, pairs, [mode], method="B2", run_id=f"{rid}/sweep_{mode}_tau{tau}_wp{w_p}",
                                 split_name="dev_train", workers=len(pairs))  # fmt: skip
                    res["sweep"].append({"mode": mode, "tau_assoc": tau, "w_p": w_p, **_pick(r["summary"][mode])})

    if val:
        vs = milestone_val_split(cfg, full_val)
        r = run_eval(cfg, read_pairs(vs), list(MODES), method="B2", run_id=f"{rid}/val_B2", split_name=vs.stem)
        summaries["val_B2"] = r
        res["val"] = {m: {**_pick(r["summary"][m]), "relations": _rel_table(r["summary"][m])} for m in MODES}
        res["val"]["metrics_path"] = r["metrics_path"]
        res["val"]["assoc_stats"] = _assoc_stats(r["metrics_path"])

    D = res["dev"]
    acc: dict[str, Any] = {}
    for m in MODES:
        a2, a0 = D["B2"][m]["assoc_acc_strict"]["value"], D["B0"][m]["assoc_acc_strict"]["value"]
        f2, f1 = D["B2"][m]["moved_F1"]["value"], D["B1"][m]["moved_F1"]["value"]
        acc[f"assoc_B2_ge_B0_{m}"] = {"value": [a2, a0], "pass": a2 is not None and a0 is not None and a2 >= a0}
        acc[f"movedF1_B2_ge_B1_{m}"] = {"value": [f2, f1], "pass": f2 is not None and f1 is not None and f2 >= f1}
    n_inv = sum(D[k][m]["graph_invalid"] for k in ("B0", "B1", "B2") for m in MODES)
    if res["val"] is not None:
        n_inv += sum(res["val"][m]["graph_invalid"] for m in MODES)
    acc["graphs_validate"] = {"value": n_inv, "pass": n_inv == 0}
    acc["relation_table_val"] = {"value": res["val"] is not None, "pass": res["val"] is not None}
    res["acceptance"] = acc
    res["wall_time_s"] = time.time() - t0
    res["metrics_path"] = str(write_metrics(run_dir, {k: v for k, v in res.items() if k != "wall_time_s"}, cfg,
                                            extra={"wall_time_s": res["wall_time_s"]}))  # fmt: skip
    res["_summaries"] = summaries
    return res


def format_summary(res: dict[str, Any]) -> str:
    """Terminal report: dev comparison, sweep, val B2 table (incl. relations), acceptance."""

    def pct(m):
        v = m["value"] if isinstance(m, dict) else m
        return "  n/a" if v is None else f"{100 * v:5.1f}"

    L = []
    if "val_B2" in res["_summaries"]:
        L.append(eval_summary(res["_summaries"]["val_B2"]))
    L.append("== M5: dev comparison (B0 / B1 / B2) ==")
    cols = ("moved_F1", "moved_P", "moved_R", "assoc_acc_strict", "unchanged_acc", "unchanged_kept", "new_P",
            "new_R", "false_deletion_rate", "false_move_rate")  # fmt: skip
    L.append("  method mode        " + " ".join(f"{c[:10]:>10s}" for c in cols) + "   loc_med")
    for method in ("B0", "B1", "B2"):
        for m in MODES:
            E = res["dev"][method][m]
            lm = E["loc_error_median"]
            L.append(f"  {method:6s} {m:11s} " + " ".join(f"{pct(E[c]):>10s}" for c in cols)
                     + f"   {'n/a' if lm is None else f'{lm:.3f}'}")  # fmt: skip
    L.append(f"  B2 association stats (dev): {res['dev']['B2']['assoc_stats']}")
    for m in MODES:
        E = res["dev"]["B2"][m]
        L.append(f"  B2 {m}: GT-unchanged with free evidence {E['free_ev_unchanged']['value'] and round(E['free_ev_unchanged']['value'] * E['free_ev_unchanged']['n'])}"
                 f" / {E['free_ev_unchanged']['n']}, of which predicted removed {pct(E['free_ev_unchanged_removed'])}%")  # fmt: skip
    if res["sweep"]:
        L.append("  sweep (dev, B2)      tau   w_p   movedF1  assoc   unch.acc  new P  new R  false-move  loc_med")
        for r in res["sweep"]:
            lm = r["loc_error_median"]
            L.append(f"    {r['mode']:11s} {r['tau_assoc']:6.2f} {r['w_p']:5.1f}   {pct(r['moved_F1'])}   {pct(r['assoc_acc_strict'])}"
                     f"    {pct(r['unchanged_acc'])}   {pct(r['new_P'])}  {pct(r['new_R'])}     {pct(r['false_move_rate'])}"
                     f"   {'n/a' if lm is None else f'{lm:.3f}'}")  # fmt: skip
    L.append("  acceptance: " + ", ".join(f"{k}={'PASS' if a['pass'] else 'FAIL'}" for k, a in res["acceptance"].items()))
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
