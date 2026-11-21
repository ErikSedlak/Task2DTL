"""M7: all ablations (DESIGN §8) on val + relchange, both modes; final results tables (`uv run lsg m7`).

Steps (each run is one `run_eval`, see `eval.ablations`; nothing is tuned, val rows are
reported once):

1. Environment: CUDA availability (DINOv2 / CLIP device) is recorded.
2. Dev (`splits/dev_train.txt`): `m7.dev_rows`. Regressions against stored dev runs
   (`m7.regression_refs`; since CHANGE 3: FULL-m7 = `m7_final/dev/FULL`): identical `metrics` on
   shared keys, thin keys excluded (M7 thin-metric fix). Stops here with `dev_only`.
3. Val (`m7.val_split`, all of `eval_val`): methods, component ablations, variants. Rows in
   `m7.details_rows` (FULL) also write per-object details, the input of the §7.4 audit.
4. τ_free sweep (`m7.tau_free_sweep`) and observed fraction (`m7.fracs` × `m7.frac_seeds`, §7.3).
5. Relchange (`m7.relchange_split`): `m7.relchange_rows`, for the relation tables (§7.2).
6. Determinism: FULL on one dev pair twice -> identical `metrics`.
7. `outputs/<run_id>/results.md` (tables with 95% bootstrap CIs and support), three plots,
   `metrics.json` with every row's summary and an `acceptance` block.

With a fixed `run_id` a rerun resumes: rows whose metrics.json matches the current config and
git commit are reused (`ablations.reusable`). Lives in `eval/` (reads GT).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lsg.config import REPO_ROOT, outputs_root
from lsg.eval.ablations import MODES, Ablation, run_ablation, registry, select
from lsg.eval.relations import REL_POLICIES, REL_SUBSET_NAMES
from lsg.io.scan3r import read_pairs
from lsg.runinfo import new_run_id, write_metrics

MAIN_KEYS = ("moved_P", "moved_R", "moved_F1", "removed_P", "removed_R", "false_deletion_rate", "false_move_rate",
             "unchanged_acc", "unchanged_kept", "new_P", "new_R", "new_ag_P", "new_ag_R", "unknown_rate",
             "assoc_acc_strict")  # fmt: skip
SHORT = {"moved_P": "mov P", "moved_R": "mov R", "moved_F1": "mov F1", "removed_P": "rem P", "removed_R": "rem R",
         "false_deletion_rate": "false-del", "false_move_rate": "false-mov", "unchanged_acc": "unch acc",
         "unchanged_kept": "unch kept", "new_P": "new P", "new_R": "new R", "new_ag_P": "new P ag",
         "new_ag_R": "new R ag", "new_ag_F1": "new F1 ag", "unknown_rate": "unknown",
         "assoc_acc_strict": "assoc"}  # fmt: skip
EXTRA_KEYS = ("removed_F1", "new_F1", "new_ag_F1", "removed_R_thin", "unchanged_kept_thin", "free_ev_removed", "free_ev_removed_thin",
              "free_ev_unchanged", "free_ev_unchanged_removed", "free_ev_absent", "seen_ev_absent",
              "conf_mean_unchanged", "conf_mean_removed", "conf_mean_absent", "cand_per_pair", "cand_explained",
              "cand_appearance_per_pair", "cand_appearance_explained", "moved_R_small", "moved_R_large",
              "new_R_small", "new_R_large")  # fmt: skip
NOT_PCT = ("cand_per_pair", "cand_appearance_per_pair")
SWEEP_KEYS = ("removed_P", "removed_R", "false_deletion_rate", "unchanged_kept", "moved_F1", "new_P", "new_R",
              "unknown_rate")  # fmt: skip
REL_SUBS = tuple(REL_SUBSET_NAMES)  # a, b, c, cb
VAL_GROUPS = ("method", "component", "variant")
COLORS = {"oracle": "#2a78d6", "perception": "#eb6834"}  # categorical slots 1-2 (dataviz palette, validated)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


# ---- summaries ----------------------------------------------------------------------------------


def _rel_keys() -> list[str]:
    keys = []
    for p in REL_POLICIES.values():
        for sub in REL_SUBS:
            for u, fl in (("gt", "all"), ("gt", "nf"), ("as", "all")):
                keys.append(f"rel_{p}_{sub}_{u}_{fl}_parent_acc")
        keys.append(f"rel_{p}_a_gt_all_type_acc")
    return keys


def summarize(r: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-row record: value / CI / n of the report keys per mode, plus run info."""
    out: dict[str, Any] = {"ablation": r.get("ablation"), "metrics_path": r.get("metrics_path"),
                           "reused": r.get("reused", False), "wall_time_s": r.get("wall_time_s"),
                           "n_pairs": r.get("n_pairs"), "split": r.get("split")}  # fmt: skip
    for mode, sm in r["summary"].items():
        M = sm["metrics"]
        keys = [k for k in (*MAIN_KEYS, *EXTRA_KEYS, *_rel_keys()) if k in M]
        out[mode] = {k: {"value": M[k]["value"], "ci": M[k]["ci"], "n": M[k]["n"]} for k in keys}
        out[mode]["loc_error"] = sm["loc_error"]
        out[mode]["graph_invalid"] = sm["counts"]["graph_invalid"]
    return out


def metrics_diff(a: Any, b: Any, ignore: Sequence[str] = ("thin",), path: str = "") -> list[str]:
    """Paths where two `metrics` blocks differ, over shared keys (keys containing `ignore` skipped)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) & set(b)):
            if not any(s in str(k) for s in ignore):
                out += metrics_diff(a[k], b[k], ignore, f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length)"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in metrics_diff(x, y, ignore, f"{path}[{i}]")]
    return [] if a == b else [path]


def _load_metrics(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"] if path.is_file() else None


# ---- runner -------------------------------------------------------------------------------------


def _run_rows(cfg, rows: Sequence[Ablation], pairs, rid: str, group_dir: str, split_name: str, workers, resume,
              store: dict[str, Any], details: Sequence[str] = ()) -> dict[str, dict[str, Any]]:  # fmt: skip
    full: dict[str, dict[str, Any]] = {}
    for ab in rows:
        t0 = time.time()
        r = run_ablation(cfg, ab, pairs, f"{rid}/{group_dir}/{ab.name}", split_name, MODES, workers, resume,
                         details=ab.name in details)  # fmt: skip
        full[ab.name] = r
        store[ab.name] = summarize(r)
        print(f"[m7] {group_dir}/{ab.name}: {'reused' if r['reused'] else f'{time.time() - t0:.0f} s'}", flush=True)
    return full


def run_m7(
    cfg: dict[str, Any],
    run_id: str | None = None,
    dev_only: bool = False,
    sweep: bool = True,
    frac: bool = True,
    relchange: bool = True,
    determinism: bool = True,
    resume: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """All M7 runs and reports; writes `outputs/<run_id>/metrics.json`, `results.md`, plots."""
    t0 = time.time()
    rid = run_id or new_run_id("m7")
    run_dir = outputs_root(cfg) / rid
    m7 = cfg["m7"]
    res: dict[str, Any] = {"env": _env(), "dev": {}, "regressions": {}, "val": {}, "relchange": {},
                           "determinism": None, "frac": {}, "dev_only": dev_only}  # fmt: skip
    print(f"[m7] run {rid}; cuda available: {res['env']['cuda_available']} ({res['env']['cuda_device']})", flush=True)

    dev_pairs = read_pairs(REPO_ROOT / "splits" / "dev_train.txt")
    n_groups = len({r for r, _ in dev_pairs})
    dev_full = _run_rows(cfg, select(cfg, m7["dev_rows"]), dev_pairs, rid, "dev", "dev_train", n_groups, resume,
                         res["dev"])  # fmt: skip
    for name, ref in m7.get("regression_refs", {}).items():
        old = _load_metrics(REPO_ROOT / ref / "metrics.json")
        if name not in dev_full or old is None:
            res["regressions"][name] = {"ref": ref, "n_diff": None, "diff": ["reference run missing"]}
            continue
        d = metrics_diff(old, {k: v for k, v in dev_full[name].items() if k in old})
        res["regressions"][name] = {"ref": ref, "n_diff": len(d), "diff": d[:20]}

    if not dev_only:
        val_pairs = read_pairs(REPO_ROOT / m7["val_split"])
        vname = Path(m7["val_split"]).stem
        reg = registry(cfg)
        rows = [a for a in reg.values() if a.group in VAL_GROUPS]
        if sweep:
            rows += [a for a in reg.values() if a.group == "tau_free"]
        if frac:
            rows += [a for a in reg.values() if a.group == "frac"]
        _run_rows(cfg, rows, val_pairs, rid, "val", vname, workers, resume, res["val"], m7.get("details_rows", ()))
        if frac:
            res["frac"] = frac_table(res["val"], cfg)
        if relchange:
            rc_pairs = read_pairs(REPO_ROOT / m7["relchange_split"])
            _run_rows(cfg, select(cfg, m7["relchange_rows"]), rc_pairs, rid, "relchange",
                      Path(m7["relchange_split"]).stem, workers, resume, res["relchange"])  # fmt: skip

    if determinism:
        pair = [dev_pairs[int(m7.get("determinism_pair", 0))]]
        full_ab = registry(cfg)["FULL"]
        a, b = (run_ablation(cfg, full_ab, pair, f"{rid}/determinism/run{k}", "dev_pair", MODES, 1, resume=False)
                for k in "ab")  # fmt: skip
        ma, mb = (_load_metrics(Path(x["metrics_path"])) for x in (a, b))
        d = metrics_diff(ma, mb, ignore=())
        res["determinism"] = {"pair": list(pair[0]), "n_diff": len(d), "diff": d[:20]}

    res["results_md"] = str(write_results(res, cfg, run_dir / "results.md"))
    res["plots"] = [] if dev_only else write_plots(res, cfg, run_dir)
    res["acceptance"] = acceptance(res, cfg)
    res["wall_time_s"] = time.time() - t0
    res["metrics_path"] = str(write_metrics(run_dir, {k: v for k, v in res.items() if k != "wall_time_s"}, cfg,
                                            extra={"wall_time_s": res["wall_time_s"]}))  # fmt: skip
    return res


def _env() -> dict[str, Any]:
    try:
        import torch

        ok = bool(torch.cuda.is_available())
        return {"cuda_available": ok, "cuda_device": torch.cuda.get_device_name(0) if ok else None}
    except Exception as e:  # pragma: no cover - torch missing / broken driver
        return {"cuda_available": False, "cuda_device": f"error: {e}"}


def frac_table(val: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Observed fraction: mean ± sd over chunk seeds per (fraction, mode, metric); 100% = the FULL row."""
    m7 = cfg["m7"]
    out: dict[str, Any] = {}
    fracs = [1.0, *[float(f) for f in m7.get("fracs", [])]]
    for f in fracs:
        names = ["FULL"] if f == 1.0 else [f"FULL_frac{f:g}_s{int(s)}" for s in m7.get("frac_seeds", [0])]
        rows = [val[n] for n in names if n in val]
        if not rows:
            continue
        out[f"{f:g}"] = {}
        for mode in MODES:
            out[f"{f:g}"][mode] = {}
            for k in SWEEP_KEYS + ("moved_R", "new_R", "removed_F1"):
                v = [r[mode][k]["value"] for r in rows if r[mode][k]["value"] is not None]
                out[f"{f:g}"][mode][k] = {"mean": float(np.mean(v)) if v else None,
                                          "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                          "n_seeds": len(v), "n": [r[mode][k]["n"] for r in rows]}  # fmt: skip
    return out


def acceptance(res: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """DESIGN §9 M7: results table with CIs, plots, determinism on one pair, valid graphs, dev regressions."""
    acc: dict[str, Any] = {}
    regs = res["regressions"]
    bad = {k: v["n_diff"] for k, v in regs.items() if v["n_diff"] not in (0, None)}
    acc["dev_regressions"] = {"value": {k: v["n_diff"] for k, v in regs.items()}, "pass": not bad}
    rows = [*res["dev"].values(), *res["val"].values(), *res["relchange"].values()]
    n_inv = sum(r[m]["graph_invalid"] for r in rows for m in MODES if m in r)
    acc["graphs_validate"] = {"value": n_inv, "pass": n_inv == 0}
    if res.get("determinism") is not None:
        acc["determinism_one_pair"] = {"value": res["determinism"]["n_diff"], "pass": res["determinism"]["n_diff"] == 0}
    acc["results_md_written"] = {"value": res["results_md"], "pass": Path(res["results_md"]).is_file()}
    if not res["dev_only"]:
        full = res["val"].get("FULL")
        acc["val_full_split"] = {"value": full and full["n_pairs"], "pass": bool(full) and full["n_pairs"] == len(
            read_pairs(REPO_ROOT / "splits" / "eval_val.txt"))}  # fmt: skip
        acc["plots_written"] = {"value": len(res["plots"]), "pass": len(res["plots"]) == 3}
    return acc


# ---- results.md ---------------------------------------------------------------------------------


def _fmt(m: Mapping[str, Any] | None, key: str = "", ci: bool = True) -> str:
    if m is None or m.get("value") is None:
        return "n/a"
    s = 1 if key in NOT_PCT else 100
    v = f"{s * m['value']:.1f}"
    lo, hi = m.get("ci") or (None, None)
    return f"{v} [{s * lo:.1f}, {s * hi:.1f}]" if ci and lo is not None else v


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def write_results(res: Mapping[str, Any], cfg: Mapping[str, Any], path: Path) -> Path:
    """Markdown tables: main (CIs), ablation support, τ_free, fraction, relations, diagnostics, checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# Results (M7 tables)", "",
         "Values in %, 95% bootstrap CI over pairs in brackets (1,000 resamples). Val rows are reported once; "
         "nothing was tuned on them. `unknown` counts as neither removed nor moved. New P / R twice: labelled "
         "(the mode's §5 rule) and class-agnostic (`ag`: geometry only, any label), so detection and naming "
         "losses show apart.", ""]  # fmt: skip
    L += _section_rows("Dev (`dev_train`, 3 pairs)", res["dev"], MAIN_KEYS)
    if res["regressions"]:
        L += ["**Dev regressions** (shared `metrics` keys, thin keys excluded):", ""]
        L += [f"- {k} vs `{v['ref']}`: {v['n_diff'] if v['n_diff'] is not None else 'n/a'} differences" for k, v in
              res["regressions"].items()] + [""]  # fmt: skip
    val = res["val"]
    if val:
        main = {k: v for k, v in val.items() if (v.get("ablation") or {}).get("group") in VAL_GROUPS}
        n = next(iter(main.values()))["n_pairs"] if main else "?"
        L += _section_rows(f"Val (`eval_val`, {n} pairs): methods and ablations", main, MAIN_KEYS)
        L += _section_rows("Val diagnostics (thin objects, evidence, confidence, candidates)", main,
                           ("removed_R_thin", "unchanged_kept_thin", "free_ev_removed", "free_ev_unchanged_removed",
                            "conf_mean_removed", "conf_mean_absent", "cand_per_pair", "cand_explained",
                            "cand_appearance_per_pair"), ci=False)  # fmt: skip
        tau = {k: v for k, v in val.items() if (v.get("ablation") or {}).get("group") == "tau_free"}
        if tau:
            t0 = float(cfg["visibility"]["tau_free"])
            tau = {f"FULL_tau{t0:g} (= FULL)": val["FULL"], **tau} if "FULL" in val else tau
            L += _section_rows("τ_free sweep (FULL; thin_max_extent stays 0.15 m)", tau, SWEEP_KEYS)
    if res.get("frac"):
        L += [f"## Observed fraction (FULL, one contiguous chunk, mean ± sd over chunk seeds {cfg['m7']['frac_seeds']})", ""]
        for mode in MODES:
            rows = [[f"{100 * float(f):g}%", *[_msd(d[mode][k]) for k in SWEEP_KEYS]] for f, d in res["frac"].items()]
            L += [f"**{mode}**", ""] + _table(["frames", *[SHORT.get(k, k) for k in SWEEP_KEYS]], rows) + [""]
    for split, rows in (("val", {k: v for k, v in val.items() if k in ("B2", "FULL", "FULL+follow", "FULL-keepparent")}),
                        ("relchange", res["relchange"])):  # fmt: skip
        if rows:
            L += _relations(split, rows)
    if res.get("determinism"):
        d = res["determinism"]
        L += ["## Determinism", "", f"FULL on {d['pair'][0][:8]} → {d['pair'][1][:8]} run twice: "
              f"{d['n_diff']} differences in `metrics`.", ""]  # fmt: skip
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _msd(d: Mapping[str, Any]) -> str:
    if d["mean"] is None:
        return "n/a"
    return f"{100 * d['mean']:.1f}" + (f" ± {100 * d['sd']:.1f}" if d["sd"] is not None else "")


def _section_rows(title: str, rows: Mapping[str, Any], keys: Sequence[str], ci: bool = True) -> list[str]:
    if not rows:
        return []
    L = [f"## {title}", ""]
    for mode in MODES:
        body = [[name, *[_fmt(r[mode].get(k), k, ci) for k in keys]] for name, r in rows.items() if mode in r]
        first = next(r for r in rows.values() if mode in r)
        L += [f"**{mode}**", ""] + _table(["row", *[SHORT.get(k, k) for k in keys]], body)
        L += ["", "support n: " + ", ".join(f"{SHORT.get(k, k)} {first[mode][k]['n']}" for k in keys if k in first[mode]), ""]
    return L


def _relations(split: str, rows: Mapping[str, Any]) -> list[str]:
    L = [f"## Relations ({split}; parent accuracy, GT universe, all parents incl. floor; R0 stale / R1 geometric / update)", ""]
    header = ["row", "mode", *[REL_SUBSET_NAMES[s] for s in REL_SUBS], "type acc (a)"]
    body = []
    for name, r in rows.items():
        for mode in MODES:
            if mode not in r:
                continue
            cells = []
            for s in REL_SUBS:
                ms = [r[mode].get(f"rel_{p}_{s}_gt_all_parent_acc") for p in REL_POLICIES.values()]
                n = ms[0]["n"] if ms[0] else 0
                cells.append(" / ".join(_fmt(m, ci=False) for m in ms) + f" (n={n})")
            cells.append(" / ".join(_fmt(r[mode].get(f"rel_{p}_a_gt_all_type_acc"), ci=False) for p in REL_POLICIES.values()))
            body.append([name, mode, *cells])
    return L + _table(header, body) + [""]


# ---- plots --------------------------------------------------------------------------------------


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def write_plots(res: Mapping[str, Any], cfg: Mapping[str, Any], out_dir: Path) -> list[str]:
    """Three PNGs: metrics vs observed fraction, vs τ_free, and each ablation against FULL."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    keys4 = ("removed_R", "false_deletion_rate", "moved_F1", "new_R")
    titles = {"removed_R": "Removed recall", "false_deletion_rate": "False-deletion rate", "moved_F1": "Moved F1",
              "new_R": "New recall", "unchanged_kept": "Unchanged kept", "removed_P": "Removed precision",
              "new_P": "New precision"}  # fmt: skip

    # 1. observed fraction
    if res.get("frac"):
        fr = sorted(res["frac"], key=float)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE)
        for ax, k in zip(axes, keys4):
            ends = {}
            for mode in MODES:
                x = [100 * float(f) for f in fr]
                y = [100 * (res["frac"][f][mode][k]["mean"] or np.nan) for f in fr]
                e = [100 * (res["frac"][f][mode][k]["sd"] or 0.0) for f in fr]
                ax.errorbar(x, y, yerr=e, color=COLORS[mode], lw=2, marker="o", ms=6, capsize=3, label=mode)
                ends[mode] = (x[-1], y[-1])
            hi = max(ends, key=lambda m: np.nan_to_num(ends[m][1]))  # direct labels: upper series above, other below
            for mode, xy in ends.items():
                ax.annotate(mode, xy, xytext=(6, 6 if mode == hi else -8), textcoords="offset points", color=INK2,
                            fontsize=7, va="center")  # fmt: skip
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
            ax.set_xlabel("observed frames [%]", color=INK2, fontsize=9)
            _style(ax, titles[k], "%" if k == keys4[0] else None)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("FULL vs observed fraction (one contiguous chunk; mean ± sd over seeds)", color=INK, fontsize=11)
        fig.tight_layout()
        p = out_dir / "m7_fraction.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(str(p))

    # 2. tau_free
    val = res["val"]
    t0 = float(cfg["visibility"]["tau_free"])
    tau = {t0: val.get("FULL")} | {float(a["ablation"]["overrides"]["visibility.tau_free"]): a for a in val.values()
                                      if (a.get("ablation") or {}).get("group") == "tau_free"}  # fmt: skip
    tau = {k: v for k, v in sorted(tau.items()) if v is not None}
    if len(tau) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), facecolor=SURFACE)
        for ax, k in zip(axes, ("removed_R", "false_deletion_rate", "unchanged_kept")):
            for mode in MODES:
                x = list(tau)
                y = np.array([100 * (v[mode][k]["value"] or np.nan) for v in tau.values()])
                ci = np.array([[100 * (c if c is not None else np.nan) for c in (v[mode][k]["ci"] or [None, None])]
                               for v in tau.values()])  # fmt: skip
                ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], color=COLORS[mode], lw=2, marker="o", ms=6,
                            capsize=3, label=mode)  # fmt: skip
            ax.axvline(t0, color=MUTED, lw=1, ls="--")
            ax.set_xlabel(f"τ_free [m] (dashed: default {t0:g})", color=INK2, fontsize=9)
            _style(ax, titles[k], "%" if k == "removed_R" else None)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("FULL vs τ_free (95% bootstrap CI)", color=INK, fontsize=11)
        fig.tight_layout()
        p = out_dir / "m7_tau_free.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(str(p))

    # 3. ablations vs FULL (dot + CI per row, one panel per metric)
    rows = {k: v for k, v in val.items() if (v.get("ablation") or {}).get("group") in VAL_GROUPS}
    if rows:
        names = list(rows)[::-1]
        keys = ("removed_R", "removed_P", "false_deletion_rate", "unchanged_kept", "moved_F1", "new_P")
        fig, axes = plt.subplots(1, len(keys), figsize=(16, 0.32 * len(names) + 1.6), sharey=True, facecolor=SURFACE)
        for ax, k in zip(axes, keys):
            for j, mode in enumerate(MODES):
                off = (j - 0.5) * 0.3
                for i, nm in enumerate(names):
                    m = rows[nm][mode][k]
                    if m["value"] is None:
                        continue
                    v = 100 * m["value"]
                    lo, hi = (100 * c if c is not None else v for c in (m["ci"] or [None, None]))
                    ax.plot([lo, hi], [i + off] * 2, color=COLORS[mode], lw=1.5)
                    ax.plot([v], [i + off], "o", color=COLORS[mode], ms=5, mec=SURFACE, mew=1,
                            label=mode if i == 0 else None)  # fmt: skip
                if "FULL" in rows and rows["FULL"][mode][k]["value"] is not None:
                    ax.axvline(100 * rows["FULL"][mode][k]["value"], color=COLORS[mode], lw=1, ls=":", alpha=0.8)
            ax.set_yticks(range(len(names)), names)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            _style(ax, titles.get(k, k) + " [%]")
            ax.grid(axis="y", visible=False)
        grp = [rows[n]["ablation"]["group"] for n in names]
        cuts = [i - 0.5 for i in range(1, len(names)) if grp[i] != grp[i - 1]]  # methods | components | variants
        for ax in axes:
            for y in cuts:
                ax.axhline(y, color=MUTED, lw=0.8)
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper right", ncol=2)
        fig.suptitle("Val: each row with its 95% CI; dotted = FULL", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out_dir / "m7_ablations.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(str(p))
    return paths


# ---- terminal -------------------------------------------------------------------------------------


def format_summary(res: Mapping[str, Any]) -> str:
    """Short terminal report; the tables are in results.md."""
    L = [f"== M7 ({'dev only' if res['dev_only'] else 'dev + val'}); cuda {res['env']['cuda_available']} =="]
    for name, r in {**{f"dev/{k}": v for k, v in res["dev"].items()},
                    **{f"val/{k}": v for k, v in res["val"].items()}}.items():  # fmt: skip
        for mode in MODES:
            if mode in r:
                L.append(f"  {name:24s} {mode:10s} " + "  ".join(f"{SHORT[k]} {_fmt(r[mode][k], k, ci=False):>5s}"
                                                               for k in MAIN_KEYS))  # fmt: skip
    for k, v in res["regressions"].items():
        L.append(f"  regression {k} vs {v['ref']}: {v['n_diff']} differences {v['diff'][:3]}")
    if res.get("determinism"):
        L.append(f"  determinism: {res['determinism']['n_diff']} differences")
    L.append("  acceptance: " + ", ".join(f"{k}={'PASS' if a['pass'] else 'FAIL'} ({a['value']})"
                                         for k, a in res["acceptance"].items()))  # fmt: skip
    L.append(f"results: {res['results_md']}")
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
