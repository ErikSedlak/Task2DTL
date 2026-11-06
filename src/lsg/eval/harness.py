"""Evaluation harness: run a method on a split in one or more observation modes, then score.

`lsg eval --split eval_val --obs oracle --method B0` and `lsg run --pair R S ...` end here.
Lives in `eval/` because it wires GT (`gt_changes`), the oracle (which reads rescan labels)
and the method together. The method itself (`lsg.change`) only gets the reference context,
the rescan's frames + refined `T_align`, and the `ObsInstance`s.

Work is split by reference scan (each reference is loaded once per task) and run in a process
pool; results are sorted back into split order, so `metrics` does not depend on scheduling.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lsg.change.baselines import METHODS
from lsg.change.pipeline import collect_observations, prepare_pair, select_frames
from lsg.config import REPO_ROOT, data_root, outputs_root
from lsg.eval.gt import gt_changes
from lsg.eval.metrics import aggregate, bucket_names, score_pair
from lsg.graph.build import build_reference_graph
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan, open_sequence, read_pairs
from lsg.oracle.source import OracleSource
from lsg.runinfo import new_run_id, write_metrics

MODES = ("oracle", "perception")


def split_path(split: str) -> Path:
    """`eval_val` -> `splits/eval_val.txt`; a path is returned as is."""
    p = Path(split)
    if p.suffix == ".txt":
        return p if p.is_absolute() else REPO_ROOT / p
    return REPO_ROOT / "splits" / f"{split}.txt"


def check_modes(modes: Sequence[str]) -> None:
    """Reject unknown modes and modes not implemented yet."""
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown obs mode {m!r} (choose from {MODES})")
        if m == "perception":
            raise NotImplementedError("perception mode (PerceptionSource) arrives at M4")


def _eval_group(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: all rescans of one reference, every requested mode. Picklable for the pool."""
    cfg, ref, rescans = task["cfg"], task["ref"], task["rescans"]
    root = data_root(cfg)
    meta = load_meta(root)
    R = load_scan(root, ref, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    method = METHODS[task["method"]]
    out = []
    for s in rescans:
        t0 = time.time()
        S = load_scan(root, s, with_intrinsics=False)
        objs_S = load_objects(root, s)
        gt = gt_changes(meta[s], R, S, objs_R, objs_S, cfg)
        t_load = time.time() - t0
        with open_sequence(root, s) as seq:
            fids = select_frames(seq.frame_ids, int(cfg["frames"]["stride"]), task["frac"], task["seed"], key=s)
            t1 = time.time()
            # method side gets only the raw T_align matrix, never meta[s]'s change lists
            ctx, info, ainfo = prepare_pair(root, R, objs_R, rels_R, seq, s, meta[s].T_align, cfg, fids, G_ref=G_ref)
            t_align = time.time() - t1
            for mode in task["modes"]:
                t2 = time.time()
                if mode != "oracle":  # guarded by check_modes; perception arrives at M4
                    raise NotImplementedError(mode)
                src = OracleSource(S, objs_S, cfg, seed=task["seed"] + int(cfg["oracle"]["seed"]))
                obs = collect_observations(seq, src, ctx, info)
                tm = src.token_map()
                truth = {o.token: (tm[o.token], o.label) for o in obs}
                t_obs = time.time() - t2
                G_upd = method(G_ref, obs, cfg, rescan_id=s, obs_mode=mode)
                sc = score_pair(G_upd, gt, mode, cfg, obs_truth=truth)
                if cfg["eval"].get("save_graphs", True) and task.get("run_dir"):
                    G_upd.save(Path(task["run_dir"]) / "graphs" / mode / f"{ref}_{s}.json")
                out.append({
                    "ref": ref, "rescan": s, "mode": mode,
                    "counts": sc["counts"], "loc_errors": sc["loc_errors"],
                    "n_frames": len(fids), "n_obs": len(obs),
                    "run": {"align": ainfo,
                            "time_s": {"load_gt": t_load, "align": t_align, "obs": t_obs,
                                       "total": time.time() - t0}},
                })  # fmt: skip
        del S
    return out


def run_eval(
    cfg: dict[str, Any],
    pairs: Sequence[tuple[str, str]],
    obs_modes: Sequence[str],
    method: str = "B0",
    frac: float = 1.0,
    seed: int = 0,
    workers: int | None = None,
    run_id: str | None = None,
    split_name: str = "pairs",
) -> dict[str, Any]:
    """Run `method` on `pairs` in each mode, score, write `outputs/<run_id>/metrics.json`.

    Args:
        cfg: Full config.
        pairs: (reference, rescan) pairs.
        obs_modes: Subset of ("oracle", "perception").
        method: Key of `lsg.change.baselines.METHODS`.
        frac, seed: Observation fraction and chunk seed (§7.3); the seed also shifts the
            oracle token permutation.
        workers: Processes (default `cfg["eval"]["workers"]`; 1 = in-process).
        run_id: Output folder name (default timestamped).
        split_name: Recorded in the metrics.

    Returns:
        The `metrics` block plus `"metrics_path"` and `"wall_time_s"`.
    """
    check_modes(obs_modes)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (available: {sorted(METHODS)})")
    t0 = time.time()
    run_dir = outputs_root(cfg) / (run_id or new_run_id(f"eval_{method}"))
    groups: dict[str, list[str]] = {}
    for r, s in pairs:
        groups.setdefault(r, []).append(s)
    tasks = [{"cfg": cfg, "ref": r, "rescans": ss, "modes": list(obs_modes), "method": method,
              "frac": float(frac), "seed": int(seed), "run_dir": str(run_dir)}
             for r, ss in sorted(groups.items(), key=lambda kv: -len(kv[1]))]  # fmt: skip
    n_workers = int(workers if workers is not None else cfg["eval"].get("workers", 1))
    results: list[dict[str, Any]] = []
    if n_workers <= 1 or len(tasks) == 1:
        for t in tasks:
            results += _eval_group(t)
            print(f"[eval] {t['ref']}: {len(t['rescans'])} rescans done ({time.time() - t0:.0f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
            for t, res in zip(tasks, ex.map(_eval_group, tasks)):
                results += res
                print(f"[eval] {t['ref']}: {len(t['rescans'])} rescans done ({time.time() - t0:.0f} s)", flush=True)
    order = {p: i for i, p in enumerate(pairs)}
    results.sort(key=lambda x: (order[(x["ref"], x["rescan"])], x["mode"]))

    metrics: dict[str, Any] = {
        "split": split_name,
        "method": method,
        "obs_modes": list(obs_modes),
        "frac": float(frac),
        "seed": int(seed),
        "n_pairs": len(pairs),
        "disp_buckets": bucket_names(cfg["eval"]["disp_buckets"]),
        "summary": {m: aggregate([x for x in results if x["mode"] == m], cfg) for m in obs_modes},
        "per_pair": [{k: x[k] for k in ("ref", "rescan", "mode", "n_frames", "n_obs", "counts", "loc_errors")}
                     for x in results],  # fmt: skip
    }
    wall = time.time() - t0
    extra = {
        "wall_time_s": wall,
        "workers": n_workers,
        "per_pair": [{"ref": x["ref"], "rescan": x["rescan"], "mode": x["mode"], **x["run"]} for x in results],
    }
    path = write_metrics(run_dir, metrics, cfg, extra=extra)
    return {**metrics, "metrics_path": str(path), "wall_time_s": wall}


def run_split(cfg: dict[str, Any], split: str, **kw) -> dict[str, Any]:
    """`run_eval` on a split file (`eval_val`, `dev_train`, or a path)."""
    sp = split_path(split)
    return run_eval(cfg, read_pairs(sp), split_name=sp.stem, **kw)


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------


def format_summary(res: dict[str, Any]) -> str:
    """Human-readable table: one row per mode (perception shown as n/a until M4)."""

    def f(m: dict[str, Any], pct: bool = True) -> str:
        """`value [lo, hi] (n)`."""
        v, (lo, hi), n = m["value"], m["ci"], m["n"]
        if v is None:
            return f"   n/a              (n={n})"
        s = 100 if pct else 1
        if lo is None:
            return f"{s * v:5.1f}  [ n/a ]     (n={n})"
        return f"{s * v:5.1f} [{s * lo:5.1f},{s * hi:5.1f}] (n={n})"

    L = [f"== eval: method {res['method']} on {res['split']} ({res['n_pairs']} pairs, frac {res['frac']}, "
         f"seed {res['seed']}) ==", "values in %, 95% bootstrap CI over pairs, n = support"]  # fmt: skip
    rows = [
        ("moved   P / R / F1", ["moved_P", "moved_R", "moved_F1"]),
        ("removed P / R / F1", ["removed_P", "removed_R", "removed_F1"]),
        ("new     P / R / F1", ["new_P", "new_R", "new_F1"]),
        ("unchanged acc / kept", ["unchanged_acc", "unchanged_kept"]),
        ("false-deletion / false-move", ["false_deletion_rate", "false_move_rate"]),
        ("unknown rate", ["unknown_rate"]),
        ("assoc strict / same-label", ["assoc_acc_strict", "assoc_acc_same_label"]),
        ("moved R by displacement", [f"moved_R_disp{b}" for b in res["disp_buckets"]]),
        ("moved R small / large", ["moved_R_small", "moved_R_large"]),
        ("new R small / large", ["new_R_small", "new_R_large"]),
    ]
    for mode in MODES:
        L.append(f"-- {mode} --")
        sm = res["summary"].get(mode)
        if sm is None:
            L.append("    n/a" + (" (arrives at M4)" if mode == "perception" else ""))
            continue
        M = sm["metrics"]
        for name, keys in rows:
            L.append(f"    {name:28s} " + " | ".join(f(M[k]) for k in keys))
        le = sm["loc_error"]
        loc = "n/a" if le["n"] == 0 else f"median {le['median']:.3f}  p90 {le['p90']:.3f}"
        L.append(f"    {'moved loc. error [m]':28s} {loc}  (n={le['n']})")
        c = sm["counts"]
        L.append(f"    GT support: moved {c['moved_tp'] + c['moved_fn']}  removed {c['removed_tp'] + c['removed_fn']}"
                 f"  new {c['new_tp'] + c['new_fn']}  unchanged {c['unch_n']}  absent_unlisted {c['absent_n']}"
                 f"  (ref nodes {c['ref_n']}, GT objects without node {c['ref_no_node']},"
                 f" pred new ignored {c['new_ignored']})")  # fmt: skip
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
