"""Evaluation harness: run a method on a split in one or more observation modes, then score.

`lsg eval --split eval_val --obs oracle --method B0` and `lsg run --pair R S ...` end here.
Lives in `eval/` because it wires GT (`gt_changes`), the oracle (which reads rescan labels)
and the method together. The method itself (`lsg.change`) only gets the reference context,
the rescan's frames + refined `T_align`, and the `ObsInstance`s.

Work is split by reference scan (each reference is loaded once per task) and run in a process
pool; results are sorted back into split order, so `metrics` does not depend on scheduling.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lsg.change.baselines import METHODS
from lsg.change.online import run_online
from lsg.change.pipeline import prepare_pair, select_frames
from lsg.config import REPO_ROOT, cache_root, data_root, outputs_root
from lsg.eval.gt import gt_changes
from lsg.eval.metrics import aggregate, bucket_names, perception_obs_truth, score_pair
from lsg.geom.render import Renderer
from lsg.graph.build import build_reference_graph
from lsg.io.dssg import label_to_nyu40, load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan, open_sequence, read_pairs
from lsg.obs.base import ObsInstance
from lsg.oracle.source import OracleSource
from lsg.perception.labeler import CLIPLabeler
from lsg.perception.source import PerceptionSource
from lsg.runinfo import new_run_id, write_metrics
from lsg.train_meta import clip_vocab_table, perception_vocab

MODES = ("oracle", "perception")


def split_path(split: str) -> Path:
    """`eval_val` -> `splits/eval_val.txt`; a path is returned as is."""
    p = Path(split)
    if p.suffix == ".txt":
        return p if p.is_absolute() else REPO_ROOT / p
    return REPO_ROOT / "splits" / f"{split}.txt"


def check_modes(modes: Sequence[str]) -> None:
    """Reject unknown modes."""
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown obs mode {m!r} (choose from {MODES})")


class _TeeSource:
    """Eval-side wiring: one online loop feeds the method's source and a visibility-only oracle.

    The oracle's visible instances define the GT-matchable set of the association metric in
    perception mode (DECISIONS M2/M4). `finish` returns only the method's observations; the
    oracle never reaches the method.
    """

    def __init__(self, method_src, oracle_src: OracleSource):
        self.src, self.oracle = method_src, oracle_src

    def begin(self, ref_ctx, seq_info) -> None:
        self.src.begin(ref_ctx, seq_info)
        self.oracle.begin(ref_ctx, seq_info)

    def observe(self, frame, ref_render) -> None:
        self.src.observe(frame, ref_render)
        self.oracle.observe(frame, ref_render)

    def finish(self) -> list[ObsInstance]:
        self.oracle.finish()
        return self.src.finish()


def make_labeler(cfg: dict[str, Any], vocab: Sequence[str], threads: int | None = None) -> CLIPLabeler:
    """CLIP labeller from `perception.clip` (model shared per process; cache in `cache/clip/`)."""
    cc = cfg["perception"]["clip"]
    cache = cache_root(cfg) / "clip" if cc.get("cache", True) else None
    return CLIPLabeler(vocab, cc["model"], cc["pretrained"], cc.get("device", "cuda"), cc["prompt"], cache, threads)


def _candidate_rows(obs: Sequence[ObsInstance], truth: dict[str, tuple[int, str]]) -> list[dict[str, Any]]:
    """Per-candidate summary for `outputs/<run>/candidates/` (inspection, overlays)."""
    return [{"token": o.token, "label": o.label, "label_topk": [[a, round(b, 4)] for a, b in o.label_topk],
             "centroid": [round(float(v), 4) for v in o.centroid],
             "aabb": [[round(float(v), 4) for v in o.aabb_min], [round(float(v), 4) for v in o.aabb_max]],
             "n_voxels": len(o.points), "n_frames": len(o.masks), "n_crops": len(o.crops),
             "gt": list(truth[o.token]) if o.token in truth else None}
            for o in obs]  # fmt: skip


def _eval_group(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: all rescans of one reference, every requested mode. Picklable for the pool."""
    cfg, ref, rescans = task["cfg"], task["ref"], task["rescans"]
    root = data_root(cfg)
    meta = load_meta(root)
    R = load_scan(root, ref, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    ref_renderer = Renderer(R.vertices, R.faces, R.face_labels)  # shared by the group's rescans
    method = METHODS[task["method"]]
    labeler = vocab_nyu = label_nyu = None
    if "perception" in task["modes"]:
        top_n = int(cfg["perception"]["clip"]["vocab_top_n"])
        vocab, vocab_nyu = perception_vocab(G_ref, clip_vocab_table(root, cfg), top_n)
        labeler = make_labeler(cfg, vocab, task.get("clip_threads"))
        label_nyu = label_to_nyu40(root)  # eval's class mapping for the §5 rule (all of objects.json)
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
                oracle = OracleSource(S, objs_S, cfg, seed=task["seed"] + int(cfg["oracle"]["seed"]))
                psrc = None
                n_enc0, n_cache0 = (labeler.n_encoded, labeler.n_cached) if labeler is not None else (0, 0)
                if mode == "perception":
                    psrc = PerceptionSource(labeler, vocab_nyu, cfg)
                    src = _TeeSource(psrc, oracle)
                else:
                    src = oracle
                snap = (Path(task["run_dir"]) / "snapshots" / mode / f"{ref}_{s}"
                        if task.get("snapshots") and task.get("run_dir") else None)  # fmt: skip
                online = run_online(ctx, info, seq, src, cfg, snapshot_dir=snap, obs_mode=mode,
                                    renderer=ref_renderer)  # fmt: skip
                obs = online.obs
                tm = oracle.token_map()
                extra_run: dict[str, Any] = {}
                if mode == "perception":
                    truth = perception_obs_truth(obs, gt, cfg, label_nyu)
                    score_kw = {"visible": set(tm.values()), "label_nyu": label_nyu, "candidates": obs}
                    extra_run = {"perception": dict(psrc.stats), "clip_device": labeler.device,
                                 "clip_encoded_crops": labeler.n_encoded - n_enc0,
                                 "clip_cache_hits": labeler.n_cached - n_cache0}  # fmt: skip
                else:
                    truth = {o.token: (tm[o.token], o.label) for o in obs}
                    score_kw = {}
                t_obs = time.time() - t2
                G_upd = method(G_ref, obs, cfg, rescan_id=s, obs_mode=mode, votes=online.votes)
                sc = score_pair(G_upd, gt, mode, cfg, obs_truth=truth, votes=online.votes, **score_kw)
                if cfg["eval"].get("save_graphs", True) and task.get("run_dir"):
                    G_upd.save(Path(task["run_dir"]) / "graphs" / mode / f"{ref}_{s}.json")
                if mode == "perception" and task.get("run_dir"):
                    cp = Path(task["run_dir"]) / "candidates" / f"{ref}_{s}.json"
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    rows = {"ref": ref, "rescan": s, "stats": psrc.stats, "candidates": _candidate_rows(obs, truth),
                            "door_dropped": [{"centroid": [round(float(v), 4) for v in c.centroid],
                                              "n_voxels": len(c.voxels)} for c in psrc.dropped_door]}  # fmt: skip
                    cp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                out.append({
                    "ref": ref, "rescan": s, "mode": mode,
                    "counts": sc["counts"], "loc_errors": sc["loc_errors"],
                    "n_frames": len(fids), "n_obs": len(obs),
                    "run": {"align": ainfo, **extra_run,
                            "n_snapshots": len(online.snapshots),
                            "time_s": {"load_gt": t_load, "align": t_align, "online": t_obs,
                                       **{f"online_{k}": v for k, v in online.time_s.items()},
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
    snapshots: bool = False,
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
        snapshots: Write tentative snapshots to `outputs/<run_id>/snapshots/<mode>/<ref>_<rescan>/`.

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
    n_workers = int(workers if workers is not None else cfg["eval"].get("workers", 1))
    n_proc = max(1, min(n_workers, len(groups)))
    clip_threads = max(1, (os.cpu_count() or 1) // n_proc) if n_proc > 1 else None
    tasks = [{"cfg": cfg, "ref": r, "rescans": ss, "modes": list(obs_modes), "method": method,
              "frac": float(frac), "seed": int(seed), "run_dir": str(run_dir),
              "snapshots": bool(snapshots), "clip_threads": clip_threads}
             for r, ss in sorted(groups.items(), key=lambda kv: -len(kv[1]))]  # fmt: skip
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
        ("free evidence: unch / removed", ["free_ev_unchanged", "free_ev_removed"]),
        ("absent: free ev. / seen ev.", ["free_ev_absent", "seen_ev_absent"]),
    ]
    for mode in MODES:
        L.append(f"-- {mode} --")
        sm = res["summary"].get(mode)
        if sm is None:
            L.append("    n/a (not run)")
            continue
        M = sm["metrics"]
        for name, keys in rows:
            L.append(f"    {name:28s} " + " | ".join(f(M[k]) for k in keys))
        if mode == "perception":
            for name, keys in (("candidates per pair", ["cand_per_pair"]),
                               ("cand. recall new / moved-to", ["cand_recall_new", "cand_recall_moved"]),
                               ("  same, labelled (§5 rule)", ["cand_recall_new_labelled", "cand_recall_moved_labelled"]),
                               ("cand. explained by any GT", ["cand_explained"])):  # fmt: skip
                L.append(f"    {name:28s} " + " | ".join(f(M[k], pct=k != "cand_per_pair") for k in keys))
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
