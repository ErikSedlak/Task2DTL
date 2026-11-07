"""M4 acceptance checks (`uv run lsg m4`, DESIGN §9): perception end-to-end on the dev pairs.

1. B0 and B1 on dev in perception + oracle mode (`run_eval`), with the candidate diagnostics:
   candidate recall vs GT new / moved-to objects, class-agnostic and with the §5 label rule
   (reported, no threshold).
2. Overlay PNGs of candidate clusters on 3 dev frames (one per dev pair: the processed frame
   with the largest total candidate mask), with projected GT new / moved-to centroids.
3. Diagnostic sweep on dev (perception, B1): door filter on / off x `min_voxels`.

The synthetic acceptance test (box added -> exactly one candidate, centroid error < 5 cm) and
the no-leak test are pytest tests (`tests/test_perception.py`, `tests/test_no_leak.py`).
Lives in `eval/` because it reads GT for the overlay markers and the scoring.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from lsg.change.online import run_online
from lsg.change.pipeline import prepare_pair, select_frames
from lsg.config import REPO_ROOT, data_root, outputs_root
from lsg.eval.gt import gt_changes
from lsg.eval.harness import format_summary as eval_summary
from lsg.eval.harness import make_labeler, run_eval
from lsg.eval.metrics import perception_obs_truth
from lsg.geom.transforms import apply, invert
from lsg.graph.build import build_reference_graph
from lsg.io.dssg import label_to_nyu40, load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan, open_sequence, read_pairs
from lsg.perception.source import PerceptionSource
from lsg.runinfo import new_run_id, write_metrics
from lsg.train_meta import clip_vocab_table, perception_vocab

REPORT_KEYS = ("cand_per_pair", "cand_recall_new", "cand_recall_moved", "cand_recall_new_labelled",
               "cand_recall_moved_labelled", "cand_explained", "new_P", "new_R", "moved_P", "moved_R",
               "removed_P", "removed_R", "unchanged_acc", "unchanged_kept", "false_deletion_rate",
               "false_move_rate", "unknown_rate", "assoc_acc_strict", "assoc_acc_same_label")  # fmt: skip


def _pick(summary: dict[str, Any]) -> dict[str, Any]:
    """value + n of the report metrics of one mode."""
    M = summary["metrics"]
    return {k: {"value": M[k]["value"], "n": M[k]["n"]} for k in REPORT_KEYS}


def _overlay(ref: str, rescan: str, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Run perception on one pair in-process and draw its best frame (CLIP labels from cache)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = data_root(cfg)
    meta = load_meta(root)
    R = load_scan(root, ref, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    S = load_scan(root, rescan, with_intrinsics=False)
    gt = gt_changes(meta[rescan], R, S, objs_R, load_objects(root, rescan), cfg)
    vocab, vocab_nyu = perception_vocab(G_ref, clip_vocab_table(root, cfg), int(cfg["perception"]["clip"]["vocab_top_n"]))
    src = PerceptionSource(make_labeler(cfg, vocab), vocab_nyu, cfg)
    with open_sequence(root, rescan) as seq:
        fids = select_frames(seq.frame_ids, int(cfg["frames"]["stride"]))
        ctx, info, _ = prepare_pair(root, R, objs_R, rels_R, seq, rescan, meta[rescan].T_align, cfg, fids, G_ref=G_ref)
        res = run_online(ctx, info, seq, src, cfg)
        obs = res.obs
        truth = perception_obs_truth(obs, gt, cfg, label_to_nyu40(root))
        area: dict[int, int] = {}
        for o in obs:
            for f in o.masks:
                area[f] = area.get(f, 0) + o.masks.area(f)
        if not area:
            return {"ref": ref, "rescan": rescan, "frame": None, "n_candidates": 0, "png": None}
        f_best = min(area, key=lambda f: (-area[f], f))
        img = seq.color(f_best)
        pose = info.T_align @ seq.pose(f_best)
    Hc, Wc = img.shape[:2]
    W, H = info.intrinsics.depth_wh
    up = np.rot90(img, k=-1)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(up)
    rows = []
    for k, o in enumerate(obs):
        if f_best not in o.masks:
            continue
        m = o.masks[f_best]
        r_idx = (np.arange(Hc) * H // Hc).clip(0, H - 1)
        c_idx = (np.arange(Wc) * W // Wc).clip(0, W - 1)
        mc = np.rot90(m[np.ix_(r_idx, c_idx)], k=-1)
        col = np.array(cmap(k % 10))
        rgba = np.zeros((*mc.shape, 4))
        rgba[mc] = [*col[:3], 0.45]
        ax.imshow(rgba)
        rr, cc = np.nonzero(mc)
        g = truth.get(o.token)
        tag = f"{o.token} {o.label}" + (f" = GT {g[1]}" if g else "")
        ax.text(cc.mean(), rr.min(), tag, color="white", fontsize=8, ha="center", va="bottom",
                bbox={"facecolor": col[:3], "alpha": 0.8, "pad": 1, "edgecolor": "none"})  # fmt: skip
        rows.append({"token": o.token, "label": o.label, "gt": list(g) if g else None, "px": int(m.sum())})
    # GT new (+) and moved-to (x) centroids, raw T_align (eval side)
    T_w2c = invert(pose)
    Kc = info.intrinsics.K_color
    for pts, mk, name in (([n.centroid for n in gt.new.values() if n.centroid is not None], "+", "GT new"),
                          ([o.centroid_new for o in gt.objects.values() if o.status == "moved" and o.centroid_new is not None],
                           "x", "GT moved-to")):  # fmt: skip
        if not pts:
            continue
        pc = apply(T_w2c, np.asarray(pts, float))
        pc = pc[pc[:, 2] > 0.1]
        if not len(pc):
            continue
        u = Kc[0, 0] * pc[:, 0] / pc[:, 2] + Kc[0, 2]
        v = Kc[1, 1] * pc[:, 1] / pc[:, 2] + Kc[1, 2]
        inside = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        ax.scatter(Hc - 1 - v[inside], u[inside], marker=mk, s=120, c="yellow", linewidths=2, label=name)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{rescan[:8]} frame {f_best}: {len(rows)} candidates", fontsize=9)
    ax.axis("off")
    png = out_dir / f"m4_overlay_{rescan[:8]}_frame{f_best:06d}.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"ref": ref, "rescan": rescan, "frame": int(f_best), "n_candidates": len(obs), "in_frame": rows,
            "png": str(png)}  # fmt: skip


def run_m4(cfg: dict[str, Any], run_id: str | None = None, sweep: bool = True) -> dict[str, Any]:
    """All M4 checks on `splits/dev_train.txt`; writes `outputs/<run_id>/metrics.json` + PNGs."""
    t0 = time.time()
    rid = run_id or new_run_id("m4")
    run_dir = outputs_root(cfg) / rid
    pairs = read_pairs(REPO_ROOT / "splits" / "dev_train.txt")
    res: dict[str, Any] = {"split": "dev_train", "n_pairs": len(pairs), "eval": {}, "sweep": [], "overlays": []}
    summaries = {}
    for method in ("B0", "B1"):
        r = run_eval(cfg, pairs, ["perception", "oracle"], method=method, run_id=f"{rid}/eval_{method}",
                     split_name="dev_train", workers=len(pairs))  # fmt: skip
        summaries[method] = r
        res["eval"][method] = {m: _pick(r["summary"][m]) for m in ("perception", "oracle")}
        res["eval"][method]["metrics_path"] = r["metrics_path"]
    if sweep:
        for door in (True, False):
            for mv in (3, 5, 10):
                c2 = copy.deepcopy(cfg)
                c2["perception"]["door_filter"] = door
                c2["perception"]["min_voxels"] = mv
                r = run_eval(c2, pairs, ["perception"], method="B1", run_id=f"{rid}/sweep_door{int(door)}_mv{mv}",
                             split_name="dev_train", workers=len(pairs))  # fmt: skip
                res["sweep"].append({"door_filter": door, "min_voxels": mv, **_pick(r["summary"]["perception"])})
    for ref, s in pairs:
        res["overlays"].append(_overlay(ref, s, cfg, run_dir))
    n_png = sum(o["png"] is not None for o in res["overlays"])
    res["acceptance"] = {
        "overlays_3_dev_frames": {"value": n_png, "pass": n_png >= 3},
        "candidate_recall_reported": {"value": res["eval"]["B1"]["perception"]["cand_recall_new"],
                                      "pass": True},  # fmt: skip
        "synthetic_and_no_leak": {"value": "pytest tests/test_perception.py tests/test_no_leak.py", "pass": None},
    }
    res["wall_time_s"] = time.time() - t0
    res["metrics_path"] = str(write_metrics(run_dir, {k: v for k, v in res.items() if k != "wall_time_s"}, cfg,
                                            extra={"wall_time_s": res["wall_time_s"]}))  # fmt: skip
    res["_summaries"] = summaries
    return res


def format_summary(res: dict[str, Any]) -> str:
    """Terminal report: B0/B1 tables, candidate recall, sweep, overlays."""

    def pct(m):
        return "  n/a " if m["value"] is None else f"{100 * m['value']:5.1f}"

    L = []
    for method in ("B0", "B1"):
        L.append(eval_summary(res["_summaries"][method]))
    L.append("== M4: candidate diagnostics on dev (perception, B1) ==")
    E = res["eval"]["B1"]["perception"]
    L.append(f"  candidates per pair {E['cand_per_pair']['value']:.1f} (n={E['cand_per_pair']['n']}); "
             f"explained by any GT instance {pct(E['cand_explained'])}%")  # fmt: skip
    L.append(f"  recall new      {pct(E['cand_recall_new'])}%  labelled {pct(E['cand_recall_new_labelled'])}%"
             f"  (n={E['cand_recall_new']['n']})")  # fmt: skip
    L.append(f"  recall moved-to {pct(E['cand_recall_moved'])}%  labelled {pct(E['cand_recall_moved_labelled'])}%"
             f"  (n={E['cand_recall_moved']['n']})")  # fmt: skip
    if res["sweep"]:
        L.append("  sweep (B1, perception)   cand/pair  rec.new  rec.moved  explained  new P  new R")
        for r in res["sweep"]:
            L.append(f"    door {'on ' if r['door_filter'] else 'off'} min_vox {r['min_voxels']:2d}       "
                     f"{r['cand_per_pair']['value']:5.1f}    {pct(r['cand_recall_new'])}    {pct(r['cand_recall_moved'])}"
                     f"     {pct(r['cand_explained'])}   {pct(r['new_P'])}  {pct(r['new_R'])}")  # fmt: skip
    L.append("  overlays:")
    for o in res["overlays"]:
        L.append(f"    {o['rescan'][:8]} frame {o['frame']}: {o['n_candidates']} candidates -> {o['png']}")
    L.append("  acceptance: " + ", ".join(f"{k}={'PASS' if a['pass'] else ('see pytest' if a['pass'] is None else 'FAIL')}"
                                         for k, a in res["acceptance"].items()))  # fmt: skip
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
