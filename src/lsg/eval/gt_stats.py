"""M1 acceptance: GT statistics, displacement histogram, reference-rule sanity.

(1) val counts reproduce the expected raw counts (rigid 390 / removed 96 / new 224 raw; 171 scored);
(2) GT displacement histogram (val + dev) saved, τ_eval_move from the config;
(3) geometric support rules vs. reference 3DSSG on **reference** scans: dev refs (tuning)
    and val refs (report only);
(4) `splits/download_textures.txt` lists the 72 references of dev + val + relchange.
The relchange split gets metadata + 3DSSG counts only (tier-2 meshes not downloaded).

Run: `uv run lsg m1 [--config configs/default.yaml]`.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lsg.classes import node_level
from lsg.config import REPO_ROOT, data_root, outputs_root
from lsg.eval.gt import PARENT_CHANGE_KINDS, gt_changes, gt_parent_changes
from lsg.graph.build import build_reference_graph
from lsg.graph.model import FLOOR_ID, ref_node_id
from lsg.graph.relations import FLOOR, gt_support
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import Meta, load_meta, load_scan, read_pairs
from lsg.io.splits import reference_ids
from lsg.runinfo import new_run_id, write_metrics


def _group(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """`{reference: [rescans]}` in file order (one reference scan loaded at a time)."""
    out: dict[str, list[str]] = {}
    for r, s in pairs:
        out.setdefault(r, []).append(s)
    return out


def _name(split_file: str) -> str:
    """`splits/eval_val.txt` -> `eval_val`."""
    return Path(split_file).stem


# --------------------------------------------------------------------------------------------
# (3) reference-rule sanity
# --------------------------------------------------------------------------------------------


def rule_records(R, objs_R, rels_R, cfg) -> list[dict[str, Any]]:
    """Geometric support parent vs. reference 3DSSG parent, per thing with a GT parent.

    GT parents are those of `gt_support` restricted to graph nodes or the floor; things with
    only structure parents (excluded) or no GT parent are skipped.
    """
    g = build_reference_graph(R, objs_R, None, "geometric", cfg)
    pred = g.support_parent()
    level = {n.id: n.level for n in g.nodes}
    label = {n.id: n.label for n in g.nodes}
    gs = gt_support(rels_R, objs_R)
    recs = []
    for n in g.nodes:
        if n.level != "thing":
            continue
        oid = int(n.id[1:])
        if oid in gs.excluded:
            continue
        gp = {
            (FLOOR_ID if p == FLOOR else ref_node_id(p)): t
            for p, t in gs.parents.get(oid, {}).items()
            if p == FLOOR or ref_node_id(p) in level
        }
        if not gp:
            continue
        pp, pt = pred.get(n.id, (None, None))
        recs.append(
            {
                "scan": R.scan_id,
                "id": n.id,
                "label": n.label,
                "gt": {k: (v, label.get(k, k)) for k, v in gp.items()},
                "pred": pp,
                "pred_label": label.get(pp, pp) if pp else None,
                "pred_type": pt,
                "parent_ok": pp in gp,
                "type_ok": pp in gp and gp[pp] == pt,
                "gt_floor": FLOOR_ID in gp,
            }
        )
    return recs


def rule_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Parent accuracy, type accuracy (given a correct parent) and error kinds.

    Reported for all things with a GT parent and for those whose GT parents exclude the floor.
    """

    def summ(rs: list[dict[str, Any]]) -> dict[str, Any]:
        """Accuracy block for one subset."""
        n = len(rs)
        ok = [r for r in rs if r["parent_ok"]]
        err = Counter()
        for r in rs:
            if r["parent_ok"]:
                continue
            if r["pred"] is None:
                err["no_prediction"] += 1
            elif r["pred"] == FLOOR_ID:
                err["pred_floor_gt_object"] += 1
            elif r["gt_floor"]:
                err["pred_object_gt_floor"] += 1
            else:
                err["wrong_object"] += 1
        return {
            "n": n,
            "parent_acc": len(ok) / n if n else float("nan"),
            "type_acc_given_parent": (sum(r["type_ok"] for r in ok) / len(ok)) if ok else float("nan"),
            "errors": dict(err),
        }

    return {
        "with_floor": summ(recs),
        "without_floor": summ([r for r in recs if not r["gt_floor"]]),
        "n_scans": len({r["scan"] for r in recs}),
    }


# --------------------------------------------------------------------------------------------
# (1)+(2) GT passes
# --------------------------------------------------------------------------------------------


def _full_pass(root, meta: Meta, pairs, cfg, rules: bool) -> dict[str, Any]:
    """Load every reference + rescan of a split: GT counts, displacements, parent changes.

    With `rules`, also run the reference-rule sanity on each reference scan.
    """
    counts: Counter = Counter()
    pc: Counter = Counter()
    disp: list[dict[str, Any]] = []
    recs: list[dict[str, Any]] = []
    size_new: Counter = Counter()
    for ref, rescans in _group(pairs).items():
        R = load_scan(root, ref, with_intrinsics=False)
        objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
        gs_R = gt_support(rels_R, objs_R)
        if rules:
            recs += rule_records(R, objs_R, rels_R, cfg)
        for s in rescans:
            S = load_scan(root, s, with_intrinsics=False)
            objs_S = load_objects(root, s)
            gt = gt_changes(meta[s], R, S, objs_R, objs_S, cfg)
            counts.update(gt.counts())
            size_new.update(n.size for n in gt.new.values())
            for o in gt.objects.values():
                if o.in_rigid:
                    disp.append({"ref": ref, "rescan": s, "id": o.id, "label": o.label,
                                 "level": o.level, "displacement": o.displacement,
                                 "status": o.status, "reason": o.reason})  # fmt: skip
            gs_S = gt_support(load_relations(root, s), objs_S)
            pc.update(gt_parent_changes(meta[s], objs_R, gs_R, gs_S).values())
            del S
        del R
    counts["pairs"] = len(pairs)
    counts["refs"] = len(_group(pairs))
    return {
        "counts": dict(counts),
        "new_by_size": dict(size_new),
        "parent_change": {k: pc.get(k, 0) for k in PARENT_CHANGE_KINDS},
        "displacements": disp,
        "rule_records": recs,
    }


def _meta_pass(root, meta: Meta, pairs) -> dict[str, Any]:
    """Counts that need only 3RScan.json + 3DSSG (no meshes): for not-yet-downloaded splits."""
    c: Counter = Counter()
    pc: Counter = Counter()
    for r, s in pairs:
        sm = meta[s]
        objs_R, objs_S = load_objects(root, r), load_objects(root, s)
        c["rigid_raw"] += len(sm.rigid)
        c["rigid_bare"] += sum(e.T is None for e in sm.rigid)
        c["removed_raw"] += len(sm.removed)
        c["nonrigid_raw"] += len(sm.nonrigid)
        c["new_raw_3dssg"] += len(set(objs_S) - set(objs_R))
        c["moved_things_raw"] += sum(
            1 for e in sm.rigid if e.id in objs_R and node_level(int(objs_R[e.id]["nyu40"])) == "thing"
        )
        gs_R = gt_support(load_relations(root, r), objs_R)
        gs_S = gt_support(load_relations(root, s), objs_S)
        pc.update(gt_parent_changes(sm, objs_R, gs_R, gs_S).values())
    c["pairs"] = len(pairs)
    c["refs"] = len(_group(pairs))
    return {"counts": dict(c), "parent_change": {k: pc.get(k, 0) for k in PARENT_CHANGE_KINDS}}


def displacement_summary(values: list[float | None], tau: float, buckets: list[float]) -> dict[str, Any]:
    """Counts below τ_eval_move, per displacement bucket, bare entries (None), and quantiles."""
    d = np.array([v for v in values if v is not None], float)
    edges = [tau, *[b for b in buckets if b > tau]]
    out: dict[str, Any] = {"n": len(values), "n_no_transform": sum(v is None for v in values)}
    out[f"below_tau ({tau})"] = int((d < tau).sum())
    for lo, hi in zip(edges, [*edges[1:], np.inf]):
        out[f"[{lo}, {hi})"] = int(((d >= lo) & (d < hi)).sum())
    if d.size:
        out["quantiles"] = {f"p{q}": float(np.percentile(d, q)) for q in (5, 10, 25, 50, 75, 90, 95)}
    return out


def plot_displacements(by_split: dict[str, list[float]], tau: float, buckets: list[float], out: Path) -> Path:
    """Small-multiple histograms (one panel per split, shared log-x) of GT displacement.

    A dashed line marks τ_eval_move, dotted lines the displacement bucket edges; each bucket is
    labelled with its count.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 (validated)
    bins = np.logspace(np.log10(0.005), np.log10(10.0), 37)
    edges = [tau, *[b for b in buckets if b > tau]]
    fig, axes = plt.subplots(len(by_split), 1, figsize=(7.5, 2.4 * len(by_split) + 0.6),
                             sharex=True, facecolor=surface, squeeze=False)  # fmt: skip
    for ax, (name, vals), col in zip(axes[:, 0], by_split.items(), colors):
        d = np.array([v for v in vals if v is not None], float)
        ax.set_facecolor(surface)
        ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, color=col, edgecolor=surface, linewidth=1.0)
        ax.set_xscale("log")
        ymax = ax.get_ylim()[1] * 1.45  # headroom for the bucket and threshold labels
        ax.set_ylim(0, ymax)
        ax.axvline(tau, color=ink, linestyle="--", linewidth=1.2)
        for b in edges[1:]:
            ax.axvline(b, color=muted, linestyle=":", linewidth=1.0)
        segs = [(bins[0], tau)] + list(zip(edges, [*edges[1:], bins[-1]]))
        for k, (lo, hi) in enumerate(segs):
            hi_n = np.inf if k == len(segs) - 1 else hi  # last bucket is open-ended
            n = int(((d >= lo) & (d < hi_n)).sum()) if k else int((d < hi).sum())
            ax.text(np.sqrt(lo * hi), ymax * 0.92, f"n = {n}", ha="center", va="top", fontsize=8, color=ink2)
        ax.set_title(f"{name}: {len(d)} moved things/anchors (unique rigid ids with a transform)",
                     loc="left", fontsize=10, color=ink)
        ax.text(tau * 0.93, ymax * 0.80, f"τ_eval_move = {tau} m\nbelow: dont_care", ha="right",
                va="top", fontsize=8, color=ink)  # fmt: skip
        ax.set_ylabel("objects", color=ink2, fontsize=9)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=ink2, labelsize=8)
    axes[-1, 0].set_xlabel("GT displacement ‖T_align·T_rigid·c_R − c_R‖ [m] (log scale)", color=ink2, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=surface)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


def run_m1(cfg: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Run the M1 checks, write `outputs/<run_id>/metrics.json` (+ histogram PNG).

    Returns:
        Results plus `"acceptance"` (pass flag per check) and `"metrics_path"`.
    """
    t0 = time.time()
    root = data_root(cfg)
    meta = load_meta(root)
    m1, ecfg = cfg["m1"], cfg["eval"]
    run_dir = outputs_root(cfg) / (run_id or new_run_id("m1"))
    tau, buckets = float(ecfg["tau_eval_move"]), [float(b) for b in ecfg["disp_buckets"]]

    full_splits = list(dict.fromkeys([m1["count_split"], *m1["hist_splits"], m1["rule_tune_split"]]))
    rule_splits = {m1["rule_tune_split"], m1["rule_report_split"]}
    res: dict[str, Any] = {"splits": {}, "tau_eval_move": tau}
    for sf in full_splits:
        print(f"[m1] {_name(sf)} (meshes)", flush=True)
        p = _full_pass(root, meta, read_pairs(REPO_ROOT / sf), cfg, rules=sf in rule_splits)
        recs = p.pop("rule_records")
        if sf in rule_splits:
            p["rules"] = rule_summary(recs)
            p["rule_records"] = recs
        p["displacement"] = displacement_summary([d["displacement"] for d in p["displacements"]], tau, buckets)
        res["splits"][_name(sf)] = p
    for sf in m1["meta_only_splits"]:
        print(f"[m1] {_name(sf)} (metadata + 3DSSG only)", flush=True)
        res["splits"][_name(sf)] = _meta_pass(root, meta, read_pairs(REPO_ROOT / sf))

    hist = {_name(sf): [d["displacement"] for d in res["splits"][_name(sf)]["displacements"]]
            for sf in m1["hist_splits"]}  # fmt: skip
    png = plot_displacements(hist, tau, buckets, run_dir / "gt_displacement_hist.png")

    # ---- texture list
    tex_path = REPO_ROOT / cfg["splits"]["texture_list"]
    tex = tex_path.read_text(encoding="utf-8").split() if tex_path.is_file() else []
    want = reference_ids([REPO_ROOT / f for f in cfg["splits"]["texture_sources"]])

    # ---- acceptance
    acc: dict[str, Any] = {}
    vc = res["splits"][_name(m1["count_split"])]["counts"]
    exp = m1["expected_counts"]
    got = {"rigid": vc.get("rigid_raw"), "removed": vc.get("removed_raw"), "new_raw": vc.get("new_raw")}
    acc["1_counts"] = {
        "gate": f"{_name(m1['count_split'])} raw counts == expected {exp}",
        "values": {**got, "new_scored (things+anchors)": vc.get("new")},
        "pass": all(got[k] == v for k, v in exp.items()),
    }
    acc["2_displacement"] = {
        "gate": "GT displacement histogram saved; tau_eval_move fixed in configs/default.yaml",
        "tau_eval_move": tau,
        "png": str(png),
        "per_split": {k: res["splits"][k]["displacement"] for k in hist},
        "pass": png.is_file() and "tau_eval_move" in ecfg,
    }
    rules = {_name(sf): res["splits"][_name(sf)]["rules"] for sf in sorted(rule_splits)}
    acc["3_rule_sanity"] = {
        "gate": "reference-scan rule parent accuracy reported (tune split + report split)",
        "values": {k: {"parent_acc": v["with_floor"]["parent_acc"],
                       "parent_acc_without_floor": v["without_floor"]["parent_acc"],
                       "n": v["with_floor"]["n"]} for k, v in rules.items()},  # fmt: skip
        "pass": all(v["with_floor"]["n"] > 0 for v in rules.values()),
    }
    acc["4_texture_list"] = {
        "gate": f"{cfg['splits']['texture_list']} lists the {m1['expected_texture_refs']} references",
        "n": len(tex),
        "pass": len(tex) == m1["expected_texture_refs"] and sorted(tex) == want,
    }
    res["acceptance"] = acc
    res["wall_time_s"] = time.time() - t0
    res["metrics_path"] = str(write_metrics(run_dir, res, cfg))
    return res


def format_summary(res: dict[str, Any]) -> str:
    """Human-readable report of `run_m1`."""
    L = ["== M1: GT statistics, displacement histogram, reference-rule sanity =="]
    pct = lambda x: f"{100 * x:5.1f}%" if x == x else "  n/a"  # noqa: E731
    keys = ["pairs", "refs", "rigid_raw", "rigid_bare", "removed_raw", "nonrigid_raw", "new_raw",
            "new_raw_3dssg", "new", "moved_things_raw", "moved", "removed", "unchanged",
            "absent_unlisted", "dont_care", "dont_care_removed_but_present", "dont_care_nonrigid",
            "dont_care_small_move"]  # fmt: skip
    L.append("(1) GT counts per split")
    for name, sp in res["splits"].items():
        c = sp["counts"]
        L.append(f"    {name}: " + "  ".join(f"{k}={c[k]}" for k in keys if k in c))
        if "new_by_size" in sp:
            L.append(f"        new by size: {sp['new_by_size']}")
        L.append(f"        moved things, parent change (3DSSG ref vs rescan): {sp['parent_change']}")
    L.append(f"(2) GT displacement (tau_eval_move = {res['tau_eval_move']} m)")
    for name, d in res["acceptance"]["2_displacement"]["per_split"].items():
        q = d.get("quantiles", {})
        bk = "  ".join(f"{k}={v}" for k, v in d.items() if k not in ("quantiles",))
        L.append(f"    {name}: {bk}")
        L.append("        quantiles [m]: " + "  ".join(f"{k}={v:.3f}" for k, v in q.items()))
    L.append(f"    histogram: {res['acceptance']['2_displacement']['png']}")
    L.append("(3) geometric support rules vs. reference 3DSSG (reference scans, things with a GT parent)")
    for name, sp in res["splits"].items():
        if "rules" not in sp:
            continue
        r = sp["rules"]
        for sub in ("with_floor", "without_floor"):
            b = r[sub]
            L.append(
                f"    {name:22s} {sub:13s} n={b['n']:4d}  parent acc {pct(b['parent_acc'])}"
                f"  type acc|parent {pct(b['type_acc_given_parent'])}  errors {b['errors']}"
            )
    L.append("-- acceptance --")
    for k, a in res["acceptance"].items():
        L.append(f"    {k}: pass={a['pass']}   {a['gate']}")
        if "values" in a:
            L.append(f"        {a['values']}")
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
