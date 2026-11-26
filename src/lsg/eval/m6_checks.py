"""M6 acceptance checks and report (`uv run lsg m6`).

1. OBJ <-> PLY correspondence over all references of `splits/download_textures.txt`:
   vertex / face counts, max vertex distance in the same order, identical faces.
2. Texture render check on the dev references' *own* sequences (T_align = I): 3 overlay PNGs
   per reference (observed | rendered | 50 % blend, upright), median patch similarity (DINOv2
   and the colour fallback), the frame-relative distance distribution over compared patches
   (p98 -> `appearance.tau_app`, per pixel / thin candidates) and over thing nodes (mean over a
   node's comparable pixels, p98 -> `appearance.tau_node`, thin-node votes): same-scan false
   rate 2 %. The dev rescans (method's refined `T_align`) are reported for comparison, split by
   GT status of the thing nodes.
3. Dev (`splits/dev_train.txt`), both modes: B2 vs FULL with appearance off / on (thin-object
   candidates off / on); readouts incl. `free_ev_removed` before / after.
4. Optional dev sweep: FULL `confidence.L_hit` in {0.5, 1, 2.5}.
5. Val (reported once, nothing tuned): FULL in both modes. Default: the milestone val subset
   (`splits/eval_val_small.txt`); `full_val=True` for all of `eval_val`.

The synthetic acceptance check (2 cm textured book removed from a table: missed by depth-only,
caught by depth + appearance) is `tests/test_appearance.py` (colour features, fast) and its
slow DINOv2 variant. Lives in `eval/` (reads GT and reference / rescan annotations).
"""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from lsg.change.pipeline import refined_alignment, select_frames
from lsg.classes import node_level
from lsg.eval.gt import gt_changes
from lsg.io.dssg import load_objects
from lsg.config import REPO_ROOT, cache_root, data_root, milestone_val_split, outputs_root
from lsg.eval.harness import format_summary as eval_summary
from lsg.eval.harness import run_eval
from lsg.eval.m5_checks import _pick
from lsg.geom.render import Renderer, TextureRenderer
from lsg.geom.transforms import invert
from lsg.io.scan3r import load_meta, load_scan, load_textured_mesh, open_sequence, read_obj, read_pairs, read_ply
from lsg.obs.base import SeqInfo
from lsg.perception.appearance import (
    AppearanceRunner,
    ColorFeatures,
    DinoV2Features,
    SimilarityCache,
    patch_index,
    resize,
    tau_node,
)
from lsg.runinfo import new_run_id, write_metrics

MODES = ("perception", "oracle")
REPORT_KEYS = ("removed_P", "removed_R", "removed_F1", "removed_R_thin", "moved_F1", "new_P", "new_R",
               "unchanged_acc", "unchanged_kept", "unchanged_kept_thin", "false_deletion_rate", "false_move_rate",
               "unknown_rate", "assoc_acc_strict", "free_ev_removed", "free_ev_removed_thin", "free_like_ev_removed",
               "free_ev_unchanged", "free_like_ev_unchanged", "free_ev_unchanged_removed", "conf_mean_unchanged",
               "conf_mean_removed", "conf_mean_absent", "cand_per_pair", "cand_appearance_per_pair",
               "cand_appearance_explained")  # fmt: skip


# ---- 1. OBJ <-> PLY ---------------------------------------------------------------------------


def _objply_one(args: tuple[str, str]) -> dict[str, Any]:
    root, ref = args
    d = Path(root) / ref
    try:
        V, F, UV = read_obj(d / "mesh.refined.v2.obj")
        ply = read_ply(d / "labels.instances.annotated.v2.ply")
    except FileNotFoundError as e:
        return {"ref": ref, "ok": False, "error": str(e)}
    PV = np.stack([ply.vertex["x"], ply.vertex["y"], ply.vertex["z"]], axis=1)
    same_n = V.shape == PV.shape and F.shape == ply.faces.shape
    dv = float(np.abs(V - PV).max()) if V.shape == PV.shape else None
    feq = bool(F.shape == ply.faces.shape and np.array_equal(F, ply.faces))
    return {"ref": ref, "n_v": [len(V), len(PV)], "n_f": [len(F), len(ply.faces)], "max_dv": dv,
            "faces_equal": feq, "uv_in_range": bool(UV.min() >= 0 and UV.max() <= 1),
            "ok": bool(same_n and feq and dv is not None and dv < 1e-6)}  # fmt: skip


def check_obj_ply(cfg: dict[str, Any], workers: int = 4) -> dict[str, Any]:
    """OBJ <-> PLY correspondence over the texture references."""
    refs = [l.strip() for l in (REPO_ROOT / cfg["splits"]["texture_list"]).read_text().splitlines() if l.strip()]
    root = str(data_root(cfg))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_objply_one, [(root, r) for r in refs]))
    return {"n": len(rows), "n_ok": sum(r["ok"] for r in rows), "bad": [r for r in rows if not r["ok"]],
            "max_dv": max((r["max_dv"] or 0.0) for r in rows if r.get("max_dv") is not None)}  # fmt: skip


# ---- 2. render check + tau_app calibration -------------------------------------------------------


def _sequence_stats(runner: AppearanceRunner, rr: Renderer, seq, T: np.ndarray, frames: list[int], cfg,
                    oids: list[int]) -> dict[str, Any]:  # fmt: skip
    """Raw similarity, frame-relative distance over compared patches, and per-node mean relative
    distance (thing nodes `oids` with >= `min_mask_px` comparable pixels) over `frames`."""
    sims, rel = [], []
    nodes: dict[int, list[float]] = {o: [] for o in oids}
    min_px = int(cfg["visibility"]["min_mask_px"])
    intr = seq.info
    for f in frames:
        fr = seq.frame(f)
        if not np.all(np.isfinite(fr.pose_c2w)):
            continue
        pose = T @ fr.pose_c2w
        d_exp, lab = rr.render(intr.K_depth, invert(pose), *intr.depth_wh)
        am = runner.frame(fr, pose, d_exp)
        ok = am.patch_ok
        if ok.sum() < int(cfg["appearance"].get("min_frame_patches", 1)):
            continue
        s = am.sim[ok]
        sims.append(s)
        dist = 1.0 - s.astype(np.float64)
        rel.append(dist - np.median(dist))
        for o in oids:
            m = (lab == o) & am.comparable_px
            if m.sum() >= min_px:
                nodes[o].append(float(am.rel_px[m].mean()))
    s = np.concatenate(sims) if sims else np.zeros(0)
    r = np.concatenate(rel) if rel else np.zeros(0)
    pct = (50, 90, 95, 98, 99)
    return {"frames": len(sims), "patches": int(s.size),
            "sim_median": float(np.median(s)) if s.size else None,
            "rel_dist_pct": dict(zip(map(str, pct), np.percentile(r, pct).round(4).tolist())) if r.size else None,
            "_rel": r, "_nodes": nodes}  # fmt: skip


def _pcts(v) -> dict[str, float] | None:
    v = np.asarray(list(v), float)
    return dict(zip(("50", "90", "98"), np.percentile(v, (50, 90, 98)).round(4).tolist())) if v.size else None


def _overlays(runner: AppearanceRunner, seq, frames: list[int], out_dir: Path, tag: str) -> list[str]:
    """observed | rendered | 50 % blend, upright, one PNG per frame."""
    from PIL import Image

    paths = []
    for f in frames:
        obs = resize(seq.color(f), runner.wh)
        exp, _ = runner.tr.render_color(runner.K, invert(seq.pose(f)), *runner.wh)
        blend = ((obs.astype(np.float32) + exp) / 2).astype(np.uint8)
        up = [np.rot90(x, k=-1) for x in (obs, exp, blend)]
        p = out_dir / f"m6_render_{tag}_{f:06d}.png"
        Image.fromarray(np.concatenate(up, axis=1)).save(p)
        paths.append(str(p))
    return paths


def render_check(cfg: dict[str, Any], out_dir: Path, frame_step: int = 15) -> dict[str, Any]:
    """M6 texture render check on the dev references' own sequences + tau_app calibration."""
    root = data_root(cfg)
    pairs = read_pairs(REPO_ROOT / "splits" / "dev_train.txt")
    meta = load_meta(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ac = cfg["appearance"]
    exts = {"dinov2": DinoV2Features(ac["model"], ac.get("device", "cuda")), "color": ColorFeatures()}
    res: dict[str, Any] = {"own": {}, "rescan": {}, "overlays": [], "frame_step": frame_step}
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in exts}
    pooled_nodes: dict[str, list[float]] = {k: [] for k in exts}
    for ref in sorted({r for r, _ in pairs}):
        R = load_scan(root, ref, with_intrinsics=False)
        things = sorted(o for o, i in R.instances.items() if node_level(i.nyu40) == "thing")
        objs_R = load_objects(root, ref)
        rr = Renderer(R.vertices, R.faces, R.face_labels)
        tr = TextureRenderer(load_textured_mesh(root, ref))
        cache = SimilarityCache(cache_root(cfg) / "appearance" if ac.get("cache", True) else None)
        with open_sequence(root, ref) as seq:
            ids = [i for i in seq.frame_ids if np.all(np.isfinite(seq.pose(i)))]
            info = SeqInfo(ref, root, seq.info, np.eye(4), ids)
            for name, ext in exts.items():
                c2 = copy.deepcopy(cfg)
                c2["appearance"]["features"] = name
                run = AppearanceRunner(tr, ext, c2, ref, cache)
                run.begin(info, seq)
                st = _sequence_stats(run, rr, seq, np.eye(4), ids[::frame_step], c2, things)
                pooled[name].append(st.pop("_rel"))
                nm = [x for v in st.pop("_nodes").values() for x in v]
                pooled_nodes[name] += nm
                st["node_mean_pct"] = _pcts(nm)
                res["own"].setdefault(ref, {})[name] = st
                if name == "dinov2":
                    res["dino_device"] = ext.device or "cached"
                    pick = [ids[len(ids) // 5], ids[len(ids) // 2], ids[4 * len(ids) // 5]]
                    res["overlays"] += _overlays(run, seq, pick, out_dir, ref[:8])
        for r_, s in pairs:
            if r_ != ref:
                continue
            gt = gt_changes(meta[s], R, load_scan(root, s, with_intrinsics=False), objs_R, load_objects(root, s), cfg)
            with open_sequence(root, s) as seq:
                fids = select_frames(seq.frame_ids, int(cfg["frames"]["stride"]), 1.0, 0, key=s)
                T, _ = refined_alignment(seq, R, s, meta[s].T_align, cfg, fids)
                info = SeqInfo(s, root, seq.info, T, fids)
                for name, ext in exts.items():
                    c2 = copy.deepcopy(cfg)
                    c2["appearance"]["features"] = name
                    run = AppearanceRunner(tr, ext, c2, ref, cache)
                    run.begin(info, seq)
                    st = _sequence_stats(run, rr, seq, T, fids[:: max(1, frame_step // 3)], c2, things)
                    r = st.pop("_rel")
                    nodes = st.pop("_nodes")
                    st["mismatch_share_at_tau"] = float((r > _tau(c2)).mean()) if r.size else None
                    tn = tau_node(c2)
                    st["node_mean_by_status"] = {}
                    for status in ("unchanged", "moved", "removed", "absent_unlisted"):
                        v = [x for o, l in nodes.items() if o in gt.objects and gt.objects[o].status == status for x in l]
                        if v:
                            st["node_mean_by_status"][status] = {**_pcts(v), "n": len(v),
                                                                 "share_above_tau_node": float(np.mean(np.asarray(v) > tn))}  # fmt: skip
                    res["rescan"].setdefault(s, {})[name] = st
    res["tau_app_p98"] = {k: round(float(np.percentile(np.concatenate(v), 98)), 4) for k, v in pooled.items() if v}
    res["tau_node_p98"] = {k: round(float(np.percentile(v, 98)), 4) for k, v in pooled_nodes.items() if v}
    res["tau_app_config"] = {k: _tau({"appearance": {**ac, "features": k}}) for k in exts}
    res["tau_node_config"] = {k: tau_node({"appearance": {**ac, "features": k}}) for k in exts}
    return res


def _tau(cfg) -> float:
    from lsg.perception.appearance import tau_app

    return tau_app(cfg)


# ---- 3-5. dev / sweep / val -----------------------------------------------------------------------


def _variant(cfg: dict[str, Any], app: bool, thin_cand: bool = True) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["appearance"]["enabled"] = app
    c["appearance"]["thin_candidates"] = thin_cand
    return c


def run_m6(cfg: dict[str, Any], run_id: str | None = None, sweep: bool = True, val: bool = True,
           objply: bool = True, full_val: bool = False) -> dict[str, Any]:  # fmt: skip
    """All M6 checks; writes `outputs/<run_id>/metrics.json`."""
    t0 = time.time()
    rid = run_id or new_run_id("m6")
    run_dir = outputs_root(cfg) / rid
    res: dict[str, Any] = {"objply": None, "render": None, "dev": {}, "sweep": [], "val": None}
    if objply:
        res["objply"] = check_obj_ply(cfg)
    rc = render_check(cfg, run_dir)
    res["render"] = {k: v for k, v in rc.items()}

    pairs = read_pairs(REPO_ROOT / "splits" / "dev_train.txt")
    variants = {"B2": ("B2", cfg), "FULL_depth": ("FULL", _variant(cfg, False)),
                "FULL_app_nocand": ("FULL", _variant(cfg, True, False)), "FULL": ("FULL", _variant(cfg, True, True))}
    summaries: dict[str, Any] = {}
    for name, (method, c) in variants.items():
        r = run_eval(c, pairs, list(MODES), method=method, run_id=f"{rid}/dev_{name}", split_name="dev_train",
                     workers=len(pairs))  # fmt: skip
        summaries[f"dev_{name}"] = r
        res["dev"][name] = {m: _pick(r["summary"][m], REPORT_KEYS) for m in MODES}
        res["dev"][name]["metrics_path"] = r["metrics_path"]

    if sweep:
        for L in (0.5, 1.0, 2.5):
            c = _variant(cfg, True, True)
            c["confidence"]["L_hit"] = L
            c["eval"]["save_graphs"] = False
            r = run_eval(c, pairs, list(MODES), method="FULL", run_id=f"{rid}/sweep_Lhit{L}", split_name="dev_train",
                         workers=len(pairs))  # fmt: skip
            res["sweep"].append({"L_hit": L, **{m: _pick(r["summary"][m], REPORT_KEYS) for m in MODES}})

    if val:
        vs = milestone_val_split(cfg, full_val)
        r = run_eval(cfg, read_pairs(vs), list(MODES), method="FULL", run_id=f"{rid}/val_FULL", split_name=vs.stem)
        summaries["val_FULL"] = r
        res["val"] = {m: _pick(r["summary"][m], REPORT_KEYS) for m in MODES}
        res["val"]["metrics_path"] = r["metrics_path"]

    acc: dict[str, Any] = {}
    if res["objply"] is not None:
        o = res["objply"]
        acc["obj_ply_correspondence"] = {"value": f"{o['n_ok']}/{o['n']}", "pass": o["n_ok"] == o["n"]}
    acc["render_overlays_written"] = {"value": len(rc["overlays"]), "pass": len(rc["overlays"]) >= 3}
    own = [v["dinov2"]["sim_median"] for v in rc["own"].values()]
    acc["median_patch_similarity_reported"] = {"value": own, "pass": all(x is not None for x in own)}
    n_inv = sum(res["dev"][k][m]["graph_invalid"] for k in variants for m in MODES)
    acc["graphs_validate"] = {"value": n_inv, "pass": n_inv == 0}
    res["acceptance"] = acc
    res["wall_time_s"] = time.time() - t0
    res["metrics_path"] = str(write_metrics(run_dir, {k: v for k, v in res.items() if k != "wall_time_s"}, cfg,
                                            extra={"wall_time_s": res["wall_time_s"],
                                                   "dino_device": rc.get("dino_device")}))  # fmt: skip
    res["_summaries"] = summaries
    return res


def format_summary(res: dict[str, Any]) -> str:
    """Terminal report: OBJ/PLY, render check, dev comparison, sweep, val, acceptance."""

    def pct(m):
        v = m["value"] if isinstance(m, dict) else m
        return "  n/a" if v is None else f"{100 * v:5.1f}"

    L = []
    if res.get("_summaries", {}).get("val_FULL"):
        L.append(eval_summary(res["_summaries"]["val_FULL"]))
    o = res.get("objply")
    if o:
        L.append(f"== M6: OBJ <-> PLY: {o['n_ok']} / {o['n']} references identical (max vertex diff {o['max_dv']:.2e})")
        for b in o["bad"][:10]:
            L.append(f"   mismatch: {json.dumps(b)}")
    rc = res["render"]
    L.append(f"== M6: texture render check (own sequences, every {rc['frame_step']}th frame; device {rc.get('dino_device')}) ==")
    for grp in ("own", "rescan"):
        for sid, d in rc[grp].items():
            for feat, st in d.items():
                extra = f"  mismatch share at tau {st['mismatch_share_at_tau']:.3f}" if "mismatch_share_at_tau" in st else ""
                L.append(f"  {grp:6s} {sid[:8]} {feat:6s} frames {st['frames']:3d} patches {st['patches']:6d} "
                         f"median sim {st['sim_median']:.3f}  rel. dist pct50/90/95/98/99 "
                         f"{list((st['rel_dist_pct'] or {}).values())}{extra}")  # fmt: skip
                if st.get("node_mean_pct"):
                    L.append(f"         node means pct50/90/98 {list(st['node_mean_pct'].values())}")
                for status, q in (st.get("node_mean_by_status") or {}).items():
                    L.append(f"         node means {status:16s} n={q['n']:4d} pct50/90/98 {[q['50'], q['90'], q['98']]}"
                             f"  share > tau_node {q['share_above_tau_node']:.3f}")  # fmt: skip
    L.append(f"  tau_app p98 (pooled own sequences): {rc['tau_app_p98']}   config: {rc['tau_app_config']}")
    L.append(f"  tau_node p98 (pooled own sequences): {rc['tau_node_p98']}   config: {rc['tau_node_config']}")
    L.append(f"  overlays: {len(rc['overlays'])} PNGs, e.g. {rc['overlays'][0] if rc['overlays'] else '-'}")
    cols = ("removed_P", "removed_R", "removed_R_thin", "free_ev_removed", "free_like_ev_removed", "free_ev_unchanged_removed",
            "unchanged_kept", "unchanged_acc", "false_deletion_rate", "new_P", "new_R", "moved_F1",
            "conf_mean_unchanged", "conf_mean_removed")  # fmt: skip
    L.append("== M6: dev comparison ==")
    L.append("  variant          mode        " + " ".join(f"{c[:10]:>10s}" for c in cols) + "  cand/pair app/pair")
    for name, d in res["dev"].items():
        for m in MODES:
            E = d[m]
            L.append(f"  {name:16s} {m:11s} " + " ".join(f"{pct(E[c]):>10s}" for c in cols)
                     + f"  {E['cand_per_pair']['value'] or 0:6.1f}  {E['cand_appearance_per_pair']['value'] or 0:6.1f}")  # fmt: skip
        L.append("    supports: " + ", ".join(f"{c} n={d['oracle'][c]['n']}" for c in ("removed_R", "removed_R_thin",
                                                                                     "free_ev_unchanged_removed")))  # fmt: skip
    for r in res["sweep"]:
        for m in MODES:
            E = r[m]
            L.append(f"  sweep L_hit {r['L_hit']:<4} {m:11s} " + " ".join(f"{pct(E[c]):>10s}" for c in cols))
    L.append("  acceptance: " + ", ".join(f"{k}={'PASS' if a['pass'] else 'FAIL'} ({a['value']})"
                                         for k, a in res["acceptance"].items()))  # fmt: skip
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
