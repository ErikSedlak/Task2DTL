"""M0 acceptance checks (DESIGN §9). Lives in `eval/` because it reads rescan labels and `rigid`.

(1) rendered vs. observed depth on dev rescans (gate: reference mesh at refined T_align @ pose)
(2) aligned rescan structure vs. reference structure: point-to-surface distance per hypothesis
(3) `rigid` transform semantics: Chamfer under H0..H4
(4) overlay PNGs, upright, 5 panels: color | rescan own-mesh labels (oracle view) |
    reference-mesh labels at refined T_align | |Δ| depth at raw T_align | |Δ| at refined T_align.
    Heat ramp: blue 0 -> yellow 10 cm -> red >= 20 cm; black = invalid depth or no render hit.

Run: `uv run lsg m0 [--config configs/default.yaml]`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from lsg.classes import STRUCTURE_NYU40
from lsg.config import REPO_ROOT, data_root, outputs_root
from lsg.geom.align import refine_from_sequence
from lsg.geom.render import Renderer
from lsg.geom.transforms import apply, invert, rotation_angle_deg
from lsg.io.scan3r import Meta, Scan, SequenceReader, load_meta, load_scan, open_sequence, read_pairs
from lsg.runinfo import new_run_id, write_metrics

# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


@dataclass
class PairCtx:
    ref: str
    rescan: str
    R: Scan
    S: Scan
    rR: Renderer
    rS: Renderer
    seq: SequenceReader
    T_raw: np.ndarray
    T_ref: np.ndarray
    align_info: dict[str, Any]


def _stats(a: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {"n": 0, "median": float("nan"), "p90": float("nan"), "mean": float("nan")}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(np.mean(a)),
    }


def chamfer(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(symmetric mean Chamfer, median one-sided B->A) in metres."""
    dab = cKDTree(B).query(A)[0]
    dba = cKDTree(A).query(B)[0]
    return 0.5 * (float(dab.mean()) + float(dba.mean())), float(np.median(dba))


def label_colors(ids: np.ndarray) -> np.ndarray:
    """Deterministic colour per objectId (same id -> same colour in ref and rescan)."""
    ids = ids.astype(np.uint64)
    h = (ids * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
    rgb = np.stack([(h >> np.uint64(s)) & np.uint64(0xFF) for s in (16, 8, 0)], axis=-1)
    rgb = (rgb.astype(np.float32) * 0.75 + 64).astype(np.uint8)
    return np.where((ids == 0)[..., None], 0, rgb).astype(np.uint8)


def heat(err: np.ndarray, valid: np.ndarray, vmax: float = 0.20) -> np.ndarray:
    """|Δ| -> RGB ramp blue (0) -> yellow -> red (>= vmax); invalid black."""
    x = np.clip(err / vmax, 0, 1)
    r = np.clip(2 * x, 0, 1)
    g = np.clip(2 - 2 * x, 0, 1) * np.clip(2 * x, 0, 1) + (x < 0.5) * 0.3 * (1 - 2 * x)
    b = np.clip(1 - 2 * x, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return np.where(valid[..., None], rgb, 0).astype(np.uint8)


def _resize_nn(img: np.ndarray, W: int, H: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((W, H), Image.NEAREST))


# --------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------


def check_depth(ctx: PairCtx, frame_step: int) -> dict[str, Any]:
    seq = ctx.seq
    K = seq.info.K_depth
    W, H = seq.info.depth_wh
    errs: dict[str, list[np.ndarray]] = {"own": [], "ref_raw": [], "ref_refined": []}
    n = 0
    t0 = time.time()
    for i in seq.frame_ids[::frame_step]:
        f = seq.frame(i)
        if not np.all(np.isfinite(f.pose_c2w)):
            continue
        n += 1
        for key, r, T in (
            ("own", ctx.rS, np.eye(4)),
            ("ref_raw", ctx.rR, ctx.T_raw),
            ("ref_refined", ctx.rR, ctx.T_ref),
        ):
            d, _ = r.render(K, invert(T @ f.pose_c2w), W, H)
            v = (f.depth > 0) & (d > 0)
            errs[key].append(np.abs(f.depth[v] - d[v]))
    out = {k: _stats(np.concatenate(v)) for k, v in errs.items()}
    out["n_frames"] = n
    out["sec_per_frame_3renders"] = (time.time() - t0) / max(n, 1)
    return out


def check_structure(ctx: PairCtx) -> dict[str, Any]:
    S, R = ctx.S, ctx.R
    s_mask = np.isin(S.vertex_nyu40, list(STRUCTURE_NYU40))
    f_mask = np.isin(R.face_nyu40, list(STRUCTURE_NYU40))
    r_struct = Renderer(R.vertices, R.faces[f_mask])
    P = S.vertices[s_mask]
    T_mm = ctx.T_raw.copy()
    T_mm[:3, 3] *= 0.001
    hyps = {
        "T_align": ctx.T_raw,
        "T_align_inv": invert(ctx.T_raw),
        "T_align_t_x0.001 (mm)": T_mm,
        "identity": np.eye(4),
        "T_refined": ctx.T_ref,
    }
    out = {name: _stats(r_struct.distance(apply(T, P))) for name, T in hyps.items()}
    out["n_points"] = int(len(P))
    return out


def check_rigid(ctx: PairCtx, meta: Meta) -> list[dict[str, Any]]:
    rows = []
    for e in meta[ctx.rescan].rigid:
        row: dict[str, Any] = {"rescan": ctx.rescan, "id": e.id}
        if e.T is None:
            row["skip"] = "bare-int entry (no transform)"
            rows.append(row)
            continue
        PR, PS = ctx.R.instance_points(e.id), ctx.S.instance_points(e.id)
        inst = ctx.R.instances.get(e.id)
        row.update(
            label=inst.label if inst else "",
            n_ref=int(len(PR)),
            n_rescan=int(len(PS)),
            t_norm=float(np.linalg.norm(e.T[:3, 3])),
            rot_deg=rotation_angle_deg(e.T),
        )
        if len(PR) == 0 or len(PS) == 0:
            row["skip"] = "instance missing in one scan"
            rows.append(row)
            continue
        Ti = invert(e.T)
        Ta = ctx.T_raw
        hyps = {
            "H0 no motion: P_R ~ T_align.P_S": (PR, apply(Ta, PS)),
            "H1 T.P_R ~ T_align.P_S (ref frame)": (apply(e.T, PR), apply(Ta, PS)),
            "H2 T.P_R ~ P_S (ref -> rescan frame)": (apply(e.T, PR), PS),
            "H3 T^-1.P_R ~ T_align.P_S": (apply(Ti, PR), apply(Ta, PS)),
            "H4 T^-1.P_R ~ P_S": (apply(Ti, PR), PS),
        }
        ch = {}
        for name, (A, B) in hyps.items():
            c, med = chamfer(A, B)
            ch[name.split()[0]] = {"chamfer": c, "median_rescan_to_ref": med}
        row["hyp"] = ch
        row["best"] = min(ch, key=lambda k: ch[k]["chamfer"])
        # displacement of the reference centroid in the reference frame, M_ref = T_align . T_rigid
        c_R = PR.mean(axis=0)
        row["gt_displacement_ref_frame"] = float(np.linalg.norm(apply(Ta @ e.T, c_R[None])[0] - c_R))
        rows.append(row)
    return rows


def make_overlays(ctx: PairCtx, n_frames: int, out_dir: Path) -> list[str]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    seq = ctx.seq
    Kc = seq.info.K_color
    Wc, Hc = seq.info.color_wh
    Kd = seq.info.K_depth
    Wd, Hd = seq.info.depth_wh
    ids = seq.frame_ids
    picks = [ids[int(round(q * (len(ids) - 1)))] for q in np.linspace(0.2, 0.8, n_frames)]
    written = []
    for i in picks:
        f = seq.frame(i)
        color = seq.color(i)
        _, lab_own = ctx.rS.render(Kc, invert(f.pose_c2w), Wc, Hc)
        _, lab_ref = ctx.rR.render(Kc, invert(ctx.T_ref @ f.pose_c2w), Wc, Hc)
        panels = [color]
        for lab in (lab_own, lab_ref):
            col = label_colors(lab)
            m = (lab > 0)[..., None]
            blend = np.where(m, (0.65 * color + 0.35 * col), color).astype(np.uint8)
            edge = np.zeros(lab.shape, bool)
            edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            edge[1:, :] |= lab[1:, :] != lab[:-1, :]
            edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)  # 2 px
            blend[edge] = 255
            panels.append(blend)
        for T in (ctx.T_raw, ctx.T_ref):
            d, _ = ctx.rR.render(Kd, invert(T @ f.pose_c2w), Wd, Hd)
            v = (f.depth > 0) & (d > 0)
            panels.append(_resize_nn(heat(np.abs(f.depth - d), v), Wc, Hc))
        upright = [np.rot90(p, k=-1) for p in panels]
        sep = np.full((upright[0].shape[0], 6, 3), 255, np.uint8)
        row = np.concatenate(sum(([p, sep] for p in upright), [])[:-1], axis=1)
        name = f"{ctx.rescan[:8]}_frame{i:06d}.png"
        Image.fromarray(row).save(out_dir / name)
        written.append(str(out_dir / name))
    return written


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


def run_m0(cfg: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    root = data_root(cfg)
    m0 = cfg["m0"]
    meta = load_meta(root)
    pairs = read_pairs(REPO_ROOT / m0["pairs_file"])
    run_dir = outputs_root(cfg) / (run_id or new_run_id("m0"))
    t_start = time.time()

    refs: dict[str, tuple[Scan, Renderer]] = {}
    res: dict[str, Any] = {"pairs": [], "depth": {}, "structure": {}, "rigid": [], "align": {}}
    overlays: list[str] = []
    for ref, rescan in pairs:
        print(f"[m0] {ref[:8]} -> {rescan[:8]}", flush=True)
        if ref not in refs:
            R = load_scan(root, ref, with_intrinsics=False)
            refs[ref] = (R, Renderer(R.vertices, R.faces, R.face_labels))
        R, rR = refs[ref]
        S = load_scan(root, rescan, with_intrinsics=False)
        seq = open_sequence(root, rescan)
        T_raw = meta[rescan].T_align
        if cfg["alignment"]["refine"]:
            T_ref, ainfo = refine_from_sequence(seq, R.vertices, R.faces, T_raw, cfg["alignment"])
        else:
            T_ref, ainfo = T_raw, {"accepted": False, "disabled": True}
        ctx = PairCtx(ref, rescan, R, S, rR, Renderer(S.vertices, S.faces, S.face_labels),
                      seq, T_raw, T_ref, ainfo)  # fmt: skip
        res["pairs"].append([ref, rescan])
        res["align"][rescan] = {**ainfo, "T_refined": T_ref}
        res["depth"][rescan] = check_depth(ctx, int(m0["depth_frame_step"]))
        res["structure"][rescan] = check_structure(ctx)
        res["rigid"].extend(check_rigid(ctx, meta))
        if rescan in m0["overlay_scans"]:
            overlays += make_overlays(ctx, int(m0["overlay_frames"]), run_dir / "overlays")
        seq.close()

    # ---- acceptance
    acc: dict[str, Any] = {}
    d_meds = {s: v["ref_refined"]["median"] for s, v in res["depth"].items()}
    acc["1_depth"] = {
        "gate": f"median |d_obs - d_render(ref mesh, refined T_align)| < {m0['accept_depth_median']} m on every dev rescan",
        "values": d_meds,
        "pass": all(v < m0["accept_depth_median"] for v in d_meds.values()),
    }
    s_meds = {s: v["T_align"]["median"] for s, v in res["structure"].items()}
    acc["2_structure"] = {
        "gate": f"median structure point-to-surface distance under raw T_align < {m0['accept_struct_median']} m",
        "values": s_meds,
        "pass": all(v < m0["accept_struct_median"] for v in s_meds.values()),
    }
    scored = [r for r in res["rigid"] if "hyp" in r]
    wins: dict[str, int] = {}
    for r in scored:
        wins[r["best"]] = wins.get(r["best"], 0) + 1
    best_h = max(wins, key=wins.get) if wins else None
    n_below = sum(r["hyp"][best_h]["chamfer"] < m0["accept_rigid_chamfer"] for r in scored) if best_h else 0
    acc["3_rigid"] = {
        "gate": f">= {m0['accept_rigid_min_objects']} objects with Chamfer < {m0['accept_rigid_chamfer']} m under the winning hypothesis",
        "winning_hypothesis": best_h,
        "wins": wins,
        "n_objects": len(scored),
        "n_below_threshold": int(n_below),
        "pass": n_below >= m0["accept_rigid_min_objects"],
    }
    acc["4_overlays"] = {"files": overlays, "pass": "manual inspection"}
    res["acceptance"] = acc
    res["wall_time_s"] = time.time() - t_start
    path = write_metrics(run_dir, res, cfg)
    res["metrics_path"] = str(path)
    return res


def format_summary(res: dict[str, Any]) -> str:
    L = []
    cm = lambda x: f"{100 * x:5.2f}"  # noqa: E731
    L.append("== M0 acceptance checks (oracle-free; dev pairs) ==")
    L.append("(1) |d_obs - d_render| [cm]           median  p90   (own mesh | ref raw T_align | ref refined T_align)")
    for s, v in res["depth"].items():
        L.append(
            f"    {s[:8]}  frames={v['n_frames']:3d}   own {cm(v['own']['median'])}/{cm(v['own']['p90'])}"
            f" | raw {cm(v['ref_raw']['median'])}/{cm(v['ref_raw']['p90'])}"
            f" | refined {cm(v['ref_refined']['median'])}/{cm(v['ref_refined']['p90'])}"
        )
    L.append("    ICP refinement of T_align:")
    for s, a in res["align"].items():
        L.append(
            f"    {s[:8]}  dt={100 * a.get('delta_t_m', 0):.1f} cm  drot={a.get('delta_rot_deg', 0):.2f} deg"
            f"  fitness={a.get('fitness') or 0:.3f}  accepted={a.get('accepted')}"
        )
    L.append("(2) structure point-to-surface, median [cm] per hypothesis")
    for s, v in res["structure"].items():
        hs = "  ".join(f"{k}={cm(h['median'])}" for k, h in v.items() if isinstance(h, dict))
        L.append(f"    {s[:8]}  n={v['n_points']}  {hs}")
    L.append("(3) rigid transforms: symmetric Chamfer [cm] per hypothesis")
    for r in res["rigid"]:
        if "hyp" not in r:
            L.append(f"    {r['rescan'][:8]} id={r['id']:3d}  SKIP {r.get('skip')}")
            continue
        hs = " ".join(f"{k}={cm(h['chamfer'])}" for k, h in r["hyp"].items())
        L.append(
            f"    {r['rescan'][:8]} id={r['id']:3d} {r['label'][:12]:12s} {hs}  best={r['best']}"
            f"  disp_ref={r['gt_displacement_ref_frame']:.2f} m"
        )
    L.append("-- acceptance --")
    for k, a in res["acceptance"].items():
        L.append(f"    {k}: pass={a['pass']}")
        if "gate" in a:
            L.append(f"        {a['gate']}")
        if k == "3_rigid":
            L.append(
                f"        winner={a['winning_hypothesis']} wins={a['wins']} "
                f"below={a['n_below_threshold']}/{a['n_objects']}"
            )
        if k == "4_overlays":
            for fp in a["files"]:
                L.append(f"        {fp}")
    L.append(f"metrics: {res['metrics_path']}  (wall {res['wall_time_s']:.0f} s)")
    return "\n".join(L)
