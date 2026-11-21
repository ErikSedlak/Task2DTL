"""§7.4 ground-truth audit (M8): sample false positives, draw one panel each, summarise verdicts.

Flow (`lsg audit sample`, then `lsg audit summarize`):
1. A val FULL run with `details=True` (both modes) writes per-object outcomes to
   `details/<mode>/<pair>.json`. Since CHANGE 3 (Part D) this is the final run's own FULL row
   (`lsg m7` writes details for `m7.details_rows`), passed as `from_run`, so the audited
   predictions are exactly the reported ones. A fresh FULL run must reproduce `compare`.
2. **Pools** per mode (DECISIONS M8):
   - `moved`: predicted moved, GT unchanged / removed / absent-unlisted (false move);
   - `removed`: predicted removed, GT unchanged / moved / absent-unlisted (false deletion);
   - `new`: predicted new node that the mode's §5 rule leaves unmatched (`fp`).
   Each row keeps its GT status, so "scored FP" and "absent-unlisted" can be reported apart.
   **Veto set** (STATUS M7): GT-removed objects that B2 deletes and FULL keeps `unknown`.
3. Seeded sample (`n_per_cat` per mode and category), one PNG panel per row, `audit.csv` and
   `audit_veto.csv` with empty `verdict` / `note` columns. Erik labels them:
   `gt_missing` (a real change the GT does not list, or a GT error), `method_error`, `unclear`.
4. `summarize`: share of `gt_missing` per mode × category with a Wilson 95% CI → `metrics.json`
   (hard rule 5) + `audit.md`.

Lives in `eval/`: reads rescan annotations and GT (hard rule 1).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lsg.geom.render import Renderer
from lsg.geom.transforms import apply, invert
from lsg.graph.model import Graph
from lsg.io.scan3r import Scan, SequenceReader, load_scan, open_sequence
from lsg.viz.topdown import MUTED, STATE_COLOR, SURFACE, room_outline

CATS = ("moved", "removed", "new")
VERDICTS = ("gt_missing", "method_error", "unclear")
MODES = ("oracle", "perception")
CSV_COLS = ("id", "mode", "category", "ref", "rescan", "node", "label", "gt_status", "gt_reason", "pred_state",
            "confidence", "seen", "free", "free_like", "panel", "verdict", "note")  # fmt: skip
POOL_GT = {"moved": ("unchanged", "removed", "absent_unlisted"), "removed": ("unchanged", "moved", "absent_unlisted")}


# --------------------------------------------------------------------------------------------
# pools + sampling
# --------------------------------------------------------------------------------------------


def load_details(run_dir: str | Path, mode: str) -> list[dict[str, Any]]:
    """All `details/<mode>/*.json` of a run, sorted by file name (= pair)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(run_dir) / "details" / mode).glob("*.json"))]


def _row(d: Mapping[str, Any], cat: str, o: Mapping[str, Any]) -> dict[str, Any]:
    v = o.get("votes") or {}
    return {"mode": d["mode"], "category": cat, "ref": d["ref"], "rescan": d["rescan"], "node": o.get("node", o.get("id")),
            "oid": o.get("oid"), "label": o["label"], "gt_status": o.get("gt_status", "new_unmatched"),
            "gt_reason": o.get("gt_reason"), "pred_state": o.get("pred_state", "new"), "confidence": o.get("confidence"),
            "seen": v.get("seen"), "free": v.get("free"), "free_like": v.get("free_like"),
            "bbox_3d": o.get("bbox_3d"), "gt_aabb_new": o.get("gt_aabb_new"), "gt_token": o.get("gt_token"),
            "displacement": o.get("displacement")}  # fmt: skip


def pools(details: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """False-positive pools of one mode (module docstring), in pair / node order."""
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in CATS}
    for d in details:
        for o in d["objects"]:
            for cat in ("moved", "removed"):
                if o["pred_state"] == cat and o["gt_status"] in POOL_GT[cat]:
                    out[cat].append(_row(d, cat, o))
        for n in d["new"]:
            if n["outcome"] == "fp":
                out["new"].append(_row(d, "new", n))
    return out


def veto_pool(details: Sequence[Mapping[str, Any]], b2_graph_dir: str | Path) -> list[dict[str, Any]]:
    """GT-removed objects that B2 deletes and FULL keeps `unknown` (the confidence veto, M6/M7)."""
    rows = []
    for d in details:
        g = Path(b2_graph_dir) / d["mode"] / f"{d['ref']}_{d['rescan']}.json"
        if not g.is_file():
            continue
        b2 = {n.id: n.state for n in Graph.load(g).nodes}
        for o in d["objects"]:
            if o["gt_status"] == "removed" and o["pred_state"] == "unknown" and b2.get(o["node"]) == "removed":
                rows.append(_row(d, "veto", o))
    return rows


def sample_rows(pool: Sequence[dict[str, Any]], n: int, seed: Sequence[int]) -> list[dict[str, Any]]:
    """`n` rows without replacement (all if fewer), seeded, returned in pool order."""
    if len(pool) <= n:
        return list(pool)
    idx = np.sort(np.random.default_rng(list(seed)).choice(len(pool), n, replace=False))
    return [pool[int(i)] for i in idx]


def pool_counts(pool: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Pool size by GT status."""
    out: dict[str, int] = {}
    for r in pool:
        out[r["gt_status"]] = out.get(r["gt_status"], 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------


def _box_corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _project(P_world: np.ndarray, pose_c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc = apply(invert(pose_c2w), P_world)
    z = Pc[:, 2]
    uv = (K @ Pc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def _inside_frac(uv: np.ndarray, z: np.ndarray, W: int, H: int) -> float:
    """Share of a projected box's image-plane AABB that lies inside the image (0 if a corner is behind)."""
    if np.any(z < 0.05):
        return 0.0
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    full = max((x1 - x0) * (y1 - y0), 1e-6)
    return max(0.0, min(x1, W) - max(x0, 0)) * max(0.0, min(y1, H) - max(y0, 0)) / full


def best_label_view(seq: SequenceReader, frame_ids: Sequence[int], T: np.ndarray, renderer: Renderer,
                    oid: int, box=None) -> tuple[int | None, int]:  # fmt: skip
    """Frame where reference object `oid` is seen best (reference render at `T @ pose`).

    Score = first-hit pixels of `oid`, times the squared share of its box projection inside the
    image when `box` = (min, max) is given (prefers whole-object views over close-ups).
    """
    K, (W, H) = seq.info.K_depth, seq.info.depth_wh
    corners = _box_corners(*box) if box is not None else None
    best, best_n, best_s = None, 0, 0.0
    for i in frame_ids:
        pose = seq.pose(int(i))
        if not np.all(np.isfinite(pose)):
            continue
        _, lab = renderer.render(K, invert(T @ pose), W, H)
        n = int((lab == oid).sum())
        if n == 0:
            continue
        s = float(n)
        if corners is not None:
            uv, z = _project(corners, T @ pose, K)
            s *= max(_inside_frac(uv, z, W, H), 0.05) ** 2
        if s > best_s:
            best, best_n, best_s = int(i), n, s
    return best, best_n


def best_box_view(seq: SequenceReader, frame_ids: Sequence[int], T: np.ndarray, lo, hi) -> int | None:
    """Frame that sees a box best: centre in view and not occluded by observed depth, largest visible footprint."""
    K, (W, H) = seq.info.K_depth, seq.info.depth_wh
    c = (np.asarray(lo, float) + np.asarray(hi, float)) / 2
    ext = float(np.max(np.asarray(hi, float) - np.asarray(lo, float)))
    corners = _box_corners(lo, hi)
    best, best_s = None, 0.0
    for i in frame_ids:
        pose = seq.pose(int(i))
        if not np.all(np.isfinite(pose)):
            continue
        pose = T @ pose
        uv, z = _project(np.vstack([c[None], corners]), pose, K)
        if z[0] < 0.3 or z[0] > 5.0 or not (0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H) or np.any(z[1:] < 0.05):
            continue
        d = seq.depth(int(i))[int(uv[0, 1]), int(uv[0, 0])]
        if 0 < d < z[0] - ext / 2 - 0.2:  # something in front of the box centre
            continue
        x0, y0 = uv[1:].min(0)
        x1, y1 = uv[1:].max(0)
        full = max((x1 - x0) * (y1 - y0), 1e-6)
        vis = max(0.0, min(x1, W) - max(x0, 0)) * max(0.0, min(y1, H) - max(y0, 0))
        s = vis * (vis / full)
        if s > best_s:
            best, best_s = int(i), s
    return best


def _draw_photo(ax, seq: SequenceReader, idx: int | None, pose: np.ndarray | None, boxes, title: str) -> None:
    """Upright colour image of frame `idx` with projected AABB wireframes `[(lo, hi, colour)]`."""
    ax.set_title(title, fontsize=8, loc="left", color="#0b0b0b")
    ax.set_xticks([])
    ax.set_yticks([])
    if idx is None or pose is None:
        ax.text(0.5, 0.5, "no view", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    img = seq.color(int(idx))
    Hs = img.shape[0]
    ax.imshow(np.rot90(img, k=-1))
    K = seq.info.K_color
    t = np.linspace(0, 1, 25)[:, None]
    for lo, hi, col in boxes:
        C = _box_corners(lo, hi)
        for a, b in _EDGES:
            P = C[a] + t * (C[b] - C[a])
            uv, z = _project(P, pose, K)
            ok = z > 0.05
            if ok.sum() < 2:
                continue
            ax.plot(Hs - 1 - uv[ok, 1], uv[ok, 0], color=col, lw=1.5)  # stored (u, v) -> upright (x, y)
    ax.set_xlim(0, Hs)
    ax.set_ylim(img.shape[1], 0)


def _ref_box(R: Scan, oid: int | None):
    inst = R.instances.get(int(oid)) if oid is not None else None
    return None if inst is None else (inst.aabb_min, inst.aabb_max)


def render_panel(row: Mapping[str, Any], root: Path, R: Scan, renderer: Renderer, T_align: np.ndarray,
                 frame_ids: Sequence[int], out: Path, ref_stride: int = 5) -> Path:  # fmt: skip
    """One audit panel (DESIGN §7.4): reference photo | rescan at the old place | rescan at the new place | top-down."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lsg.viz.topdown import _draw_box, _setup_ax

    cat = row["category"]
    ref_box = _ref_box(R, row.get("oid"))
    pred = row.get("bbox_3d")
    pred_box = (pred["min"], pred["max"]) if pred else None
    col_ref = STATE_COLOR["removed" if cat in ("removed", "veto") else "unknown"]
    col_pred = STATE_COLOR["moved" if cat == "moved" else "new"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 7.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    I4 = np.eye(4)
    with open_sequence(root, row["ref"]) as rseq:
        ids = rseq.frame_ids[::ref_stride]
        if ref_box is not None:
            i, _ = best_label_view(rseq, ids, I4, renderer, int(row["oid"]), ref_box)
        else:
            i = best_box_view(rseq, ids, I4, *pred_box) if pred_box else None
        box = ref_box or pred_box
        _draw_photo(axes[0], rseq, i, rseq.pose(i) if i is not None else None,
                    [(box[0], box[1], col_ref)] if box else [], f"reference scan, frame {i}")  # fmt: skip
    with open_sequence(root, row["rescan"]) as sseq:
        if ref_box is not None:
            j, _ = best_label_view(sseq, frame_ids, T_align, renderer, int(row["oid"]), ref_box)
            _draw_photo(axes[1], sseq, j, T_align @ sseq.pose(j) if j is not None else None,
                        [(ref_box[0], ref_box[1], col_ref)], f"rescan at the reference place, frame {j}")  # fmt: skip
        else:
            _draw_photo(axes[1], sseq, None, None, [], "rescan at the reference place: n/a (new node)")
        if cat in ("moved", "new") and pred_box is not None:
            k = best_box_view(sseq, frame_ids, T_align, *pred_box)
            _draw_photo(axes[2], sseq, k, T_align @ sseq.pose(k) if k is not None else None,
                        [(pred_box[0], pred_box[1], col_pred)], f"rescan at the predicted place, frame {k}")  # fmt: skip
        elif row.get("gt_aabb_new"):
            lo, hi = row["gt_aabb_new"]
            k = best_box_view(sseq, frame_ids, T_align, lo, hi)
            _draw_photo(axes[2], sseq, k, T_align @ sseq.pose(k) if k is not None else None,
                        [(lo, hi, STATE_COLOR["moved"])], f"rescan at the GT new place, frame {k}")  # fmt: skip
        else:
            _draw_photo(axes[2], sseq, None, None, [], "rescan at the predicted place: n/a")
    ax = axes[3]
    _setup_ax(ax, room_outline(R.vertices, R.vertex_nyu40, max_points=8000), "top-down (reference frame)")
    if ref_box is not None:
        _draw_box(ax, np.asarray(ref_box[0])[:2], np.asarray(ref_box[1])[:2], "removed" if cat in ("removed", "veto")
                  else "unknown", 3, lw=1.5)  # fmt: skip
    if row.get("gt_aabb_new"):
        lo, hi = row["gt_aabb_new"]
        _draw_box(ax, np.asarray(lo)[:2], np.asarray(hi)[:2], "unknown", 3.5, lw=1.5)
        ax.text(hi[0], hi[1], "GT new place", fontsize=7, color=MUTED, ha="left", va="bottom")
    if pred_box is not None and cat in ("moved", "new"):
        _draw_box(ax, np.asarray(pred_box[0])[:2], np.asarray(pred_box[1])[:2], "moved" if cat == "moved" else "new", 4)
    lines = [f"{row.get('id', '')}   {row['mode']} / {cat}",
             f"pair {row['ref'][:8]} → {row['rescan'][:8]}   node {row['node']}   label '{row['label']}'",
             f"predicted {row['pred_state']}   confidence {row['confidence'] if row['confidence'] is not None else '-'}",
             f"counted votes seen {row.get('seen')} / free {row.get('free')} / free-like {row.get('free_like')}",
             f"GT status {row['gt_status']}" + (f" ({row['gt_reason']})" if row.get("gt_reason") else "")
             + (f", displacement {row['displacement']:.2f} m" if row.get("displacement") is not None else "")]  # fmt: skip
    if row.get("gt_token"):
        lines.append(f"observation = rescan instance {row['gt_token'][0]} '{row['gt_token'][1]}'")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8.5, color="#0b0b0b", va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, facecolor=SURFACE)
    plt.close(fig)
    return out


def render_panels(rows: Sequence[dict[str, Any]], root: Path, details_by_pair: Mapping[tuple, Mapping[str, Any]],
                  out_dir: Path) -> None:  # fmt: skip
    """Panels for all rows (each reference loaded once); sets `row["panel"]`."""
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ref.setdefault(r["ref"], []).append(r)
    for k, (ref, rs) in enumerate(sorted(by_ref.items())):
        R = load_scan(root, ref, with_intrinsics=False)
        renderer = Renderer(R.vertices, R.faces, R.face_labels)
        for r in rs:
            d = details_by_pair[(r["mode"], r["ref"], r["rescan"])]
            p = out_dir / "panels" / f"{r['id']}.png"
            render_panel(r, root, R, renderer, np.asarray(d["T_align"], float), d["frame_ids"], p)
            r["panel"] = str(p.relative_to(out_dir)).replace("\\", "/")
        print(f"[audit] panels for {ref[:8]} ({k + 1}/{len(by_ref)})", flush=True)


# --------------------------------------------------------------------------------------------
# CSV + summary
# --------------------------------------------------------------------------------------------


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Rows with empty `verdict` / `note` columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in CSV_COLS}, **{k: ("" if r.get(k) is None else r[k]) for k in CSV_COLS},
                        "verdict": r.get("verdict", ""), "note": r.get("note", "")})  # fmt: skip
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a (labelled) audit CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval of k / n."""
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def summarize_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Verdict counts and the `gt_missing` share per (mode, category), per mode, and by GT status.

    Raises:
        ValueError: if a verdict is missing or not one of `VERDICTS`.
    """
    bad = [r["id"] for r in rows if r.get("verdict", "").strip() not in VERDICTS]
    if bad:
        raise ValueError(f"{len(bad)} rows without a valid verdict ({', '.join(VERDICTS)}): {bad[:5]}")

    def block(sel: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        n = len(sel)
        c = {v: sum(r["verdict"].strip() == v for r in sel) for v in VERDICTS}
        lo, hi = wilson(c["gt_missing"], n)
        return {"n": n, **c, "gt_missing_share": c["gt_missing"] / n if n else None, "ci95": [lo, hi]}

    out: dict[str, Any] = {}
    for key in sorted({(r["mode"], r["category"]) for r in rows}):
        sel = [r for r in rows if (r["mode"], r["category"]) == key]
        out[f"{key[0]}/{key[1]}"] = block(sel)
        for gs in sorted({r["gt_status"] for r in sel}):
            out[f"{key[0]}/{key[1]}/{gs}"] = block([r for r in sel if r["gt_status"] == gs])
    for m in sorted({r["mode"] for r in rows}):
        out[f"{m}/all"] = block([r for r in rows if r["mode"] == m])
    out["all"] = block(rows)
    return out


def format_summary_md(summ: Mapping[str, Any], title: str) -> str:
    """Markdown table of `summarize_rows`."""
    lines = [f"### {title}", "", "| group | n | gt_missing | method_error | unclear | gt_missing share [95% CI] |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for k, b in summ.items():
        sh = "–" if b["gt_missing_share"] is None else (
            f"{100 * b['gt_missing_share']:.0f}% [{100 * b['ci95'][0]:.0f}, {100 * b['ci95'][1]:.0f}]")
        lines.append(f"| {k} | {b['n']} | {b['gt_missing']} | {b['method_error']} | {b['unclear']} | {sh} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------


def compare_metrics(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Differences between two `metrics` blocks in `summary` and `per_pair` (empty = identical)."""
    diffs = []
    for m in a["summary"]:
        if a["summary"][m] != b["summary"].get(m):
            diffs.append(f"summary[{m}] differs")
    pa = {(x["ref"], x["rescan"], x["mode"]): x for x in a["per_pair"]}
    pb = {(x["ref"], x["rescan"], x["mode"]): x for x in b["per_pair"]}
    if set(pa) != set(pb):
        diffs.append("per_pair keys differ")
    for k in sorted(set(pa) & set(pb)):
        if pa[k]["counts"] != pb[k]["counts"] or pa[k]["loc_errors"] != pb[k]["loc_errors"]:
            diffs.append(f"per_pair {k} differs")
    return diffs


def run_audit_sample(
    cfg: dict[str, Any],
    run_id: str,
    split: str = "eval_val",
    from_run: str | Path | None = None,
    compare: str | Path | None = None,
    b2_run: str | Path | None = None,
    n_per_cat: int = 10,
    seed: int = 0,
    workers: int | None = None,
    max_pairs: int | None = None,
    m7_method: bool = False,
) -> dict[str, Any]:
    """`lsg audit sample`: FULL with details (or reuse `from_run`), pools, sample, panels, CSVs.

    The audited method is the current default (CHANGE 1 + 2 + 3). `m7_method=True` audits the
    M7 FULL instead (`ablations.M7_OVERRIDES`, row FULL-m7; compare with `m7_final/val/FULL`).

    Args:
        run_id: Output folder (`outputs/<run_id>/`; the FULL run goes to `<run_id>/val_FULL`).
        split: Pair list of the FULL run.
        from_run: Reuse a finished run with `details/` instead of running FULL.
        compare: Run dir (or id under outputs) whose `metrics` the FULL run must reproduce; None = skip.
        b2_run: Run dir with B2 graphs (`graphs/<mode>/`) for the veto set; None = no veto set.
        n_per_cat: Sample size per mode and category.
        max_pairs: Only the first N pairs (dry runs).
        m7_method: Force the M7 method (CHANGE 1, 2 and 3 off).
    """
    from lsg.config import data_root, outputs_root
    from lsg.eval.ablations import M7_OVERRIDES, apply_overrides
    from lsg.eval.harness import run_eval, split_path
    from lsg.io.scan3r import read_pairs
    from lsg.runinfo import write_metrics

    if m7_method:
        cfg = apply_overrides(cfg, M7_OVERRIDES)
    out_root = outputs_root(cfg)
    out_dir = out_root / run_id
    resolve = lambda p: None if p is None else (Path(p) if Path(p).is_dir() else out_root / p)  # noqa: E731
    check: dict[str, Any] = {}
    if from_run is None:
        sp = split_path(split)
        pairs = read_pairs(sp)[:max_pairs] if max_pairs else read_pairs(sp)
        res = run_eval(cfg, pairs, MODES, method="FULL", workers=workers, run_id=f"{run_id}/val_FULL",
                       split_name=sp.stem, details=True)  # fmt: skip
        run_dir = Path(res["metrics_path"]).parent
    else:
        run_dir = resolve(from_run)
        if not (run_dir / "details").is_dir():
            raise FileNotFoundError(f"{run_dir} has no details/ (run FULL with details, e.g. `lsg m7`)")
    cmp_dir = resolve(compare)
    if cmp_dir is not None and max_pairs is None:
        a = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        b = json.loads((cmp_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        diffs = compare_metrics(a, b)
        check = {"compared_to": str(cmp_dir), "identical": not diffs, "differences": diffs[:20]}
        print(f"[audit] metrics vs {cmp_dir}: {'identical' if not diffs else f'{len(diffs)} differences'}", flush=True)
    root = data_root(cfg)
    rows: list[dict[str, Any]] = []
    veto: list[dict[str, Any]] = []
    sizes: dict[str, Any] = {}
    details_by_pair: dict[tuple, Mapping[str, Any]] = {}
    for mi, mode in enumerate(MODES):
        det = load_details(run_dir, mode)
        details_by_pair.update({(mode, d["ref"], d["rescan"]): d for d in det})
        pl = pools(det)
        for ci, cat in enumerate(CATS):
            sizes[f"{mode}/{cat}"] = {"n": len(pl[cat]), "by_gt_status": pool_counts(pl[cat])}
            for k, r in enumerate(sample_rows(pl[cat], n_per_cat, (seed, mi, ci))):
                rows.append({**r, "id": f"{mode}_{cat}_{k:02d}"})
        b2 = resolve(b2_run)
        if b2 is not None:
            vp = veto_pool(det, b2 / "graphs")
            sizes[f"{mode}/veto"] = {"n": len(vp)}
            veto += [{**r, "id": f"{mode}_veto_{k:02d}"} for k, r in enumerate(vp)]
    render_panels(rows + veto, root, details_by_pair, out_dir)
    write_csv(rows, out_dir / "audit.csv")
    write_csv(veto, out_dir / "audit_veto.csv")
    metrics = {"split": split, "fp_run": str(run_dir), "n_per_cat": n_per_cat, "seed": seed, "pool_sizes": sizes,
               "n_rows": len(rows), "n_veto": len(veto), "consistency": check}  # fmt: skip
    write_metrics(out_dir, metrics, cfg)
    return {**metrics, "out_dir": str(out_dir)}


def run_audit_summarize(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """`lsg audit summarize`: verdict shares from the labelled CSVs → metrics.json + audit.md."""
    from lsg.config import outputs_root
    from lsg.runinfo import write_metrics

    out_dir = outputs_root(cfg) / run_id if not Path(run_id).is_dir() else Path(run_id)
    prev = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    summ = {"fp": summarize_rows(read_csv(out_dir / "audit.csv"))}
    md = format_summary_md(summ["fp"], "False positives (§7.4), FULL on val")
    vp = out_dir / "audit_veto.csv"
    if vp.is_file() and read_csv(vp):
        summ["veto"] = summarize_rows(read_csv(vp))
        md += "\n" + format_summary_md(summ["veto"], "GT-removed objects kept by FULL's confidence veto")
    metrics = {**{k: v for k, v in prev.items() if k != "summary"}, "summary": summ}
    write_metrics(out_dir, metrics, cfg)
    (out_dir / "audit.md").write_text(md, encoding="utf-8")
    return {"summary": summ, "markdown": md, "out_dir": str(out_dir)}
