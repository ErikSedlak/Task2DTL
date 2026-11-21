"""Anytime evaluation of the graph during the walk (CHANGE 3 Part B; `uv run lsg anytime`).

With snapshot method runs (`run_eval(..., snapshots=True)`) every pair and mode records, per snapshot,
`(walk fraction, counts, reference-node states)`: fraction 0 = the reference graph before the first frame
(every node `unknown`, no new node), fraction 1 = the final graph.

- **anytime curve**: for f in `FRACS`, each pair contributes the counts of its latest snapshot at or before
  f (a step function); the counts are summed over pairs and turned into micro metrics + bootstrap CIs by
  `metrics.aggregate` (the same definitions as the final tables);
- **stability**: per reference node, the number of state changes across its snapshots, the share of nodes
  with at least one change, and the settling fraction (the earliest snapshot fraction after which the
  state equals the final one), median per GT status (eval reads GT: allowed here).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lsg.eval.metrics import aggregate

FRACS = [round(0.1 * i, 1) for i in range(11)]
KEYS = ("moved_F1", "removed_P", "removed_R", "false_deletion_rate", "unchanged_kept", "false_move_rate", "new_P",
        "new_R", "new_ag_P", "new_ag_R")  # fmt: skip


def snapshot_row(frac: float, frame_idx: int | None, counts: Mapping[str, int], G: Any) -> dict[str, Any]:
    """One snapshot record: walk fraction, frame, counts, reference-node states, new-node tokens."""
    return {"frac": round(float(frac), 6), "frame": None if frame_idx is None else int(frame_idx),
            "counts": {k: int(v) for k, v in counts.items()},
            "states": {n.id: n.state for n in G.nodes if n.id.startswith("r") and n.state is not None},
            "new": sorted(n.obs_token or "" for n in G.nodes if n.state == "new")}  # fmt: skip


def at_fraction(snaps: Sequence[Mapping[str, Any]], f: float) -> Mapping[str, Any]:
    """The latest snapshot with fraction <= f (snapshots sorted by fraction; the first is fraction 0)."""
    best = snaps[0]
    for s in snaps:
        if s["frac"] <= f + 1e-9:
            best = s
    return best


def curve(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    """Micro metrics with CIs at every `FRACS` value over the pairs of one mode."""
    per = [sorted(r["anytime"], key=lambda s: s["frac"]) for r in rows if r["mode"] == mode and r.get("anytime")]
    out = []
    for f in FRACS:
        agg = aggregate([{"counts": at_fraction(p, f)["counts"], "loc_errors": []} for p in per], cfg)
        out.append({"frac": f, **{k: agg["metrics"][k] for k in KEYS if k in agg["metrics"]}})
    return out


def flips_and_settling(snaps: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, float]]:
    """Per reference node: (state changes over the snapshots, settling fraction)."""
    snaps = sorted(snaps, key=lambda s: s["frac"])
    final = snaps[-1]["states"]
    out = {}
    for nid, fin in final.items():
        seq = [(s["frac"], s["states"].get(nid)) for s in snaps]
        flips = sum(1 for (_, a), (_, b) in zip(seq, seq[1:]) if a != b)
        settle = 0.0
        for f, st in seq:
            if st != fin:
                settle = None  # type: ignore[assignment]
            elif settle is None:
                settle = f
        out[nid] = (flips, float(settle if settle is not None else 1.0))
    return out


def stability(rows: Sequence[Mapping[str, Any]], mode: str, gt_status: Mapping[tuple[str, str], Mapping[str, str]]
              ) -> dict[str, Any]:  # fmt: skip
    """Mean flips per node, share of nodes with >= 1 flip, median settling fraction per GT status."""
    flips, by_status = [], {}
    for r in rows:
        if r["mode"] != mode or not r.get("anytime"):
            continue
        st = gt_status.get((r["ref"], r["rescan"]), {})
        for nid, (fl, settle) in flips_and_settling(r["anytime"]).items():
            flips.append(fl)
            by_status.setdefault(st.get(nid, "other"), []).append(settle)
    a = np.asarray(flips, float)
    return {"n_nodes": int(len(a)), "mean_flips": float(a.mean()) if len(a) else None,
            "share_flipping": float((a > 0).mean()) if len(a) else None,
            "settling_median": {k: float(np.median(v)) for k, v in sorted(by_status.items())},
            "settling_n": {k: len(v) for k, v in sorted(by_status.items())}}  # fmt: skip


def anytime_summary(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], modes: Sequence[str]) -> dict[str, Any]:
    """Curve + stability per mode from `run_eval` rows with `anytime` records (`gt_status` in each row)."""
    gts = {(r["ref"], r["rescan"]): r.get("gt_status", {}) for r in rows}
    return {m: {"curve": curve(rows, cfg, m), "stability": stability(rows, m, gts)} for m in modes}


def format_anytime(summ: Mapping[str, Any]) -> str:
    """Printable curve table per mode + stability lines."""
    lines = []
    for m, s in summ.items():
        lines.append(f"== {m}: metrics over the walk (latest snapshot at or before each fraction) ==")
        lines.append("frac " + "".join(f"{k[:12]:>13}" for k in KEYS))
        for row in s["curve"]:
            vals = [(row.get(k) or {}).get("value") for k in KEYS]
            cells = "".join(f"{'-' if v is None else format(100 * v, '.1f'):>13}" for v in vals)
            lines.append(f"{row['frac']:.1f} " + cells)
        st = s["stability"]
        if st["n_nodes"]:
            lines.append(f"stability: {st['n_nodes']} nodes, mean flips {st['mean_flips']:.2f}, "
                         f"{100 * st['share_flipping']:.1f} % flip at least once; median settling fraction by GT status: "
                         + ", ".join(f"{k} {v:.2f} (n={st['settling_n'][k]})" for k, v in st["settling_median"].items()))  # fmt: skip
    return "\n".join(lines)


def plot_anytime(summ: Mapping[str, Any], path) -> None:
    """Moved F1, removed R, new P / R and false deletion over the walk fraction, one panel per mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(summ)
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), squeeze=False)
    for ax, m in zip(axes[0], modes):
        cv = summ[m]["curve"]
        x = [r["frac"] for r in cv]
        for k in ("moved_F1", "removed_R", "new_P", "new_R", "false_deletion_rate", "unchanged_kept"):
            y = [None if r.get(k, {}).get("value") is None else 100 * r[k]["value"] for r in cv]
            ax.plot(x, [np.nan if v is None else v for v in y], marker="o", ms=3, label=k)
        ax.set_title(f"{m}: graph quality during the walk")
        ax.set_xlabel("walk fraction (processed frames)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
