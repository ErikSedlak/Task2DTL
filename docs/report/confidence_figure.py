"""Confidence of a few reference objects during the walk (update-policy figure, not yet in the report).

Replays the frame loop of FULL on one val pair (the report's example pair) and records every node's counted votes
after each processed frame. The confidence of an unmatched node is p = sigmoid(logit(prior_p) + clip(L_hit·seen -
L_miss·free)) (`change.confidence`); a node is deleted at the end if it has free evidence (>= k_free counted free
votes) and p < p_del. The chosen objects are all unmatched in perception mode and not thin (their votes come from depth
alone, so no appearance model is needed). The script checks that the replay ends with the same vote counts as the
saved run (`details/perception`), and prints the result.

Run:  uv run python docs/report/confidence_figure.py [--run outputs/chg3_final/val/FULL]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsg.change.confidence import node_logodds, prior_logodds, probability
from lsg.change.online import run_online
from lsg.change.pipeline import prepare_pair, select_frames
from lsg.config import data_root, load_config
from lsg.graph.build import build_reference_graph
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan, open_sequence
from lsg.viz.topdown import GRID, INK, INK2, MUTED, STATE_COLOR, SURFACE

REF, RESCAN = "8eabc45f-5af7-2f32-8528-640861d2a135", "5341b7b3-8a66-2cdd-856d-9d70e194568b"
# node -> (short name in the figure, line colour, line style)
NODES = {
    "r71": ("pillow, removed (correct)", STATE_COLOR["removed"], "-"),
    "r36": ("vase, deleted: scored false deletion", STATE_COLOR["removed"], "--"),
    "r5": ("bench, unchanged", INK2, "-"),
    "r18": ("wood, mixed evidence: unknown", MUTED, "-"),
    "r40": ("cabinet, never seen: unknown", MUTED, ":"),
}


class _NoObs:
    """Observation source that observes nothing: the votes do not depend on it."""

    def begin(self, ref_ctx, seq_info):
        pass

    def observe(self, frame, ref_render):
        pass

    def finish(self):
        return []

    def peek(self):
        return []


def replay(cfg: dict) -> tuple[list[int], dict[str, list], list[int]]:
    """Counted votes of every node after each processed frame. Returns (frame ids, node -> [NodeVotes], frame ids)."""
    root = data_root(cfg)
    meta = load_meta(root)
    R = load_scan(root, REF, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, REF), load_relations(root, REF)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    c = copy.deepcopy(cfg)
    c.setdefault("online", {})["snapshot_every"] = 1
    hist: dict[str, list] = {n: [] for n in NODES}
    frames: list[int] = []

    def on_snap(o_now, v_now, views_now, fidx, k):
        frames.append(int(fidx))
        for n in NODES:
            hist[n].append(v_now[n])

    with open_sequence(root, RESCAN) as seq:
        fids = select_frames(seq.frame_ids, int(c["frames"]["stride"]), 1.0, 0, key=RESCAN)
        ctx, info, _ = prepare_pair(root, R, objs_R, rels_R, seq, RESCAN, meta[RESCAN].T_align, c, fids, G_ref=G_ref)
        run_online(ctx, info, seq, _NoObs(), c, contrast=bool(c["visibility"].get("seen_needs_contrast")),
                   on_snapshot=on_snap)  # fmt: skip
    return frames, hist, fids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/chg3_final/val/FULL")
    ap.add_argument("--out", default="docs/figures/confidence_walk.png")
    a = ap.parse_args()
    cfg = load_config()
    frames, hist, fids = replay(cfg)

    det = json.loads((Path(a.run) / "details" / "perception" / f"{REF}_{RESCAN}.json").read_text(encoding="utf-8"))
    assert det["frame_ids"] == fids, "processed frames differ from the saved run"
    saved = {o["node"]: o for o in det["objects"]}
    l0 = prior_logodds(None, None, None, cfg)
    p_del = float(cfg["confidence"]["p_del"])
    for n in NODES:
        v, s = hist[n][-1], saved[n]
        ok = (v.seen, v.free, v.free_like) == (s["votes"]["seen"], s["votes"]["free"], s["votes"]["free_like"])
        p_end = probability(node_logodds(l0, v, cfg))
        print(f"{n:5s} {s['label']:10s} gt={s['gt_status']:15s} pred={s['pred_state']:9s} obs={s['obs_token']} "
              f"seen={v.seen} free={v.free} p_end={p_end:.3f} saved p={s['confidence']:.3f} votes match: {ok}")  # fmt: skip
        assert ok and s["obs_token"] is None and v.free_like == 0, n

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.6, 2.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(1, len(frames) + 1))
    ends = []
    for n, (name, col, ls) in NODES.items():
        p = [probability(node_logodds(l0, v, cfg)) for v in hist[n]]
        ax.step([0, *x], [probability(l0), *p], where="post", color=col, ls=ls, lw=2, solid_capstyle="round")
        # counted free votes as small markers (they are what lowers p)
        fx = [x[i] for i in range(len(x)) if hist[n][i].free > (hist[n][i - 1].free if i else 0)]
        ax.plot(fx, [p[x.index(f)] for f in fx], "v", ms=4.5, color=col, mec=SURFACE, mew=1.0, zorder=3)
        ends.append([p[-1], name])
    ax.axhline(p_del, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.text(1, p_del + 0.015, f"delete below p = {p_del} (with free evidence)", color=INK2, va="bottom", fontsize=7)
    ax.axhline(probability(l0), color=GRID, lw=0.8, zorder=0)
    # direct labels at the line ends, nudged apart
    ends.sort(key=lambda e: -e[0])
    ys = []
    for y, _ in ends:
        ys.append(min(y, ys[-1] - 0.075) if ys else y)
    for (y0, name), y in zip(ends, ys):
        ax.text(x[-1] + 1.2, y, name, va="center", color=INK, fontsize=7)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("processed frame (every 3rd frame of the rescan)", color=INK2)
    ax.set_ylabel("confidence p", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([], [], "v", color=INK2, ms=4.5, label="counted free vote")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
