"""Node-edge picture of one pair's graph before and after the update (for the report, Figure 2).

Shows only the part of the graph that changed: every node that is moved, removed or new in the updated graph, plus the
parents they hang on (arrows child -> parent, solid `on`, dashed `in`). The positions are the same in both panels.
Reads only the reference graph (3DSSG) and the saved updated graph, no rescan annotation.

Run:  uv run python docs/report/graph_figure.py [--run outputs/chg3_final/val/FULL] [--mode perception]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from lsg.config import data_root, load_config
from lsg.graph.build import build_reference_graph
from lsg.graph.model import Graph
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import load_scan
from lsg.viz.topdown import INK, INK2, MUTED, STATE_COLOR, SURFACE

REF, RESCAN = "8eabc45f-5af7-2f32-8528-640861d2a135", "5341b7b3-8a66-2cdd-856d-9d70e194568b"
FLOOR = "floor"


def support(G: Graph) -> dict[str, tuple[str, str]]:
    """child -> (parent, 'on' | 'in')."""
    return {e.source: (e.target, e.type) for e in G.edges if e.type in ("on", "in")}


def layout(S: list[str], par_ref: dict, par_upd: dict, labels: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Horizontal tree: the floor on the left, parents to the right of it, then their children. A node hangs under
    its reference parent (new nodes under their new parent). Leaves get one row each. Returns node id -> (x, y)."""
    parent = {n: (par_ref.get(n) or par_upd.get(n) or (FLOOR, ""))[0] for n in S if n != FLOOR}
    kids: dict[str, list[str]] = {}
    for n, p in parent.items():
        kids.setdefault(p, []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    row = [0.0]

    def place(n: str, depth: int) -> float:
        ks = sorted(kids.get(n, []), key=lambda k: (bool(kids.get(k)), labels[k], k))
        if not ks:
            y = -row[0]
            row[0] += 1.0
        else:
            ys = [place(k, depth + 1) for k in ks]
            y = sum(ys) / len(ys)
        pos[n] = (depth * 1.55, y)
        return y

    place(FLOOR, 0)
    for n in S:  # nodes whose parent chain does not reach the floor (should not happen)
        pos.setdefault(n, (0.0, -row[0]))
    return pos


def draw(ax, nodes: dict[str, tuple[str, str]], pos, edges: dict, title: str, orig_state: dict | None) -> None:
    """`nodes`: id -> (label, state)."""
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.axis("off")
    for child, (par, typ) in edges.items():
        if child not in pos or par not in pos:
            continue
        (x0, y0), (x1, y1) = pos[child], pos[par]
        ax.annotate("", xy=(x1 + 0.47, y1), xytext=(x0 - 0.47, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, ls="-" if typ == "on" else (0, (3, 2)),
                                    shrinkA=0, shrinkB=0, mutation_scale=6))  # fmt: skip
        ax.text(x0 - 0.52, y0 + 0.05, typ, fontsize=5.5, color=MUTED, ha="right", va="bottom")
    for nid, (label, state) in nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        col = STATE_COLOR.get(state, MUTED)
        ax.add_patch(FancyBboxPatch((x - 0.45, y - 0.36), 0.9, 0.72, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=col, ec=col, alpha=0.9 if state != "unchanged" else 0.55, lw=0.8,
                                    ls="-" if state != "unknown" else (0, (2, 1.5))))  # fmt: skip
        txt = label if nid == FLOOR else f"{label} {nid}"
        ax.text(x, y, txt, fontsize=5.8, ha="center", va="center", color="white" if state in ("moved", "removed", "new") else INK)
    ax.set_xlim(min(p[0] for p in pos.values()) - 0.6, max(p[0] for p in pos.values()) + 0.6)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.6, max(p[1] for p in pos.values()) + 0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/chg3_final/val/FULL")
    ap.add_argument("--mode", default="perception")
    ap.add_argument("--out", default="docs/figures/graph_before_after.png")
    a = ap.parse_args()
    cfg = load_config(None)
    root = data_root(cfg)
    R = load_scan(root, REF, with_intrinsics=False)
    G_ref = build_reference_graph(R, load_objects(root, REF), load_relations(root, REF), "gt", cfg)
    G_upd = Graph.load(Path(a.run) / "graphs" / a.mode / f"{REF}_{RESCAN}.json")
    lab = {n.id: n.label for n in G_ref.nodes} | {n.id: n.label for n in G_upd.nodes} | {FLOOR: "floor"}
    par_ref, par_upd = support(G_ref), support(G_upd)
    state_upd = {n.id: n.state for n in G_upd.nodes}
    changed = [i for i, s in state_upd.items() if s in ("moved", "removed", "new") and i != "room"]
    S = list(changed)
    for n in changed:
        for par in (par_ref, par_upd):
            if n in par and par[n][0] not in S:
                S.append(par[n][0])
    pos = layout(S, par_ref, par_upd, lab)
    nodes_before = {n: (lab[n], "unchanged") for n in S if not n.startswith("n")}  # new nodes do not exist before
    nodes_after = {n: (lab[n], state_upd.get(n, "unchanged") if n != FLOOR else "unchanged") for n in S}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), facecolor=SURFACE)
    draw(axes[0], nodes_before, pos, {c: par_ref[c] for c in S if c in par_ref}, "before: reference graph (changed part)", None)
    draw(axes[1], nodes_after, pos, {c: par_upd[c] for c in S if c in par_upd}, f"after: updated graph ({a.mode})", None)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=STATE_COLOR[s], ec=STATE_COLOR[s]) for s in ("unchanged", "moved", "removed", "new")]
    fig.legend(handles, ["unchanged", "moved", "removed", "new"], loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out, f"({len(S)} nodes)")


if __name__ == "__main__":
    main()
