"""Top-down before/after figure of a reference graph and its updated graph.

Panels: reference graph | updated graph | (optional) GT. Every node is its AABB footprint in the
reference frame (xy, metres), coloured by state. The room outline comes from the reference
mesh's structure vertices.

Colours (dataviz palette; blue / orange / aqua validated all-pairs, light surface):
moved = blue, removed = orange (dashed, hatched), new = aqua (labelled: aqua is < 3:1 on the
surface, so it never carries meaning alone), unchanged = grey fill, unknown = grey dotted outline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lsg.classes import STRUCTURE_NYU40
from lsg.graph.model import Graph, Node

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
STATE_COLOR = {
    "moved": "#2a78d6",
    "removed": "#eb6834",
    "new": "#1baf7a",
    "unchanged": "#c3c2b7",
    "unknown": MUTED,
}
STATE_ORDER = ("unchanged", "unknown", "moved", "removed", "new")
# GT statuses drawn in the GT panel, mapped onto the state styles
GT_STYLE = {"unchanged": "unchanged", "moved": "moved", "removed": "removed",
            "absent_unlisted": "unknown", "dont_care": "unknown", "new": "new"}  # fmt: skip


@dataclass
class RoomOutline:
    """xy points of the reference room (floor faint, other structure darker)."""

    floor_xy: np.ndarray  # (N, 2)
    struct_xy: np.ndarray  # (M, 2)


def room_outline(vertices: np.ndarray, vertex_nyu40: np.ndarray, max_points: int = 30000,
                 seed: int = 0) -> RoomOutline:  # fmt: skip
    """Subsampled structure vertices of the reference mesh (floor = nyu40 2; ceiling dropped)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(vertices, float)
    nyu = np.asarray(vertex_nyu40)

    def pick(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) > max_points:
            idx = np.sort(rng.choice(idx, max_points, replace=False))
        return V[idx, :2]

    struct = np.isin(nyu, sorted(STRUCTURE_NYU40 - {2, 20, 22}))
    return RoomOutline(floor_xy=pick(nyu == 2), struct_xy=pick(struct))


def _footprint(n: Node) -> tuple[np.ndarray, np.ndarray] | None:
    if n.bbox_3d is None:
        return None
    return np.asarray(n.bbox_3d["min"], float)[:2], np.asarray(n.bbox_3d["max"], float)[:2]


def _area(n: Node) -> float:
    fp = _footprint(n)
    return 0.0 if fp is None else float(np.prod(fp[1] - fp[0]))


def _draw_box(ax, lo, hi, style: str, z: float, lw: float = 1.0, alpha_fill: float = 0.55) -> None:
    from matplotlib.patches import Rectangle

    col = STATE_COLOR[style]
    w, h = hi - lo
    if style == "unknown":
        ax.add_patch(Rectangle(lo, w, h, fill=False, ec=col, lw=lw, ls=(0, (1.5, 1.5)), zorder=z))
    elif style == "removed":
        ax.add_patch(Rectangle(lo, w, h, fc="none", ec=col, lw=lw + 0.5, ls=(0, (4, 2)), hatch="////",
                               zorder=z))  # fmt: skip
    elif style == "unchanged":
        ax.add_patch(Rectangle(lo, w, h, fc=col, ec=MUTED, lw=0.6, alpha=0.6, zorder=z))
    else:
        ax.add_patch(Rectangle(lo, w, h, fc=col, ec=SURFACE, lw=1.0, alpha=alpha_fill, zorder=z))
        ax.add_patch(Rectangle(lo, w, h, fill=False, ec=col, lw=lw, zorder=z + 0.01))


def _setup_ax(ax, outline: RoomOutline | None, title: str) -> None:
    ax.set_facecolor(SURFACE)
    if outline is not None:
        if len(outline.floor_xy):
            ax.scatter(outline.floor_xy[:, 0], outline.floor_xy[:, 1], s=0.3, c=GRID, lw=0, zorder=0,
                       rasterized=True)  # fmt: skip
        if len(outline.struct_xy):
            ax.scatter(outline.struct_xy[:, 0], outline.struct_xy[:, 1], s=0.3, c=MUTED, lw=0, zorder=0.5,
                       rasterized=True)  # fmt: skip
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID)


def draw_graph(
    ax,
    G: Graph,
    G_ref: Graph | None = None,
    states: Mapping[str, str] | None = None,
    label_states: Sequence[str] = ("moved", "removed", "new"),
    edges: bool = False,
    new_boxes: Sequence[tuple[str, Any, Any]] = (),
    moved_to: Mapping[str, tuple[Any, Any]] | None = None,
) -> dict[str, int]:
    """Draw a graph's nodes on `ax`; returns per-style counts (for the legend).

    Args:
        G: Graph to draw (things + anchors with a bbox).
        G_ref: Reference graph; moved nodes then get a faint old footprint and an old→new arrow.
        states: Override `{node id: style}` (GT panel); nodes missing from it are not drawn.
        label_states: Styles whose nodes get a text label.
        edges: Draw support edges child→parent (thin lines between centroids).
        new_boxes: Extra `(label, min, max)` boxes drawn as `new` (GT new instances).
        moved_to: `{node id: (min, max)}` new footprints for `moved` nodes of the GT panel.
    """
    nodes = [n for n in G.nodes if n.level in ("anchor", "thing") and n.bbox_3d is not None]
    ref = G_ref.node_map() if G_ref is not None else {}
    counts = dict.fromkeys(STATE_ORDER, 0)
    for n in sorted(nodes, key=_area, reverse=True):  # large first, small on top
        style = states.get(n.id) if states is not None else n.state
        if style is None:
            continue
        lo, hi = _footprint(n)  # type: ignore[misc]
        z = 2 + (n.level == "thing")
        counts[style] += 1
        if style == "moved":
            old = ref.get(n.id)
            src = _footprint(old) if old is not None else None
            if moved_to is not None and n.id in moved_to:  # GT panel: n is the reference node
                src = (lo, hi)
                lo, hi = (np.asarray(moved_to[n.id][0], float)[:2], np.asarray(moved_to[n.id][1], float)[:2])
            if src is not None:
                from matplotlib.patches import Rectangle

                ax.add_patch(Rectangle(src[0], *(src[1] - src[0]), fill=False, ec=STATE_COLOR["moved"], lw=0.8,
                                       ls=(0, (2, 2)), alpha=0.8, zorder=z))  # fmt: skip
                a, b = (src[0] + src[1]) / 2, (lo + hi) / 2
                if np.linalg.norm(b - a) > 1e-3:
                    ax.annotate("", xy=b, xytext=a, zorder=5,
                                arrowprops={"arrowstyle": "-|>", "color": STATE_COLOR["moved"], "lw": 1.2})  # fmt: skip
        _draw_box(ax, lo, hi, style, z)
        if style in label_states:
            ax.text(hi[0], hi[1], n.label, fontsize=6, color=INK2, ha="left", va="bottom", zorder=6, clip_on=True)
    for lab, lo, hi in new_boxes:
        lo, hi = np.asarray(lo, float)[:2], np.asarray(hi, float)[:2]
        _draw_box(ax, lo, hi, "new", 4)
        counts["new"] += 1
        if "new" in label_states:
            ax.text(hi[0], hi[1], lab, fontsize=6, color=INK2, ha="left", va="bottom", zorder=6, clip_on=True)
    if edges:
        nm = G.node_map()
        for child, (parent, _t) in G.support_parent().items():
            if child in nm and parent in nm and nm[child].centroid and nm[parent].centroid:
                a, b = np.asarray(nm[child].centroid)[:2], np.asarray(nm[parent].centroid)[:2]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, lw=0.4, alpha=0.6, zorder=4)
    return counts


def _legend(fig, names: Mapping[str, str] | None = None) -> None:
    from matplotlib.patches import Rectangle

    handles, labels = [], []
    for s in STATE_ORDER:
        col = STATE_COLOR[s]
        if s == "unknown":
            h = Rectangle((0, 0), 1, 1, fill=False, ec=col, ls=(0, (1.5, 1.5)))
        elif s == "removed":
            h = Rectangle((0, 0), 1, 1, fc="none", ec=col, hatch="////", ls=(0, (4, 2)))
        else:
            h = Rectangle((0, 0), 1, 1, fc=col, ec=col if s != "unchanged" else MUTED, alpha=0.7)
        handles.append(h)
        labels.append((names or {}).get(s, s))
    fig.legend(handles, labels, loc="lower center", ncol=len(STATE_ORDER), frameon=False, fontsize=8,
               labelcolor=INK2)  # fmt: skip


def topdown_png(
    G_ref: Graph,
    G_upd: Graph,
    out: str | Path,
    outline: RoomOutline | None = None,
    gt: Mapping[str, Any] | None = None,
    title: str | None = None,
    edges: bool = False,
    dpi: int = 150,
) -> Path:
    """Write the before/after figure.

    Args:
        G_ref: Reference graph (drawn all `unchanged`).
        G_upd: Updated graph (states as predicted).
        out: PNG path (parents created).
        outline: Room outline (`room_outline`); None = no background.
        gt: Optional GT panel, computed in `lsg.eval` (hard rule 1):
            `{"states": {node id: GT status}, "new": [(label, min, max)], "moved_to": {id: (min, max)}}`.
        title: Figure title.
        edges: Also draw support edges.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pan = 3 if gt is not None else 2
    ratio = 1.0
    if outline is not None and len(outline.floor_xy) + len(outline.struct_xy) > 10:
        xy = np.concatenate([outline.floor_xy, outline.struct_xy])
        ext = np.ptp(xy, axis=0) + 0.4
        ratio = float(np.clip(ext[1] / max(ext[0], 1e-6), 0.4, 2.0))
    fig, axes = plt.subplots(1, n_pan, figsize=(4.6 * n_pan, 4.6 * ratio + 1.1), facecolor=SURFACE,
                             sharex=True, sharey=True)  # fmt: skip
    _setup_ax(axes[0], outline, "reference graph")
    draw_graph(axes[0], G_ref, states={n.id: "unchanged" for n in G_ref.nodes}, label_states=(), edges=edges)
    mode = G_upd.meta.get("obs_mode") or ""
    _setup_ax(axes[1], outline, f"updated graph ({mode})" if mode else "updated graph")
    draw_graph(axes[1], G_upd, G_ref=G_ref, edges=edges)
    if gt is not None:
        _setup_ax(axes[2], outline, "ground truth (dotted: absent-unlisted / don't care)")
        st = {k: GT_STYLE[v] for k, v in gt.get("states", {}).items() if v in GT_STYLE}
        draw_graph(axes[2], G_ref, states=st, new_boxes=gt.get("new", ()), moved_to=gt.get("moved_to"))
    _legend(fig)
    if title:
        fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97 if title else 1))
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return p
