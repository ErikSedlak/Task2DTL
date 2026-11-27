"""Render-and-compare at one frame (method figures, not yet in the report).

Replays the frame loop of FULL on one pair and takes exactly what the method used at one processed frame: the measured
depth, and the reference mesh rendered at the frame's aligned pose (the online `T_align` estimate at that moment).
Five panels, all on the depth image grid and turned upright; first the reference scan, then the rescan:

  (a) the reference scan's textured mesh rendered at the frame's pose (what the room looked like from here at the first
      visit; for the reader only, the depth comparison does not use colour),
  (b) reference depth rendered at the same pose,
  (c) the rescan's colour image (resampled onto the depth grid with the two intrinsics),
  (d) measured depth (same colour scale as (b)),
  (e) a verdict per pixel of the reference objects:
      - seen through: measured depth more than tau_free behind the reference surface (free space);
      - seen: depth agrees within tau_near, and the next reference surface behind is more than tau_free farther, so a
        missing object would show;
      - agrees, no contrast: depth agrees, but the surface behind is too close (a pillow on a sofa), so agreement
        proves nothing (no information);
      - something in front: measured depth more than tau_near nearer (occlusion; no information);
      - no depth / in between: no information;
      walls / floor / ceiling do not vote.
  One object is outlined in every panel: its first-hit pixels in the reference render (where it was at the first visit,
  seen from this pose; the pixels its frame vote is computed from). Its vote is in the title of (e).

Run one figure:   uv run python docs/report/render_compare_figure.py --ref R --rescan S --node r71 --frame 66 --out F
Run the examples: uv run python docs/report/render_compare_figure.py --examples   (writes docs/figures/render_compare*.png)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lsg.change.online import run_online
from lsg.change.pipeline import prepare_pair, select_frames
from lsg.change.visibility import frame_votes
from lsg.config import data_root, load_config
from lsg.geom.render import Renderer, TextureRenderer
from lsg.geom.transforms import invert
from lsg.graph.build import build_reference_graph
from lsg.graph.model import object_id_of
from lsg.io.dssg import load_objects, load_relations
from lsg.io.scan3r import load_meta, load_scan, load_textured_mesh, open_sequence
from lsg.viz.topdown import INK, INK2, STATE_COLOR, SURFACE

# pixel classes of panel (e), in drawing order (index = class id)
VERDICT = [
    ("walls / floor / ceiling (no vote)", "#efeee9"),
    ("no depth / in between", "#d3d2ca"),
    ("something in front", "#8f8e86"),
    ("agrees, no contrast", "#a9c8ef"),
    ("seen: depth agrees", STATE_COLOR["moved"]),
    ("seen through: free space", STATE_COLOR["removed"]),
]
KIND_TEXT = {"free": "seen through", "seen": "seen", "noinfo": "no information", "unobserved": "not in view"}
DEPTH_CMAP = "viridis"  # perceptually uniform, wide lightness range: near = dark, far = bright

# name -> (ref, rescan, node, rescan frame, what it shows), for `--examples`; frames picked by eye among the node's
# frames with that vote (the colour image should be sharp). Outcomes in perception mode from outputs/chg3_final/val/FULL.
_P1 = ("8eabc45f-5af7-2f32-8528-640861d2a135", "5341b7b3-8a66-2cdd-856d-9d70e194568b")
_P2 = ("10b17940-3938-2467-8a7a-958300ba83d3", "c2d99341-1947-2fbf-817a-5aa9b44f724f")
_P3 = ("ddc73797-765b-241a-9e2c-097c5989baf6", "c7895f07-339c-2d13-8176-7418b6e8d7ce")
_P4 = ("4138582f-a238-2435-8332-6902542c2823", "5341b7a5-8a66-2cdd-8751-70b98263cb8d")
_P5 = ("280d8ebb-6cc6-2788-9153-98959a2da801", "4731976c-f9f7-2a1a-95cc-31c4d1751d0b")
EXAMPLES = {
    "render_compare_seen": (*_P1, "r5", 117, "bench unchanged: depth agrees with contrast behind it (seen; unchanged, correct)"),
    "render_compare_partial": (*_P4, "r4", 12, ("ottoman moved 0.2 m: half its old place agrees, half is seen through, "
                               "neither reaches half: no information this frame (moved, correct)")),
    "render_compare_replaced": (*_P5, "r46", 246, ("pillow moved 1.0 m, a blanket now lies in its place: depth agrees "
                                "(seen; kept unchanged, a missed move)")),
    "render_compare": (*_P1, "r71", 66, "removed pillow: its place on the sofa is seen through (removed, correct)"),
    "render_compare_moved": (*_P2, "r12", 93, "couch moved 2.3 m: its old place is seen through (moved, correct)"),
    "render_compare_occluded": (*_P2, "r24", 153, ("table hidden behind the moved couch: something in front, no "
                                "information (stays unknown)")),
    "render_compare_tv": (*_P3, "r42", 126, ("TV still there: its black screen gives no depth and the few valid pixels "
                          "see through to the wall (deleted: a false deletion)")),
}  # fmt: skip


@dataclass
class Walk:
    """What the frame loop saw on one pair: per processed frame the measured depth, reference render and pose."""

    ref: str
    rescan: str
    cfg: dict[str, Any]
    frames: dict[int, dict[str, np.ndarray]]
    fids: list[int]
    K: np.ndarray
    K_color: np.ndarray
    W: int
    H: int
    renderer: Renderer
    voting: set[int]
    label_of: dict[int, str]
    texture: TextureRenderer | None = None  # loaded on first use (`reference_color`)


class _Capture:
    """Observation source that keeps the inputs of every processed frame."""

    def __init__(self):
        self.frames: dict[int, dict] = {}

    def begin(self, ref_ctx, seq_info):
        self.info = seq_info  # the walk's own SeqInfo: its T_align is updated during the walk

    def observe(self, frame, ref_render):
        pose = np.asarray(self.info.T_align, float) @ frame.pose_c2w  # the pose this frame was rendered at
        self.frames[int(frame.idx)] = {"depth": frame.depth, "d_exp": ref_render.depth, "lab": ref_render.labels,
                                       "pose": pose}  # fmt: skip

    def finish(self):
        return []

    def peek(self):
        return []


def replay(ref: str, rescan: str, cfg: dict[str, Any]) -> Walk:
    """Run the frame loop of FULL (votes with contrast) on one pair and keep every frame's inputs."""
    cfg = copy.deepcopy(cfg)
    root = data_root(cfg)
    R = load_scan(root, ref, with_intrinsics=False)
    objs_R, rels_R = load_objects(root, ref), load_relations(root, ref)
    G_ref = build_reference_graph(R, objs_R, rels_R, "gt", cfg)
    voting = {object_id_of(n.id) for n in G_ref.nodes if n.level in ("anchor", "thing") and object_id_of(n.id)}
    label_of = {object_id_of(n.id): n.label for n in G_ref.nodes if object_id_of(n.id)}
    r = Renderer(R.vertices, R.faces, R.face_labels)
    cap = _Capture()
    with open_sequence(root, rescan) as seq:
        fids = select_frames(seq.frame_ids, int(cfg["frames"]["stride"]), 1.0, 0, key=rescan)
        T_raw = load_meta(root)[rescan].T_align
        ctx, info, _ = prepare_pair(root, R, objs_R, rels_R, seq, rescan, T_raw, cfg, fids, G_ref=G_ref)
        run_online(ctx, info, seq, cap, cfg, renderer=r, contrast=True)
    W, H = info.intrinsics.depth_wh
    return Walk(ref, rescan, cfg, cap.frames, fids, info.intrinsics.K_depth, info.intrinsics.K_color, W, H, r,
                voting, label_of)  # fmt: skip


def classify(w: Walk, f: int) -> tuple[np.ndarray, dict[int, Any]]:
    """Pixel classes (index into VERDICT) and the frame votes {objectId: FrameVote} of frame `f`."""
    vc = w.cfg["visibility"]
    tau_near, tau_free = float(vc["tau_near"]), float(vc["tau_free"])
    c = w.frames[f]
    d_obs, d_exp, lab = c["depth"].astype(float), c["d_exp"].astype(float), c["lab"]
    on_node = np.isin(lab, list(w.voting))
    db = w.renderer.depth_behind(w.K, invert(c["pose"]), w.W, w.H, c["d_exp"], lab, on_node,
                                 float(vc.get("contrast_eps", 1e-3)))  # fmt: skip
    votes = frame_votes(w.voting, c["d_exp"], lab, c["depth"], w.cfg, d_behind=db)
    valid = (d_obs > 0) & (d_exp > 0)
    diff = d_obs - d_exp
    agree = valid & (np.abs(diff) <= tau_near)
    cls = np.where(on_node, 1, 0)
    cls[on_node & valid & (diff < -tau_near)] = 2
    cls[on_node & agree] = 3
    cls[on_node & agree & (db > d_exp + tau_free)] = 4
    cls[on_node & valid & (diff > tau_free)] = 5
    return cls, votes


def color_on_depth_grid(rgb: np.ndarray, Kc: np.ndarray, Kd: np.ndarray, H: int, W: int, s: int = 4) -> np.ndarray:
    """The colour image resampled onto the depth grid (s x finer), stored orientation. Both cameras share the centre."""
    v, u = np.mgrid[0 : H * s, 0 : W * s]
    ud, vd = (u + 0.5) / s - 0.5, (v + 0.5) / s - 0.5
    uc = (ud - Kd[0, 2]) / Kd[0, 0] * Kc[0, 0] + Kc[0, 2]
    vc = (vd - Kd[1, 2]) / Kd[1, 1] * Kc[1, 1] + Kc[1, 2]
    uc = np.clip(np.rint(uc).astype(int), 0, rgb.shape[1] - 1)
    vc = np.clip(np.rint(vc).astype(int), 0, rgb.shape[0] - 1)
    return rgb[vc, uc]


def reference_color(w: Walk, f: int, s: int = 4) -> np.ndarray:
    """The reference scan's textured mesh rendered at frame `f`'s pose, on the depth grid s x finer (stored
    orientation); white where the mesh has no surface."""
    if w.texture is None:
        w.texture = TextureRenderer(load_textured_mesh(data_root(w.cfg), w.ref))
    Ks = w.K.astype(float).copy()  # pixel centres: u_fine = (u + 0.5) * s - 0.5, as in `color_on_depth_grid`
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (w.K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (w.K[1, 2] + 0.5) * s - 0.5
    rgb, hit = w.texture.render_color(Ks, invert(w.frames[f]["pose"]), w.W * s, w.H * s)
    rgb[~hit] = 255
    return rgb


def up(a: np.ndarray) -> np.ndarray:
    """Stored orientation -> upright (see `SequenceReader.color`)."""
    return np.rot90(a, k=-1)


def node_frames(w: Walk, node: str) -> list[tuple[int, str, int]]:
    """(frame id, vote kind, first-hit pixels) of every processed frame in which `node` is in view."""
    oid = object_id_of(node)
    out = []
    for f, c in w.frames.items():
        v = frame_votes([oid], c["d_exp"], c["lab"], c["depth"], w.cfg)[oid]  # kind without contrast (fast)
        if v.kind != "unobserved":
            out.append((f, v.kind, v.n_px))
    return out


def draw(w: Walk, node: str, f: int, out: str | Path) -> None:
    """The five-panel figure of `node` at rescan frame `f`."""
    oid = object_id_of(node)
    cls, votes = classify(w, f)
    v = votes[oid]
    c = w.frames[f]
    with open_sequence(data_root(w.cfg), w.rescan) as seq:
        rgb = color_on_depth_grid(seq.color(f), w.K_color, w.K, w.H, w.W)
    ref_rgb = reference_color(w, f)
    print(f"{w.ref[:8]}_{w.rescan[:8]} frame {f} (processed #{w.fids.index(f) + 1} of {len(w.fids)}): {node} "
          f"{w.label_of[oid]} vote {v.kind}, {v.n_px} px, {v.n_valid} valid, {v.n_present} agree, {v.n_free} through, "
          f"{v.n_contrast} with contrast")  # fmt: skip

    d_obs, d_exp = c["depth"].astype(float), c["d_exp"].astype(float)
    both = np.r_[d_obs[d_obs > 0], d_exp[d_exp > 0]]
    vmin, vmax = (float(x) for x in np.percentile(both, [2, 90]))  # far tail clipped: contrast where the objects are
    mask = c["lab"] == oid
    s = 4
    big = np.kron(mask, np.ones((s, s), bool))

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(9.0, 3.25), dpi=200)
    # reference scan (a, b) | small gap | rescan (c, d) | verdict (e)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.1, 1, 1, 1], height_ratios=[1, 0.045], hspace=0.12, wspace=0.06)
    cols = [0, 1, 3, 4, 5]
    axs = [fig.add_subplot(gs[0, i]) for i in cols]
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.get_cmap(DEPTH_CMAP).copy()
    cmap.set_bad("#ffffff")
    titles = ["(a) reference scan,\nrendered at this pose", "(b) reference depth,\nrendered at this pose",
              "(c) rescan image", "(d) measured depth",
              f"(e) verdict per pixel\n{w.label_of[oid]}: {KIND_TEXT.get(v.kind, v.kind)}"]  # fmt: skip
    for ax, img in ((axs[0], ref_rgb), (axs[2], rgb)):
        ax.imshow(up(img))
        ax.contour(up(big), levels=[0.5], colors=[SURFACE], linewidths=2.4)
        ax.contour(up(big), levels=[0.5], colors=[INK], linewidths=1.0)
    for ax, d in ((axs[1], d_exp), (axs[3], d_obs)):
        im = ax.imshow(up(np.ma.masked_less_equal(d, 0)), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(up(mask), levels=[0.5], colors=["#ffffff"], linewidths=2.4)
        ax.contour(up(mask), levels=[0.5], colors=[INK], linewidths=1.0)
    axs[4].imshow(up(cls), cmap=ListedColormap([col for _, col in VERDICT]), vmin=-0.5, vmax=len(VERDICT) - 0.5,
                  interpolation="nearest")  # fmt: skip
    axs[4].contour(up(mask), levels=[0.5], colors=[INK], linewidths=1.0)
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=8, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 1:5]), orientation="horizontal", extend="both")  # for (b) and (d)
    cb.set_label("depth [m] (white: no depth; ends: beyond the scale)", color=INK2, fontsize=7)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    present = set(np.unique(cls).tolist()) | {4, 5}
    handles = [Patch(color=col, label=name) for i, (name, col) in enumerate(VERDICT) if i in present][::-1]
    axs[4].legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.01), frameon=False, fontsize=6.5,
                  handlelength=1.2)  # fmt: skip
    lax = fig.add_subplot(gs[1, 0])  # the outline is in every panel: its key goes under (a)
    lax.axis("off")
    outline = Line2D([], [], color=INK, lw=1.0, path_effects=[pe.Stroke(linewidth=2.4, foreground=SURFACE), pe.Normal()])
    lax.legend(handles=[outline], labels=[(f"outline: the {w.label_of[oid]}\nat the first visit, seen\n"
                                          "from this pose (the pixels\nits vote is computed from)")],
               loc="upper left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=6.5, handlelength=2.0)  # fmt: skip
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true", help="draw every entry of EXAMPLES into docs/figures/")
    ap.add_argument("--ref")
    ap.add_argument("--rescan")
    ap.add_argument("--node")
    ap.add_argument("--frame", type=int, help="rescan frame id; omitted: list the frames in which the node is in view")
    ap.add_argument("--out", default="docs/figures/render_compare.png")
    a = ap.parse_args()
    cfg = load_config()
    if a.examples:
        walks: dict[tuple[str, str], Walk] = {}
        for name, (ref, rescan, node, f, what) in EXAMPLES.items():
            if (ref, rescan) not in walks:
                walks[ref, rescan] = replay(ref, rescan, cfg)
            print(f"{name}: {what}")
            draw(walks[ref, rescan], node, f, Path("docs/figures") / f"{name}.png")
        return
    w = replay(a.ref, a.rescan, cfg)
    if a.frame is None:
        for f, kind, npx in node_frames(w, a.node):
            print(f"  frame {f}: {kind} ({npx} px)")
        return
    draw(w, a.node, a.frame, a.out)


if __name__ == "__main__":
    main()
