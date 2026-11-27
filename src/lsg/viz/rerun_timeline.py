"""Rerun timeline of one pair's online walk (M8).

Input = a run directory written by `lsg run` (snapshots are on by default):
`snapshots/<mode>/<ref>_<rescan>/frame_XXXXXX.json` + `seq_info.json`, and `graphs/<mode>/<ref>_<rescan>.json`.

Recording (timeline `frame` = rescan frame index):
- static: the reference mesh (PLY vertex colours) in the reference frame, z up;
- per processed frame: the rescan camera at `T_align @ pose` (pinhole + trajectory so far),
  optionally its colour image (JPEG bytes as stored, rotated 90°);
- per snapshot: node boxes grouped by tentative state (`world/nodes/<state>`), labels with confidence; with
  snapshot method runs (CHANGE 3 Part B) also mid-walk `moved` arrows (from `reference.json`) and `new` boxes;
  perception runs also have a sidecar `candidates_frame_XXXXXX.json` (CHANGE 2): the live candidates
  as `world/candidates`, grey (p 0.5) -> green (p 1) by confidence, labelled "candidate <p>";
- after the last frame (`frame` = last + 1): the final updated graph, moved nodes with an
  old→new arrow, new nodes, removed nodes.

Reads method inputs and outputs only (reference mesh, rescan frames + poses, graphs).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lsg.graph.model import Graph
from lsg.io.scan3r import SequenceReader, read_ply
from lsg.viz.topdown import STATE_COLOR, STATE_ORDER


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def snapshot_paths(snap_dir: str | Path) -> list[Path]:
    """Snapshot files of one pair and mode, in frame order."""
    return sorted(Path(snap_dir).glob("frame_*.json"))


def _log_nodes(rec, rr, base: str, G: Graph, moved_from: dict[str, Any] | None = None) -> None:
    """Boxes per state under `base/<state>`; empty groups are cleared."""
    groups: dict[str, list] = {s: [] for s in STATE_ORDER}
    for n in G.nodes:
        if n.level in ("anchor", "thing") and n.bbox_3d is not None and n.state in groups:
            groups[n.state].append(n)
    for state, nodes in groups.items():
        path = f"{base}/{state}"
        if not nodes:
            rec.log(path, rr.Clear(recursive=False))
            continue
        lo = np.array([n.bbox_3d["min"] for n in nodes], float)
        hi = np.array([n.bbox_3d["max"] for n in nodes], float)
        labels = [n.label + (f" {n.confidence:.2f}" if n.confidence is not None else "") for n in nodes]
        rec.log(path, rr.Boxes3D(mins=lo, sizes=hi - lo, colors=[_rgb(STATE_COLOR[state])] * len(nodes),
                                 labels=labels, radii=0.006 if state in ("unchanged", "unknown") else 0.012))  # fmt: skip
    if moved_from is not None:
        a, v = [], []
        for n in groups["moved"]:
            old = moved_from.get(n.id)
            if old is not None and n.centroid is not None:
                a.append(old)
                v.append(np.asarray(n.centroid, float) - np.asarray(old, float))
        if a:
            rec.log(f"{base}/moves", rr.Arrows3D(origins=a, vectors=v, colors=[_rgb(STATE_COLOR["moved"])] * len(a)))
        else:
            rec.log(f"{base}/moves", rr.Clear(recursive=False))


def confidence_color(p: float | None) -> tuple[int, int, int]:
    """Grey (p 0.5 or unknown) -> green (p 1) ramp for live candidates."""
    lo, hi = np.array([150, 150, 150], float), np.array(_rgb(STATE_COLOR["new"]), float)
    if p is None:
        return tuple(int(x) for x in lo)  # type: ignore[return-value]
    t = float(np.clip((float(p) - 0.5) / 0.5, 0.0, 1.0))
    return tuple(int(round(x)) for x in lo + t * (hi - lo))  # type: ignore[return-value]


def _log_candidates(rec, rr, path: str, side: dict[str, Any]) -> None:
    """Live perception candidates of one snapshot (CHANGE 2 sidecar): boxes coloured by confidence."""
    rows = side.get("candidates", [])
    if not rows:
        rec.log(path, rr.Clear(recursive=False))
        return
    lo = np.array([r["min"] for r in rows], float)
    hi = np.array([r["max"] for r in rows], float)
    labels = ["candidate" + (f" {r['confidence']:.2f}" if r.get("confidence") is not None else "") for r in rows]
    rec.log(path, rr.Boxes3D(mins=lo, sizes=hi - lo, colors=[confidence_color(r.get("confidence")) for r in rows],
                             labels=labels, radii=0.008))  # fmt: skip


def rerun_timeline(
    root: str | Path,
    snap_dir: str | Path,
    out: str | Path,
    graph_final: str | Path | None = None,
    images: bool = False,
    mesh: bool = True,
    spawn: bool = False,
) -> Path:
    """Write the `.rrd` recording of one pair and mode.

    Args:
        root: Data root (reference PLY for the mesh, rescan sequence for poses / images).
        snap_dir: `snapshots/<mode>/<ref>_<rescan>/` of a run (with `seq_info.json`).
        out: Output `.rrd` path.
        graph_final: The run's final G_upd (`graphs/<mode>/<pair>.json`); None = snapshots only.
        images: Also log the colour image of every processed frame (bigger file).
        mesh: Log the reference mesh (off in tests).
        spawn: Open the Rerun viewer on the file afterwards.
    """
    import rerun as rr

    snap_dir = Path(snap_dir)
    info = json.loads((snap_dir / "seq_info.json").read_text(encoding="utf-8"))
    T_align = np.asarray(info["T_align"], float)
    snaps = snapshot_paths(snap_dir)
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {snap_dir} (run `lsg run` without --no-snapshots)")
    by_frame = {int(Graph.load(p).meta["frame_idx"]): p for p in snaps}

    rec = rr.RecordingStream("lsg", recording_id=f"{info['ref']}_{info['rescan']}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.save(str(out))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if mesh:
        ply = read_ply(Path(root) / info["ref"] / "labels.instances.annotated.v2.ply")
        V = np.stack([ply.vertex[k] for k in ("x", "y", "z")], axis=1).astype(np.float32)
        C = np.stack([ply.vertex[k] for k in ("red", "green", "blue")], axis=1).astype(np.uint8)
        rec.log("world/reference_mesh", rr.Mesh3D(vertex_positions=V, triangle_indices=ply.faces.astype(np.uint32),
                                                  vertex_colors=C), static=True)  # fmt: skip

    # arrow origins = reference centroids (CHANGE 3 Part B: snapshots hold moved nodes, so the first one may not)
    ref_file = snap_dir / "reference.json"
    G0 = Graph.load(ref_file if ref_file.is_file() else snaps[0])
    ref_centroid = {n.id: n.centroid for n in G0.nodes if n.centroid is not None}
    traj: list[np.ndarray] = []
    with SequenceReader(Path(root) / info["rescan"]) as seq:
        intr = seq.info
        K, (W, H) = (intr.K_color, intr.color_wh) if images else (intr.K_depth, intr.depth_wh)
        rec.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[W, H], image_plane_distance=0.3),
                static=True)  # fmt: skip
        for idx in info["frame_ids"]:
            pose = seq.pose(int(idx))
            if not np.all(np.isfinite(pose)):
                continue
            pose = T_align @ pose
            rec.set_time("frame", sequence=int(idx))
            rec.log("world/camera", rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
            traj.append(pose[:3, 3].copy())
            rec.log("world/trajectory", rr.LineStrips3D([np.array(traj)], colors=[(82, 81, 78)], radii=0.005))
            if images:
                data = seq.read_bytes(f"frame-{int(idx):06d}.color.jpg")
                rec.log("world/camera/image", rr.EncodedImage(contents=data, media_type="image/jpeg"))
            if int(idx) in by_frame:
                _log_nodes(rec, rr, "world/nodes", Graph.load(by_frame[int(idx)]), moved_from=ref_centroid)
                side = snap_dir / f"candidates_frame_{int(idx):06d}.json"
                if side.is_file():
                    _log_candidates(rec, rr, "world/candidates", json.loads(side.read_text(encoding="utf-8")))
    if graph_final is not None:
        last = max(int(i) for i in info["frame_ids"]) + 1
        rec.set_time("frame", sequence=last)
        rec.log("world/candidates", rr.Clear(recursive=False))  # the final new / moved nodes replace them
        _log_nodes(rec, rr, "world/nodes", Graph.load(graph_final), moved_from=ref_centroid)
    rec.flush()
    rec.disconnect()
    if spawn:
        # On a hybrid-GPU Windows laptop the viewer's default graphics backend exits silently after ~3 s;
        # Vulkan stays open. An explicit WGPU_BACKEND of the user wins.
        env = dict(os.environ)
        if sys.platform == "win32":
            env.setdefault("WGPU_BACKEND", "vulkan")
        subprocess.Popen([sys.executable, "-m", "rerun", str(out)], env=env)
    return out
