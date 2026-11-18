"""`lsg.viz` (M8): top-down figure and Rerun recording on synthetic graphs (no data needed)."""

import json

import numpy as np
from PIL import Image

from lsg.graph.model import Edge, Graph, Node, bbox_dict
from lsg.viz.rerun_timeline import rerun_timeline
from lsg.viz.topdown import RoomOutline, room_outline, topdown_png


def _node(nid, lo, hi, state="unchanged", level="thing", label="box", conf=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Node(id=nid, level=level, label=label, bbox_3d=bbox_dict(lo, hi), centroid=list((lo + hi) / 2),
                state=state, confidence=conf)


def _graphs():
    meta = Graph.new_meta("REF", "RES", "oracle")
    ref = Graph(dict(meta), [Node("room", "room", "room"),
                             _node("r1", [0, 0, 0], [2, 1, 0.8], level="anchor", label="table"),
                             _node("r2", [0.2, 0.2, 0.8], [0.4, 0.4, 1.0], label="cup"),
                             _node("r3", [3, 0, 0], [3.5, 0.5, 0.5], label="chair"),
                             _node("r4", [3, 2, 0], [3.4, 2.4, 0.4], label="bin")],
                [Edge("r2", "r1", "on")])
    upd = Graph(dict(meta), [Node("room", "room", "room"),
                             _node("r1", [0, 0, 0], [2, 1, 0.8], level="anchor", label="table"),
                             _node("r2", [0.2, 0.2, 0.8], [0.4, 0.4, 1.0], "unknown", label="cup"),
                             _node("r3", [1, 2, 0], [1.5, 2.5, 0.5], "moved", label="chair", conf=0.99),
                             _node("r4", [3, 2, 0], [3.4, 2.4, 0.4], "removed", label="bin", conf=0.1),
                             _node("n0", [4, 0, 0], [4.3, 0.3, 0.3], "new", label="bag")],
                [Edge("r2", "r1", "on")])
    return ref, upd


def test_room_outline_keeps_structure_drops_ceiling():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 2.5], [5, 5, 0.5]], float)
    nyu = np.array([2, 1, 22, 5])  # floor, wall, ceiling, chair
    o = room_outline(V, nyu)
    assert o.floor_xy.tolist() == [[0, 0]]
    assert o.struct_xy.tolist() == [[1, 0]]


def test_topdown_png_two_and_three_panels(tmp_path):
    ref, upd = _graphs()
    outline = RoomOutline(np.random.default_rng(0).uniform(-1, 5, (500, 2)), np.array([[-1, -1], [5, 3]], float))
    p2 = topdown_png(ref, upd, tmp_path / "a.png", outline=outline, title="toy", edges=True)
    gt = {"states": {"r1": "unchanged", "r2": "absent_unlisted", "r3": "moved", "r4": "removed"},
          "new": [("bag", [4, 0, 0], [4.3, 0.3, 0.3])], "moved_to": {"r3": ([1, 2, 0], [1.5, 2.5, 0.5])}}
    p3 = topdown_png(ref, upd, tmp_path / "b.png", outline=outline, gt=gt)
    w2, h2 = Image.open(p2).size
    w3, h3 = Image.open(p3).size
    assert w3 > w2 > h2 > 100  # three panels are wider; landscape figure


def _write_sequence(root, scan_id, n=6):
    d = root / scan_id / "sequence"
    d.mkdir(parents=True)
    K = "100 0 50 0 0 100 40 0 0 0 1 0 0 0 0 1"
    (d / "_info.txt").write_text(
        f"m_colorWidth = 100\nm_colorHeight = 80\nm_depthWidth = 100\nm_depthHeight = 80\nm_depthShift = 1000\n"
        f"m_calibrationColorIntrinsic = {K}\nm_calibrationDepthIntrinsic = {K}\nm_frames.size = {n}\n",
        encoding="ascii")
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.1 * i
        (d / f"frame-{i:06d}.pose.txt").write_text("\n".join(" ".join(map(str, r)) for r in T), encoding="ascii")


def test_rerun_timeline_writes_recording(tmp_path):
    import rerun.dataframe as rdf

    ref, upd = _graphs()
    _write_sequence(tmp_path, "RES")
    snap = tmp_path / "snaps"
    snap.mkdir()
    (snap / "seq_info.json").write_text(json.dumps({"ref": "REF", "rescan": "RES", "T_align": np.eye(4).tolist(),
                                                    "frame_ids": [0, 1, 2, 3, 4, 5]}), encoding="utf-8")
    for k in (2, 5):
        g = Graph(dict(ref.meta, frame_idx=k, tentative=True), ref.nodes, ref.edges)
        g.save(snap / f"frame_{k:06d}.json")
    gp = upd.save(tmp_path / "final.json")
    out = rerun_timeline(tmp_path, snap, tmp_path / "x.rrd", graph_final=gp, mesh=False)
    rec = rdf.load_recording(str(out))
    ents = {c.entity_path for c in rec.schema().component_columns()}
    assert {"/world/camera", "/world/trajectory", "/world/nodes/unchanged", "/world/nodes/moved",
            "/world/nodes/moves"} <= ents
    frames = rec.view(index="frame", contents="/world/nodes/**").select().read_all().column("frame").to_pylist()
    assert frames == [2, 5, 6]  # two snapshots + the final graph after the last frame
