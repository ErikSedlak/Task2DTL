"""Online cumulative alignment (CHANGE 3 Part B, B3)."""

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_perception import graph_of, orbit, room  # noqa: E402
from synth import INTR, K, T_ALIGN, FakeSeq, look_at, make_scan  # noqa: E402

from lsg.change.online import online_aligner, run_online
from lsg.config import load_config
from lsg.geom.align import OnlineAligner, mesh_target
from lsg.geom.transforms import invert, rotation_angle_deg
from lsg.obs import RefContext, SeqInfo
from lsg.oracle.source import OracleSource

CFG = load_config()


def _cfg(**online):
    c = copy.deepcopy(CFG)
    c["alignment"]["online"].update(enabled=True, **online)
    return c


def _offset(T, dt=(0.05, -0.03, 0.0), deg=2.0):
    a = np.radians(deg)
    E = np.eye(4)
    E[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    E[:3, 3] = dt
    return E @ T


def _err(Ta, Tb):
    D = Ta @ invert(Tb)
    return float(np.linalg.norm(D[:3, 3])), rotation_angle_deg(D)


def _dense(parts, it=5):
    """Subdivided meshes: the ICP target is the mesh's vertices (dense on real scans, 4 per quad here)."""
    import open3d as o3d

    out = {}
    for oid, (V, F, nyu, lab) in parts.items():
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        m = m.subdivide_midpoint(number_of_iterations=it)
        out[oid] = (np.asarray(m.vertices), np.asarray(m.triangles), nyu, lab)
    return out


def _scene():
    parts = _dense(room())
    R = make_scan(parts, "REF")
    S = make_scan(parts, "S", T=invert(T_ALIGN))
    # views into the corner x = 4 / y = 4 (two walls + floor + the basket): every axis is constrained
    tg = [(4, 4, 0.5), (4, 2.5, 0.3), (2.5, 4, 0.3), (4, 3.2, 1.0), (3.2, 4, 1.0), (2.2, 2.2, 0.2)]
    poses = [look_at([1.6 + 0.2 * (i % 3), 1.6 + 0.2 * (i // 3 % 2), 1.4], t) for i, t in enumerate(tg * 2)]
    return R, S, poses


def test_aligner_recovers_a_known_offset_and_needs_enough_frames():
    R, S, poses = _scene()
    seq = FakeSeq(S, poses)
    T_raw = _offset(T_ALIGN)
    al = OnlineAligner(mesh_target(R.vertices, R.faces), T_raw, dict(_cfg()["alignment"]))
    for f in seq.frame_ids[:2]:
        al.add_frame(seq.frame(f).depth, K, seq.frame(f).pose_c2w)
    assert al.update() == (False, {"frames": 2, "status": "too_few_frames"})
    for f in seq.frame_ids[2:]:
        al.add_frame(seq.frame(f).depth, K, seq.frame(f).pose_c2w)
    changed, info = al.update()
    assert changed and info["status"] == "accepted"
    dt, dr = _err(al.T, T_ALIGN)
    assert dt < 0.01 and dr < 0.5, (dt, dr)


def test_one_wall_is_degenerate():
    R, S, _ = _scene()
    wall = [look_at([3.0, y, 1.5], [4.0, y, 1.5]) for y in np.linspace(-1, 1, 5)]  # only the wall x = 4 in view
    seq = FakeSeq(S, wall)
    al = OnlineAligner(mesh_target(R.vertices, R.faces), _offset(T_ALIGN, dt=(0.0, 0.02, 0.0), deg=0.0),
                       dict(_cfg(min_cloud_frames=1)["alignment"]))  # fmt: skip
    for f in seq.frame_ids:
        al.add_frame(seq.frame(f).depth, K, seq.frame(f).pose_c2w)
    changed, info = al.update()
    assert not changed and info["status"] in ("degenerate", "rejected")


def test_run_online_updates_t_align_only_when_enabled():
    R, S, poses = _scene()
    seq = FakeSeq(S, poses)
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    for enabled in (False, True):
        c = _cfg(every=4, min_cloud_frames=2)
        c["alignment"]["frame_step"] = 1
        c["alignment"]["online"]["enabled"] = enabled
        info = SeqInfo("S", Path("."), INTR, _offset(T_ALIGN), list(seq.frame_ids))
        raw = info.T_align.copy()
        res = run_online(ctx, info, seq, OracleSource(S, {}, c), c, obs_mode="oracle")
        if enabled:
            assert _err(info.T_align, T_ALIGN)[0] < 0.01 and res.stats["align_online"][-1]["status"] == "accepted"
        else:
            assert np.array_equal(info.T_align, raw) and "align_online" not in res.stats
    assert online_aligner(ctx, info, CFG) is not None  # default: online (Erik 2026-11-21)
    off = copy.deepcopy(CFG)
    off["alignment"]["online"]["enabled"] = False
    assert online_aligner(ctx, info, off) is None


def test_structure_target_and_pipeline_skip():
    from lsg.change.pipeline import refined_alignment

    R, S, poses = _scene()
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    info = SeqInfo("S", Path("."), INTR, T_ALIGN.copy(), [])
    al = online_aligner(ctx, info, _cfg(target="structure", min_target_points=3))
    n_struct = len(np.asarray(al.target.points))
    assert n_struct < len(R.vertices)  # the basket is not in the target
    T, ainfo = refined_alignment(None, R, "S", T_ALIGN, _cfg(), [], use_cache=False)
    assert np.array_equal(T, T_ALIGN) and ainfo["online"] is True
