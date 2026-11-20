"""Perception pipeline (DESIGN §3 candidate steps, §6.6; M4 acceptance: box added -> exactly one
candidate, centroid error < 5 cm).

Synthetic room with real ray casting: floor (z = 0) and four walls at ±4 m, a basket r11 in
both scans. The rescan lives in its own frame (`T_align` != I). Rescan depth is rendered from
the rescan mesh, so unchanged surfaces agree exactly with the reference render.
"""

import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from lsg.change.online import run_online
from lsg.config import load_config
from lsg.geom.transforms import invert
from lsg.graph.model import ROOM_ID, Graph, Node, bbox_dict
from lsg.obs import RefContext, SeqInfo
from lsg.perception import labeler as labeler_mod
from lsg.perception import source as source_mod
from lsg.perception.cluster import cluster, connected_components, door_regions, drop_door_clusters
from lsg.perception.labeler import CLIPLabeler, StubLabeler
from lsg.perception.residual import nearer_mask, nearer_points
from lsg.perception.source import UNLABELLED, PerceptionSource
from lsg.perception.voxels import VoxelEvidence, pack_keys, unpack_keys
from lsg.train_meta import label_table, perception_vocab, train_scans
from synth import INTR, K, T_ALIGN, FakeSeq, H, W, box, look_at, make_scan, quad

NEW_BOX = ([0.35, 0.35, 0.0], [0.65, 0.65, 0.3])  # 0.3 m cube, added in the rescan
BASKET = ([2.0, 2.0, 0.0], [2.4, 2.4, 0.4])


def room(extra=None):
    p = {
        1: (*quad([-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]), 2, "floor"),
        2: (*quad([-6, 4, -1], [6, 4, -1], [6, 4, 4], [-6, 4, 4]), 1, "wall"),
        3: (*quad([-6, -4, -1], [6, -4, -1], [6, -4, 4], [-6, -4, 4]), 1, "wall"),
        4: (*quad([4, -6, -1], [4, 6, -1], [4, 6, 4], [4, -6, 4]), 1, "wall"),
        5: (*quad([-4, -6, -1], [-4, 6, -1], [-4, 6, 4], [-4, -6, 4]), 1, "wall"),
        11: (*box(*BASKET), 40, "basket"),
    }
    p.update(extra or {})
    return p


def orbit(center, n=12, radius=1.5, height=0.5, target_z=None):
    c = np.asarray(center, float)
    tz = c[2] if target_z is None else target_z
    return [look_at([c[0] + radius * np.cos(a), c[1] + radius * np.sin(a), height], [c[0], c[1], tz])
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]  # fmt: skip


def graph_of(R):
    nodes = [Node(ROOM_ID, "room", "room", last_seen=R.scan_id)]
    i = R.instances[11]
    nodes.append(Node("r11", "thing", "basket", 40, bbox_3d=bbox_dict(i.aabb_min, i.aabb_max),
                      centroid=list(i.centroid), last_seen=R.scan_id))  # fmt: skip
    meta = Graph.new_meta(R.scan_id)
    meta["floor_z"] = 0.0
    return Graph(meta, nodes, [])


@pytest.fixture
def cfg():
    """Relaxed candidate rules and candidate log-odds OFF: these tests cover the strict M4 path."""
    c = copy.deepcopy(load_config())
    c["perception"]["relaxed"]["enabled"] = False
    c["confidence"]["candidates"]["enabled"] = False
    return c


@pytest.fixture
def rcfg():
    """CHANGE 1 (relaxed candidate rules) ON, CHANGE 2 candidate log-odds OFF: view counts."""
    c = copy.deepcopy(load_config())
    assert c["perception"]["relaxed"]["enabled"] is True
    c["confidence"]["candidates"]["enabled"] = False
    return c


@pytest.fixture
def ccfg():
    """The default config (CHANGE 1 and CHANGE 2 on) with the start candidate weights 2.5 / 2.5 and p_new_min 0.92
    (CHANGE 3 Part C mechanics; the learned dev_tune values make every confidence ~0.5)."""
    c = copy.deepcopy(load_config())
    assert c["perception"]["relaxed"]["enabled"] is True and c["confidence"]["candidates"]["enabled"] is True
    c["confidence"]["candidates"]["p_new_min"] = 0.92
    c["confidence"]["candidates"]["track"].update(L_seen=2.5, L_free=2.5)
    return c


def run_perception(cfg, ref_parts, rescan_parts, poses, labeler=None, with_crops=False, later_parts=None,
                   n_first=None, snapshot_dir=None):  # fmt: skip
    """Online walk; with `later_parts` the frames from index `n_first` on show that scene instead."""
    R = make_scan(ref_parts, "REF")
    S = make_scan(rescan_parts, "S", T=invert(T_ALIGN))
    seq = FakeSeq(S, poses)
    if later_parts is not None:
        late = FakeSeq(make_scan(later_parts, "S2", T=invert(T_ALIGN)), poses)
        for i, f in enumerate(seq.frame_ids):
            if i >= n_first:
                seq._frames[f] = late._frames[f]
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    info = SeqInfo("S", Path("."), INTR, T_ALIGN, list(seq.frame_ids))
    src = PerceptionSource(labeler, {"box": 29}, cfg, with_crops=with_crops)
    res = run_online(ctx, info, seq, src, cfg, obs_mode="perception", snapshot_dir=snapshot_dir)
    return res, src


# ---- acceptance: synthetic box added ---------------------------------------------------------


def test_box_added_gives_exactly_one_candidate(cfg):
    res, src = run_perception(cfg, room(), room({20: (*box(*NEW_BOX), 29, "box")}), orbit([0.5, 0.5, 0.15]))
    assert len(res.obs) == 1, src.stats
    o = res.obs[0]
    # GT centroids in eval are means of mesh vertices, i.e. of the reconstructed surface: an
    # object's bottom is never observed. Reference = area-weighted centroid of the cube's top +
    # 4 side faces (z = 0.18 m); the solid centre is z = 0.15 m (DECISIONS M4).
    lo, hi = np.array(NEW_BOX[0]), np.array(NEW_BOX[1])
    surf_c = np.array([*(0.5 * (lo + hi))[:2], (hi[2] + 4 * 0.5 * (lo[2] + hi[2])) / 5])
    err = float(np.linalg.norm(o.centroid - surf_c))
    assert err < 0.05, f"centroid error {err:.3f} m"
    assert np.linalg.norm(o.centroid[:2] - 0.5 * (lo + hi)[:2]) < 0.01  # xy: no bias
    # points are in the reference frame, inside the box (plus half a voxel)
    assert np.all(o.points >= np.array(NEW_BOX[0]) - 0.03) and np.all(o.points <= np.array(NEW_BOX[1]) + 0.03)
    assert o.token == "c0" and o.label == UNLABELLED and o.label_topk == [] and o.nyu40 is None
    assert len(o.masks) >= 3 and all(o.masks.area(f) > 0 for f in o.masks)
    assert src.stats["clusters"] == 1 and src.stats["door_dropped"] == 0
    # the basket and the room are unchanged: no other residual survives the filters
    assert res.votes["r11"].free == 0


def test_unchanged_scene_gives_no_candidate(cfg):
    res, src = run_perception(cfg, room(), room(), orbit([0.5, 0.5, 0.15]))
    assert res.obs == [] and src.stats["residual_pts"] == 0


def test_one_viewpoint_repeated_gives_no_candidate(cfg):
    poses = [orbit([0.5, 0.5, 0.15])[0]] * 30
    res, src = run_perception(cfg, room(), room({20: (*box(*NEW_BOX), 29, "box")}), poses)
    assert src.stats["residual_pts"] > 0 and res.obs == []  # 1 decorrelated view < min_views
    cfg["decorrelation"]["enabled"] = False  # ablation: every frame counts
    res, _ = run_perception(cfg, room(), room({20: (*box(*NEW_BOX), 29, "box")}), poses)
    assert len(res.obs) == 1


def test_object_flat_against_wall_is_filtered(cfg):
    poster = ([0.0, 3.94, 0.8], [0.6, 3.99, 1.4])  # 5 cm deep, within struct_dist of the wall
    poses = [look_at([x, 1.5, 1.1], [0.3, 4.0, 1.1]) for x in np.linspace(-1.5, 2.0, 8)]
    res, src = run_perception(cfg, room(), room({20: (*box(*poster), 29, "box")}), poses)
    assert src.stats["nearer_px"] == 0 or res.obs == []
    # the same object 30 cm off the wall is kept
    off = ([0.0, 3.5, 0.8], [0.6, 3.7, 1.4])
    res, _ = run_perception(cfg, room(), room({20: (*box(*off), 29, "box")}), poses)
    assert len(res.obs) == 1


def test_swung_door_is_dropped_by_the_swing_region(cfg):
    closed = quad([1.0, 3.98, 0.0], [2.0, 3.98, 0.0], [2.0, 3.98, 2.0], [1.0, 3.98, 2.0])
    opened = quad([1.0, 3.98, 0.0], [1.0, 2.98, 0.0], [1.0, 2.98, 2.0], [1.0, 3.98, 2.0])  # 90° about x = 1
    ref_p, res_p = room({30: (*closed, 8, "door")}), room({30: (*opened, 8, "door")})
    poses = [look_at([x, 1.0, 1.2], [1.0, 3.5, 1.0]) for x in np.linspace(1.6, 3.2, 8)]
    res, src = run_perception(cfg, ref_p, res_p, poses)
    assert res.obs == [] and src.stats["door_dropped"] >= 1 and src.stats["doors"] == 1
    cfg["perception"]["door_filter"] = False
    res, _ = run_perception(cfg, ref_p, res_p, poses)
    assert len(res.obs) >= 1  # the leaf would be a false "new" candidate


# ---- crops + labels ----------------------------------------------------------------------------


class _ColorSeq:
    """Stands in for `open_sequence(...)` in `PerceptionSource._crops`: grey 540x960 frames."""

    calls: list[int] = []

    def color(self, idx):
        self.calls.append(idx)
        return np.full((540, 960, 3), 127, np.uint8)


def test_crops_go_to_the_labeler(cfg, monkeypatch):
    @contextmanager
    def fake_open(root, scan_id):
        yield _ColorSeq()

    monkeypatch.setattr(source_mod, "open_sequence", fake_open)
    lab = StubLabeler([("box", 0.7), ("basket", 0.2)])
    res, _ = run_perception(cfg, room(), room({20: (*box(*NEW_BOX), 29, "box")}), orbit([0.5, 0.5, 0.15]),
                            labeler=lab, with_crops=True)  # fmt: skip
    (o,) = res.obs
    assert o.label == "box" and o.nyu40 == 29 and o.label_topk == [("box", 0.7), ("basket", 0.2)]
    k = int(cfg["perception"]["crop_k"])
    assert lab.calls == [k] and len(o.crops) == k
    # crops are upright (rotated from the stored orientation) and inside the 960x540 image
    assert all(c.ndim == 3 and c.shape[0] <= 960 and c.shape[1] <= 540 for c in o.crops)


class _FakeCLIP(CLIPLabeler):
    def probs(self, crops):
        self.n_encoded += len(crops)
        p = np.array([0.1, 0.45, 0.45, 0.0])[: len(self.vocab)]
        return p / p.sum()


def test_clip_topk_order_ties_and_cache(tmp_path):
    lab = _FakeCLIP(["chair", "box", "bag", "zebra"], cache_dir=tmp_path)  # sorted: bag, box, chair, zebra
    crop = np.zeros((10, 12, 3), np.uint8)
    top = lab.topk([crop], 3)
    assert [a for a, _ in top] == ["box", "chair", "bag"]  # 0.45 tie broken by label
    assert lab.n_encoded == 1 and len(list(tmp_path.glob("*.json"))) == 1
    assert lab.topk([crop], 3) == top and lab.n_encoded == 1 and lab.n_cached == 1
    assert lab.topk([], 3) == []
    other = _FakeCLIP(["chair", "box", "bag", "zebra", "cup"], cache_dir=tmp_path)
    assert other._key([crop]) != lab._key([crop])  # vocabulary is part of the key
    assert labeler_mod._vocab_hash(["a"], "{}") != labeler_mod._vocab_hash(["a"], "a {}")


# ---- units -------------------------------------------------------------------------------------


def test_nearer_mask_rules(cfg):
    d_exp = np.full((4, 5), 2.0)
    d_obs = np.full((4, 5), 2.0)
    d_obs[0, 0] = 1.85  # 15 cm nearer -> nearer
    d_obs[0, 1] = 1.95  # 5 cm -> not
    d_obs[0, 2] = 0.0  # invalid
    d_obs[0, 3] = 1.0
    d_exp[0, 3] = 0.0  # no reference hit -> excluded
    d_obs[1, 0] = 2.6  # farther (free) -> not nearer
    d_obs[1, 1], d_exp[1, 1] = 4.5, 5.0  # beyond max_depth
    m = nearer_mask(d_obs, d_exp, cfg)
    assert np.argwhere(m).tolist() == [[0, 0]]


def test_nearer_points_backprojects_into_the_reference_frame(cfg):
    d_exp = np.full((H, W), 3.0, np.float32)
    d_obs = d_exp.copy()
    d_obs[80:90, 100:110] = 2.0
    pose = T_ALIGN @ look_at([0, 0, 1], [0, 3, 1])
    pts, pix = nearer_points(d_obs, d_exp, K, pose, None, cfg)
    assert len(pts) == 100 and len(pix) == 100
    v, u = np.divmod(pix, W)
    assert v.min() == 80 and u.max() == 109
    cam = (pts - pose[:3, 3]) @ pose[:3, :3]  # back into the camera frame
    np.testing.assert_allclose(cam[:, 2], 2.0, atol=1e-6)


def test_voxel_evidence_counts_decorrelated_views():
    ve = VoxelEvidence(0.04, d_dec=0.3, theta_dec_deg=20)
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.01], [0.5, 0.5, 0.5]])
    P = look_at([0, -2, 0], [0, 0, 0])
    for _ in range(5):
        idx = ve.add(pts, P)
    assert idx[0] == idx[1] != idx[2] and len(ve) == 2
    assert ve.views[:2].tolist() == [1, 1] and ve.hits[:2].tolist() == [5, 5]
    P2 = P.copy()
    P2[:3, 3] += [0.4, 0, 0]  # 40 cm away -> a new view
    ve.add(pts[:1], P2)
    ve.add(pts[:1], look_at([0, -2, 0], [1.5, 0, 0]))  # turned ~37° -> new view
    assert ve.views[:2].tolist() == [3, 1]
    assert ve.kept(3).tolist() == [int(idx[0])]
    np.testing.assert_allclose(ve.mean_points([idx[2]]), [[0.5, 0.5, 0.5]])
    off = VoxelEvidence(0.04, 0.3, 20, enabled=False)
    for _ in range(5):
        off.add(pts[:1], P)
    assert off.views[0] == 5


def test_voxel_keys_roundtrip():
    ijk = np.array([[0, 0, 0], [-5, 7, -100], [100000, -100000, 3]])
    np.testing.assert_array_equal(unpack_keys(pack_keys(ijk)), ijk)


def test_cluster_components_size_and_order(cfg):
    a = [[x, y, 0] for x in range(3) for y in range(3)]  # 9 voxels
    b = [[10 + x, 0, 0] for x in range(6)] + [[16, 1, 1]]  # 7 voxels, diagonal neighbour
    c = [[30, 30, 30], [31, 30, 30]]  # 2 voxels -> dropped (min_voxels 5)
    coords = np.array(b + a + c)
    lab = connected_components(coords)
    assert len(set(lab.tolist())) == 3 and lab[0] == 0 and lab[7] == 1  # numbered by first voxel
    gap = np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0]])  # one-voxel gap, then a two-voxel gap
    assert connected_components(gap, link=1).tolist() == [0, 1, 2]
    assert connected_components(gap, link=2).tolist() == [0, 0, 1]
    pts = (coords + 0.5) * 0.04
    cfg["perception"]["link_voxels"] = 1
    cands = cluster(np.arange(len(coords)), coords, pts, cfg)
    assert [len(k.voxels) for k in cands] == [9, 7]
    assert sorted(cands[1].voxels.tolist()) == list(range(7))
    assert len(connected_components(np.array(b), connectivity=6)) == 7
    assert len(set(connected_components(np.array(b), connectivity=6).tolist())) == 2


def test_door_regions_cover_both_hinges(cfg):
    R = make_scan(room({30: (*quad([1.0, 3.98, 0.0], [2.0, 3.98, 0.0], [2.0, 3.98, 2.0], [1.0, 3.98, 2.0]), 8, "door")}), "R")
    ctx = RefContext("R", R, graph_of(R), 0.0, {30: 8})
    (r,) = door_regions(ctx, 0.1)
    np.testing.assert_allclose(r.hinges, [[1.0, 3.98], [2.0, 3.98]])
    assert r.radius == pytest.approx(1.1)
    inside = r.contains(np.array([[1.0, 3.0, 1.0], [2.9, 3.98, 1.0], [1.5, 3.5, 2.5], [3.5, 3.5, 1.0]]))
    assert inside.tolist() == [True, True, False, False]
    from lsg.perception.cluster import Candidate

    c_in = Candidate(np.arange(4), np.array([[1.0, 3.2, z] for z in (0.5, 0.8, 1.1, 1.4)]))
    c_out = Candidate(np.arange(4), np.array([[-1.0, 1.0, z] for z in (0.5, 0.8, 1.1, 1.4)]))
    kept, dropped = drop_door_clusters([c_in, c_out], [r], 0.5)
    assert kept == [c_out] and dropped == [c_in]


# ---- vocabulary (train metadata) --------------------------------------------------------------


def test_train_scans_and_label_table():
    from lsg.io.scan3r import Meta, ScanMeta

    I = np.eye(4)
    meta = Meta({
        "A": ScanMeta("A", "A", True, "train", I), "a1": ScanMeta("a1", "A", False, "train", I),
        "B": ScanMeta("B", "B", True, "train", I), "b1": ScanMeta("b1", "B", False, "train", I),
        "V": ScanMeta("V", "V", True, "validation", I), "v1": ScanMeta("v1", "V", False, "validation", I),
    })  # fmt: skip
    assert train_scans(meta, {"B"}) == ["A", "a1"]  # dev/relchange room B and val room V excluded
    tab = label_table({
        "A": {1: {"label": "chair", "nyu40": 5}, 2: {"label": "wall", "nyu40": 1}, 3: {"label": "chair", "nyu40": 5}},
        "a1": {1: {"label": "chair", "nyu40": 5}, 4: {"label": "table", "nyu40": 7}, 5: {"label": "x"}},
    })  # fmt: skip
    assert tab == [("chair", 5, 3), ("table", 7, 1)]  # structure and nyu40-less objects dropped
    G = Graph(Graph.new_meta("R"), [Node("r1", "thing", "pillow", 18), Node("r2", "thing", "chair", 5)], [])
    vocab, nyu = perception_vocab(G, tab, top_n=1)
    assert vocab == ["chair", "pillow"] and nyu == {"chair": 5, "pillow": 18}


# ---- CHANGE 1: relaxed candidate rules (size decides the evidence) ----------------------------

ONE_POSE = [orbit([0.5, 0.5, 0.15])[0]] * 30  # one viewpoint, repeated: 1 decorrelated view
TWO_POSES = [orbit([0.5, 0.5, 0.15])[0], orbit([0.5, 0.5, 0.15])[6]]  # 3 m apart: 2 decorrelated views
WITH_BOX = {20: (*box(*NEW_BOX), 29, "box")}


def test_nearer_mask_tau_override(cfg):
    d_exp = np.full((4, 4), 2.0)
    d_obs = np.full((4, 4), 1.93)  # 7 cm nearer
    assert not nearer_mask(d_obs, d_exp, cfg).any()  # tau_new = 0.10
    assert nearer_mask(d_obs, d_exp, cfg, tau=0.05).all()


def test_large_cluster_needs_one_view(rcfg, cfg):
    res, src = run_perception(rcfg, room(), room(WITH_BOX), ONE_POSE)
    assert len(res.obs) == 1, src.stats
    assert src.stats["large_clusters"] == 1 and src.stats["clusters"] == 1
    assert src.stats["large_replaced"] == 0  # no strict cluster: a single view is below min_views_small
    lo, hi = np.array(NEW_BOX[0]), np.array(NEW_BOX[1])
    assert np.all(res.obs[0].points >= lo - 0.03) and np.all(res.obs[0].points <= hi + 0.03)
    assert len(res.obs[0].masks) >= 1
    res, _ = run_perception(cfg, room(), room(WITH_BOX), ONE_POSE)  # strict path: 1 view < min_views
    assert res.obs == []


def test_small_cluster_needs_two_views(rcfg, cfg):
    rcfg["perception"]["relaxed"]["large_min_voxels"] = 10**6  # no cluster counts as large: strict tier only
    res, src = run_perception(rcfg, room(), room(WITH_BOX), ONE_POSE)
    assert res.obs == [] and src.stats["large_clusters"] == 0
    res, _ = run_perception(rcfg, room(), room(WITH_BOX), TWO_POSES)
    assert len(res.obs) == 1  # min_views_small = 2
    res, _ = run_perception(cfg, room(), room(WITH_BOX), TWO_POSES)
    assert res.obs == []  # the M7 rule needs 3


def test_large_cluster_replaces_its_strict_clusters(rcfg, cfg):
    poses = orbit([0.5, 0.5, 0.15])
    res_s, _ = run_perception(cfg, room(), room(WITH_BOX), poses)
    res, src = run_perception(rcfg, room(), room(WITH_BOX), poses)
    assert len(res.obs) == 1 and len(res_s.obs) == 1  # merged, not duplicated
    assert src.stats["large_clusters"] == 1 and src.stats["large_replaced"] == 1
    assert len(res.obs[0].points) >= len(res_s.obs[0].points)  # the lenient cluster contains the strict one
    assert res.obs[0].token == "c0"


def test_relaxed_is_deterministic(rcfg):
    runs = [run_perception(rcfg, room(), room(WITH_BOX), orbit([0.5, 0.5, 0.15]))[0].obs for _ in range(2)]
    assert [o.token for o in runs[0]] == [o.token for o in runs[1]]
    assert all(np.array_equal(a.points, b.points) for a, b in zip(*runs))


def test_relaxed_with_neutral_values_equals_strict(rcfg, cfg):
    rcfg["perception"]["relaxed"].update(min_views_small=int(cfg["perception"]["min_views"]), large_min_voxels=10**6)
    a, _ = run_perception(cfg, room(), room(WITH_BOX), orbit([0.5, 0.5, 0.15]))
    b, _ = run_perception(rcfg, room(), room(WITH_BOX), orbit([0.5, 0.5, 0.15]))
    assert [o.token for o in a.obs] == [o.token for o in b.obs]
    assert all(np.array_equal(x.points, y.points) for x, y in zip(a.obs, b.obs))


def test_merge_add_keeps_the_strict_cluster_and_link_knob(rcfg, cfg):
    poses = orbit([0.5, 0.5, 0.15])
    res_s, _ = run_perception(cfg, room(), room(WITH_BOX), poses)
    rcfg["perception"]["relaxed"]["merge"] = "add"
    res, src = run_perception(rcfg, room(), room(WITH_BOX), poses)
    assert len(res.obs) == 1 and src.stats["large_replaced"] == 0 and src.stats["large_clusters"] == 0
    assert np.array_equal(res.obs[0].points, res_s.obs[0].points)  # the strict candidate is untouched
    # a large cluster without a strict cluster is still added
    res, src = run_perception(rcfg, room(), room(WITH_BOX), ONE_POSE)
    assert len(res.obs) == 1 and src.stats["large_clusters"] == 1
    # the connectivity knob reaches the lenient tier
    pts = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])  # gaps of 2 voxels
    c = {"perception": {"link_voxels": 2, "min_voxels": 1}}
    assert len(cluster(np.arange(3), pts, pts * 0.04, c)) == 1
    assert len(cluster(np.arange(3), pts, pts * 0.04, c, link=1)) == 3


# ---- CHANGE 2: candidate confidence as per-voxel log-odds --------------------------------------

ORBIT = orbit([0.5, 0.5, 0.15])


def _same(a, b):
    return [o.token for o in a] == [o.token for o in b] and all(np.array_equal(x.points, y.points) for x, y in zip(a, b))


def test_candconf_with_neutral_values_equals_change1(ccfg, rcfg, cfg):
    """L_corr 0, no see-through votes, l_keep 5.0 / 2.5 = CHANGE 1's 2 / 1 views (same tokens and points)."""
    ccfg["confidence"]["candidates"].update(L_corr=0.0, free_votes=False, l_keep_small=5.0, l_keep_large=2.5)
    for poses in (ORBIT, ONE_POSE, TWO_POSES):
        a, _ = run_perception(rcfg, room(), room(WITH_BOX), poses)
        b, _ = run_perception(ccfg, room(), room(WITH_BOX), poses)
        assert _same(a.obs, b.obs), len(poses)
    # without the relaxed rules the strict grid needs L_hit * min_views (M7's 3 views)
    ccfg["perception"]["relaxed"]["enabled"] = False
    for poses in (ORBIT, TWO_POSES):
        a, _ = run_perception(cfg, room(), room(WITH_BOX), poses)
        b, _ = run_perception(ccfg, room(), room(WITH_BOX), poses)
        assert _same(a.obs, b.obs)


def test_node_weights_do_not_change_candidates(ccfg):
    """CHANGE 3 Part C: the voxel grid has its own weights (L_vox_*); the node weights L_hit / L_miss do not move it."""
    w = copy.deepcopy(ccfg)
    w["confidence"].update(L_hit=0.7, L_miss=4.0, L_free_like=4.0)
    for kw in ({}, {"later_parts": room(), "n_first": 4}):
        a, sa = run_perception(ccfg, room(), room(WITH_BOX), ORBIT, **kw)
        b, sb = run_perception(w, room(), room(WITH_BOX), ORBIT, **kw)
        assert _same(a.obs, b.obs) and [o.confidence for o in a.obs] == [o.confidence for o in b.obs]
        assert sa.stats == sb.stats


def test_cluster_votes_keep_the_see_through_signal(ccfg):
    """CHANGE 3 Part C: a box seen in 6 of 12 views and then seen through gets 6 seen / 6 free cluster votes
    (p 0.5, not emitted at p_new_min); CHANGE 2's median voxel log-odds still says 0.99."""
    from lsg.change.baselines import emits_new

    res, _ = run_perception(ccfg, room(), room(WITH_BOX), ORBIT)
    assert [o.cand_votes for o in res.obs] == [(12, 0)] and res.obs[0].confidence > 0.999
    on, _ = run_perception(ccfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=6)
    assert [o.cand_votes for o in on.obs] == [(6, 6)] and on.obs[0].confidence == 0.5
    assert not emits_new(on.obs[0], ccfg)
    off_cfg = copy.deepcopy(ccfg)
    off_cfg["confidence"]["candidates"]["track"]["enabled"] = False
    off, _ = run_perception(off_cfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=6)
    assert off.obs[0].cand_votes is None and off.obs[0].confidence > 0.99 and emits_new(off.obs[0], off_cfg)


def test_one_observation_is_enough_and_confidence_is_set(ccfg):
    """Erik: a candidate is added after one decorrelated observation, small clusters too."""
    ccfg["perception"]["relaxed"]["large_min_voxels"] = 10**6  # strict tier only
    res, src = run_perception(ccfg, room(), room(WITH_BOX), ONE_POSE)
    assert len(res.obs) == 1 and src.stats["free_votes"] == 0
    p = res.obs[0].confidence
    assert 0.92 <= p < 0.98  # one decorrelated hit (2.5) + capped correlated hits (<= 1.0)
    res, src = run_perception(ccfg, room(), room(WITH_BOX), ORBIT)
    assert len(res.obs) == 1 and res.obs[0].confidence > 0.999  # 12 views: clipped at +8
    import json

    json.dumps(src.stats)  # stats go into metrics.json: plain Python types only
    assert type(res.obs[0].confidence) is float


def test_transient_is_removed_by_see_through_votes(ccfg, rcfg):
    """A box seen in the first frame only, then seen through from other viewpoints, gives no candidate."""
    a, _ = run_perception(rcfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=1)
    assert len(a.obs) == 1  # CHANGE 1: one view is enough for a large cluster, nothing can lower it
    b, src = run_perception(ccfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=1)
    assert b.obs == [] and src.stats["free_votes"] > 0 and src.stats["large_seen_through_voxels"] > 0
    # the same walk without see-through votes keeps it (ablation FULL-candconf-free)
    ccfg["confidence"]["candidates"]["free_votes"] = False
    c, src = run_perception(ccfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=1)
    assert len(c.obs) == 1 and "free_votes" in src.stats and src.stats["free_votes"] == 0


def test_object_that_stays_is_not_seen_through(ccfg):
    res, src = run_perception(ccfg, room(), room(WITH_BOX), ORBIT)
    assert len(res.obs) == 1 and src.stats["seen_through_voxels"] == 0
    assert src.stats["large_seen_through_voxels"] == 0


def test_candconf_is_deterministic(ccfg):
    runs = [run_perception(ccfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=4)[0].obs
            for _ in range(2)]  # fmt: skip
    assert _same(runs[0], runs[1]) and [o.confidence for o in runs[0]] == [o.confidence for o in runs[1]]


def _one_voxel(**kw):
    """A VoxelEvidence with one voxel at (0, 0, 2) hit once from the identity pose (camera at 0, +z)."""
    ve = VoxelEvidence(0.04, 0.30, 20.0, L_corr=0.25, corr_cap=1.0, clip=8.0, **kw)
    ve.add(np.array([[0.01, 0.01, 2.01]]), np.eye(4))
    return ve


def _moved(dx):
    P = np.eye(4)
    P[0, 3] = dx
    return P


def test_free_vote_rules_occlusion_invalid_out_of_view_and_window():
    from lsg.perception.voxels import window_min_depth

    bg = np.full((H, W), 5.0)
    ve = _one_voxel()
    assert ve.observe_free(window_min_depth(bg, 1), K, _moved(0.5)) == 1  # seen through
    assert ve.logodds([0])[0] == pytest.approx(0.0)
    for depth in (np.full((H, W), 1.0),  # something in front: occlusion
                  np.zeros((H, W))):     # invalid depth  # fmt: skip
        ve = _one_voxel()
        assert ve.observe_free(window_min_depth(depth, 1), K, _moved(0.5)) == 0
    ve = _one_voxel()
    P = np.eye(4)
    P[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking the other way: out of view (behind)
    assert ve.observe_free(window_min_depth(bg, 1), K, P) == 0
    assert ve.observe_free(window_min_depth(bg, 1), K, np.eye(4), skip=np.array([0])) == 0  # hit this frame
    # silhouette: one neighbouring pixel at the voxel's depth blocks the 3x3 window, not a 1x1 one
    ve = _one_voxel()
    X = ve.mean_points([0])[0] - np.array([0.5, 0, 0])
    u, v = int(round(K[0, 0] * X[0] / X[2] + K[0, 2])), int(round(K[1, 1] * X[1] / X[2] + K[1, 2]))
    edge = bg.copy()
    edge[v, u + 1] = 2.0
    assert ve.observe_free(window_min_depth(edge, 1), K, _moved(0.5)) == 0
    assert ve.observe_free(window_min_depth(edge, 0), K, _moved(0.5)) == 1
    assert window_min_depth(bg, 1, max_depth=4.0).max() == 0.0  # beyond max_depth = invalid


def test_correlated_votes_are_capped():
    ve = _one_voxel()
    for _ in range(30):  # the same viewpoint again and again
        ve.add(np.array([[0.01, 0.01, 2.01]]), np.eye(4))
    assert ve.views[0] == 1 and ve.logodds([0])[0] == pytest.approx(2.5 + 1.0)
    ve.add(np.array([[0.01, 0.01, 2.01]]), _moved(0.5))  # decorrelated: full vote, cap resets
    assert ve.logodds([0])[0] == pytest.approx(6.0)
    for _ in range(10):
        ve.add(np.array([[0.01, 0.01, 2.01]]), _moved(0.5))
    assert ve.logodds([0])[0] == pytest.approx(7.0)
    from lsg.perception.voxels import window_min_depth

    dm = window_min_depth(np.full((H, W), 5.0), 1)
    for _ in range(10):  # first see-through vote counts fully, the correlated ones are capped too
        ve.observe_free(dm, K, _moved(0.5))
    assert ve.logodds([0])[0] == pytest.approx(7.0 - 2.5 - 1.0)
    assert ve.kept_by_logodds(3.5).tolist() == [0] and ve.kept_by_logodds(3.6).size == 0


def test_live_candidates_go_to_the_snapshot_sidecar(ccfg, tmp_path):
    import json

    ccfg["online"]["snapshot_every"] = 1
    res, src = run_perception(ccfg, room(), room(WITH_BOX), ORBIT, later_parts=room(), n_first=1,
                              snapshot_dir=tmp_path)  # fmt: skip
    side = sorted(tmp_path.glob("candidates_frame_*.json"))
    assert len(side) == len(ORBIT) and len(sorted(tmp_path.glob("frame_*.json"))) == len(ORBIT)
    first = json.loads(side[0].read_text(encoding="utf-8"))["candidates"]
    last = json.loads(side[-1].read_text(encoding="utf-8"))["candidates"]
    assert len(first) == 1 and first[0]["confidence"] > 0.9 and first[0]["tier"] == "large"
    assert last == [] and res.obs == []
    assert src.stats["frames"] == len(ORBIT)  # live calls do not touch the stats of the walk
