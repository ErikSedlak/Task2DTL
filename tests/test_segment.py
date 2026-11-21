"""Co-segmentation of candidate voxels (CHANGE 3 Part N, N1): geometric frame segments, SAM cache
reader, majority segments, cluster linking, and the PerceptionSource wiring."""

import copy

import numpy as np
import pytest

from lsg.config import load_config
from lsg.perception.cluster import cluster
from lsg.perception.sam_cache import paint, rescan_dir, to_depth_grid
from lsg.perception.segment import SamSegments, cosegment_cluster, seg_method, segment_frame, voxel_segments

CFG = load_config()


def test_segment_frame_cuts_depth_steps_not_ramps():
    d = np.ones((6, 8))
    d[:, 4:] = 1.5  # 0.5 m step between columns 3 and 4
    pix = np.arange(48)
    seg = segment_frame(d, pix, CFG)
    assert len(set(seg[d.ravel() == 1.0])) == 1 and len(set(seg[d.ravel() == 1.5])) == 1
    assert seg[0] != seg[4]
    ramp = 1.0 + 0.01 * np.tile(np.arange(8), (6, 1))  # 1 cm per pixel < 3 cm + 2 %
    assert len(set(segment_frame(ramp, pix, CFG))) == 1
    # only residual pixels form the graph: two residual islands -> two segments
    seg2 = segment_frame(np.ones((6, 8)), np.array([0, 1, 6, 7]), CFG)
    assert seg2[0] == seg2[1] and seg2[2] == seg2[3] and seg2[0] != seg2[2]


def test_voxel_segments_majority_and_unknown():
    v, s = voxel_segments(np.array([5, 5, 5, 2, 9]), np.array([1, 1, 0, 3, -1]))
    assert v.tolist() == [2, 5] and s.tolist() == [3, 1]  # voxel 9 has only unknown pixels


def _two_bars():
    """Two 4-voxel bars side by side (adjacent within link 1)."""
    coords = np.array([[x, y, 0] for y in (0, 1) for x in range(4)])
    pts = coords * 0.04
    return np.arange(8), coords, pts


def test_cosegment_cluster_splits_by_segments():
    vi, co, pts = _two_bars()
    split = [(vi, np.array([0, 0, 0, 0, 1, 1, 1, 1]))] * 2
    c, st = cosegment_cluster(vi, co, pts, split, 8, min_voxels=1, link=1)
    assert sorted(len(x.voxels) for x in c) == [4, 4] and st["cut_edges"] > 0
    one = [(vi, np.zeros(8, np.int64))]
    assert len(cosegment_cluster(vi, co, pts, one, 8, 1, 1)[0]) == 1
    # never co-observed -> spatial link as before; equal to cluster.cluster
    c0, _ = cosegment_cluster(vi, co, pts, [], 8, 1, 1)
    ref = cluster(vi, co, pts, {"perception": {"min_voxels": 1, "link_voxels": 1}})
    assert [x.voxels.tolist() for x in c0] == [x.voxels.tolist() for x in ref]
    # a tie (one frame same, one frame different) keeps the link
    tie = [(vi, np.zeros(8, np.int64)), (vi, np.array([0, 0, 0, 0, 1, 1, 1, 1]))]
    assert len(cosegment_cluster(vi, co, pts, tie, 8, 1, 1)[0]) == 1


def test_sam_label_map_rotation_and_sampling(tmp_path):
    # upright 8 x 4 (H x W) image: two masks, the smaller one painted last (wins)
    big = np.zeros((8, 4), bool)
    big[:, :] = True
    small = np.zeros((8, 4), bool)
    small[0:2, 0:2] = True
    lab_up = paint([{"segmentation": small, "area": 4}, {"segmentation": big, "area": 32}], (8, 4))
    assert lab_up[0, 0] == 1 and lab_up[7, 3] == 0
    stored = to_depth_grid(lab_up, (8, 4))  # stored = 4 x 8 (H x W), depth grid equal size
    assert stored.shape == (4, 8)
    assert np.array_equal(np.rot90(stored, k=-1), lab_up)  # viewing rotation gives the upright map back
    # reader: missing frame -> clear error; present frame -> per-pixel ids
    sam = SamSegments(tmp_path)
    with pytest.raises(FileNotFoundError, match="lsg sam cache"):
        sam.labels(3)
    np.savez_compressed(tmp_path / "000003.npz", labels=stored.astype(np.int16))
    assert sam.segment(3, np.array([0, 31])).tolist() == [stored.ravel()[0], stored.ravel()[31]]


def test_seg_method_and_cache_key():
    c = copy.deepcopy(CFG)
    assert seg_method(c) == "geometric"  # CHANGE 3 Part N default (N1 winner)
    c["perception"]["segment"]["method"] = "kmeans"
    with pytest.raises(ValueError):
        seg_method(c)
    c2 = copy.deepcopy(CFG)
    c2["perception"]["segment"]["sam"]["points_per_side"] = 16
    assert rescan_dir(CFG, "S") != rescan_dir(c2, "S") and rescan_dir(CFG, "S").name == "S"


def test_geometric_cosegmentation_splits_two_touching_boxes():
    """A box against a taller, offset box: one blob without co-segmentation, two candidates with
    geometric segments (crease + any-disagreement cut + no bridge for unseen pairs)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_perception import box, orbit, room, run_perception

    A = ([0.2, 0.35, 0.0], [0.5, 0.65, 0.3])
    B = ([0.5, 0.45, 0.0], [0.8, 0.75, 0.6])
    scene = room({20: (*box(*A), 29, "box"), 21: (*box(*B), 29, "box")})
    out = {}
    for m in ("none", "geometric"):
        c = copy.deepcopy(CFG)
        c["perception"]["segment"]["method"] = m
        res, _ = run_perception(c, room(), scene, orbit([0.5, 0.5, 0.15], radius=1.8, height=1.0))
        out[m] = res.obs
    assert len(out["none"]) == 1
    assert len(out["geometric"]) == 2
    a, b = sorted(out["geometric"], key=lambda o: o.centroid[0])
    assert a.centroid[0] < 0.5 < b.centroid[0]  # one candidate per box
