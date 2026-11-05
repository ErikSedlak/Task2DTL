import numpy as np
import pytest

from conftest import box_points, make_scan
from lsg.config import load_config
from lsg.eval.gt import gt_changes, gt_parent_changes, parent_change_kind
from lsg.graph.relations import FLOOR, GTSupport
from lsg.io.scan3r import RigidEntry, ScanMeta


def tr(x, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


T_ALIGN = tr(0.5)  # rescan -> reference


def obj(lo, hi, nyu, label, shift=0.0):
    """Box object; `shift` moves it along x (to express it in the rescan frame)."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    lo[0] += shift
    hi[0] += shift
    return (box_points(lo, hi, 0.05), nyu, label)


@pytest.fixture
def pair():
    s = -0.5  # unchanged objects in the rescan frame: p_S = T_align^-1 p_R
    R = make_scan({
        1: obj([-2, -2, -1.4], [3, 3, -1.4], 2, "floor"),
        10: obj([0, 0, -1.4], [0.5, 0.5, -0.5], 5, "chair"),
        11: obj([1, 0, -1.0], [1.4, 0.3, -0.9], 18, "pillow"),
        12: obj([2, 0, -0.7], [2.08, 0.08, -0.6], 40, "cup"),
        13: obj([0, 2, -1.4], [0.3, 2.3, -0.4], 35, "lamp"),
        14: obj([1, 2, -0.7], [1.1, 2.1, -0.5], 40, "vase"),
        15: obj([2, 2, -1.0], [2.8, 2.6, -0.9], 39, "blanket"),
        16: obj([1, 1, -0.7], [2, 1.8, -0.68], 7, "table"),
        17: obj([1.2, 1.2, -0.68], [1.4, 1.3, -0.65], 23, "book"),
    }, "R")  # fmt: skip
    S = make_scan({
        1: obj([-2, -2, -1.4], [3, 3, -1.4], 2, "floor", s),
        10: obj([0, 0, -1.4], [0.5, 0.5, -0.5], 5, "chair", s + 1.0),  # moved 1 m in ref frame
        11: obj([1, 0, -1.0], [1.4, 0.3, -0.9], 18, "pillow", s + 0.03),  # 3 cm: dont_care
        12: obj([2, 0, -0.7], [2.08, 0.08, -0.6], 40, "cup", s + 0.4),
        14: obj([1, 2, -0.7], [1.1, 2.1, -0.5], 40, "vase", s),
        15: obj([2, 2, -1.0], [2.8, 2.6, -0.9], 39, "blanket", s),
        16: obj([1, 1, -0.7], [2, 1.8, -0.68], 7, "table", s),
        20: obj([0, -1, -1.4], [0.3, -0.7, -1.0], 29, "box", s),  # new thing
        21: obj([-2, 0, -1.4], [-2, 0.8, 0.6], 8, "door", s),  # new structure: not scored
    }, "S")  # fmt: skip
    # T_rigid maps reference points into the rescan frame: M_ref = T_align @ T_rigid
    sm = ScanMeta(
        "S", "R", False, "train", T_ALIGN,
        rigid=[RigidEntry(10, np.linalg.inv(T_ALIGN) @ tr(1.0)),
               RigidEntry(11, np.linalg.inv(T_ALIGN) @ tr(0.03)),
               RigidEntry(12, None)],
        removed=[13, 14], nonrigid=[15],
    )  # fmt: skip
    return sm, R, S


def test_gt_statuses(pair):
    sm, R, S = pair
    gt = gt_changes(sm, R, S, {}, {}, load_config())
    st = {i: (o.status, o.reason) for i, o in gt.objects.items()}
    assert 1 not in st  # structure is not scored
    assert st[10] == ("moved", None)
    assert st[11] == ("dont_care", "small_move")
    assert st[12] == ("moved", None)  # bare-int entry
    assert st[13] == ("removed", None)
    assert st[14] == ("dont_care", "removed_but_present")
    assert st[15] == ("dont_care", "nonrigid")
    assert st[16] == ("unchanged", None)
    assert st[17] == ("absent_unlisted", None)


def test_gt_displacement_and_new_location(pair):
    sm, R, S = pair
    gt = gt_changes(sm, R, S, {}, {}, load_config())
    chair = gt.objects[10]
    assert chair.displacement == pytest.approx(1.0)
    np.testing.assert_allclose(chair.M_ref, tr(1.0), atol=1e-12)
    np.testing.assert_allclose(chair.centroid_new, chair.centroid_ref + [1.0, 0, 0], atol=1e-9)
    assert gt.objects[11].displacement == pytest.approx(0.03)
    assert gt.objects[12].displacement is None and gt.objects[12].M_ref is None
    # unchanged table: rescan copy maps back onto the reference copy with raw T_align
    np.testing.assert_allclose(gt.objects[16].centroid_new, gt.objects[16].centroid_ref, atol=1e-9)


def test_gt_new_and_counts(pair):
    sm, R, S = pair
    gt = gt_changes(sm, R, S, {}, {}, load_config())
    assert set(gt.new) == {20}  # door (structure) dropped
    n = gt.new[20]
    assert n.label == "box" and n.level == "thing" and n.size == "small"
    np.testing.assert_allclose(n.centroid, [0.15, -0.85, -1.2], atol=1e-9)  # reference frame
    c = gt.counts()
    assert (c["rigid_raw"], c["rigid_bare"], c["removed_raw"], c["nonrigid_raw"], c["new_raw"]) == (3, 1, 2, 1, 2)
    assert (c["moved"], c["removed"], c["unchanged"], c["absent_unlisted"], c["dont_care"]) == (2, 1, 1, 1, 3)
    assert c["new"] == 1 and c["moved_things_raw"] == 3


def test_size_class(pair):
    sm, R, S = pair
    gt = gt_changes(sm, R, S, {}, {}, load_config())
    assert gt.objects[12].size == "small"  # cup, 8 cm
    assert gt.objects[10].size == "large"  # chair, 0.9 m tall
    assert gt.objects[16].size == "large"  # anchor, regardless of size


def test_dssg_nyu40_overrides_ply(pair):
    sm, R, S = pair
    objs_R = {17: {"nyu40": 1}}  # 3DSSG says the "book" is a wall -> structure -> not scored
    gt = gt_changes(sm, R, S, objs_R, {}, load_config())
    assert 17 not in gt.objects


def test_parent_change_kinds():
    gR = GTSupport(parents={1: {5: "on"}, 2: {5: "on"}, 4: {5: "on"}, 6: {FLOOR: "on"}}, excluded={7})
    gS = GTSupport(parents={1: {5: "on", 6: "on"}, 2: {8: "on"}, 3: {5: "in"}, 7: {5: "on"}})
    assert parent_change_kind(1, gR, gS) == "same"
    assert parent_change_kind(2, gR, gS) == "swapped"
    assert parent_change_kind(3, gR, gS) == "gained"
    assert parent_change_kind(4, gR, gS) == "lost"
    assert parent_change_kind(9, gR, gS) == "none"
    assert parent_change_kind(7, gR, gS) == "excluded"
    sm = ScanMeta("S", "R", False, None, np.eye(4), rigid=[RigidEntry(2, None), RigidEntry(5, None)])
    objs = {2: {"nyu40": 40}, 5: {"nyu40": 7}}  # 5 is an anchor -> not counted
    assert gt_parent_changes(sm, objs, gR, gS) == {2: "swapped"}
