import numpy as np
import pytest

from lsg.config import load_config
from lsg.graph.relations import (
    FLOOR,
    SupportCandidate,
    aabb_gap,
    estimate_floor_z,
    gt_support,
    infer_support,
    pick_parent,
    support_type,
)


@pytest.fixture
def rcfg():
    return load_config()["relations"]


def box_points(lo, hi, step=0.01):
    """Surface-ish point grid filling an axis-aligned box (dense enough for the rules)."""
    axes = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return g


FLOOR_Z = -1.4


def test_contact_gives_on(rcfg):
    table = SupportCandidate("table", "anchor", box_points([0, 0, FLOOR_Z + 0.70], [1, 1, FLOOR_Z + 0.72]))
    cup = box_points([0.4, 0.4, FLOOR_Z + 0.725], [0.48, 0.48, FLOOR_Z + 0.83])
    p, t, s = infer_support(cup, [table], FLOOR_Z, rcfg)
    assert (p, t) == ("table", "on") and s > 0.9


def test_containment_in_anchor_gives_in(rcfg):
    # cabinet as a hollow shell: points only on its walls, nothing directly under the child
    lo, hi = np.array([0, 0, FLOOR_Z]), np.array([0.6, 0.6, FLOOR_Z + 1.0])
    pts = box_points(lo, hi, 0.02)
    shell = pts[np.any((pts <= lo + 0.011) | (pts >= hi - 0.011), axis=1)]
    shell = shell[~((shell[:, 2] <= lo[2] + 0.011) & np.all((shell[:, :2] > 0.05) & (shell[:, :2] < 0.55), axis=1))]
    cab = SupportCandidate("cab", "anchor", shell)
    towel = box_points([0.2, 0.2, FLOOR_Z + 0.30], [0.4, 0.4, FLOOR_Z + 0.35])
    p, t, _ = infer_support(towel, [cab], FLOOR_Z, rcfg)
    assert (p, t) == ("cab", "in")


def test_containment_needs_anchor(rcfg):
    lo, hi = np.array([0, 0, FLOOR_Z + 0.5]), np.array([0.6, 0.6, FLOOR_Z + 1.0])
    pts = box_points(lo, hi, 0.02)
    shell = pts[np.any((pts[:, :2] <= lo[:2] + 0.011) | (pts[:, :2] >= hi[:2] - 0.011), axis=1)]
    box = SupportCandidate("box", "thing", shell)  # a thing, not an anchor
    child = box_points([0.2, 0.2, FLOOR_Z + 0.7], [0.4, 0.4, FLOOR_Z + 0.75])
    assert infer_support(child, [box], FLOOR_Z, rcfg) == (None, None, 0.0)


def test_floor_fallback_and_floating(rcfg):
    chair = box_points([0, 0, FLOOR_Z + 0.02], [0.4, 0.4, FLOOR_Z + 0.9], 0.05)
    assert infer_support(chair, [], FLOOR_Z, rcfg) == (FLOOR, "on", 1.0)
    lamp = box_points([0, 0, FLOOR_Z + 1.5], [0.2, 0.2, FLOOR_Z + 1.7], 0.05)
    assert infer_support(lamp, [], FLOOR_Z, rcfg) == (None, None, 0.0)


def test_anchor_beats_thing_on_ties(rcfg):
    top = [0, 0, FLOOR_Z + 0.70], [1, 1, FLOOR_Z + 0.72]
    a = SupportCandidate("tab", "anchor", box_points(*top))
    b = SupportCandidate("mat", "thing", box_points(*top))  # identical geometry
    cup = box_points([0.4, 0.4, FLOOR_Z + 0.725], [0.48, 0.48, FLOOR_Z + 0.83])
    assert infer_support(cup, [b, a], FLOOR_Z, rcfg)[0] == "tab"


def test_best_score_wins(rcfg):
    # child half over a table, half over a box -> whichever covers more slice points
    table = SupportCandidate("table", "anchor", box_points([0, 0, FLOOR_Z + 0.70], [0.43, 1, FLOOR_Z + 0.72]))
    box = SupportCandidate("box", "thing", box_points([0.43, 0, FLOOR_Z + 0.70], [1, 1, FLOOR_Z + 0.72]))
    child = box_points([0.3, 0.4, FLOOR_Z + 0.725], [0.7, 0.6, FLOOR_Z + 0.80])
    assert infer_support(child, [table, box], FLOOR_Z, rcfg)[0] == "box"


def test_self_is_skipped(rcfg):
    pts = box_points([0, 0, FLOOR_Z + 1.0], [0.2, 0.2, FLOOR_Z + 1.1])
    me = SupportCandidate("me", "thing", pts)
    assert infer_support(pts, [me], FLOOR_Z, rcfg, child_id="me")[0] is None


def test_predicate_mapping():
    assert support_type("standing on") == "on"
    assert support_type("hanging on") == "on"
    assert support_type("build in") == "in"
    assert support_type("inside") == "in"
    assert support_type("close by") is None
    assert support_type("attached to") is None


def test_gt_support_floor_wall_and_next_to():
    objects = {1: {"nyu40": 2}, 2: {"nyu40": 1}, 3: {"nyu40": 7}, 10: {"nyu40": 40},
               11: {"nyu40": 40}, 12: {"nyu40": 40}, 13: {"nyu40": 40}}  # fmt: skip
    rels = [
        (10, 3, "standing on"),  # cup on table
        (10, 3, "standing in"),  # same parent, "on" kept
        (11, 1, "standing on"),  # on floor
        (12, 2, "hanging on"),  # on wall only -> excluded
        (13, 2, "attached to"),  # not support
        (13, 3, "lying in"),
        (13, 2, "hanging on"),  # wall + valid parent -> not excluded
        (10, 11, "close by"),
        (11, 10, "close by"),
        (10, 99, "standing on"),  # unknown parent ignored
    ]
    gs = gt_support(rels, objects)
    assert gs.parents[10] == {3: "on"}
    assert gs.parents[11] == {FLOOR: "on"}
    assert 12 not in gs.parents and gs.excluded == {12}
    assert gs.parents[13] == {3: "in"}
    assert gs.next_to == {(10, 11)}


def test_pick_parent_order():
    lv = {"a": "anchor", "t": "thing", "a2": "anchor"}.get
    assert pick_parent({FLOOR: "on", "t": "on", "a": "in"}, lv) == ("a", "in")
    assert pick_parent({FLOOR: "on", "t": "in"}, lv) == ("t", "in")
    assert pick_parent({"a2": "in", "a": "on"}, lv) == ("a", "on")
    assert pick_parent({"r12": "on", "r5": "on"}, lambda _: "thing") == ("r5", "on")


def test_floor_estimate():
    V = np.array([[0, 0, -1.40], [1, 0, -1.41], [0, 1, -1.39], [0, 0, 1.0]] * 50, float)
    nyu = np.array([2, 2, 2, 1] * 50)
    assert estimate_floor_z(V, nyu, {"floor_fallback_percentile": 1.0}) == pytest.approx(-1.40)
    nyu[:] = 1  # no floor labels -> low percentile of all z
    assert estimate_floor_z(V, nyu, {"floor_fallback_percentile": 1.0}) == pytest.approx(-1.41)


def test_aabb_gap():
    z = np.zeros(3)
    assert aabb_gap(z, np.ones(3), np.full(3, 0.5), np.full(3, 2)) == 0.0
    assert aabb_gap(z, np.ones(3), np.array([1.3, 0, 0]), np.array([2, 1, 1])) == pytest.approx(0.3)


def test_floor_level_neighbour_does_not_support(rcfg):
    # basket on the floor right next to a washing machine: the machine's bottom edge is at
    # floor level and must not count as support (support_min_height)
    machine = SupportCandidate("wm", "anchor", box_points([0, 0, FLOOR_Z], [0.6, 0.6, FLOOR_Z + 0.85], 0.02))
    basket = box_points([0.58, 0.1, FLOOR_Z + 0.01], [0.9, 0.4, FLOOR_Z + 0.4], 0.01)
    assert infer_support(basket, [machine], FLOOR_Z, rcfg)[0] == FLOOR
    no_guard = {**rcfg, "support_min_height": 0.0, "contact_min_frac": 0.05}
    assert infer_support(basket, [machine], FLOOR_Z, no_guard)[0] == "wm"
