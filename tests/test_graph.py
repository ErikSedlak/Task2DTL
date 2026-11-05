import numpy as np
import pytest

from lsg.config import load_config
from lsg.graph.build import anchor_of, build_reference_graph
from lsg.graph.model import Edge, Graph, Node, object_id_of, ref_node_id, validate
from conftest import box_points, make_scan

FZ = -1.4


@pytest.fixture
def scene():
    z = lambda h: FZ + h  # noqa: E731
    objs = {
        1: (box_points([-1, -1, z(0)], [4, 3, z(0)], 0.05), 2, "floor"),
        2: (box_points([-1, 3, z(0)], [4, 3, z(2.5)], 0.05), 1, "wall"),
        3: (np.concatenate([box_points([0, 0, z(0.70)], [1, 1, z(0.72)]),  # table top + leg
                            box_points([0, 0, z(0.0)], [0.04, 0.04, z(0.70)])]), 7, "table"),
        4: (box_points([2, 0, z(0.40)], [3, 0.4, z(0.42)]), 15, "shelf"),
        10: (box_points([0.4, 0.4, z(0.725)], [0.48, 0.48, z(0.83)], 0.01), 40, "cup"),
        11: (box_points([2.2, 0.1, z(0.425)], [2.5, 0.3, z(0.60)]), 29, "box"),
        12: (box_points([2.25, 0.15, z(0.605)], [2.45, 0.25, z(0.63)], 0.01), 23, "book"),
        13: (box_points([0.0, 2.0, z(0.02)], [0.4, 2.4, z(0.9)], 0.05), 5, "chair"),
    }
    scan = make_scan(objs)
    objects = {i: {"id": str(i), "label": lab, "nyu40": n} for i, (_, n, lab) in objs.items()}
    rels = [
        (10, 3, "standing on"),
        (11, 4, "standing on"),
        (12, 11, "lying on"),
        (12, 4, "lying on"),  # two GT parents: pick_parent prefers the anchor
        (13, 1, "standing on"),
        (3, 1, "standing on"),  # anchor on floor: no support edge for anchors
        (10, 13, "close by"),
    ]
    return scan, objects, rels


def test_gt_graph(scene):
    scan, objects, rels = scene
    g = build_reference_graph(scan, objects, rels, "gt", load_config())
    assert validate(g) == []
    ids = {n.id for n in g.nodes}
    assert ids == {"room", "r3", "r4", "r10", "r11", "r12", "r13"}  # no structure nodes
    sp = g.support_parent()
    assert sp == {"r10": ("r3", "on"), "r11": ("r4", "on"), "r12": ("r4", "on"), "r13": ("floor", "on")}
    nm = g.node_map()
    assert nm["r10"].anchor == "r3" and nm["r12"].anchor == "r4" and nm["r13"].anchor is None
    assert nm["r3"].level == "anchor" and nm["r3"].anchor is None
    assert Edge("r10", "r13", "next_to") in g.edges
    assert sum(e.type == "hierarchy" for e in g.edges) == 6
    assert g.meta["scan_ref"] == "REF" and g.meta["relations"] == "gt"
    assert g.meta["floor_z"] == pytest.approx(FZ)


def test_geometric_graph_matches_scene(scene):
    scan, objects, _ = scene
    g = build_reference_graph(scan, objects, None, "geometric", load_config())
    assert validate(g) == []
    sp = g.support_parent()
    assert sp == {"r10": ("r3", "on"), "r11": ("r4", "on"), "r12": ("r11", "on"), "r13": ("floor", "on")}
    nm = g.node_map()
    assert nm["r12"].anchor == "r4"  # book -> box -> shelf: first anchor up the chain


def test_mode_errors(scene):
    scan, objects, rels = scene
    with pytest.raises(ValueError):
        build_reference_graph(scan, objects, None, "gt", load_config())
    with pytest.raises(ValueError):
        build_reference_graph(scan, objects, rels, "geometric", {})
    with pytest.raises(ValueError):
        build_reference_graph(scan, objects, rels, "other", load_config())


def test_json_round_trip(scene, tmp_path):
    scan, objects, rels = scene
    g = build_reference_graph(scan, objects, rels, "gt", load_config())
    p = g.save(tmp_path / "g.json")
    h = Graph.load(p)
    assert h.to_json() == g.to_json()
    n = h.node("r10")
    assert n.bbox_3d is not None and len(n.bbox_3d["min"]) == 3
    assert n.state == "unchanged" and n.last_seen == "REF" and n.label_topk is None


def test_anchor_of_cycle_safe():
    lv = {"a": "thing", "b": "thing", "s": "anchor"}
    assert anchor_of("a", {"a": "b", "b": "a"}, lv) is None
    assert anchor_of("a", {"a": "b", "b": "s"}, lv) == "s"
    assert anchor_of("a", {"a": "floor"}, lv) is None


def test_validate_catches_problems():
    meta = Graph.new_meta("R")
    room = Node("room", "room", "room")
    t = Node("r1", "thing", "cup", bbox_3d={"min": [0, 0, 0], "max": [1, 1, 1]}, centroid=[0, 0, 0])
    a = Node("r2", "anchor", "table", bbox_3d={"min": [0, 0, 0], "max": [1, 1, 1]}, centroid=[0, 0, 0])
    g = Graph(meta, [room, t, a], [Edge("r1", "r2", "on"), Edge("r1", "floor", "on"),
                                   Edge("r2", "floor", "on"), Edge("r1", "r9", "next_to")])  # fmt: skip
    probs = validate(g)
    assert any("r1: 2 support edges" in p for p in probs)
    assert any("non-thing r2" in p for p in probs)
    assert any("r9" in p for p in probs)
    t.anchor = "r1"
    assert any("anchor r1 is not an anchor" in p for p in validate(g))


def test_ids():
    assert ref_node_id(12) == "r12" and object_id_of("r12") == 12
    assert object_id_of("room") is None and object_id_of("n3") is None
