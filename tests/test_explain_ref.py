"""Reference explain-away of perception candidates (CHANGE 3 Part N, N2)."""

import copy

import numpy as np

from lsg.change.association import explained_by_reference
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.graph.model import Node, bbox_dict
from lsg.obs import ObsInstance

CFG = load_config()


def plane(z, n=10, x0=0.0, y0=0.0, size=0.4):
    g = np.linspace(0, size, n)
    X, Y = np.meshgrid(x0 + g, y0 + g)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def node(nid, P):
    return Node(nid, "thing", "table", bbox_3d=bbox_dict(P.min(0), P.max(0)), centroid=list(P.mean(0)))


def cand(tok, P, kind="depth"):
    return ObsInstance(tok, "chair", [("chair", 1.0)], None, P, kind=kind)


def test_candidate_close_to_a_standing_node_is_explained():
    top = plane(0.7)
    n = node("r1", top)
    near = cand("c0", plane(0.77))  # 7 cm above the table top: an offset re-detection
    far = cand("c1", plane(0.95))  # 25 cm above: something standing on the table
    app = cand("c2", plane(0.77), kind="appearance")  # thin-object candidates are never tested
    out = explained_by_reference([near, far, app], [n], {"r1": top}, {"r1": NodeVotes(seen=3)}, set(), CFG)
    assert out == {"c0": "r1"}
    # a node with free evidence at its place (it left) or committed removed explains nothing
    assert explained_by_reference([near], [n], {"r1": top}, {"r1": NodeVotes(free=2)}, set(), CFG) == {}
    assert explained_by_reference([near], [n], {"r1": top}, {"r1": NodeVotes(seen=3)}, {"r1"}, CFG) == {}
    # min_frac: a candidate only partly near the node stays new
    half = cand("c3", np.concatenate([plane(0.77), plane(1.2), plane(1.3)]))
    assert explained_by_reference([half], [n], {"r1": top}, {}, set(), CFG) == {}
    c = copy.deepcopy(CFG)
    c["association"]["explain_ref"]["dist"] = 0.30
    assert explained_by_reference([far], [n], {"r1": top}, {}, set(), c) == {"c1": "r1"}


def test_explain_ref_blocks_emission_only_in_full_perception():
    """Through `full`: with the switch on, the explained candidate is not a new node; oracle unaffected."""
    from lsg.change.baselines import full
    from lsg.graph.model import ROOM_ID, Graph
    from lsg.obs import RefContext
    from lsg.io.scan3r import Instance, Scan

    top = plane(0.7)
    scan = Scan("R", top, np.zeros((0, 3), np.int64), np.full(len(top), 7), np.full(len(top), 7),
                np.zeros(0, np.int64), np.zeros(0, np.int64),
                {7: Instance(7, "table", 7, top.mean(0), top.min(0), top.max(0), len(top))}, None)  # fmt: skip
    G = Graph(Graph.new_meta("R"), [Node(ROOM_ID, "room", "room"), node("r7", top)], [])
    ctx = RefContext("R", scan, G, 0.0)
    obs = [cand("c0", plane(0.77))]
    votes = {"r7": NodeVotes(seen=1)}
    c = copy.deepcopy(CFG)
    c["association"]["tau_assoc"] = {"oracle": 0.0, "perception": 0.0}  # nothing matches: c0 is left over
    off = full(G, obs, c, "S", "perception", votes, ctx)
    assert [n.obs_token for n in off.nodes if n.state == "new"] == ["c0"]
    c["association"]["explain_ref"]["enabled"] = True
    on = full(G, obs, c, "S", "perception", votes, ctx)
    assert not [n for n in on.nodes if n.state == "new"] and on.meta["assoc"]["explained_ref"] == 1
    orc = full(G, obs, c, "S", "oracle", votes, ctx)
    assert [n.obs_token for n in orc.nodes if n.state == "new"] == ["c0"]
