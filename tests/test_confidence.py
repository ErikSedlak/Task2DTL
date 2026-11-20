"""Log-odds confidence with class priors and FULL's removal rule (DESIGN §6.11, M6)."""

import copy

import numpy as np
import pytest

from lsg.change.baselines import b2, full
from lsg.change.confidence import (
    below_p_del,
    new_logodds,
    node_logodds,
    prior_logodds,
    probability,
)
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.graph.model import ROOM_ID, Edge, Graph, Node, bbox_dict, validate
from lsg.io.scan3r import Meta, ScanMeta
from lsg.obs.base import ObsInstance
from lsg.train_meta import PriorTable, logit, prior_counts


@pytest.fixture
def cfg():
    """The M7 confidence (class priors, flat fallback 0.9, p_del 0.2) = ablation row FULL+classprior."""
    c = copy.deepcopy(load_config())
    c["confidence"].update(class_priors=True, prior_p=0.9, p_del=0.2)
    return c


@pytest.fixture
def cfg_a():
    """The default: no class prior (CHANGE 2 Part A), flat 0.9 with p_del 0.2 (CHANGE 3)."""
    return copy.deepcopy(load_config())


def table(**labels):
    """PriorTable from {label: (present, removed)}, all things of nyu40 40 unless named 'bed'."""
    by_l = {k: v for k, v in labels.items()}
    by_n: dict[int, list[int]] = {}
    for k, (p, r) in by_l.items():
        n = 4 if k == "bed" else 40
        by_n.setdefault(n, [0, 0])
        by_n[n][0] += p
        by_n[n][1] += r
    tot = (sum(p for p, _ in by_l.values()), sum(r for _, r in by_l.values()))
    return PriorTable(by_l, {k: (v[0], v[1]) for k, v in by_n.items()}, tot, 20.0)


def test_prior_smoothing_and_backoff():
    t = table(chair=(1047, 35), bed=(138, 0), rare=(1, 1))
    pg = t.p_global
    assert pg == pytest.approx(1186 / 1222)
    assert t.p_persist("chair", 40) == pytest.approx((1047 + 20 * t.p_nyu(40)) / (1082 + 20))
    assert 0.99 < t.p_persist("bed", 4) < 1.0  # 0 removed of 138 -> smoothed below 1
    assert t.p_persist("rare", 40) > 0.85  # 1 / 2 is pulled toward the nyu40 rate
    assert t.p_persist("unseen label", 40) == t.p_nyu(40)
    assert t.p_persist("unseen label", 99) == pg == t.p_nyu(None)


def test_prior_counts_from_metadata():
    meta = Meta({
        "R": ScanMeta("R", "R", True, "train", np.eye(4)),
        "S1": ScanMeta("S1", "R", False, "train", np.eye(4), removed=[2]),
        "S2": ScanMeta("S2", "R", False, "train", np.eye(4)),
    })  # fmt: skip
    objs = {"R": {1: {"label": "chair", "nyu40": 5}, 2: {"label": "chair", "nyu40": 5},
                  3: {"label": "wall", "nyu40": 1}, 4: {"label": "pillow", "nyu40": 18}},
            "S1": {1: {"label": "chair", "nyu40": 5}, 4: {"label": "pillow", "nyu40": 18}},
            "S2": {1: {"label": "chair", "nyu40": 5}}}  # fmt: skip
    t = prior_counts(meta, ["R", "S1", "S2"], lambda s: objs[s])
    # S1: chair 1 present, chair 2 removed, pillow present; S2: chair 1 present, chair 2 + pillow absent-unlisted
    assert t.by_label == {"chair": (2, 1), "pillow": (1, 0)} and t.total == (3, 1)
    assert "wall" not in t.by_label and t.by_nyu[5] == (2, 1)


def test_worked_examples_from_the_spec_decision(cfg):
    """DECISIONS M6: chair 2 free -> removed; bed 2 free kept, 3 removed; mixed 7 free / 12 seen kept."""
    chair = logit(0.968)
    bed = logit(0.996)
    assert probability(node_logodds(chair, NodeVotes(free=2), cfg)) == pytest.approx(0.166, abs=0.01)
    assert below_p_del(node_logodds(chair, NodeVotes(free=2), cfg), cfg)
    assert not below_p_del(node_logodds(bed, NodeVotes(free=2), cfg), cfg)
    assert below_p_del(node_logodds(bed, NodeVotes(free=3), cfg), cfg)
    cfg["confidence"]["L_hit"] = 2.5  # with L_hit = L_miss the mixed node is kept
    assert not below_p_del(node_logodds(chair, NodeVotes(free=7, seen=12), cfg), cfg)


def test_clip_and_matched_and_new(cfg):
    l0 = logit(0.9)
    assert node_logodds(l0, NodeVotes(free=100), cfg) == pytest.approx(l0 - cfg["confidence"]["clip"])
    # matched: free votes at the old place are ignored, the observation's views count
    assert node_logodds(l0, NodeVotes(free=5), cfg, matched=True, views=2) == pytest.approx(l0 + 2 * cfg["confidence"]["L_hit"])
    assert new_logodds(0, cfg) == 0.0 and probability(new_logodds(3, cfg)) > 0.9
    assert 0 <= probability(-1e3) <= probability(1e3) <= 1


def test_priors_off_uses_fallback(cfg):
    t = table(chair=(10, 10))
    assert prior_logodds("chair", 40, t, cfg) < logit(0.9)
    cfg["confidence"]["class_priors"] = False
    assert prior_logodds("chair", 40, t, cfg) == pytest.approx(logit(0.9))
    assert prior_logodds("chair", 40, None, cfg) == pytest.approx(logit(0.9))


def test_part_a_default_has_no_prior_and_one_margin_for_every_class(cfg_a):
    """No class prior: one l0 for chair and bed alike; 2 net free votes delete, 1 cannot, mixed protects."""
    cc = cfg_a["confidence"]
    assert cc["class_priors"] is False and cc["prior_p"] == 0.9 and cc["p_del"] == 0.2
    t = table(chair=(968, 32), bed=(996, 4))
    l_chair = prior_logodds("chair", 40, t, cfg_a)
    l_bed = prior_logodds("bed", 4, t, cfg_a)
    assert l_chair == l_bed == pytest.approx(logit(0.9))
    for l0 in (l_chair, l_bed):
        assert below_p_del(node_logodds(l0, NodeVotes(free=2), cfg_a), cfg_a)
        assert not below_p_del(node_logodds(l0, NodeVotes(free=1), cfg_a), cfg_a)
        assert not below_p_del(node_logodds(l0, NodeVotes(free=2, seen=1), cfg_a), cfg_a)
        assert below_p_del(node_logodds(l0, NodeVotes(free=3, seen=1), cfg_a), cfg_a)
    assert probability(node_logodds(l_chair, None, cfg_a)) == pytest.approx(0.9)  # no votes: the flat start value


def test_default_decisions_equal_change2_no_prior(cfg_a):
    """CHANGE 3's flat 0.9 / 0.2 deletes exactly where CHANGE 2's l0 = 0 / p_del 0.027 did (every vote combination)."""
    m7 = copy.deepcopy(cfg_a)
    m7["confidence"].update(class_priors=False, prior_p=0.5, p_del=0.027)
    l_a, l_m = prior_logodds("x", None, None, cfg_a), prior_logodds("x", None, None, m7)
    for seen in range(6):
        for free in range(6):
            for fl in range(3):
                v = NodeVotes(seen=seen, free=free, free_like=fl)
                assert below_p_del(node_logodds(l_a, v, cfg_a), cfg_a) == below_p_del(node_logodds(l_m, v, m7), m7)


def test_new_node_takes_candidate_confidence(cfg_a):
    """CHANGE 2 Part B: a candidate's own confidence becomes the new node's; oracle obs keep views."""
    G = room()
    cand = ObsInstance("c0", "box", [("box", 1.0)], 29, cube([6, 6, 0.2]), masks={}, confidence=0.97)
    orc = ObsInstance("o1", "box", [("box", 1.0)], 29, cube([9, 9, 0.2]), masks={})
    g = full(G, [cand, orc], cfg_a, "S", "perception", votes={}, obs_views={"c0": 1, "o1": 3})
    conf = {n.obs_token: n.confidence for n in g.nodes if n.state == "new"}
    assert conf["c0"] == pytest.approx(0.97, abs=1e-6)
    assert conf["o1"] == pytest.approx(probability(3 * cfg_a["confidence"]["L_hit"]))


def test_low_confidence_candidates_are_not_emitted(cfg_a):
    """Candidates below p_new_min are dropped from the new nodes (Erik: drop); p_new_min 0 emits all."""
    G = room()
    hi = ObsInstance("c0", "box", [("box", 1.0)], 29, cube([6, 6, 0.2]), masks={}, confidence=0.99)
    lo = ObsInstance("c1", "box", [("box", 1.0)], 29, cube([9, 9, 0.2]), masks={}, confidence=0.6)
    for method in (b2, full):
        g = method(G, [hi, lo], cfg_a, "S", "perception", votes={})
        assert sorted(n.obs_token for n in g.nodes if n.state == "new") == ["c0"]
        assert [n.id for n in g.nodes if n.state == "new"] == ["n0"] and validate(g) == []
    c0 = copy.deepcopy(cfg_a)
    c0["confidence"]["candidates"]["p_new_min"] = 0.0
    assert len([n for n in full(G, [hi, lo], c0, "S", "perception", votes={}).nodes if n.state == "new"]) == 2
    c0["confidence"]["candidates"].update(p_new_min=0.92, enabled=False)
    assert len([n for n in b2(G, [hi, lo], c0, "S", "perception", votes={}).nodes if n.state == "new"]) == 2


# ---- FULL vs B2 on a point-box room ------------------------------------------------------------


def cube(c, e=0.4, n=6):
    g = np.linspace(-e / 2, e / 2, n)
    return np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) + np.asarray(c, float)


def room():
    nodes = [Node(ROOM_ID, "room", "room")]
    for k, (lab, nyu, c) in enumerate((("chair", 5, [0, 0, 0.2]), ("bed", 4, [3, 0, 0.2]),
                                        ("item", 40, [0, 3, 0.2]), ("lamp", 35, [3, 3, 0.2]))):  # fmt: skip
        P = cube(c)
        nodes.append(Node(f"r{k + 1}", "anchor" if lab == "bed" else "thing", lab, nyu,
                          bbox_3d=bbox_dict(P.min(0), P.max(0)), centroid=list(P.mean(0))))  # fmt: skip
    return Graph(Graph.new_meta("R"), nodes, [Edge(ROOM_ID, n.id, "hierarchy") for n in nodes[1:]])


def test_full_vetoes_mixed_evidence_and_sets_confidence(cfg):
    G = room()
    priors = table(chair=(968, 32), bed=(996, 4), item=(908, 92), lamp=(990, 10))
    votes = {
        "r1": NodeVotes(free=2),  # chair: 2 free, nothing else -> removed by both
        "r2": NodeVotes(free=2),  # bed: 2 free is not enough for its prior -> unknown in FULL
        "r3": NodeVotes(free=3, seen=12),  # mixed evidence -> B2 removes, FULL keeps
        "r4": NodeVotes(seen=4),  # confirmed in place
    }
    new = ObsInstance("o1", "box", [("box", 1.0)], 29, cube([6, 6, 0.2]), masks={})
    g2 = b2(G, [new], cfg, "S", "oracle", votes=votes)
    gf = full(G, [new], cfg, "S", "oracle", votes=votes, obs_views={"o1": 4}, priors=priors)
    s2 = {n.id: n.state for n in g2.nodes}
    sf = {n.id: n.state for n in gf.nodes}
    assert (s2["r1"], s2["r2"], s2["r3"], s2["r4"]) == ("removed", "removed", "removed", "unchanged")
    assert (sf["r1"], sf["r2"], sf["r3"], sf["r4"]) == ("removed", "unknown", "unknown", "unchanged")
    conf = {n.id: n.confidence for n in gf.nodes}
    assert conf["r1"] < 0.2 <= conf["r2"] and conf["r4"] > 0.99 and conf["n0"] > 0.9
    assert conf[ROOM_ID] is None and all(n.confidence is None for n in g2.nodes)
    assert gf.meta["method"] == "FULL" and gf.meta["assoc"]["conf_veto"] == 2
    assert validate(gf) == []
