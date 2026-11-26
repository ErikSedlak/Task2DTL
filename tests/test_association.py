"""Association (M5): label terms, Hungarian with gate, confirmed-node absorption."""

import copy
import math

import numpy as np

from conftest import box_points
from lsg.change.association import (
    absorb_confirmed,
    associate,
    inside_frac,
    label_cost,
    pair_cost,
    size_cost,
)
from lsg.change.baselines import greedy_label_match
from lsg.config import load_config
from lsg.obs import ObsInstance
from test_b0 import obs, ref_node

CFG = load_config()


def pobs(token, topk, c, h=0.2, nyu=None):
    """Perception-like observation with a CLIP top-k."""
    c = np.asarray(c, float)
    return ObsInstance(token, topk[0][0], list(topk), nyu, box_points(c - h, c + h, 0.1))


def test_oracle_label_gate_and_nyu_fallback():
    n = ref_node("r1", "chair", (0, 0, 0), nyu=5)
    assert label_cost(n, obs("o0", "chair", (0, 0, 0)), "oracle", CFG) == 0.0
    assert label_cost(n, obs("o1", "armchair", (0, 0, 0), nyu=5), "oracle", CFG) == CFG["association"]["nyu_penalty"]
    assert label_cost(n, obs("o2", "table", (0, 0, 0), nyu=7), "oracle", CFG) == math.inf
    assert pair_cost(n, obs("o2", "table", (0, 0, 0), nyu=7), "oracle", CFG) == math.inf


def test_perception_soft_label_term():
    n = ref_node("r1", "pillow", (0, 0, 0))
    o = pobs("c0", [("cushion", 0.6), ("pillow", 0.3)], (0, 0, 0))
    assert math.isclose(label_cost(n, o, "perception", CFG), -math.log(0.3))
    o2 = pobs("c1", [("lamp", 0.9)], (0, 0, 0))  # pillow not in top-k -> p_min
    assert math.isclose(label_cost(n, o2, "perception", CFG), -math.log(CFG["association"]["p_min"]))


def test_size_term_zero_for_equal_boxes_and_grows():
    n = ref_node("r1", "chair", (0, 0, 0), h=0.2)
    assert size_cost(n, obs("o0", "chair", (1, 0, 0), h=0.2), CFG) < 1e-9
    assert size_cost(n, obs("o1", "chair", (1, 0, 0), h=0.5), CFG) > 0.5


def test_hungarian_beats_greedy():
    # r1 at 0, r2 at 1; o0 at 0.6, o1 at 1.6. Greedy takes the shortest pair first (r2-o0,
    # 0.4 m) and is left with r1-o1 (1.6 m): total 2.0 m. Optimal: r1-o0 + r2-o1 = 1.2 m.
    nodes = [ref_node("r1", "chair", (0, 0, 0)), ref_node("r2", "chair", (1.0, 0, 0))]
    O = [obs("o0", "chair", (0.6, 0, 0)), obs("o1", "chair", (1.6, 0, 0))]
    greedy = {n.id: o.token for n, o, _ in greedy_label_match(nodes, O)}
    assert greedy == {"r2": "o0", "r1": "o1"}
    pairs, un, uo = associate(nodes, O, "oracle", CFG)
    assert {n.id: o.token for n, o, _ in pairs} == {"r1": "o0", "r2": "o1"} and not un and not uo


def test_gate_leaves_bad_pairs_unmatched():
    cfg = copy.deepcopy(CFG)
    cfg["association"]["tau_assoc"]["perception"] = 1.0
    n = ref_node("r1", "pillow", (0, 0, 0))
    o = pobs("c0", [("lamp", 0.9)], (0, 0, 0))  # label cost -log(0.05) = 3 >= 1
    pairs, un, uo = associate([n], [o], "perception", cfg)
    assert pairs == [] and [x.id for x in un] == ["r1"] and [x.token for x in uo] == ["c0"]


def test_hungarian_global_optimum_with_competition():
    # one obs, two same-label nodes: the cheaper (closer) node gets it, the other is unmatched
    nodes = [ref_node("r1", "chair", (0, 0, 0)), ref_node("r2", "chair", (2.0, 0, 0))]
    pairs, un, _ = associate(nodes, [obs("o0", "chair", (1.8, 0, 0))], "oracle", CFG)
    assert [(n.id, o.token) for n, o, _ in pairs] == [("r2", "o0")] and [n.id for n in un] == ["r1"]


def test_associate_deterministic_under_input_order():
    nodes = [ref_node(f"r{i}", "chair", (i * 0.7, 0, 0)) for i in range(5)]
    O = [obs(f"o{i}", "chair", (i * 0.7 + 0.05, 0.1, 0)) for i in range(5)]
    a = associate(nodes, O, "oracle", CFG)[0]
    b = associate(nodes[::-1], O[::-1], "oracle", CFG)[0]
    assert [(n.id, o.token) for n, o, _ in a] == [(n.id, o.token) for n, o, _ in b]


def test_absorption_oracle_own_instance():
    n = ref_node("r1", "chair", (0, 0, 0), h=0.3)
    own = obs("o0", "chair", (0.02, 0, 0), h=0.25)
    other = obs("o1", "chair", (2, 0, 0))
    ab = absorb_confirmed([n], [own, other], "oracle", CFG)
    assert ab.token_of == {"r1": "o0"} and ab.absorbed == {"o0": "r1"}


def test_absorption_perception_junk_many_to_one_and_label_rule():
    tv = ref_node("r1", "tv", (0, 0, 0), h=0.4)
    hole1 = pobs("c0", [("tv", 0.5), ("monitor", 0.3)], (0.1, 0, 0), h=0.1)
    hole2 = pobs("c1", [("monitor", 0.6), ("tv", 0.2)], (-0.1, 0, 0.1), h=0.1)
    cup = pobs("c2", [("cup", 0.9)], (0, 0, 0.2), h=0.05)  # new object inside the AABB, other label
    ab = absorb_confirmed([tv], [hole1, hole2, cup], "perception", CFG)
    assert set(ab.absorbed) == {"c0", "c1"} and ab.token_of["r1"] in {"c0", "c1"}
    assert "c2" not in ab.absorbed


def test_absorption_needs_points_inside_aabb():
    n = ref_node("r1", "chair", (0, 0, 0), h=0.2)
    o = obs("o0", "chair", (0.5, 0, 0), h=0.2)  # mostly outside AABB + 10 cm
    assert inside_frac(n, o, 0.10) < 0.5
    assert absorb_confirmed([n], [o], "oracle", CFG).absorbed == {}


def test_nyu_fallback_excludes_catch_all_classes():
    # nyu40 40 (otherprop) / 39 (otherfurniture) are catch-alls: keyboard vs blanket must not pair
    n = ref_node("r1", "blanket", (0, 0, 0), nyu=40)
    assert label_cost(n, obs("o0", "keyboard", (0, 0, 0), nyu=40), "oracle", CFG) == math.inf
    n39 = ref_node("r2", "tv stand", (0, 0, 0), nyu=39)
    assert label_cost(n39, obs("o1", "bench", (0, 0, 0), nyu=39), "oracle", CFG) == math.inf
    cfg = copy.deepcopy(CFG)
    cfg["association"]["nyu_fallback_exclude"] = []
    assert label_cost(n, obs("o0", "keyboard", (0, 0, 0), nyu=40), "oracle", cfg) == cfg["association"]["nyu_penalty"]
