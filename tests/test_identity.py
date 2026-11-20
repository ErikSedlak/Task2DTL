"""Identity check in association (CHANGE 3 Part N, N3)."""

import copy

import numpy as np

from lsg.change.association import associate, both_gates, identity_gate, old_place_gate
from lsg.change.visibility import NodeVotes
from lsg.config import load_config
from lsg.geom.transforms import apply, invert
from lsg.graph.model import Node, bbox_dict
from lsg.obs import ObsInstance
from lsg.perception.appearance import ColorFeatures
from lsg.perception.identity import IdentityScorer, look_at, square_crop

CFG = load_config()


def test_look_at_projects_the_target_to_the_image_centre_upright():
    T = look_at([3.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    p = apply(invert(T), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert abs(p[0, 0]) < 1e-9 and abs(p[0, 1]) < 1e-9 and p[0, 2] > 0
    assert p[1, 1] < p[0, 1]  # a point above the target is higher in the image (smaller y)


def test_square_crop_keeps_the_mask():
    img = np.zeros((100, 60, 3), np.uint8)
    m = np.zeros((100, 60), bool)
    m[40:50, 20:40] = True
    rgb, mm = square_crop(img, m, 0.1, 56)
    assert rgb.shape == (56, 56, 3) and mm.shape == (56, 56) and 0.1 < mm.mean() < 0.9


class _Seq:
    def __init__(self, colour):
        self.colour = colour

    def color(self, idx):
        img = np.full((54, 96, 3), 90, np.uint8)  # stored orientation, grey background
        img[10:40, 30:70] = self.colour
        return img


class _Renderer:
    def render(self, K, Tw2c, W, H):
        lab = np.zeros((H, W), np.int64)
        lab[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 7
        return np.ones((H, W)), lab


class _Tex:
    def render_color(self, K, Tw2c, W, H):
        img = np.full((H, W, 3), 90, np.uint8)
        img[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = (200, 30, 30)  # a red object
        return img, np.ones((H, W), bool)


def _obs(tok):
    m = np.zeros((17, 22), bool)
    m[3:13, 7:17] = True  # depth-resolution mask over the coloured block
    return ObsInstance(tok, "box", [("box", 1.0)], None, np.zeros((5, 3)), masks={0: m})


def _node():
    return Node("r7", "thing", "box", bbox_3d=bbox_dict([0, 0, 0], [0.3, 0.3, 0.3]), centroid=[0.15, 0.15, 0.15])


def _scorer(colour):
    c = copy.deepcopy(CFG)
    c["association"]["identity"].update(crop_px=56, view_px=56, min_patches=1)
    return IdentityScorer(_Seq(colour), None, _Tex(), _Renderer(), ColorFeatures(), c)


def test_scorer_prefers_the_same_colour_and_logs():
    red, blue = _scorer((200, 30, 30)), _scorer((30, 30, 200))
    s_red, s_blue = red.similarity(_node(), _obs("o1")), blue.similarity(_node(), _obs("o1"))
    assert s_red is not None and s_blue is not None and s_red > s_blue
    assert red.log == [("r7", "o1", s_red)]
    # memoised: a second call does not recompute descriptors
    red.similarity(_node(), _obs("o1"))
    assert len(red._node) == 1 and len(red._obs) == 1


class _Fixed:
    def __init__(self, s):
        self.s, self.log = s, []

    def similarity(self, n, o):
        self.log.append((n.id, o.token, self.s))
        return self.s


def _cfg(tau=0.5, inside_too=False):
    c = copy.deepcopy(CFG)
    c["association"]["identity"].update(enabled=True, tau_id=tau, inside_too=inside_too)
    return c


def _pts(c, h=0.1):
    g = np.linspace(-h, h, 4)
    return np.array([[c[0] + x, c[1] + y, c[2] + z] for x in g for y in g for z in g])


def test_identity_gate_applies_only_to_nodes_that_may_move():
    n = Node("r1", "thing", "chair", bbox_3d=bbox_dict([0, 0, 0], [0.4, 0.4, 0.4]), centroid=[0.2, 0.2, 0.2])
    far = ObsInstance("o1", "chair", [("chair", 1.0)], None, _pts([2.0, 0.2, 0.2]))
    here = ObsInstance("o2", "chair", [("chair", 1.0)], None, _pts([0.2, 0.2, 0.2]))
    low = _Fixed(0.2)
    g = identity_gate({"r1": NodeVotes(free=2)}, low, _cfg())
    assert g(n, far) is False and g(n, here) is True  # inside its own place: not checked
    assert identity_gate({"r1": NodeVotes(free=2)}, low, _cfg(inside_too=True))(n, here) is False
    assert identity_gate({"r1": NodeVotes(free=2)}, _Fixed(0.8), _cfg())(n, far) is True
    assert identity_gate({"r1": NodeVotes(free=2)}, _Fixed(None), _cfg())(n, far) is True  # no information
    assert identity_gate({"r1": NodeVotes(seen=3)}, low, _cfg())(n, far) is True  # net seen: Part A's business
    off = copy.deepcopy(CFG)
    assert identity_gate({}, low, off) is None and identity_gate({}, None, _cfg()) is None
    # through associate: the dissimilar far observation is not taken
    gate = both_gates(old_place_gate({"r1": NodeVotes(free=2)}, CFG), g)
    pairs, un_n, un_o = associate([n], [far], "oracle", CFG, gate=gate)
    assert pairs == [] and [x.token for x in un_o] == ["o1"]


def test_idstats_labels_and_pick(tmp_path):
    import json

    from lsg.eval.identitystats import identity_stats

    d = tmp_path / "details" / "oracle"
    d.mkdir(parents=True)
    rows = [["r1", "o1", 0.9, True], ["r1", "o2", 0.3, True], ["r2", "o2", None, True], ["r2", "o9", 0.1, True],
            ["r1", "o1", 0.9, True]]  # duplicate pair counts once; o9 has no GT instance
    json.dump({"token_truth": {"o1": [1, "chair"], "o2": [5, "chair"]}, "identity": rows}, open(d / "p.json", "w"))
    s = identity_stats(tmp_path, ("oracle",))
    r = s["modes"]["oracle"]
    assert r["n"] == {"same": 1, "different": 2} and r["no_descriptor"]["different"] == 1
    assert s["pick"] == 0.9  # the only same pair has 0.9
    c = s["at_pick"]["oracle"]
    assert c["keep_same"] == 1.0 and c["reject_different"] == 0.5  # 0.3 rejected, the None pair kept
