"""CHANGE 3 Part E: detection cache helpers and detector tracks (stub detections, no models)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import INTR, K, T_ALIGN, FakeSeq, H, W, box, look_at, make_scan  # noqa: E402
from test_perception import NEW_BOX, graph_of, orbit, room  # noqa: E402

import lsg.perception.detect_cache as dc
from lsg.change.baselines import emits_new
from lsg.change.online import run_online
from lsg.config import load_config
from lsg.geom.render import Renderer
from lsg.geom.transforms import invert
from lsg.obs import ObsInstance, RefContext, SeqInfo
from lsg.perception.source import PerceptionSource

PROMPTS = ["basket", "box"]


def test_prompt_helpers_and_frames():
    ids = {"bedside table": [1, 2], "lamp": [3], "box": [4]}
    used, spans = dc.fit_prompts(["bedside table", "lamp", "box"], lambda t: ids[t], max_tokens=8)
    assert used == ["bedside table", "lamp"] and spans == [(1, 3), (4, 5)]  # [CLS] bed side . lamp . [SEP]
    assert dc.ordered_prompts(["lamp", "zebra"], ["box", "chair", "lamp"], ["box", "chair", "lamp", "vase"]) == \
        ["lamp", "box", "chair", "vase"]  # reference first, then by frequency, then the rest; never out of vocab
    assert dc.detect_frames(list(range(20)), 3, 2) == [0, 6, 12, 18]
    probs = np.zeros((2, 8))
    probs[0, 1:3] = [0.8, 0.6]
    probs[1, 4] = 0.5
    np.testing.assert_allclose(dc.label_scores(probs, spans), [[0.7, 0.0], [0.0, 0.5]])
    B = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert dc.nms(B, np.array([0.5, 0.9, 0.4]), 0.7).tolist() == [1, 2]


def test_cache_round_trip(tmp_path):
    m = np.zeros((3, H, W), bool)
    m[0, 5:9, 7:20] = True
    m[2, -3:, :4] = True
    det = {"boxes": np.ones((3, 4), np.float32), "scores": np.array([0.5, 0.3, 0.9], np.float32),
           "labels": np.array([1, 0, 1], np.int16), "label_scores": np.random.rand(3, 2).astype(np.float16), "masks": m}
    dc.save_frame(tmp_path / "000006.npz", det)
    back = dc.load_frame(tmp_path / "000006.npz")
    assert np.array_equal(back["masks"], m) and np.array_equal(back["labels"], det["labels"])
    np.testing.assert_allclose(back["label_scores"], det["label_scores"].astype(np.float32))


class StubCache:
    """Detections = the rescan's own instances rendered at each frame (perfect detector), with fixed labels."""

    def __init__(self, S, seq, oids_labels, every=1):
        self.prompts = list(PROMPTS)
        r = Renderer(S.vertices, S.faces, S.face_labels)
        self.frames = {}
        for k, f in enumerate(seq.frame_ids):
            if k % every:
                continue
            fr = seq.frame(f)
            _, lab = r.render(K, invert(fr.pose_c2w), W, H)
            ms, ls = [], []
            for oid, name in oids_labels.items():
                m = lab == oid
                if m.sum() >= 20:
                    ms.append(m)
                    v = np.zeros(len(PROMPTS), np.float32)
                    v[PROMPTS.index(name)] = 0.6
                    ls.append(v)
            n = len(ms)
            self.frames[f] = {"boxes": np.zeros((n, 4), np.float32), "scores": np.full(n, 0.6, np.float32),
                              "labels": np.array([int(np.argmax(v)) for v in ls], np.int16),
                              "label_scores": np.array(ls, np.float32).reshape(n, len(PROMPTS)),
                              "masks": np.array(ms, bool).reshape(n, H, W)}  # fmt: skip

    def frame(self, fidx):
        return self.frames.get(int(fidx))

    def n_frames(self):
        return len(self.frames)

    def complete(self):
        return True


def _dcfg(mode="replace", **kw):
    c = copy.deepcopy(load_config())
    c["perception"]["detector"].update(enabled=True, mode=mode, min_score=0.3, **kw)
    return c


def run_det(cfg, poses, monkeypatch, every=1):
    R = make_scan(room(), "REF")
    S = make_scan(room({20: (*box(*NEW_BOX), 29, "box")}), "S", T=invert(T_ALIGN))
    seq = FakeSeq(S, poses)
    stub = StubCache(S, seq, {20: "box", 11: "basket"}, every)
    monkeypatch.setattr(dc, "DetectionCache", lambda *a, **k: stub)
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    info = SeqInfo("S", Path("."), INTR, T_ALIGN.copy(), list(seq.frame_ids))
    src = PerceptionSource(None, {"box": 29, "basket": 40}, cfg, with_crops=False, det_prompts=PROMPTS)
    res = run_online(ctx, info, seq, src, cfg, obs_mode="perception")
    return res, src


def test_tracks_fuse_frames_and_newness_separates_new_from_reference(monkeypatch):
    cfg = _dcfg(p_new_min=0.5)
    poses = orbit([1.2, 1.2, 0.15], n=12, radius=2.2, height=1.0, target_z=0.2)
    res, src = run_det(cfg, poses, monkeypatch)
    det = [o for o in res.obs if o.kind == "detector"]
    assert [o.kind for o in res.obs].count("depth") == 0  # replace: no residual candidates
    by = {o.label: o for o in det}
    assert set(by) == {"box", "basket"} and len(det) == 2, [(o.token, o.label, len(o.points)) for o in det]
    assert by["box"].newness > 0.8 and by["basket"].newness < 0.2
    assert emits_new(by["box"], cfg) and not emits_new(by["basket"], cfg)
    assert by["box"].confidence is None and by["box"].label_topk[0] == ("box", 1.0)
    lo, hi = np.array(NEW_BOX[0]), np.array(NEW_BOX[1])
    assert np.linalg.norm(by["box"].centroid[:2] - (lo + hi)[:2] / 2) < 0.05
    assert src.stats["det_frames"] == 12 and src.stats["det_instances"] == 2
    # snapshot tokens are the finish tokens (track ids)
    assert sorted(o.token for o in src.peek() if o.kind == "detector") == sorted(o.token for o in det)


def test_views_are_decorrelated(monkeypatch):
    cfg = _dcfg(min_views=2)
    pose = look_at([2.5, 2.5, 1.2], [0.5, 0.5, 0.15])
    res, src = run_det(cfg, [pose] * 8, monkeypatch)  # one viewpoint, 8 frames: one view per track
    assert not [o for o in res.obs if o.kind == "detector"]
    res, _ = run_det(_dcfg(min_views=1), [pose] * 8, monkeypatch)
    assert len([o for o in res.obs if o.kind == "detector"]) >= 1


def test_union_drops_residual_candidates_on_detections(monkeypatch):
    poses = orbit([0.5, 0.5, 0.15], n=12, radius=1.5, height=0.5)
    res, src = run_det(_dcfg(mode="union"), poses, monkeypatch)
    kinds = [o.kind for o in res.obs]
    assert "detector" in kinds and "depth" not in kinds, (kinds, src.stats)
    assert src.stats["det_union_dropped"] >= 1


def test_emits_new_per_kind():
    cfg = _dcfg(p_new_min=0.5)
    o = ObsInstance("d0", "box", [("box", 1.0)], 29, np.zeros((3, 3)), kind="detector", newness=0.4)
    assert not emits_new(o, cfg)
    o.newness = 0.6
    assert emits_new(o, cfg)
    c = ObsInstance("c0", "box", [("box", 1.0)], 29, np.zeros((3, 3)), confidence=0.1)  # residual: candidate rule
    assert emits_new(c, cfg) is (0.1 >= cfg["confidence"]["candidates"]["p_new_min"])


def test_missing_cache_is_an_error(tmp_path):
    cfg = _dcfg()
    cfg["cache_root"] = str(tmp_path)
    R = make_scan(room(), "REF")
    ctx = RefContext("REF", R, graph_of(R), 0.0, {oid: inst.nyu40 for oid, inst in R.instances.items()})
    src = PerceptionSource(None, {}, cfg, with_crops=False, det_prompts=PROMPTS)
    with pytest.raises(FileNotFoundError):
        src.begin(ctx, SeqInfo("S", Path("."), INTR, T_ALIGN.copy(), [0]))


def test_part_e_grid_and_selection_rules():
    from lsg.eval.ablations import apply_overrides
    from lsg.eval.chg3_checks import part_e_grid, select_part_e, select_part_e_step1

    cfg = load_config()
    grid = part_e_grid(cfg["chg3"]["part_e"])
    assert len(grid) == 8 and all(apply_overrides(cfg, ov) for ov in grid.values())  # every key exists
    curves = {"a": {"curve": [{"p_new_min": 0.0, "f1": 0.10, "precision": 0.1, "recall": 0.1, "ag_f1": 0.2},
                              {"p_new_min": 0.5, "f1": 0.20, "precision": 0.3, "recall": 0.15, "ag_f1": 0.3}]},
              "b": {"curve": [{"p_new_min": 0.3, "f1": 0.20, "precision": 0.2, "recall": 0.2, "ag_f1": 0.3}]}}
    assert select_part_e_step1(curves)["variant"] == "a"  # tie on F1: the earlier variant
    base = {"moved_F1": 0.45, "removed_F1": 0.40, "unchanged_kept": 0.90, "false_deletion_rate": 0.05, "new_F1": 0.05}
    rows = {"residual": {"perception": base, "oracle": {"x": 1}},
            "det": {"perception": {**base, "new_F1": 0.30, "moved_F1": 0.40}, "oracle": {"x": 1}},  # fails moved guard
            "det_new": {"perception": {**base, "new_F1": 0.20}, "oracle": {"x": 1}},
            "det_union": {"perception": {**base, "new_F1": 0.204}, "oracle": {"x": 2}}}  # tie with det_new
    sel = select_part_e(rows, {"tie": 0.005, "f1_drop": 0.02, "rate_drop": 0.01}, list(rows))
    assert not sel["verdict"]["det"]["pass"] and sel["winner"] == "det_new" and sel["oracle_differs"] == ["det_union"]
