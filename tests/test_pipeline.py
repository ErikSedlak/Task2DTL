"""Frame selection (§7.3 contiguous chunks) and the refined-alignment cache."""

import copy
from types import SimpleNamespace

import numpy as np

import lsg.change.pipeline as pl
from lsg.config import load_config


def test_select_frames_full_and_chunk():
    ids = list(range(100))
    assert pl.select_frames(ids, 3) == ids[::3]
    a = pl.select_frames(ids, 3, frac=0.5, seed=0, key="S1")
    assert len(a) == 17 and a == pl.select_frames(ids, 3, frac=0.5, seed=0, key="S1")
    assert all(b - x == 3 for x, b in zip(a, a[1:]))  # contiguous in the strided sequence
    assert set(a) <= set(ids[::3])
    starts = {pl.select_frames(ids, 3, frac=0.25, seed=s, key="S1")[0] for s in range(10)}
    assert len(starts) > 1  # the seed moves the chunk
    assert len(pl.select_frames(ids, 3, frac=0.1, seed=0)) == 3
    assert pl.select_frames([], 3, frac=0.5) == []


def test_alignment_cache(tmp_path, monkeypatch):
    calls = []

    def fake_refine(seq, V, F, T, acfg, frame_ids):
        calls.append(list(frame_ids))
        out = np.array(T, float).copy()
        out[0, 3] += 0.01
        return out, {"accepted": True, "delta_t_m": 0.01}

    monkeypatch.setattr(pl, "refine_from_sequence", fake_refine)
    cfg = copy.deepcopy(load_config())
    cfg["cache_root"] = str(tmp_path)
    cfg["alignment"]["online"]["enabled"] = False  # the offline pre-pass (CHANGE 3 Part B: online is the default)
    ref = SimpleNamespace(scan_id="R", vertices=np.zeros((3, 3)), faces=np.zeros((1, 3), int))
    T0 = np.eye(4)
    T1, i1 = pl.refined_alignment(None, ref, "S", T0, cfg, [0, 3, 6])
    T2, i2 = pl.refined_alignment(None, ref, "S", T0, cfg, [0, 3, 6])
    assert len(calls) == 1 and not i1["cached"] and i2["cached"]
    np.testing.assert_array_equal(T1, T2)
    assert T1[0, 3] == 0.01
    pl.refined_alignment(None, ref, "S", T0, cfg, [0, 3])  # other chunk -> new entry
    cfg["alignment"]["voxel"] = 0.03  # other ICP setting -> new entry
    pl.refined_alignment(None, ref, "S", T0, cfg, [0, 3, 6])
    assert len(calls) == 3
    cfg["alignment"]["refine"] = False
    T, info = pl.refined_alignment(None, ref, "S", T0, cfg, [0, 3, 6])
    np.testing.assert_array_equal(T, T0) and info["disabled"]
    assert len(calls) == 3


def test_alignment_cache_key_ignores_the_online_block():
    """CHANGE 3 Part B: the online settings do not change the pre-pass result, so they are not in its key."""
    acfg = copy.deepcopy(load_config())["alignment"]
    k1 = pl._align_key("R", "S", np.eye(4), acfg, [0, 3])
    acfg["online"]["every"] = 7
    acfg["online"]["enabled"] = not acfg["online"]["enabled"]
    assert pl._align_key("R", "S", np.eye(4), acfg, [0, 3]) == k1
    acfg["voxel"] = 0.05
    assert pl._align_key("R", "S", np.eye(4), acfg, [0, 3]) != k1
