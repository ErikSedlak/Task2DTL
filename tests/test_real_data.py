"""Smoke tests on local 3RScan data. Slow; run with `uv run pytest -q -m slow`."""

import numpy as np
import pytest

from lsg.geom.render import Renderer
from lsg.geom.transforms import invert
from lsg.io.scan3r import load_meta, load_scan, open_sequence

RESCAN = "0ad2d3a1-79e2-2212-9b99-a96495d9f7fe"


@pytest.mark.slow
def test_render_own_mesh_matches_depth(real_data_root):
    meta = load_meta(real_data_root)
    assert not meta[RESCAN].is_reference
    S = load_scan(real_data_root, RESCAN)
    assert S.intrinsics is not None and S.intrinsics.depth_wh == (224, 172)
    r = Renderer(S.vertices, S.faces, S.face_labels)
    with open_sequence(real_data_root, RESCAN) as seq:
        f = seq.frame(seq.frame_ids[len(seq.frame_ids) // 2])
        W, H = seq.info.depth_wh
        d, lab = r.render(seq.info.K_depth, invert(f.pose_c2w), W, H)
    v = (f.depth > 0) & (d > 0)
    assert v.mean() > 0.5
    assert np.median(np.abs(f.depth[v] - d[v])) < 0.03
    assert len(np.unique(lab[lab > 0])) >= 2


@pytest.mark.slow
def test_val_gt_counts_reproduce_design(real_data_root):
    """DESIGN §2.2 / M1 acceptance (1): 390 rigid / 96 removed / 224 new raw (171 scored)."""
    from collections import Counter

    from lsg.config import REPO_ROOT, load_config
    from lsg.eval.gt import gt_changes
    from lsg.io.dssg import load_objects
    from lsg.io.scan3r import read_pairs

    cfg = load_config()
    meta = load_meta(real_data_root)
    c: Counter = Counter()
    refs: dict = {}
    for r, s in read_pairs(REPO_ROOT / "splits/eval_val.txt"):
        if r not in refs:
            refs.clear()
            refs[r] = (load_scan(real_data_root, r, with_intrinsics=False), load_objects(real_data_root, r))
        R, oR = refs[r]
        S = load_scan(real_data_root, s, with_intrinsics=False)
        c.update(gt_changes(meta[s], R, S, oR, load_objects(real_data_root, s), cfg).counts())
    assert (c["rigid_raw"], c["removed_raw"], c["new_raw"], c["new"]) == (390, 96, 224, 171)
    assert c["dont_care_small_move"] == 134


@pytest.mark.slow
def test_m2_oracle_b0_one_dev_pair_deterministic(real_data_root, tmp_path):
    """M2 end-to-end on one dev pair: oracle + B0, scored; two runs give identical `metrics`."""
    import copy
    import json

    from lsg.config import load_config
    from lsg.eval.harness import run_eval

    cfg = copy.deepcopy(load_config())
    cfg["outputs_root"] = str(tmp_path)
    pair = ("0ad2d39b-79e2-2212-99ae-830c292cd079", RESCAN)
    a = run_eval(cfg, [pair], ["oracle"], "B0", workers=1, run_id="a")
    b = run_eval(cfg, [pair], ["oracle"], "B0", workers=1, run_id="b")
    ma = json.loads((tmp_path / "a" / "metrics.json").read_text())
    mb = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert ma["metrics"] == mb["metrics"]
    assert ma["config_hash"] == mb["config_hash"] and "run" in ma
    s = a["summary"]["oracle"]
    assert s["n_pairs"] == 1 and s["counts"]["ref_n"] > 10
    for k in ("moved_F1", "removed_P", "new_R", "false_deletion_rate", "false_move_rate",
              "unchanged_kept", "assoc_acc_strict", "moved_R_small"):
        assert k in s["metrics"]
    assert (tmp_path / "a" / "graphs" / "oracle" / f"{pair[0]}_{pair[1]}.json").is_file()
    assert b["per_pair"][0]["n_obs"] == a["per_pair"][0]["n_obs"] > 5
