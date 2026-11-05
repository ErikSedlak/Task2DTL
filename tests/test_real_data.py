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
