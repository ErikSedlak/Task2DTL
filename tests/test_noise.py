"""Thin-object candidate confidence (CHANGE 3 Part N, N5; the noise filters of N5 lost and were removed)."""

import copy

from lsg.config import load_config


def test_thin_object_candidates_get_a_confidence_when_switched_on():
    import test_appearance as ta
    from lsg.change.online import run_online
    from lsg.perception.labeler import StubLabeler
    from lsg.perception.source import PerceptionSource

    c = copy.deepcopy(load_config())  # the colour-feature setup of test_appearance's cfg fixture
    c["appearance"]["features"] = "color"
    c["appearance"]["tau_app"]["color"] = 0.15
    c["appearance"]["tau_node"]["color"] = 0.15
    out = {}
    for flag in (False, True):
        cc = copy.deepcopy(c)
        cc["confidence"]["candidates"]["track"]["appearance"] = flag
        ctx, info, seq, runner = ta.setup(cc, ref_book=False, rescan_book=True)
        src = PerceptionSource(StubLabeler((("book", 1.0),)), {"book": 23}, cc, with_crops=False)
        res = run_online(ctx, info, seq, src, cc, obs_mode="perception", appearance=runner)
        assert len(res.obs) == 1 and res.obs[0].kind == "appearance"
        out[flag] = res.obs[0]
    assert out[False].confidence is None and out[False].cand_votes is None
    s, f = out[True].cand_votes
    assert s >= 3 and f == 0 and out[True].confidence is not None and out[True].confidence > 0.5
