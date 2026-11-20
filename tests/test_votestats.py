"""`lsg.eval.votestats` (CHANGE 3 Part C): vote weights as likelihood ratios."""

import json
import math

import pytest

from lsg.config import load_config
from lsg.eval.votestats import kind_rates, node_vote_weights, weights_from_units

CFG = load_config()


def test_rates_weight_every_unit_once_and_smooth():
    # one object with 30 seen votes and one with 1 free vote count the same
    r = kind_rates([{"seen": 30}, {"free": 1}, {}])
    assert r["seen"] == pytest.approx((1 + 1 / 3) / 3) and r["free"] == pytest.approx((1 + 1 / 3) / 3)
    assert r["free_like"] == pytest.approx((1 / 3) / 3)
    assert sum(r.values()) == pytest.approx(1.0)


def test_weights_are_likelihood_ratios_with_fallback():
    present = [{"seen": 9, "free": 1}] * 25  # unchanged: mostly seen (25 free votes)
    gone = [{"seen": 1, "free": 3}] * 25  # removed: mostly free (25 seen votes)
    w = weights_from_units(present, gone, min_votes=20, fallback=2.5)
    rp, rg = w["rates"]["present"], w["rates"]["gone"]
    assert w["L_hit"]["value"] == pytest.approx(math.log(rp["seen"] / rg["seen"])) and w["L_hit"]["value"] > 0
    assert w["L_miss"]["value"] == pytest.approx(math.log(rg["free"] / rp["free"])) and w["L_miss"]["value"] > 0
    assert w["L_free_like"]["fallback"] and w["L_free_like"]["value"] == 2.5  # no free-like votes at all
    assert w["votes"]["gone"]["free"] == 75 and w["units"] == {"present": 25, "gone": 25}


def test_node_vote_weights_from_details(tmp_path):
    d = tmp_path / "details" / "oracle"
    d.mkdir(parents=True)
    for k in range(4):
        rows = [{"gt_status": "unchanged", "votes": {"seen": 6, "free": 1, "free_like": 0}} for _ in range(5)]
        rows += [{"gt_status": "removed", "votes": {"seen": 3, "free": 6, "free_like": 0}} for _ in range(2)]
        rows += [{"gt_status": "absent_unlisted", "votes": {"seen": 0, "free": 9, "free_like": 0}}]  # ignored
        (d / f"R{k}_S{k}.json").write_text(json.dumps({"objects": rows}), encoding="utf-8")
    w = node_vote_weights(tmp_path, CFG, "oracle")
    assert w["n_pairs"] == 4 and w["units"] == {"present": 20, "gone": 8}
    assert not w["L_hit"]["fallback"] and not w["L_miss"]["fallback"] and w["L_free_like"]["fallback"]
    assert w["L_hit"]["value"] > 0 and w["L_miss"]["value"] > w["L_hit"]["value"]
    lo, hi = w["L_miss"]["ci"]
    assert lo <= w["L_miss"]["estimate"] <= hi
