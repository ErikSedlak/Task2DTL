"""M7 ablation registry, resume logic, and the results writers (on fake run results)."""

import copy

import numpy as np
import pytest

import lsg.eval.ablations as abl
from lsg.config import load_config
from lsg.eval import m7_checks as m7
from lsg.runinfo import write_metrics

CFG = load_config()


def test_apply_overrides_copies_and_rejects_unknown_keys():
    c = abl.apply_overrides(CFG, {"appearance.enabled": False, "visibility.tau_free": 0.3})
    assert c["appearance"]["enabled"] is False and c["visibility"]["tau_free"] == 0.3
    assert CFG["appearance"]["enabled"] is True and CFG["visibility"]["tau_free"] == 0.15
    with pytest.raises(KeyError):
        abl.apply_overrides(CFG, {"appearance.enabeld": False})
    with pytest.raises(KeyError):
        abl.apply_overrides(CFG, {"nope.x": 1})


def test_registry_overrides_exist_and_sweeps_come_from_config():
    reg = abl.registry(CFG)
    for a in reg.values():
        abl.apply_overrides(CFG, a.overrides)  # every key exists
    mc = CFG["m7"]
    assert sum(a.group == "frac" for a in reg.values()) == len(mc["fracs"]) * len(mc["frac_seeds"])
    assert sum(a.group == "tau_free" for a in reg.values()) == len(mc["tau_free_sweep"])
    assert reg["FULL-contrast"].overrides == {"visibility.seen_needs_contrast": False}
    assert reg["FULL-relaxed"].overrides == {"perception.relaxed.enabled": False}  # CHANGE 1 off
    # CHANGE 2 rows; every override key exists (apply_overrides raises on typos)
    assert reg["FULL-candconf"].overrides == {"confidence.candidates.enabled": False}
    assert reg["FULL-candconf-free"].overrides == {"confidence.candidates.free_votes": False}
    assert reg["FULL+classprior"].overrides == {"confidence.class_priors": True, "confidence.p_del": 0.2}
    assert reg["FULL-m7"].overrides == abl.M7_OVERRIDES
    assert reg["FULL-moveev"].overrides == {"association.old_place.enabled": False}
    assert abl.M7_OVERRIDES["association.old_place.enabled"] is False
    assert all(abl.M7_OVERRIDES[f"confidence.{k}"] == 2.5 for k in ("L_hit", "L_miss", "L_free_like"))
    assert reg["FULL-learnedL"].overrides["confidence.L_hit"] == 2.5
    assert reg["FULL-candtrack"].overrides == {"confidence.candidates.track.enabled": False}
    assert abl.M7_OVERRIDES["confidence.candidates.track.enabled"] is False
    assert reg["FULL-segment"].overrides == {"perception.segment.method": "none"}  # CHANGE 3 Part N
    assert abl.M7_OVERRIDES["perception.segment.method"] == "none"
    assert reg["FULL-appconf"].overrides == {"confidence.candidates.track.appearance": False}
    assert reg["FULL-offlinealign"].overrides == {"alignment.online.enabled": False}  # CHANGE 3 Part B
    assert abl.M7_OVERRIDES["alignment.online.enabled"] is False
    for a in reg.values():
        abl.apply_overrides(CFG, a.overrides)
    m7 = abl.apply_overrides(CFG, abl.M7_OVERRIDES)
    assert not m7["perception"]["relaxed"]["enabled"] and not m7["confidence"]["candidates"]["enabled"]
    assert m7["confidence"]["class_priors"] and m7["confidence"]["p_del"] == 0.2 and m7["confidence"]["prior_p"] == 0.9
    # CHANGE 3: the default vocabulary also excludes dev_tune's rooms; FULL-m7 keeps the M7 exclusions
    assert "splits/dev_tune.txt" in CFG["train_meta"]["exclude_splits"]
    assert m7["train_meta"]["exclude_splits"] == ["splits/dev_train.txt", "splits/eval_relchange_train.txt"]
    for name in (*mc["dev_rows"], *mc["relchange_rows"], *mc["regression_refs"]):
        assert name in reg


def test_select_by_name_group_and_all():
    assert [a.name for a in abl.select(CFG, "FULL,B2,FULL")] == ["FULL", "B2"]
    assert all(a.group == "frac" for a in abl.select(CFG, "group:frac"))
    assert len(abl.select(CFG, "all")) == len(abl.registry(CFG))
    with pytest.raises(KeyError):
        abl.select(CFG, "FULL-nothing")


def _fake_eval_result(cfg, tmp_path, run_id):
    metrics = {"split": "x", "obs_modes": list(abl.MODES), "summary": {}, "n_pairs": 1}
    p = write_metrics(tmp_path / run_id, metrics, cfg, extra={"wall_time_s": 1.0})
    return metrics, p


CLEAN = {"sha": "1" * 40, "dirty": False}


def test_resume_only_with_same_config_and_commit(tmp_path, monkeypatch):
    import lsg.runinfo

    cfg = copy.deepcopy(CFG)
    cfg["outputs_root"] = str(tmp_path)
    monkeypatch.setattr(abl, "outputs_root", lambda c=None: tmp_path)
    monkeypatch.setattr(lsg.runinfo, "git_sha", lambda: dict(CLEAN, dirty=True))
    monkeypatch.setattr(abl, "git_sha", lambda: CLEAN)
    _, p = _fake_eval_result(cfg, tmp_path, "r/FULL")
    assert abl.reusable(p, cfg) is None  # made from a dirty tree: never reused
    monkeypatch.setattr(lsg.runinfo, "git_sha", lambda: CLEAN)
    _, p = _fake_eval_result(cfg, tmp_path, "r/FULL")
    assert abl.reusable(p, cfg) is not None
    c2 = abl.apply_overrides(cfg, {"visibility.tau_free": 0.2})
    assert abl.reusable(p, c2) is None  # other config
    monkeypatch.setattr(abl, "git_sha", lambda: {"sha": "0" * 40, "dirty": False})
    assert abl.reusable(p, cfg) is None  # other commit

    calls = []
    monkeypatch.setattr(abl, "run_eval", lambda *a, **k: calls.append(k) or {"metrics_path": "m", "wall_time_s": 0})
    r = abl.run_ablation(cfg, abl.registry(cfg)["FULL"], [("R", "S")], "r/FULL", "x")
    assert calls and r["reused"] is False  # commit differs -> rerun
    monkeypatch.setattr(abl, "git_sha", lambda: CLEAN)
    calls.clear()
    monkeypatch.setattr(abl, "run_eval", lambda *a, **k: calls.append(k) or {"metrics_path": "m", "wall_time_s": 0})
    r = abl.run_ablation(cfg, abl.registry(cfg)["FULL"], [("R", "S")], "r/FULL", "x")
    assert not calls and r["reused"] is True and r["ablation"]["name"] == "FULL"
    r = abl.run_ablation(cfg, abl.registry(cfg)["FULL"], [("R", "S")], "r/FULL", "x", resume=False)
    assert len(calls) == 1 and calls[0]["frac"] == 1.0


def test_metrics_diff_shared_keys_and_ignore():
    a = {"x": 1, "thin_n": 3, "l": [1, {"y": 2}], "only_a": 0}
    b = {"x": 1, "thin_n": 4, "l": [1, {"y": 3}], "only_b": 0}
    assert m7.metrics_diff(a, b) == [".l[1].y"]
    assert m7.metrics_diff(a, b, ignore=()) == [".l[1].y", ".thin_n"]


def _fake_summary(rng, name, group, overrides=None):
    keys = [*m7.MAIN_KEYS, *m7.EXTRA_KEYS, *m7._rel_keys()]
    r = {"summary": {}, "n_pairs": 110, "split": "eval_val", "metrics_path": "x", "wall_time_s": 1.0,
         "ablation": {"name": name, "group": group, "overrides": overrides or {}}}  # fmt: skip
    for mode in m7.MODES:
        M = {}
        for k in keys:
            v = float(rng.uniform(0.1, 0.9))
            M[k] = {"value": v, "ci": [v - 0.05, v + 0.05], "n": 50}
        r["summary"][mode] = {"metrics": M, "loc_error": {"n": 3, "median": 0.1, "p90": 0.3},
                              "counts": {"graph_invalid": 0}}  # fmt: skip
    return m7.summarize(r)


def test_results_tables_plots_and_acceptance(tmp_path):
    rng = np.random.default_rng(0)
    reg = abl.registry(CFG)
    val = {a.name: _fake_summary(rng, a.name, a.group, dict(a.overrides)) for a in reg.values()}
    res = {"env": {"cuda_available": False}, "dev": {"FULL": val["FULL"]}, "val": val,
           "relchange": {"FULL": val["FULL"]}, "regressions": {"B2": {"ref": "r", "n_diff": 0, "diff": []}},
           "determinism": {"pair": ["a" * 8, "b" * 8], "n_diff": 0, "diff": []}, "dev_only": False}  # fmt: skip
    res["frac"] = m7.frac_table(val, CFG)
    assert set(res["frac"]) == {"1", "0.5", "0.25", "0.1"}
    assert res["frac"]["0.5"]["oracle"]["removed_R"]["n_seeds"] == 3
    assert res["frac"]["1"]["oracle"]["removed_R"]["sd"] is None
    res["results_md"] = str(m7.write_results(res, CFG, tmp_path / "results.md"))
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    for s in ("FULL-contrast", "τ_free sweep", "Observed fraction", "Relations (relchange", "Determinism", "["):
        assert s in md
    res["plots"] = m7.write_plots(res, CFG, tmp_path)
    assert len(res["plots"]) == 3 and all((tmp_path / p).exists() for p in res["plots"])
    acc = m7.acceptance(res, CFG)
    assert acc["graphs_validate"]["pass"] and acc["determinism_one_pair"]["pass"] and acc["plots_written"]["pass"]
    assert acc["val_full_split"]["pass"]
    res.update(acceptance=acc, metrics_path="m", wall_time_s=1.0)
    assert "acceptance" in m7.format_summary(res)
