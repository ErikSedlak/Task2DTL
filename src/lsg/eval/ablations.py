"""Ablation registry and runner (DESIGN §8, M7): `lsg ablate`, used by `lsg m7`.

An ablation is one `run_eval` call: a method, config overrides (dotted keys, e.g.
`{"appearance.enabled": False}`), and an observed fraction + chunk seed (§7.3). Every run
writes its own `outputs/<run_id>/<group>/<name>/metrics.json` (hard rule 5). Nothing here tunes
anything: each row on val is run once and reported.

Resume: a run whose `metrics.json` already exists is reused only if its config hash **and**
git commit equal the current ones and it was made from a clean tree (`git.dirty` false), so
results of uncommitted code are never reused. `metrics.json` is written at the end of
`run_eval`, so a crashed run leaves nothing to reuse.

Lives in `eval/` (runs the harness, which reads GT).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsg.config import config_hash, outputs_root
from lsg.eval.harness import run_eval
from lsg.runinfo import git_sha

MODES = ("perception", "oracle")

# The M7 method (= outputs/m7_final, REPORT tables): CHANGE 1 (relaxed candidates) and CHANGE 2 (no class
# prior, candidate log-odds) off. Ablation row FULL-m7 and the §7.4 audit rerun use it.
M7_OVERRIDES: dict[str, Any] = {
    "perception.relaxed.enabled": False,
    "confidence.candidates.enabled": False,
    "confidence.class_priors": True,
    "confidence.prior_p": 0.9,
    "confidence.p_del": 0.2,
    # M7 vocabulary (CHANGE 3 also excludes dev_tune's rooms): same CLIP vocabulary and cache keys as m7_final
    "train_meta.exclude_splits": ["splits/dev_train.txt", "splits/eval_relchange_train.txt"],
    "association.old_place.enabled": False,  # CHANGE 3 Part A off
    "confidence.L_hit": 2.5, "confidence.L_miss": 2.5, "confidence.L_free_like": 2.5,  # CHANGE 3 Part C off
    "confidence.candidates.track.enabled": False,
    "perception.segment.method": "none",  # CHANGE 3 Part N off
    "confidence.candidates.track.appearance": False,
    "alignment.online.enabled": False,  # CHANGE 3 Part B off (the M7 pre-pass)
}

# CHANGE 1 + 2 as fixed by Erik on 2026-11-20 with the CHANGE 3 Part 0 tuning-set changes (flat start value 0.9 /
# p_del 0.2, dev_tune rooms out of the CLIP vocabulary) = the config of `outputs/chg3_dev_baseline/FULL`. Every
# CHANGE 3 part after Part 0 is off. Ablation row FULL-chg2 (docs/CHANGE_online_graph.md §7.4).
CHG2_OVERRIDES: dict[str, Any] = {
    "association.old_place.enabled": False,  # Part A
    "confidence.L_hit": 2.5, "confidence.L_miss": 2.5, "confidence.p_del": 0.2,  # Part C
    "confidence.candidates.track.enabled": False,
    "confidence.candidates.p_new_min": 0.92,
    "perception.segment.method": "none",  # Part N
    "confidence.candidates.track.appearance": False,
    "alignment.online.enabled": False,  # Part B
}


@dataclass(frozen=True)
class Ablation:
    """One row of the M7 tables."""

    name: str
    method: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    frac: float = 1.0
    seed: int = 0
    group: str = "variant"  # method | component | variant | tau_free | frac
    desc: str = ""


def apply_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Deep copy of `cfg` with dotted-key overrides; every key must already exist (no typos)."""
    c = copy.deepcopy(dict(cfg))
    for key, val in overrides.items():
        parts = key.split(".")
        d = c
        for p in parts[:-1]:
            if not isinstance(d, dict) or p not in d:
                raise KeyError(f"override {key!r}: no config block {p!r}")
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"override {key!r}: no config key {parts[-1]!r}")
        d[parts[-1]] = val
    return c


def registry(cfg: Mapping[str, Any]) -> dict[str, Ablation]:
    """All M7 rows (DESIGN §8 + the M6 / M7 switches), in report order.

    Sweep values (`m7.tau_free_sweep`, `m7.fracs`, `m7.frac_seeds`) come from the config.
    """
    m7 = cfg.get("m7", {})
    rows = [
        Ablation("B0", "B0", group="method", desc="naive same-label matching"),
        Ablation("B1", "B1", group="method", desc="+ visibility votes, decorrelated, deferred"),
        Ablation("B2", "B2", group="method", desc="+ confirmed-in-place skip, Hungarian, ICP moved test"),
        Ablation("FULL", "FULL", group="method", desc="+ confidence / priors, appearance, seen needs contrast"),
        Ablation("FULL-contrast", "FULL", {"visibility.seen_needs_contrast": False}, group="component",
                 desc="seen votes without contrast (= M6 FULL)"),
        Ablation("FULL-appearance", "FULL", {"appearance.enabled": False}, group="component",
                 desc="depth only (the brief's ablation)"),
        Ablation("FULL-thincand", "FULL", {"appearance.thin_candidates": False}, group="component",
                 desc="no thin-object candidates (perception)"),
        Ablation("FULL-relaxed", "FULL", {"perception.relaxed.enabled": False}, group="component",
                 desc="CHANGE 1 off: strict candidate geometry, voxels need L_hit * min_views"),
        Ablation("FULL-candconf", "FULL", {"confidence.candidates.enabled": False}, group="component",
                 desc="CHANGE 2 Part B off: candidates by view counts (= CHANGE 1)"),
        Ablation("FULL-candconf-free", "FULL", {"confidence.candidates.free_votes": False}, group="component",
                 desc="candidate log-odds without see-through votes"),
        Ablation("FULL+classprior", "FULL", {"confidence.class_priors": True, "confidence.p_del": 0.2},
                 group="component", desc="CHANGE 2 Part A off: learned class priors, p_del 0.2 (M7)"),
        Ablation("FULL-priors", "FULL", {"confidence.class_priors": False, "confidence.prior_p": 0.9,
                                         "confidence.p_del": 0.2}, group="component",
                 desc="flat prior p = 0.9, p_del 0.2 at the current vote weights (the M7 row had 2.5 each)"),
        Ablation("FULL-candtrack", "FULL", {"confidence.candidates.track.enabled": False}, group="component",
                 desc="candidate confidence = CHANGE 2 median voxel log-odds (CHANGE 3 Part C cluster votes off)"),
        Ablation("FULL-learnedL", "FULL", {"confidence.L_hit": 2.5, "confidence.L_miss": 2.5,
                                           "confidence.L_free_like": 2.5, "confidence.p_del": 0.2}, group="component",
                 desc="CHANGE 3 Part C off: equal hand-set vote weights 2.5, p_del 0.2"),
        Ablation("FULL-appconf", "FULL", {"confidence.candidates.track.appearance": False}, group="component",
                 desc="thin-object candidates without confidence, always emitted (CHANGE 3 Part N5 off)"),
        Ablation("FULL-offlinealign", "FULL", {"alignment.online.enabled": False}, group="component",
                 desc="T_align refined by a pre-pass over all processed frames (CHANGE 3 Part B online alignment off)"),
        Ablation("FULL-segment", "FULL", {"perception.segment.method": "none"}, group="component",
                 desc="candidate voxels linked by space only (CHANGE 3 Part N co-segmentation off)"),
        Ablation("FULL-moveev", "FULL", {"association.old_place.enabled": False}, group="component",
                 desc="a move needs no old-place evidence (CHANGE 3 Part A off)"),
        Ablation("FULL-chg2", "FULL", CHG2_OVERRIDES, group="component",
                 desc="CHANGE 3 off after Part 0 (= CHANGE 1 + 2, the method before CHANGE 3; outputs/chg3_dev_baseline)"),
        Ablation("FULL-m7", "FULL", M7_OVERRIDES, group="component",
                 desc="the M7 method (CHANGE 1, 2 and 3 off) = outputs/m7_final"),
        Ablation("FULL-decorr", "FULL", {"decorrelation.enabled": False}, group="component",
                 desc="every vote counts (no viewpoint decorrelation)"),
        Ablation("B1-decorr", "B1", {"decorrelation.enabled": False}, group="component",
                 desc="B1 without viewpoint decorrelation"),
        Ablation("FULL+immediate", "FULL", {"states.commitment": "immediate"}, group="variant",
                 desc="removed committed at the first tentative removed frame"),
        Ablation("FULL+follow", "FULL", {"update.children_follow_parent": True}, group="variant",
                 desc="unobserved things follow their moved anchor"),
        Ablation("FULL-keepparent", "FULL", {"update.keep_supported_parent": False}, group="variant",
                 desc="re-checked things always get infer_support's parent"),
        Ablation("FULL+detector", "FULL", {"perception.detector.enabled": True}, group="variant",
                 desc="perception candidates from Grounding DINO + SAM 2 (CHANGE 3 Part E, det_new; needs the detection cache)"),
    ]
    for t in m7.get("tau_free_sweep", []):
        rows.append(Ablation(f"FULL_tau{float(t):g}", "FULL", {"visibility.tau_free": float(t)}, group="tau_free",
                             desc=f"tau_free = {float(t):g} m"))  # fmt: skip
    for f in m7.get("fracs", []):
        for s in m7.get("frac_seeds", [0]):
            rows.append(Ablation(f"FULL_frac{float(f):g}_s{int(s)}", "FULL", frac=float(f), seed=int(s), group="frac",
                                 desc=f"{100 * float(f):g}% of the frames, chunk seed {int(s)}"))  # fmt: skip
    return {a.name: a for a in rows}


def select(cfg: Mapping[str, Any], names: str | Sequence[str]) -> list[Ablation]:
    """Rows by name: a comma list, a list, `all`, or a group name (`group:frac`)."""
    reg = registry(cfg)
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out: list[Ablation] = []
    for n in names:
        if n == "all":
            out += list(reg.values())
        elif n.startswith("group:"):
            out += [a for a in reg.values() if a.group == n[6:]]
        elif n in reg:
            out.append(reg[n])
        else:
            raise KeyError(f"unknown ablation {n!r} (available: {', '.join(reg)})")
    seen: set[str] = set()
    return [a for a in out if not (a.name in seen or seen.add(a.name))]


def reusable(path: Path, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The stored metrics.json if it was made with this config, this git commit and a clean tree, else None."""
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sha = git_sha().get("sha")
    g = d.get("git", {})
    if d.get("config_hash") != config_hash(dict(cfg)) or sha is None or g.get("sha") != sha or g.get("dirty") is not False:
        return None
    return d


def run_ablation(
    cfg: Mapping[str, Any],
    ab: Ablation,
    pairs: Sequence[tuple[str, str]],
    run_id: str,
    split_name: str,
    modes: Sequence[str] = MODES,
    workers: int | None = None,
    resume: bool = True,
    details: bool = False,
) -> dict[str, Any]:
    """One ablation row: `run_eval` with the overrides (or the reused result).

    `details=True` also writes per-object outcomes (`details/`, the §7.4 audit's input); a stored
    row is reused then only if its `details/` folder exists.

    Returns:
        `run_eval`'s result (`metrics` block + `metrics_path`, `wall_time_s`) plus
        `"ablation"` (name, method, overrides, frac, seed) and `"reused"`.
    """
    c = apply_overrides(cfg, ab.overrides)
    meta = {"name": ab.name, "method": ab.method, "overrides": dict(ab.overrides), "frac": ab.frac,
            "seed": ab.seed, "group": ab.group, "desc": ab.desc}  # fmt: skip
    path = outputs_root(c) / run_id / "metrics.json"
    if resume and (d := reusable(path, c)) is not None and list(d["metrics"].get("obs_modes", [])) == list(modes) \
            and (not details or (path.parent / "details").is_dir()):
        return {**d["metrics"], "metrics_path": str(path), "wall_time_s": d.get("run", {}).get("wall_time_s", 0.0),
                "ablation": meta, "reused": True}  # fmt: skip
    r = run_eval(c, pairs, list(modes), method=ab.method, frac=ab.frac, seed=ab.seed, workers=workers,
                 run_id=run_id, split_name=split_name, details=details)  # fmt: skip
    return {**r, "ablation": meta, "reused": False}
