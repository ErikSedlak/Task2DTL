"""Vote weights from sensor statistics (CHANGE 3 Part C, `lsg chg3 votestats`).

FULL's log-odds confidence adds L_hit per counted seen vote and subtracts L_miss / L_free_like per
counted free / free-like vote. Part C measures these weights as likelihood ratios on the tuning set
instead of setting them by hand:

- each unit (a reference object, later a candidate) contributes the **shares** of its counted votes
  per kind (seen, free, free_like), so every unit weighs 1 and long views do not dominate; units
  without counted votes carry no information and are skipped;
- P(kind | class) = (sum of the units' shares + 1/3) / (number of units + 1): one pseudo-unit with
  uniform shares (Laplace smoothing);
- reference nodes: class `present` = GT unchanged, `gone` = GT removed (scored statuses only);
  L_hit = ln P(seen | present) / P(seen | gone), L_miss = ln P(free | gone) / P(free | present),
  L_free_like likewise;
- fallback (written before the run): a kind with fewer than `min_votes` counted votes in either
  class keeps `fallback` (2.5);
- 95% CIs by a bootstrap over pairs (seeded).

Reads a run's `details/<mode>/*.json` (`run_eval(details=True)`); lives in `eval/` (GT). Writes
`outputs/<run>/votestats/metrics.json` (hard rule 5). The weights go into `configs/default.yaml` by
hand (hard rule 4: thresholds live in the config).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("seen", "free", "free_like")
# weight name, kind, class whose rate is the numerator ("present" for positive evidence)
WEIGHTS = (("L_hit", "seen", "present"), ("L_miss", "free", "gone"), ("L_free_like", "free_like", "gone"))


def kind_rates(units: Sequence[Mapping[str, int]]) -> dict[str, float]:
    """Smoothed P(kind | class) from units' counted votes (module docstring)."""
    sh = np.zeros(len(KINDS))
    n = 0
    for u in units:
        v = np.array([float(u.get(k, 0)) for k in KINDS])
        if v.sum() <= 0:
            continue
        sh += v / v.sum()
        n += 1
    return {k: float((sh[i] + 1.0 / 3.0) / (n + 1)) for i, k in enumerate(KINDS)}


def weights_from_units(present: Sequence[Mapping[str, int]], gone: Sequence[Mapping[str, int]],
                       min_votes: int, fallback: float) -> dict[str, Any]:  # fmt: skip
    """Likelihood-ratio weights + the counts behind them; a kind below `min_votes` keeps `fallback`."""
    rp, rg = kind_rates(present), kind_rates(gone)
    tot = {c: {k: int(sum(int(u.get(k, 0)) for u in us)) for k in KINDS} for c, us in (("present", present), ("gone", gone))}
    out: dict[str, Any] = {"rates": {"present": rp, "gone": rg}, "votes": tot,
                           "units": {"present": len(present), "gone": len(gone)}}  # fmt: skip
    for name, kind, num in WEIGHTS:
        den = "gone" if num == "present" else "present"
        enough = min(tot["present"][kind], tot["gone"][kind]) >= int(min_votes)
        est = math.log((rp if num == "present" else rg)[kind] / (rg if num == "present" else rp)[kind])
        out[name] = {"estimate": est, "value": est if enough else float(fallback), "fallback": not enough,
                     "num_class": num, "den_class": den}  # fmt: skip
    return out


def node_units(details_mode_dir: Path) -> dict[str, dict[str, list[dict[str, int]]]]:
    """Per pair: {"present": [votes of GT-unchanged objects], "gone": [votes of GT-removed objects]}."""
    out: dict[str, dict[str, list[dict[str, int]]]] = {}
    for f in sorted(Path(details_mode_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        pr: dict[str, list[dict[str, int]]] = {"present": [], "gone": []}
        for r in d["objects"]:
            cls = {"unchanged": "present", "removed": "gone"}.get(r["gt_status"])
            if cls is None or not r.get("votes"):
                continue
            pr[cls].append({k: int(r["votes"].get(k, 0)) for k in KINDS})
        out[f.stem] = pr
    return out


def bootstrap(per_pair: Mapping[str, Mapping[str, Sequence[Mapping[str, int]]]], n_boot: int, seed: int,
              min_votes: int, fallback: float) -> dict[str, list[float]]:  # fmt: skip
    """95% percentile CIs of the weight estimates, resampling pairs."""
    keys = sorted(per_pair)
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {n: [] for n, _, _ in WEIGHTS}
    for _ in range(int(n_boot)):
        pick = [keys[i] for i in rng.integers(len(keys), size=len(keys))]
        pres = [u for k in pick for u in per_pair[k]["present"]]
        gone = [u for k in pick for u in per_pair[k]["gone"]]
        w = weights_from_units(pres, gone, min_votes, fallback)
        for n, _, _ in WEIGHTS:
            samples[n].append(w[n]["estimate"])
    return {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in samples.items()}


def node_vote_weights(run_dir: str | Path, cfg: Mapping[str, Any], mode: str = "oracle") -> dict[str, Any]:
    """The reference-node weights of one run (module docstring), with CIs."""
    vc = cfg["chg3"]["part_c"]["votestats"]
    per = node_units(Path(run_dir) / "details" / mode)
    pres = [u for p in per.values() for u in p["present"]]
    gone = [u for p in per.values() for u in p["gone"]]
    w = weights_from_units(pres, gone, int(vc["min_votes"]), float(vc["fallback"]))
    ci = bootstrap(per, int(vc["n_boot"]), int(vc["seed"]), int(vc["min_votes"]), float(vc["fallback"]))
    for n in ci:
        w[n]["ci"] = ci[n]
    w["n_pairs"] = len(per)
    w["mode"] = mode
    return w


def candidate_units(details_mode_dir: Path) -> dict[str, dict[str, list[dict[str, int]]]]:
    """Per pair: depth candidates' cluster votes; `present` = explained by a GT instance, `gone` = not."""
    out: dict[str, dict[str, list[dict[str, int]]]] = {}
    for f in sorted(Path(details_mode_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        pr: dict[str, list[dict[str, int]]] = {"present": [], "gone": []}
        for r in d.get("candidates") or []:
            if r.get("kind", "depth") != "depth" or not r.get("cand_votes"):
                continue
            s, fr = (int(x) for x in r["cand_votes"])
            pr["present" if r["explained"] else "gone"].append({"seen": s, "free": fr})
        out[f.stem] = pr
    return out


def candidate_vote_weights(run_dir: str | Path, cfg: Mapping[str, Any], mode: str = "perception") -> dict[str, Any]:
    """Candidate cluster-vote weights: L_seen (seen | explained vs unexplained) and L_free, with CIs."""
    vc = cfg["chg3"]["part_c"]["votestats"]
    per = candidate_units(Path(run_dir) / "details" / mode)
    pres = [u for p in per.values() for u in p["present"]]
    gone = [u for p in per.values() for u in p["gone"]]
    w = weights_from_units(pres, gone, int(vc["min_votes"]), float(vc["fallback"]))
    ci = bootstrap(per, int(vc["n_boot"]), int(vc["seed"]), int(vc["min_votes"]), float(vc["fallback"]))
    out = {"rates": w["rates"], "votes": w["votes"], "units": w["units"], "n_pairs": len(per), "mode": mode}
    for src, dst in (("L_hit", "L_seen"), ("L_miss", "L_free")):
        out[dst] = {**w[src], "ci": ci[src]}
    return out


def format_candidate_weights(res: Mapping[str, Any]) -> str:
    """Terminal summary of `candidate_vote_weights`."""
    lines = [f"candidates ({res['mode']}), {res['n_pairs']} pairs; explained {res['units']['present']}, "
             f"unexplained {res['units']['gone']}"]  # fmt: skip
    for c, lab in (("present", "explained"), ("gone", "unexpl.")):
        r, v = res["rates"][c], res["votes"][c]
        lines.append(f"  {lab:<9} P(seen)={r['seen']:.3f} ({v['seen']} votes)  P(free)={r['free']:.3f} ({v['free']} votes)")
    for n in ("L_seen", "L_free"):
        w = res[n]
        lines.append(f"  {n:<8} estimate {w['estimate']:+.2f} [{w['ci'][0]:+.2f}, {w['ci'][1]:+.2f}]  -> value "
                     f"{w['value']:.2f}{'  (fallback: too few votes)' if w['fallback'] else ''}")  # fmt: skip
    return "\n".join(lines)


def format_weights(res: Mapping[str, Any]) -> str:
    """Terminal summary of `node_vote_weights`."""
    lines = [f"mode {res['mode']}, {res['n_pairs']} pairs; units present {res['units']['present']}, gone {res['units']['gone']}"]
    for c in ("present", "gone"):
        r, v = res["rates"][c], res["votes"][c]
        lines.append(f"  {c:<8} " + "  ".join(f"P({k})={r[k]:.3f} ({v[k]} votes)" for k in KINDS))
    for n, _, _ in WEIGHTS:
        w = res[n]
        lines.append(f"  {n:<12} estimate {w['estimate']:+.2f} [{w['ci'][0]:+.2f}, {w['ci'][1]:+.2f}]  -> value "
                     f"{w['value']:.2f}{'  (fallback: too few votes)' if w['fallback'] else ''}")  # fmt: skip
    return "\n".join(lines)
