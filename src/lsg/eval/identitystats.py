"""Identity-check statistics (CHANGE 3 Part N, N3; `uv run lsg chg3 idstats --run <details run>`).

From a FULL details run with `association.identity.enabled` (`tau_id` = -1 scores every gated pair
without rejecting any), every scored (node, observation) pair is labelled with the GT (eval only):
**same** = the observation is the node's own rescan instance (a true move or the object in place),
**different** = it is another GT instance; observations without a GT instance are not labelled.
Per threshold: share of *same* pairs kept and share of *different* pairs rejected, per mode.

Pre-written pick (DECISIONS, CHANGE 3 Part N): `tau_id` = the largest threshold that keeps >= 95 % of the
oracle *same* pairs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lsg.graph.model import object_id_of

GRID = [round(x, 2) for x in np.arange(-0.2, 1.0001, 0.01)]


def labelled_pairs(run: Path, mode: str) -> list[tuple[str, float | None]]:
    """[(label, similarity)] of every scored pair in `details/<mode>/`; label same / different."""
    out = []
    for f in sorted(Path(run, "details", mode).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        truth = d.get("token_truth") or {}
        seen = set()
        for nid, tok, sim, _ in d.get("identity") or []:
            if (nid, tok) in seen or tok not in truth:
                continue
            seen.add((nid, tok))
            oid = object_id_of(nid)
            out.append(("same" if oid is not None and int(truth[tok][0]) == int(oid) else "different", sim))
    return out


def rates(pairs: list[tuple[str, float | None]]) -> dict[str, Any]:
    """Counts, similarity quantiles per label, and keep / reject rates per threshold (None sims always kept)."""
    sims = {k: np.array([s for lab, s in pairs if lab == k and s is not None], float) for k in ("same", "different")}
    none = {k: sum(1 for lab, s in pairs if lab == k and s is None) for k in ("same", "different")}
    n = {k: len(sims[k]) + none[k] for k in sims}
    q = {k: (np.round(np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]), 3).tolist() if len(v) else None) for k, v in sims.items()}
    curve = []
    for t in GRID:
        keep = (int((sims["same"] >= t).sum()) + none["same"]) / n["same"] if n["same"] else None
        rej = int((sims["different"] < t).sum()) / n["different"] if n["different"] else None
        curve.append({"tau": t, "keep_same": keep, "reject_different": rej})
    return {"n": n, "no_descriptor": none, "quantiles_5_25_50_75_95": q, "curve": curve}


def pick_tau(r: Mapping[str, Any], keep_min: float = 0.95) -> float | None:
    """Largest threshold keeping >= `keep_min` of the *same* pairs."""
    ok = [c["tau"] for c in r["curve"] if c["keep_same"] is not None and c["keep_same"] >= keep_min]
    return max(ok) if ok else None


def at(r: Mapping[str, Any], tau: float) -> dict[str, Any]:
    """The curve row nearest to `tau`."""
    return min(r["curve"], key=lambda c: abs(c["tau"] - tau))


def identity_stats(run: Path, modes=("oracle", "perception")) -> dict[str, Any]:
    """Per mode `rates`, the pre-written oracle pick, and every mode's rates at the pick."""
    res = {m: rates(labelled_pairs(run, m)) for m in modes if Path(run, "details", m).is_dir()}
    pick = pick_tau(res["oracle"]) if "oracle" in res else None
    return {"modes": res, "pick": pick, "at_pick": {m: (at(r, pick) if pick is not None else None) for m, r in res.items()}}


def format_stats(s: Mapping[str, Any]) -> str:
    """Printable summary."""
    lines = []
    for m, r in s["modes"].items():
        lines.append(f"{m}: pairs same {r['n']['same']} (no descriptor {r['no_descriptor']['same']}), different "
                     f"{r['n']['different']} (no descriptor {r['no_descriptor']['different']})")  # fmt: skip
        lines.append(f"  similarity quantiles 5/25/50/75/95 %: same {r['quantiles_5_25_50_75_95']['same']}, "
                     f"different {r['quantiles_5_25_50_75_95']['different']}")  # fmt: skip
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            c = at(r, t)
            ks = "-" if c["keep_same"] is None else f"{100 * c['keep_same']:.1f}"
            rd = "-" if c["reject_different"] is None else f"{100 * c['reject_different']:.1f}"
            lines.append(f"  tau {t:.2f}: keep same {ks} %, reject different {rd} %")
    lines.append(f"pick (largest tau keeping >= 95 % of oracle same pairs): {s['pick']}")
    for m, c in s["at_pick"].items():
        if c is not None:
            lines.append(f"  {m} at the pick: keep same {c['keep_same']}, reject different {c['reject_different']}")
    return "\n".join(lines)
