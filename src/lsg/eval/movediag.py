"""Breakdown of predicted moves by old-place evidence (CHANGE 3 Part A diagnostic, `lsg movediag`).

Reads the per-object rows of a run with details (`run_eval(details=True)` ->
`details/<mode>/<ref>_<rescan>.json`) and counts every reference node predicted `moved` by:

- GT status: `moved` (TP) / `unchanged` / `absent_unlisted` / `removed` / `dont_care` (incl. nonrigid);
- old-place evidence from its counted votes (the classes the Part A rules act on):
  `net_seen` (seen > free + free-like, checked first), `free_ev` (free evidence), `unobserved` (no counted
  vote), `mixed` (anything else);
- association: `own` (the matched observation is the node's own rescan instance), `other`, `none`
  (no GT instance behind the observation);
- predicted move distance ‖c_pred − c_ref‖ in the eval displacement buckets.

So the effect of a rule on false and true moves can be read off before the rule exists.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lsg.change.visibility import NodeVotes, old_place_class

STATUSES = ("moved", "unchanged", "absent_unlisted", "removed", "dont_care")
EVIDENCE = ("net_seen", "free_ev", "unobserved", "mixed")


def evidence_class(votes: Mapping[str, int] | None, cfg: Mapping[str, Any]) -> str:
    """Old-place evidence class of one details row (`change.visibility.old_place_class`, the rule's own test)."""
    if not votes:
        return old_place_class(None, cfg)
    return old_place_class(NodeVotes(seen=int(votes.get("seen", 0)), free=int(votes.get("free", 0)),
                                     free_like=int(votes.get("free_like", 0))), cfg)  # fmt: skip


def assoc_class(row: Mapping[str, Any]) -> str:
    """`own` / `other` / `none` for the observation the node was matched to."""
    t = row.get("obs_gt")
    if not t:
        return "none"
    return "own" if int(t[0]) == int(row["oid"]) else "other"


def distance_bucket(row: Mapping[str, Any], edges: Sequence[float]) -> str:
    """Predicted move distance bucket, e.g. `[0.1,0.3)`; `?` without a reference centroid."""
    if row.get("centroid_ref") is None or row.get("centroid") is None:
        return "?"
    d = float(np.linalg.norm(np.asarray(row["centroid"], float) - np.asarray(row["centroid_ref"], float)))
    e = [float(x) for x in edges]
    if d < e[0]:
        return f"<{e[0]:g}"
    for lo, hi in zip(e, e[1:]):
        if d < hi:
            return f"[{lo:g},{hi:g})"
    return f">={e[-1]:g}"


def gt_class(status: str) -> str:
    """GT status folded into `STATUSES` (nonrigid and other dont_care reasons -> `dont_care`)."""
    return status if status in STATUSES else "dont_care"


def move_breakdown(run_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Per mode: counts of predicted-moved nodes by (GT, evidence), (GT, association), (GT, distance).

    Returns:
        `{mode: {"n_pairs", "n_moved", "by_evidence", "by_assoc", "by_distance"}}`; each `by_*` maps
        `"<gt>|<class>"` -> count.
    """
    edges = cfg["eval"]["disp_buckets"]
    out: dict[str, Any] = {}
    root = Path(run_dir) / "details"
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        ev: Counter = Counter()
        asc: Counter = Counter()
        dist: Counter = Counter()
        files = sorted(mdir.glob("*.json"))
        n = 0
        for f in files:
            for r in json.loads(f.read_text(encoding="utf-8"))["objects"]:
                if r["pred_state"] != "moved":
                    continue
                n += 1
                g = gt_class(r["gt_status"])
                ev[f"{g}|{evidence_class(r.get('votes'), cfg)}"] += 1
                asc[f"{g}|{assoc_class(r)}"] += 1
                dist[f"{g}|{distance_bucket(r, edges)}"] += 1
        out[mdir.name] = {"n_pairs": len(files), "n_moved": n, "by_evidence": dict(ev), "by_assoc": dict(asc),
                          "by_distance": dict(dist)}  # fmt: skip
    return out


def format_breakdown(res: Mapping[str, Any]) -> str:
    """Terminal tables: rows = GT status, columns = class."""
    lines: list[str] = []
    for mode, r in res.items():
        lines.append(f"## {mode}: {r['n_moved']} predicted moved over {r['n_pairs']} pairs")
        for key in ("by_evidence", "by_assoc", "by_distance"):
            cols = sorted({k.split("|", 1)[1] for k in r[key]}, key=lambda c: (EVIDENCE + ("own", "other", "none")).index(c)
                          if c in EVIDENCE + ("own", "other", "none") else 99)  # fmt: skip
            lines.append(f"  {key[3:]:<16}" + "".join(f"{c:>12}" for c in cols) + f"{'total':>8}")
            for g in STATUSES:
                vals = [int(r[key].get(f"{g}|{c}", 0)) for c in cols]
                if sum(vals):
                    lines.append(f"  {g:<16}" + "".join(f"{v:>12}" for v in vals) + f"{sum(vals):>8}")
    return "\n".join(lines)
