"""Log-odds node confidence (DESIGN §6.11; M6, values rescaled by Erik; CHANGE 2).

l0 = logit(`prior_p`) with `class_priors: false` (default since CHANGE 2 Part A: `prior_p` 0.5,
l0 = 0, no dataset prior; the deletion margin is set by `p_del` alone), or logit(p_persist(class))
from train metadata (`train_meta.class_priors`; label -> nyu40 -> global back-off) with
`class_priors: true` (M7 FULL, ablation row FULL+classprior).

- Reference node, unmatched or confirmed in place:
  l = l0 + clip(L_hit·seen - L_miss·free - L_free_like·free_like, ±clip), counted votes only.
- Matched reference node: l = l0 + clip(L_hit·views), where `views` = decorrelated viewpoints
  in which its observation was seen (`OnlineResult.obs_views`); free votes at the old place
  are ignored (they are explained by the move / the match).
- New node: l = clip(L_hit·views) from l = 0 (no prior for a newly observed object); a perception
  candidate with its own confidence (CHANGE 2 Part B, per-voxel log-odds) keeps that one.

Removal (FULL): unmatched, §3 free evidence, and p = sigmoid(l) < p_del.

With the spec's L = 0.4 and clip ±3, train priors of 0.91-1.0 could never reach p_del = 0.2;
L_miss = 2.5 makes 2 counted free votes delete a median-prior thing (chair: p = 0.17), while a
bed (p_persist ~0.996) needs 3 and mixed seen / free evidence protects a node (DECISIONS M6).
Without priors (CHANGE 2): l0 = 0 and p_del = 0.027 (margin 3.6) -> 2 net free votes delete any
class, 1 cannot; the same decisions as M7 `FULL-priors` (0.9 / 0.2, margin 3.58).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from lsg.change.visibility import NodeVotes
from lsg.train_meta import PriorTable, logit


def prior_logodds(label: str | None, nyu40: int | None, priors: PriorTable | None, cfg: Mapping[str, Any]) -> float:
    """l0 of a reference node (class prior, or the flat `prior_p`)."""
    cc = cfg["confidence"]
    if priors is None or not cc.get("class_priors", False):
        return logit(float(cc["prior_p"]))
    return logit(priors.p_persist(label, nyu40))


def _clip(x: float, cfg: Mapping[str, Any]) -> float:
    c = float(cfg["confidence"]["clip"])
    return max(-c, min(c, x))


def evidence(v: NodeVotes | None, cfg: Mapping[str, Any]) -> float:
    """Clipped log-odds change from a node's counted votes (unmatched / confirmed nodes)."""
    if v is None:
        return 0.0
    cc = cfg["confidence"]
    d = float(cc["L_hit"]) * v.seen - float(cc["L_miss"]) * v.free - float(cc["L_free_like"]) * v.free_like
    return _clip(d, cfg)


def node_logodds(
    l0: float,
    v: NodeVotes | None,
    cfg: Mapping[str, Any],
    matched: bool = False,
    views: int | None = None,
) -> float:
    """Final log-odds of a reference node (module docstring).

    Args:
        l0: Prior log-odds.
        v: Counted votes at the reference location.
        matched: The node was associated with an observation (its old-place free votes are ignored).
        views: Decorrelated views of the matched observation; None -> the node's counted seen votes.
    """
    if not matched:
        return l0 + evidence(v, cfg)
    n = views if views is not None else (v.seen if v is not None else 0)
    return l0 + _clip(float(cfg["confidence"]["L_hit"]) * n, cfg)


def new_logodds(views: int, cfg: Mapping[str, Any]) -> float:
    """Log-odds of a new node: clip(L_hit·views) from 0."""
    return _clip(float(cfg["confidence"]["L_hit"]) * int(views), cfg)


def obs_logodds(confidence: float | None, views: int, cfg: Mapping[str, Any]) -> float:
    """Log-odds of a new node: its candidate's own confidence (CHANGE 2) if set, else `new_logodds`."""
    if confidence is None:
        return new_logodds(views, cfg)
    p = min(max(float(confidence), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def probability(l: float) -> float:
    """sigmoid(l), rounded to 1e-6 (stable JSON / metrics)."""
    x = float(l)
    p = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))  # no overflow
    return round(p, 6)


def below_p_del(l: float, cfg: Mapping[str, Any]) -> bool:
    """p < p_del (the confidence half of the FULL removal rule)."""
    return probability(l) < float(cfg["confidence"]["p_del"])
