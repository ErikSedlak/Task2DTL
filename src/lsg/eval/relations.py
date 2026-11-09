"""Relation scoring (DESIGN §7.2): support parent and type of things vs. the rescan's 3DSSG.

GT = `gt_support` of the **rescan** relations (§5 predicate mapping; things whose only parents
are non-floor structure are excluded; several GT parents -> any one counts).

Scored things (Erik, M5):
- **GT-defined universe** (headline): reference things present in the rescan (GT status
  unchanged / moved, or `dont_care` small move) with a GT rescan parent, plus GT-new things
  matched to a predicted new node (mode's §5 rule). The node's current support edge is scored
  whatever its predicted state;
- **associated only** (`as`): the subset the method associated (predicted unchanged / moved;
  matched new nodes always count).

Subsets: (a) all; (b) GT-moved things; (c) moved things whose GT parent was **swapped**;
(cb, "c-broad") swapped or **gained** (mostly reference annotation gaps). Each with all GT
parents (`all`) and without floor parents (`nf`).

Parent ids are compared in rescan objectId space: GT parent = rescan objectId or `floor`;
predicted parent `r<oid>` -> oid, `n<k>` -> its token's GT rescan instance (`obs_truth`),
`floor` -> floor.

Counts: `rel_{policy}_{subset}_{universe}_{floor}_{n|par|typ}`; parent accuracy = par / n,
type accuracy = typ / par (given the correct parent).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lsg.eval.gt import GTLabels
from lsg.graph.model import FLOOR_ID, Graph, object_id_of, ref_node_id
from lsg.graph.relations import FLOOR, GTSupport

REL_POLICIES = {"stale": "R0", "geometric": "R1", "update": "upd"}  # policy -> count prefix
REL_SUBSETS = ("a", "b", "c", "cb")
REL_SUBSET_NAMES = {"a": "(a) all", "b": "(b) moved", "c": "(c) parent swapped", "cb": "(c-broad) swapped+gained"}
REL_UNIVERSES = ("gt", "as")
REL_FLOORS = ("all", "nf")


def relation_count_keys() -> list[str]:
    """All relation count names (fixed order)."""
    return [f"rel_{p}_{s}_{u}_{f}_{x}" for p in REL_POLICIES.values() for s in REL_SUBSETS for u in REL_UNIVERSES
            for f in REL_FLOORS for x in ("n", "par", "typ")]  # fmt: skip


def relation_metric_defs() -> dict[str, tuple[list[str], list[str], list[str]]]:
    """`{metric: (num, den, support)}` for parent and type accuracy of every row."""
    d = {}
    for p in REL_POLICIES.values():
        for s in REL_SUBSETS:
            for u in REL_UNIVERSES:
                for f in REL_FLOORS:
                    k = f"rel_{p}_{s}_{u}_{f}"
                    d[f"{k}_parent_acc"] = ([f"{k}_par"], [f"{k}_n"], [f"{k}_n"])
                    d[f"{k}_type_acc"] = ([f"{k}_typ"], [f"{k}_par"], [f"{k}_par"])
    return d


def _gt_parents(gs: GTSupport, oid: int) -> dict[Any, str]:
    """GT parents of one rescan object: `{rescan objectId | FLOOR_ID: "on"|"in"}`."""
    return {(FLOOR_ID if p == FLOOR else int(p)): t for p, t in gs.parents.get(oid, {}).items()}


def score_relations(
    G: Graph,
    gt: GTLabels,
    gs_S: GTSupport,
    parent_kinds: Mapping[int, str],
    new_tp: Mapping[str, int],
    obs_truth: Mapping[str, tuple[int, str]] | None,
    prefix: str,
) -> dict[str, int]:
    """Relation counts of one pair under one relation policy (module docstring).

    Args:
        G: Predicted graph (after `update_graph` with the policy).
        gt: The pair's change GT (statuses, scored new objects).
        gs_S: `gt_support` of the rescan's 3DSSG relations.
        parent_kinds: `gt_parent_changes` (moved things -> same / swapped / gained / ...).
        new_tp: Predicted new node id -> GT new objectId (`match_new(...)["tp"]`).
        obs_truth: Token -> (GT rescan objectId, label), for predicted parents that are new nodes.
        prefix: Count prefix (`REL_POLICIES[policy]`).
    """
    truth = obs_truth or {}
    nodes = G.node_map()
    sup = G.support_parent()
    c = {k: 0 for k in relation_count_keys() if k.startswith(f"rel_{prefix}_")}

    def pred_parent(nid: str) -> tuple[Any, str | None]:
        """Predicted parent in rescan id space, and its type."""
        if nid not in sup:
            return None, None
        p, t = sup[nid]
        if p == FLOOR_ID:
            return FLOOR_ID, t
        oid = object_id_of(p)
        if oid is not None:
            return oid, t
        tok = nodes[p].obs_token if p in nodes else None
        hit = truth.get(tok or "")
        return (hit[0] if hit else None), t

    def add(nid: str, gp: dict[Any, str], subsets: list[str], associated: bool) -> None:
        pp, pt = pred_parent(nid)
        par = pp is not None and pp in gp
        typ = par and gp[pp] == pt
        floors = ["all"] + ([] if FLOOR_ID in gp else ["nf"])
        univs = ["gt"] + (["as"] if associated else [])
        for s in subsets:
            for u in univs:
                for f in floors:
                    k = f"rel_{prefix}_{s}_{u}_{f}"
                    c[f"{k}_n"] += 1
                    c[f"{k}_par"] += par
                    c[f"{k}_typ"] += typ

    for oid, o in sorted(gt.objects.items()):
        if o.level != "thing" or oid in gs_S.excluded:
            continue
        if not (o.status in ("unchanged", "moved") or (o.status == "dont_care" and o.reason == "small_move")):
            continue
        gp = _gt_parents(gs_S, oid)
        n = nodes.get(ref_node_id(oid))
        if not gp or n is None:
            continue
        subsets = ["a"]
        if o.status == "moved":
            subsets.append("b")
        kind = parent_kinds.get(oid)
        if kind == "swapped":
            subsets += ["c", "cb"]
        elif kind == "gained":
            subsets.append("cb")
        add(n.id, gp, subsets, n.state in ("unchanged", "moved"))
    for pid, gid in sorted(new_tp.items()):
        g = gt.new.get(gid)
        if g is None or g.level != "thing" or gid in gs_S.excluded:
            continue
        gp = _gt_parents(gs_S, gid)
        if gp:
            add(pid, gp, ["a"], True)
    return c
