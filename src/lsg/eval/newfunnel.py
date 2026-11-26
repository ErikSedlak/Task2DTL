"""Where GT new objects are lost in perception (CHANGE 3 Part E; `lsg newfunnel --run <dir>`).

From a run with per-object details (`details/<mode>/`) and candidate files (`candidates/`), every GT new object
of a pair gets one fate (first that applies):
- `not visible`: the oracle of the same run saw no instance of it (neither a correct new node nor an observation
  taken by a reference node);
- `no candidate`: no perception observation is assigned to it (`perception_obs_truth`, the candidate file's `gt`);
- `new node, correct`: a new node scored TP (labelled matching rule);
- `new node, scored fp`: its observation became a new node, but the matching rule did not match it (label / box);
- `taken by a reference node -> <state>`: its observation was associated to a reference node;
- `below p_new_min`: its observation was not emitted (candidate confidence / detector newness);
- `other`: absorbed by a confirmed node or dropped.

Also counts the observations by class (`new` / `ref` / `none`: explains a GT new object, a reference object's
rescan instance, nothing) and emitted or not. Lives in `eval/` (reads GT-derived details).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def new_funnel(run_dir: str | Path) -> dict[str, Any]:
    """Fate counts of GT new objects (perception) and observation classes of a run."""
    run_dir = Path(run_dir)
    seen_oracle: set[tuple[str, str, int]] = set()
    have_oracle = (run_dir / "details" / "oracle").is_dir()
    for g in sorted((run_dir / "details" / "oracle").glob("*.json")) if have_oracle else []:
        e = _load(g)
        gt_new = set(e["gt_new_fn"]) | {n["gt_id"] for n in e["new"] if n["outcome"] == "tp"}
        vis = {n["gt_id"] for n in e["new"] if n["outcome"] == "tp"}
        vis |= {o["obs_gt"][0] for o in e["objects"] if o.get("obs_gt")} & gt_new
        seen_oracle |= {(e["ref"], e["rescan"], int(i)) for i in vis}
    fate: Counter = Counter()
    cls: Counter = Counter()
    n_gt = 0
    for g in sorted((run_dir / "details" / "perception").glob("*.json")):
        e = _load(g)
        cf = run_dir / "candidates" / f"{e['ref']}_{e['rescan']}.json"
        cands = {c["token"]: c for c in _load(cf)["candidates"]} if cf.is_file() else {}
        for c in e.get("candidates", []):
            cls[f"{c.get('cls', '?')}/{'emitted' if cands.get(c['token'], {}).get('emits_new') else 'not emitted'}"] += 1
        taken = {o["obs_token"]: o for o in e["objects"] if o.get("obs_token")}
        new_by_tok = {n["obs_token"]: n for n in e["new"] if n.get("obs_token")}
        gt_tp = {n["gt_id"] for n in e["new"] if n["outcome"] == "tp"}
        for gid in sorted(set(e["gt_new_fn"]) | gt_tp):
            n_gt += 1
            toks = [t for t, c in cands.items() if c.get("gt") and int(c["gt"][0]) == int(gid)]
            if have_oracle and (e["ref"], e["rescan"], int(gid)) not in seen_oracle and gid not in gt_tp:
                fate["not visible"] += 1
            elif gid in gt_tp:
                fate["new node, correct"] += 1
            elif not toks:
                fate["no candidate"] += 1
            else:
                t = max(toks, key=lambda x: cands[x].get("n_voxels", 0))
                if t in new_by_tok:
                    fate["new node, scored fp"] += 1
                elif t in taken:
                    fate[f"taken by a reference node -> {taken[t]['pred_state']}"] += 1
                elif not cands[t].get("emits_new", True):
                    fate["below p_new_min"] += 1
                else:
                    fate["other"] += 1
    summary = {"gt_new": n_gt, **dict(fate.most_common())}
    return {"run": str(run_dir), "summary": summary, "observations": dict(sorted(cls.items()))}


def format_funnel(res: dict[str, Any]) -> str:
    """Plain-text funnel."""
    L = [f"GT new objects (perception), {res['run']}: {res['summary']['gt_new']}"]
    L += [f"  {v:5d}  {k}" for k, v in res["summary"].items() if k != "gt_new"]
    L.append("observations by class / emitted: " + ", ".join(f"{k} {v}" for k, v in res["observations"].items()))
    return "\n".join(L)
