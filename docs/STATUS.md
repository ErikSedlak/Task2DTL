# STATUS — what is finished, what is deferred

Handoff file between milestone sessions: each milestone is done in a fresh chat.
- **At the start of a milestone:** read this file after CLAUDE.md, then the open items for that milestone.
- **At the end:** move finished items to "Finished", add new deferred items under the milestone that has to handle them, and update "Current state".

DESIGN.md (the spec) and DECISIONS.md (the dated log, with the reasons) stay the source of truth. This file is the to-do view on top of them.

## Current state (2026-11-06)
- **M0 ✅** (commit `dd66381`) and **M1 ✅** (commits `5ed8260` code, `b025428` docs) are done.
- **Next: M2** — `obs` interface + `OracleSource`, the eval harness, B0 (oracle), and the no-leak import test.
- Both M1 commits are on local `main`. Not pushed as of 2026-11-06.

---

## Finished

### M1 (2026-11-05), acceptance passed — `uv run lsg m1` → `outputs/m1_final/metrics.json`

**Acceptance results:**

| Check | Result |
|---|---|
| Val counts vs. DESIGN §2.2 | 390 rigid / 96 removed / 224 new raw (171 new scored) |
| Displacement histogram | `outputs/m1_final/gt_displacement_hist.png`; τ_eval_move = 0.10 m → 134 of 370 val moves are `dont_care` |
| Rule parent accuracy (reference scans) | dev 92.9% (n = 28, tuned); val 85.2% (n = 575, reported once); type accuracy given the correct parent 100% / 85.3% |
| `splits/download_textures.txt` | 72 references |

**Code:**
- `src/lsg/graph/model.py`: `Graph` / `Node` / `Edge` (DESIGN §4 JSON), `save` / `load`, `validate()`.
- `src/lsg/graph/relations.py`:
  - `gt_support` (§5 predicate mapping), `pick_parent`, `estimate_floor_z`;
  - `infer_support` (contact / containment / floor, plus `support_min_height`), `aabb_gap`.
- `src/lsg/graph/build.py`: `build_reference_graph(scan, objects, relations, mode="gt"|"geometric", cfg)`.
- `src/lsg/eval/gt.py`:
  - `gt_changes` → `GTLabels`: per-object status + reason, displacement, `M_ref`, `centroid_new`, size class; new objects in the reference frame;
  - `gt_parent_changes`.
- `src/lsg/eval/gt_stats.py`: `run_m1` (counts, histogram, rule sanity, texture-list check).
- `src/lsg/io/splits.py`: `reference_ids`, `write_texture_list`.
- `Scan.semseg_labels` = O_R / O_S for §5.
- CLI: `lsg m1`, `lsg splits textures`, `lsg graph --scan R --mode gt|geometric`.

**Config:** blocks `relations` (tuned on dev refs), `eval` (`tau_eval_move`, `small_max_extent`, `disp_buckets`), `splits`, `m1`.

**Tests:** `test_relations.py`, `test_graph.py`, `test_gt.py`, plus the slow val-count test. Results: 47 fast tests pass, 2 slow tests pass.

**Data:**
- Textures for the 72 references are downloaded: `mesh.refined.v2.obj`, `.mtl`, `_0.png`; 216 files, 1.1 GB.
- All 98 dev + val reference graphs validate, in both modes.

**Checked:**
- The anchor label list (DECISIONS 2026-11-05) is correct. Shelf, cabinet, table, sofa, counter, desk and bed are anchors.
- nyu40 39 "otherfurniture" (tv stand, bench, rack) are things. They can still be support parents.

**Docs:** DESIGN §2.1 / §2.2 / §2.3 / §5 / §6 / §9; DECISIONS "2026-11-05 — M1"; CODE_STRUCTURE; AI_USAGE.

### M0 (2026-11-05), acceptance passed — see DESIGN §9 and DECISIONS "M0".

---

## Open items by milestone

### M2 — obs interface, oracle, eval harness, B0

**Reuse from M1:**
- `eval.gt.gt_changes` gives the statuses. Scored = things + anchors with status ≠ `dont_care`.
- `GTLabels.by_status("absent_unlisted")` is the denominator of the false-deletion rate.
- `GTObject.centroid_new` (raw `T_align`) gives the localisation error.
- `GTObject.size` and `GTNew.size` give the size buckets.
- `displacement` gives the §7.1 buckets. Bare-int rigid entries have `None`; there are none in val, so handle `None` but don't expect it.

**Data facts:**
- Duplicate `rigid` ids exist (7 entries in val, identical transforms). `gt_changes` deduplicates them.
- `OracleSource` must deduplicate them too if it reads `rigid` (it shouldn't need to).

**Required work:**
- **No-leak import test** (DESIGN §9 M2): `perception/` imports neither `oracle/` nor `eval/`.
- `graph/` must also stay free of `eval/` and `oracle/` imports, because perception may use it. It is clean now; include it in the test.
- Perception matching rule (§5): label → nyu40 comes from 3DSSG `objects.json`, already loaded by `lsg.io.dssg.load_objects`.

**Expected runtime:** loading all val GT takes ≈ 70 s (`lsg m1`), well inside the < 15 min budget for `lsg eval`.

### M3 — online loop, B1
- No M1 carry-overs beyond the spec.
- `graph.relations.estimate_floor_z` is available wherever heights are needed (CLAUDE.md: heights are relative to the floor).

### M4 — perception
- **Door swings → false "new" candidates.**
  - In val, 12 doors are in `rigid`. Nearly all rotate 4°–120° about the vertical axis (hinge swings).
  - Structure is rendered from the *reference* mesh. An opened door therefore leaves "nearer" residuals, and the §3 step-2 structure filter (walls / ceiling / floor only) does not remove them.
  - Fix: add reference **door** surfaces (nyu40 8), probably all structure classes, to the structure filter, or drop candidates overlapping a reference door's swing region.
  - Check the effect on dev, then log it in DECISIONS.
- `perception/` may import `lsg.graph` (relations / build). It must not import `eval/` or `oracle/`.

### M5 — association, moved test, update_graph, relation eval
- **Decide the §7.2(c) definition** ("things whose GT parent changed"). Breakdown (DESIGN §2.2, DECISIONS M1):

  | Split | Gained | Lost | Swapped | Structure-only |
  |---|---|---|---|---|
  | val (12) | 7 | 1 | 4 | +4 |
  | relchange (73) | 47 | 4 | 18 | 4 |

  "Gained" mostly means the reference 3DSSG annotation is missing. `eval.gt.parent_change_kind` already computes the categories.
- **Relation type accuracy is weak** on val references: 85.3%, and 75.5% without floor parents. The errors are mostly `on` vs `in` for objects on shelves and cabinets.
  - Tuning is allowed on dev only, and dev has just 7 non-floor parent cases.
  - If relations matter, consider adding more *train* pairs to dev. That needs a download plus a DECISIONS entry. Never tune on `eval_*` (CLAUDE.md hard rule 4).
- **Anchors carry no support edge** (provisional, DECISIONS M1). Anchors on anchors (a shelf on a cabinet) are not represented. Revisit if `update_graph`'s "parent moved or removed" rule needs it.
- `update_graph` re-runs `infer_support` for moved and new things:
  - build `SupportCandidate`s from reference points with `T_move` applied (moved) or from `ObsInstance.points` (new);
  - remove deleted nodes from the candidates.
- G_ref for runs = `build_reference_graph(..., mode="gt")`. Baseline R1 = `mode="geometric"`.
- `next_to` rule (AABB gap ≤ 0.15 m) is provisional and exported only.

### M6 — confidence, texture rendering, appearance
- OBJ ↔ PLY vertex/face correspondence is still **unverified** (DESIGN §2.1). Check it before using textures.
- Textures are in `data/<ref>/mesh.refined.v2.obj|.mtl|_0.png` for all 72 references.

### M7 — ablations
- Relchange set (`eval_relchange_train.txt`) needs the **tier-2 download** (63 scans: semseg, PLY, sequence; ≈ 3 GB; `splits/download_tier2.txt`). **Not done yet.**
  - Its 23 references already have their textures, but no meshes or sequences.
  - Start it early: any session can run it in the background with the command in DESIGN §2.2.

### M8 — write-up
- State in the limitations: **12 door swings in val are articulation changes, out of scope** (not scored; doors are structure, §3).
- Mention the GT findings:
  - 36% of val moves are < 10 cm;
  - "new" = 224 raw / 171 scored;
  - most "parent changes" are annotation gaps.

---

## Decisions needed from Erik
- **Doors / articulation.** Current: unscored (structure, §3/§5), recorded as out of scope. Claude's recommendation:
  1. keep doors out of the change metrics;
  2. add the M4 door filter;
  3. *optionally*, after M7, add a separate "door open/closed changed" diagnostic on door nodes (n = 12 in val), not mixed into moved F1.

  Erik has not decided on point 3 yet.
- **AI_USAGE.md:** "Erik: (fill in)" is still open for the M0 and M1 rows.
- **DECISIONS 2026-11-05 "Milestones re-scoped to M1–M8"** is still marked "provisional — Erik to confirm".
