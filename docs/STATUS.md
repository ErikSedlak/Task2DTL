# STATUS — what is finished, what is deferred

Handoff file between milestone sessions: each milestone is done in a fresh chat.
- **At the start of a milestone:** read this file after CLAUDE.md, then the open items for that milestone.
- **At the end:** move finished items to "Finished", add new deferred items under the milestone that has to handle them, and update "Current state".

DESIGN.md (the spec) and DECISIONS.md (the dated log, with the reasons) stay the source of truth. This file is the to-do view on top of them.

## Current state (2026-11-06)
- **M0 ✅** (commit `dd66381`), **M1 ✅** (commits `5ed8260` code, `b025428` docs) and **M2 ✅** (commits `49b9b61` + `4577e4b` code, then the docs commit) are done.
- **Next: M3** — online loop (`change.online`, `visibility`, `Decorrelator`), deferred commitment, snapshots, B1 (oracle).
- All commits are on local `main`. Not pushed as of 2026-11-06.

---

## Finished

### M2 (2026-11-06), acceptance passed — `uv run lsg eval --split eval_val --obs oracle --method B0` → `outputs/m2_final/metrics.json` (git `4577e4b`, clean)

**Acceptance results:**

| Check | Result |
|---|---|
| `lsg eval` on `eval_val` < 15 min | **69 s** cold (alignment cache empty), 44 s cached; 4 workers |
| Metric unit tests, 5-object toy case incl. perception rule | `tests/test_metrics.py` (11 tests) |
| No-leak test | `tests/test_no_leak.py`: AST import scan of `perception/ obs/ change/ graph/ geom/ io/` + `classes/config/runinfo`; fresh-interpreter import closure; rescan-annotation reads in `perception/`; planted-violation self-tests |
| Determinism (§7.5) | two cold runs + one cached run → identical `metrics` block |

**B0 oracle on val** (110 pairs; 95% bootstrap CI in metrics.json):

| moved P / R / F1 | removed P / R | new P / R / F1 | unchanged acc | false-deletion | false-move | assoc strict / same-label |
|---|---|---|---|---|---|---|
| 38.2 / 24.7 / 30.0 | 21.2 / 69.9 | 85.0 / 59.6 / 70.1 | 80.9 | **95.0** (n = 737) | 2.4 | 91.4 / 97.4 (n = 1,472) |

- Moved recall by displacement: 0% in [0.1, 0.3) m (below B0's 0.5 m), 27.6% in [0.3, 1) m, 89.5% at ≥ 1 m. Localisation error median 6.0 cm, p90 55 cm (n = 58).
- GT support: 235 moved / 73 removed / 171 new / 1,401 unchanged / 737 absent-unlisted; this matches DESIGN §2.2.
- Dev (B1's acceptance compares against these): false-deletion rate **100%** (12 / 12); moved F1 33.3%; association 90.9% strict (n = 44).

**Code:**
- `obs/base.py`: `ObsInstance`, `ObsSource` protocol, `RefContext`, `SeqInfo`, `RefRender`, `PackedMasks`.
- `oracle/source.py`: `OracleSource(rescan_scan, objects_S, cfg, seed)`; `token_map()` is for eval only.
- `change/pipeline.py`: `select_frames`, `refined_alignment` (cache in `cache/align/`), `prepare_pair`, `collect_observations`.
- `change/baselines.py`: `b0`, `greedy_label_match`, `add_new_nodes`, `METHODS = {"B0": b0}`.
- `eval/metrics.py`: `match_new`, `score_pair`, `aggregate`, `metric_defs`, `aabb_iou`, buckets.
- `eval/harness.py`: `run_eval`, `run_split`, `format_summary`.
- CLI: `lsg eval`, `lsg run`.
- Other changes:
  - `GTObject.aabb_new_min/max`;
  - `dssg.label_to_nyu40`;
  - `refine_from_sequence(frame_ids=)`;
  - `config.cache_root`;
  - `write_metrics(extra=)` writes the non-deterministic `run` block.
- `perception/__init__.py` is a stub.

**Config:** new blocks `oracle`, `baselines.b0`; `eval` gains the match thresholds, bootstrap settings, `workers` and `save_graphs`; new `cache_root`.

**Tests:** 81 fast (≈ 7 s) + 3 slow (≈ 1.5 min) pass.

**Docs:**
- DECISIONS "2026-11-06 — M2": Erik's scoring definitions are logged there;
- DESIGN §6 / §7.1 / §7.5 / §9;
- CODE_STRUCTURE; AI_USAGE.

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

### M3 — online loop, B1
- **Plug into M2:**
  - `change.online.run_online` replaces `change.pipeline.collect_observations`. Pass a real `obs.RefRender` to `ObsSource.observe` (it is `None` now).
  - Reuse `select_frames`, `prepare_pair` and the alignment cache. Add `"B1"` to `change.baselines.METHODS`; the harness picks methods from there.
  - B1 reuses B0's matching (`greedy_label_match`).
- **B1 acceptance baseline:** "B1 false-deletion rate < B0's on dev". B0 dev = **100%** (12 / 12), val = 95.0%.
- **Determinism:** keep floats that derive from ICP out of the `metrics` block, or round them (see the localisation-error rounding, DECISIONS M2). Reruns use the cached `T_align`.
- **ICP rejections:** refinement was rejected on 5 / 110 val pairs, which keep the raw `T_align` (2.9–3.8 cm residual). Check whether these pairs produce fake `free` votes; the per-pair `align` info is in the `run` block of metrics.json.
- `RefContext.floor_z` (= `graph.relations.estimate_floor_z`) is available wherever heights are needed (CLAUDE.md: heights are relative to the floor).

### M4 — perception
- **Door swings → false "new" candidates.**
  - In val, 12 doors are in `rigid`. Nearly all rotate 4°–120° about the vertical axis (hinge swings).
  - Structure is rendered from the *reference* mesh. An opened door therefore leaves "nearer" residuals, and the §3 step-2 structure filter (walls / ceiling / floor only) does not remove them.
  - Fix: add reference **door** surfaces (nyu40 8), probably all structure classes, to the structure filter, or drop candidates overlapping a reference door's swing region.
  - Check the effect on dev, then log it in DECISIONS.
- `perception/` may import `lsg.graph` (relations / build). It must not import `eval/` or `oracle/`.
- `PerceptionSource` implements `obs.ObsSource`, with tokens `c<k>`. `test_no_leak.py` already scans `perception/` for imports **and** for rescan-annotation reads (`load_scan`, `.rigid`, `"semseg"`, …). Keep it green.
- **Harness work for perception mode** (`eval/harness.py` raises `NotImplementedError` for it now):
  - `obs_truth`: token → GT rescan instance, using the §5 rule against GT new instances **and** the rescan instances of reference objects (`GTObject.centroid_new / aabb_new_*`);
  - `visible`: GT-visible rescan ids for the association denominator; run `OracleSource`'s visibility pass inside `eval/`;
  - `label_nyu = dssg.label_to_nyu40(root)` for `match_new`.
- `dssg.label_to_nyu40` uses **all** of `objects.json`. That is fine for eval's class mapping. The method's CLIP vocabulary must still come from train metadata only (§2.3).

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
- G_ref for runs = `build_reference_graph(..., mode="gt")` (done in `prepare_pair`). Baseline R1 = `mode="geometric"`.
- B0 keeps the reference relations unchanged; `update_graph` should be added on top.
- Methods must keep setting `obs_token` on associated reference nodes: the association metric reads it.
- `next_to` rule (AABB gap ≤ 0.15 m) is provisional and exported only.

### M6 — confidence, texture rendering, appearance
- OBJ ↔ PLY vertex/face correspondence is still **unverified** (DESIGN §2.1). Check it before using textures.
- Textures are in `data/<ref>/mesh.refined.v2.obj|.mtl|_0.png` for all 72 references.

### M7 — ablations
- `--frac/--seed` already work (`select_frames`, a contiguous chunk with its start drawn from `[seed, crc32(rescan)]`). The alignment cache key includes the frame ids, so each chunk gets its own ICP pre-pass, as §7.3 requires.
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
- **AI_USAGE.md:** "Erik: (fill in)" is still open for the M0, M1 and M2 rows.
- **DECISIONS 2026-11-05 "Milestones re-scoped to M1–M8"** is still marked "provisional — Erik to confirm".
