# STATUS — what is finished, what is deferred

Handoff file between milestone sessions: each milestone is done in a fresh chat.
- **At the start of a milestone:** read this file after CLAUDE.md, then the open items for that milestone.
- **At the end:** move finished items to "Finished", add new deferred items under the milestone that has to handle them, and update "Current state".

DESIGN.md (the spec) and DECISIONS.md (the dated log, with the reasons) stay the source of truth. This file is the to-do view on top of them.

## Current state (2026-11-09)
- **M0 ✅** (commit `dd66381`), **M1 ✅** (commits `5ed8260` code, `b025428` docs), **M2 ✅** (commits `49b9b61` + `4577e4b` code, then the docs commit), **M3 ✅** (commit `6e71d0c` code, then the docs commit), **M4 ✅** (commit `9d67e8b` code, then the docs commit) and **M5 ✅** (commit `7cadc26` code, then the docs commit) are done.
- **Next: M6** — confidence + class priors (FULL), texture rendering, appearance residual (DINOv2): thin-node free-like votes, thin-object voxels.
- `origin/main` is at `b237be2` (M3 docs). The M4 and M5 commits are on local `main` only; not pushed as of 2026-11-09.

---

## Finished

### M5 (2026-11-07/09), acceptance passed — `uv run lsg m5 --run-id m5_final2` → `outputs/m5_final2/metrics.json` (dev B0 / B1 / B2 both modes + val B2 both modes); dev sweep in `outputs/m5_final/sweep_*`

**Acceptance results (dev):**

| Check | Result |
|---|---|
| Association strict ≥ B0 | oracle **90.9** vs 90.9%; perception **6.8** vs 6.8% (n = 44; ties) |
| Moved F1 ≥ B1 | oracle **86.7** vs 33.3%; perception **60.9** vs 45.5% (n = 15) |
| Relation table (a/b/c) R0 vs R1 on val | written, both modes (below) |
| JSON validates | 0 invalid graphs (`validate`, extended at M5) over 3 methods × 2 modes, dev + val |
| Determinism | dev `metrics` of `m5_dev2` and `m5_final2` identical; slow test runs B2 twice on one dev pair |

**Val B2** (110 pairs; val run **twice**, see DECISIONS M5: the nyu40 catch-all fix came from the first run):

| | moved P / R / F1 | removed P / R | new P / R | unch. acc / kept | false-del. / false-move | unknown | assoc strict | loc. median |
|---|---|---|---|---|---|---|---|---|
| B1 oracle (M3) | 38.2 / 24.7 / 30.0 | 66.0 / 45.2 | 85.0 / 59.6 | 80.9 / 92.9 | 9.5 / 2.4 | 31.6 | 91.4 | 0.060 |
| **B2 oracle** | 45.7 / 68.5 / **54.9** | 57.7 / 41.1 | 80.5 / 60.2 | 76.9 / 86.6 | 9.5 / 4.1 | 28.1 | 90.5 | 0.090 |
| B1 perception (M4) | 22.4 / 8.1 / 11.9 | 17.7 / 57.5 | 3.3 / 12.9 | 2.2 / 90.4 | 9.4 / 3.7 | 81.2 | 2.0 | 1.64 |
| **B2 perception** | 52.9 / 23.0 / **32.0** | 17.7 / 45.2 | 2.4 / 9.4 | **57.9** / 92.7 | 9.5 / 1.5 | 45.3 | 2.8 | 0.194 |

- Oracle moved recall by displacement: 55.5 / 75.9 / 89.5% (B1: 0 / 27.6 / 89.5).
- Oracle unchanged kept drops (92.9 → 86.6%), because the ICP test at τ_move = 0.10 m turns some unchanged objects into small "moves".

**Relations on val** (parent accuracy, GT universe; R0 stale / R1 geometric / update):

| | oracle | perception |
|---|---|---|
| (a) all | 92.0 / 79.5 / **92.4** (n = 994) | 94.6 / 80.2 / 90.7 (n = 966) |
| (b) moved | 94.0 / 65.5 / 86.3 (n = 168) | 94.0 / 75.6 / 83.9 (n = 168) |
| (c) swapped | 0 / 66.7 / 66.7 (n = 3) | same |
| (c-broad) swapped + gained | 0 / 20 / 20 (n = 10) | same |

- Type accuracy given the parent (oracle): 99.0 / 83.4 / 94.8%.
- "Associated only" rows and no-floor rows are in `metrics.json`.

**Code:**
- `change/association.py`: label / size / position costs, `absorb_confirmed` (stage 0), `associate` (Hungarian with gate).
- `change/classify.py`: `icp_move` (partial obs → reference points, two inits), `classify`, `Decision`.
- `change/update.py`: `update_graph(policy = update | stale | geometric)`, `node_points`.
- `change/baselines.py`: `b2`, `ref_node_points`; every method takes `ref_ctx=`.
- `graph/relations.py`: `still_supports`. `graph/model.validate`: `T_move`, `label_topk`, `n<k>` ⇔ `new`.
- `eval/relations.py`: `score_relations` + count keys / metric defs.
- `eval/metrics.py`: relation counts, `graph_invalid`, `free_ev_unchanged_removed`.
- `eval/harness.py`: passes `ref_ctx`, runs all 3 relation policies, scores relations, validates graphs, relation table in `format_summary`.
- `eval/m5_checks.py` + CLI `lsg m5 [--no-sweep] [--no-val]`; `--method B2`.

**Config:** new blocks `association` (incl. `absorb`, `nyu_fallback_exclude`), `classify`, `update`.

**Tests:**
- 145 fast (≈ 8–14 s) + 7 slow (≈ 2.3 min) pass.
- New: `test_association.py`, `test_classify.py`, `test_b2.py`, `test_update.py`, `test_relations_eval.py`, a `validate` test in `test_graph.py`, and the slow `test_m5_b2_both_modes_one_dev_pair`. `test_no_leak.py` imports the new modules.

**Docs:** DECISIONS "2026-11-07 — M5"; DESIGN §3 / §6.7 / §6.9 / §6.10 / §6.12 / §6.13 / §7.2 / CLI / §9; CODE_STRUCTURE; AI_USAGE.

### M4 (2026-11-07), acceptance passed — `uv run lsg m4` → `outputs/m4_final/metrics.json` + 3 overlay PNGs; val report `outputs/m4_val_B1`, `outputs/m4_val_B0` (all git `9d67e8b`, clean)

**Acceptance results:**

| Check | Result |
|---|---|
| Synthetic: box added → exactly one candidate, centroid error < 5 cm | `test_perception.py::test_box_added_gives_exactly_one_candidate`: 1 candidate, **2.1 cm** from the box's observable-surface centroid (xy 0.0 cm; 5.1 cm from the solid centre, all in z, DECISIONS M4) |
| Overlay PNGs on 3 dev frames | `outputs/m4_final/m4_overlay_*.png` (0ad2d3a1 frame 51, 19eda6f4 frame 81, e2847ff7 frame 339), inspected: masks cover the moved chair / bin, moved-to markers inside |
| Candidate recall on dev (B1, perception) | new **66.7%** (2 / 3; labelled 66.7%), moved-to **66.7%** (10 / 15; labelled 26.7%); 27.3 candidates per pair, 36.6% explained by a GT instance |
| Determinism | three `lsg m4` runs (two before the commit, one at `9d67e8b`) and two val runs per method → identical `metrics` (apart from output paths) |
| No-leak test on the real `perception/` code | `test_no_leak.py` (also checks that residual / voxels / cluster / labeler / source exist and are scanned; import closure includes them and `train_meta`) |

**B0 / B1 on val, perception vs oracle** (110 pairs, reported once, nothing tuned; oracle rows identical to M2 / M3):

| | moved P / R | removed P / R | new P / R | unch. acc / kept | false-del. | unknown | assoc strict |
|---|---|---|---|---|---|---|---|
| B0 oracle | 38.2 / 24.7 | 21.2 / 69.9 | 85.0 / 59.6 | 80.9 / 80.9 | 95.0 | 0 | 91.4 |
| B0 perception | 22.4 / 8.1 | 4.2 / 91.8 | 3.3 / 12.9 | 2.2 / 2.2 | 95.9 | 0 | 2.0 |
| B1 oracle | 38.2 / 24.7 | 66.0 / 45.2 | 85.0 / 59.6 | 80.9 / 92.9 | 9.5 | 31.6 | 91.4 |
| **B1 perception** | 22.4 / 8.1 | 17.7 / 57.5 | 3.3 / 12.9 | 2.2 / **90.4** | **9.4** | 81.2 | 2.0 |

- Val candidates: 7.8 per pair, 45.8% explained by a GT instance. Recall: new 27.5% (labelled 17.5%, n = 171), moved-to 46.4% (labelled 22.1%, n = 235).
- Door filter on val: 115 of 969 clusters dropped, in 35 pairs.
- Perception B0 / B1 are weak because of B0's matching, not the removal rule:
  - unchanged objects produce no candidate, so they are never matched;
  - same-label candidates pull far nodes into `moved` (localisation error median 1.64 m);
  - new candidates are absorbed by same-label nodes.

**Code:**
- `perception/residual.py`: `nearer_mask`, `nearer_points`, `SurfaceFilter`.
- `perception/voxels.py`: `VoxelEvidence`, `pack_keys` / `unpack_keys`.
- `perception/cluster.py`: `connected_components` (Chebyshev link), `cluster`, `Candidate`, `door_regions`, `drop_door_clusters`.
- `perception/labeler.py`: `CLIPLabeler` (cached), `StubLabeler`.
- `perception/source.py`: `PerceptionSource`.
- `train_meta.py`: `clip_vocab_table`, `perception_vocab`, `train_scans`, `label_table`.
- `obs/base.py`: `RefContext.nyu40`, `.object_nyu40()`, `.faces_of()`. `change/pipeline.prepare_pair` fills `nyu40`.
- `eval/metrics.py`: `perception_obs_truth`, `candidate_counts`, `gt_rescan_boxes`, `obs_node`; `score_pair(candidates=)`; metrics `cand_*`; `use_label=` on the §5 rule.
- `eval/harness.py`:
  - perception mode with `_TeeSource` (visibility-only oracle for `visible`);
  - `make_labeler`;
  - `outputs/<run>/candidates/*.json`;
  - per-pair `run.perception` stats, `clip_device`, CLIP crop / cache counts.
- `eval/m4_checks.py` + CLI `lsg m4 [--no-sweep]`.

**Config:** new blocks `train_meta` and `perception` (incl. `perception.clip`).

**Dependencies:** `torch` 2.5.1 + `torchvision` 0.20.1 (cu121 index), `open_clip_torch`.

**Tests:**
- 115 fast (≈ 10 s) + 6 slow (≈ 2 min) pass.
- New: `test_perception.py` (14), shared `tests/synth.py` (helpers moved out of `test_online.py`), a candidate-metrics toy test, the slow `test_m4_perception_b1_one_dev_pair` and `test_clip_labeler_real_model`.

**Docs:** DECISIONS "2026-11-07 — M4"; DESIGN §3 / §6.4 / §6.6 / §6.13 / CLI / §9; CODE_STRUCTURE; AI_USAGE.

### M3 (2026-11-06), acceptance passed — `uv run lsg eval --split dev_train --obs oracle --method B1` → `outputs/m3_dev_final/metrics.json`; val report `outputs/m3_final/metrics.json` (git `6e71d0c`, clean)

**Acceptance results:**

| Check | Result |
|---|---|
| Decorrelation unit test | `test_visibility.py::test_same_viewpoint_30x_counts_once` (30 identical poses → 1 counted); end-to-end in `test_online.py` (30 raw free → 1 counted → `unknown`; decorrelation off → `removed`) |
| Synthetic scene | `test_online.py`: box removed → ≥ 2 counted free votes → B1 `removed` at the end; removed but occluded → `unknown`; out of frustum → `unknown` (B0: `removed` in all three) |
| Dev: B1 false-deletion < B0 | **16.7%** (2 / 12) vs 100% (12 / 12) |
| Determinism | two dev B1 runs (`m3_dev_B1`, `m3_dev_B1b`) → identical `metrics` |

**B1 vs B0, oracle, val** (110 pairs, reported once, nothing tuned):

| | removed P / R | false-deletion | unchanged acc / kept | unknown rate |
|---|---|---|---|---|
| B0 (`m2_final`) | 21.2 / 69.9 | 95.0 | 80.9 / 80.9 | 0 |
| **B1** (`m3_final`) | **66.0 / 45.2** | **9.5** (n = 737) | 80.9 / **92.9** | 31.6 |

- Moved, new and association are unchanged from B0, because B1 uses the same matching.
- Evidence diagnostics on val: 60.3% of GT-removed objects get ≥ 2 counted free votes. So do 6.3% of GT-unchanged and 10.3% of absent-unlisted objects. 20.1% of absent-unlisted objects get a counted *seen* vote, i.e. they are seen in place: annotation gaps.
- ICP-rejected pairs (5): free evidence on unchanged 6.7% vs 6.3% elsewhere, so no fake free votes (open item closed).
- Dev diagnostic (DECISIONS M3): 4 / 34 unchanged objects with free evidence; erosion 0 / 1 / 2 px changes nothing, so `erode_px` stays 0. The 4 are a probable unlisted pillow move (§2.1) and three mixed-evidence "item"s.

**Code:**
- `change/visibility.py`: `frame_votes`, `FrameVote`, `erode_labels`, `Decorrelator`, `NodeVotes`, `has_free_evidence`, `confirmed_in_place`.
- `change/online.py`: `run_online` → `OnlineResult` (obs, votes, snapshots, timings); `tentative_state` / `tentative_graph`.
- `change/baselines.py`: `b1`; shared `_match_b0`; every method takes `votes=`; `METHODS = {"B0", "B1"}`.
- `change/pipeline.py`: `collect_observations` removed.
- `eval/harness.py`: runs `run_online` with one reference `Renderer` per group; snapshots behind `snapshots=`; timings under `run.per_pair[].time_s.online_*`.
- `eval/metrics.py`: `score_pair(votes=)` → `diag_*` counts; metrics `free_ev_unchanged / _removed / _absent`, `seen_ev_absent`.
- CLI: `lsg eval --snapshots`, `lsg run --no-snapshots`, `--method B0|B1`.

**Config:** `visibility.vote_frac`, `visibility.erode_px`; new blocks `decorrelation`, `states`, `online`.

**Tests:** 100 fast (≈ 8 s) + 4 slow (≈ 2 min) pass. New: `test_visibility.py`, `test_online.py`, `test_b1.py`, a diagnostics test in `test_metrics.py`, the slow `test_m3_oracle_b1_one_dev_pair`. `test_no_leak.py` also imports `change.online` / `change.visibility`.

**Docs:** DECISIONS "2026-11-06 — M3"; DESIGN §3 / §4 / §6.4 / §6.7 / §6.8 / CLI / §9; CODE_STRUCTURE; AI_USAGE.

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

### M6 — confidence, texture rendering, appearance
- **Thin / soft removed objects get `seen`, not `free`.** Example: dev pillow r50 on e2847ff7 (GT removed): 23 counted seen, 0 free. On val, 39.7% of GT-removed objects have < 2 counted free votes. Some are unseen; some are where depth cannot tell, which is the case the appearance residual targets. Measure with `free_ev_removed` before and after.
- Log-odds confidence (§6.11) builds on `NodeVotes` counted seen / free. The per-visit clip and class priors are still to do. `Node.confidence` is still None.
- OBJ ↔ PLY vertex/face correspondence is still **unverified** (DESIGN §2.1). Check it before using textures.
- Textures are in `data/<ref>/mesh.refined.v2.obj|.mtl|_0.png` for all 72 references.
- **GPU availability is intermittent.** The GTX 1650 is a hybrid-laptop dGPU. It was powered down at the start of M4 and was unavailable for the val run (CLIP fell back to CPU: 3,912 crops, ~6 min for B1 val incl. everything). DINOv2 patch features per frame need the GPU.
  - Check `torch.cuda.is_available()` before long runs; the `run` block records `clip_device`.
  - Consider a Windows power setting / NVIDIA control panel "High-performance GPU" for python.exe.
- Thin removed objects: perception mode gets the same votes as oracle (B1 removal identical), so the appearance residual helps both modes equally.
- **GT-unchanged objects with free evidence are deleted** (from M5). Dev: 4 / 34 unchanged objects have ≥ 2 counted free votes; B2 removes 4 / 4 in perception (no candidate matches them) and 1 / 4 in oracle. They are the probable unlisted pillow move and three mixed-evidence "item"s (DECISIONS M3). The confirmed-in-place rule cannot protect them (free > 0). The log-odds confidence should weigh their many seen votes against the few free ones; measure with `free_ev_unchanged_removed`.
- **Perception association stays near B0** (dev 6.8%, val 2.8%): unchanged objects produce no candidates, and confirmed nodes absorb few (dev 10). This is expected; the change metrics are what matter. Thin-object voxels (appearance output ii) may add candidates.
- **False candidates** (M4, still open): dev 63% / val 54% unexplained (plants, black TV screens, clutter). Absorption by confirmed nodes now removes some (dev perception: 10 absorbed). Perception new P on val is still 2.4% (n = 667 predicted).

### M7 — ablations
- Switches already in place: `decorrelation.enabled` (on / off), `visibility.tau_free` (sweep), `visibility.erode_px`. **Immediate vs deferred commitment** is not implemented yet. Idea: a method variant that commits `removed` at the first frame with ≥ k_free counted free votes, and keeps it even if the node is matched later.
- `--frac/--seed` already work (`select_frames`, a contiguous chunk with its start drawn from `[seed, crc32(rescan)]`). The alignment cache key includes the frame ids, so each chunk gets its own ICP pre-pass, as §7.3 requires.
- **Tier-2 download done (2026-11-06):** all 63 relchange scans (23 references + 40 rescans) have semseg, PLY and `sequence.zip`. That is 189 files, 4.0 GB, downloaded in 16 min. Every zip passes `testzip`, and every scan loads with `load_scan` + `open_sequence`. Textures for the 23 references were already there.
  - `configs/default.yaml` `m1.meta_only_splits` still lists the relchange split as metadata-only. Now that meshes exist, it could get the full GT pass (`lsg m1`). Optional; the relation eval at M5/M7 does not depend on it.
- **Relation eval on the relchange set** (§7.2(c), n = 18 swapped / 65 c-broad): `lsg eval --split eval_relchange_train --obs perception,oracle --method B2` writes the relation table. Never tuned on.
- **"Children follow parent"** ablation (§6.12 optional rule) is not implemented yet. Idea: an unobserved thing (no counted votes) whose parent anchor moved gets the parent's `T_move`.
- **`update.keep_supported_parent`** on / off (M5) is a cheap extra ablation row.
- **Relation type accuracy** (M1, still open): R1 type accuracy given the parent is 83.4% on val (oracle), mostly `on` vs `in` on shelves / cabinets. Dev has only 7 non-floor parent cases; adding train pairs to dev needs a download + DECISIONS entry.
- **Anchors carry no support edge** (M1, provisional): anchors on anchors are not represented. `update_graph` did not need it at M5.
- **τ_move / τ_assoc sweeps:** oracle unchanged kept is 86.6% (B1 92.9%) because τ_move = 0.10 m turns ICP noise on some unchanged objects into small moves; perception trades moved F1 against localisation via τ_assoc (dev sweep, DECISIONS M5). Both are §8-style sweeps (dev for tuning, val for reporting).
- **Perception same-label swaps** (M5 dev diagnosis): moved pillows take the candidates of *new* pillows, and a "bag" candidate with "pillow" in its top-3 is matched to a moved pillow. Localisation errors of 1.1–1.7 m on these.
- **Perception small moves become `removed`** (dev: tv stand, chairs, table moved 0.16–0.37 m): the old place gets free votes, but the new-place candidate is cut by `surf_dist` or merged. Visible in the moved recall [0.1, 0.3) bucket (perception val 14.5%).
- **Door filter cost** and **`min_voxels`** (M4 items): revisit on dev if moved recall near doors matters (moved rack on 0ad2d3a1).

### M8 — write-up
- State in the limitations: **12 door swings in val are articulation changes, out of scope** (not scored; doors are structure, §3).
- State that **val was run twice at M5** (the nyu40 catch-all fix and the keep-parent rule were triggered by the first val run; DECISIONS M5), and report the first run's numbers too.
- R0 (stale relations) is a strong baseline on 3RScan: most moved things keep their parent (val (b) R0 94.0% vs update 86.3%); (c) swapped has n = 3 on val.
- `next_to` (AABB gap ≤ 0.15 m) is provisional and exported only.
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
- **AI_USAGE.md:** "Erik: (fill in)" is still open for the M0, M1, M2, M3, M4 and M5 rows.
- **DECISIONS 2026-11-05 "Milestones re-scoped to M1–M8"** is still marked "provisional — Erik to confirm".
