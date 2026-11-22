# STATUS — what is finished, what is deferred

Handoff file between milestone sessions: each milestone is done in a fresh chat.
- **At the start of a milestone:** read this file after CLAUDE.md, then the open items for that milestone.
- **At the end:** move finished items to "Finished", add new deferred items under the milestone that has to handle them, and update "Current state".

DESIGN.md (the spec) and DECISIONS.md (the dated log, with the reasons) stay the source of truth. This file is the to-do view on top of them.

## Current state (2026-11-22)
- **The final method is CHANGE 1 + 2 + 3** (DESIGN, DECISIONS). **The final val run is done:** `outputs/chg3_final` (`lsg m7 --run-id chg3_final`, git 4a4ba76, acceptance all PASS; DECISIONS 2026-11-22). REPORT, README, slides and figures describe it. Val FULL (oracle | perception): removed P / R 58.3 / 28.8 | 24.3 / 37.0, false deletion 9.1 | 7.2, unchanged kept 90.9 | 93.4, moved F1 56.5 | 42.5, new P / R 67.5 / 66.7 | 3.1 / 3.5. The detector (CHANGE 3 Part E) is the ablation row `FULL+detector` (lost its dev_tune rule).
- **Erik, to finish:**
  1. Label the audit: `outputs/chg3_audit/audit.csv` (60 rows) and `audit_veto.csv` (11 rows), columns `verdict` (`gt_missing | method_error | unclear`) and `note`, looking at `outputs/chg3_audit/panels/`. Then Claude runs `uv run lsg audit summarize --run-id chg3_audit` and fills REPORT §6 and slide 11.
  2. Acceptance check from a clean clone (README §3; expected result updated for the final method).
  3. The "Erik: (fill in)" lines in `docs/AI_USAGE.md`; push (nothing pushed since `b237be2`).
  4. Open choices: keep or remove the unused `segment-anything` dependency (SAM 1, Part N variant; Part E uses SAM 2 via `transformers`); delete the old SAM timing cache `cache/sam/1ec391361533/`.

## State of 2026-11-20
- **CHANGE 1 + 2 is the final method** (Erik, 2026-11-20). **CHANGE 3 proposed** (`docs/CHANGE_online_graph.md`, DECISIONS 2026-11-20): tuning set `dev_tune` (33 train pairs, download ≈ 2–3 GB), move needs old-place evidence, incremental graph during the walk, confidence policy from sensor statistics, class-agnostic new P / R. All §8 questions answered. **Part 0 done** (DECISIONS 2026-11-20 "CHANGE 3 Part 0"): `splits/dev_tune.txt` (33 pairs), `download_tier3.txt` (60 scans), 102 texture references, vocabulary without dev_tune rooms, flat start value 0.9 / p_del 0.2, hard rule 4 = tune on dev_tune.
  - **Done 2026-11-20:** tier-3 download (60 scans + 30 textures, checked), regression `FULL-m7` on dev_train = `m7_final` (0 differences), baseline `outputs/chg3_dev_baseline/FULL` (DECISIONS: oracle false move 18.9%, moved P 39.5%; perception new P 2.4%). **Part A done** (default `association.old_place` = sweep row `a_u_c02`, Erik; DECISIONS "CHANGE 3 Part A"): dev_tune oracle false move 18.9 → 9.2%, unchanged kept 78.8 → 86.4%, moved F1 52.5 → 55.1; perception 13.3 → 7.1%, 82.7 → 91.2%, 39.6 → 44.9. **Part C done** (DECISIONS "CHANGE 3 Part C"): learned vote weights L_hit 0.63 / L_miss 1.47, `p_del` 0.4, candidate cluster votes, `p_new_min` 0.514996; dev_tune vs Part A: perception new P 2.5 → 5.6 (R 11.2 → 7.0), moved F1 +0.9 / +2.0, false move +2.0 / +2.6, false deletion oracle 2.6 → 3.1. **Part N done** (new objects, DECISIONS "CHANGE 3 Part N"): geometric co-segmentation of candidate voxels, re-estimated candidate weights, `p_new_min` 0.517493; explain-away off and identity check dropped by their rules / Erik; dev_tune perception new P / R 5.6 / 7.0 → 12.2 / 5.6 (F1 6.2 → 7.7; N5 thin-object confidence on, noise rejection off by its rule), removed R 24.1 → 33.7; oracle unchanged. Open from Part N: Erik to confirm the identity-audit verdicts (`outputs/audit_identity/cases.json`); the SAM dependency is unused by the default (keep as a variant or remove). **Part B done** (DECISIONS "CHANGE 3 Part B"): the method runs at every snapshot during the walk (`peek()`, track ids, `lsg anytime` curve + stability, viz with mid-walk moves); online cumulative alignment is the default (Erik, despite the perception cost: dev_tune moved F1 46.9 → 43.0, false deletion 4.1 → 5.6, new F1 7.7 → 3.3; oracle slightly better); `FULL-offlinealign` row. Identity audit verdicts final (Erik: 00, 02 same; 03 unclear). **Part E done** (DECISIONS "CHANGE 3 Part E"; Erik asked for the stretch-goal detector after the stopped val run showed 6 of 171 new objects found): Grounding DINO + SAM 2 detections fused into 3D tracks; on dev_tune perception new F1 3.3 → 15.6 but removed F1 40.9 → 32.5, so the pre-written rule keeps the residual candidates and the detector is ablation row `FULL+detector`. **Part D code done** (c4bd661). **Next:** the final `lsg m7 --run-id chg3_final` (after the val detection cache), then REPORT / README / slides on its numbers, `lsg audit sample` for Erik's labels.
- **CHANGE 2 (no class prior; candidate confidence with see-through votes) is implemented and ON by default** (`confidence`, `confidence.candidates`, docs/CHANGE_confidence_unified.md, DECISIONS 2026-11-20). `FULL-m7` (`ablations.M7_OVERRIDES`) is the M7 method and reproduces `m7_final` bit for bit on dev; REPORT, slides, the README headline table and `outputs/m7_final` describe it.
  - **Val FULL measured** (`outputs/chg2_val/FULL`, DECISIONS 2026-11-20): new R 8.8 → 14.6%, moved F1 31.7 → 37.4, false deletion 7.7 → 6.9%, but unchanged kept 95.3 → 88.8% (outside the CI) and false move 1.6 → 6.4%, new P 2.0 → 1.0%. Erik decides whether the defaults stay.
  - Still open on val: comparison rows `FULL-candconf`, `FULL-candconf-free`, `FULL+classprior`, `FULL-relaxed` (`lsg ablate`, > 1 min each); `lsg newcurve --run <FULL run>` for new P / R over `p_new_min`. Judge against the change document §6.4.
  - The M8 audit (step 1) forces `M7_OVERRIDES` by itself; the README expected result (step 3) is regenerated for the new default (perception: 11 `new` nodes).

## State of 2026-11-19
- **CHANGE 1 (relaxed candidate rules) is implemented and ON by default** (`perception.relaxed`, docs/CHANGE_candidate_detection.md, DECISIONS 2026-11-19). **Its val effect is not measured.** Until it is, REPORT, slides, the README headline table and `outputs/m7_final` describe the strict method = ablation row `FULL-relaxed`.
  - Erik: `uv run lsg ablate --names FULL,FULL-relaxed --split eval_val` (both modes, > 1 min), then judge against the change document §5.4 (new R up; false deletion and unchanged kept within the FULL CIs). If it stays the default: `uv run lsg m7 --run-id m7_relaxed` (111 min) refreshes the report numbers; rebuild the slides.
  - If the relaxed rules lose on val: set `perception.relaxed.enabled: false` in `configs/default.yaml` (everything is then as at M7).
  - The M8 steps below are affected: step 1 (`lsg audit sample`) forces the strict rules by itself (`--relaxed` audits the new default); **step 3 (clean-clone README run) now gives the relaxed perception graph: 4 `new` nodes, see the README expected result.**
- **M0–M7 ✅** (see Finished). **M8: code, README, report and slides done; acceptance pending** (Erik's runs, below); state of 2026-11-18:
- **M0–M7 ✅** (see Finished). **M8: code, README, report and slides done; acceptance pending** (Erik's runs, below).
  - M8 commits: `96b85b4` (viz), `6dc69db` (audit), `f1e296a` / `fff1775` (README, report), `d00c0ec` (slides), then the M8 docs commit.
- **To finish M8** (in this order):
  1. Erik: `uv run lsg audit sample --run-id m8_audit` (≈ 3.5 min FULL val + ≈ 85 panels; must print "metrics identical … True"), then fill `verdict` (`gt_missing | method_error | unclear`) and `note` in `outputs/m8_audit/audit.csv` and `audit_veto.csv`, looking at `outputs/m8_audit/panels/`.
  2. Claude: `uv run lsg audit summarize --run-id m8_audit` → numbers into REPORT §6 (placeholder) and slide 11 (`docs/slides/build_deck.js`, rebuild), STATUS / DECISIONS.
  3. Erik: **acceptance check**: fresh `git clone` into an empty folder → `uv sync` → `LSG_DATA` = the data folder → the README §3 command (cold `cache/`). Compare with README "expected result" (Claude can diff the graphs against `outputs/m7_final/dev/FULL/graphs`); paste the output into the final M8 commit.
  4. Erik: one "checked by hand" line per milestone for AI_USAGE.md (M0–M8); then DESIGN §9 M8 ✅.
  5. Optional: a Rerun screenshot (`lsg viz rerun … --spawn`) for the report / slides.
- **Rule (Erik, 2026-11-09):** milestone val reports run on `splits/eval_val_small.txt`; full `eval_val` only for the final tables (done at M7).
- `origin/main` is at `b237be2` (M3 docs). The M4–M8 commits are on local `main` only; not pushed as of 2026-11-18.

---

## Finished

### M7 (2026-11-15/18), acceptance passed — `uv run lsg m7 --run-id m7_final` → `outputs/m7_final/` (`results.md`, `m7_fraction.png`, `m7_tau_free.png`, `m7_ablations.png`, `metrics.json`; git `565fc44`, clean, 111 min)

**Acceptance results:**

| Check | Result |
|---|---|
| Results table with CIs, both modes | `outputs/m7_final/results.md` (val 110 pairs: 13 method / ablation rows, τ_free, fraction; relations val + relchange) |
| Ablation plots | 3 PNGs, inspected |
| Determinism on one pair | FULL on 0ad2d39b → 0ad2d3a1 twice: 0 differences |
| Dev regressions | B0 / B1 / B2 vs M5 / M6 dev, FULL-contrast vs M6 dev FULL: 0 differences each (thin keys excluded) |
| Graphs / tests | 0 invalid graphs; 185 fast tests pass; 10 slow tests pass (run on 2026-11-18 at `565fc44`, before the Hub-offline commit) |

**Val** (removed P / R, false deletion, unchanged kept, moved F1, new P / R; oracle | perception):

| | oracle | perception |
|---|---|---|
| B0 | 21.2 / 69.9, 95.0, 80.9, 30.0, 85.0 / 59.6 | 4.2 / 91.8, 95.9, 2.2, 11.9, 3.3 / 12.9 |
| B1 | 66.0 / 45.2, 9.5, 92.9, 30.0, 85.0 / 59.6 | 17.7 / 57.5, 9.4, 90.4, 11.9, 3.3 / 12.9 |
| B2 | 57.7 / 41.1, 9.5, 86.6, 54.9, 80.5 / 60.2 | 17.7 / 45.2, 9.5, 92.7, 32.0, 2.4 / 9.4 |
| **FULL** | 55.6 / 27.4, 8.1, 86.4, 54.6, 81.6 / 59.6 | 24.2 / 32.9, 7.7, 95.3, 31.7, 2.0 / 8.8 |
| FULL − contrast (= M6) | 54.3 / 26.0, 8.1, 86.6, 54.6, 81.0 / 59.6 | 24.0 / 31.5, 7.7, 95.4, 31.2, 2.0 / 8.8 |
| FULL − appearance | 55.6 / 27.4, 8.1, 86.7, 54.8, 81.1 / 60.2 | 23.5 / 31.5, 7.9, 95.9, 32.5, 2.4 / 9.4 |
| FULL − priors | 56.4 / 30.1, 8.4, 86.3, 54.6, 81.6 / 59.6 | 23.7 / 37.0, 8.0, 94.8, 31.7, 2.0 / 8.8 |
| FULL − decorrelation | 56.1 / 31.5, 9.1, 87.4, 55.4, 82.3 / 59.6 | 24.1 / 37.0, 8.7, 94.9, 31.0, 2.0 / 11.7 |
| FULL + immediate | 19.6 / 42.5, 9.5, 85.4, 34.5, 50.9 / 64.3 | 19.6 / 42.5, 9.5, 94.1, 18.4, 2.1 / 9.9 |

- τ_free and observed fraction: DECISIONS 2026-11-18. Thin candidates, follow parent and keep-parent rows: `results.md`.
- Relations (FULL, parent accuracy R0 / R1 / update): val (a) 92.0 / 79.4 / 92.2 (oracle); relchange (c) swapped 0 / 50.0 / 42.9 (n = 14).

**Code:**
- `geom/render.py` `Renderer.depth_behind`; `change/visibility.py` contrast rule, `NodeVotes.commit_removed_frame`; `change/online.py` `contrast=`;
- `change/baselines.py` immediate commitment (`committed_removed`), `follow_parent`, `_apply_move`;
- `eval/gt.py` `min_extent_ref`, `eval/metrics.py` thin by reference AABB;
- `eval/ablations.py`, `eval/m7_checks.py`, CLI `lsg ablate`, `lsg m7`;
- `config.setup_hub_offline` (no Hub requests once cached).

**Config:** `visibility.seen_needs_contrast`, `visibility.contrast_eps`, `states.commitment`, `update.children_follow_parent`, `perception.clip.hf_repo`, `models.hub_offline`, block `m7`.

**Tests:** 185 fast + 10 slow pass. New: `test_contrast.py`, `test_m7_variants.py`, `test_ablations.py`, a hub-offline test in `test_config.py`.

**Docs:** DECISIONS "2026-11-15", "2026-11-17 — M7 code", "2026-11-18 — M7 results"; DESIGN §3 / §6 / §8 / CLI / §9; CODE_STRUCTURE; AI_USAGE.


### M6 (2026-11-09), acceptance passed — `uv run lsg m6 --run-id m6_final` → `outputs/m6_final/metrics.json` (OBJ/PLY check, render check + overlays, dev B2 / FULL variants both modes, L_hit sweep, val FULL both modes on the full `eval_val`)

**Acceptance results:**

| Check | Result |
|---|---|
| Texture render check (own sequences) | 3 overlays per dev reference (`outputs/m6_final/m6_render_*.png`), inspected: visibly aligned (black = mesh holes); median DINOv2 patch similarity **0.766** (0ad2d39b) / **0.720** (531cff08); rescans 0.56–0.61 |
| OBJ ↔ PLY (§2.1) | identical vertex order, positions (max diff 0.0) and faces on **72 / 72** references |
| Synthetic 2 cm book removed from a table | `test_appearance.py`: FULL depth-only → `unchanged` (missed), FULL + appearance → `removed`; colour features (fast) and DINOv2 (slow) |
| Graphs / determinism | 0 invalid graphs; dev `dev_FULL` = `sweep_Lhit2.5` metrics; slow test runs FULL on one dev pair twice |
| B0–B2 regression | `lsg m5 --no-sweep --no-val` dev metrics identical to `m5_final2` on all shared keys |

**Dev** (3 pairs, both modes; perception / oracle):

| | unch. kept | unchanged w/ free ev. deleted | false-del. | new P | app. cand. / pair |
|---|---|---|---|---|---|
| B2 | 85.3 / 91.2 | 100 / 25 (n = 4) | 8.3 / 16.7 | 1.8 / 50.0 | – |
| FULL depth-only | 94.1 / 94.1 | 25 / 0 | 8.3 / 16.7 | 1.8 / 50.0 | – |
| FULL | 94.1 / 94.1 | 25 / 0 | 8.3 / 25.0 | 1.4 / 50.0 | 5.3 |

- Dev has 1 GT-removed object (pillow r50, a same-class replacement that DINOv2 does not flag). The extra oracle false deletion is the robot vacuum r68, which is really gone but not listed (DECISIONS M6).
- L_hit sweep 0.5 / 1 / 2.5 → 2.5 (lowest false deletion).

**Val FULL** (110 pairs, reported once, nothing tuned; vs B2 at M5):

| | removed P / R | false-del. | unch. kept | new P / R | moved F1 |
|---|---|---|---|---|---|
| B2 oracle | 57.7 / 41.1 | 9.5 | 86.6 | 80.5 / 60.2 | 54.9 |
| **FULL oracle** | 54.3 / **26.0** | **8.1** | 86.6 | 81.0 / 59.6 | 54.6 |
| B2 perception | 17.7 / 45.2 | 9.5 | 92.7 | 2.4 / 9.4 | 32.0 |
| **FULL perception** | **24.0 / 31.5** | **7.7** | **95.4** | 2.0 / 8.8 | 31.2 |

- The confidence veto (33 oracle / 120 perception B2 deletions → `unknown`) trades ~11 removed TPs per mode for ~10–13 fewer absent-unlisted deletions.
- Appearance: 2 / 5 GT-removed objects on val are thin; 1 / 73 removed and 3 / 1,401 unchanged get free-like evidence; thin-object candidates 0.9 per pair, 16% explained.

**Code:**
- `io/scan3r.py`: `read_obj`, `TexturedMesh`, `load_textured_mesh`.
- `geom/render.py`: `TextureRenderer`, `sample_bilinear`, `Renderer.render(return_prim=)`, `face_normal_z`.
- `perception/appearance.py`: `DinoV2Features` (timm), `ColorFeatures`, `patch_similarity`, `mismatch` → `AppearanceMap` (`comparable_px`, `mismatch_px`, `rel_px`), `SimilarityCache` (`cache/appearance/`), `AppearanceRunner`, `tau_app` / `tau_node`.
- `change/visibility.py`: `FREE_LIKE`, `thin_votes`, `NodeVotes.free_like`, free evidence / confirmed rules incl. free-like.
- `change/online.py`: `run_online(appearance=, priors=)`, `obs_views`, `stats`, tentative confidence in snapshots.
- `change/confidence.py` (new); `train_meta.py`: `PriorTable`, `prior_counts`, `class_priors`, `logit`.
- `change/baselines.py`: `full`, shared `_b2_core`; every method takes `obs_views=`, `priors=`; `METHODS["FULL"]`.
- `perception/source.py`: thin-object candidates (separate voxel grid, `kind = "appearance"`, `wants_appearance`); `residual.SurfaceFilter.keep_struct`.
- `obs/base.py`: `RefRender.appearance` / `.up`, `RefContext.thin_oids`, `ObsInstance.kind`; `graph/model.validate`: confidence in [0, 1].
- `eval/metrics.py`: free evidence incl. free-like, `free_like_ev_*`, `removed_R_thin`, `free_ev_removed_thin`, `unchanged_kept_thin`, `conf_mean_*`, `cand_appearance_*`.
- `eval/harness.py`: FULL wiring (texture per reference group, extractor per process, priors). `eval/m6_checks.py` + CLI `lsg m6 [--no-sweep] [--no-val] [--no-objply] [--full-val]`.
- `io/splits.py`: `subset_pairs`, `write_pairs`; CLI `lsg splits subset`; `config.milestone_val_split`; `lsg m5 --full-val`.

**Config:** new blocks `confidence`, `appearance` (incl. per-feature `tau_app` / `tau_node`), `states.k_free_like`, `splits.milestone_val`.

**Data:** `splits/eval_val_small.txt` (20 pairs, seed 0).

**Dependencies:** `timm` explicit (was already installed via open_clip); DINOv2 ViT-S/14 weights from the HF hub.

**Tests:**
- 166 fast (≈ 15 s) + 10 slow (≈ 2.4 min) pass.
- New: `test_texture.py`, `test_appearance.py`, `test_confidence.py`, textured scenes in `synth.py`, slow `test_m6_full_both_modes_one_dev_pair`, `test_dinov2_texture_render_on_own_sequence`, `test_book_removed_with_dinov2`; no-leak test covers `appearance.py` / `change.confidence`.

**Docs:** DECISIONS "2026-11-09 — M6"; DESIGN §2.1 / §2.3 / §3 / §6 / §8 / CLI / §9; CODE_STRUCTURE; AI_USAGE; CLAUDE.md (milestone val subset).

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

### Carried over from M7 (report as limitations or later work)
- **Contrast did not fix the confidence veto** (removed R +1 object per mode). Use the §7.4 audit to check what the vetoed GT-removed objects really are: replacements, partial removals, or GT errors.
- **Perception association stays near B0** (val FULL ~3% strict): unchanged objects produce no candidates.
- **False candidates:** 8.7 per pair, 43% explained by GT (plants, black TV screens, clutter); perception new P 2.0%.
- **τ_move / τ_assoc sweeps** (not in §8): oracle unchanged kept 86.4% (B1 92.9%) because τ_move = 0.10 m turns ICP noise into small moves.
- **Perception same-label swaps** and **small moves become `removed`** (M5 dev diagnosis).
- **Relation type accuracy** (R1 83% given the parent, `on` vs `in`); **anchors carry no support edge**.
- **Door filter cost** and **`min_voxels`** (M4); optional DINOv2 association term (old B3) not done.

### M8 — done in the write-up (docs/REPORT.md §2, §4, §6, §8)
All items that were listed here are in the report: door swings out of scope; val run twice at M5 with the first run's numbers; the M6 audit cases (r50, r68, r69); spec deviations; R0 strong; `next_to` provisional; GT findings; the M7 carry-overs as limitations.

### M8 — still open
- Audit results (§7.4) into REPORT §6 and slide 11 (after Erik's labels).
- Acceptance check (clean clone) and AI_USAGE lines (Erik).

---

## Decisions needed from Erik
- **CHANGE 3 (2026-11-20):** §8 answered; Part 0 done incl. download and baseline. Nothing open for Erik before Part A.
- **CHANGE 2 (2026-11-20): class-agnostic prior + one confidence for new objects.** Proposed, not implemented; all questions of `docs/CHANGE_confidence_unified.md` section 7 answered, ready to implement; answered by Erik: drop low-confidence candidates, no prior value (l0 = 0, margin via `p_del` = 0.027), correlated views keep the small weight, one observation is enough for every cluster, ON by default, confidence logged into the snapshots. Builds on CHANGE 1, whose val effect is not measured yet.
- **Candidate detection change (2026-11-19):** relax `min_views` for large clusters (1) and small (2), relax `tau_new` for big objects. Decided, not implemented; four open questions in `docs/CHANGE_candidate_detection.md` section 6 (notably small = 2 or 3 views; the current value is 3).
- **Doors / articulation:** decided 2026-11-18, the door diagnostic is skipped and stated as a limitation (DECISIONS M8). Closed.
- **AI_USAGE.md:** "Erik: (fill in)" is still open for the M0–M8 rows (Erik sends the lines at the end of M8).
- **DECISIONS 2026-11-05 "Milestones re-scoped to M1–M8"** is still marked "provisional — Erik to confirm".
