# CHANGE 3 — tuning set, move evidence, incremental graph, confidence policy (proposal, 2026-11-20)

**Status (2026-11-20): PROPOSAL, all questions answered (§8), nothing implemented.** Builds on CHANGE 1 + 2,
which Erik fixed as the **final method** (2026-11-20). Comes from a logic review of the whole solution (chat,
2026-11-20). Nothing here changes DESIGN.md before sign-off.

## 1. Why

Four problems, ordered by how much they limit the results:

1. **There is no real tuning set.** `dev_train` has 3 pairs, with 1 GT-removed and 3 GT-new objects. Almost no
   threshold could be chosen there: most kept their start values, or were chosen from one-object differences
   (L_hit, `large_min_voxels`, `l_keep_*`). In practice val became the design loop: the M5 catch-all fix and
   keep-parent rule, contrast (from the M6 val veto), CHANGE 1 (from the M7 val new-recall funnel), CHANGE 2
   Part A (from the `FULL-priors` val row), and now the CHANGE 2 defaults. REPORT's "every threshold was
   tuned on 3 training pairs" is no longer accurate.
2. **"Moved" ignores the old place; "removed" needs evidence there.** The core rule of the project is: don't
   claim something is gone without free-space evidence. A move claims the same thing (the object is no longer
   where it was), but `associate` never looks at votes (`baselines.py` `_b2_core`, `association.py`
   `pair_cost`). A matched node's confidence also ignores its old-place votes (`confidence.node_logodds`). So a
   node seen in place twice (one short of "confirmed"), or never observed at all, becomes `moved` when a
   label-compatible candidate appears anywhere within the gate. Likely consequences:
   - false move 1.6 → 6.4% and unchanged kept 95.3 → 88.8% with CHANGE 2 (3× more candidates, same free
     matching), the part of the CHANGE 2 criterion that failed;
   - oracle unchanged kept 86.4% vs B1's 92.9% (ICP noise on partial views above τ_move = 0.10 m);
   - same-label swaps.
3. **The graph is only maintained at the end.** Votes accumulate online, but clustering, CLIP, association,
   ICP, relations, and the `T_align` refinement (a pre-pass over the whole chunk) all run after the last
   frame. Snapshots therefore never contain `moved` or `new`. Erik (2026-11-20): an incremental graph update
   during the walk is the way to go.
4. **The confidence policy is mostly not used or not meaningful.** The brief leaves the policy to us (Erik,
   2026-11-20).
   - After CHANGE 2 Part A, a reference node with no new evidence reports p = 0.5, while a candidate seen from
     one view reports ≥ 0.92.
   - Candidate confidence is the median over voxels that already passed lv ≥ 2.5, so it is always ≥ 0.924, and
     `p_new_min` = 0.92 never filters anything (DECISIONS 2026-11-20, choice 5). Voxels that are seen through
     just leave the cluster. The strongest transient signal ("40% of this cluster was seen through") is lost.
   - L_hit = L_miss = 2.5 is not backed by sensor data. A "seen" vote is weaker evidence that the *same* object
     is there than a "free" vote is that it is gone (a replacement also gives "seen"). This is the likely cause
     of the confidence veto (removed R 27 vs 41 for B2).

## 2. Part 0 — a real tuning set (`splits/dev_tune.txt`)

**Data available (computed from metadata, 2026-11-20).** The train split has 718 pairs in 360 rooms that are
neither dev nor relchange, with 304 listed removed, 1,006 new and 2,136 rigid entries (things + anchors).
Removals are concentrated: only 158 of the 718 pairs have one.

| selection (one rescan per room) | removed | rigid (incl. < 10 cm) | new |
|---|---|---|---|
| 30 random rooms (seed 0) | 9 | 100 | 24 |
| **20 change-heavy + 10 random rooms (proposal)** | **85** | **106** | **212** |
| 30 change-heavy rooms | 104 | 130 | 261 |
| val, for comparison (110 pairs, scored) | 73 | 235 (≥ 10 cm) | 171 |

- **Proposal:** 20 rooms with the most changes (score 2 · removed + new + 0.5 · rigid, best rescan per room)
  plus 10 random rooms (seed 0), plus the 3 `dev_train` pairs = **33 pairs**. That gives more removed and new
  objects than val. The random third keeps a realistic share of quiet rooms, so the false-deletion side is
  tuned on ordinary rescans, not only on change-heavy ones. Selection is by GT, which is allowed on train.
- **Download** (hard rule 6): `splits/download_tier3.txt` (60 scans: semseg, PLY, `sequence.zip`) and the 30
  references added to `splits/download_textures.txt` (`lsg splits textures`). Measured scans are 30–36 MB
  including textures, so this is ≈ 2–3 GB. Erik runs the download.
- **Vocabulary:** the CLIP vocabulary excludes the dev rooms (`train_meta.exclude_splits`). Excluding 30 more
  rooms swaps 6 of the top-100 labels (coffee machine, flower, flowers, rug, stuffed animal, toy; 3 in, 3 out).
  That changes CLIP outputs and so every perception number slightly (question §8.2).
- **Rule change (CLAUDE.md hard rule 4, needs Erik):** tune on `dev_tune.txt`; `dev_train.txt` stays as the
  3-pair smoke / bit-for-bit regression set.
- **Val protocol from here on:** val is not used to choose anything in this change. One final run (`lsg m7`)
  produces the report tables. REPORT §8 says plainly which earlier decisions val influenced (list in §1.1).
- Code: `lsg splits devtune` (the selection rule above, written to `splits/dev_tune.txt` and
  `splits/download_tier3.txt`); `lsg eval --split dev_tune` works as is. A `dev_tune` FULL run takes ≈ 1.5 min
  (val: 274 s for 110 pairs).

## 3. Part A — a move needs consistent evidence at the old place

**Rule (default on, `association.move_needs_old_place`).** Before association, a reference node is not
eligible to be matched to an observation outside its own footprint if its old place is net-seen:
counted seen > counted free + free-like. Such a node keeps the state its votes give (`unchanged` if
confirmed, else `unknown`). The rule mirrors the removal veto: you cannot say an object left a place where the
sensor keeps seeing it.

Example (perception, today): chair r12 gets 2 counted seen votes at its old place (one short of confirmed) and
0 free. A "chair" candidate appears 2.4 m away (in reality a new chair). Hungarian pairs them, ICP gives 2.4 m,
and r12 becomes `moved` (a false move), while the new chair is lost (a missed `new`). With the rule, r12 stays
`unknown` (kept) and the candidate becomes `new`.

**Variants to compare on `dev_tune` (tuning, hard rule 4):**
- (a) the rule above;
- (b) (a), plus nodes whose old place was never observed (no counted votes at all) are matched only within
  `move_max_dist_unseen` (e.g. 1.5 m). Nodes with old-place free evidence keep the full gate;
- (c) oracle ICP noise: moved needs displacement > τ_move **and** (old-place free evidence or ICP fitness ≥ f)
  — targets the oracle unchanged-kept loss.

**Matched-node confidence** keeps its formula, l = l0 + L_hit · views at the new place, but it is now
consistent. Free votes at the old place agree with the move, so not subtracting them is correct. Net seen
votes at the old place can no longer exist for a moved node, because the rule above prevents the match.

**Expected effect:** false move and unchanged kept back toward the M7 values. Moved recall drops only where the
old place was seen. Report moved R by displacement bucket to show that.

## 4. Part B — incremental graph during the walk

**Idea.** Every `online.snapshot_every` processed frames (20 now; ≈ 10–20 per pair) the method runs on the
*current* state and writes a full tentative graph: association, ICP, states including `moved` and `new`,
relations (`update_graph`) and confidence. It is the same function that produces the final graph, so the last
snapshot equals the final graph by construction (a test checks this). Deferred commitment stays: states in
snapshots are tentative, and `removed` / `moved` are final only at the end (§3 of DESIGN unchanged).

**What has to change:**
1. **`ObsSource.peek()`** (new, next to `finish()`): returns the current instances without ending the walk.
   Oracle: instances so far. Perception: clustering of the current voxel grids (cheap, scipy).
2. **Stable ids across snapshots.** Candidate tokens `c<k>` are assigned by cluster size at `finish()` today,
   so they would change from snapshot to snapshot. New: a candidate track keeps its id while its voxels overlap
   the previous snapshot's cluster (voxel IoU ≥ 0.3, greedy). A merge keeps the older id; a split creates a
   new id. New nodes `n<k>` follow their track.
3. **CLIP at snapshots.** Labels are recomputed only for tracks whose voxel count grew by > 25% since their
   last label (otherwise the cached label is reused). Rough estimate: ≈ 27 candidates × 5 crops × 10–20
   snapshots ≈ 1–3k crops per pair, about 5–10 min extra for val on the GTX 1650; to be measured.
4. **Alignment refinement online** (Erik, §8.4: incremental). Today the ICP pre-pass uses every frame of the
   chunk, including future ones. New: **one cumulative estimate**. At each snapshot, ICP is re-run on the
   frames processed so far (every 10th, as now), starting from the previous estimate. Votes use the raw
   `T_align` until the first accepted estimate (≥ `alignment.min_frames_online` frames). The same rejection
   rules apply (> 20 cm / 5°), plus a degeneracy check (smallest eigenvalue of the point-to-plane normal
   matrix), so a few frames that see only one wall are not used.
   - **Why not ICP on the current frame alone** (discussed 2026-11-20): the error is one rigid offset shared
     by all frames. The rescan's poses fit its own mesh to 1.1–1.3 cm, and one global correction already
     reaches 1.9–2.2 cm, so per-frame estimation can gain at most ≈ 1 cm. It would re-estimate the same 6
     parameters from one frame's noisy points (jitter between frames, and the votes depend on it). It would
     also pull the pose toward changed geometry when a moved object fills the frame, erasing the free /
     nearer pixels the method looks for. And single 224×172 frames are often degenerate (one plane).
   - **Variants to test on `dev_tune`** (residual and change metrics): (i) the cumulative estimate against the
     current offline pre-pass; (ii) ICP target restricted to reference static surfaces (wall, floor, ceiling;
     no doors), allowed because it reads reference labels only; (iii) a per-frame correction regularised
     toward the cumulative estimate, kept only if it lowers the residual.
5. **Viz:** `lsg viz rerun` already draws snapshot graphs; it now also shows `moved` arrows and `new` boxes
   mid-walk.

**Evaluation (new, cheap):**
- **anytime curve**: score every snapshot against the GT (moved F1, new P / R, false deletion, unchanged kept)
  over walk fraction 0–100%, both modes. This shows the graph getting better during the walk, and it partly
  replaces the observed-fraction ablation;
- **state flips per node**: how often a tentative state changes (stability);
- final metrics must equal the end-of-sequence run, apart from the effects of online alignment (item 4).

## 5. Part C — confidence policy

**Definition (proposal):** `confidence` = the belief that the node is present at the position the graph gives
it (reference position, or the new position for `moved` and `new`). One log-odds scale for all nodes:
`l = l0 + Σ counted-vote weights`, deleted / not emitted below a threshold.

1. **Vote weights from sensor statistics, not chosen by hand.** On `dev_tune`, measure per counted vote kind
   how often it occurs for GT-unchanged vs GT-removed objects, and set
   `L_kind = log(P(vote | present) / P(vote | removed))` (Laplace-smoothed). This is the log-odds model as
   intended. It describes the sensor and the vote rules, not how often rooms change, so it does not bring back
   the dataset-specific class prior that CHANGE 2 removed. Expected result: |L_seen| < |L_free|, which weakens
   the veto exactly where the review found it too strong. The free-evidence rule (≥ 2 counted free) and
   decorrelation stay as hard conditions.
2. **Reference-node start value** (Erik, §8.3): **flat l0 = logit(0.9), p_del 0.2** for every reference
   node (was l0 = 0, p_del 0.027). The decisions are the same (margin 3.59 vs 3.6); an untouched reference
   node now reports 0.9 instead of 0.5. With learned weights (item 1) `p_del` is re-tuned on `dev_tune`.
3. **Candidate confidence over the cluster's full history.** Per candidate track, `l = L_hit · (decorrelated
   views of the track) − L_free · (decorrelated see-through votes on any of its voxels, including voxels that
   fell below the keep threshold)`, with the same weights as item 1 (estimated on `dev_tune` for candidates:
   GT-explained vs unexplained). This keeps the transient signal that is lost today, and the value varies, so
   it can rank.
4. **Emission threshold `p_new_min` tuned on `dev_tune`** from the new-node precision / recall curve
   (`lsg newcurve`), so that it actually filters. Candidates below it still take part in association (they can
   explain a move); they only do not become `new` nodes.

## 6. Part D — reporting

- New P / R are reported twice: **labelled** (§5 rule, as now) and **class-agnostic** (geometry only), so the
  detection loss and the CLIP naming loss are visible separately. The numbers exist already
  (`candidate_counts`); they move into the headline tables.
- REPORT, README, slides and the audit move to the final method (CHANGE 1 + 2 + 3). The audit stops forcing
  `M7_OVERRIDES` and is sampled from the final val run. REPORT §8 gets the val-influence list of §1.1.

## 7. How to evaluate (CLAUDE.md rules)

1. Tune only on `dev_tune` (Part 0 first). Every chosen value goes into `configs/default.yaml` with its dev
   evidence in DECISIONS.
2. **Equivalence tests:** each part has a switch; all switches off = CHANGE 2 bit for bit on `dev_train`; the
   last snapshot = the final graph; `peek()` followed by `finish()` = `finish()` alone.
3. **Success criterion on `dev_tune` (written before tuning), against CHANGE 2 on `dev_tune`:**
   - Part A: false move and unchanged-kept loss at least halved, moved F1 not lower by more than 2 points;
   - Part C: removed R up at equal or lower false deletion; new P up at equal new R;
   - Part B: final metrics equal to the end-only run (item 4 aside); anytime curve reported.
4. **One val run at the end** (`lsg m7`, Erik starts it), with ablation rows `FULL-moveev`, `FULL-online`,
   `FULL-learnedL`, `FULL-chg2` (= the current default). Reported once, not tuned on.
5. No new dependency.

## 8. Questions (all answered by Erik, 2026-11-20)

**Answers:** (1) mixed selection, 20 change-heavy + 10 random + the 3 `dev_train` pairs; (2) exclude the
tuning rooms from the CLIP vocabulary; (3) flat 0.9 start value (§5.2); (4) incremental alignment, as one
cumulative estimate (§4 item 4; per-frame-only ICP discussed and not chosen); (5) order 0 → A → C → B → D.

Original questions:

1. **Tuning-set selection** (§2): 20 change-heavy + 10 random rooms (proposal), 30 change-heavy, or 30 random?
   Random alone gives 9 removed objects, too few to tune the removal side.
2. **Vocabulary:** exclude the new dev rooms from the CLIP vocabulary (spec-consistent; 6 labels of the top-100
   change, every perception number shifts slightly, a final val rerun is needed anyway), or freeze the
   current vocabulary?
3. **Reference-node start value:** l0 = 0 (untouched reference node reports 0.5; a one-view candidate 0.92) or
   flat 0.9 (untouched node 0.9; same decisions)? With learned weights (§5.1) the margin is set by `p_del`
   again, tuned on `dev_tune`.
4. **Online alignment:** refine `T_align` incrementally (strictly online; early frames use the raw alignment,
   2.9–3.8 cm residual instead of 1.9–2.2 cm), or keep the whole-chunk pre-pass and state that it uses future
   frames?
5. **Scope / order:** proposal Part 0 → A → C → B → D (A and C are cheap and target the failed CHANGE 2
   criterion; B is the largest piece of code). Or B earlier?

## 9. Work list

- [x] answers to §8
- [ ] Part 0: `lsg splits devtune`, split + download lists; **Erik:** download tier 3 + textures; CLAUDE.md
      hard rule 4; baseline run of CHANGE 2 on `dev_tune` (≈ 1.5 min, Erik)
- [ ] Part A: rule + variants, tests, dev_tune comparison
- [ ] Part C: vote statistics on dev_tune, weights, candidate track confidence, `p_new_min`
- [ ] Part B: `peek()`, candidate tracks, snapshot method runs, CLIP relabel rule, online alignment (per §8.4),
      anytime curve, viz
- [ ] Part D: class-agnostic new in the tables; audit on the final method
- [ ] DECISIONS, DESIGN (§3, §6.4, §6.7, §6.9, §6.11, §7), CODE_STRUCTURE, STATUS, after sign-off
- [ ] **Erik:** final `lsg m7` on val; then REPORT / README / slides / audit refresh
