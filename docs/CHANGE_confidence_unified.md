# CHANGE 2 — one confidence for reference nodes and new objects (proposal, 2026-11-20)

**Status: proposed by Erik, NOT implemented.** No code or config has changed. Part A needs no code (a config
flip). Part B is a design for review; the open questions in §7 must be answered first.
Builds on CHANGE 1 (`docs/CHANGE_candidate_detection.md`, implemented 2026-11-19, val effect not measured yet).

## 1. Why

Two observations from Erik (2026-11-20):

1. **The class prior is specific to this dataset.** `l0 = logit(p_persist(class))` is learned from the training
   rescans (chair 0.968, bed 0.996). It also makes a bed need 3 counted free votes (l0 = 5.52 → −1.98) where a
   chair needs 2, even if the bed is clearly gone.
   - The M7 ablation `FULL-priors` (flat p = 0.9 for every class, all 110 val pairs) shows the prior buys almost
     nothing: oracle removed P / R 56.4 / 30.1 (FULL 55.6 / 27.4), false deletion 8.4 (8.1); perception
     23.7 / 37.0 (24.2 / 32.9), false deletion 8.0 (7.7). All within the CIs of FULL.
2. **New objects have no confidence process.** A perception candidate becomes a `new` node at the end of the
   walk if its voxels pass a view count (`min_views_*`, CHANGE 1). After that its confidence is fixed
   (`l = clip(2.5 · views)` from 0). Nothing can lower it. A transient (a person who stands still, a bag carried
   through the room) is filtered only by the view count, never by later views seeing through it, although this
   is exactly the free-space evidence that already protects the reference nodes (hard rule 3).

Erik's proposal: add an object after **one observation**, then let later observations confirm or delete it, with
**decorrelated views counting more**, using the same confidence machinery as for reference nodes.

## 2. What is decided (Erik) and what is open

Decided:
- No class prior learned from the dataset (Part A).
- A candidate is added after a single (decorrelated) observation; its confidence is then updated by later
  frames (confirm / see through), decorrelated views carrying a larger vote (Part B).

Open (§7): the value of the class-agnostic prior, the evidence needed for small clusters, whether low-confidence
candidates are dropped or kept, the default (on / off), the weight of correlated views.

## 3. Part A — class-agnostic prior (no new code)

`confidence.class_priors: false` already exists: `l0 = logit(confidence.fallback_p)` for every node (ablation
row `FULL-priors`, M7). Change: make that the default and rename the meaning: `fallback_p` becomes `prior_p`,
**one** hyper-parameter, "a reference object is assumed to be still there", not learned from the data.

Effect of the value (L = 2.5 per counted vote, deletion needs p < `p_del` = 0.2, i.e. l < −1.39, **and** free
evidence, i.e. ≥ 2 counted free votes):

| `prior_p` | l0 | 2 free votes | 1 seen + 2 free votes (mixed evidence) | 1 free vote |
|---|---|---|---|---|
| 0.5 (neutral) | 0 | −5.0 → p 0.007: deleted | −2.5 → p 0.076: **deleted** | −2.5, blocked by the ≥ 2 votes rule |
| 0.9 (= M7 `FULL-priors`) | 2.20 | −2.80 → p 0.057: deleted | −0.30 → p 0.43: protected | blocked |
| 0.968 (old chair prior) | 3.41 | −1.59 → p 0.17: deleted | 0.91 → p 0.71: protected | blocked |

So with a neutral prior, mixed evidence (one seen vote cancelling one free vote) no longer protects a node. The
protection of mixed evidence needs `prior_p` ≥ about 0.8. Proposal: keep 0.9 (its M7 numbers are known) and test
0.5 / 0.8 / 0.9 on dev; the remaining assumption is stated openly in the report.

`train_meta.class_priors` (the learned table) stays in the code as the ablation row `FULL+classprior`
(= the M7 FULL).

## 4. Part B — candidate confidence as per-voxel log-odds (perception only)

### 4.1 Idea
Each tracked residual voxel carries a log-odds `lv`, updated per frame like a reference node. Candidates are
the connected components of voxels whose confidence passes a threshold. The `min_views` rules become a special
case (`lv = 2.5 · views`), and one new kind of vote is added: **seeing through the voxel**.

### 4.2 Votes on a voxel (per processed frame; the voxel's mean residual point `x` is projected with the pose)
| vote | condition | effect |
|---|---|---|
| **hit** | a residual point of this frame falls into the voxel (existing step 1–3 of the candidate pipeline) | `+L_hit` (decorrelated) |
| **free** | `x` projects into the valid image area, the **minimum** depth in a 3×3 window around its pixel is > `z(x)` + `free_margin` (0.15 m = τ_free) | `−L_free` (decorrelated) |
| **no information** | `x` outside the image, depth invalid (also the never-valid columns 0–20, 205–223), or the window depth < `z(x)` − margin (something in front = occlusion) | none |

- The 3×3 window (`free_window: 1`) makes a free vote conservative: a voxel on an object silhouette whose pixel
  falls on the background because of a small pose error is not "seen through".
- Free evidence is a free-space observation, so hard rule 3 is respected: occlusion and out-of-view never lower
  a confidence.

### 4.3 Weights: decorrelated views count more
- **Decorrelated** vote (the existing rule: camera moved ≥ 0.30 m or turned ≥ 20° since the voxel's last counted
  vote of that kind): `L_hit = L_free = 2.5`.
- **Correlated** vote (same viewpoint region): `L_corr` = 0.25, with the sum of correlated additions since the last
  decorrelated vote capped at `corr_cap` = 1.0. This keeps 30 frames from one spot from adding up to a full
  confidence (frames are strongly correlated), but a repeated view is no longer worth exactly zero.
- Per-visit clip as before.
- `L_corr = 0`, `corr_cap = 0`, free votes off reproduces the CHANGE 1 candidates exactly (§6, test).

### 4.4 Thresholds replace `min_views` (the CHANGE 1 tiers stay)
The two tiers of CHANGE 1 keep their grids, their `tau_new` / `tau_new_large`, their connectivity and the
"large replaces small" merge. Only the voxel filter changes:

| CHANGE 1 | CHANGE 2 |
|---|---|
| strict grid: voxels with `views ≥ min_views_small` (2) | voxels with `lv ≥ l_keep_small` (5.0 = two decorrelated hits) |
| lenient grid: voxels with `views ≥ min_views_large` (1), clusters ≥ 40 voxels | voxels with `lv ≥ l_keep_large` (2.5 = **one** decorrelated hit), clusters ≥ 40 voxels |

Erik's wish "add after one observation" is the lenient tier (2.5). Whether small clusters may also pass with one
observation is an open question (§7.2).

### 4.5 Candidate confidence and the `new` node
- Candidate confidence `p = sigmoid(median lv over its voxels)`. A median is robust to a few noisy edge voxels.
- `ObsInstance` gets an optional field `confidence` (perception only; oracle instances keep `None`).
- `confidence.py`: a new node takes `l = logit(p)` of its candidate when the field is set; oracle keeps
  `clip(L_hit · views)` from 0.
- **Emission:** a candidate becomes a `new` node only with `p ≥ p_new_min` (config; default = the confidence of
  one decorrelated hit, 0.92). Dropped candidates are counted in the diagnostics. All candidates are written to
  `candidates/<pair>.json` with their confidence, so a precision / recall curve over `p_new_min` can be scored
  from saved files (small helper in `eval/`, like the audit) without rerunning perception.
- Association, absorption, the ICP moved test and the removal rule of reference nodes are **unchanged**. A
  candidate matched to a reference node still feeds `matched: l = l0 + clip(L_hit · views)`.

### 4.6 Not part of this change
Thin-object (appearance) candidates (keep `min_views: 3`), the reference-node votes and the confirmed-in-place rule,
oracle mode, relations, the SAM idea, strength-weighted free votes by pixel coverage (discussed in chat, a
separate change).

## 5. Expected effects and risks

- **Transients are deleted by evidence, not only filtered by a view count.** A person seen in 2 decorrelated views
  and then seen through: 5.0 − 2.5 = 2.5, p 0.92 (still emitted) → a second see-through → 0, p 0.5 (dropped).
  The margin needs to be tuned: a person standing still for a minute can pass the hit threshold before any
  see-through vote exists.
- **More `new` nodes with low confidence** if `p_new_min` is low. Expected: new R up, new P down at the same
  threshold; the curve over `p_new_min` is the result that matters.
- **False see-through votes** at silhouettes, grazing angles, depth noise; mitigations: the 3×3 minimum-depth window,
  the 0.15 m margin, decorrelation. A real object is not removed by one view (needs 2 free votes against 2 hits).
- **Moving camera, hidden then visible:** a voxel behind an obstacle is no-info, not free. Doors opening or
  sliding objects can still produce see-through votes (limitation, same as for reference nodes).
- **Cost:** about 13–17k tracked voxels per pair × 240–390 frames, vectorised projection; to be measured against the
  current ~3 s per pair. Two grids double it.
- **Prior value (Part A):** with a flat 0.9, anchors lose their protection (a bed now needs 2 free votes). On val
  the M7 numbers say this is neutral overall; the audit veto set shows which GT-removed objects are involved.
- **Dev is tiny** (3 pairs, 3 GT-new objects): the thresholds `l_keep_*`, `p_new_min` and the weights cannot be
  tuned reliably there; val is reported once.

## 6. How to evaluate (CLAUDE.md rules)

1. **Equivalence test** (unit + dev): with `L_corr = 0`, free votes off, `l_keep_small = 5.0`, `l_keep_large = 2.5` the candidates equal CHANGE 1's (same tokens, same points). With `candidates.enabled: false` the code path is CHANGE 1.
2. **Dev check** (`dev_train`, candidate geometry as in CHANGE 1): candidates per pair, share explained by GT, candidate recall (new / moved-to), number of candidates deleted by see-through votes, overlays. Tune only here (hard rule 4); report if the evidence is too thin.
3. **Ablation rows on the full `eval_val`** (both modes, `lsg ablate`), each reported once with 95% bootstrap CIs:
   - `FULL` (CHANGE 1 default) as the baseline;
   - `FULL-priors` (exists; flat prior only, Part A);
   - `FULL+candconf` (Part B with the flat prior);
   - `FULL+candconf-free` (Part B without see-through votes: isolates the free-space effect);
   - `FULL+candconf` at 3 values of `p_new_min` (precision / recall curve).
4. **Success criterion (written before the run):**
   - new P at equal new R is higher than the baseline at at least one operating point of the `p_new_min` curve, and new R at the default threshold is not lower than the baseline's;
   - false deletion and unchanged kept within the CIs of FULL;
   - removed recall not worse than FULL's CI (Part A);
   - report moved F1, candidates per pair and the share deleted by see-through whichever way they move.
5. **Audit:** the `new` rows of `lsg audit` (§7.4) say whether the removed candidates were real or GT gaps.
6. Log in DECISIONS; update DESIGN §3 (candidate step 4) and §6.11 (confidence) only after sign-off.

No new dependency (numpy / scipy).

## 7. Open questions (answer before implementing)

1. **Value of the class-agnostic prior** (`prior_p`): 0.9 (known M7 behaviour, keeps mixed-evidence protection) or neutral 0.5? Proposal: 0.9, tested at 0.5 / 0.8 / 0.9 on dev.
2. **Small clusters:** one observation for every cluster (`l_keep_small` = 2.5) or two (5.0, as in CHANGE 1)? "After one observation" literally means 2.5; the size rule of CHANGE 1 (≥ 40 voxels for the 1-view tier) was added because 1-view voxels percolate into blobs.
3. **Low-confidence candidates:** drop below `p_new_min`, or emit them as `new` nodes with their confidence and let the evaluation threshold decide? Proposal: drop (smaller graph), all kept in `candidates/` for the curve.
4. **Default:** on or off until the val ablation? CHANGE 1 was set ON by Erik; the same here would make every perception number change again (REPORT already carries a note).
5. **Weight of correlated views** (`L_corr` 0.25, `corr_cap` 1.0): fine as a starting point, or should repeats count zero as today?
6. **Snapshots / visualisation:** log candidate confidence into the snapshots so `lsg viz rerun` shows candidates appearing and being confirmed or deleted? (Small, optional; fits Erik's earlier wish for a viewer that follows the scan.)

## 8. Work list (not started)

- [ ] answer §7 with Erik
- [ ] Part A: config flip (`class_priors: false`, `prior_p`), ablation row `FULL+classprior`, dev check of 0.5 / 0.8 / 0.9
- [ ] `perception/voxels.py`: per-voxel log-odds, correlated cap, `kept_by_logodds`; see-through update (`observe_free(depth, K, pose)`); `perception/source.py`: call it per frame, thresholds in `finish`, candidate confidence; `obs/base.py`: `ObsInstance.confidence`; `change/confidence.py`: new node from candidate confidence; emission threshold in the update step
- [ ] config block `confidence.candidates` (`enabled`, `L_corr`, `corr_cap`, `free_margin`, `free_window`, `l_keep_small`, `l_keep_large`, `p_new_min`)
- [ ] unit tests: equivalence with CHANGE 1; a transient (box present for a few frames, then removed) is dropped by see-through votes; occlusion and out-of-view give no free vote; silhouette window; correlated cap; determinism; `tests/test_no_leak.py`
- [ ] harness / eval: candidates with confidence saved; `new_pr_curve` helper; ablation rows; `lsg ablate` run by Erik (> 1 min)
- [ ] DECISIONS, DESIGN §3 / §6.11, CODE_STRUCTURE, REPORT note, STATUS
