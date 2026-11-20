# CHANGE 2 — one confidence for reference nodes and new objects (proposal, 2026-11-20)

**Status: decided by Erik, all open questions answered (2026-11-20), NOT implemented.** No code or config has changed. Part A needs no code (a config
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

Answered 2026-11-20 (Erik): **no prior value is needed** (Part A: l0 = 0, the deletion margin is set through
`p_del`); **correlated views keep the small weight** (§4.3).

Also answered 2026-11-20 (Erik): **one decorrelated observation is enough for every cluster, small ones too**
(`l_keep_small` = `l_keep_large` = 2.5); **ON by default**; **candidate confidence is logged into the snapshots**.

Last answer (Erik, 2026-11-20): candidates below `p_new_min` are **dropped** from the graph (all stay in `candidates/<pair>.json`). **No open question is left; the design is ready to implement.**

## 3. Part A — no prior at all (no new code)

**The prior and `p_del` are one parameter.** A node is deleted when `l0 + E < logit(p_del)`, with `E` the net
evidence (counted votes, in log-odds units). That is the same as `E < −M` with the **margin**
`M = l0 − logit(p_del)`. Only `M` matters: how much net evidence the method needs before it deletes.
- M7 FULL: `M` depends on the class. Chair: 3.41 + 1.39 = 4.80 (2 votes); bed: 5.52 + 1.39 = 6.91 (3 votes).
- M7 `FULL-priors` (flat p = 0.9, `p_del` 0.2): `M` = 2.20 + 1.39 = 3.59 for every class.

So no prior value has to be chosen. Proposal: **`l0 = 0` for every reference node (p = 0.5, "no information")**
and the margin set directly through `p_del`: `M` = 3.6, i.e. `p_del = sigmoid(−3.6)` = **0.027**. Net evidence of
2 counted votes (−5.0) deletes, 1 vote (−2.5) cannot. This is exactly the behaviour of `FULL-priors`, so its val
numbers (oracle removed P / R 56.4 / 30.1, false deletion 8.4; perception 23.7 / 37.0, 8.0) apply to it:

| evidence (L = 2.5 per counted vote) | E | p = sigmoid(E) | deleted? (needs ≥ 2 counted free votes **and** p < 0.027) |
|---|---|---|---|
| 2 free | −5.0 | 0.007 | yes |
| 1 free | −2.5 | 0.076 | no (also blocked by the ≥ 2 votes rule) |
| 1 seen + 2 free (mixed) | −2.5 | 0.076 | no: protected |
| 1 seen + 3 free | −5.0 | 0.007 | yes |
| any class, bed or chair | same | same | same |

Consequences:
- A bed needs the same evidence as a chair, whatever the dataset says about beds.
- The confidence score becomes a pure evidence score on one scale for reference nodes and new nodes (both
  start at 0). A node with no counted votes reports p = 0.5 ("no evidence"), not 0.97. This changes what the
  `confidence` field means (documented in DESIGN §6.11 after sign-off).
- Config: `confidence.class_priors: false`, `confidence.prior_p: 0.5` (was `fallback_p`), `confidence.p_del: 0.027`.
  `M` is the only caution parameter left (not learned from data); checked on dev against `FULL-priors` and varied
  between 1 and 2 votes (`p_del` 0.076 to 0.007).
- The learned table `train_meta.class_priors` stays in the code as the ablation row `FULL+classprior` (= M7 FULL).

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
  **Decided by Erik (2026-11-20): correlated views keep this small weight.**
- Per-visit clip as before.
- `L_corr = 0`, `corr_cap = 0`, free votes off reproduces the CHANGE 1 candidates exactly (§6, test).

### 4.4 Thresholds replace `min_views` (the CHANGE 1 tiers stay)
The two tiers of CHANGE 1 keep their grids, their `tau_new` / `tau_new_large`, their connectivity and the
"large replaces small" merge. Only the voxel filter changes:

| CHANGE 1 | CHANGE 2 |
|---|---|
| strict grid: voxels with `views ≥ min_views_small` (2) | voxels with `lv ≥ l_keep_small` (**2.5** = one decorrelated hit; Erik: one observation is enough) |
| lenient grid: voxels with `views ≥ min_views_large` (1), clusters ≥ 40 voxels | voxels with `lv ≥ l_keep_large` (2.5 = one decorrelated hit), clusters ≥ 40 voxels |

Both tiers therefore use the same evidence threshold. What still separates them is the geometry (strict: `tau_new`
0.10, 2-voxel linking, ≥ 5 voxels; lenient: `tau_new_large` 0.05, 26-connectivity, ≥ 40 voxels), and the
"large replaces small" merge. `l_keep_small` stays a config value (5.0 reproduces CHANGE 1).

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
- **Small clusters from a single view (decided by Erik) are the main precision risk.** A 5-voxel cluster (8 cm) from
  one view is hard to tell from depth noise, and new P is already 2%. Their defence is the later see-through
  votes and the output threshold; the dev check reports candidates per pair and the explained share at
  `l_keep_small` 2.5 against 5.0 so the cost is visible.
- **More `new` nodes with low confidence** if `p_new_min` is low. Expected: new R up, new P down at the same
  threshold; the curve over `p_new_min` is the result that matters.
- **False see-through votes** at silhouettes, grazing angles, depth noise; mitigations: the 3×3 minimum-depth window,
  the 0.15 m margin, decorrelation. A real object is not removed by one view (needs 2 free votes against 2 hits).
- **Moving camera, hidden then visible:** a voxel behind an obstacle is no-info, not free. Doors opening or
  sliding objects can still produce see-through votes (limitation, same as for reference nodes).
- **Cost:** about 13–17k tracked voxels per pair × 240–390 frames, vectorised projection; to be measured against the
  current ~3 s per pair. Two grids double it.
- **No class prior (Part A):** anchors lose their extra protection (a bed now needs 2 net free votes like a chair).
  On val the M7 `FULL-priors` numbers say this is neutral overall; the audit veto set shows which GT-removed
  objects are involved.
- **Dev is tiny** (3 pairs, 3 GT-new objects): the thresholds `l_keep_*`, `p_new_min` and the weights cannot be
  tuned reliably there; val is reported once.

## 6. How to evaluate (CLAUDE.md rules)

1. **Equivalence tests** (unit + dev): with `L_corr = 0`, free votes off, `l_keep_small = 5.0`, `l_keep_large = 2.5` the candidates equal CHANGE 1's (same tokens, same points). With `candidates.enabled: false` the code path is CHANGE 1. `FULL-m7` reproduces `m7_final/dev/FULL`.
2. **Dev check** (`dev_train`, candidate geometry as in CHANGE 1): candidates per pair, share explained by GT, candidate recall (new / moved-to), number of candidates deleted by see-through votes, overlays. Tune only here (hard rule 4); report if the evidence is too thin.
3. **Ablation rows on the full `eval_val`** (both modes, `lsg ablate`), each reported once with 95% bootstrap CIs:
   - `FULL` = the new default (Part A and Part B on, CHANGE 1 on);
   - `FULL-candconf` (Part B off = CHANGE 1 candidates, Part A on): isolates Part B;
   - `FULL-candconf-free` (Part B without see-through votes): isolates the free-space effect;
   - `FULL+classprior` (Part A off, the learned prior with `p_del` 0.2): isolates Part A; `FULL-priors` (M7, flat 0.9) has the same margin as Part A;
   - `FULL-relaxed` (CHANGE 1 off) and **`FULL-m7`** (all three changes off = `m7_final` bit for bit, regression check);
   - `FULL` at 3 values of `p_new_min` (precision / recall curve).
4. **Success criterion (written before the run):**
   - new P at equal new R is higher than the baseline at at least one operating point of the `p_new_min` curve, and new R at the default threshold is not lower than the baseline's;
   - false deletion and unchanged kept within the CIs of FULL;
   - removed recall not worse than FULL's CI (Part A);
   - report moved F1, candidates per pair and the share deleted by see-through whichever way they move.
5. **Audit:** the `new` rows of `lsg audit` (§7.4) say whether the removed candidates were real or GT gaps.
6. Log in DECISIONS; update DESIGN §3 (candidate step 4) and §6.11 (confidence) only after sign-off.

No new dependency (numpy / scipy).

## 7. Open questions (answer before implementing)

1. ~~Value of the class-agnostic prior~~: **answered**, no prior value needed (§3): `l0 = 0`, margin via `p_del` = 0.027.
2. ~~Small clusters~~: **answered** (Erik): one observation is enough, `l_keep_small` = 2.5.
3. ~~Low-confidence candidates~~: **answered** (Erik): **drop**. Background: a candidate whose confidence has fallen below `p_new_min` (0.92 = one decorrelated hit), for example a person who was hit once and then seen through (2.5 − 2.5 = 0, p 0.5). Drop it from the graph, or emit it as a `new` node carrying p = 0.5? Decision: **drop** (smaller graph; every candidate stays in `candidates/<pair>.json` with its confidence, and the curve over `p_new_min` is scored from those files). `p_new_min = 0` emits everything, so the choice is a config value, not a code change.
4. ~~Default~~: **answered** (Erik): ON. Every perception number changes again after CHANGE 1; REPORT / README notes are updated when the val result is in.
5. ~~Weight of correlated views~~: **answered** (Erik): they keep the small weight (`L_corr` 0.25, `corr_cap` 1.0).
6. ~~Snapshots / visualisation~~: **answered** (Erik): yes. Snapshots get a candidates sidecar (box, label, confidence per candidate at that frame) and `lsg viz rerun` draws it as `world/candidates`, coloured by confidence, so candidates are seen appearing, being confirmed and being deleted.

## 8. Work list (not started)

- [x] all questions of §7 answered (the last: drop low-confidence candidates)
- [ ] Part A: config flip (`class_priors: false`, `prior_p: 0.5`, `p_del: 0.027`), ablation row `FULL+classprior`, dev check that it reproduces `FULL-priors` and a margin between 1 and 2 votes
- [ ] `perception/voxels.py`: per-voxel log-odds, correlated cap, `kept_by_logodds`; see-through update (`observe_free(depth, K, pose)`); `perception/source.py`: call it per frame, thresholds in `finish`, candidate confidence; `obs/base.py`: `ObsInstance.confidence`; `change/confidence.py`: new node from candidate confidence; emission threshold in the update step
- [ ] config block `confidence.candidates` (`enabled: true`, `L_corr`, `corr_cap`, `free_margin`, `free_window`, `l_keep_small` 2.5, `l_keep_large` 2.5, `p_new_min`)
- [ ] snapshots: per-frame candidate sidecar (`candidates_frame_XXXXXX.json`: box, label, confidence), `viz/rerun_timeline.py` draws `world/candidates` coloured by confidence (the live label stays "candidate"; moved / new are assigned at the end)
- [ ] unit tests: equivalence with CHANGE 1; a transient (box present for a few frames, then removed) is dropped by see-through votes; occlusion and out-of-view give no free vote; silhouette window; correlated cap; determinism; `tests/test_no_leak.py`
- [ ] harness / eval: candidates with confidence saved; `new_pr_curve` helper; ablation rows; `lsg ablate` run by Erik (> 1 min)
- [ ] DECISIONS, DESIGN §3 / §6.11, CODE_STRUCTURE, REPORT note, STATUS
