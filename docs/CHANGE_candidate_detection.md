# CHANGE 1 — relax perception candidate detection (proposal, 2026-11-19)

**Status (2026-11-19): IMPLEMENTED, default ON; val effect not measured yet.** Code: `perception.relaxed` in
`configs/default.yaml`, `perception/source.py`, `cluster.py`, `residual.py`. Every number in `docs/REPORT.md`,
`outputs/m7_final/`, the README headline table and the slides is the **M7 method with strict candidates**, which
is now the ablation row `FULL-relaxed` (`perception.relaxed.enabled: false`, reproduces `m7_final` bit for
bit on the dev pairs). Values and dev evidence: DECISIONS "2026-11-19 — CHANGE 1 implemented".

**Differences from the sketch below** (found on dev_train, see DECISIONS): the lenient tier is 26-connected
(`link_voxels_large: 1`; with 2, 1-view voxels percolate into room-sized blobs), `large_min_voxels` is 40 (not
about 20: 20 adds candidates but no GT-explained ones), and `merge: replace | add` is a switch (`replace` as designed).

## 1. Why

New-object recall in perception mode is 8.8% (oracle 59.6%). Following the 171 scored GT-new objects of val
through the pipeline (DECISIONS M4/M5, B1 numbers; REPORT §5):

| stage | share of the 171 |
|---|---|
| a candidate exists near the object (class-agnostic) | 27.5% |
| ... and CLIP gives a matching label | 17.5% |
| survives association in FULL (final new R) | 8.8% |

The largest loss is the first step: 72% of GT-new objects never produce a candidate. Reasons:
- A voxel is kept only if it was hit from **≥ 3 decorrelated viewpoints** (`perception.min_views: 3`; a counted
  view needs the camera to move ≥ 0.30 m or turn ≥ 20° since the last counted one).
- The method already skips frames (`online.stride: 3`), which leaves fewer counted views per voxel. With fewer
  frames, new R collapses: 8.8 / 5.5 / 0.6 / 0.2% at 100 / 50 / 25 / 10% of the frames (M7 observed-fraction run).
- A pixel counts as "nearer" only if `d_obs < d_exp − 0.10 m` (`perception.tau_new`).

The three views and the 0.10 m rule were written for the worst case, a small object, which is the case where
noise, people and mesh holes dominate. A large object (a chair-sized cluster) is far less likely to be noise.
Erik decided to apply the strict rules only to small clusters.

## 2. What Erik decided

1. **Large clusters: one decorrelated viewpoint is enough** (was 3).
2. **Small clusters (a couple of pixels): 2 viewpoints** (Erik wrote "keep 2"; the current value is **3**, so this
   is also a relaxation. To confirm, see §6).
3. **Relax the nearer-pixel threshold for big objects** (`tau_new`, now 0.10 m for everything).

Not part of this change: the association losses (17.5 → 8.8%), CLIP labelling, the appearance candidates
(thin objects) and the precision filters discussed in chat. They are separate changes.

## 3. Design sketch (to review before implementing)

Size is a property of a *cluster*, but `min_views` and `tau_new` act earlier, on voxels and pixels. So the rule
needs two passes over the same data:

**Tier S (small, strict)**: the existing path, with `min_views_small: 2`. Voxels with ≥ 2 views → connected
components → clusters with ≥ `min_voxels` (5) voxels.

**Tier L (large, lenient)**: a second set of voxels with `min_views_large: 1` (and, for the `tau_new` part, a
second residual grid filled with `tau_new_large` < `tau_new`; the appearance grid `vox_app` is the code
precedent for a second `VoxelEvidence`). Voxels → connected components → **keep only clusters that count as
large**. A tier-L cluster that touches a tier-S cluster is merged into it, not duplicated (same de-duplication
the appearance candidates use, `link_voxels`).

"Large" is a cluster rule, proposed as: voxel count ≥ `large_min_voxels` and largest AABB extent ≥
`large_min_extent` (m). Starting values are not chosen here; they are set on `dev_train` (§5).

Parameter sketch (all in `configs/default.yaml` under `perception:`; nothing else hard-coded):

| key | now | proposed role |
|---|---|---|
| `min_views` | 3 | split into `min_views_small` (2) and `min_views_large` (1) |
| `tau_new` | 0.10 m | stays for tier S; new `tau_new_large` for tier L (floor ≈ 0.05 m, i.e. `surf_dist` and about twice the 2 cm alignment residual; lower values would turn alignment error into candidates) |
| `large_min_voxels`, `large_min_extent` | – | define "large" |

Unchanged: voxel size 4 cm, `link_voxels`, `min_voxels` for tier S, the door filter (applies to both tiers),
`max_depth`, `surf_dist` / `struct_dist`, labelling.

The `tau_new_large` part means a second residual pass per frame (the first stores only pixels nearer than
0.10 m). Cost: a second set of voxels per pair; it should stay within the current ~3 s per pair once caches are
warm, to be measured.

## 4. Expected effects and risks

- **More candidates per pair** (now 8.7, 43% explained by GT). Expected: new R up, new P unchanged or lower.
- **People and moving foliage.** With 1 view, a person standing still for a short time is a large cluster seen from
  one pose. Mesh holes (p1 of `obs − exp` ≈ −0.30 m on an unchanged scene) give large, flat clusters. Mitigations
  to test: the door filter, the structure filters, and a flatness test on tier-L clusters (rejecting
  wall-parallel slabs).
- **Moved objects.** Candidates also feed association. More large candidates can mean more moved matches
  (good for moved R, risk for same-label swaps) and more false `new` nodes.
- **False-deletion rate.** Candidates only affect which observations exist. Votes and the free-space rule are not
  changed. Still, check that the false-deletion rate and unchanged kept do not get worse.
- **Determinism.** The two-tier clustering must keep the candidate order deterministic (size descending, then
  centroid), so the same config and seed give the same `metrics`.

## 5. How to evaluate (rules of CLAUDE.md)

1. **Tune on `dev_train` only** (hard rule 4). Caveat: it has only 3 pairs, 3 GT-new objects and 15 moves, so it
   cannot choose values reliably; DECISIONS 2026-11-05 already records this limit. Use the candidate-level
   numbers (`metrics.candidate_counts`: candidate recall, share explained by GT, by size class) and look at
   overlays, not only the end metrics.
2. **Ablation row** `FULL+relaxed-candidates` (`lsg ablate`) beside FULL, both modes, on the full `eval_val`
   (110 pairs, ~3 min per row), reported once with 95% bootstrap CIs. No second val run to adjust values.
3. **Report all of:** candidates per pair, share explained by GT, candidate recall (class-agnostic and
   labelled), new P / R, moved P / R / F1, false deletion, unchanged kept, removed P / R.
4. **Success criterion (written before the run):** new R up in perception mode (8.8% now), false-deletion rate and
   unchanged kept within the CIs of FULL, new P and moved F1 reported whichever way they move. If new P falls
   further, say so; it is the cost of the trade.
5. **Audit:** the `new` rows of `lsg audit` (§7.4) say how many of the added false candidates are GT gaps.
6. Log in DECISIONS (values, dev evidence, val result). Update DESIGN §3 steps 4–5 and the "≥ 3-view voxels"
   mitigation in DESIGN §10 only after sign-off, as the CLAUDE.md rule on spec changes requires.

No new dependency (numpy / scipy only).

## 6. Open questions (answered by Erik, 2026-11-19: small = 2 views; `tau_new`; ON by default; large about 20 voxels, voxel count only)

1. **Small clusters: 2 or 3 views?** The config has `min_views: 3` today. "Keep 2" suggests 2 was assumed; this
   document proposes 2 (relaxed from 3). Say if 3 should stay for small clusters.
2. **"Relax the threshold for big objects"**: read here as the nearer-pixel threshold `tau_new` (lower for large
   clusters). The other reading is `min_voxels` (cluster size). Say which one is meant, or both.
3. **Definition of "large"** (voxel count and extent; see §3).
4. **Appearance (thin-object) candidates**: left unchanged. They share `min_views` today (`source.py`
   `_appearance_candidates`), so the split must keep their behaviour at the current value of 3 or be applied
   deliberately.

## 7. Work list

- [x] confirm §6 with Erik
- [x] config keys, `perception/source.py` (second grid, `finish`: two tiers), `cluster.py` (`is_large`, `link`, `min_voxels` arguments), `residual.py` (`tau` argument); `voxels.py` unchanged
- [x] unit tests (`tests/test_perception.py`): large box from one view, small box needs 2 views, merge replace / add, determinism, neutral values = strict path, link knob, `tau` override; ablation row and audit override tests
- [x] `tests/test_no_leak.py` still passes
- [x] dev_train check: strict path = `m7_final/dev/FULL`; candidate-level variants (DECISIONS); overlays not drawn
- [x] DECISIONS, DESIGN §3 step 4 / §6.6 / §10, CODE_STRUCTURE, README, REPORT note, STATUS
- [ ] **Erik:** `uv run lsg ablate --names FULL,FULL-relaxed --split eval_val` (both modes, > 1 min), then judge against §5.4
- [ ] **Erik:** `uv run lsg m7 --run-id m7_relaxed` (111 min) to refresh REPORT / slides if the relaxed rules stay the default
- [ ] slides: footnote that the numbers are the strict method (rebuild `docs/slides.pptx`), after the val result
