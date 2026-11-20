# DECISIONS — dated log

Format: **date — decision.** Why. *Alternatives considered.* (status: fixed / provisional / to verify)

## 2026-11-04 — before the spec (Erik)
- **Framing: object-level semantic graph maintenance** (identity, relations, confidence/deletion), not class-agnostic geometric change detection. Prior work (Objects Can Move, Has Anything Changed?) already covers that; this project's value is the graph. *Alt: compete on point-level change detection.* (fixed)
- **Reference graph from GT**: `semseg.v2.json` + 3DSSG support relations mapped to on/in/next_to; anchor/thing split by class. The brief says "given the graph of the reference scan". (fixed)
- **Rescan into the reference frame** via 3RScan.json `T_align`; no own alignment or pose estimation. (fixed)
- **Visibility via render-and-compare** of expected vs. observed depth; **only free-space evidence lowers confidence**. (fixed)
- **Oracle rescan instances first**; a real detector (Grounding DINO + SAM 2) is a stretch goal. (fixed)
- **Association**: Hungarian matching on class + geometry + appearance; position only weakly weighted. (fixed; appearance is an ablation)
- **States**: unchanged / moved (ICP) / removed (needs free evidence) / unknown → keep / new. (fixed)
- **Relations recomputed with geometric rules** (footprint contact + height). (fixed)
- **Confidence**: log-odds, optional class-dependent priors from the train split. (fixed)
- **Baselines** B0 → B1 (+visibility) → B2 (+appearance) → full; observation fraction 100/50/25/10% as contiguous chunks. (fixed; renumbered 2026-11-05, see below)
- **Workflow**: repo docs are the source of truth for Claude Code; milestones M0–M7 with acceptance checks. (fixed)

## 2026-11-05 — spec session (Claude, confirmed by Erik where noted)
- **Package manager uv; Python 3.11.** Erik's preference. (fixed)
- **Renderer: Open3D `RaycastingScene` on CPU.** Measured ~5 ms per 224×172 depth frame and ~50 ms at 960×540, so GPU rendering would not save meaningful time. `primitive_ids` give instance masks from the same cast. Rays built from intrinsics return z-depth (verified). *Alt: pyrender (needs EGL/OSMesa headless), rio_renderer (C++/OpenGL build).* (fixed)
- **Relation scope: exactly one support parent per thing (`on`|`in`, parent = anchor, thing or `floor`); `next_to` exported but not a headline metric.** Erik agreed. *Alt: full multi-relation 3DSSG graph — noisier to evaluate, much more work.* (fixed)
- **Headline = oracle instances; reference graph = GT.** The brief describes the input as the reference graph; the oracle isolates the change logic. Every table is labelled `oracle`. (provisional — ask organisers, non-blocking)
- **Pose convention camera→world, depth mm/1000, +Z up, images rotated 90°.** Verified on 19eda6f4 (back-projection residual 0.9–1.5 cm; inverse convention 0.5–2.6 m) and on floor normals of 4 scans. (fixed)
- **3RScan.json transforms are column-major and treated as metres**, contrary to the FAQ's "mm". Evidence: scan-alignment translations have median 1.0 m and max 9.7 m, which in mm would be implausible (≤ 1 cm). (to verify at M0 with a mesh overlay)
- **Direction/frame of `rigid` transforms unknown** → M0 Chamfer check before any GT displacement is computed. (to verify)
- **Handle bare-int `rigid` entries** (390 of 3,323): moved, but with no GT transform → displacement unknown → counted as moved, excluded from the displacement-bucket analysis. (fixed)
- **"Removed" GT = the listed `removed` only.** In val, 1,018 reference objects are missing from the rescan annotation, but only 96 are listed removed: missing ≠ removed because rescans are partial. The rest form `absent-unlisted`, used for a false-deletion rate. *Alt: derive removals by ID difference (Objects Can Move style) — would turn every unscanned object into a "removal".* (fixed)
- **"New" GT = rescan IDs not in the reference** (things/anchors only), audited manually because it is noisy. (fixed)
- **Anchor/thing/structure split by 3DSSG `nyu40`** (table in DESIGN §3). It is per-object in `objects.json`, so no extra class-mapping CSV is needed. Chairs are things. (provisional — check the anchor label list at M1)
- **Data subset**: dev = 3 train pairs; main eval = all 110 val pairs; relation-change eval = 40 train pairs with the most parent-changed moved things (73 events). Val alone has only 12 such events. Downloads: semseg + PLY + sequence only (≈ 8 GB tier 1, ≈ 3 GB tier 2). Lists in `splits/`. Erik prefers not to download the full dataset. (fixed)
- **Tune only on dev**; class priors from train metadata excluding the dev and relchange references. (fixed)
- **Visibility thresholds start at τ_near = 0.10 m, τ_free = 0.15 m.** On a scene compared with its own mesh, only 0.4% of pixels exceed +0.15 m, while p1 of (obs − exp) is −0.30 m (thin or unreconstructed occluders → treated as occluded). (provisional, tune on dev)
- **Frame stride 3.** Consecutive frames are only ~6 cm / 5° apart; a 10 cm / 10° keyframe rule kept 398/717 frames, so it saved little. (provisional)
- **Moved = translation-only test in v1** (ICP centroid displacement > 0.10 m); rotation-only moves are out of scope. Recall is reported by displacement bucket. (provisional, τ fixed at M1)
- **Baselines renumbered**: B0 naive, B1 + visibility, B2 + Hungarian geometry, B3 + appearance, FULL + confidence/priors. Splitting association from appearance makes the appearance ablation clean, and B3 can be dropped if time runs short. (fixed)
- **Removed nodes are kept in the JSON with `state: "removed"`**; deletion is a view filter (`--prune`). Keeps before/after visualisation and audits simple. (fixed)
- **`bbox_3d` = axis-aligned {min, max} in the reference frame.** (provisional — confirm the shared-format convention with the organisers)

## 2026-11-05 — M0 (Claude Code, confirmed by Erik where noted)
- **3RScan.json transforms are metres; `T_align` maps rescan → reference.** M0 check (2): rescan structure vertices lie 1.6 / 1.9 / 3.0 cm (median point-to-surface) from reference structure on the 3 dev rescans; `T⁻¹` gives 12–204 cm, translation ×0.001 gives 6–46 cm, identity 7–63 cm. Never scale by 0.001. (fixed)
- **`rigid` transforms map reference-frame points into the rescan frame (p_S = T_rigid · p_R).** M0 check (3): hypothesis H2 wins on 20/21 dev objects (Chamfer 2.8–7.8 cm, 17 < 5 cm); H1 (both in reference frame) gives 5–358 cm. Motion in the reference frame is `M_ref = T_align · T_rigid`, so **DESIGN §5's GT displacement is corrected to ‖T_align·T_rigid·c_R − c_R‖** (Erik approved). *Alt: the original formula, which mixes frames and would give metre-scale fake displacements.* (fixed)
- **Refine `T_align` with label-free ICP on the method side** (`geom.align`; Erik approved). Raw `T_align` leaves a 2.9–3.8 cm median depth residual (M0 gate 3 cm; own-mesh residual is 1.1–1.3 cm), and a direct check on unchanged same-label pixels still gives 2.4–3.1 cm, so the error is in `T_align`, not change contamination. Robust point-to-plane ICP (Tukey, max corr. 10/5/3 cm) of back-projected rescan depth (every 10th frame, 2 cm voxels, ≤ 4 m) onto reference mesh vertices+normals, init `T_align`, corrects 1.6–8.4 cm / 0.4–1.5° and gives 1.9–2.2 cm. Uses no rescan labels (hard rule 1). Corrections > 20 cm or > 5° are rejected (raw kept). GT/eval keeps the raw 3RScan transforms. *Alt: relax the gate to 4 cm and decide at M3; gate on own-mesh render; ICP on structure only (needs rescan labels → not allowed on the method side).* (fixed; ICP params provisional in `configs/default.yaml`)
- **M0 check (1) gates on the reference mesh rendered at refined `T_align @ pose`** (what the visibility votes use), not on the rescan's own mesh (reported too). (Erik) (fixed)
- **Check (2) uses point-to-surface distance** (Open3D `compute_distance` to the reference structure faces) rather than vertex-to-vertex NN, which would add mesh-sampling distance. (fixed)
- **Depth PGM is big-endian uint16** (PGM spec). Little-endian reading gives a median of 36 m. (fixed)
- **Pixel convention: integer pixel coordinates, no +0.5 offset** — x = (u − cx)·z/fx, for both back-projection and rendering rays (consistent with each other; offsets of 0.5 px are below depth noise). Ray directions are unnormalised with camera-z = 1, so `t_hit` is z-depth. (fixed)
- **Sequences are read straight from `sequence.zip`** (random access via `zipfile`), or from `sequence/` if extracted. Avoids ~8 GB of extracted files. (fixed)
- **Rescan access API split** for hard rule 1: the method reads rescans only through `open_sequence` / `iter_frames(seq, stride, idx)` (RGB-D + poses); `load_scan` (labels) on a rescan is only called from `oracle/` and `eval/`. DESIGN §6.1's `iter_frames(scan, …)` therefore takes a `SequenceReader`, not a `Scan`. (fixed)
- **Depth border**: stored depth columns 0–~20 and ~205–223 are never valid (≈ 18% of pixels). These pixels give "no information" under §3 automatically. (fixed)
- **Dependencies**: `numpy<2` (Open3D wheels are built against numpy 1.x), `scipy` (cKDTree for Chamfer), `open3d` (raycasting, ICP), `pillow` (JPEG/PNG), `pyyaml` (config); dev: `pytest`. Open3D pulls dash/flask/etc. transitively; its pytest plugin is disabled with `-p no:dash`. Own ASCII PLY reader instead of `plyfile` (30 lines, 1.9 s for a 145k-vertex scan). (fixed)

## 2026-11-05 — Spec change v2: perception headline, oracle upper bound (Erik; drafted by Claude Code)
- **Headline = perception mode; oracle = upper bound.** The method's input is only the reference graph + reference scan data (mesh, texture) and the rescan's RGB-D frames + poses + `T_align`. Oracle instances fail the brief as a headline. Supersedes "Oracle instances first" (2026-11-04) and "Headline = oracle instances" (2026-11-05). Every results table gets an `oracle` and a `perception` row. *Alt: oracle-only headline with a detector as stretch goal (v1).* (fixed)
- **One interface `ObsSource` → `ObsInstance`** (token, label, label_topk, nyu40, points in the reference frame, per-frame masks at depth resolution, upright crops), implemented by `oracle.OracleSource` and `perception.PerceptionSource`. The change logic only sees the interface, so both modes share association, classification, update and eval. (fixed)
- **New hard rule: `perception/` never imports `oracle/` or `eval/`** and never reads rescan semseg / PLY labels / 3DSSG or the `rigid/removed/nonrigid` lists. Enforced by an import-check test from M2 on. (fixed)
- **Frame-sequential (online) processing** in time order with the config stride; per frame: render → votes → confidence → residual accumulation; snapshots every N frames. (fixed)
- **Viewpoint decorrelation**: a vote (per node and kind) or voxel hit counts only if the camera moved ≥ 0.30 m or ≥ 20° since the last counted one. The per-visit log-odds clip [−3, +3] stays. Why: consecutive frames are ~6 cm / 5° apart, so raw frame counts mostly measure how long the camera lingered. *Alt: clip only (v1); keyframe selection (saves little, see stride decision).* (provisional, tune on dev)
- **Deferred commitment**: `moved` / `removed` are final only at the end of the sequence (or chunk); earlier states are tentative. Why: a moved object can appear at its new place hundreds of frames after its old place was seen empty. *Alt: immediate commitment — kept as an ablation.* (fixed)
- **ICP alignment refinement stays a pre-pass**, run on the processed chunk's frames only (signature gains `frame_ids`). It is not strictly online, but commitment is deferred to the end of the chunk anyway and it uses only allowed inputs. *Alt: incremental refinement during the walk — more complex, little gain expected.* (provisional)
- **Perception candidates from depth residuals**: nearer pixels (d_obs < d_exp − τ_new, τ_new = 0.10 m), back-projected; drop points within 5 cm of the reference mesh and near structure (except standing on the floor); 4 cm voxels kept if hit from ≥ 3 decorrelated viewpoints; DBSCAN / connected components with a minimum size. Why: the multi-view rule filters people, sensor noise and mesh holes (p1 of obs − exp ≈ −0.30 m on an unchanged scene). *Alt: full-frame detector (Grounding DINO + SAM 2) — out of scope; per-frame clustering — noisier.* (provisional, tune on dev)
- **Appearance residual for thin objects** (< ~10 cm, invisible to depth): render the reference texture at the rescan pose (Open3D `primitive_uvs` → OBJ per-face UVs → `mesh.refined_0.png`), compare with observed colour only where depth agrees, using DINOv2 ViT-S/14 patch-feature cosine distance (fallback: mean/std-normalised colour). Used as free-like votes on thin nodes and as thin-object voxels; never deletes a node alone (≥ 3 agreeing viewpoints). This gives the brief's depth-only vs depth + appearance ablation. *Alt: raw RGB difference — not exposure-robust.* (provisional)
- **"Appearance" now means the appearance residual.** DINOv2 *embedding-based association* (old B3) becomes an optional ablation, off by default. Baselines renumbered: B0, B1, B2, FULL (DESIGN §8). (fixed)
- **Candidate labelling with CLIP ViT-B/32**, zero-shot on the K = 5 largest upright crops; vocabulary = reference-graph labels ∪ most frequent 3RScan thing/anchor labels from train metadata (excluding dev and relchange references). Stores `label_topk`. Fits a GTX 1650 (4 GB). *Alt: Grounding DINO as labeller — optional only.* (provisional)
- **Association**: reference nodes confirmed in place (≥ 3 counted seen, 0 counted free) are `unchanged` without a match and skip Hungarian matching. In perception mode the label term is soft, −log max(p(label), p_min), instead of the hard gate. (provisional)
- **Perception eval matching**: same label (or nyu40) and (centroid ≤ 0.30 m or 3D IoU ≥ 0.25), greedy by IoU, one-to-one; reference nodes are compared by id in both modes. New metrics: moved localisation error, new/moved recall by size (`small` = thing with max extent < 0.5 m). (fixed)
- **Determinism** is defined on the `metrics` block of metrics.json (`created`/`run_id` differ by design); GPU features are cached per (scan, frame, model). (fixed)
- **Textures for the 72 reference scans** of dev + val + relchange (`splits/download_textures.txt`, written by `lsg splits textures`; ≈ 17 MB/scan → ≈ 1.2 GB). Rescans don't need them. OBJ/PLY face-order correspondence is checked at M6 before relying on it. (fixed)
- **Milestones re-scoped to M1–M8** (DESIGN §9): M0 kept as finished; new order graph+GT → oracle+eval+B0 → online loop+B1 → perception → association+update → confidence+appearance → ablations → viz+write-up. Cut order defined; the depth vs depth + appearance ablation is not cuttable. (provisional — Erik to confirm)
- **Planned dependencies** (each gets its own entry when added): `torch` (CUDA build for the GTX 1650, CPU fallback), `open_clip_torch` (CLIP), DINOv2 via `torch.hub` or `transformers` (decided at M6), `scikit-learn` (DBSCAN), `rerun-sdk` (M8). (planned)

## 2026-11-05 — M1 (Claude Code, confirmed by Erik where noted)
- **τ_eval_move = 0.10 m** (Erik). Of 370 unique thing/anchor moves in val, 134 are < 0.10 m (median 0.17 m, p25 0.06 m; dev: 5 / 20). These become `dont_care` for moved/unchanged; 235 scored moves remain. 0.10 m is above the method's resolution floor (~2 cm alignment residual, 4 cm voxels) and equals the method's τ_move start value. The method's τ_move is still tuned on dev only. *Alt: 0.05 m (79 dont_care; 5–10 cm moves near the depth noise floor), 0.20 m (214 dont_care, too few positives).* (fixed)
- **"New" count: 224 raw, 171 scored** (Erik). §2.2's 224 is the raw rescan-ID difference including structure (doors, pictures, …). §5 scores things and anchors only → 171 (105 large, 66 small). `lsg m1` reports both; §2.2 amended. *Alt: score all 224 — would make new structure graph nodes, against §3.* (fixed)
- **Parent-change counts are mostly annotation gaps** (Erik: report now, decide §7.2(c) at M5). §2.2's 12 (val) / 73 (relchange) count any difference in the support-parent set. Breakdown: val 7 gained (no reference parent annotated), 1 lost, 4 swapped (+4 with structure-only parents); relchange 47 gained, 4 lost, 18 swapped, 4 structure-only. Only the swaps are clearly real parent changes. (open → M5)
- **GT status precedence**: listed removed (→ `dont_care` if still in O_S) > nonrigid > rigid (→ `dont_care` if < τ_eval_move; bare-int → moved with unknown displacement) > present in O_S → unchanged > absent-unlisted. Removed and nonrigid are explicit annotations of a different change kind, so they win over `rigid`. (fixed)
- **Duplicate `rigid` ids are deduplicated** per rescan. Val has 7 duplicate entries (1d234024, 4a9a43d4), all with identical transforms, so nothing is lost. Raw counts (390) still count entries, to match §2.2. (fixed)
- **Object class for levels: 3DSSG `objects.json` nyu40 first, PLY vertex majority as fallback** (§3 says 3DSSG). O_R / O_S are the semseg objectIds (`Scan.semseg_labels`, new field). (fixed)
- **GT-mode reference graph: one parent per thing, chosen deterministically** when 3DSSG lists several: anchor > thing > floor, then `on` > `in`, then the lowest id. If one child has both `on` and `in` to the same parent, `on` is kept. A thing whose only support parents are non-floor structure (wall, ceiling, …) is excluded; a structure parent next to a valid parent is just dropped. *Alt: exclude any thing that has a structure parent — loses representable relations.* (fixed)
- **Anchors carry no support edge** (§4: "at most one per thing"); their `anchor` field is null. Anchors on anchors (e.g. a shelf on a cabinet) are therefore not represented. (provisional)
- **Floor height = median z of the reference's floor-labelled vertices** (nyu40 = 2), with the 1st percentile of all z as fallback (< 100 floor vertices). Reference annotations are allowed method input. (fixed)
- **Relation rules tuned on the 2 dev references only** (n = 28 things with a GT parent; hard rule 4). Default rules: 78.6%. Changes:
  1. **`support_min_height` = 0.05 m (new rule):** supporting candidate points must lie at least 5 cm above the floor. This fixed a basket standing on the floor that was "supported" by the washing machine's bottom edge.
  2. **`floor_max_gap` 0.10 → 0.20 m:** the bottoms of 3 boxes/items were unreconstructed, 16–19 cm above the floor.

  Result: 92.9% (26/28). The two remaining errors look like GT or mesh issues: a plant on a stand that 3DSSG puts on the floor, and a stand whose lowest vertex is 47 cm up. `contact_min_frac` 0.2–0.5, `contact_xy` 3–8 cm, `contain_min_frac` 0.5–0.8 and `bottom_slice` 2–5 cm made no difference on dev, so they keep the spec values. Val references were evaluated once, after tuning, and not tuned on: 85.2% parent accuracy (n = 575), 85.9% without floor parents; type accuracy given the correct parent 85.3% (75.5% without floor), mostly `on` vs `in` on shelves/cabinets. (provisional; revisit with more dev data if relations matter at M5)
- **Geometric `next_to`** = AABB gap ≤ 0.15 m between any two anchor/thing nodes; GT `next_to` = 3DSSG `close by`. Exported only, not scored. (provisional)
- **Dependency: `matplotlib`** (≥ 3.8) for the M1 displacement histogram; also planned for the M7 ablation plots. Pure-Python wheels, no numpy-2 conflict. (fixed)
- **Texture download done**: 72 references × {`mesh.refined.v2.obj`, `mesh.refined.mtl`, `mesh.refined_0.png`} = 216 files, 1.1 GB, via `download_files.py --id … --type …`. For the 23 relchange references only the texture files exist so far (tier 2 is still pending). (fixed)

## 2026-11-06 — after M1 (Claude Code; Erik's decision pending where noted)
- **Anchor label list checked** (open item from 2026-11-05). On the dev + val references, anchors are shelf, cabinet / kitchen cabinet / bath cabinet / commode, table / coffee / side table, sofa / couch, counter, desk, bookshelf, bed and nightstand: all as intended. nyu40 39 "otherfurniture" (tv stand, bench, rack) are things; they can still be support parents, but a TV's `anchor` field stays null. (fixed)
- **Doors are not scored; door swings are out of scope for now** (Erik's decision pending on the optional diagnostic). The 12 moved doors in val nearly all rotate 4–120° about the vertical axis, so they are hinge swings (articulation), not rearrangement. Doors are structure (§3), so they have no node to score. Scoring them as `moved` would count a swing as a ~0.5 m centroid move and would mix articulation into moved F1. Recommendation: keep doors out of the change metrics; at M4, stop swung doors from producing false "new" candidates (add reference door surfaces to the structure filter); *optionally*, after M7, add a separate door open/closed diagnostic (n = 12 in val). *Alt: door nodes with a `state` attribute in every pair, scored in all metrics — new scope, a hinge model is needed.* (provisional)
- **Handoff file `docs/STATUS.md`** (Erik: one chat per milestone). It records finished work and open items per milestone; CLAUDE.md now says to read it at the start of a milestone and update it at the end. DESIGN and DECISIONS stay the source of truth. (fixed)

## 2026-11-06 — M2 (Claude Code, confirmed by Erik where noted)
- **Scoring universe for reference objects** (Erik). Moved P/R, removed P/R and unchanged accuracy are computed over things/anchors with GT status `moved | removed | unchanged`; `dont_care` and `absent_unlisted` are excluded from all three. Why: the GT state of `absent_unlisted` objects is unknown (mostly unscanned parts of a partial rescan). *Alt: `absent_unlisted` as negatives for moved — would mix GT gaps into moved precision.* (fixed)
- **False-move rate** (Erik): share of `absent_unlisted` objects predicted `moved`, reported next to the false-deletion rate (share predicted `removed`). Why: without it a `moved` claim on these objects would go unmeasured while a `removed` claim is measured; both are changes the GT cannot confirm, and both go into the §7.4 audit. These objects have no oracle instance, but B0 can still pair their node with a *different* same-label instance (val, oracle: 2.4%). (fixed)
- **Unchanged: two numbers** (Erik): strict accuracy (predicted `unchanged`) and **kept rate** (predicted `unchanged` or `unknown`). Why: from B1 on, `unknown` is the correct answer for an unseen object; strict accuracy alone would hide B1's main gain over B0. The unknown rate (over all scored reference nodes, incl. `dont_care`/`absent_unlisted`) is reported separately. (fixed)
- **Predicted new on a `dont_care` reference object is ignored** (Erik): neither TP nor FP. Oracle: its token maps to a `dont_care` objectId; perception: it passes the §5 rule against that object's rescan instance (raw `T_align`), checked only after matching to GT new. Other unmatched predictions are FP. (fixed)
- **B0 matching is greedy one-to-one** (Erik): all same-label (node, obs) pairs sorted by centroid distance (ties: node id, token). One observed chair cannot explain two reference chairs, so leftover nodes become `removed` and leftover observations `new`. Exact top-1 label equality, `near_dist` = 0.5 m in `baselines.b0`. Moved nodes get the observed centroid, the bbox shifted by the same translation, and `T_move` = that translation. *Alt: each node takes its nearest obs independently (many-to-one) — inflates `unchanged` and hides new objects.* (fixed)
- **Oracle pixels need valid sensor depth.** An instance pixel counts only where the rendered rescan mesh hits it **and** `d_obs > 0`. The ~18% border columns that are never valid (§2.1) therefore observe nothing, as for perception. Points = unique vertices of hit faces whose face label is the instance, mapped with the **refined** `T_align` (the method's alignment; GT/eval keeps the raw one). Structure instances are emitted; the method ignores observations whose nyu40 is structure, so both modes are treated alike. *Alt: rendered label only — the oracle would see objects in pixels the sensor never measured.* (fixed; `oracle.min_points` = 50 provisional)
- **Oracle tokens** `o<k>` come from a permutation seeded with `[oracle.seed (+ run seed), crc32(rescan_id)]` (stable across processes, unlike Python's `hash`). The token → objectId map is read only by `eval/`. (fixed)
- **Masks are stored bit-packed** (`obs.PackedMasks`, a read-only `{frame: bool[172,224]}` mapping). Raw bool masks of every instance in every processed frame would take ~230 MB per pair. (fixed)
- **Refined `T_align` is cached** in `cache/align/<rescan>_<key>.json`; the key hashes the reference id, raw `T_align`, the `alignment` config block and the processed frame ids, so a chunk (§7.3) or an ICP setting change gets its own entry. `refine_from_sequence` gains `frame_ids` (planned in v2). (fixed)
- **Method side never sees the rescan's `ScanMeta`**: `change.pipeline.prepare_pair` takes only the raw `T_align` matrix, because `ScanMeta` also carries the `rigid/removed/nonrigid` GT lists. (fixed)
- **Eval harness in `eval/harness.py`**; process pool over reference groups (`eval.workers` = 4), results sorted back into split order. `metrics.json` keeps everything deterministic under `metrics` (incl. per-pair counts); wall times, worker count, alignment info and cache hits go to a separate `run` block (`write_metrics(..., extra=)`), so §7.5 compares `metrics` only. Undefined metrics (n = 0) are written as `null`, not NaN. (fixed)
- **Association accuracy (oracle)**: GT-matchable = reference objects with GT status moved/unchanged whose rescan instance is among the oracle instances of the processed frames (≥ `min_points`). Strict: the node's `obs_token` maps to its own objectId; same-label tolerant: to an object with the same label. For perception (M4) eval must compute the GT-visible set itself (same oracle visibility pass, run in `eval/`). (fixed for oracle)
- **GT objects without a graph node** (semseg id without PLY vertices) are counted (`ref_no_node`) but not scored. 0 on dev. (fixed)
- **Frame chunks** (§7.3): `select_frames` = every `stride`-th frame, then one contiguous run of `round(frac · n)` frames with a start drawn from `[seed, crc32(rescan_id)]`. (fixed)
- **No new dependencies.** (fixed)
- **Determinism with fresh ICP** (§7.5). Two cold val runs differed only in localisation errors at ~1e-15 m (Open3D's multi-threaded ICP is not bit-exact; `T_align` differed by 5e-15 m); all counts were identical. Localisation errors are now rounded to 1 µm. Result: two cold runs and one cached run give identical `metrics`. A decision that flips on a 1e-15 m difference is possible in principle but was not observed; cached reruns are bit-exact anyway. *Alt: single-threaded Open3D (slower), or round `T_align` itself.* (fixed)
- **M2 acceptance result, B0 oracle on val (110 pairs)**: moved P 38.2 / R 24.7 / F1 30.0%; removed P 21.2 / R 69.9%; new P 85.0 / R 59.6%; unchanged acc 80.9%; **false-deletion rate 95.0%** (n = 737), false-move rate 2.4%; association 91.4% strict / 97.4% same-label (n = 1,472); moved recall 0% in [0.1, 0.3) m (below `near_dist`), 27.6% in [0.3, 1) m, 89.5% ≥ 1 m; localisation error median 6.0 cm. As DESIGN §8 expects: B0 deletes nearly everything it doesn't see. It also loses new objects to vanished same-label reference nodes (dev: a new bag and two new pillows took over the matches of a moved bag / pillows). Reported only, nothing tuned. (fixed)
- **ICP refinement rejected on 5 / 110 val pairs** (correction > 20 cm or > 5°); the raw `T_align` is kept there. Revisit at M3, when free-space votes depend on alignment. (open → M3)
- **Tier-2 download done** (Erik asked, 2026-11-06): 63 scans × {`semseg.v2.json`, `labels.instances.annotated.v2.ply`, `sequence.zip`} = 189 files, 4.0 GB (DESIGN's ≈ 3 GB estimate was low), via `download_files.py --id … --type …` (single-scan mode, no prompt). Verified: no failures, all zips pass `testzip`, all 63 scans load. The relchange split (§2.3) can now be run end to end. (fixed)

## 2026-11-06 — M3 (Claude Code, confirmed by Erik where noted)
- **Scope of tuning at M3** (Erik): spec threshold values (τ_near 0.10, τ_free 0.15, 50%, 15 px, d_dec 0.30 m / θ_dec 20°, k_free 2, k_seen 3), plus a dev diagnostic for false free evidence, plus an optional mask-erosion knob `visibility.erode_px` (default 0), to be tuned on dev only if the diagnostic shows silhouette problems. *Alt: full dev sweep of τ_free / vote fraction / erosion now (3 dev pairs, too noisy); spec values without a diagnostic.* (fixed)
- **Erosion stays off** (`erode_px` = 0). On dev, 4 / 34 GT-unchanged objects get ≥ 2 counted free votes at erode_px 0, 1 and 2 alike, so silhouette edges are not the cause. The 4 are pillow r49 on 19eda6f4 (10 counted free, 1 seen: the probable unlisted move of DESIGN §2.1) and three "item"s on e2847ff7 (3–7 counted free mixed with 12–19 seen). Unchanged matched nodes are not affected in B1; this matters for B2's confirmed-in-place rule and FULL's confidence. (fixed)
- **Vote needs ≥ `min_mask_px` *valid* pixels.** |P_i,f| < 15 → `unobserved` (§3). If |P_i,f| ≥ 15 but fewer than 15 of them have valid sensor and rendered depth → `noinfo`. Otherwise `seen` / `free` at ≥ `vote_frac` = 50% of the valid pixels. Why: the never-valid border columns (§2.1) would otherwise let 1–2 pixels decide a vote. (fixed)
- **Decorrelation** per key `(node, kind)` for kinds `seen` and `free`, against the pose of the last *counted* vote: the camera centre moved ≥ d_dec, or the optical axis (camera +z in the reference frame) turned ≥ θ_dec. The first vote of a key always counts. `noinfo` / `unobserved` are not counted. `decorrelation.enabled: false` is the ablation switch. (fixed)
- **Tentative states in snapshots are vote-only**: `removed` with ≥ k_free counted free votes, `unchanged` if confirmed in place, else `unknown`. Association is deferred to the end of the sequence, so snapshots carry no `moved` and no `new`. Final states come from the method after `finish()`. (fixed)
- **B1** = B0's greedy same-label matching (unchanged); an unmatched node is `removed` only with ≥ k_free counted free votes, else `unknown`. Free votes of matched nodes are ignored. `last_seen` = rescan for unmatched nodes with a raw `seen` vote. Every method now takes `votes=` (`OnlineResult.votes`); B0 ignores it. (fixed)
- **`run_online(ref_ctx, seq_info, seq, obs_source, cfg, snapshot_dir, obs_mode, renderer)`**: G_ref comes from `ref_ctx.graph` and the frame ids from `seq_info`, instead of DESIGN §6.7's separate `G_ref` / `frame_ids` arguments (same information; DESIGN updated). The harness builds one reference `Renderer` per reference group. `change.pipeline.collect_observations` is removed. (fixed)
- **Snapshots**: every `online.snapshot_every` = 20 processed frames, plus one after the last frame. Written by `lsg run` by default (`--no-snapshots`); `lsg eval` only with `--snapshots` (110 pairs × ~10 files otherwise). Path `outputs/<run>/snapshots/<mode>/<ref>_<rescan>/frame_XXXXXX.json`: §4 plus a `<mode>` level, so both modes can be written in one run. (fixed)
- **Evidence diagnostics in `metrics`** (method-independent, from the online votes, computed in `eval/`): `free_ev_unchanged` / `free_ev_removed` / `free_ev_absent` = share of GT-unchanged / removed / absent-unlisted objects with ≥ k_free counted free votes; `seen_ev_absent` = share of absent-unlisted objects with any counted seen vote. Reported only, not a headline metric. (fixed)
- **ICP-rejected pairs give no extra fake free votes** (STATUS M3 open item, closed). On the 5 val pairs whose refinement was rejected (raw `T_align`), 5 / 75 GT-unchanged objects have free evidence (6.7%), vs 83 / 1,326 (6.3%) on the other 105 pairs; their false-deletion rate is 1 / 63. τ_free = 0.15 m is well above the 3–4 cm raw residual. Reported on val, nothing tuned. (fixed)
- **M3 acceptance result.**
  - **Dev, B1 oracle:** false-deletion rate **16.7%** (2 / 12) vs B0's 100%; unchanged kept 97.1%; unknown rate 20.5%.
  - **Val, B1 oracle (reported once):** false-deletion rate **9.5%** (n = 737) vs B0's 95.0%; removed P **66.0%** / R 45.2% (B0: 21.2 / 69.9); unchanged kept 92.9%; unknown rate 31.6%.
  - **Unchanged from B0** (same matching): moved, new and association metrics.
  - **Evidence on val:** 60.3% of GT-removed objects have free evidence. The rest are unseen in the processed frames, or depth agrees where they were. Example: dev pillow r50 on e2847ff7 (GT removed) gets 23 counted seen votes and 0 free. Either a thin object whose supporting surface lies within τ_near (the M6 appearance-residual case) or a GT error; to check in the §7.4 audit.
  
  (fixed)
- **No new dependencies** (`scipy.ndimage` for erosion is part of scipy, already a dependency). (fixed)

## 2026-11-07 — M4 (Claude Code, confirmed by Erik where noted)
- **Dependencies: `torch` 2.5.1 + `torchvision` 0.20.1, CUDA 12.1 build, and `open_clip_torch`** (Erik: CUDA build now).
  - Source: an explicit `pytorch-cu121` index in `pyproject.toml` (`[[tool.uv.index]]` + `[tool.uv.sources]`). torch 2.5 works with `numpy<2`.
  - open_clip pulls in `timm`, `huggingface_hub`, `ftfy` and `regex`. The CLIP weights (~600 MB) go to the user's HF cache on first use.
  - torch is imported lazily, inside `perception.labeler` only. Fast tests and oracle runs never load it.
  - `device: cuda` falls back to CPU when CUDA is unavailable. On this laptop the GTX 1650 was powered down at first (`torch.cuda.is_available()` False, device status "Unknown"); a few minutes later it was available without any change. The `clip_device` field in the `run` block records which device each run used.
  - *Alt: CPU-only torch now (~200 MB), CUDA at M6.* (fixed)
- **CLIP model = `ViT-B-32-quickgelu` with `openai` weights.** The OpenAI checkpoint was trained with QuickGELU. Plain `ViT-B-32` loads it with GELU, and open_clip warns about the mismatch. (fixed)
- **Clustering = connected components** (Erik; no scikit-learn): scipy `cKDTree.query_pairs` (Chebyshev metric) + `scipy.sparse.csgraph.connected_components`.
  - Two kept voxels are linked if their Chebyshev distance is ≤ `link_voxels` = **2** voxels. The plan said 26-connectivity (link 1).
  - Why 2: in the synthetic tests, one box and one wall-adjacent block each split into 2 clusters. Grazing-angle sampling of the top face, and side-face edges seen from fewer than 3 viewpoints, leave one-voxel gaps inside one surface. Link 2 bridges these gaps; it also merges objects closer than ~8 cm.
  - *Alt: DBSCAN (eps = 2 voxels, same effect, extra dependency); 26-connectivity (splits objects).* (provisional, tune on dev)
- **Structure filter (§3 step 2)** = points ≤ `struct_dist` = 10 cm from reference faces of **every structure class except floor and floor mat** (nyu40 2, 20). That covers walls, ceiling, doors, windows, pictures, curtains, … Floor and floor mat are excluded so objects standing on the floor survive; their bottom ~5 cm already falls under the 5 cm surface filter. (provisional, tune on dev)
- **Door swing-region filter** (Erik: surface filter + swing region; STATUS M4 open item).
  - Rule: a cluster is dropped if ≥ 50% of its voxels lie inside a reference door's swing region. The region is two vertical cylinders of radius door width + 10 cm, one around each possible hinge edge (the ends of the longer horizontal AABB side), over the door's height ± 10 cm.
  - Dev effect (3 pairs, B1, `min_voxels` 5): 12 clusters dropped (27.3 instead of 31.3 candidates per pair).
    - 3 of them correspond to GT instances: the **moved rack** on 0ad2d3a1 (GT moved 0.28 m, standing next to the door), plus a plant and clothes (both nonrigid, `dont_care`).
    - 9 are unexplained by any GT instance. The 295-voxel cluster on e2847ff7 next to door 12 is consistent with a swung leaf. Door swings are not annotated in the dev rescans, so this cannot be confirmed from GT.
    - Moved-to candidate recall 73.3% → 66.7% (n = 15); new recall unchanged.
  - Kept **on** (Erik's choice). The cost (objects standing within a door width of a hinge) is now measured; revisit with val's `door_dropped` counts.
  - *Alt: surface filter only.* (provisional)
- **Reference annotations in `perception/` go through `RefContext`**: new fields `nyu40` (3DSSG, PLY fallback; filled in `prepare_pair`) and `faces_of(oids)`.
  - The no-leak test bans `.face_labels` in `perception/` by attribute name; it cannot tell reference from rescan. The reference sub-mesh is therefore built in `obs/`.
  - Reference annotations are allowed method input (DESIGN §1); the ban stays as it is. (fixed)
- **CLIP vocabulary** = reference-graph labels ∪ the 100 most frequent thing/anchor labels of the train split (`lsg.train_meta`).
  - Source: 3RScan.json `type == "train"` scans (references and rescans), excluding the rooms of `dev_train.txt` and `eval_relchange_train.txt`; labels from 3DSSG `objects.json`.
  - A label's nyu40 comes from the G_ref node if there is one, else from the train majority.
  - The module lives outside `perception/`, which may not read 3DSSG.
  - It reads train *rescans'* object labels, i.e. training data of other rooms. The rescan under evaluation is never read.
  - *Alt: train references only.* (fixed)
- **CLIP labelling.** Per candidate, the image embeddings of the 5 largest-mask crops are normalised and averaged, then softmaxed against `"a photo of a {label}"`. The top-5 is stored, rounded to 1e-4 with ties broken by label, and cached in `cache/clip/` (key = crop pixels + model + vocabulary). Candidates without a usable crop get the label `object` and an empty `label_topk`. (provisional)
- **Candidate geometry and masks.**
  - Points: one mean residual point per kept voxel (≈ surface sampling, like GT's vertex means).
  - Masks: the pixels whose residual point fell into one of the cluster's voxels. They are recorded during `observe`; no re-projection pass.
  - Crops: mask bounding box +10% per side, depth → colour pixels by the per-axis resolution scale (as `OracleSource`), then `rot90(k=-1)`.
  - Pixels without a reference hit (`d_exp = 0`) are never nearer pixels; depth > 4 m is ignored. (provisional)
- **Synthetic acceptance check** (box added → exactly one candidate, centroid error < 5 cm) is measured against the **observable-surface centroid** (top + 4 sides, area-weighted). This is the analogue of eval's GT centroids, which are means of reconstructed mesh vertices, and object bottoms are never observed.
  - Result: 1 candidate, error **2.1 cm** (xy 0.0 cm).
  - Against the solid cube centre the error is 5.1 cm, all of it in z: the top face is seen, the bottom never, and the lowest 5 cm are cut by `surf_dist`. (fixed)
- **Perception eval wiring.**
  - `obs_truth`: §5 rule, one-to-one, greedy by IoU, of candidates against GT new ∪ the rescan instances of reference objects.
  - `visible`: a visibility-only `OracleSource` fed by the same online loop (eval-side `_TeeSource`); it never reaches the method.
  - `label_nyu = dssg.label_to_nyu40`.
  - **Candidate diagnostics** (method-independent; perception pairs only):
    - recall vs GT new and vs GT moved-to (the rescan instance of GT-moved objects), class-agnostic (centroid ≤ 0.30 m or IoU ≥ 0.25, one-to-one over all targets jointly) and with the §5 label rule;
    - candidates per pair;
    - share of candidates within the geometric gate of *any* GT rescan instance ("explained").
  - Per-pair candidate lists go to `outputs/<run>/candidates/`. (fixed)
- **M4 dev results** (`lsg m4`, reported, nothing tuned).
  - Candidate recall (B1, perception): new **66.7%** (2 / 3), labelled 66.7%; moved-to **66.7%** (10 / 15), labelled 26.7%.
  - 27.3 candidates per pair, of which 36.6% are explained by a GT instance.
  - Unexplained candidates are mostly plants (foliage moves and is poorly reconstructed), black TV screens (mesh holes) and clutter.
  - Sweep: `min_voxels` 3 / 5 / 10 → 33.3 / 27.3 / 22.0 candidates per pair, moved-to recall 73.3 / 66.7 / 66.7%. Spec value 5 kept.
  - B0 / B1 in perception mode are as expected for the B0 matching. Unchanged objects produce no candidates, so B0 deletes them (false-deletion rate 91.7%) and B1 keeps them `unknown` (unchanged kept 85.3%, false-deletion 16.7%). Labelled candidates of new objects are taken over as same-label `moved` matches (new R 0 / 3).
  - B2's confirmed-in-place skip and Hungarian matching (M5) are what make perception usable.
  - Determinism: the default-config sweep run and `eval_B1` gave identical perception `metrics`. A slow test runs one dev pair twice. (fixed)
- **M4 val report, B0 / B1 in both modes** (`outputs/m4_val_B1`, `m4_val_B0`; run once, nothing tuned). The oracle rows reproduce M2 (B0) and M3 (B1) exactly.
  - Perception B1:
    - moved P 22.4 / R 8.1%; removed P 17.7 / R 57.5%; new P 3.3 / R 12.9%;
    - unchanged strict 2.2%, kept 90.4%; false-deletion **9.4%** (n = 737; B0 perception 95.9%); false-move 3.7%; unknown rate 81.2%; association 2.0% strict.
  - Candidates: 7.8 per pair, 45.8% explained by a GT instance. Recall: new **27.5%** (labelled 17.5%, n = 171), moved-to **46.4%** (labelled 22.1%, n = 235).
  - Door filter: 115 of 969 clusters dropped, in 35 of 110 pairs.
  - The gap to oracle is mostly the B0 matching:
    - perception sees only changed regions, so unchanged objects are never matched (strict accuracy 2%);
    - same-label candidates pull far reference nodes into `moved` (localisation error median 1.64 m, n = 19);
    - new candidates are absorbed by same-label nodes.
  - The B1 removal rule works identically in both modes (same votes).
  - CLIP ran on CPU for this run (CUDA unavailable at the time; `run.per_pair[].clip_device`). The CLIP cache key does not include the device, so the first computed labels are reused; a cold GPU run could differ in the 4th decimal of the scores. (fixed)
- **No new dependency besides torch / torchvision / open_clip_torch** (scipy's KD-tree and csgraph were already available). (fixed)

## 2026-11-07 — M5 (Claude Code, confirmed by Erik where noted)
- **§7.2(c) = moved things whose GT parent was swapped** (Erik). Swapped = a parent annotated in both reference and rescan 3DSSG, and the two differ.
  - Secondary row **(c-broad)** = swapped + **gained** (no reference parent annotated). It is labelled "mostly reference annotation gaps".
  - "Lost" (no rescan parent) cannot be scored.
  - R0 is 0% on (c) by construction.
  - Scored n on val: (c) 3, (c-broad) 10. M1 counted 4 / 11 over all `rigid` things; one is nonrigid / removed and drops out of the relation universe.
  - *Alt: swapped only; swapped + gained as one row.* (fixed)
- **Relation universe for (a)/(b)** (Erik): the headline is **GT-defined**.
  - It covers reference things present in the rescan (GT unchanged / moved / `dont_care` small move; not nonrigid, not removed-but-present, not structure-only) with a GT rescan parent, plus GT-new things matched to a predicted new node (mode's §5 rule).
  - The node's current support edge is scored whatever its predicted state.
  - A second row, **associated only**, keeps the things the method associated (predicted unchanged / moved, plus matched new).
  - Parents are compared in rescan objectId space (`n<k>` → its token's GT instance).
  - *Alt: prediction-defined only. The scored set would then differ between methods and between modes.* (fixed)
- **Confirmed-in-place absorption, stage 0 of §6.9** (Erik; spec addition).
  - Problem: confirmed nodes skip the matching, but their own oracle instance is still observed. Without a rule it would become a false `new` and a free target for an unseen same-label node ("move"), and the confirmed node would count as wrong in the association metric.
  - Rule: an observation is absorbed by a confirmed node if both hold:
    - label compatible: oracle same label or nyu40; perception node label in `label_topk` with p ≥ p_min;
    - ≥ 50% of its points inside the node's AABB + 10 cm.
  - Many-to-one: each observation goes to its best node (point share, then distance, then id). The node's `obs_token` = its best absorbed observation.
  - In perception it also removes candidates from mesh holes / foliage inside a confirmed TV or plant.
  - Dev: oracle 29 absorbed for 36 confirmed; perception 10.
  - *Alt: confirmed nodes join the Hungarian with a fixed state; strict spec (no absorption, false new nodes).* (fixed; `absorb.*` provisional)
- **Method signature gains `ref_ctx`**: `method(G_ref, obs, cfg, rescan_id, obs_mode, votes, ref_ctx=None)`. B2 needs the reference node points (mesh vertices) for ICP; B0 / B1 ignore it. `classify` gains `ref_points` for the same reason. (fixed)
- **Methods decide states; `update_graph` decides relations.**
  - `update_graph(G_states, ref_ctx, obs, cfg, policy)`, with policy `update` (method) / `stale` (R0) / `geometric` (R1).
  - The harness runs all three on the same method output and scores all three.
  - `update.relations` (= `update`) picks the saved graph and the graph scored for change metrics (relations do not affect those).
  - R0 gives new nodes no parent (strict "keep reference relations").
  - Removed nodes keep their old edge in the file but are never a parent.
  - *Alt: relations inside each method (no shared R0 / R1 comparison).* (fixed)
- **Association details** (§6.9; values *(tune)*, kept at the starting values after the dev sweep below):
  - weights: w_g = w_p = 1, position scale 3 m, extents floored at 2 cm;
  - label term: oracle nyu40 penalty 0.3; perception p_min = 0.05;
  - gate τ_assoc per mode: oracle 2.0, perception 3.5 (a perception label cost alone can reach −log 0.05 = 3.0);
  - Hungarian = `scipy.optimize.linear_sum_assignment` on costs clipped at τ, then pairs ≥ τ dropped. This equals "leave unmatched at cost τ" and adds no dependency.
  - (provisional)
- **Dev sweep (B2, `outputs/m5_final/sweep_*`): τ_assoc (×0.6 / 0.8 / 1.0 / 1.2) × w_p (0.5 / 1 / 2).**
  - Oracle hardly reacts: moved F1 86.7% everywhere except τ 1.2 / w_p 2 (82.8%).
  - Perception trades moved F1 against localisation and new R:
    - τ 2.1 → moved F1 12.5%, localisation median 0.085 m, new R 66.7%;
    - τ 3.5 / w_p 1 → 60.9%, 0.446 m, 33.3%;
    - τ 4.2 / w_p 0.5 → 68.6%, 0.627 m, 0%.
  - Dev has 15 moves and 3 new objects, so there is no clear win: **starting values kept**. (fixed for M5)
- **ICP moved test** (§6.10).
  - Point-to-point; the **partial observation (source) is registered onto the reference node points (target)**; T_move = inverse of the result.
  - Schedule 0.20 / 0.10 / 0.05 m (the spec gave no values; 0.20 first because a partial view biases the centroid init by up to half an extent). Max 30 iterations, ≤ 4,000 points per cloud (seeded subsample).
  - **Two inits** (deviation from §6.10's single centroid-offset init): centroid offset and identity. Higher fitness wins; near ties (≤ 0.02) go to the lower RMSE. Below fitness 0.3 the centroid offset is used.
  - Why: from the biased partial-view centroid init, ICP on planar faces slid into a local minimum (synthetic half-seen box: 0.16 m fake displacement, fitness 1.0). For an unchanged object the identity init converges to zero motion.
  - Dev numbers were identical with one and two inits; val was run only with two.
  - Moved nodes get centroid / AABB of T_move · reference points, and `T_move` = the full 4×4.
  - *Alt: point-to-plane; centroid-offset init only (spec).* (provisional)
- **Oracle nyu40 fallback excludes the catch-all classes 39 (otherfurniture) and 40 (otherprop)** (Erik; found on val, so val was run twice).
  - The first val run (`outputs/m5_final/val_B2`) had oracle new R **24.0%** (B1 59.6%) and false-move 9.8% (B1 2.4%).
  - Diagnosis (`diag_new.py`, val oracle, nothing tuned): 96 GT-new objects were taken as `moved` by reference nodes, 58 of them with a different label of the same catch-all class. Examples: keyboard → blanket, plant → blanket, monitor → laptop. The taking nodes were mostly absent-unlisted (66) or GT-removed (18), i.e. unseen nodes.
  - Fix (`association.nyu_fallback_exclude: [39, 40]`): no nyu40 fallback for those classes; specific classes (chair = 5, table = 7, …) keep it. Principled, not tuned.
  - Dev is unchanged by it.
  - Second val run (`outputs/m5_final2`): new R **60.2%**, false-move 4.1%. Both runs are reported here.
  - *Alt: drop the fallback entirely; keep the spec and report.* (fixed)
- **`update` keeps a re-checked thing's old parent if it still supports it** (Erik; `update.keep_supported_parent`, `graph.relations.still_supports`).
  - The old parent is tested with the same contact / containment / floor rules at the new pose; only if that fails does `infer_support` pick a new parent.
  - Why: most moved things (chairs, pillows) keep their parent. Re-inferring it with the ~85%-accurate rules lost accuracy.
  - Before the change:
    - dev (b) oracle: R0 13/13, update 11/13;
    - first val run (b) oracle: R0 94.0 / update 72.0%.
  - After: dev 13/13; second val run 86.3%.
  - Checked on dev only. The first val run showed the same pattern, so val influenced the decision to look at it; logged here for that reason.
  - *Alt: keep the full re-inference and report update < R0.* (fixed)
- **`validate()` extended**: `T_move` must be 4×4; `label_topk` must be `[[label, score], …]`; `n<k>` ids ⇔ state `new`. The harness counts `graph_invalid` per pair (in `metrics`). No jsonschema dependency. (fixed)
- **Diagnostic `free_ev_unchanged_removed`**: GT-unchanged objects with ≥ k_free counted free votes that the method deletes (STATUS M5 item).
  - Dev: 4 / 34 such objects. B2 removes 4 / 4 in perception (no candidate to match them) and 1 / 4 in oracle. B1 perception removes the same 4.
  - They are the probable unlisted pillow move and three mixed-evidence "item"s (DECISIONS M3). They are not confirmed (free > 0), so the confirmed-in-place rule cannot protect them. Left for M6 confidence / the §7.4 audit. (fixed)
- **M5 acceptance** (`uv run lsg m5 --run-id m5_final2`, dev + val, sweep from `m5_final`):

  | dev | moved F1 B1 → B2 | assoc strict B0 → B2 |
  |---|---|---|
  | oracle | 33.3 → **86.7** | 90.9 → **90.9** |
  | perception | 45.5 → **60.9** | 6.8 → **6.8** |

  - All graphs validate (0 invalid over 3 methods × 2 modes × dev + val).
  - The dev `metrics` blocks of `m5_dev2` and `m5_final2` are identical (determinism).
  - Perception association stays at 6.8% because unchanged objects produce no candidates; confirmed nodes absorb few.
  - (fixed)
- **Val B2 (second run, reported; nothing tuned on val).**
  - Oracle:
    - moved P 45.7 / R 68.5 / F1 54.9% (B1 30.0); moved R by displacement 55.5 / 75.9 / 89.5% (B1 0 / 27.6 / 89.5);
    - localisation median 0.090 m, p90 0.448 (n = 161);
    - removed P 57.7 / R 41.1%; new P 80.5 / R 60.2%;
    - unchanged strict 76.9 / kept 86.6% (B1 80.9 / 92.9: the 10 cm ICP threshold turns some unchanged objects into small "moves");
    - false-deletion 9.5%, false-move 4.1%; association 90.5% strict (B0 / B1 91.4).
  - Perception:
    - moved P 52.9 / R 23.0 / F1 32.0% (B1 11.9); localisation median 0.194 m (B1 1.64);
    - unchanged strict **57.9%** (B1 2.2), kept 92.7%;
    - false-deletion 9.5%, false-move 1.5%; new P 2.4 / R 9.4%; association 2.8%; unknown rate 45.3% (B1 81.2).
  - **Relations, parent accuracy, GT universe, R0 / R1 / update:**

    | row | oracle | perception |
    |---|---|---|
    | (a) all | 92.0 / 79.5 / **92.4** (n = 994) | 94.6 / 80.2 / 90.7 (n = 966) |
    | (b) moved | 94.0 / 65.5 / 86.3 (n = 168) | 94.0 / 75.6 / 83.9 (n = 168) |
    | (c) swapped | 0 / 66.7 / 66.7 (n = 3) | 0 / 66.7 / 66.7 (n = 3) |
    | (c-broad) | 0 / 20 / 20 (n = 10) | 0 / 20 / 20 (n = 10) |

    - Type accuracy given the parent: R0 99.0 / R1 83.4 / update 94.8% (oracle).
    - R0 is strong because most things keep their parent; (c) is anecdotal (n = 3).
  - (fixed)
- **No new dependencies** (`scipy.optimize`, Open3D ICP already available). (fixed)

## 2026-11-09 — M6 (Claude Code, confirmed by Erik where noted)
- **Confidence rescaled** (Erik; spec §6.11 values replaced). With the spec values (L_hit = L_miss = 0.4, per-visit clip ±3, p_del 0.2) the delete rule could never fire. Train priors are 0.91–1.0 (global 0.968 over 718 train rescans), so the lowest reachable confidence was p = σ(logit(0.968) − 3) = 0.60 for a chair, and 0.31 even for the 0.9 fallback.
  - New: L_miss = L_free_like = **2.5**, so 2 counted free votes delete a median-prior thing (chair 0.968 → p 0.17); per-visit clip **±8**; p_del 0.2.
  - **L_hit = 2.5** (= L_miss, one seen vote offsets one free vote): dev sweep 0.5 / 1 / 2.5 → false-deletion rate perception 16.7 / 16.7 / **8.3%**, oracle 33.3 / 33.3 / **25.0%**; GT-unchanged objects with free evidence that get deleted: perception 50 / 25 / 25%, oracle 0 / 0 / 0%. Dev has 1 GT-removed object, so the cost in removed recall is only visible on val (reported, not tuned).
  - *Alt: spec values with confidence exported only (FULL removal = B2's rule, priors without effect); tune all four values on dev.* (fixed; L_hit provisional)
- **Class priors** (§6.11, `train_meta.class_priors`): per (reference object, train rescan) pair, present = in the rescan's 3DSSG objects, removed = in its `removed` list; absent-unlisted pairs are not counted (unknown fate, as in the eval universe). Beta smoothing κ = 20 toward the nyu40 rate, then the global rate. Same scene exclusions as the CLIP vocabulary (dev + relchange rooms, train split only). Reading train rescans' `removed` lists is training data of other rooms (DESIGN §2.3), never the evaluated rescan. `confidence.class_priors: false` → p = 0.9 for every node (ablation). (fixed)
- **Matched / new nodes** (§6.11 "seen at the new location"): `OnlineResult.obs_views` = decorrelated viewpoints (same `Decorrelator` rule) of the frames in which an observation's mask has ≥ 15 px. A matched node gets l₀ + clip(L_hit · views), and its old-place free votes are ignored. A new node gets clip(L_hit · views) from l = 0. Every anchor / thing / new node of FULL's graph carries `confidence`; `validate()` checks [0, 1]. (fixed)
- **`unknown` wording**: an unknown node carries its log-odds confidence. With no counted votes this is the prior (= "unchanged"); one counted free vote lowers it (allowed by hard rule 3). DESIGN §3 clarified. (fixed, in the approved plan)
- **Thin nodes: the appearance verdict replaces the depth vote** (Erik). A thin node's depth `seen` / `noinfo` vote is replaced; a depth `free` vote stays. Free-like votes block "confirmed in place". Free evidence = counted free + free-like ≥ k_free, where free-like alone needs ≥ k_free_like = 3. Why: a removed 2 cm book is `seen` by depth (the table behind lies within τ_near); adding free-like votes on top of 10 seen votes would net to zero, and the §9 synthetic check would fail. *Alt: add free-like votes on top of the depth votes (spec literal).* (fixed)
- **Thin threshold 0.15 m = τ_free** (Erik; spec 0.10 m). Things 0.10–0.15 m thick leave neither `present` nor `free` pixels when removed, so depth can never vote on them. Example: pillow r50, 0.14 m. Dev example: the 12 cm robot vacuum r68 on 19eda6f4 (below). *Alt: 0.10 m (spec); tune on dev.* (fixed)
- **Comparable pixels** = valid depth with −τ_near ≤ d_obs − d_exp ≤ τ_free (not occluded, not seen through), instead of the spec's "`present` only". Without it, objects of the 0.10–0.15 m band would never be compared, which would defeat the threshold decision. A patch is compared if ≥ 50% of its depth pixels are comparable; a frame only with ≥ 20 compared patches. (fixed, in the approved plan)
- **Frame-relative appearance distance** (dev finding). DINOv2 patch cosine distance between the photo and the rendered reference texture has a median of 0.23–0.28 on the references' *own* sequences, but 0.40–0.45 on the dev rescans of the same rooms (different lighting; the texture is baked from the reference visit). A fixed threshold calibrated on own sequences would flag 5–8% of rescan patches. Subtracting each frame's median distance over compared patches makes the distributions match (p98: own 0.38–0.42, rescans 0.31–0.41). (fixed)
- **Thin-node vote uses the node's mean relative distance** (dev finding; replaces the plan's "≥ 50% of the node's pixels mismatch"). Per patch, a 2% false-rate threshold (0.38) almost never makes half of a node mismatch: r50's patches lie at 0.1–0.36. The mean over the node's comparable pixels has much less variance, so its own threshold can be lower. Thin-object candidates (output ii) stay per pixel. (fixed)
- **Thresholds calibrated on the dev references' own sequences** (reference data, unchanged by construction; `lsg m6`, every 15th frame; 2% same-scan false rate, analogous to τ_free's 0.4%):
  - DINOv2: per patch τ_app = **0.38** (p98 0.384), per node τ_node = **0.30** (p98 0.297);
  - colour fallback: τ_app 0.85, τ_node 0.54.
  - On the dev rescans (method's refined `T_align`), DINOv2 node means above τ_node: GT-unchanged 0% (n = 50 node views), moved 0–6%, absent-unlisted 17–67%, GT-removed r50 0%.
  - The colour fallback is not usable on rescans: 9–50% of GT-unchanged node views exceed its threshold (lighting). DINOv2 stays the default. (fixed)
- **Dev appearance cases** (for the §7.4 audit):
  - **r50 pillow** (GT removed on e2847ff7; 23 counted seen votes at M3): a *white* pillow now lies where the reference had a blue-grey one. Depth sees "something there", and DINOv2, being largely semantic, sees "a pillow where a pillow was" (node means mostly < 0.1). Colour features would flag it but also flag unchanged objects. Not caught; a same-class replacement.
  - **r68 robot vacuum** (absent-unlisted on 19eda6f4): the reference texture shows it, the rescan photo shows bare floor, so it is gone and the GT does not list it. Depth gave only noinfo (12 cm, the band above). Appearance deletes it, which counts as a "false deletion".
  - **r69 shoes** (same rescan, absent-unlisted): displaced or partly gone; deleted.
- **Thin-object candidates (output ii)** implemented (Erik), switch `appearance.thin_candidates` (on in FULL).
  - Mismatching pixels whose depth agrees (|Δd| ≤ τ_near) on a support surface (first hit a non-thin anchor / thing or floor / floor mat, face |normal z| ≥ 0.8) go into a **separate** voxel grid. It bypasses the 5 cm surface filter (thin objects lie on the surface) but keeps the structure filter. The same ≥ 3-view / clustering / door rules apply; clusters touching a depth candidate (Chebyshev ≤ `link_voxels`) are dropped. `ObsInstance.kind = "appearance"`.
  - Synthetic book added: 1 candidate, < 5 cm from the top-surface centre; depth alone finds none.
  - Dev: 5.3 appearance candidates per pair, 10% explained by a GT instance; perception new P 1.8 → 1.4%, new R unchanged (33.3%). (provisional, report on val)
- **DINOv2 via timm** (`vit_small_patch14_dinov2.lvd142m`, `dynamic_img_size=True`, frozen, eval mode).
  - `timm` was already installed (open_clip dependency) and is now explicit in `pyproject.toml`; the lock changed by 2 lines. Weights ~88 MB from the HF hub. *Alt: `torch.hub` (fetches GitHub code at run time); `transformers` (new dependency).*
  - Speed on the GTX 1650: 53 ms per image pair at 448×252 (CPU 235 ms); texture render 55 ms per frame (CPU).
  - Similarities are cached per frame in `cache/appearance/` (key: rescan, frame, rounded aligned pose, reference, features, render size; values rounded to 1e-3; ~3 KB per frame). Features are not cached (~22 GB for val). The device is not part of the key (as for CLIP). (fixed)
- **Texture rendering**: own `RaycastingScene` on the OBJ geometry, barycentric `primitive_uvs` interpolation of the corner UVs, bilinear lookup with v flipped; checked by a unit test (four-colour quad).
  - OBJ ↔ PLY verified on **all 72** texture references: same vertex order and positions (max diff 0.0), identical faces in the same order. DESIGN §2.1 updated.
  - Render check overlays (`outputs/m6_final/m6_render_*.png`, 3 frames × 2 dev references, inspected) are visibly aligned. Black areas are mesh holes (mirror, toilet bowl).
  - Median DINOv2 patch similarity on own sequences: 0.766 (0ad2d39b), 0.720 (531cff08); on the dev rescans 0.56–0.61. (fixed)
- **Appearance runs only where needed.** Oracle mode: frames where a thin node has a depth seen / noinfo vote. Perception with thin candidates: every frame. B0–B2 are unchanged: the dev `metrics` of `lsg m5 --no-sweep --no-val` are identical to `m5_final2` on all shared keys (regression check). (fixed)
- **Milestone val subset** (Erik, 2026-11-09: a full val run after every milestone takes too long; the FULL val run took ~25 min). Per-milestone val reports use `splits/eval_val_small.txt`: 20 of the 47 val references, one rescan each, drawn with seed 0 (`lsg splits subset`, `io.splits.subset_pairs`; config `splits.milestone_val`). One pair per room, so rooms with many rescans do not dominate. GT support: 51 moved / 17 removed / 19 new / 229 unchanged / 148 absent-unlisted (about 1/5 of val). `lsg m5` / `lsg m6` default to it; `--full-val` runs all 110 pairs, reserved for the final tables (M7). Like val, never tuned on. The M6 val report below was still run on full val (started before this decision). *Alt: stratified by GT change counts (selection by GT); a prefix of the file (only a few rooms).* (fixed)
- **M6 acceptance** (`uv run lsg m6 --run-id m6_final`, dirty tree before the code commit):
  - texture render check: overlays of 3 frames × 2 dev references visibly aligned (inspected); median DINOv2 patch similarity 0.766 / 0.720; OBJ ↔ PLY identical on 72 / 72;
  - synthetic 2 cm book removed from a table (`tests/test_appearance.py`): FULL depth-only → `unchanged` (missed; free 0, seen ≥ 3), FULL + appearance → `removed` (≥ 3 counted free-like votes, p < 0.2), with colour features (fast) and with DINOv2 (slow);
  - 0 invalid graphs; dev `metrics` of `dev_FULL` and `sweep_Lhit2.5` identical; slow test runs FULL on one dev pair twice.
  - Dev, FULL vs B2 (both modes): unchanged kept 94.1 / 94.1% (B2 85.3 / 91.2); GT-unchanged objects with free evidence that get deleted 25 / 0% (B2 100 / 25%); false deletion perception 8.3% (B2 8.3), oracle 25.0% (B2 16.7: robot vacuum r68, a real unlisted removal). (fixed)
- **Val FULL** (full `eval_val`, reported once, nothing tuned; `outputs/m6_final/val_FULL`, 907 s, DINOv2 on CUDA, 4 workers, GPU memory 3.6 / 4 GB):

  | | removed P / R | false-del. | unch. kept | new P / R | moved F1 | unknown |
  |---|---|---|---|---|---|---|
  | B2 oracle (M5) | 57.7 / 41.1 | 9.5 | 86.6 | 80.5 / 60.2 | 54.9 | 28.1 |
  | FULL oracle | 54.3 / **26.0** | **8.1** | 86.6 | 81.0 / 59.6 | 54.6 | 29.6 |
  | B2 perception (M5) | 17.7 / 45.2 | 9.5 | 92.7 | 2.4 / 9.4 | 32.0 | 45.3 |
  | FULL perception | **24.0** / **31.5** | **7.7** | **95.4** | 2.0 / 8.8 | 31.2 | 50.2 |

  - The confidence veto turns 33 (oracle) / 120 (perception) of B2's deletions into `unknown`. It costs ~11 removed true positives per mode (GT-removed objects that also got counted seen votes; with L_hit = L_miss one seen cancels one free) and saves ~10–13 absent-unlisted deletions, some of which are real unlisted removals (dev r68). Dev (1 removed object) could not show this; L_hit was **not** re-tuned on val. Decision for Erik / M7 (STATUS).
  - Mean confidence: GT-unchanged 0.988 / 0.974, GT-removed 0.696 / 0.634, absent-unlisted 0.889 / 0.892 (oracle / perception).
  - Appearance matters little on val: only 2 (oracle) / 5 (perception) scored GT-removed objects are thin; 1 / 73 GT-removed and 3 / 1,401 GT-unchanged objects get ≥ 3 counted free-like votes; thin unchanged kept 90.9 / 90.6%. Thin-object candidates: 0.9 per pair, 16% explained by a GT instance; perception new P 2.4 → 2.0%.
  - Relations unchanged within noise: (a) update 92.2 / 92.4%, (b) 85.7 / 87.5%. (fixed)

## 2026-11-15 — before M7: confidence veto trade-off (Erik; drafted by Claude Code)
- **Seen votes need contrast** (Erik: option 3 of the M6 trade-off, STATUS M7). A pixel of P_i may count toward a `seen` vote only if it could also have counted toward a `free` vote.
  - **Rule:** d_behind = depth of the next reference surface along the pixel's ray that does not belong to node i (no surface → ∞). The pixel is *contrastive* if d_behind > d_exp + τ_free, i.e. if node i were gone, depth would show it.
  - **Frame vote** (`frame_votes`): `free` unchanged (≥ `vote_frac` of all valid pixels seen through). `seen` = ≥ `vote_frac` of the valid *contrastive* pixels are present (|d_obs − d_exp| ≤ τ_near), counted over at least `min_mask_px` valid contrastive pixels. Otherwise `noinfo`. Non-contrastive present pixels are no information.
  - **Why:** a present pixel whose background is within τ_free looks the same whether the object is there or gone (a removed pillow on a sofa shows the sofa just behind it). Counting it as `seen` is not evidence. Since M6, FULL's confidence counts every seen vote against a free vote (L_hit = L_miss = 2.5), so these votes veto real removals. This is hard rule 3 applied to the positive side: evidence that cannot discriminate is no information.
  - **Thin nodes** are unchanged. Their pixels are almost never contrastive (thickness < τ_free), which is why appearance replaces their depth seen / noinfo vote (M6).
  - **Honesty note:** the trade-off was found on the M6 val report (fixed, reported once). A per-object breakdown of FULL's vetoed val deletions (vs B2 at M5, diagnostic only) showed that the lost real removals were soft things on furniture: 4 pillows, 2 clothes, a cushion, 2 side tables, an organizer, a laptop. All have a smallest extent ≥ 0.16 m, so none were eligible for appearance, and their seen − free spread looks like that of the vetoed absent-unlisted objects. The rule is chosen on the sensor argument above, **fixed now before any further val run, and not chosen among variants on val numbers**. L_hit / L_miss / p_del and all thresholds stay as in M6; nothing is re-tuned on val.
  - **Scope:** config switch `visibility.seen_needs_contrast`, **on for FULL only**. B0–B2 keep the M5 vote rule, so their reported rows stay as they are. M7 ablation row: FULL with the switch off (= M6 FULL). Scope confirmed by Erik on 2026-11-15 (B0 ignores votes and B1 uses only free votes, so the alternative would only have changed B2, which has no confidence veto).
  - **Expected cost:** "confirmed in place" (≥ `k_seen` counted seen votes) uses the new seen votes, so some unchanged objects lying on support surfaces become `unknown` instead of `unchanged`. They are kept, so false deletion is not affected, but "unchanged kept" may drop. This will be reported, not tuned away.
  - **Implementation (first step of M7):** second-surface depth per frame from the reference mesh (method side, reference data only). Ray past the first hit, next intersection with a different label: Open3D `list_intersections`, or a second `cast_rays` from the hit point offset by ε, repeated while the label is i. About one extra cast per frame (≈ 5 ms at 224×172). Code in `geom/render.py` + `change/visibility.py`.
  - **Checks:**
    - unit tests: a pillow on a cushion removed → old rule `seen`, new rule `noinfo`; an object 1 m in front of a wall → `seen` under both;
    - dev: FULL with the switch off identical to `m6_final/dev_FULL`, B0–B2 dev unchanged, FULL on / off dev table;
    - val: once, in the M7 final tables (full `eval_val`, both modes), next to the switch-off row.
  - *Alt: 1 keep FULL as is and report the trade-off; 2 add train pairs with removed objects to dev and re-tune L_hit (download; would likely move along the same trade-off); appearance on all nodes (reverses M6 decision 2; DINOv2's semantic invariance limits it); a lower L_hit set a priori; scope FULL + B2.* (fixed)

## 2026-11-17 — M7 code (Claude Code; Erik decided scope, ablation set and "immediate")
- **Seen-needs-contrast scope fixed: FULL only** (Erik, 2026-11-15). B0 ignores votes and B1 uses only free votes, so only B2 could have changed; B2 has no confidence veto. (fixed)
- **Second-surface depth via `list_intersections`**, not repeated `cast_rays`. The first implementation restarted the ray 1 mm behind every hit with the node's label. A unit test (pillow box against a backrest, its back face coplanar with the backrest's front) showed that this skips a coincident surface of another label and finds the backrest's *far* side, so the removed pillow still voted `seen`. Now all intersections of the active rays are listed and the nearest other-label hit at t ≥ d_exp − 1 mm wins (no iteration cap needed). Cost ~7.6 ms per depth frame on 531cff08 (render ~6 ms); 95% (median) of node pixels are contrastive on that reference's own sequence. *Alt: iterative cast with an ε restart (fails on coincident faces); per-node depth peeling.* (fixed)
- **Dev check of the contrast rule** (reported, nothing tuned):
  - FULL with `seen_needs_contrast: false` vs `outputs/m6_final/dev_FULL`: 0 differences in `metrics`; B2 (default config) vs `m6_final/dev_B2`: 0; B0 vs `m5_final2/dev_B0`: 0.
  - FULL with contrast on dev: headline metrics identical to M6; one perception removed false positive moves from e2847ff7 to 19eda6f4. Dev has 1 GT-removed object, so the effect can only show on val. Contrast costs 1.0–1.4 s per pair. (fixed)
- **Immediate commitment = removal only** (Erik). `NodeVotes.commit_removed_frame` = first frame whose tentative state is `removed` (≥ k_free counted free / free-like evidence; with priors, i.e. FULL, also tentative p < p_del, the snapshot formula). With `states.commitment: immediate` such nodes are `removed` before absorption and matching, and FULL never vetoes them. Synthetic case: the committed chair's observation is then taken by an unseen same-label chair 3 m away (false move) or becomes a spurious `new` node, which is what deferral prevents. Implemented in `_b2_core` (works for B2 too); the M7 row is FULL only. *Alt: also freeze `unchanged` once confirmed mid-walk.* (fixed)
- **Children follow parent** (§6.12 optional rule): in FULL (a state decision, so in the method, not in `update_graph`). A thing in state `unknown` with no counted seen / free / free-like vote whose **reference** support parent is an **anchor** in state `moved` gets the anchor's `T_move` and state `moved`. Default off; M7 rows on val and relchange. *Alt: any moved parent (thing parents too); propagate up chains.* (fixed)
- **Thin objects in the metrics use the reference AABB** (`GTObject.min_extent_ref`, same `Instance` AABB as the reference graph) instead of the node's bbox in G_upd (STATUS M6 item: moved nodes changed their thin status, so oracle and perception counted different objects). Only thin keys change (dev B2 oracle `diag_thin_unch_n`). (fixed)
- **Ablation runner** (`eval/ablations.py`, `lsg ablate`, `lsg m7`): every row is one `run_eval` with dotted config overrides (unknown keys raise), its own `metrics.json` (hard rule 5). Rows:
  - methods B0, B1, B2, FULL, all rerun at one commit;
  - FULL minus contrast / appearance / thin candidates / class priors / decorrelation;
  - `B1-decorr` (Claude Code's addition: decorrelation acts directly on B1's removal rule; cheap);
  - FULL + immediate / + follow / − keep-supported-parent;
  - τ_free 0.05 / 0.10 / 0.20 / 0.30 (0.15 = FULL). `appearance.thin_max_extent` stays 0.15 m, so the sweep varies the vote threshold (and the appearance-comparable band), not which nodes are thin;
  - observed fraction 50 / 25 / 10% × chunk seeds 0 / 1 / 2 (the seed also shifts the oracle token permutation, as in `run_eval`); 100% = the FULL row;
  - relchange: B2, FULL, FULL+follow, FULL−keepparent (relation tables);
  - determinism: FULL on dev pair 0, twice.
  - (fixed)
- **Resume**: a row is reused only if its stored config hash and git commit equal the current ones and it was made from a clean tree. A crash during the multi-hour val run then costs only the unfinished row. (fixed)
- **Plots**: matplotlib (already a dependency), two-hue categorical palette (oracle blue `#2a78d6`, perception orange `#eb6834`) checked with the dataviz validator (CVD ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1). (fixed)
- **No new dependencies.** (fixed)

## 2026-11-18 — M7 results (val reported once, nothing tuned; Claude Code)
- **Run:** `uv run lsg m7 --run-id m7_final` (Erik started it; commit `565fc44`, clean tree; 111 min; DINOv2 / CLIP features cached for val, relchange computed on CUDA). Acceptance: dev regressions B0 / B1 / B2 / FULL-contrast 0 differences, 0 invalid graphs, determinism 0 differences, val on 110 pairs, `results.md` + 3 plots. All tables: `outputs/m7_final/results.md`.
- **Reproducibility:** the val rows B0 / B1 / B2 equal the M2–M5 val numbers, and FULL-contrast equals M6 FULL, on every reported value.
- **Seen votes need contrast did not resolve the M6 trade-off.** Removed recall: oracle 26.0 → 27.4% [17.8, 37.5], perception 31.5 → 32.9% [23.6, 42.9] (n = 73, about one object per mode). False deletion is unchanged (8.1 / 7.7%), and unchanged strict accuracy is −0.1 / −0.2 points. B2 has 41.1 / 45.2% removed recall. So the vetoed real removals mostly get seen votes from pixels *with* contrast behind them: the object (or something in its place) is really seen there. They are likely replacements, partial removals or GT issues, which is a question for the §7.4 audit (M8). Kept as fixed on 2026-11-15; not re-tuned. (fixed)
- **Ablations on val** (removed P / R, false deletion; oracle | perception):
  - FULL 55.6 / 27.4 / 8.1 | 24.2 / 32.9 / 7.7;
  - appearance off: 55.6 / 27.4 / 8.1 | 23.5 / 31.5 / 7.9. Expected: only 4 scored GT-removed objects are thin (now counted by the reference AABB in both modes);
  - thin candidates off: change metrics within one object; perception new P 2.0 → 2.3%;
  - class priors off: 56.4 / 30.1 / 8.4 | 23.7 / 37.0 / 8.0;
  - decorrelation off: 56.1 / 31.5 / 9.1 | 24.1 / 37.0 / 8.7. For B1 the false deletion goes 9.5 → 10.7 / 9.4 → 10.6%. Decorrelation trades a little recall for fewer false deletions, as §8 expected;
  - **immediate commitment**: 19.6 / 42.5 / 9.5 in both modes. Oracle moved F1 54.6 → 34.5% and new P 81.6 → 50.9%: committing early deletes moved objects and turns their observations into false new nodes. This is the clearest ablation result and supports deferred commitment. The two modes are identical because the commitment depends only on depth votes;
  - children follow parent: oracle unchanged kept 86.4 → 85.9%, moved F1 54.6 → 53.9%; relations unchanged. Rarely applies, slightly harmful; stays off;
  - keep-supported-parent off: change metrics identical (relations only).
- **τ_free sweep** (0.05 / 0.10 / 0.15 / 0.20 / 0.30 m):
  - removed R: oracle 35.6 / 37.0 / 27.4 / 24.7 / 11.0%, perception 38.4 / 38.4 / 32.9 / 28.8 / 12.3%;
  - false deletion: oracle 9.4 / 9.5 / 8.1 / 7.3 / 6.6%, perception 9.1 / 8.8 / 7.7 / 6.9 / 6.1%;
  - a monotone trade-off. The dev-chosen 0.15 m is not changed on val.
- **Observed fraction** (FULL, mean ± sd over 3 chunk seeds; 100 / 50 / 25 / 10%):
  - oracle: removed R 27.4 / 21.9 ± 2.7 / 10.0 ± 4.2 / 4.6 ± 0.8; false deletion 8.1 / 4.6 / 2.3 / 1.2; unknown 29.5 / 43.9 / 60.8 / 76.9;
  - perception: moved F1 31.7 / 20.5 / 10.7 / 2.7; new R 8.8 / 5.5 / 0.6 / 0.2.
  - Less observation leads to more `unknown`, not to more false deletions. Perception's moved / new collapse fastest, because candidates need ≥ 3 decorrelated views.
- **Relations** (parent accuracy, GT universe, R0 / R1 / update):
  - val FULL: (a) oracle 92.0 / 79.4 / 92.2 (n = 993), perception 94.7 / 80.4 / 92.4; (b) 94.0 / 64.9 / 85.7 and 94.0 / 75.0 / 87.5 (n = 168); (c) n = 3 as at M5;
  - relchange FULL (40 pairs, never tuned on): (a) oracle 84.4 / 70.3 / 83.4 (n = 441); (b) 48.4 / 39.8 / 48.4 (n = 93); (c) swapped 0 / 50.0 / 42.9 (n = 14); (c-broad) 0 / 32.8 / 24.6 (n = 61);
  - on the parent-change set, update beats R0 on (c), as constructed, and is level with R0 on (b). R1 alone is better on (c) but worse elsewhere.
  - (fixed)
- **No Hugging Face Hub requests once the weights are cached** (Erik asked why the runs contact the Hub):
  - timm (DINOv2) and open_clip (CLIP, whose openai weights now live in `timm/vit_base_patch32_clip_224.openai`) load through `huggingface_hub`, which sends a metadata request per model load even with a full cache: once per worker process per run.
  - `config.setup_hub_offline`, called by `lsg` before any model import: `models.hub_offline: auto` sets `HF_HUB_OFFLINE=1` when both repos have a cached snapshot. A fresh machine still downloads them once; `true` / `false` force it; an environment variable wins.
  - The library reads the flag at import time, so a per-call toggle is not possible.
  - *Alt: tell users to export the variable (forgotten easily); always offline (breaks a clean `uv sync` for the M8 README).* (fixed)

## 2026-11-18 — M8 (Claude Code; Erik decided the scope)
- **Deliverables** (Erik): write-up = `docs/REPORT.md` (~6–8 pages, audience: organisers / reviewers) + a ~10–12 slide PPTX (`docs/slides.pptx`) built from the same figures and tables. Figures used by them are copied to `docs/figures/` (outputs/ is git-ignored). (fixed)
- **Door open / closed diagnostic: skipped** (Erik). The 12 door swings in val stay out of scope (DESIGN §10) and are stated as a limitation. Closes the open STATUS item. (fixed)
- **AI_USAGE "checked by hand"** (Erik): Erik sends one line per milestone at the end of M8; Claude pastes them in. (fixed)
- **`rerun-sdk` 0.23.1** (planned 2026-11-05) for the snapshot timeline (`viz/rerun_timeline.py`). Newer rerun-sdk releases need NumPy 2, and `pyarrow` ≥ 22 (pulled in by rerun-sdk) refuses NumPy 1.26, so `[tool.uv] constraint-dependencies = ["pyarrow<22"]` (resolves to 21.0.0). NumPy stays < 2 (open3d / torch 2.5 wheels). *Alt: matplotlib animation (no 3D navigation); Open3D visualiser (needs OpenGL, not available headless); upgrade to NumPy 2 (risks every earlier result).* (fixed)
- **`seq_info.json` next to the snapshots** (`run_online`, only when snapshots are written): refined `T_align` + processed frame ids, so `lsg viz rerun` needs only the run directory and the data. Output files only; votes, states and `metrics` are unchanged (dev pair graphs identical to `m7_final/dev/FULL`, nodes and edges, both modes). (fixed)
- **`lsg run --snapshot-every N`** overrides `online.snapshot_every` for one run (denser timelines on short sequences: the dev pair 0ad2d39b → 0ad2d3a1 has 24 processed frames, i.e. 2 snapshots at the default 20). It changes the config hash, not the `metrics` (checked: identical). (fixed)
- **Top-down colours** (dataviz palette, validator run `--pairs all`, light surface): moved blue `#2a78d6`, removed orange `#eb6834` (also dashed + hatched), new aqua `#1baf7a` (always labelled: 2.7:1 contrast), unchanged grey fill, unknown grey dotted outline. All-pairs CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 24. The GT panel is computed in `eval/figures.py` and passed to `viz` as a dict (hard rule 1); `viz` is in the no-leak import scan. (fixed)
- **§7.4 audit design** (`eval/audit.py`, `lsg audit sample | summarize`):
  - **Source:** a val FULL rerun with per-object details (`run_eval(details=True)`: a run argument, not config, so the config hash is unchanged). The oracle's token permutation depends on the observed instances, so the M7 graphs alone cannot rebuild oracle new-node matches. The rerun must reproduce `m7_final/val/FULL` (`summary` + every `per_pair` count; checked automatically). A 3-pair dry run matched.
  - **Pools:** moved = predicted moved with GT unchanged / removed / absent-unlisted; removed = predicted removed with GT unchanged / moved / absent-unlisted; new = unmatched predicted new nodes. Absent-unlisted predictions are included because they are the likely GT gaps. Each row keeps its GT status, so the summary splits scored FPs from absent-unlisted ones. Simple random sample per mode × category (seed 0, n = 10), not stratified by GT status.
  - **Veto set** (STATUS M7): GT-removed objects that B2 deletes and FULL keeps `unknown` (`m7_final/val/B2` graphs). All of them, no sampling.
  - **Panels:** reference photo at the node's best reference view; rescan photo at the best view of the reference place (most first-hit pixels, weighted by how much of the box is in the image); rescan at the predicted / GT new place (box view, not occluded by observed depth); top-down; text with GT status, votes and confidence. Verdicts `gt_missing | method_error | unclear`, labelled by Erik; summary = `gt_missing` share with Wilson 95% CIs.
  - *Alt: stratified pools (5 scored + 5 absent-unlisted); Claude labels first (Erik chose to label himself).* (fixed)
- **Report figure pairs are chosen by rule** (no cherry-picking): the example is the `eval_val_small` pair with the most scored GT changes (8eabc45f → 5341b7b3, 15). The failure case is the pair with the most perception false positives (4a9a43d2 → 4a9a43d4, 39). Both come from `m7_final/val/FULL` per-pair counts; the README demo pair is also shown. (fixed)
- **Slides** built with `pptxgenjs` from `docs/slides/build_deck.js` (a build-time tool, not a project dependency; `npm install pptxgenjs` in a scratch directory). Numbers are copied from `m7_final/results.md`. Checked with the pptx skill's validator and by exporting through PowerPoint to PDF. The script strips the ZIP's folder entries (cloud co-editing). (fixed)

## 2026-11-19 — Change proposal: relax perception candidate detection (Erik decided; not implemented)
- **Decision** (Erik): candidates from large clusters need only **1** decorrelated viewpoint; small clusters (a couple of pixels) **2**; the nearer-pixel threshold `tau_new` is relaxed for big objects. Why: 72% of the 171 scored GT-new objects never produce a candidate (new R 8.8% perception), the method already skips frames (`stride: 3`), and a large cluster is rarely noise.
- Nothing is implemented; M7 / M8 numbers are unchanged. Design, risks, evaluation plan (dev_train tuning only, one val ablation row, success criterion written beforehand) and four open questions are in `docs/CHANGE_candidate_detection.md`. *Alt: leave candidate detection as is and fix only association / labelling.* (proposal, open questions in the document)

## 2026-11-19 — CHANGE 1 implemented (Claude Code; answers by Erik)
- **Erik's answers to the four open questions:** small clusters **2** views (was 3); relax **`tau_new`** for large clusters (not `min_voxels`); relaxed rules **ON by default**; "large" = about 20 voxels, voxel count only (`large_min_extent: 0`, extent rule present but off). Thin-object (appearance) candidates keep `min_views: 3`.
- **What was built** (`perception.relaxed` in `configs/default.yaml`; `perception/source.py`, `cluster.py`, `residual.py`): strict tier = existing grid with `min_views_small` 2 at `tau_new` 0.10; lenient tier = a second grid at `tau_new_large` 0.05 with `min_views_large` 1; lenient clusters with ≥ `large_min_voxels` voxels replace the strict clusters inside them (`merge: replace`; `add` = only where no strict cluster is). One residual pass per frame; the strict tier gets exactly the pixels nearer than `tau_new`, so `relaxed.enabled: false` reproduces the M7 candidates and `metrics` bit for bit (checked: `lsg ablate --names FULL-relaxed --split dev_train` = `m7_final/dev/FULL`, both modes).
- **Tuned on dev_train only** (3 pairs; geometry check without CLIP, `candidate_counts`; strict: 82 candidates, 30 explained by GT = 37%, new 2/3, moved-to 10/15):
  - the first version (`link_voxels` 2, 20 voxels, replace) built room-sized blobs from 1-view voxels (largest clusters 3,669 and 6,281 voxels, replacing 40 and 46 strict clusters): 77 candidates, 26% explained, moved-to 6/15. **This is why `link_voxels_large` = 1 (26-connectivity) exists.**
  - strict tier at 2 views alone: 96 candidates, 42% explained, moved-to 12/15.
  - lenient tier at 26-connectivity, replace: 20 voxels 125 candidates (29%), 30 voxels 108 (32%), 40 voxels 97 (36%); explained by GT 36 / 35 / 35, new 3/3, moved-to 10/15. **`large_min_voxels` = 40** (Erik's "about 20" was a starting guess; 20 adds candidates but no explained ones).
  - lenient tier at 2 views: 91-99 candidates, 38-42% explained, moved-to 13/15, new 2/3. 1 view was kept as decided.
  - With 3 GT-new objects and 15 moved-to objects these are single-object differences. Treat dev as a smoke test; **val decides**.
- **Dev end-to-end FULL, perception, strict -> relaxed** (n tiny): new P 1.4 -> 1.3%, new R 33 -> 33% (candidate recall new 2/3 -> 3/3), moved F1 60.9 -> 69.2%, false deletion 8.3 -> 25.0% (n = 12: 1 -> 3 objects), false move 16.7 -> 0%, unchanged kept 94.1 -> 91.2%. Demo pair: 4 `new` nodes instead of 1 (the pair has no GT-new object). Oracle unchanged.
- **Consequences of ON by default:** every perception number (B0 / B1 / B2 / FULL share the candidates) changes; `m7_final`, REPORT, slides and the README headline table describe the strict method (= ablation row `FULL-relaxed`). `lsg audit sample` runs the strict rules unless `--relaxed` is given, so it still reproduces `m7_final/val/FULL`. Config hash changes (new keys).
- **Open:** val effect (Erik runs `uv run lsg ablate --names FULL,FULL-relaxed --split eval_val`), success criterion in the change document §5 (new R up; false deletion and unchanged kept within the FULL CIs), then a `lsg m7` rerun to refresh the report. *Alt: default off until the val run (rejected by Erik).* (implemented; values provisional until val)

## 2026-11-20 — Change proposal: class-agnostic prior and one confidence for new objects (Erik decided the direction; not implemented)
- **Decision** (Erik): (A) no class prior learned from the dataset: it is dataset-specific, and the M7 ablation `FULL-priors` (flat p = 0.9) shows it buys almost nothing (oracle removed R 30.1 vs 27.4, false deletion 8.4 vs 8.1; perception 37.0 vs 32.9, 8.0 vs 7.7; all within the CIs). (B) New objects get the same confidence treatment as reference nodes: a candidate is added after one (decorrelated) observation, later views confirm it or lower it by seeing through it, decorrelated views count more.
- Nothing is implemented. Design (per-voxel log-odds for the candidate voxels, `min_views` rules as a special case, see-through votes with a 3x3 minimum-depth window, correlated-view cap, candidate confidence = median voxel log-odds, emission threshold `p_new_min`, precision / recall curve from saved candidates), risks, evaluation plan with a success criterion written beforehand, and six open questions are in `docs/CHANGE_confidence_unified.md`. A neutral prior (0.5) would remove the protection of mixed evidence (1 seen + 2 free deletes a node): the document proposes 0.9 and tests 0.5 / 0.8 / 0.9 on dev. *Alt: keep class priors; keep candidates view-count-only.* (proposal, open questions in the document)
- **Answers (Erik, 2026-11-20)** to the CHANGE 2 questions: no prior value is needed; correlated views keep the small weight (`L_corr` 0.25, cap 1.0). Reason for the first: prior and `p_del` are one parameter (the deletion margin `M = l0 - logit(p_del)`), so the design uses `l0 = 0` and sets `M` = 3.6 through `p_del` = 0.027 (2 net counted votes delete, 1 cannot, mixed evidence protects), the same behaviour as the M7 row `FULL-priors` (flat 0.9, `p_del` 0.2, `M` 3.59). Four questions remain open (small clusters, low-confidence candidates, default, snapshot logging).
- **More answers (Erik, 2026-11-20)**: one decorrelated observation is enough for every cluster, small ones too (`l_keep_small` = `l_keep_large` = 2.5, 5.0 reproduces CHANGE 1; risk: single-view small clusters are close to depth noise, the dev check reports candidates and explained share at 2.5 vs 5.0); CHANGE 2 is **ON by default** (ablation rows `FULL-candconf`, `FULL-candconf-free`, `FULL+classprior`, `FULL-relaxed`, and `FULL-m7` = all off = `m7_final`); candidate confidence is logged into the snapshots and drawn in `lsg viz rerun`. Still open: whether candidates below `p_new_min` are dropped (proposal) or emitted.
- **Last answer (Erik, 2026-11-20)**: candidates whose confidence is below `p_new_min` are **dropped** from the graph; all candidates are saved with their confidence in `candidates/<pair>.json` for the precision / recall curve; `p_new_min = 0` emits everything. All CHANGE 2 questions are answered; the design is ready to implement. (fixed)

## 2026-11-20 — CHANGE 2 implemented (no class prior; candidate confidence as per-voxel log-odds), default ON
- **Decision** (Erik asked for the implementation of `docs/CHANGE_confidence_unified.md`): Part A = `confidence.class_priors: false`, `prior_p: 0.5` (renamed from `fallback_p`; l0 = 0), `p_del: 0.027`. Part B = `confidence.candidates` (`L_corr` 0.25, `corr_cap` 1.0, `free_votes`, `free_margin` 0.15, `free_window` 1, `l_keep_small` = `l_keep_large` = 2.5, `p_new_min` 0.92). Ablation rows: `FULL-candconf` (Part B off = CHANGE 1), `FULL-candconf-free` (no see-through votes), `FULL+classprior` (learned prior, p_del 0.2), `FULL-priors` (now explicit 0.9 / 0.2 = the M7 row), `FULL-m7` (`ablations.M7_OVERRIDES`: CHANGE 1 and 2 off). The §7.4 audit forces `M7_OVERRIDES` (it must reproduce `m7_final/val/FULL`).
- **Choices where the document was silent:** (1) voxel lv = clip(hit_sum − free_sum, ±8), the same total clip as reference nodes (order-independent); (2) see-through: invalid / > `max_depth` pixels count as 0 before the 3x3 minimum filter, so any invalid pixel in the window, the image border, out of view or behind the camera = no info; voxels hit in the same frame get no free vote; free votes have their own decorrelation state and their own correlated cap (−0.25, at most 1.0 since the last decorrelated free vote); (3) `relaxed.enabled: false` with Part B on: strict voxels need L_hit · `min_views` = 7.5 (M7's 3 views), so `FULL-relaxed` keeps meaning "M7 candidate geometry"; (4) the emission filter (`change.baselines.emits_new`) applies to the new nodes of every method, because the candidates are shared; oracle instances have no confidence and are never filtered; (5) **at the defaults the emission threshold is inactive**: kept voxels have lv ≥ 2.5, so the median gives p ≥ 0.924 ≥ 0.92 (dev: `below_p_new_min` = 0 on all pairs); the deletion Erik asked for happens at voxel level (seen-through voxels fall below 2.5 and leave the cluster); (6) instead of 3 val rows at different `p_new_min`, `lsg newcurve --run <dir>` re-scores new P / R from the saved graphs at `m7.p_new_sweep` (0.95 / 0.99 / 0.999) plus the default (exact for thresholds ≥ the run's, association is not rerun). Snapshots get `candidates_frame_XXXXXX.json` (live candidates: box, confidence, size, tier; no labels mid-walk), drawn by `lsg viz rerun` as `world/candidates`.
- **Dev check** (`lsg ablate --names FULL-m7,FULL-candconf,FULL --split dev_train --run-id chg2_dev`, 3 pairs, both modes): `FULL-m7` reproduces `m7_final/dev/FULL` exactly (summary and per-pair); `FULL-candconf` reproduces the CHANGE 1 dev numbers, and its oracle states equal M7's (Part A changes no decision on dev, only confidence values). Perception, CHANGE 1 → CHANGE 2: candidates per pair 36.0 → 57.3, share explained by GT 33 → 23%, candidate recall new 3/3 → 2/3 and moved-to 10/15 → 12/15, new P / R 1.3 / 33 → 0.7 / 33%, moved F1 69.2 → 75.9 (R 60 → 73), false deletion 25.0 → 8.3%, false move 0 → 25% (3 of 12 absent-unlisted), unchanged kept 91.2 → 88.2%. Oracle unchanged. See-through votes pull 17–19% of the voxels below 2.5 (`seen_through_voxels`); the strict kept voxels almost double (one observation is enough: 5982 → 10711 on 19eda6f4). The lost GT-new candidate (e2847ff7, table 54) was not seen through: see-through votes trimmed the cluster (531 → 388 voxels), CLIP then said "bottle" and the §5 gate fails; a sofa + pillow blob split into sofa and pillow. Observe time per pair +3 to +10% (3.1 → 3.2 s, 3.9 → 4.3 s). Dev has 3 GT-new objects: no value is tuned here (hard rule 4). Main risk as predicted: single-view small clusters raise the candidate count and lower new P. *Alt: `l_keep_small` 5.0 (two views for small clusters, the CHANGE 1 value).* (fixed, val pending)
- **Val result (2026-11-20, Erik asked for the FULL validation; `lsg ablate --names FULL --split eval_val --run-id chg2_val`, 110 pairs, both modes, 274 s; the new default = CHANGE 1 + CHANGE 2)** against M7 FULL (`m7_final/val/FULL` = `FULL-m7`), 95% bootstrap CIs. Perception: new P 2.0 [1.0, 3.0] → 1.0 [0.6, 1.4], new R 8.8 [4.6, 13.2] → 14.6 [9.1, 20.4] (TP 15 → 25 of 171, new nodes 757 → 2547), moved P / R / F1 49.1 / 23.4 / 31.7 → 36.1 / 38.7 / 37.4 [32.1, 42.7] (TP 55 → 91), removed P / R 24.2 / 32.9 → 25.6 / 28.8, false deletion 7.7 [4.8, 11.4] → 6.9 [4.2, 10.0], **false move 1.6 [0.7, 2.8] → 6.4 [4.5, 8.9]**, **unchanged kept 95.3 [93.9, 96.7] → 88.8 [86.8, 90.7]**, candidates per pair 8.7 → 27.4 (explained by GT 43 → 31%), candidate recall new 27.5 → 40.9%, moved-to 47.2 → 63.4%. Oracle: new / moved unchanged; removed P / R / false deletion 55.6 / 27.4 / 8.1 → 56.4 / 30.1 / 8.4 = exactly the M7 `FULL-priors` row (Part A = the flat-prior decisions, as designed). Against the criterion of the change document §6.4: new R up (met); false deletion within the CI (met); removed recall within the CI (met); **unchanged kept outside the CI (not met)**, the cost of the many more candidates (more wrong moved matches of unchanged nodes, false move ×4); new P at equal recall needs `lsg newcurve --run outputs/chg2_val/FULL`. This run does not separate CHANGE 1 from CHANGE 2: `FULL-candconf` (= CHANGE 1 alone) and `FULL-m7` on val would. Defaults not changed; Erik decides. (measured)

## 2026-11-20 — CHANGE 1 + 2 = final method; CHANGE 3 proposed (Erik decided; drafted by Claude Code after a logic review)
- **Final method = CHANGE 1 + 2** (Erik), although its val run misses one item of the CHANGE 2 criterion (unchanged kept 88.8% vs FULL's CI [93.9, 96.7]; false move 1.6 → 6.4%). The review traces this to association: a move needs no evidence at the old place, so more candidates means more wrong moves of unconfirmed nodes. CHANGE 3 Part A targets it. REPORT / README / slides / audit move to the final method after the final val run. *Alt: revert to `FULL-m7` (the criterion as written).* (fixed)
- **Incremental graph during the walk** (Erik): association, states, relations and confidence at every snapshot, not only at the end; the brief leaves the confidence policy to us (Erik). (direction fixed; design in `docs/CHANGE_online_graph.md`)
- **A larger tuning set** (Erik: plan it). Train metadata: 718 eligible pairs in 360 rooms with 304 removed / 1,006 new; 30 random rooms give only 9 removed objects, so the proposal is 20 change-heavy + 10 random rooms + the 3 `dev_train` pairs (85 removed, 212 new; ≈ 2–3 GB). It needs a CLAUDE.md hard-rule-4 change and a vocabulary decision.
- Proposal in `docs/CHANGE_online_graph.md`: Part 0 tuning set, A move needs consistent old-place evidence, B incremental graph (`ObsSource.peek`, stable candidate tracks, snapshot method runs, anytime curve), C confidence policy (vote weights from sensor statistics on the tuning set, candidate confidence over the track's full history, `p_new_min` tuned), D class-agnostic new P / R in the tables. Five open questions (§8). Nothing implemented. (proposal)
- **Answers to CHANGE 3 §8** (Erik, 2026-11-20): (1) tuning set = 20 change-heavy + 10 random train rooms + the 3 `dev_train` pairs; (2) the tuning rooms are excluded from the CLIP vocabulary (6 of the top-100 labels change); (3) reference nodes start at a flat p = 0.9 with p_del 0.2 (same decisions as l0 = 0 / 0.027; an untouched node reports 0.9, not 0.5); (4) incremental alignment; (5) order 0 → A → C → B → D.
- **Online alignment = one cumulative estimate** re-estimated at each snapshot from the frames so far (warm start, raw `T_align` until enough frames, same rejection rules + a degeneracy check), not ICP on the current frame alone. Why: the error is one rigid offset shared by all frames (own-mesh fit 1.1–1.3 cm, one global correction reaches 1.9–2.2 cm, so per-frame estimation gains ≤ ≈ 1 cm); a single frame gives a noisier estimate, can be pulled toward a moved object that fills the frame (erasing the evidence of the change), and is often degenerate (one wall). *Alt: per-frame ICP (Erik's question); per-frame correction regularised toward the cumulative estimate and ICP against reference static surfaces only — both tested on `dev_tune`.* (fixed; variants open)

## 2026-11-20 — CHANGE 3 Part 0 implemented: tuning set `dev_tune`, flat start value 0.9 (Claude Code; Erik's answers)
- **Reference start value = flat 0.9, p_del 0.2** (`confidence.prior_p`, `p_del`; was 0.5 / 0.027). Unit test: deletes exactly where CHANGE 2's values did, over every seen / free / free-like combination (margins 3.58 vs 3.6, votes in steps of 2.5). Only the exported `confidence` changes: an untouched reference node reports 0.9, matched nodes start from logit(0.9). `FULL-priors` now equals the default on this point. (fixed)
- **`lsg splits devtune`** (`eval/devtune.py`, in `eval/` because it reads train rescans' `removed` / `rigid` lists, hard rule 1; config `splits.dev_tune`). Metadata only: per eligible train pair (718 pairs in 360 rooms, dev and relchange rooms excluded, both scans in 3DSSG) the things / anchors removed (listed and absent from the rescan objects), rigid (unique ids, incl. < 10 cm) and new. Room score 2 · removed + new + 0.5 · rigid, best rescan per room; the 20 highest rooms, then 10 random rooms (`numpy.random.default_rng(0)`, as `subset_pairs`), then the 3 `dev_train` pairs. Result `splits/dev_tune.txt` (33 pairs): heavy 81 removed / 78 rigid / 204 new, random 1 / 24 / 8, total of the 30 new pairs **82 / 102 / 212** (val, scored: 73 removed / 235 moved / 171 new). The 30 new rooms are disjoint from val, relchange and dev (checked). *Alt: 30 change-heavy (104 removed, only busy rooms); 30 random (9 removed).* (fixed)
- **Downloads** (hard rule 6): `splits/download_tier3.txt` = the 60 scans of `dev_tune` not in tier 1 / 2; `splits.texture_sources` += `dev_tune`, so `download_textures.txt` lists 102 references (30 new). (fixed; download pending, Erik)
- **CLIP vocabulary excludes `dev_tune`'s rooms** (Erik, answer 2; `train_meta.exclude_splits`). Top-100 labels vs the M7 vocabulary: in coffee machine, flowers, toy; out flower, rug, stuffed animal. Every default perception number shifts slightly (new vocabulary, new CLIP cache keys). `ablations.M7_OVERRIDES` pins the M7 exclusion list, so `FULL-m7` and the audit still reproduce `m7_final`. (fixed)
- **CLAUDE.md hard rule 4: tune on `splits/dev_tune.txt`**; `dev_train` stays the 3-pair smoke / regression set (Erik's answer 1). (fixed)
- No new dependency. (fixed)
- **Download + baseline (2026-11-20, Erik asked Claude Code to run the steps).** Tier 3: 60 scans (3.3 GB incl. textures) + 30 texture sets, all `sequence.zip` pass `testzip`, all 66 `dev_tune` scans load; two downloads failed on dropped connections (one file already present from the first attempt, `bcb0fe1b…/semseg.v2.json` re-downloaded). Regression: `lsg ablate --names FULL-m7 --split dev_train --run-id chg3_reg_m7` = `m7_final/dev/FULL`, **0 differences in 2,898 summary + per-pair values** (43 s). Baseline `lsg ablate --names FULL --split dev_tune --run-id chg3_dev_baseline` (git 7a98d01, clean; 153 s on CUDA after a CPU start that was stopped when CUDA came back; ≈ 10 pairs' features were computed on CPU, cached, device not in the key). GT support: moved 79, removed 83, new 215, unchanged 353, absent-unlisted 196 (791 reference nodes).

  | FULL on dev_tune (CHANGE 1 + 2 + flat 0.9) | oracle | perception |
  |---|---|---|
  | removed P / R | 83.3 / 24.1 | 50.0 / 25.3 |
  | false deletion | 2.6 | 4.6 |
  | **false move** | **18.9** (37 / 196) | **13.3** (26 / 196) |
  | unchanged strict / kept | 71.1 / 78.8 | 53.3 / 82.7 |
  | moved P / R / F1 | 39.5 / 78.5 / 52.5 (95 FP) | 33.6 / 48.1 / 39.6 (75 FP) |
  | new P / R | 92.7 / 53.0 | 2.4 / 10.2 (935 new nodes) |
  | unknown rate | 24.9 | 39.7 |
  | candidates per pair / explained / recall new, moved-to | – | 34.8 / 30.1% / 34.0%, 74.7% |

  Reading: the largest error is wrong moves (oracle 95 moved FP against 62 TP; 19% of absent-unlisted nodes moved), which is exactly CHANGE 3 Part A's target; removed recall is low (24–25%) at a high oracle precision, Part C's target (seen-vote weight). These are the comparison numbers for Parts A and C (`docs/CHANGE_online_graph.md` §7.3). (measured)
