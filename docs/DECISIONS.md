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
