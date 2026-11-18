# DESIGN — Living Scene Graphs, Subproject D: change detection and graph maintenance

Status: **v2 (2026-11-05)**.
- v1 was written before M0; the M0 results are in §2.1 and §5.
- v2 is a spec change: the headline is now **perception mode** (the method sees only reference data plus rescan RGB-D, poses and `T_align`). **Oracle mode** is kept as an upper bound behind the same interface (DECISIONS 2026-11-05, "Spec change v2").

Numbers marked *(tune)* are starting values. They are fixed on `splits/dev_train.txt`, and every change is logged in DECISIONS.md.

---

## 1. Problem and framing

**Input of the method (both modes):**
- the scene graph G_ref of a reference scan (from GT `semseg.v2.json` and 3DSSG of the **reference**);
- the reference mesh, with its texture;
- the RGB-D frames and poses of a rescan of the same room, **in time order**;
- 3RScan's `T_align` (rescan → reference).

Nothing else about the rescan is input.

**Produce:**
- a per-node change state: `unchanged | moved | removed | unknown`;
- a list of `new` nodes (with `label_topk`);
- the updated graph G_upd: moved nodes with recomputed support parent and relation, deleted nodes, added nodes, `confidence`, and `last_seen`;
- graph snapshots during the walk (tentative states, for visualisation).

**Contribution.** Object-level *semantic graph maintenance* from raw RGB-D:
- node identity across visits;
- relation updates;
- a confidence/deletion policy that distinguishes "gone" from "not seen", with viewpoint-decorrelated evidence and deferred commitment.

This project does **not** compete with class-agnostic geometric change detection (Objects Can Move, Has Anything Changed?). Perception mode finds candidate objects from depth and appearance residuals (§3, §6.6), but this candidate step is a means only. It is evaluated through the object-level graph metrics (§7), not as point-level change detection.

**Two observation modes behind one interface (`ObsSource`, §6.4):**
- **`perception`** (headline; answers the brief):
  - rescan objects are discovered from depth residuals against the rendered reference, plus appearance residuals for thin objects;
  - they are then labelled zero-shot (CLIP);
  - it never reads rescan annotations.
- **`oracle`** (upper bound):
  - rescan objects come from the rescan's GT instance segmentation: masks, class labels and points visible in the used frames, with IDs shuffled;
  - this isolates the change logic and measures what perception costs.

**Every results table has two rows per method: `oracle` and `perception`.**

## 2. Data

### 2.1 Facts verified on local data (2026-11-05)

| Item | Value | Status |
|---|---|---|
| Color / depth resolution | 960×540 / 224×172, same FOV; separate intrinsics in `_info.txt` | verified |
| Depth encoding | PGM P5, **big-endian** uint16 mm, 0 = invalid, `m_depthShift = 1000` | verified |
| Pose | camera→world in that scan's mesh frame. Back-projected depth lies 0.9–1.5 cm (median) from the mesh; the inverse convention gives 0.5–2.6 m | verified |
| Up axis | +Z (floor normal ≈ ±z on 4 scans); floor z ≈ −1.4…−1.5 | verified |
| Image orientation | stored rotated 90° (portrait capture); `np.rot90(img, k=-1)` is upright | verified visually (M0 overlays) |
| Depth noise vs. own mesh | \|d_obs − d_render\| median 1.4–2.1 cm, p90 5–9 cm. Signed p99 = +8 cm (obs farther), p1 = −30 cm (obs nearer: mesh holes, thin/unreconstructed surfaces) | 3 frames, 1 scan |
| Open3D `t_hit` with intrinsic rays | equals camera-z depth (max diff 0.0) | verified |
| Render speed (CPU) | ~5 ms per 224×172 frame, ~50 ms at 960×540 | verified (cloud CPU) |
| PLY | ASCII; per-vertex `objectId, globalId, NYU40, Eigen13, RIO27`; 99.5% of faces have one label | verified |
| OBJ vs. PLY | `mesh.refined.v2.obj` and the PLY have the same vertices in the same order (max difference 0.0) and identical faces in the same order on **all 72** texture references; one UV per vertex (`f v/vt/vn` with equal indices); texture 4096² RGB, one material | verified M6 (`lsg m6`) |
| Texture vs. photos | Reference texture rendered at a reference frame's own pose: DINOv2 patch cosine median 0.72–0.77 (dev references); on rescans 0.56–0.61 (other lighting) → appearance distances are taken **relative to the frame median** | verified M6 |
| 3RScan.json transforms | 16 floats, column-major, orthonormal R, **metres** (not the mm the FAQ states). `T_align` maps rescan → reference: rescan structure lies 1.6 / 1.9 / 3.0 cm (median, point-to-surface) from reference structure on the 3 dev rescans; T⁻¹ gives 12–204 cm, translation ×0.001 gives 6–46 cm | verified M0 |
| `T_align` accuracy | Aligned-reference depth residual median 2.9–3.8 cm vs. 1.1–1.3 cm for the rescan's own mesh. Label-free ICP of rescan depth onto the reference mesh corrects 1.6–8.4 cm / 0.4–1.5° → 1.9–2.2 cm. The method uses the **refined** alignment (`geom.align`); GT/eval uses raw 3RScan transforms | verified M0 (3 dev rescans) |
| Depth valid area | Stored depth columns 0–~20 and ~205–223 are never valid (sensor border, ≈ 18% of pixels); depth is otherwise ~85% valid | verified M0 |
| `rigid` entries | 2,933 dicts + 390 bare ints (no transform); `instance_reference == instance_rescan` always | verified |
| Rigid-transform semantics | `T_rigid` maps reference-frame instance points **into the rescan frame**: p_S = T_rigid · p_R. Best of 5 hypotheses on 20/21 dev objects (Chamfer 2.8–7.8 cm, 17 < 5 cm; the 21st is a toilet that barely moved). Motion in the reference frame: **M_ref = T_align · T_rigid** | verified M0 |
| Instance IDs | consistent within an environment (same objectId = same object) | verified via 3DSSG `objects.json` |
| 3DSSG | 1,482 scans in `objects.json`, 1,335 in `relationships.json` (train+val refs and rescans; test rescans have none). Objects carry `nyu40/eigen13/rio27`. 41 predicates with `none` = id 0; `inside` never occurs; `classes.txt` has 528 lines | verified |
| Ref objects absent from rescan semseg (val) | 1,018, vs. only 96 listed `removed` → absence in the rescan annotation ≠ removal (partial rescans) | verified |
| GT completeness (example) | On 19eda6f4, pillow 49 is not listed as moved, but its reference and rescan footprints barely overlap (IoU 0.14 in a shared view) → probably an unlisted move | observed after M0; one example |
| Duplicate `rigid` ids | A rescan's `rigid` list can repeat an id (7 extra entries in val, all with identical transforms) → deduplicated per rescan | verified M1 |
| Small GT moves | 134 of the 370 unique thing/anchor moves in val have a displacement < 0.10 m (median 0.17 m; p25 0.06 m) | verified M1 |

### 2.2 Change ground-truth statistics (validation split, 110 rescans of 47 references)

- rigid (moved): 390 entries = 370 unique things/anchors + 7 duplicate entries + 13 structure (mostly doors). 134 of the 370 move < τ_eval_move = 0.10 m and become `dont_care` (§5), which leaves **235 scored `moved`**;
- removed (listed): 96 (73 scored; 20 still present in the rescan → `dont_care`, of which 12 are things/anchors);
- nonrigid: 54;
- new (derived, rescan IDs not in reference): **224 raw** (structure included); **171** things/anchors are scored (§5): 105 large, 66 small;
- moved *things*: 278 entries (274 unique). Their support-parent set differs between reference and rescan 3DSSG for **12**, but only **4 are real parent→parent swaps**: 7 *gained* a parent (none annotated in the reference), 1 *lost* it (verified M1). 4 more have structure-only parents (excluded);
- relchange set (§2.3): the 73 "parent-changed" events = 47 gained + 4 lost + 18 swapped + 4 structure-only. Most look like gaps in the reference annotation, not real changes; the §7.2(c) definition is fixed at M5 (DECISIONS M1);
- train split: 174 parent-changed things over 793 rescans (same broad definition).

Most-moved classes: chair 80, pillow 31, armchair 23, table 22, ottoman 20, coffee table 17.

→ "Cup moves from table to shelf" is **rare** in 3RScan. Most changes are furniture rearrangement. The relation-change evaluation therefore needs an extra set (§2.3).

### 2.3 Subset (files in `splits/`)

| Split file | Content | Use |
|---|---|---|
| `dev_train.txt` | 3 train pairs (531cff08→19eda6f4, 531cff08→e2847ff7, 0ad2d39b→0ad2d3a1) | development + **all threshold tuning** |
| `eval_val.txt` | all 110 validation pairs (47 refs) with 3DSSG relations | main evaluation, change types: **final tables only** (M7, `--full-val`) |
| `eval_val_small.txt` | 20 val pairs: 20 of the 47 references, one rescan each, seed 0 (`lsg splits subset`). GT support: 51 moved, 17 removed, 19 new, 229 unchanged, 148 absent-unlisted | per-milestone val report (M6, Erik); never tuned on |
| `eval_relchange_train.txt` | 40 train pairs with the most parent-changed moved things (73 events) | relation-update evaluation only; never tuned on |
| `download_tier1.txt` | 162 scan IDs (dev + val) | downloaded (M0) |
| `download_tier2.txt` | 63 more scan IDs (relchange set) | downloaded 2026-11-06 (189 files, 4.0 GB) |
| `download_textures.txt` | the **72 reference scans** of the three pair lists above (written by `lsg splits textures`, M1) | texture files for the appearance residual (§3, §6.6); downloaded at M1 (216 files, 1.1 GB) |

**Class priors (§6.10)** and the **CLIP vocabulary (§6.6)** come from train-split metadata (3RScan.json + 3DSSG objects), **excluding** the references in `eval_relchange_train.txt` and `dev_train.txt`. This needs no downloads.

**Download.** Scan data, three file types per scan (≈50 MB/scan → tier 1 ≈ 8 GB, tier 2 ≈ 3 GB):

```bash
while read id; do for t in semseg.v2.json labels.instances.annotated.v2.ply sequence.zip; do
  python download_files.py -o data --id "$id" --type "$t"; done; done < splits/download_tier1.txt
```

Textures, **reference scans only** (≈17 MB/scan → ≈ 1.2 GB; rescans don't need them):

```bash
while read id; do for t in mesh.refined.v2.obj mesh.refined.mtl mesh.refined_0.png; do
  python download_files.py -o data --id "$id" --type "$t"; done; done < splits/download_textures.txt
```

`train-/val-scans.tfrecords` (2.5 GB) are not used.

## 3. Definitions

- **Reference frame**: the reference scan's mesh frame (z-up, metres). All rescan geometry and poses are mapped into it via `T_align`. On the method side, `T_align` is refined by a label-free ICP pre-pass over the frames being processed (§2.1, `geom.align`).
- **Online processing**: rescan frames are processed in time order with stride `s = 3` *(tune)*. Frames move only ~6 cm / 5° per step, so consecutive frames are strongly correlated.
- **Node classes**: decided from 3DSSG `nyu40` id.
  - `anchor`: nyu40 ∈ {3 cabinet, 4 bed, 6 sofa, 7 table, 10 bookshelf, 12 counter, 14 desk, 15 shelves, 17 dresser, 32 nightstand}.
  - `structure`: nyu40 ∈ {1 wall, 2 floor, 8 door, 9 window, 11 picture, 13 blinds, 16 curtain, 19 mirror, 20 floor mat, 22 ceiling, 28 shower curtain, 30 whiteboard, 38 otherstructure}.
  - `thing`: everything else.
  - Structure is rendered as geometry but is **not** a graph node, except one `room` node (and `floor` as a pseudo-parent).
  - **Thin node**: a thing whose smallest AABB extent is < `appearance.thin_max_extent` = **0.15 m = τ_free** (M6, Erik; spec v2 said 0.10 m). Depth alone cannot tell whether it is still there: a removed thing up to τ_near thick leaves `present` pixels, one between τ_near and τ_free thick leaves neither present nor free (e.g. pillow r50, 0.14 m).
- **Pixel classes** for frame f. Render the full reference mesh from the aligned rescan pose. With d_obs = observed depth and d_exp = rendered depth:
  - `invalid`: d_obs = 0 or no hit;
  - `present`: |d_obs − d_exp| ≤ τ_near = 0.10 m *(tune)*;
  - `free` (seen through): d_obs > d_exp + τ_free = 0.15 m *(tune; same-scan false rate 0.4% of pixels)*;
  - `occluded` / `nearer`: d_obs < d_exp − τ_near. On a node's pixels this is occlusion (no information). Anywhere, it is residual evidence for something new in front (§ Candidate below, with τ_new).
- **Frame vote** for node i. P_i,f = pixels where i is the first hit. A vote is cast only if |P_i,f| ≥ 15 px *(tune)*, and only from ≥ 15 pixels with valid observed depth (fewer → `no-info`; M3):
  - `seen` if present ≥ 50% of valid pixels;
  - `free` if free ≥ 50% of valid pixels;
  - `no-info` otherwise;
  - `unobserved` if |P_i,f| < 15 px (out of frustum or hidden).
- **Seen votes need contrast** (FULL only, M7; Erik, DECISIONS 2026-11-15). d_behind = depth of the nearest reference surface along the pixel's ray whose label is not node i (no surface → ∞). A pixel is *contrastive* if d_behind > d_exp + τ_free: if node i were gone, depth would show it. FULL's `seen` vote needs ≥ 15 valid contrastive pixels, ≥ 50% of them present; `free` is unchanged (≥ 50% of all valid pixels); everything else is `no-info`. A present pixel whose background lies within τ_free (pillow on a sofa) is no information, like occlusion (hard rule 3 applied to the positive side). Switch `visibility.seen_needs_contrast`; B0–B2 keep the rule above. Thin nodes are unaffected when appearance is on (their seen / no-info vote is replaced, below).
- **Viewpoint decorrelation.** A vote (or a voxel hit, below) **counts** only if the camera centre moved ≥ d_dec = 0.30 m, or the viewing direction rotated ≥ θ_dec = 20° *(tune)*, since the last *counted* vote of the same kind for the same node (or voxel). Repeating one viewpoint therefore adds one vote. The per-visit log-odds clip (§6.10) stays as a second guard.
- **Deferred commitment.** During the walk, states are **tentative** (visible in snapshots). `removed` and `moved` become final only at the end of the sequence (or chunk). Reason: a moved object can appear at its new place hundreds of frames after its old place was seen empty.
  - *Immediate commitment* (ablation, M7, Erik: removal only): `removed` is final at the first frame where the node's tentative state is `removed` (free evidence; FULL: also tentative confidence < p_del). The node then skips absorption, matching and FULL's veto, so its later observation can only be taken by another node or become `new`. `moved` cannot be immediate (association runs at the end). Switch `states.commitment: deferred | immediate`.
- **Confirmed in place**: a reference node with ≥ k_seen = 3 *(tune)* counted `seen` votes and **no** counted `free` or `free-like` votes. It is `unchanged` without needing a matched instance, and it does not enter the Hungarian matching. It does absorb the observations that lie inside it (§6.9 stage 0, M5), so its own oracle instance or a mesh-hole candidate is not taken for a `new` object.
- **Candidate** (perception mode). Built in five steps:
  1. **Nearer pixels:** pixels with valid depth and d_obs < d_exp − τ_new (0.10 m *(tune)*).
  2. **Back-project and filter.** Back-project them to the reference frame. Drop points within 5 cm of the reference mesh surface. Drop points near structure surfaces (wall, ceiling, floor), except points standing on the floor.
  3. **Voxel grid:** accumulate the remaining points in a 4 cm *(tune)* grid.
  4. **Multi-view filter:** keep a voxel only if it was hit from ≥ 3 decorrelated viewpoints. This filters people, sensor noise and mesh holes.
  5. **Cluster:** at the end, cluster the kept voxels: connected components, two voxels linked if their Chebyshev distance is ≤ `link_voxels` = 2 voxels *(tune)* (M4: bridges one-voxel gaps of sparse grazing-angle sampling inside one surface). Drop clusters below `min_voxels` = 5 *(tune)*. Drop clusters with ≥ 50% of their voxels inside a reference door's swing region (M4, door swings).

  Each remaining cluster is a candidate. Appearance mismatches off reference nodes, on supporting surfaces, add thin-object voxels (§6.6).
- **Appearance residual** (M6): the reference texture is rendered at the rescan pose and compared with the observed colour patch by patch (14 px patches at 448×252).
  - **Comparable pixels**: valid depth with −τ_near ≤ d_obs − d_exp ≤ τ_free (not occluded, not seen through; M6: includes the thin-object band, not only `present`). A patch is compared if ≥ 50% of its pixels are comparable, and a frame only if it has ≥ 20 compared patches.
  - Features: DINOv2 ViT-S/14 patch-feature cosine distance (default), or mean/std-normalised colour (fallback). The distance is taken **relative to the frame's median** over compared patches (M6: rescans are lit differently from the textured visit).
  - **Thin-node vote** (M6, Erik: the appearance verdict *replaces* the depth vote): a thin node's depth `seen` / `noinfo` vote becomes `free-like` if the **mean** relative distance over its comparable pixels is > τ_node (0.30, DINOv2), else `seen`; < 15 comparable pixels → `noinfo`. A depth `free` vote stays `free`.
  - Thresholds are the p98 of the same quantities on the dev references' *own* sequences (2% same-scan false rate): τ_node per node, τ_app (0.38) per pixel for thin-object candidates.
  - Appearance evidence alone never deletes a node: it needs ≥ k_free_like = 3 counted (decorrelated) free-like votes.
- **Free evidence**: ≥ k_free = 2 *(tune)* counted free + free-like votes, where free-like votes alone (no counted depth free vote) need ≥ k_free_like = 3 (M6). Without appearance this is "≥ 2 counted free votes".
- **Unknown**: a reference node that is not confirmed in place, has no matched instance, and has no free evidence (FULL: or its confidence is still ≥ p_del). It is **kept** and its state is `unknown`. Its confidence is the log-odds value of its counted votes (§6.11): with no counted votes this is the prior, i.e. unchanged (M6 wording).
- **Moved** (prediction): matched, and the ICP-refined centroid displacement is > τ_move = 0.10 m *(tune at M1 from the GT displacement distribution)*. Final at the end of the sequence. Rotation-only moves are out of scope; recall is reported per displacement bucket.
- **Removed** (prediction): unmatched, with free evidence, and (FULL) confidence p < p_del = 0.2 *(tune)*. Final at the end of the sequence. B1 / B2 use the free-evidence part only.
- **New**: an observed instance (oracle) or candidate (perception) not assigned to any reference node.

## 4. Graph JSON (shared format + the fields this project adds)

```json
{"meta": {"scan_ref": "...", "scan_obs": "...|null", "frame": "ref", "units": "m", "version": 2,
          "obs_mode": "perception|oracle|null", "frame_idx": null, "tentative": false},
 "nodes": [{"id": "r12", "level": "thing", "label": "pillow", "nyu40": 18,
            "label_topk": null,
            "bbox_3d": {"min": [x,y,z], "max": [x,y,z]}, "centroid": [x,y,z],
            "embedding": null, "confidence": 0.83, "last_seen": "scan_id|null",
            "state": "unchanged|moved|removed|unknown|new", "anchor": "r5|null",
            "T_move": null, "obs_token": null}],
 "edges": [{"source": "r12", "target": "r5", "type": "on"},
           {"source": "room", "target": "r5", "type": "hierarchy"}]}
```

- `level`:
  - `room`: one per scan;
  - `place`: unused, single room;
  - `anchor`;
  - `thing`.
- `bbox_3d`: axis-aligned in the reference frame. This is a design choice; if the organisers' shared format specifies an OBB, add an `obb` field.
- Edges:
  - `hierarchy`: room→anchor and room→thing;
  - support edges `on|in`: child→parent, **at most one per thing**; the parent is an anchor, a thing, or `"floor"`;
  - `next_to`: computed and exported, but not a headline metric.
- `anchor` = first anchor up the support chain (it can differ from the direct parent, e.g. book on box on shelf).
- Removed nodes stay in the file with `state: "removed"` (deletion is a view filter). `--prune` drops them.
- `label_topk`: for new nodes in perception mode, `[[label, score], …]` from CLIP; `label` is the top-1. For reference nodes, null.
- `obs_token`:
  - oracle mode: the shuffled oracle instance id, written only for evaluation mapping and never read by the method;
  - perception mode: the candidate id `c<k>`.
- **Snapshots** are written every N processed frames *(config `online.snapshot_every`)* and after the last frame, to `outputs/<run_id>/snapshots/<mode>/<ref>_<rescan>/frame_XXXXXX.json` (M3). They use the same schema with `meta.frame_idx` set and `meta.tentative: true`. Tentative states are vote-only (`removed` with ≥ k_free counted free votes, `unchanged` if confirmed in place, else `unknown`), because association is deferred to the end.

## 5. Ground truth (only in `src/lsg/eval/`)

Per pair (ref R, rescan S). O_R and O_S are the objectIds in their semseg files.

- **Precedence** (M1): listed removed > nonrigid > rigid > present in O_S (unchanged) > absent-unlisted. Duplicate `rigid` ids count once.
- **moved**: IDs in S's `rigid` list (dict or int form).
  - GT displacement = ‖T_align · T_rigid · c_R − c_R‖, where c_R is the reference centroid (raw 3RScan `T_align`). `T_rigid` alone maps into the rescan frame (§2.1, confirmed at M0), so M_ref = T_align · T_rigid is the motion in the reference frame.
  - GT moved with displacement < τ_eval_move = **0.10 m** *(fixed at M1 from the histogram, `lsg m1`: 134 of 370 val moves)* → `dont_care` for the moved/unchanged decision.
  - GT new location (for localisation error) = centroid of the rescan instance's points, mapped with raw `T_align`.
- **removed**: IDs in `removed`. If a listed removed ID still appears in O_S (20 cases in val) → `dont_care`.
- **nonrigid**: `dont_care` for change-type metrics (either `unchanged` or `moved` is accepted). Excluded from relation metrics.
- **new**: O_S \ O_R, things and anchors only (structure dropped). Known to be noisy (Objects Can Move reports inconsistent "new" annotation) → audited (§7.4).
- **unchanged**: (O_R ∩ O_S) minus the above.
- **absent-unlisted**: O_R \ O_S that are not listed removed. Excluded from removed P/R. Used for the **false-deletion rate** (§7.1).
- Only `thing` + `anchor` nodes are scored.
- **Size class** (for breakdowns): `small` = thing with largest AABB extent < 0.5 m; `large` = anchors and things ≥ 0.5 m.
- **Prediction ↔ GT matching:**
  - Reference nodes keep their id (`r<objectId>`) in both modes, so their states are compared with GT by id.
  - Predicted **new** nodes are matched to GT instances by the mode's rule:
    - **oracle:** by `obs_token` → rescan objectId (unshuffled in `eval/`);
    - **perception:** a prediction and a GT instance can match if they have the same label (or the same nyu40, with label → nyu40 from 3DSSG `objects.json`), **and** centroid distance ≤ 0.30 m or 3D AABB IoU ≥ 0.25. Matching is greedy by IoU, one-to-one. GT instances are taken in the reference frame (raw `T_align`).
  - For moved nodes: **localisation error** = ‖predicted new centroid − GT new centroid‖.
- **Relations GT** from rescan 3DSSG, child = subject. Only things are scored:
  - `on` ⇐ {standing on, lying on, supported by, hanging on};
  - `in` ⇐ {standing in, lying in, hanging in, inside, build in};
  - parent `floor` (structure "floor") is kept as a valid parent;
  - parent wall/ceiling/window/other structure → the thing is excluded;
  - `close by` → `next_to` (reported only);
  - everything else is ignored: attached to, left/right/front/behind, comparatives, same-*, part of, connected to, belonging to, cover, leaning against;
  - multiple GT parents → any one counts as correct (1 case in val).
  - In perception mode, relations of new nodes are scored only for nodes matched to a GT instance.

## 6. Modules and interfaces (inputs → outputs; no implementation here)

`src/lsg/` — every function is pure where possible and works in the reference frame. ✅ = exists (M0–M7).

1. `io` ✅
   - `scan3r.load_meta(root) -> Meta` (3RScan.json, column-major transforms) ✅;
   - `scan3r.load_scan(root, id) -> Scan` (vertices, faces, vertex objectId, per-face label by majority, instances {id: label, nyu40, centroid, aabb}, intrinsics) ✅. Annotation reader: on rescans only from `oracle/` and `eval/`;
   - `scan3r.open_sequence(root, id) -> SequenceReader` and `iter_frames(seq, stride, idx=None) -> Frame` (idx, pose_c2w, depth[m]; colour on demand) ✅;
   - `scan3r.load_textured_mesh(root, id) -> TexturedMesh` (OBJ vertices, faces, per-face-corner UVs, texture image; **reference scans only**; own OBJ parser `read_obj`) ✅ (M6);
   - `dssg.load_relations / load_objects / load_predicates` ✅;
   - `splits.write_texture_list(split_files, out)` → `splits/download_textures.txt` (CLI `lsg splits textures`) ✅.
2. `geom` ✅
   - `render.Renderer(vertices, faces, face_labels)`, `.render(K, T_w2c, W, H) -> (depth[m], label_map)`, `.distance(points)` ✅. Open3D RaycastingScene on CPU; label map via `primitive_ids`.
   - `render.TextureRenderer(textured_mesh)`, `.render_color(K, T_w2c, W, H) -> (rgb, hit)`: own RaycastingScene on the OBJ geometry; `primitive_ids` + `primitive_uvs` (barycentric, hit = (1−a−b)·c0 + a·c1 + b·c2) → interpolate corner UVs → bilinear lookup in `mesh.refined_0.png`, v flipped ✅ (M6; ~55 ms at 448×252). `Renderer.render(..., return_prim=True)` and `.face_normal_z()` give the per-pixel support-surface test.
   - `render.Renderer.depth_behind(K, T_w2c, W, H, depth, labels, active, eps) -> d_behind` ✅ (M7): all intersections of the active pixels' rays (`RaycastingScene.list_intersections`); the nearest hit whose label differs from the first hit's, at t ≥ d_exp − ε (ε = 1 mm, so a surface coincident with the node's back face counts); ∞ if none. ~7.6 ms per 224×172 frame on a dev reference (render ~6 ms).
   - `align.refine_from_sequence(seq, ref_V, ref_F, T_align, cfg, frame_ids) -> (T_refined, info)` ✅. A pre-pass over the frames of the processed chunk only (not online; DECISIONS). Corrections beyond `max_delta_t` / `max_delta_rot_deg` are rejected.
3. `graph` ✅
   - `graph.model`: `Graph` / `Node` / `Edge`, §4 JSON (`save` / `load`), `validate(graph)` ✅.
   - `graph.build.build_reference_graph(scan, objects, relations=None, mode="gt"|"geometric", cfg) -> Graph` ✅. GT mode picks one parent when 3DSSG gives several: anchor > thing > floor, then `on` > `in`, then lowest id. Anchors carry no support edge.
   - `graph.relations`: `infer_support(child_pts, candidates, floor_z, cfg) -> (parent_id|"floor"|None, "on"|"in", score)` ✅; `gt_support` (§5 mapping), `estimate_floor_z` (median z of floor-labelled reference vertices). Rules (all *(tune)* on reference scans of dev; values tuned at M1 in `configs/default.yaml`):
     - contact: ≥ 30% of the child's bottom-slice points (lowest 3 cm) have a candidate point within 5 cm horizontally and 0–10 cm below → `on`; supporting points must lie ≥ `support_min_height` = 5 cm above the floor (M1: a neighbour's bottom edge at floor level does not carry a thing standing on the floor);
     - containment: ≥ 60% of child points inside the candidate's AABB, and the candidate is an anchor → `in`;
     - best score wins, anchor beats thing on ties;
     - otherwise `floor` if the child's bottom is < `floor_max_gap` above floor_z (spec start 10 cm; **20 cm** after M1 dev tuning, because object bottoms are often unreconstructed).
4. `obs` — the observation interface ✅ (M2).
   - `ObsInstance`:
     - `token`, `label`, `label_topk: list[(label, score)]` (oracle: `[(label, 1.0)]`), `nyu40`;
     - `points` (N×3, **reference frame**);
     - `masks: {frame_idx: bool[172, 224]}` (depth resolution; stored bit-packed, `PackedMasks`);
     - `crops`: best colour crops, upright.
   - `ObsSource` protocol:
     - `begin(ref_ctx, seq_info)`;
     - `observe(frame, ref_render) -> None`, called per processed frame, in time order;
     - `finish() -> list[ObsInstance]`.
   - Implementations: `oracle.OracleSource` ✅, `perception.PerceptionSource` ✅ (M4). The change logic depends only on this interface.
   - Also `RefContext` (reference scan, G_ref, floor_z, per-object nyu40; `faces_of(oids)` gives a reference sub-mesh, M4; `thin_oids(max_extent)`, M6) and `SeqInfo` (rescan id, intrinsics, **refined** `T_align`, processed frame ids). `ref_render` (`RefRender`: reference depth + objectId labels at the frame's aligned pose; M6: + `appearance` map and `up` = |normal z| of the first-hit face, when appearance ran on the frame) is passed by `run_online` from M3 on.
   - `ObsInstance.kind` (M6): `depth` or `appearance` (perception thin-object candidate).
5. `oracle` (upper bound) ✅ (M2)
   - `OracleSource`: renders the rescan's own labelled mesh at each frame, collecting per-instance masks, points (vertices of hit triangles, *these frames only*, mapped with the refined `T_align`) and crops (optional, off by default). A pixel counts only where the sensor depth is valid. IDs are shuffled with a seeded permutation; `token_map()` is read only by `eval/`.
   - It is the only reader of rescan labels outside `eval/`.
   - Instances with < 50 observed points *(tune)* are dropped.
6. `perception` (headline) ✅ (M4; `appearance` M6). **Must never import `oracle/` or `eval/`, and never read rescan semseg, rescan PLY labels, rescan 3DSSG or the `rigid/removed/nonrigid` lists.**
   - `residual.nearer_points(depth, d_exp, K, pose_ref, filt, cfg) -> (points, pixels)` ✅: §3 candidate steps 1–2. `SurfaceFilter(ref_ctx, cfg)`: drop points ≤ `surf_dist` = 5 cm from the reference mesh and ≤ `struct_dist` = 10 cm from reference structure faces (all structure classes incl. doors, except floor and floor mat). `d_exp` comes from `run_online`'s reference render; pixels without a reference hit are excluded; depth > 4 m is ignored.
   - `voxels.VoxelEvidence(voxel=0.04, d_dec, θ_dec)` ✅: `.add(points, cam_pose)` counts decorrelated viewpoints per voxel (the `Decorrelator` rule, vectorised); `.kept(min_views=3)`; per-voxel mean point.
   - `cluster.cluster(voxel_idx, coords, points, cfg) -> list[Candidate]` ✅: connected components (scipy KD-tree + sparse graph, Chebyshev link ≤ `link_voxels`), `min_voxels` *(tune)*; `door_regions` / `drop_door_clusters`: swing region = vertical cylinders of radius door width + 10 cm around both possible hinge edges.
   - `appearance` ✅ (M6):
     - `patch_similarity(rgb_obs, rgb_exp, extractor)`: per-patch cosine, images upright for the network, grid mapped back to the stored orientation; `mismatch(sim, d_obs, d_exp, cfg, render_wh) -> AppearanceMap` (`patch_ok`, `comparable_px`, `mismatch_px`, `rel_px`; §3 rules);
     - features: `DinoV2Features` (timm `vit_small_patch14_dinov2.lvd142m`, frozen, CUDA → CPU fallback; default) or `ColorFeatures` (normalised colour; fallback, fast tests);
     - `AppearanceRunner`: per frame colour → resize, texture render, similarity (cached in `cache/appearance/`, rounded to 1e-3), map;
     - output (i): thin-node votes (`change.visibility.thin_votes`); output (ii): `PerceptionSource` back-projects `mismatch_px` pixels whose depth agrees (|Δd| ≤ τ_near) and whose first hit is a support surface (non-thin anchor / thing or floor, |normal z| ≥ 0.8) into a **separate** voxel grid (structure filter only), clustered with the §3 rules; clusters touching a depth candidate are dropped; switch `appearance.thin_candidates`.
   - `labeler.CLIPLabeler(vocab, model="ViT-B-32-quickgelu", pretrained="openai")` ✅ (`open_clip_torch`, frozen):
     - `.topk(crops, k) -> label_topk`: mean normalised image embedding of the K = 5 largest upright crops → softmax over `"a photo of a {label}"` → top-k; cached in `cache/clip/`;
     - vocabulary = reference-graph labels ∪ the 100 most frequent thing/anchor labels of train metadata (`lsg.train_meta`, §2.3; outside `perception/`, which may not read 3DSSG).
   - `source.PerceptionSource(labeler, label_nyu, cfg)` ✅: assembles the above into `ObsInstance`s. Points = one mean point per voxel; masks = the pixels whose residual point fell into one of the cluster's voxels (stored during `observe`, no re-projection); crops = upright colour crops of the 5 largest masks (depth → colour by resolution scale, +10% margin); label = CLIP top-1; nyu40 from the vocabulary; tokens `c<k>` by cluster size.
7. `change.pipeline` ✅ (M2) and `change.online` ✅ (M3)
   - `pipeline.select_frames` (stride + §7.3 chunk); `pipeline.prepare_pair` (G_ref + cached refined `T_align`; it gets only the raw `T_align` matrix, never the rescan's change lists); (M2's `collect_observations` was replaced by `run_online` at M3).
   - `baselines.b0` ✅ (§8; greedy one-to-one same-label matching); `baselines.b1` ✅ (M3; B0 matching + the §3 removed / unknown rule from counted free votes); `baselines.b2` ✅ (M5; §6.9 + §6.10). Every method is `method(G_ref, obs, cfg, rescan_id, obs_mode, votes=, ref_ctx=)` (M5: `ref_ctx` gives B2 the reference node points). Methods decide the states; `update_graph` (§6.12) recomputes the relations afterwards.
   - `run_online(ref_ctx, seq_info, seq, obs_source, cfg, snapshot_dir=None, obs_mode=None, renderer=None, appearance=None, priors=None, contrast=False) -> OnlineResult` ✅ (obs instances, per-node `NodeVotes`, snapshot paths, timings; M6: `obs_views` = decorrelated viewpoints per observation, appearance `stats`). Appearance runs on frames where a thin node has a depth seen / noinfo vote, or on every frame if the source `wants_appearance` (thin candidates). G_ref comes from `ref_ctx.graph` and the frame ids from `seq_info` (M3). Per processed frame, in time order:
     1. render the reference (depth, labels, and colour if appearance is on);
     2. compute frame votes (`change.visibility`; with `contrast` (FULL, M7) from `depth_behind`);
     3. replace thin nodes' votes by the appearance verdict (M6), apply decorrelation (confidence is computed from the counted votes; snapshots carry the tentative value);
     4. call `obs_source.observe` (residual accumulation in perception mode);
     5. write a snapshot every N frames.
   - At the end it calls `obs_source.finish()`, then association, then classification. States become final only then.
   - Every node records `NodeVotes.commit_removed_frame`, the first frame with a tentative `removed` state (with `priors`: also tentative p < p_del); read only with `states.commitment: immediate` (M7).
8. `change.visibility` ✅ (M3)
   - `frame_votes(node_oids, ref_depth, ref_labels, obs_depth, cfg, d_behind=None) -> dict[objectId, FrameVote]` (one frame, §3 rules; `unobserved` included; `d_behind` given → seen needs contrast, M7, `FrameVote.n_contrast` / `n_present_contrast`). A vote needs ≥ `min_mask_px` first-hit pixels (else `unobserved`) **and** ≥ `min_mask_px` valid ones (else `noinfo`). Optional per-label mask erosion `visibility.erode_px` (default 0).
   - `Decorrelator(d_dec, θ_dec, enabled)`: per key (node, vote kind) or voxel, the pose of the last counted event; `.counts(key, cam_pose) -> bool` (centre moved ≥ d_dec or optical axis turned ≥ θ_dec).
   - `NodeVotes` (raw and counted seen / free / free-like, noinfo, first free / free-like / last seen frame); `has_free_evidence`, `confirmed_in_place` (§3, incl. free-like, M6).
   - `thin_votes(depth_votes, thin_oids, ref_labels, comparable_px, rel_px, cfg, tau_node)` (M6): the §3 thin-node vote.
9. `change.association` ✅ (M5)
   - **Stage 0, absorption** (M5, Erik): `absorb_confirmed(confirmed, obs, mode, cfg)`. An observation is absorbed by a confirmed-in-place node if its label is compatible (oracle: same label or nyu40; perception: the node's label in `label_topk` with p ≥ p_min) and ≥ `absorb.min_frac` = 50% of its points lie inside the node's AABB + `absorb.margin` = 10 cm. Many-to-one: each observation goes to its best node (largest point share, then centroid distance, then id); the node's `obs_token` = its best absorbed observation. Absorbed observations are neither `new` nor matched.
   - `associate(ref_nodes, obs_instances, mode, cfg) -> (pairs, unmatched_ref, unmatched_obs)`.
   - Only reference nodes **not confirmed in place** and observations not absorbed take part.
   - Cost:
     - label term: oracle = hard gate (+∞ if labels differ; fallback same nyu40 + 0.3 penalty, **except** the catch-all classes 39 otherfurniture / 40 otherprop, M5 Erik); perception = soft, −log max(p(label_ref), p_min) from `label_topk` *(p_min tune)*;
     - plus w_g · size term (mean |log| ratio of sorted AABB extents, extents floored at 2 cm) + w_p · min(dist / 3 m, 1) (position, weak); w_g = w_p = 1 *(tune)*;
     - optional + w_a · (1 − cos(DINOv2 embedding)) — the old B3, **off by default** (not implemented).
   - Hungarian assignment (`scipy.optimize.linear_sum_assignment` on costs clipped at τ), gate cost < τ_assoc *(tune; per mode: oracle 2.0, perception 3.5)*; p_min = 0.05, nyu40 penalty 0.3.
10. `change.classify` ✅ (M5)
    - `classify(pairs, unmatched_ref, votes, confirmed, ref_points, cfg) -> dict[node_id, Decision]`, run at the end of the sequence (M5: `ref_points` added, because ICP needs the reference node points).
    - ICP (point-to-point) of the **partial observed points onto the reference node points** gives T_move (as its inverse) and the moved/unchanged decision: ‖T_move·c_R − c_R‖ > τ_move. Schedule 0.20 / 0.10 / 0.05 m. Two inits (M5): the centroid offset and identity ("not moved"); the higher fitness wins, near ties (≤ 0.02) go to the lower RMSE. Below fitness 0.3 the centroid offset is used.
    - Unmatched reference node → `removed` if it has free evidence (§3), else `unknown`. Unmatched observation → `new`.
11. `change.confidence` ✅ (M6)
    - Log-odds per node, starting at l₀ = logit(p_persist(class)) from train priors (`train_meta.class_priors`: per (reference object, train rescan) pair, present in the rescan's 3DSSG objects vs listed `removed`, absent-unlisted not counted; Beta-smoothed with κ = 20, label → nyu40 → global back-off; global 0.968). `confidence.class_priors: false` → fallback p = 0.9 (ablation).
    - From **counted** (decorrelated) votes: Δl = L_hit · n_seen − L_miss · n_free − L_free_like · n_free_like, clipped per visit to [−8, +8]. **Rescaled at M6 (Erik):** L_miss = L_free_like = 2.5 (2 free votes delete a median-prior thing: chair p 0.968 → 0.17; a bed, p ≈ 0.996, needs 3), L_hit *(tune on dev: 0.5 / 1 / 2.5)*. With the spec's 0.4 / ±3, priors of 0.91–1.0 could never reach p_del.
    - A matched node gets l₀ + L_hit per decorrelated viewpoint of its observation (`OnlineResult.obs_views`, "seen at the new location"); its free votes at the old location are **ignored** (applied at the end, because association is deferred). New nodes: L_hit · views from l = 0.
    - Delete rule: §3 `removed` (free evidence and p < p_del). Every node of FULL's graph carries `confidence`.
    - FULL's seen votes are contrastive (§3, M7): a seen vote from a pixel whose background is within τ_free can no longer veto a deletion.
12. `change.update` ✅ (M5)
    - `update_graph(G_states, ref_ctx, obs, cfg, policy) -> G_upd`. M5 split: the method (B2) moves nodes (T_move applied to the reference points → centroid and bbox), adds new nodes (with `label_topk`) and sets `state` / `last_seen`; `update_graph` handles the relations. `confidence` stays None until M6.
    - `policy="update"` (the method): re-checks moved and new things **and** things whose parent moved or was removed; the rest keep their reference edge. A re-checked thing keeps its old parent if that parent still carries it at the new pose (`relations.still_supports`, the same contact / containment / floor rules; M5, Erik); otherwise `infer_support` picks a new parent. Candidates = non-removed anchor/thing nodes with their current points (reference mesh vertices, `T_move` applied when moved; new nodes: observation points). Removed nodes keep their old edge in the file (view filter) but are never a parent. `anchor` is recomputed up the chain; `next_to` edges of moved / new nodes are recomputed.
    - `policy="stale"` = baseline **R0** (reference relations; new nodes get no parent); `policy="geometric"` = baseline **R1** (`infer_support` for every thing). The harness runs all three on the same method output; config `update.relations` picks the saved / scored graph.
    - Optional rule (ablation, M7) ✅: "children follow parent", `update.children_follow_parent` (default off). In FULL (`baselines.follow_parent`, a state decision, so it lives in the method, before `update_graph`): a thing in state `unknown` with no counted votes whose reference support parent is an anchor in state `moved` gets the anchor's `T_move` and state `moved`; `update_graph` then re-checks its parent as for every moved thing.
13. `eval`
    - `gt.gt_changes(scan_meta, R, S, objs_R, objs_S, cfg) -> GTLabels` (§5) ✅; `gt.gt_parent_changes` ✅; `gt_stats.run_m1` (`lsg m1`) ✅;
    - `metrics.match_new(pred_new, gt, mode, cfg)` (§5 matching rules) ✅;
    - `metrics.perception_obs_truth` (perception token → GT rescan instance, §5 rule against GT new ∪ rescan instances of reference objects) and `metrics.candidate_counts` (candidate recall vs GT new / moved-to, class-agnostic and labelled; share of candidates explained by any GT instance) ✅ (M4);
    - `metrics.score_pair(G_upd, gt, mode, cfg, obs_truth)` → per-pair counts, and `metrics.aggregate` (micro-average + bootstrap CI), including localisation error and size buckets ✅ (the spec's `score_changes`);
    - `harness.run_eval` / `run_split` (`lsg eval`, `lsg run`) ✅;
    - `relations.score_relations(G, gt, gs_S, parent_kinds, new_tp, obs_truth, prefix)` ✅ (M5; §7.2 counts per policy, subset, universe and floor split);
    - `audit` ✅ (M8; `lsg audit sample | summarize`): a FULL val rerun with per-object details (`run_eval(details=True)`, `score_pair(details=True)`; must reproduce the M7 FULL `metrics`), false-positive pools per mode (moved / removed incl. absent-unlisted, unmatched new) + the confidence-veto set, a seeded sample of 10 per mode × category, one photo panel per case (reference view, rescan at the old place, rescan at the predicted place, top-down), `audit.csv` for manual verdicts (`gt_missing | method_error | unclear`), Wilson CIs; `figures.gt_panel` / `topdown_for_pair` (GT panel for the top-down figure).
14. `viz` ✅ (M8; reads method inputs / outputs only, no `oracle` / `eval` imports)
    - `rerun_timeline.rerun_timeline(root, snap_dir, out, graph_final, images)`: `.rrd` with the reference mesh, the rescan camera (pinhole + trajectory) on a `frame` timeline, node boxes by tentative state per snapshot, and the final graph with old→new arrows. `run_online` writes `seq_info.json` (refined `T_align`, frame ids) next to the snapshots;
    - `topdown.topdown_png(G_ref, G_upd, out, outline, gt)`: before / after (/ GT) footprints, coloured by state (dataviz palette, DECISIONS M8).

CLI:
- `uv run lsg run --pair R S --obs perception,oracle --method full --frac 1.0 --seed 0 [--no-snapshots]` ✅ (both modes, B0 / B1; writes snapshots; perception also writes `candidates/<ref>_<rescan>.json`);
- `uv run lsg eval --split eval_val --obs perception,oracle --method … [--workers N] [--snapshots]` ✅ (both modes, B0 / B1);
- `uv run lsg ablate --names <rows|all|group:…> --split eval_val [--obs …] [--run-id …] [--list]` ✅ (M7; one metrics.json per row, `eval.ablations`);
- `uv run lsg m7 [--run-id m7_final] [--dev-only] [--no-sweep] [--no-frac] [--no-relchange] [--no-determinism] [--no-resume] [--workers N]` ✅ (M7; all rows, tables, plots; a rerun with the same id resumes rows made from the same config + clean git commit);
- `uv run lsg viz topdown --run <dir> --pair R S [--mode …] [--gt]` ✅, `uv run lsg viz rerun --run <dir> --pair R S [--images] [--spawn]` ✅ (M8); `lsg run --snapshot-every N` ✅;
- `uv run lsg audit sample [--run-id m8_audit] [--from-run …] [--n 10]` ✅, `uv run lsg audit summarize` ✅ (M8, §7.4);
- `uv run lsg splits textures` ✅;
- `uv run lsg graph --scan R --mode gt|geometric [--out f.json]` ✅;
- `uv run lsg m0` ✅, `uv run lsg m1` ✅, `uv run lsg m4 [--no-sweep]` ✅, `uv run lsg m5 [--no-sweep] [--no-val]` ✅, `uv run lsg m6 [--no-sweep] [--no-val] [--no-objply]` ✅;
- `--method B0|B1|B2|FULL` ✅ (FULL: M6; appearance on / off via `appearance.enabled`).

## 7. Evaluation protocol

Every table reports **two rows per method**, `oracle` (upper bound) and `perception`, with support counts.

### 7.1 Change detection (per pair, then micro-averaged over the split)

- Precision / recall / F1 for **moved**, **removed** and **new**, plus accuracy on **unchanged**.
  - Reference-node states are compared with GT by object ID. New nodes are matched by the mode's rule (§5).
  - Scored universe for moved P/R, removed P/R and unchanged accuracy: things/anchors with GT status `moved | removed | unchanged`. `dont_care` **and `absent-unlisted`** objects are excluded (M2, Erik).
  - A prediction of `unknown` counts as *not removed* and *not moved*.
  - Unchanged is reported twice (M2, Erik): **strict accuracy** (predicted `unchanged`) and **kept rate** (`unchanged` or `unknown`).
  - A predicted new node that corresponds to a `dont_care` reference object is ignored (neither TP nor FP; M2, Erik).
- **False-deletion rate**: share of `absent-unlisted` reference objects that the method deletes. Lower is better; the expected baseline failure is that B0 deletes many of them. Report it next to removed P/R, because GT precision is unreliable.
- **False-move rate** (M2, Erik): share of `absent-unlisted` reference objects predicted `moved`, next to the false-deletion rate.
- **Unknown rate**: share of reference nodes left `unknown`.
- **Association accuracy**: correct ref→obs pairs / GT-matchable pairs. Report both strict and *same-label tolerant* (swaps between identical-label objects not penalised; chairs dominate). In perception mode, an obs counts as correct for a reference node if it matches that node's GT rescan instance (§5 rule).
- Moved recall broken down by GT displacement buckets: [0.1, 0.3), [0.3, 1), ≥ 1 m.
- **Moved localisation error** (m): median and p90 over correctly detected moves.
- **New / moved recall by size class** (`small` vs `large`, §5).
- Uncertainty: 95% bootstrap CI over pairs (1,000 resamples). **Always print support counts** (removed n ≈ 96 in val).

### 7.2 Relations (things only; GT = rescan 3DSSG, §5)

- **Parent accuracy** (exact parent node; `floor` counts) and **type accuracy** (on/in | correct parent). Reported on (definitions fixed at M5, Erik):
  - (a) all things with a GT rescan parent. **GT-defined universe** (headline): reference things present in the rescan (GT unchanged / moved / `dont_care` small move; not nonrigid, not removed-but-present, not structure-only), plus GT-new things matched to a predicted new node (mode's §5 rule). The node's current support edge is scored whatever its predicted state. A second row, **associated only**, keeps the things the method associated (predicted unchanged / moved; matched new nodes);
  - (b) GT-moved things;
  - (c) moved things whose GT parent was **swapped** (reference and rescan parents both present and different): val n = 4, relchange n = 18. **(c-broad)**: swapped + **gained** (no reference parent annotated): val 11, relchange 65; mostly reference annotation gaps. "Lost" (no rescan parent) cannot be scored. R0 is 0% on (c) by construction;
  - each also reported **with and without floor parents**;
  - parents are compared in rescan objectId space: predicted `r<id>` → id, `n<k>` → its token's GT rescan instance.
- Baselines:
  - R0 "stale graph": keep reference relations; perfect on unchanged things, so (b) and (c) are the interesting rows;
  - R1 geometric rules.
- Sanity: rule accuracy on **reference scans** against reference 3DSSG (no change involved). This must reach a reasonable level before the relation results count.

### 7.3 Observation-fraction ablation

- 100 / 50 / 25 / 10% of frames as **one contiguous chunk** with a random start (3 seeds), mean ± sd. A prefix is the special case start = 0.
- Both modes see only the chunk: the oracle sees only instances visible in the chunk and only their visible points; perception accumulates residuals only from the chunk. The ICP alignment pre-pass also uses only the chunk's frames.

### 7.4 Ground-truth audit

- Manually inspect 30 false positives (moved, removed and new) on val, in both modes, using the crops from `audit_sample`.
- Report the share that is actually a missing GT label (e.g. the unlisted pillow move in §2.1).
- This is the honest answer to "the GT is not exhaustive".

### 7.5 Reproducibility

- Same config + seed → identical `metrics` block in `metrics.json`. The `created` and `run_id` fields differ by design, and so does the `run` block (wall times, cache hits).
- Fresh multi-threaded ICP (`geom.align`) is not bit-exact between runs (~1e-15 m). The refined `T_align` is cached in `cache/align/`, and localisation errors are rounded to 1 µm, so cold and cached runs give the same `metrics` (verified at M2 on val).
- GPU models run in eval mode with fixed seeds. Their outputs are cached per (scan, frame, model) in `cache/`, so reruns read identical features.

## 8. Baselines and ablations (each in `oracle` and `perception` mode)

| ID | Association | Removal rule | Confidence |
|---|---|---|---|
| B0 naive | nearest same-label (top-1) observation within 0.5 m → unchanged; same label farther → moved | unmatched → removed | none |
| B1 + visibility | B0 matching | online votes, decorrelated, deferred: unmatched → removed only with ≥ k_free free votes, else unknown | none |
| B2 + Hungarian | §6.9: confirmed-in-place skip, Hungarian on geometry + label term (hard in oracle, soft in perception) | as B1 | none |
| FULL ✅ (M6, M7) | B2 (+ thin-object candidates in perception) | §6.11 log-odds + class priors (removed needs p < p_del); depth **+ appearance residual** (thin-node votes); seen votes need contrast (M7) | yes |

Ablations (M7: all ✅ as rows of `eval.ablations.registry`, run by `lsg m7`; each one switch off / on relative to FULL, both modes, full `eval_val`):
- **depth-only vs depth + appearance residual** (the brief's ablation; `FULL-appearance`);
- seen votes need contrast off (`FULL-contrast` = M6 FULL);
- thin-object candidates off (`FULL-thincand`);
- decorrelation on / off (`FULL-decorr`, and `B1-decorr`, where it acts directly on the removal rule);
- deferred vs immediate commitment (`FULL+immediate`, §3);
- oracle vs perception (every table);
- observed fraction (§7.3; 50 / 25 / 10% × chunk seeds 0–2, `FULL_frac*`);
- class priors on / off (`FULL-priors`, fallback p = 0.9);
- "children follow parent" on / off (`FULL+follow`, §6.12);
- keep-supported-parent off (`FULL-keepparent`, §6.12);
- τ_free sweep (0.05 / 0.10 / 0.20 / 0.30 m; 0.15 = FULL; `appearance.thin_max_extent` stays 0.15 m);
- relations on the relchange set: B2, FULL, `FULL+follow`, `FULL-keepparent`;
- *optional, not done:* + DINOv2 embedding term in association (old B3; cut-list item 1).

Expected story:
- B0→B1 is the big win (false deletions on partial rescans).
- The oracle→perception gap is largest for `new` and for small things.
- Appearance residual helps mainly for thin things (books, towels) and may be small overall; that is still a valid finding.
- Decorrelation mostly protects against many votes from one standpoint.

## 9. Milestones (dates assume 9–27 Nov; M0 and M1 finished early on 5 Nov)

| M | Dates | Deliverable | Acceptance check |
|---|---|---|---|
| M0 ✅ | done 5 Nov | loaders, renderer, `T_align` (+ ICP refinement), overlay script; finalise §2.1 | **Passed** (commit `dd66381`): (1) aligned depth median 1.9–2.2 cm < 3 cm; (2) structure 1.6–3.0 cm < 5 cm (metres); (3) `rigid` → rescan frame, 17/21 < 5 cm; (4) overlays checked; (5) pytest green |
| M1 ✅ | done 5 Nov | reference graph (GT + geometric relations), `gt_changes`, GT statistics; `lsg splits textures` + texture download started | **Passed** (`lsg m1`, git 5ed8260): val counts 390 / 96 / 224 raw (171 new scored); histogram saved → τ_eval_move = 0.10 m (134 / 370 val moves `dont_care`); rule parent accuracy on reference scans 92.9% dev (n = 28, tuned) / 85.2% val (n = 575; 85.9% without floor parents), type accuracy given parent 100% / 85.3%; `download_textures.txt` lists 72 references, all downloaded |
| M2 ✅ | done 6 Nov | `obs` interface + `OracleSource`; eval harness (both matching rules, size buckets, localisation error, bootstrap); B0 (oracle) | **Passed** (`lsg eval --split eval_val --obs oracle --method B0`, git 4577e4b): metrics.json written in **69 s** cold (alignment cache empty; 44 s cached; 4 workers), GT support 235 moved / 73 removed / 171 new / 1,401 unchanged / 737 absent-unlisted; `test_metrics.py` 5-object toy case incl. the perception rule; **no-leak test** (`test_no_leak.py`: AST import scan of all method-side packages, fresh-interpreter import closure, rescan-annotation reads in `perception/`); two cold runs + one cached run → identical `metrics`. B0 oracle on val: moved F1 30.0%, removed P 21.2% / R 69.9%, new F1 70.1%, **false-deletion rate 95.0%** |
| M3 ✅ | done 6 Nov | online loop (`change.online`, `visibility`, `Decorrelator`), deferred commitment, snapshots; B1 (oracle) | **Passed** (git 6e71d0c): **decorrelation unit test** (`test_visibility.py`: the same viewpoint 30× → 1 counted vote; also end-to-end in `test_online.py`: 30 raw free → 1 counted → `unknown`); **synthetic scene** (`test_online.py`: box removed → ≥ 2 counted free votes → `removed` at the end; occluded / out of frustum → `unknown`, B0 → `removed`); **dev**: B1 false-deletion rate **16.7%** (2 / 12) < B0's 100%; **determinism**: two dev runs → identical `metrics`. B1 oracle on val (reported once): false-deletion rate **9.5%** (B0 95.0%), removed P 66.0% / R 45.2%, unchanged kept 92.9%, unknown rate 31.6% |
| M4 ✅ | done 7 Nov | perception end-to-end: residual points, `VoxelEvidence`, clustering, CLIP labelling, `PerceptionSource`; B0 + B1 in perception mode | **Passed** (`lsg m4` → `outputs/m4_final`, git 9d67e8b): **synthetic** box added → exactly one candidate, centroid error 2.1 cm vs the observable-surface centroid (5.1 cm vs the solid centre, all in z; DECISIONS M4); **overlay PNGs** on 3 dev frames, inspected; **candidate recall on dev** (B1, perception): new 66.7% (2 / 3), moved-to 66.7% (10 / 15; labelled 26.7%), 27.3 candidates per pair; **no-leak test** on the real `perception/` code. Val (reported once): perception B1 false-deletion 9.4%, unchanged kept 90.4%, candidate recall new 27.5% / moved-to 46.4%; B0 matching limits perception (unchanged strict 2.2%, new P 3.3%) until B2 |
| M5 ✅ | done 7–9 Nov | association changes (confirmed-in-place skip + absorption, soft label term), ICP moved test (B2, both modes); `update_graph` + relation eval | **Passed** (`lsg m5` → `outputs/m5_final2`): association strict on dev B0 → B2 oracle 90.9 → 90.9%, perception 6.8 → 6.8% (≥ holds; ties); moved F1 on dev B1 → B2 oracle 33.3 → 86.7%, perception 45.5 → 60.9%; 0 invalid graphs (`validate`, 3 methods × 2 modes, dev + val); relation table on val (parent acc., R0 / R1 / update): (a) oracle 92.0 / 79.5 / 92.4%, perception 94.6 / 80.2 / 90.7%; (b) oracle 94.0 / 65.5 / 86.3%; (c) swapped n = 3: 0 / 2 / 2. Val was run twice (nyu40 catch-all fix, DECISIONS M5): B2 oracle moved F1 54.9%, new R 60.2%; perception unchanged strict 57.9% (B1 2.2%) |
| M6 ✅ | done 9 Nov | confidence + priors (FULL); texture rendering + appearance residual (DINOv2 patch features; thin-node free-like votes; thin-object voxels) | **Passed** (`lsg m6` → `outputs/m6_final`): **texture render check** on both dev references' own sequences, 3 overlays each visibly aligned, median DINOv2 patch similarity 0.766 / 0.720, OBJ = PLY on 72 / 72 references; **synthetic** 2 cm striped book removed from a table: depth-only `unchanged`, depth + appearance `removed` (colour features and DINOv2). Val FULL (reported once): false deletion oracle 8.1% / perception 7.7% (B2 9.5 / 9.5), unchanged kept 86.6 / 95.4%, removed R 26.0 / 31.5% (B2 41.1 / 45.2: confidence veto, DECISIONS M6) |
| M7 ✅ | done 18 Nov | seen votes need contrast (FULL); immediate commitment + children-follow-parent ablations; all ablations (§8) on val + relchange, both modes (`lsg m7`) | **Passed** (`lsg m7 --run-id m7_final`, git 565fc44, 111 min): `results.md` with oracle + perception rows and 95% CIs; 3 plots (fraction, τ_free, ablations); determinism on one pair 0 differences; dev regressions B0 / B1 / B2 / FULL-contrast 0 differences; 0 invalid graphs; val 110 pairs. FULL removed R 27.4 / 32.9% (contrast off 26.0 / 31.5; B2 41.1 / 45.2), false deletion 8.1 / 7.7%; immediate commitment: oracle removed P 55.6 → 19.6%, moved F1 54.6 → 34.5% |
| M8 | 26–27 Nov | Rerun timeline of snapshots + top-down before/after, audit (§7.4), README, write-up, slides | `README` "how to run" reproduces one pair end-to-end (both modes) on a clean `uv sync`; AI_USAGE.md complete |

**Buffer and cut order.** There is no slack inside 9–27 Nov. The 3–4 days gained by finishing M0 early are the buffer. If a milestone slips, cut in this order:
1. optional DINOv2 embedding association (old B3);
2. thin-object *candidates* from appearance (§6.6 output ii), keeping thin-node removal evidence (output i);
3. relchange set (tier-2 download) → relations evaluated on val only (n = 12, reported as anecdotal);
4. "children follow parent" ablation;
5. incremental snapshots (keep only the final graph).

The depth-only vs depth + appearance ablation is required by the brief and is not on the cut list.

## 10. Out of scope (unless time remains after M7)

- a real full-frame detector (Grounding DINO + SAM 2 over whole frames). Perception candidates come from residuals; Grounding DINO is at most an optional alternative *labeller* for candidate crops;
- estimating poses or alignment beyond the ICP refinement of `T_align`;
- rotation-only moved detection;
- articulation changes such as door swings (12 moved doors in val; doors are structure, so they are not scored; DECISIONS 2026-11-06);
- multi-visit chains (ref → rescan₁ → rescan₂ …);
- multi-room hierarchy / `place` level;
- LLM queries;
- learning anything beyond the class-prior table (CLIP and DINOv2 are used frozen, zero-shot).

## 11. Top risks

1. **Tiny positive counts**: 96 removed and 12 parent-changed in val. Mitigation: relchange set, CIs, support counts in every table.
2. **Incomplete or noisy GT**: unlisted removals and moves (§2.1 example), noisy "new". Mitigation: false-deletion rate, audit (§7.4), recall emphasised. Perception will be penalised for finding real but unannotated changes; the audit quantifies this.
3. ~~Transform semantics~~ — resolved at M0.
4. **Misalignment → fake `free` votes and fake candidates**. Mitigation: ICP refinement (M0), τ_free / τ_new, ≥ 2 free votes, ≥ 3-view voxels, decorrelation, per-pair alignment residual reported.
5. **Perception candidate quality**: people walking through, mesh holes (p1 of obs − exp ≈ −0.30 m on an unchanged scene), reflective or transparent surfaces, objects smaller than a few voxels, small moves where the old and new footprints overlap. Mitigation: multi-view voxel filter, surface/structure filters, size breakdown and displacement buckets that make the failures visible. The oracle row bounds the loss.
6. **Zero-shot labels on small, blurry, rotated crops**: CLIP may mislabel. Mitigation: upright crops, K = 5 crops, top-k with a soft label term instead of a hard gate, reference-graph labels in the vocabulary.
7. **Texture rendering**: OBJ/PLY correspondence unverified; texture seams and exposure differences between visits. Mitigation: check at M6, exposure-robust features, compare only where depth agrees, ≥ 3 agreeing views.
8. **Compute**: GTX 1650 with 4 GB. CLIP ViT-B/32 and DINOv2 ViT-S/14 fit, but per-frame patch features for 110 pairs take hours. Mitigation: features only on the processed (strided) frames, a feature cache in `cache/`, appearance only in FULL and its ablation.
9. **Schedule**: v2 adds perception, online processing and appearance to a fixed 27 Nov end. Mitigation: cut order (§9); M0's early finish is the buffer.
10. **Download time and disk**: tier 1 done; tier 2 done (4.0 GB, 16 min); textures done (1.1 GB).

## 12. Open questions for the organisers (non-blocking)

- Is "perception headline + oracle upper bound" the framing they expect? (v2 assumes yes.)
- Exact `bbox_3d` convention in the shared format (AABB vs. OBB).
