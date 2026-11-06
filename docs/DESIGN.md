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
| OBJ vs. PLY | Same vertex and face counts on 5 checked scans; **face order and vertex order not verified** | **check at M6** (texture rendering) |
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
| `eval_val.txt` | all 110 validation pairs (47 refs) with 3DSSG relations | main evaluation, change types |
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
  - **Thin node**: a thing whose smallest AABB extent is < 0.10 m (e.g. a book, a towel). Depth alone cannot tell whether it is still there.
- **Pixel classes** for frame f. Render the full reference mesh from the aligned rescan pose. With d_obs = observed depth and d_exp = rendered depth:
  - `invalid`: d_obs = 0 or no hit;
  - `present`: |d_obs − d_exp| ≤ τ_near = 0.10 m *(tune)*;
  - `free` (seen through): d_obs > d_exp + τ_free = 0.15 m *(tune; same-scan false rate 0.4% of pixels)*;
  - `occluded` / `nearer`: d_obs < d_exp − τ_near. On a node's pixels this is occlusion (no information). Anywhere, it is residual evidence for something new in front (§ Candidate below, with τ_new).
- **Frame vote** for node i. P_i,f = pixels where i is the first hit. A vote is cast only if |P_i,f| ≥ 15 px *(tune)*:
  - `seen` if present ≥ 50% of valid pixels;
  - `free` if free ≥ 50% of valid pixels;
  - `no-info` otherwise;
  - `unobserved` if |P_i,f| < 15 px (out of frustum or hidden).
- **Viewpoint decorrelation.** A vote (or a voxel hit, below) **counts** only if the camera centre moved ≥ d_dec = 0.30 m, or the viewing direction rotated ≥ θ_dec = 20° *(tune)*, since the last *counted* vote of the same kind for the same node (or voxel). Repeating one viewpoint therefore adds one vote. The per-visit log-odds clip (§6.10) stays as a second guard.
- **Deferred commitment.** During the walk, states are **tentative** (visible in snapshots). `removed` and `moved` become final only at the end of the sequence (or chunk). Reason: a moved object can appear at its new place hundreds of frames after its old place was seen empty.
- **Confirmed in place**: a reference node with ≥ k_seen = 3 *(tune)* counted `seen` votes and **no** counted `free` votes. It is `unchanged` without needing a matched instance, and it does not enter association.
- **Candidate** (perception mode). Built in five steps:
  1. **Nearer pixels:** pixels with valid depth and d_obs < d_exp − τ_new (0.10 m *(tune)*).
  2. **Back-project and filter.** Back-project them to the reference frame. Drop points within 5 cm of the reference mesh surface. Drop points near structure surfaces (wall, ceiling, floor), except points standing on the floor.
  3. **Voxel grid:** accumulate the remaining points in a 4 cm *(tune)* grid.
  4. **Multi-view filter:** keep a voxel only if it was hit from ≥ 3 decorrelated viewpoints. This filters people, sensor noise and mesh holes.
  5. **Cluster:** at the end, cluster the kept voxels (DBSCAN or connected components *(tune)*), and drop clusters below a minimum size *(tune)*.

  Each remaining cluster is a candidate. Appearance mismatches off reference nodes, on supporting surfaces, add thin-object voxels (§6.6).
- **Appearance residual**: the reference texture is rendered at the rescan pose and compared with the observed colour, only where depth agrees (`present`).
  - Comparison uses exposure-robust features: DINOv2 patch-feature cosine distance (default), or mean/std-normalised colour (fallback).
  - On a **thin node's** pixels, a mismatch is a `free-like` vote.
  - Appearance evidence alone never deletes a node: it needs ≥ 3 decorrelated viewpoints that agree.
- **Unknown**: a reference node that is not confirmed in place, has no matched instance, and has fewer than k_free = 2 *(tune)* counted free (or free-like) votes. It is **kept**, its confidence is unchanged, and its state is `unknown`.
- **Moved** (prediction): matched, and the ICP-refined centroid displacement is > τ_move = 0.10 m *(tune at M1 from the GT displacement distribution)*. Final at the end of the sequence. Rotation-only moves are out of scope; recall is reported per displacement bucket.
- **Removed** (prediction): unmatched, with ≥ k_free counted free or free-like votes, and confidence p < p_del = 0.2 *(tune)*. Final at the end of the sequence.
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
- **Snapshots** are written every N frames *(config)* to `outputs/<run_id>/snapshots/<ref>_<rescan>/frame_XXXXXX.json`. They use the same schema with `meta.frame_idx` set and `meta.tentative: true`.

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

`src/lsg/` — every function is pure where possible and works in the reference frame. ✅ = exists (M0–M2).

1. `io` ✅
   - `scan3r.load_meta(root) -> Meta` (3RScan.json, column-major transforms) ✅;
   - `scan3r.load_scan(root, id) -> Scan` (vertices, faces, vertex objectId, per-face label by majority, instances {id: label, nyu40, centroid, aabb}, intrinsics) ✅. Annotation reader: on rescans only from `oracle/` and `eval/`;
   - `scan3r.open_sequence(root, id) -> SequenceReader` and `iter_frames(seq, stride, idx=None) -> Frame` (idx, pose_c2w, depth[m]; colour on demand) ✅;
   - `scan3r.load_textured_mesh(root, id) -> TexturedMesh` (OBJ vertices, faces, per-face-corner UVs, texture image; **reference scans only**) — M6;
   - `dssg.load_relations / load_objects / load_predicates` ✅;
   - `splits.write_texture_list(split_files, out)` → `splits/download_textures.txt` (CLI `lsg splits textures`) ✅.
2. `geom` ✅
   - `render.Renderer(vertices, faces, face_labels)`, `.render(K, T_w2c, W, H) -> (depth[m], label_map)`, `.distance(points)` ✅. Open3D RaycastingScene on CPU; label map via `primitive_ids`.
   - `render.TextureRenderer(textured_mesh)`, `.render_color(K, T_w2c, W, H) -> rgb`: `primitive_ids` + `primitive_uvs` → interpolate per-face UVs → sample `mesh.refined_0.png`. Uses OBJ geometry — M6.
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
   - Implementations: `oracle.OracleSource` ✅, `perception.PerceptionSource` (M4). The change logic depends only on this interface.
   - Also `RefContext` (reference scan, G_ref, floor_z) and `SeqInfo` (rescan id, intrinsics, **refined** `T_align`, processed frame ids). `ref_render` is None until M3.
5. `oracle` (upper bound) ✅ (M2)
   - `OracleSource`: renders the rescan's own labelled mesh at each frame, collecting per-instance masks, points (vertices of hit triangles, *these frames only*, mapped with the refined `T_align`) and crops (optional, off by default). A pixel counts only where the sensor depth is valid. IDs are shuffled with a seeded permutation; `token_map()` is read only by `eval/`.
   - It is the only reader of rescan labels outside `eval/`.
   - Instances with < 50 observed points *(tune)* are dropped.
6. `perception` (headline). **Must never import `oracle/` or `eval/`, and never read rescan semseg, rescan PLY labels, rescan 3DSSG or the `rigid/removed/nonrigid` lists.**
   - `residual.nearer_points(frame, d_exp, ref_ctx, cfg) -> points`: §3 candidate steps 1–2 (τ_new, surface and structure filters).
   - `voxels.VoxelEvidence(voxel=0.04)`: `.add(points, cam_pose)` counts decorrelated viewpoints per voxel; `.kept(min_views=3)`.
   - `cluster.cluster(voxels, cfg) -> list[Candidate]`: DBSCAN (scikit-learn) or connected components, min size *(tune)*.
   - `appearance`:
     - `.mismatch(rgb_obs, rgb_exp, present_mask, cfg) -> mismatch map`;
     - features: DINOv2 ViT-S/14 patch features (default; upright for the network, mapped back to the stored orientation) or normalised colour (fallback);
     - output (i): free-like votes on thin-node pixels; output (ii): thin-object voxels off-node on supporting surfaces.
   - `labeler.CLIPLabeler(vocab, model="ViT-B/32")`:
     - `.topk(crops) -> label_topk` from the K = 5 largest projected upright crops;
     - vocabulary = reference-graph labels ∪ the most frequent 3RScan thing/anchor labels (train metadata, §2.3).
   - `source.PerceptionSource`: assembles the above into `ObsInstance`s (points, masks by projecting the cluster, crops, `label_topk`).
7. `change.pipeline` ✅ (M2) and `change.online` (M3)
   - `pipeline.select_frames` (stride + §7.3 chunk); `pipeline.prepare_pair` (G_ref + cached refined `T_align`; it gets only the raw `T_align` matrix, never the rescan's change lists); `pipeline.collect_observations` (the M2 frame loop, replaced by `run_online` at M3).
   - `baselines.b0` ✅ (§8; greedy one-to-one same-label matching).
   - `run_online(G_ref, ref_ctx, seq, obs_source, cfg, frame_ids) -> OnlineResult` (votes, obs instances, snapshots). Per processed frame, in time order:
     1. render the reference (depth, labels, and colour if appearance is on);
     2. compute frame votes (`change.visibility`);
     3. apply decorrelation and update confidence (`change.confidence`);
     4. call `obs_source.observe` (residual accumulation in perception mode);
     5. write a snapshot every N frames.
   - At the end it calls `obs_source.finish()`, then association, then classification. States become final only then.
8. `change.visibility`
   - `frame_votes(nodes, render, frame, cfg) -> dict[node_id, Vote]` (one frame, §3 rules; `unobserved` included).
   - `Decorrelator(d_dec, θ_dec)`: per key (node, vote kind) or voxel, the pose of the last counted event; `.counts(key, cam_pose) -> bool`.
9. `change.association`
   - `associate(ref_nodes, obs_instances, cfg, mode) -> (pairs, unmatched_ref, unmatched_obs)`.
   - Only reference nodes **not confirmed in place** take part.
   - Cost:
     - label term: oracle = hard gate (+∞ if labels differ; fallback same nyu40 + 0.3 penalty); perception = soft, −log max(p(label_ref), p_min) from `label_topk` *(p_min tune)*;
     - plus w_g · size term (mean |log| ratio of sorted AABB extents) + w_p · min(dist / 3 m, 1) (position, weak);
     - optional + w_a · (1 − cos(DINOv2 embedding)) — the old B3, **off by default**.
   - Hungarian assignment, gate cost < τ_assoc *(tune)*.
10. `change.classify`
    - `classify(pairs, unmatched_ref, votes, confirmed, cfg) -> dict[node_id, State]`, run at the end of the sequence.
    - ICP (point-to-point, init = centroid offset) gives T_move and the moved/unchanged decision.
    - Unmatched reference node → `removed` if it has free evidence (§3), else `unknown`. Unmatched observation → `new`.
11. `change.confidence`
    - Log-odds per node, starting at l₀ = logit(p_persist(class)) from train priors (fallback p = 0.9).
    - Updated online from **counted** (decorrelated) votes: Δl = L_hit · n_seen − L_miss · (n_free + n_free_like), clipped per visit to [−3, +3]. L_hit = L_miss = 0.4 *(tune)*.
    - A matched node gets +L_hit per counted seen vote at its *new* location. Free votes at the old location of a matched node are **ignored**, which is applied at the end because association is deferred.
    - Delete rule: §3 `removed`.
12. `change.update`
    - `update_graph(G_ref, states, obs, cfg) -> G_upd`.
    - Moves nodes (apply T_move to the centroid and bbox), recomputes `infer_support` for moved and new things **and** for things whose parent moved or was removed, adds new nodes (with `label_topk`), sets `last_seen`, `confidence`, `state`.
    - Optional rule (ablation): an unobserved thing whose parent anchor moved moves with it.
13. `eval`
    - `gt.gt_changes(scan_meta, R, S, objs_R, objs_S, cfg) -> GTLabels` (§5) ✅; `gt.gt_parent_changes` ✅; `gt_stats.run_m1` (`lsg m1`) ✅;
    - `metrics.match_new(pred_new, gt, mode, cfg)` (§5 matching rules) ✅;
    - `metrics.score_pair(G_upd, gt, mode, cfg, obs_truth)` → per-pair counts, and `metrics.aggregate` (micro-average + bootstrap CI), including localisation error and size buckets ✅ (the spec's `score_changes`);
    - `harness.run_eval` / `run_split` (`lsg eval`, `lsg run`) ✅;
    - `score_relations(G_upd, gt_rel) -> dict`;
    - `audit_sample(...)` writes image crops for manual FP review.
14. `viz`
    - `rerun_timeline(snapshots, meshes)`: snapshots on a frame timeline, nodes coloured by (tentative) state;
    - `topdown_png(G_ref, G_upd)`: static before/after figure for slides/report.

CLI:
- `uv run lsg run --pair R S --obs perception|oracle --method full --frac 1.0 --seed 0` ✅ (oracle, B0);
- `uv run lsg eval --split eval_val --obs perception,oracle --method … [--workers N]` ✅ (oracle, B0);
- `uv run lsg ablate …`;
- `uv run lsg splits textures` ✅;
- `uv run lsg graph --scan R --mode gt|geometric [--out f.json]` ✅;
- `uv run lsg m0` ✅, `uv run lsg m1` ✅.

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

- **Parent accuracy** (exact parent node; `floor` counts) and **type accuracy** (on/in | correct parent). Reported on:
  - (a) all matched things with a GT parent;
  - (b) moved things;
  - (c) things whose GT parent changed: val (n = 12) + relchange set (n = 73).
  - Each also reported **with and without floor parents**.
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
| FULL | B2 | §6.11 log-odds + class priors; depth **+ appearance residual** | yes |

Ablations:
- **depth-only vs depth + appearance residual** (the brief's ablation; FULL with appearance off vs on);
- decorrelation on / off;
- deferred vs immediate commitment;
- oracle vs perception (every table);
- observed fraction (§7.3);
- class priors on / off;
- "children follow parent" on / off;
- τ_free sweep (0.05–0.30 m);
- *optional:* + DINOv2 embedding term in association (old B3).

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
| M3 | 13 + 16 Nov | online loop (`change.online`, `visibility`, `Decorrelator`), deferred commitment, snapshots; B1 (oracle) | **Decorrelation unit test**: the same viewpoint repeated 30× gives 1 counted vote. **Synthetic scene**: box removed → free votes → `removed` at end; box occluded / out of frustum → `unknown`. On dev, B1's false-deletion rate < B0's. **Determinism**: two runs → identical `metrics` |
| M4 | 17–18 Nov | perception end-to-end: residual points, `VoxelEvidence`, clustering, CLIP labelling, `PerceptionSource`; B0 + B1 in perception mode | **Synthetic**: box added → exactly one candidate, centroid error < 5 cm. Overlay PNGs of candidate clusters on 3 dev frames. Candidate recall vs GT new / moved-to objects on dev reported (no threshold yet). No-leak test passes on the real `perception/` code |
| M5 | 19–20 Nov | association changes (confirmed-in-place skip, soft label term), ICP moved test (B2, both modes); `update_graph` + relation eval | Association accuracy ≥ B0 on dev (both modes); moved F1 ≥ B1 on dev (both modes); relation table (a/b/c) for R0 vs R1 on val; JSON validates against the schema |
| M6 | 23–24 Nov | confidence + priors (FULL); texture rendering + appearance residual (DINOv2 patch features; thin-node free-like votes; thin-object voxels) | **Texture render check**: on a reference scan's *own* sequence, rendered vs observed colour overlay for 3 frames visibly aligned; median patch-feature similarity reported; OBJ/PLY correspondence checked (§2.1). **Synthetic**: a 2 cm flat textured "book" removed from a table → missed by depth-only, caught by depth + appearance |
| M7 | 25 Nov | all ablations (§8) on val + relchange, both modes | Full results table (oracle + perception rows) with CIs; ablation plots; determinism re-checked on one pair |
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
