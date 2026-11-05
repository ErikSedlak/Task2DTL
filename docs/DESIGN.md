# DESIGN — Living Scene Graphs, Subproject D: change detection and graph maintenance

Status: v1 (2026-11-05), written before M0; §2.1, §3, §5, §6.2 and §11.4 updated with the M0 results (see DECISIONS, M0). Numbers marked *(tune)* are starting values. They are fixed on `splits/dev_train.txt`, and every change is logged in DECISIONS.md.

---

## 1. Problem and framing

**Given:**
- the scene graph G_ref of a reference scan (from GT `semseg.v2.json` and 3DSSG),
- the reference mesh,
- the RGB-D frames and poses of a rescan of the same room (poses aligned to the reference frame with 3RScan's `T_align`).

**Produce:**
- a per-node change state: `unchanged | moved | removed | unknown`;
- a list of `new` nodes;
- the updated graph G_upd: moved nodes with recomputed support parent and relation, deleted nodes, added nodes, `confidence`, and `last_seen`.

**Contribution.** Object-level *semantic graph maintenance*:
- node identity across visits,
- relation updates,
- a confidence/deletion policy that distinguishes "gone" from "not seen".

This project does **not** compete with class-agnostic geometric change detection (Objects Can Move, Has Anything Changed?).

**Headline setting = oracle instances.** The rescan's objects come from the rescan's GT instance segmentation: masks, class labels, and points visible in the used frames, with IDs shuffled. This isolates the change logic. Every result is labelled `oracle` in tables and plots. A real detector is a stretch goal (§10).

## 2. Data

### 2.1 Facts verified on local data (2026-11-05)

| Item | Value | Status |
|---|---|---|
| Color / depth resolution | 960×540 / 224×172, same FOV; separate intrinsics in `_info.txt` | verified |
| Depth encoding | PGM P5, uint16 mm, 0 = invalid, `m_depthShift = 1000` | verified |
| Pose | camera→world in that scan's mesh frame. Back-projected depth lies 0.9–1.5 cm (median) from the mesh; the inverse convention gives 0.5–2.6 m | verified |
| Up axis | +Z (floor normal ≈ ±z on 4 scans); floor z ≈ −1.4…−1.5 | verified |
| Image orientation | stored rotated 90° (portrait capture) | verified visually |
| Depth noise vs. own mesh | |d_obs − d_render| median 1.4–2.1 cm, p90 5–9 cm. Signed p99 = +8 cm (obs farther), p1 = −30 cm (obs nearer) | 3 frames, 1 scan |
| Open3D `t_hit` with intrinsic rays | equals camera-z depth (max diff 0.0) | verified |
| Render speed (CPU) | ~5 ms per 224×172 frame, ~50 ms at 960×540 | verified (cloud CPU) |
| PLY | ASCII; per-vertex `objectId, globalId, NYU40, Eigen13, RIO27`; 99.5% of faces have one label | verified |
| 3RScan.json transforms | 16 floats, column-major, orthonormal R, **metres** (not the mm the FAQ states). `T_align` maps rescan → reference: rescan structure lies 1.6 / 1.9 / 3.0 cm (median, point-to-surface) from reference structure on the 3 dev rescans; T⁻¹ gives 12–204 cm, translation ×0.001 gives 6–46 cm | verified M0 |
| `T_align` accuracy | Aligned-reference depth residual median 2.9–3.8 cm vs. 1.1–1.3 cm for the rescan's own mesh. Label-free ICP of rescan depth onto the reference mesh corrects 1.6–8.4 cm / 0.4–1.5° → 1.9–2.2 cm. The method uses the **refined** alignment (`geom.align`); GT/eval uses raw 3RScan transforms | verified M0 (3 dev rescans) |
| Depth valid area | Stored depth columns 0–~20 and ~205–223 are never valid (sensor border, ≈ 18% of pixels); depth is otherwise ~85% valid | verified M0 |
| `rigid` entries | 2,933 dicts + 390 bare ints (no transform); `instance_reference == instance_rescan` always | verified |
| Rigid-transform semantics | `T_rigid` maps reference-frame instance points **into the rescan frame**: p_S = T_rigid · p_R. Best of 5 hypotheses on 20/21 dev objects (Chamfer 2.8–7.8 cm, 17 < 5 cm; the 21st is a toilet that barely moved). Motion in the reference frame: **M_ref = T_align · T_rigid** | verified M0 |
| Instance IDs | consistent within an environment (same objectId = same object) | verified via 3DSSG `objects.json` |
| 3DSSG | 1,482 scans in `objects.json`, 1,335 in `relationships.json` (train+val refs and rescans; test rescans have none). Objects carry `nyu40/eigen13/rio27`. 41 predicates with `none` = id 0; `inside` never occurs; `classes.txt` has 528 lines | verified |
| Ref objects absent from rescan semseg (val) | 1,018, vs. only 96 listed `removed` → absence in the rescan annotation ≠ removal (partial rescans) | verified |

### 2.2 Change ground-truth statistics (validation split, 110 rescans of 47 references)

- rigid (moved): 390;
- removed (listed): 96;
- nonrigid: 54;
- new (derived, rescan IDs not in reference): 224;
- moved *things*: 278, of which only **12 change support parent**;
- train split: 174 parent-changed things over 793 rescans.

Most-moved classes: chair 80, pillow 31, armchair 23, table 22, ottoman 20, coffee table 17.

→ "Cup moves from table to shelf" is **rare** in 3RScan. Most changes are furniture rearrangement. The relation-change evaluation therefore needs an extra set (§2.3).

### 2.3 Subset (files in `splits/`)

| Split file | Content | Use |
|---|---|---|
| `dev_train.txt` | 3 train pairs (531cff08→19eda6f4, 531cff08→e2847ff7, 0ad2d39b→0ad2d3a1); 2 of their scans are already local | development + **all threshold tuning** |
| `eval_val.txt` | all 110 validation pairs (47 refs) with 3DSSG relations | main evaluation, change types |
| `eval_relchange_train.txt` | 40 train pairs with the most parent-changed moved things (73 events) | relation-update evaluation only; never tuned on |
| `download_tier1.txt` | 162 scan IDs (dev + val) | download first |
| `download_tier2.txt` | 63 more scan IDs (relchange set) | download in week 1–2 |

**Class priors (§6.7)** are estimated from train-split metadata (3RScan.json + 3DSSG objects) **excluding** references in `eval_relchange_train.txt` and `dev_train.txt`. This needs no downloads.

**Download** (only three file types; ≈50 MB/scan → tier 1 ≈ 8 GB, tier 2 ≈ 3 GB):

```bash
while read id; do for t in semseg.v2.json labels.instances.annotated.v2.ply sequence.zip; do
  python download_files.py -o data --id "$id" --type "$t"; done; done < splits/download_tier1.txt
```

`mesh.refined.v2.obj` + `mesh.refined_0.png` are needed only for 1–2 textured visualisation scenes. `train-/val-scans.tfrecords` (2.5 GB) are not used.

## 3. Definitions

- **Reference frame**: the reference scan's mesh frame (z-up, metres). All rescan geometry and poses are mapped into it via `T_align`, refined by label-free ICP on the method side (§2.1, `geom.align`).
- **Frame subsampling**: stride `s = 3` *(tune)*. Frames move only ~6 cm / 5° per step, so consecutive frames are strongly correlated.
- **Node classes**: decided from 3DSSG `nyu40` id.
  - `anchor`: nyu40 ∈ {3 cabinet, 4 bed, 6 sofa, 7 table, 10 bookshelf, 12 counter, 14 desk, 15 shelves, 17 dresser, 32 nightstand}.
  - `structure`: nyu40 ∈ {1 wall, 2 floor, 8 door, 9 window, 11 picture, 13 blinds, 16 curtain, 19 mirror, 20 floor mat, 22 ceiling, 28 shower curtain, 30 whiteboard, 38 otherstructure}.
  - `thing`: everything else.
  - Structure is rendered as geometry but is **not** a graph node, except one `room` node (and `floor` as a pseudo-parent).
- **Visibility test** for node i in frame f: render the full reference mesh from the aligned rescan pose. P_i,f = pixels where i is the first hit. Per pixel, with d_obs = observed depth and d_exp = rendered depth:
  - `invalid`: d_obs = 0 or no hit;
  - `present`: |d_obs − d_exp| ≤ τ_near = 0.10 m *(tune)*;
  - `absent` (free space): d_obs > d_exp + τ_free = 0.15 m *(tune; same-scan false rate 0.4% of pixels)*;
  - `occluded`: d_obs < d_exp − τ_near.
- **Frame vote** (only if |P_i,f| ≥ 15 px *(tune)*):
  - `seen` if present ≥ 50% of valid pixels;
  - `free` if absent ≥ 50% of valid pixels;
  - otherwise `no-info`.
- **Unknown**: a reference node with no matched rescan instance and fewer than k_free = 2 *(tune)* `free` votes. It is **kept**, its confidence is unchanged, and its state is `unknown`.
- **Moved** (prediction): matched, and the ICP-refined centroid displacement is > τ_move = 0.10 m *(tune at M1 from the GT displacement distribution)*. Rotation-only moves are out of v1; recall is reported per displacement bucket.
- **Removed** (prediction): unmatched, with ≥ k_free `free` votes, and confidence p < p_del = 0.2 *(tune)*.
- **New**: a rescan instance not assigned to any reference node by the association.

## 4. Graph JSON (shared format + the fields this project adds)

```json
{"meta": {"scan_ref": "...", "scan_obs": "...|null", "frame": "ref", "units": "m", "version": 1},
 "nodes": [{"id": "r12", "level": "thing", "label": "pillow", "nyu40": 18,
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
- `obs_token` = the shuffled oracle instance id. It is written only for evaluation mapping and never read by the method.

## 5. Ground truth (only in `src/lsg/eval/`)

Per pair (ref R, rescan S). O_R and O_S are the objectIds in their semseg files.

- **moved**: IDs in S's `rigid` list (dict or int form).
  - GT displacement = ‖T_align · T_rigid · c_R − c_R‖, where c_R is the reference centroid (raw 3RScan `T_align`). `T_rigid` alone maps into the rescan frame (§2.1, confirmed at M0), so M_ref = T_align · T_rigid is the motion in the reference frame.
  - GT moved with displacement < τ_eval_move = 0.10 m *(set at M1)* → `dont_care` for the moved/unchanged decision.
- **removed**: IDs in `removed`. If a listed removed ID still appears in O_S (20 cases in val) → `dont_care`.
- **nonrigid**: `dont_care` for change-type metrics (either `unchanged` or `moved` is accepted). Excluded from relation metrics.
- **new**: O_S \ O_R, things and anchors only (structure dropped). Known to be noisy (Objects Can Move reports inconsistent "new" annotation) → audited (§7.4).
- **unchanged**: (O_R ∩ O_S) minus the above.
- **absent-unlisted**: O_R \ O_S that are not listed removed. Excluded from removed P/R. Used for the **false-deletion rate** (§7.1).
- Only `thing` + `anchor` nodes are scored.
- **Relations GT** from rescan 3DSSG, child = subject. Only things are scored:
  - `on` ⇐ {standing on, lying on, supported by, hanging on};
  - `in` ⇐ {standing in, lying in, hanging in, inside, build in};
  - parent `floor` (structure "floor") is kept as a valid parent;
  - parent wall/ceiling/window/other structure → the thing is excluded;
  - `close by` → `next_to` (reported only);
  - everything else is ignored: attached to, left/right/front/behind, comparatives, same-*, part of, connected to, belonging to, cover, leaning against;
  - multiple GT parents → any one counts as correct (1 case in val).

## 6. Modules and interfaces (inputs → outputs; no implementation here)

`src/lsg/` — every function is pure where possible and works in the reference frame.

1. `io.scan3r`
   - `load_meta(root) -> Meta` (3RScan.json, column-major transforms);
   - `load_scan(root, id) -> Scan` (vertices, faces, vertex objectId, per-face label by majority, instances {id: label, nyu40, centroid, aabb}, intrinsics);
   - `iter_frames(scan, stride, idx=None) -> Frame` (idx, pose_c2w, depth[m], color path).
   - `io.dssg`: `load_relations(root, scan_id) -> list[(child, parent, name)]`, `load_objects(...)`.
2. `geom.render`
   - `Renderer(vertices, faces, face_labels)`;
   - `.render(K, T_w2c, W, H) -> (depth[m], label_map)`.
   - Open3D RaycastingScene on CPU; label map via `primitive_ids`.
   - `geom.align.refine_from_sequence(seq, ref_V, ref_F, T_align, cfg) -> (T_refined, info)`: robust point-to-plane ICP of rescan depth points onto the reference mesh (no rescan labels); corrections beyond `max_delta_t` / `max_delta_rot_deg` are rejected.
3. `graph.build`
   - `build_reference_graph(scan, relations=None, mode="gt"|"geometric") -> Graph`.
   - `graph.relations`: `infer_support(node, candidates, floor_z, cfg) -> (parent_id|"floor"|None, "on"|"in", score)`. Rules (all *(tune)* on reference scans of dev):
     - contact: ≥ 30% of the child's bottom-slice points (lowest 3 cm) have a candidate point within 5 cm horizontally and 0–10 cm below → `on`;
     - containment: ≥ 60% of child points inside the candidate's AABB, and the candidate is an anchor → `in`;
     - best score wins, anchor beats thing on ties;
     - otherwise `floor` if the child's bottom is < 10 cm above floor_z.
4. `oracle.rescan_instances`
   - `observe_instances(S, frames, perm_seed) -> list[ObsInstance]` (token, label, nyu40, points seen in *these frames only* = vertices of hit triangles, per-frame masks at depth resolution, best color crops).
   - It is the only reader of rescan labels outside `eval/`.
   - Instances with < 50 observed points *(tune)* are dropped.
5. `change.visibility`
   - `frame_votes(G_ref_nodes, ref_renderer, frames) -> dict[node_id, list[Vote]]` (§3 rules).
6. `change.association`
   - `associate(ref_nodes, obs_instances, cfg, use_appearance) -> (pairs, unmatched_ref, unmatched_obs)`.
   - Cost:
     - +∞ if the labels differ (fallback: same nyu40 + 0.3 penalty);
     - otherwise w_g · size term (mean |log| ratio of sorted AABB extents) + w_p · min(dist / 3 m, 1) (position, weak) + w_a · (1 − cos(emb)).
   - Hungarian assignment, gate cost < τ_assoc *(tune)*.
7. `change.classify`
   - `classify(pairs, unmatched_ref, votes, cfg) -> dict[node_id, State]`.
   - ICP (point-to-point, init = centroid offset) gives T_move and the moved/unchanged decision.
8. `change.confidence`
   - Log-odds per node, starting at l₀ = logit(p_persist(class)) from train priors (fallback p = 0.9).
   - Per visit: Δl = L_hit · n_seen − L_miss · n_free, clipped to [−3, +3] (correlated frames). L_hit = L_miss = 0.4 *(tune)*.
   - A matched node gets +L_hit per seen frame at its *new* location. Free votes at the old location of a matched node are **ignored**.
   - Delete rule: §3 `removed`.
9. `change.update`
   - `update_graph(G_ref, states, obs, cfg) -> G_upd`.
   - Moves nodes (apply T_move to the centroid and bbox), recomputes `infer_support` for moved and new things **and** for things whose parent moved or was removed, adds new nodes, sets `last_seen`, `confidence`, `state`.
   - Optional rule (ablation): an unobserved thing whose parent anchor moved moves with it.
10. `eval`
    - `gt_changes(meta, R, S) -> GTLabels` (§5);
    - `score_changes(states, new_tokens, gt) -> dict`;
    - `score_relations(G_upd, gt_rel) -> dict`;
    - `audit_sample(...)` writes image crops for manual FP review.
11. `viz`
    - `rerun_pair(G_ref, G_upd, meshes)`: before/after nodes and edges coloured by state;
    - `topdown_png(G_ref, G_upd)`: static figure for slides/report.

CLI: `uv run lsg run --pair R S --method full --frac 1.0 --seed 0`, `uv run lsg eval --split eval_val --method …`, `uv run lsg ablate …`.

## 7. Evaluation protocol

### 7.1 Change detection (per pair, then micro-averaged over the split)

- Precision / recall / F1 for **moved**, **removed** and **new**, plus accuracy on **unchanged**.
  - Ground truth and predictions are compared by object ID. The oracle mapping recovers rescan IDs.
  - A prediction of `unknown` counts as *not removed*.
  - `dont_care` objects are excluded.
- **False-deletion rate**: share of `absent-unlisted` reference objects that the method deletes. Lower is better; the expected baseline failure is that B0 deletes many of them. Report it next to removed P/R, because GT precision is unreliable.
- **Unknown rate**: share of reference nodes left `unknown`.
- **Association accuracy**: correct ref→obs pairs / GT-matchable pairs. Report both strict and *same-label tolerant* (swaps between identical-label objects not penalised; chairs dominate).
- Moved recall broken down by GT displacement buckets: [0.1, 0.3), [0.3, 1), ≥ 1 m.
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
- Sanity: rule accuracy on **reference scans** against reference 3DSSG (no change involved). This must reach a reasonable level before M5 counts.

### 7.3 Observation-fraction ablation

- 100 / 50 / 25 / 10% of frames as **one contiguous chunk**, start chosen with 3 seeds, mean ± sd.
- The oracle sees only instances visible in the chunk, and only their visible points.

### 7.4 Ground-truth audit

- Manually inspect 30 false positives (moved, removed and new) on val using the crops from `audit_sample`.
- Report the share that is actually a missing GT label.
- This is the honest answer to "the GT is not exhaustive".

## 8. Baselines and ablations (all oracle)

| ID | Association | Removal rule | Confidence |
|---|---|---|---|
| B0 naive | nearest same-label instance within 0.5 m → unchanged; same label farther → moved | unmatched → removed | none |
| B1 + visibility | B0 matching | unmatched → removed only with ≥ k_free free votes, else unknown | none |
| B2 + Hungarian geometry | §6.6 without appearance | as B1 | none |
| B3 + appearance | §6.6 with DINOv2 | as B1 | none |
| FULL | best of B2/B3 | §6.8 log-odds + class priors | yes |

Ablations:
- geometry vs. geometry + appearance (B2 vs B3);
- observed fraction (§7.3);
- class priors on/off;
- "children follow parent" on/off;
- τ_free sweep (0.05–0.30 m).

The expected story: B0→B1 is the big win (false deletions on partial rescans). Appearance helps mainly for same-label duplicates and may be small; that is still a valid finding.

## 9. Milestones (dates assume 9–27 Nov)

| M | Dates | Deliverable | Acceptance check |
|---|---|---|---|
| M0 | 9–10 Nov | loaders, renderer, `T_align`, overlay script; finalise §2.1 | (1) Rendered vs. observed depth on a dev rescan: median |Δ| < 3 cm. (2) Aligned rescan mesh vs. reference mesh on structure classes: median NN distance < 5 cm, which settles the transform units. (3) Applying a `rigid` transform to the reference instance points gives Chamfer < 5 cm to the rescan instance on ≥ 2 dev objects, which settles the semantics. (4) Overlay PNG of the instance mask on color for 3 frames looks right. (5) `pytest` green. |
| M1 | 11 Nov | reference graph (GT and geometric relations), `gt_changes`, GT statistics | Counts on val reproduce §2.2 (390 / 96 / 224); GT displacement histogram saved → τ_eval_move fixed; reference-scan rule parent accuracy reported |
| M2 | 12–13 Nov | B0 + full eval harness + bootstrap | `lsg eval --method B0` on `eval_val` writes metrics.json in < 15 min; metric unit tests on a 5-object toy case |
| M3 | 16–17 Nov | visibility votes + B1 | Synthetic test (a cube removed / occluded / out of frustum) gives free / no-info / no-info. On dev, B1's false-deletion rate is lower than B0's |
| M4 | 18–19 Nov | Hungarian association + ICP moved test (B2) | Association accuracy ≥ B0 on dev; moved F1 ≥ B1 on dev |
| M5 | 20 Nov | `update_graph` + relation eval | Relation table (a/b/c) for R0 vs R1 on val and relchange; JSON validates against the schema |
| M6 | 23–24 Nov | confidence + priors (FULL), appearance (B3), all ablations | Full results table with CIs; ablation plots |
| M7 | 25–27 Nov | Rerun + top-down figures, audit, README, write-up, slides | `README` "how to run" reproduces one pair end-to-end on a clean `uv sync`; AI_USAGE.md complete |

Buffer: B3 (appearance) can be dropped without breaking anything downstream.

## 10. Out of scope (unless time remains after M6)

- real detector (Grounding DINO + SAM 2);
- estimating poses or alignment;
- rotation-only moved detection;
- multi-visit chains (ref → rescan₁ → rescan₂ …);
- multi-room hierarchy / `place` level;
- LLM queries;
- learning anything beyond the class-prior table.

## 11. Top risks

1. **Tiny positive counts**: 96 removed and 12 parent-changed in val. Mitigation: relchange set, CIs, support counts in every table.
2. **Incomplete or noisy GT**: unlisted removals, noisy "new". Mitigation: false-deletion rate, audit (§7.4), recall emphasised.
3. **Transform semantics** (units, direction of `rigid`) are not confirmed until M0. M0 is a hard gate.
4. **Misalignment → fake `free` votes**. Mitigation: τ_free, ≥ 2 votes, report per-pair alignment residual. ICP refinement of `T_align` adopted at M0 (raw `T_align` is off by up to 8 cm / 1.5° on dev).
5. **The oracle makes "new" and association easy**. Say so plainly in the write-up; the contribution is the maintenance logic.
6. **Download time and disk** (~11 GB). Start tier 1 before 9 Nov.

## 12. Open questions for the organisers (non-blocking)

- Is an oracle-instance headline acceptable, given the brief's own "given the graph of the reference scan"?
- Exact `bbox_3d` convention in the shared format (AABB vs. OBB).
