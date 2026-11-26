# Code structure — what is where and what it does

State: milestone M8 (2026-11-18). Covers everything in `src/lsg/`, `tests/` and `configs/`. This file explains *what* the code does; the report (`docs/report/`) explains the method and the results.

## 1. The project in one paragraph

3RScan contains rooms that were scanned several times. The first scan of a room is the **reference**; later visits are **rescans**. Between visits, furniture moves, objects disappear and new ones appear. The goal is to take the reference room's scene graph (objects + "cup is on the table" relations), look at a rescan's camera frames, and update the graph: which objects are unchanged, moved, removed, unknown (not seen) or new.

M0 built the foundation: reading the data, drawing ("rendering") a scanned room from any camera position, lining up the rescan with the reference, and checking on real data that all of this is correct. M1 added the reference scene graph (objects + support relations, from annotations or from geometry) and the ground truth of what changed between two visits. M2 added the observation interface (what the method sees of a rescan), the oracle version of it, the first (naive) change detector B0, and the scoring harness `lsg eval`. M3 added the online frame loop: every rescan frame is compared with the rendered reference, each object gets "seen" / "seen through" votes, votes from nearly the same camera position count once, and B1 deletes an object only if it was seen through from at least two different viewpoints. Otherwise the object becomes `unknown`. M4 added perception mode: the method finds rescan objects itself, from places where the measured depth is nearer than the rendered reference (something new stands there), keeps only spots seen from several viewpoints, groups them into candidate objects and names each one with CLIP from colour crops. M5 added B2: objects confirmed in place keep their place (and claim the observations lying inside them), the rest are paired with observations by an optimal assignment on label, size and position, and a paired object counts as moved only if aligning its reference shape onto the observation (ICP) moves it more than 10 cm. After that, "what stands on what" is recomputed for objects that moved, are new, or lost their support, and the relations are scored against the rescan's annotation. M6 added FULL: every object gets a confidence (how likely it still exists), starting from how often objects of its class disappear in the training rooms and moved by its "seen" / "seen through" votes; an object is deleted only if the confidence drops below 20%. It also added an appearance check: the reference room's photo texture is drawn from the rescan camera and compared with the real photo (DINOv2 features), which tells whether a flat object (book, towel, pillow) is still there where depth cannot, and finds new flat objects lying on tables or the floor. M7 made FULL's "seen" votes stricter: a pixel only counts as "still there" if the room behind the object is far enough away that its absence would show in the depth (a pillow lying on a sofa looks the same whether it is there or not). M7 also added two experimental switches (deciding "removed" immediately instead of at the end; unseen things moving along with the furniture they stand on) and the machinery that runs every experiment (ablation) and writes the final results tables and plots. M8 added the pictures (a top-down before / after figure and a Rerun replay of the walk), the hand check of the ground truth (audit), the README, the report and the slides.

## 2. Folder map

```
Task2/
├── configs/default.yaml     all tunable numbers (thresholds, ICP, relation rules, eval definitions, gates)
├── docs/                    this file, the report, the slides and their figures
├── splits/                  lists of scan ids / (reference, rescan) pairs to use
├── src/lsg/                 the Python package (run as `uv run lsg …`)
│   ├── cli.py               the `lsg` command
│   ├── config.py            loads the YAML config, finds data/outputs folders
│   ├── runinfo.py           run names, git commit, writes metrics.json
│   ├── train_meta.py        train-split metadata: the CLIP vocabulary and the class priors (M6)
│   ├── classes.py           which object classes are "structure", "anchor", "thing"
│   ├── io/                  READING DATA
│   │   ├── scan3r.py        3RScan: metadata, meshes + labels, camera frames
│   │   ├── dssg.py          3DSSG: object attributes and relations
│   │   └── splits.py        split lists: which references need textures
│   ├── graph/               SCENE GRAPHS
│   │   ├── model.py         Graph / Node / Edge, JSON in and out, validate()
│   │   ├── relations.py     "what stands on what": from 3DSSG or from geometry
│   │   └── build.py         builds the reference graph of a scan
│   ├── geom/                GEOMETRY
│   │   ├── transforms.py    moving points between coordinate frames, depth -> 3D points
│   │   ├── render.py        draws a mesh as seen by a camera (depth + object ids; texture colour, M6)
│   │   └── align.py         fine-tunes how the rescan lines up with the reference
│   ├── obs/                 OBSERVATION INTERFACE (what the method sees of a rescan)
│   │   └── base.py          ObsInstance, ObsSource, RefContext, SeqInfo, PackedMasks
│   ├── change/              CHANGE LOGIC (method side)
│   │   ├── pipeline.py      frame selection, per-pair preparation (G_ref, refined T_align)
│   │   ├── visibility.py    per-frame votes (seen / free / free-like / no info), viewpoint decorrelation
│   │   ├── confidence.py    log-odds confidence (M6; no class prior since CHANGE 2)
│   │   ├── online.py        the online frame loop + tentative snapshots
│   │   ├── association.py   B2 matching: confirmed nodes absorb their observations, then Hungarian
│   │   ├── classify.py      final states: ICP moved test, removed / unknown rule
│   │   ├── update.py        update_graph: support relations (update / stale R0 / geometric R1)
│   │   └── baselines.py     B0: naive same-label matching; B1: + visibility removal rule; B2; FULL (M6)
│   ├── eval/                CHECKING / SCORING (may read rescan ground truth)
│   │   ├── m0_checks.py     the four M0 acceptance checks
│   │   ├── gt.py            ground truth per pair: moved / removed / new / ...
│   │   ├── gt_stats.py      M1 checks: GT counts, displacement histogram, rule sanity
│   │   ├── metrics.py       scoring: matching rules, per-pair counts, averages + bootstrap CIs
│   │   ├── harness.py      `lsg eval` / `lsg run`: method + observations + GT -> metrics.json
│   │   ├── relations.py     relation scoring: parent / type accuracy per subset
│   │   ├── m4_checks.py     `lsg m4`: perception on dev, candidate recall, overlay PNGs, sweep
│   │   ├── m5_checks.py     `lsg m5`: B0 / B1 / B2 on dev, sweep, B2 on val + relation table
│   │   ├── m6_checks.py     `lsg m6`: OBJ/PLY check, texture render check, FULL variants on dev, val
│   │   ├── ablations.py     the list of M7 experiments (rows) and `lsg ablate`
│   │   ├── m7_checks.py     `lsg m7`: every row on dev / val / relchange, results.md, plots, determinism
│   │   ├── audit.py         `lsg audit`: sample false positives, photo panels, CSV for manual verdicts (M8)
│   │   ├── newcurve.py      `lsg newcurve`: new-object P / R over the candidate-confidence threshold (CHANGE 2)
│   │   ├── devtune.py       `lsg splits devtune`: picks the tuning set dev_tune from train metadata (CHANGE 3)
│   │   ├── movediag.py      `lsg movediag`: predicted moves split by old-place evidence (CHANGE 3 Part A)
│   │   ├── chg3_checks.py   `lsg chg3 parta` / `pdel`: CHANGE 3 variant sweeps on dev_tune + selection rules
│   │   ├── votestats.py     `lsg chg3 votestats`: vote weights as likelihood ratios (CHANGE 3 Part C)
│   │   ├── anytime.py       `lsg anytime`: metrics over the walk from snapshot graphs, state flips (CHANGE 3 Part B)
│   │   ├── newfunnel.py     `lsg newfunnel`: where GT new objects are lost in perception (CHANGE 3 Part E)
│   │   └── figures.py       the GT panel of the top-down figure (M8)
│   ├── viz/                 PICTURES (M8; reads only method inputs / outputs)
│   │   ├── topdown.py       before / after top-down PNG, nodes coloured by state
│   │   └── rerun_timeline.py  Rerun recording: mesh, camera path, node boxes per snapshot, final graph
│   ├── oracle/              ground-truth rescan objects: the upper-bound mode
│   │   └── source.py        OracleSource
│   └── perception/          finds rescan objects from depth/colour alone: the headline mode
│       ├── residual.py      "nearer" depth pixels -> 3D points, minus reference surfaces / structure
│       ├── voxels.py        4 cm voxel grid counting distinct viewpoints per voxel; per-voxel log-odds with see-through votes (CHANGE 2)
│       ├── cluster.py       groups kept voxels into candidates; door swing-region filter
│       ├── segment.py       per-frame image segments + co-segmentation clustering (CHANGE 3 Part N)
│       ├── detect_cache.py  `lsg detect cache`: Grounding DINO + SAM 2 detections per frame (CHANGE 3 Part E)
│       ├── detector.py      detections -> 3D tracks with a newness score (Part E, row FULL+detector)
│       ├── labeler.py       CLIP zero-shot labels from colour crops (cached)
│       ├── appearance.py    rendered texture vs photo, DINOv2 patch features (cached) (M6)
│       └── source.py        PerceptionSource: the above as an ObsSource
├── docs/report/, docs/slides.pptx, docs/figures/   the report (Typst source + PDF), the slides (built by docs/slides/build_deck.js) and their figures
├── README.md                setup, data for one pair, run it end-to-end, evaluation commands
├── tests/                   automated tests (`uv run pytest -q`)
├── data/      (git-ignored) the downloaded dataset
├── cache/     (git-ignored) refined alignments (cache/align/), CLIP labels (cache/clip/), appearance similarities (cache/appearance/)
└── outputs/   (git-ignored) results of runs: outputs/<run_id>/metrics.json, PNGs, graphs/
```

## 3. How the pieces fit together

```
 data/3RScan.json ──► io.scan3r.load_meta ──► which scans belong together,
                                              T_align, moved/removed lists
 data/<scan>/*.ply ─► io.scan3r.load_scan ──► mesh + object labels ──► geom.render.Renderer
 data/<scan>/sequence.zip ─► io.scan3r.open_sequence ──► frames (depth, pose, color)
                                   │                            │
                                   ▼                            ▼
                    geom.align.refine_from_sequence     render the reference room from
                    (better T_align, using depth only)  the rescan camera and compare
                                   │                    with the measured depth
                                   └──────────► eval.m0_checks ──► outputs/<run>/metrics.json
                                                                   + overlay PNGs
```

## 4. Key ideas, in plain words

| Term | Meaning |
|---|---|
| **Mesh** | The 3D surface of a scanned room, made of triangles. Each vertex carries an object id (which object it belongs to) and a class id (NYU40, e.g. 5 = chair). |
| **Frame** | One moment of the scan: a color photo, a depth image (distance per pixel, in metres) and a **pose** (where the camera was and where it looked). |
| **Coordinate frame** | Every scan has its own "origin and axes". A point's numbers depend on which frame it is written in. |
| **`T_align`** | A 4x4 matrix from 3RScan.json that moves rescan coordinates into the reference's coordinates, so both rooms overlap. |
| **Refined `T_align`** | `T_align` is a few cm off. `geom/align.py` improves it with ICP (below) using only the rescan's depth images. |
| **ICP** | "Iterative closest point": repeatedly pair each rescan point with the nearest reference surface and shift the rescan to reduce the gaps. |
| **Rendering** | Computing what a camera *would* see of a mesh: per pixel, the depth and the object id of the first surface hit. |
| **Render-and-compare** | Render the *reference* room from the rescan camera and compare with the depth the rescan *measured*. Where they differ, something changed (used from M3 on). |
| **`rigid` transform** | Ground truth for a moved object: maps its reference copy onto its rescan copy. M0 found it lands in the *rescan's* frame. |

## 5. Module by module

### `config.py` — settings
- `load_config()` reads `configs/default.yaml` into a dict. All thresholds live there, never hard-coded.
- `config_hash()` gives a short fingerprint of the settings. It is saved with every result.
- `data_root()` finds the dataset: environment variable `LSG_DATA`, else `./data`. `outputs_root()` finds `./outputs`.
- `setup_hub_offline(cfg)` (M7): once the DINOv2 and CLIP weights are in the local Hugging Face cache, `lsg` tells the Hugging Face library not to contact the internet at all (`HF_HUB_OFFLINE=1`); on a fresh machine they are downloaded once. Config `models.hub_offline`.

### `runinfo.py` — bookkeeping for results
- `write_metrics()` saves `outputs/<run_id>/metrics.json` with the results, the full config, its hash and the git commit. Any number can be traced back to the exact code and settings.
- `git_sha()` and `new_run_id()` are its helpers.

### `classes.py` — object categories
- `node_level(nyu40)` sorts a class into **structure** (walls, floor, doors…; not graph nodes), **anchor** (big furniture like tables, beds, shelves) or **thing** (everything else: chairs, pillows, cups…).

### `io/scan3r.py` — reading 3RScan

| Function / class | What it gives you |
|---|---|
| `load_meta(root)` → `Meta` | All of 3RScan.json. `meta[id]` tells you if a scan is a reference or rescan, its `T_align`, and (ground truth) which objects moved, were removed or deformed. |
| `read_pairs(file)` | The (reference, rescan) pairs from a `splits/` file. |
| `load_scan(root, id)` → `Scan` | A scan's labelled mesh: vertices, triangles, object id and class per vertex and per triangle, plus an `Instance` summary per object (label, class, centre, bounding box). **Reads annotations**; see section 6. |
| `load_textured_mesh(root, id)` → `TexturedMesh` | A **reference** scan's photo-textured mesh (`mesh.refined.v2.obj` + texture PNG), M6. Same vertices and triangles as the PLY (checked on all 72 references). |
| `open_sequence(root, id)` → `SequenceReader` | Access to the camera frames. Reads straight from `sequence.zip`, so nothing has to be unzipped. |
| `SequenceReader.pose/depth/color(i)` | Pose (4x4), depth in metres (0 = no measurement) and color image of frame `i`. Images stay rotated 90°, as stored. |
| `iter_frames(seq, stride)` | Loops over every n-th frame, skipping frames with a broken pose. |
| `read_ply`, `read_pgm16`, `parse_info` | Low-level file decoders (mesh file, depth image, camera settings). |

### `io/dssg.py` — reading 3DSSG
- `load_objects(root, id)`: per object, its label, class ids and attributes.
- `load_relations(root, id)`: relations such as `(12, 5, "standing on")` = object 12 stands on object 5.

The big JSON files are parsed only once per run and then reused.

### `geom/transforms.py` — coordinate maths
- `mat4_from_colmajor`: turns the 16 numbers in 3RScan.json into a 4x4 matrix (they are stored column by column).
- `apply(T, points)` moves points with a transform. `invert(T)` gives the reverse transform.
- `backproject(depth, K, pose)`: depth image → 3D points. `project(points, K)` goes the other way (3D → pixel).
- `rotation_angle_deg(T)`: how much a transform rotates.

### `geom/render.py` — drawing a mesh from a camera
- `Renderer(vertices, faces, labels)` prepares a mesh once, then `render(K, T_w2c, W, H)` returns, for a camera, a **depth image** and an **object-id image** (~5 ms per depth-sized frame, CPU only).
- `distance(points)` gives each point's distance to the nearest surface of the mesh (used by the checks).
- `depth_behind(...)` (M7): for chosen pixels, how far away the next surface *behind* the first-hit object is (the first surface along the ray that belongs to something else). It lists every surface the pixel's ray passes through, so a backrest touching the back of a pillow is found too.
- `TextureRenderer(textured_mesh).render_color(K, T_w2c, W, H)` (M6): what the camera *would* see of the reference room's photo texture (finds the triangle hit by each pixel ray, interpolates its texture coordinates, reads the texture image). ~55 ms per 448x252 frame.

### `geom/align.py` — improving the rescan↔reference alignment
- `refine_from_sequence(seq, ref_mesh, T_align, cfg)` builds a point cloud from the rescan's depth images and runs ICP against the reference mesh, starting from `T_align`.
- Moved furniture is ignored automatically, because points far from the reference surface don't count.
- If the correction is implausibly large (> 20 cm or > 5°), it is rejected and the original `T_align` is kept.
- It uses no labels, so the method itself may use it.

### `eval/m0_checks.py` — the M0 acceptance checks (`uv run lsg m0`)

| Check | Question it answers | Function |
|---|---|---|
| (1) depth | Does the rendered reference room match the rescan's measured depth? (median error < 3 cm) | `check_depth` |
| (2) structure | After alignment, do walls and floors of both scans coincide? Also settles the units (metres, not mm). | `check_structure` |
| (3) rigid | What exactly does a "moved object" transform map from and to? | `check_rigid` |
| (4) overlays | Do the labels visually line up with the photos? Saves PNGs. | `make_overlays` |

`run_m0()` runs all four on the dev pairs, applies the pass/fail thresholds from the config and writes `metrics.json`. `format_summary()` prints the table you see in the terminal.

### `io/splits.py` — split lists
- `reference_ids(split_files)`: the distinct reference scans of some pair lists.
- `write_texture_list(split_files, out)`: writes `splits/download_textures.txt` (the 72 references whose textures are needed; rescans never need them).

### `graph/model.py` — the scene graph and its JSON
- `Node`, `Edge`, `Graph`: the fields of the graph JSON. Ids: `r<objectId>` for reference objects, `n<k>` for new ones, `room` for the room. `floor` is only a parent name, not a node.
- `Graph.save(path)` / `Graph.load(path)` write and read the JSON; `support_parent()` gives `{child: (parent, "on"|"in")}`.
- `validate(graph)` lists schema problems (missing keys, more than one support edge per thing, edges to unknown nodes, ...). Empty list = valid.

### `graph/relations.py` — what stands on / in what
- `gt_support(relations, objects)`: turns 3DSSG relations ("standing on", "lying in", ...) into support parents (`on` / `in`), with the floor as a special parent. Things that only hang on walls are marked `excluded`.
- `pick_parent(parents, level_of)`: if 3DSSG lists several parents, picks one (anchor first, then thing, then floor).
- `estimate_floor_z(...)`: height of the floor (it is at z ≈ −1.4, not 0).
- `infer_support(child_pts, candidates, floor_z, cfg)`: the **geometric** rules. "On": enough of the child's lowest points rest on a candidate just below them (but not on a neighbour's bottom edge at floor level). "In": most of the child lies inside a piece of furniture's box. Otherwise "on the floor" if the child's bottom is near the floor.

### `graph/build.py` — the reference graph
- `build_reference_graph(scan, objects, relations, mode, cfg)`: one node per anchor/thing (structure such as walls gets none), a `room` node, `hierarchy` edges, at most one support edge per thing, `next_to` edges, and per thing its `anchor` (first piece of furniture up the chain: book → box → shelf gives shelf).
- `mode="gt"` takes relations from 3DSSG; `mode="geometric"` computes them with `infer_support` (baseline R1, and later used for moved / new objects).

### `eval/gt.py` — what really changed (ground truth)
- `gt_changes(scan_meta, R, S, objs_R, objs_S, cfg)`: for every reference anchor/thing, one status: `moved`, `removed`, `unchanged`, `absent_unlisted` (not in the rescan, but not listed as removed either) or `dont_care` (with a reason: removed-but-still-there, non-rigid, or moved less than 10 cm). Also the true displacement and new position of moved objects, the new objects (in the reference frame), and a size class (small / large).
- `gt_parent_changes(...)`: for moved things, did their support parent change between the two visits (same / swapped / gained / lost)?

### `eval/gt_stats.py` — the M1 checks (`uv run lsg m1`)
- Counts per split (must reproduce the expected raw counts), a histogram of how far objects moved (`outputs/<run>/gt_displacement_hist.png`), and how often the geometric relation rules agree with 3DSSG on the reference scans.

### `obs/base.py` — the observation interface
- `ObsInstance`: one object seen in the rescan. It has an opaque `token`, a label (+ `label_topk`), a class id, its observed 3D points in the **reference** frame, and a mask per frame showing where it was seen.
- `ObsSource`: anything that produces `ObsInstance`s frame by frame: `begin(...)`, then `observe(frame, ...)` for every processed frame in time order, then `finish()` returns the instances. The oracle implements it now; perception (M4) will too. The change logic only ever talks to this interface.
- `RefContext` (the reference scan + its graph) and `SeqInfo` (rescan id, camera parameters, the refined `T_align`, which frames are processed) bundle what the method is allowed to know.
- `PackedMasks` stores the per-frame masks bit-packed (8x smaller) but reads like a dict of boolean images.

### `train_meta.py` — what the method may learn from the train split
- `clip_vocab_table(root, cfg)`: thing/anchor labels of the train scans with their class id and count. The rooms of the dev and relchange pair lists are excluded, and validation / test scans are never read.
- `perception_vocab(G_ref, table, top_n)`: the labels CLIP chooses from = the reference graph's labels plus the 100 most frequent train labels.
- `class_priors(root, cfg)` → `PriorTable` (M6): per class, how often a reference object was still there vs listed removed in the training rescans (e.g. chair 96.8%, box 93.9%, bed ~100%). `p_persist(label, nyu40)` smooths small classes toward their NYU40 class and the overall rate.

### `perception/` — finding objects without annotations (headline mode, M4)
The rule: no imports from `oracle/` / `eval/`, and no rescan annotations. Inputs are the reference (mesh, labels, graph) and the rescan's depth, colour, poses and `T_align`.

| File | What it does |
|---|---|
| `residual.py` | Per frame: pixels where the camera measured something at least 10 cm **in front of** the rendered reference surface ("nearer"), turned into 3D points. Points within 5 cm of the reference mesh are dropped, and so are points within 10 cm of walls, doors, windows and other structure (not the floor, so objects standing on it survive). `SurfaceFilter` does the distance tests. |
| `voxels.py` | `VoxelEvidence`: puts the points into 4 cm cubes and counts from how many *different* viewpoints each cube was hit, with the same 30 cm / 20° rule as the votes. A person walking by or sensor noise hits a cube from one viewpoint only; real objects are seen from many. CHANGE 2: each cube also has a log-odds confidence; a hit from a new viewpoint adds 2.5, a repeated viewpoint 0.25 (at most 1.0 in a row), and a later frame that sees *through* the cube (the depth around its pixel is > 15 cm behind it) subtracts the same. One new-viewpoint hit keeps a cube; a person who walks away is seen through and drops out. |
| `cluster.py` | Cubes seen from enough viewpoints (CHANGE 1: 2 for small clusters; a second, lenient pass with 1 viewpoint keeps only clusters of at least 40 cubes, which replace the small ones inside them; `perception.relaxed`) are grouped into candidates (cubes at most 2 cubes apart belong together); groups of fewer than 5 cubes are dropped. Groups lying mostly in the swing area of a reference door are dropped too: a door opened since the reference visit would otherwise look like a new object. |
| `labeler.py` | `CLIPLabeler`: shows CLIP the colour crops of a candidate and scores them against "a photo of a {label}" for every vocabulary label; keeps the top 5. Results are cached in `cache/clip/`, so reruns give identical labels. `StubLabeler` gives fixed answers for tests. |
| `appearance.py` | (M6) Compares the real photo with the reference texture drawn from the same camera, in 14x14-pixel patches, with DINOv2 features (or a plain colour fallback). Only where depth says nothing is in front and nothing was seen through. Distances are taken relative to the frame's typical distance, because rescans are lit differently. `AppearanceRunner` does this per frame and caches the result in `cache/appearance/`. |
| `source.py` | `PerceptionSource`: runs the above frame by frame (`observe`) and at the end (`finish`) turns each candidate into an `ObsInstance`: points, per-frame masks, 5 upright crops, CLIP label, token `c0`, `c1`, … (M6) With appearance on, pixels on tables / the floor whose photo no longer matches the texture are collected in a second voxel grid: flat new objects that depth cannot see (`kind = "appearance"`). |

### `oracle/source.py` — the upper-bound observations
- `OracleSource` renders the rescan's own **annotated** mesh from each rescan camera. Every object that shows up (where the sensor also has valid depth) becomes an `ObsInstance` with its real label and the mesh vertices that were seen. Objects with fewer than 50 seen vertices are dropped.
- Object ids are replaced by shuffled tokens (`o0`, `o1`, ...) so the method cannot cheat by id; `token_map()` undoes the shuffle and is only used by `eval/`.

### `change/pipeline.py` — preparing a pair
- `select_frames(...)`: every 3rd frame, or one contiguous chunk of them (for the "observe only 50 / 25 / 10%" experiments).
- `prepare_pair(...)`: builds the reference graph and the refined alignment. The refined alignment is cached in `cache/align/`, so ICP runs once per rescan (and chunk).

### `change/visibility.py` — what one frame says about each object
- `frame_votes(...)`: the reference room is rendered from the rescan camera. For each object, look at the pixels where it should be visible and compare the measured depth with the expected depth:
  - **seen**: at least half the pixels agree (within 10 cm);
  - **free**: at least half are measured *behind* where the object should be (by more than 15 cm), i.e. the camera saw through the object's old place;
  - **no info**: something is in front (occlusion), the depth is missing, or the evidence is mixed;
  - **unobserved**: fewer than 15 pixels (out of view or hidden).
- **Seen needs contrast** (M7, FULL only, `visibility.seen_needs_contrast`): `frame_votes(..., d_behind=...)` counts a pixel toward "seen" only if the surface behind the object is more than 15 cm farther away. Otherwise the pixel would look the same with the object gone, so it is no information.
- `Decorrelator`: a vote counts only if the camera moved at least 30 cm or turned at least 20° since the last *counted* vote of the same kind for the same object. Standing still for 30 frames gives one vote, not 30.
- `NodeVotes`: the tally per object (raw and counted seen / free / free-like votes; M7: `commit_removed_frame`, the first frame at which the object would tentatively have been called removed). `has_free_evidence` (≥ 2 counted free votes; appearance alone needs ≥ 3 free-like votes) and `confirmed_in_place` (≥ 3 counted seen, no free / free-like) are the rules built on it.
- `thin_votes(...)` (M6): for thin objects (thinner than 15 cm), depth agrees whether they are there or not, so their vote comes from the appearance check instead: photo still matches the texture → `seen`, clearly different → `free_like`.

### `change/online.py` — the online loop
- CHANGE 3 Part B: with `on_snapshot` (given by the eval harness) the loop calls back every `online.snapshot_every` frames with `source.peek()`, a copy of the votes and the views so far; the harness runs the method on that state, saves and scores the snapshot. With `alignment.online` the loop feeds `geom.align.OnlineAligner` and swaps in its accepted `T_align` estimates (sources read `seq_info.T_align` every frame).
- `run_online(...)`: walks the processed frames in time order. Per frame it renders the reference, casts and counts votes, and hands the frame plus the reference render to the observation source. At the end it collects the observations. Returns an `OnlineResult` (observations, votes, snapshot files, timings).
- Snapshots (`tentative_graph`) are graph JSONs written every 20 processed frames with *tentative* states: `removed` once there is enough free evidence, `unchanged` once confirmed in place, else `unknown`. The real decision is only made at the end (deferred commitment).

### `change/baselines.py` — B0, the naive change detector
- `b0(G_ref, obs, cfg)`: pairs each reference object with the nearest observed object of the **same label** (one-to-one, closest pairs first). Within 0.5 m it is `unchanged`, farther `moved` (and its position is updated); without a partner `removed`; observed objects nobody claimed become `new` nodes. It returns the updated graph.
- It ignores visibility, so anything simply not seen in the rescan is deleted. Later baselines fix exactly that.
- `b1(G_ref, obs, cfg, votes=...)`: the same matching, but an unmatched object is `removed` only with ≥ 2 counted free votes; otherwise it is `unknown` and kept. On val this cuts false deletions from 95% to 9.5%.
- `b2(G_ref, obs, cfg, votes=..., ref_ctx=...)` (M5): objects confirmed in place (≥ 3 counted seen votes, no free vote) are `unchanged` and absorb their observations; the rest go through `association.associate`; `classify.classify` decides moved / unchanged / removed / unknown; leftover observations become `new`. Moved objects get the ICP pose: centroid and box of the moved reference points, `T_move` = the full 4x4 motion.
- `full(...)` (M6): B2, plus a confidence for every object (`change.confidence`); an unmatched object with "seen through" evidence is deleted only if its confidence is below 20%, otherwise it stays `unknown`. With appearance on, thin objects' votes come from the appearance check, and perception also gets the flat-object candidates.
- Every method only decides states and geometry; relations are recomputed afterwards by `update.update_graph`.
- M7 switches: `states.commitment: immediate` makes "removed" final at the frame where it was first tentatively reached, even if the object turns up later (B2 / FULL; an experiment showing why waiting until the end matters). `update.children_follow_parent` (FULL, `follow_parent`): an unseen thing standing on furniture that moved moves with it.

### `change/confidence.py` — how sure are we that an object still exists (M6, CHANGE 2)
- Start value: 50% for every object since CHANGE 2 ("no information", log-odds 0); before, the class prior (e.g. 96.8% for a chair), still available as the ablation row `FULL+classprior`. Each counted "seen" vote adds `L_hit` = 2.5, each counted "seen through" (or appearance free-like) vote subtracts 2.5; the total change is capped at ±8.
- An object is deleted below 2.7% (`p_del` 0.027): 2 net "seen through" votes delete any object, 1 cannot, an object seen once and seen through twice is kept. These are the same decisions as M7's flat-prior row (90% / 20%).
- A matched object gets credit for the viewpoints its observation was seen from; its old-place "seen through" votes are ignored (it moved). A new object takes its candidate's confidence (perception, CHANGE 2) or credit for its views (oracle).

### `change/association.py` — who is who (M5)
- `absorb_confirmed(...)`: an observation lying mostly (≥ 50% of its points) inside the box (+10 cm) of an object confirmed in place, with a compatible label, belongs to that object. It is then neither new nor available for matching. In oracle mode this is the object's own instance; in perception mode it is usually a fake candidate from a mesh hole (black TV screen) or a plant.
- `associate(nodes, obs, mode, cfg)`: optimal one-to-one assignment (Hungarian, `scipy.optimize.linear_sum_assignment`). Cost = label term + size term + weak position term. Label term: oracle needs the same label (or the same class, +0.3, except the catch-all classes "otherprop" / "otherfurniture", where e.g. a keyboard and a blanket share a class); perception uses how likely CLIP thinks the object's label is (−log p, at least p_min = 0.05). Pairs costing ≥ `tau_assoc` stay unmatched.

### `change/classify.py` — moved or not (M5)
- `icp_move(ref_pts, obs_pts, cfg)`: aligns the (often partial) observed points onto the object's reference points with ICP, trying two starting guesses (the centroid offset and "not moved") and keeping the better fit. Returns the object's motion `T_move`.
- `classify(...)`: moved if the ICP motion shifts the centroid by > 10 cm, else unchanged; unmatched objects: removed with ≥ 2 counted free votes, else unknown.

### `change/update.py` — relations after the change (M5)
- `update_graph(G, ref_ctx, obs, cfg, policy)`. `update` re-checks only moved and new things and things whose parent moved or was removed: a re-checked thing keeps its old parent if that parent still carries it at the new position (`graph.relations.still_supports`), otherwise `infer_support` picks a new one. `stale` (baseline R0) keeps the reference relations. `geometric` (baseline R1) recomputes every thing. A removed object is never anyone's parent; the `anchor` field and `next_to` edges are refreshed.

### `eval/metrics.py` — scoring
- `match_new(...)`: which predicted new nodes are real new objects. Oracle: by token. Perception: same label (or class) and close (<= 30 cm) or overlapping (box IoU >= 0.25), best overlaps first, one-to-one.
- `score_pair(G_upd, gt, ...)`: counts for one pair: hits / false alarms / misses for moved, removed and new; unchanged right / kept; false deletions and false moves on objects the rescan annotation doesn't mention; how often the method paired an object with the right observation; moved recall per distance and size bucket; how far off the new position of a correctly detected move is.
- `aggregate(...)`: sums the counts over all pairs, turns them into percentages, and adds a 95% confidence interval by resampling pairs (bootstrap). Every number comes with its support `n`.

### `eval/relations.py` — scoring relations (M5)
- `score_relations(...)`: per thing with a support parent in the rescan's 3DSSG annotation, is the predicted parent the same object (parent accuracy), and is `on` / `in` right (type accuracy)? Counted for (a) all such things, (b) moved things, (c) moved things whose annotated parent changed (swapped), (c-broad) swapped or gained; each with and without floor parents, and once over all things ("GT universe") and once only over the things the method associated.

### `eval/m5_checks.py` — the M5 checks (`uv run lsg m5`)
- Runs B0, B1 and B2 on the dev pairs in both modes and checks: B2's association accuracy ≥ B0's, B2's moved F1 ≥ B1's, all graphs valid. Optional sweep of `tau_assoc` × `w_p` on dev. Then B2 on val in both modes, with the relation table R0 / R1 / update.

### `eval/ablations.py` — the experiments (`uv run lsg ablate`, M7)
- `registry(cfg)`: every row of the M7 tables, each a method plus a few config changes (e.g. `FULL-appearance` = FULL with `appearance.enabled: false`), and optionally a share of the frames. `lsg ablate --list` prints them.
- `run_ablation(...)`: runs one row with `run_eval` into its own folder. If that folder already holds a result made with the same settings and the same committed code, it is reused (so a crashed multi-hour run can be restarted).

### `eval/devtune.py` — the tuning set (`uv run lsg splits devtune`, CHANGE 3)
- `pair_counts(meta, objects_of, exclude_refs)`: per train (reference, rescan) pair, how many things / anchors were removed, moved (`rigid`) and added, from 3RScan.json + 3DSSG `objects.json` only (no downloads). It reads train rescans' change lists, so it lives in `eval/`.
- `select_dev_tune(...)`: the 20 rooms with the most changes (best rescan each) plus 10 random rooms (seeded); the CLI appends the 3 `dev_train` pairs and writes `splits/dev_tune.txt` and `splits/download_tier3.txt` (scans not yet downloaded).
- Settings in `splits.dev_tune` of `configs/default.yaml`. All tuning happens on `dev_tune` (CLAUDE.md hard rule 4); `dev_train` stays the 3-pair regression set.

### `perception/segment.py` — CHANGE 3 Part N (co-segmentation)
- `segment_frame`: one frame's residual pixels form a graph of neighbouring pixels; it is cut where the depth jumps and on concave creases (`normals_from_depth`, `concave_band`), and its connected parts are the frame's segments.
- `cosegment_cluster`: kept voxels next to each other are joined only if no frame that sees both puts them in different segments; voxels never seen in one frame are not joined across gaps. Used by both candidate tiers in `source.py` when `perception.segment.method` is `geometric`.
- Removed 2026-11-26 (lost their dev_tune selections, no reported row used them): the SAM segment source and `lsg sam cache`, the N2 reference explain-away, the N3 identity check (`lsg chg3 idstats`, `partn-ref`) and the N5 noise filters.

### `perception/detect_cache.py`, `perception/detector.py` — CHANGE 3 Part E (detector, off by default)
- `DetectorRunner.detect`: Grounding DINO on the upright colour frame with the vocabulary as one prompt (`fit_prompts`, `ordered_prompts`: reference labels first), a score per label from the token logits (`label_scores`), NMS, then SAM 2 masks from the boxes at depth resolution; `cache_rescan` writes one `.npz` per frame and `done.json`.
- `DetectorTracks`: per frame, each detection's mask becomes points (erosion, depth outliers, structure filter), 4 cm voxels; detections join the track that holds most of their voxels, tracks that overlap are merged; `instances()` gives points, masks, views, summed label scores and `newness`.
- `PerceptionSource` with `perception.detector.enabled` uses them (`mode: replace | union`); `change.baselines.emits_new` gates detector instances by newness. `eval/chg3_checks.run_part_e` is the dev_tune sweep (`lsg chg3 parte`).

### `eval/votestats.py` — CHANGE 3 Part C (`uv run lsg chg3 votestats`, `uv run lsg chg3 pdel`)
- `node_vote_weights`: from a run's `details/`, the share of seen / free / free-like votes of every GT-unchanged and every GT-removed object (each object counts once), turned into the weights L_hit, L_miss, L_free_like as likelihood ratios, with bootstrap intervals; too few votes of a kind → the fallback 2.5.
- `candidate_vote_weights`: the same for perception candidates' cluster votes, explained vs unexplained candidates.
- The cluster votes themselves are computed in `perception/source.py` (`cluster_votes`, `_history`) from the logged voxel hits and see-through events (`VoxelEvidence.free_log`).

### `eval/movediag.py` and `eval/chg3_checks.py` — CHANGE 3 Part A (`uv run lsg movediag`, `uv run lsg chg3 parta`)
- `movediag`: reads a run's `details/` and counts every node predicted `moved` by its GT status, the evidence at its old place (`net_seen`, `free_ev`, `unobserved`, `mixed`; the same `old_place_class` the method uses), whether it took its own instance or another object's, and how far it moved.
- `chg3_checks.run_part_a`: runs each variant of `chg3.part_a.variants` (FULL on dev_tune, both modes) and applies the selection rule written before the run (`select`).
- The rule itself lives on the method side: `change.association.old_place_gate` (pair gate), `change.classify.classify(old_place=True)` (`tau_move_seen`), wired in `baselines._b2_core` for FULL only.

### `eval/m7_checks.py` — the M7 run (`uv run lsg m7`)
- Dev rows first, compared with the stored M5 / M6 dev results (must be identical where nothing changed); then all val rows, the τ_free sweep, the observed-fraction runs (3 seeds each), the relation runs on the relchange set, and FULL twice on one pair (must be identical).
- Writes `outputs/<run>/results.md` (all tables with 95% confidence intervals and support counts), three plots (`m7_fraction.png`, `m7_tau_free.png`, `m7_ablations.png`) and `metrics.json` with the acceptance checks.

### `eval/audit.py` — checking the ground truth by hand (`uv run lsg audit`, M8)
- `lsg audit sample` reruns FULL on val with a `details` switch: for every object it writes what GT says, what the method predicted, its votes and confidence (`details/<mode>/<pair>.json`). The rerun must give exactly the M7 numbers (checked).
- From these it collects the false positives per mode (moved / removed, including objects the annotation simply lacks, and new nodes that match nothing), draws 10 of each at random (seed 0), and adds every truly removed object that B2 deletes but FULL keeps.
- For each it draws one panel: the reference photo of the object, the rescan photo of the same place, the rescan photo of the predicted new place, and a top-down sketch. `audit.csv` lists them with an empty `verdict` column.
- Each panel asks one plain question ("did this chair really move?"); after a person answers `yes` / `no` / `unsure` in the CSV, `verdict_of` maps the answer to `gt_missing` / `method_error` / `unclear` (inverted for the veto set) and `lsg audit summarize` gives the share of real GT gaps with 95% confidence intervals.

### `eval/figures.py` and `viz/` — pictures (`uv run lsg viz`, M8)
- `viz/topdown.py`: the room seen from above, every object as its box footprint: the reference graph, the updated graph (moved = blue with an arrow from the old place, removed = orange hatched, new = aqua with its label, unchanged = grey, unknown = dotted) and optionally the ground truth (computed in `eval/figures.py`, because only `eval/` may read it).
- `viz/rerun_timeline.py`: a recording for the Rerun viewer. Scrubbing the `frame` timeline shows the camera walking through the reference room and the boxes changing their tentative state, ending with the final graph. It reads the snapshots of a `lsg run` (plus `seq_info.json`, written next to them) and the rescan's poses.

### `eval/m6_checks.py` — the M6 checks (`uv run lsg m6`)
- Checks on all 72 references that the textured OBJ and the labelled PLY are the same mesh.
- Draws the texture from the reference's own camera path and saves photo | render | blend PNGs; reports how similar they are and sets the appearance thresholds (the 98th percentile of the differences on these unchanged sequences).
- Runs B2 and FULL (appearance off, on without / with flat-object candidates) on dev in both modes, an optional `L_hit` sweep, and FULL on val.

### `eval/m4_checks.py` — the M4 checks (`uv run lsg m4`)
- Runs B0 and B1 on the dev pairs in both modes and reports how many real changes got a candidate: **candidate recall** against new objects and against the new places of moved objects, with and without requiring the right label.
- Draws one overlay PNG per dev pair: the colour frame with the candidates coloured and labelled, plus yellow markers where GT says a new / moved object is.
- Sweeps the door filter (on / off) and the minimum cluster size (3 / 5 / 10) as a diagnostic.

### `eval/harness.py` — `lsg eval` and `lsg run`
- For every pair: load the GT, prepare the pair, run the online loop with the oracle, run the method with the observations and votes, recompute relations with all three policies (`update_graph`), score changes and relations, check the graph with `validate`, and save the updated graph to `outputs/<run>/graphs/<mode>/` (snapshots to `outputs/<run>/snapshots/<mode>/` if asked).
- Besides the scores it reports *evidence diagnostics* that do not depend on the method: how many truly unchanged / removed / absent objects got free (seen-through) evidence.
- Perception mode: the frames also go to a hidden `OracleSource` (only to know which GT objects were visible, for the association score; the method never sees it). Candidates are matched to GT objects with the matching rule (same label or class, and close). The candidate lists are saved to `outputs/<run>/candidates/`.
- Pairs of the same reference run in one worker process (4 in parallel by default). Results are sorted back into a fixed order, so the `metrics` block of metrics.json is identical across reruns; timings go into a separate `run` block.

### `cli.py` — the command line
- `uv run lsg m0` runs the M0 checks. Exit code 0 means checks (1)–(3) passed.
- `uv run lsg m1` runs the M1 checks (about 70 s). Exit code 0 means all four passed.
- `uv run lsg splits textures` writes `splits/download_textures.txt`.
- `uv run lsg graph --scan <reference id> --mode gt|geometric [--out file.json]` exports one reference graph and validates it.
- `uv run lsg eval --split eval_val --obs oracle --method B0|B1 [--workers N] [--frac F --seed S] [--snapshots]` runs and scores a whole split (~65 s for val); `uv run lsg run --pair <ref> <rescan> --obs oracle --method B1 [--no-snapshots]` does one pair and writes snapshots. Both print a results table and write `outputs/<run>/metrics.json`. `--obs perception,oracle` runs both modes.
- `uv run lsg m4 [--no-sweep]` runs the M4 checks on dev (~4 min, GPU for CLIP if available).
- `uv run lsg m5 [--no-sweep] [--no-val]` runs the M5 checks (dev ~1 min, sweep ~8 min, val a few minutes with the CLIP cache warm). `--method B2` works in `lsg eval` / `lsg run`.
- `uv run lsg m6 [--no-sweep] [--no-val] [--no-objply]` runs the M6 checks. `--method FULL` works in `lsg eval` / `lsg run` (DINOv2 on the GPU if available; appearance results are cached).
- `uv run lsg ablate --names FULL,FULL-appearance --split eval_val` runs single experiment rows; `uv run lsg m7 --run-id m7_final` runs all of M7 (several hours; rerunning the same command resumes).

## 6. The one rule that shapes the layout

The method must not peek at the rescan's answers (CLAUDE.md, hard rule 1). Rescan **annotations** are:
- the rescan's object labels (PLY / semseg / 3DSSG);
- the lists of moved, removed and non-rigid objects in 3RScan.json.

These may only be read inside `eval/` (scoring, checks) and `oracle/` (the upper-bound stand-in for perception). `perception/`, and every other method-side package (`obs/`, `change/`, `graph/`, `geom/`, `io/`), may not even import `oracle/` or `eval/`; `tests/test_no_leak.py` checks this. The method also never receives the rescan's `ScanMeta` (which holds the moved/removed lists), only its `T_align` matrix. Everything else reads a rescan only through `open_sequence` (photos, depth, poses). That is why `load_scan` and `SequenceReader` are separate. It is also why `align.py` uses depth only, and why the M0 checks live in `eval/`.

## 7. Tests (`tests/`)

| File | What it checks (all on small synthetic data unless noted) |
|---|---|
| `test_config.py` | Config loads; hash ignores key order; `LSG_DATA` override; metrics.json contents. |
| `test_transforms.py` | Column-major parsing; inverse; depth → 3D → pixel round trip. |
| `test_io.py` | Depth images (big-endian, comments); zip and folder give identical frames; both kinds of `rigid` entries; PLY parsing; object summaries; 3DSSG; split files. |
| `test_render.py` | A flat wall renders at the right depth everywhere; a box in front gets the right label; empty pixels are 0; a posed camera's depth back-projects onto the mesh. |
| `test_align.py` | ICP recovers a known 5 cm / 1° misalignment; oversized corrections are rejected. |
| `test_relations.py` | 3DSSG → on/in mapping; contact, containment, floor and tie rules on synthetic boxes; floor-level neighbours don't count as support. |
| `test_graph.py` | A synthetic room built in both modes: no structure nodes, one parent per thing, anchor chain, JSON round trip, `validate`. |
| `test_gt.py` | A synthetic pair with every GT case (moved, small move, bare-int, removed, removed-but-present, non-rigid, new, absent); parent-change kinds. |
| `test_no_leak.py` | Method-side packages never import `oracle/` or `eval/` (source scan + a fresh-interpreter import check); `perception/` contains no rescan-annotation reads; the checker itself catches planted violations. |
| `test_metrics.py` | 5-object toy pair: P/R/F1, unchanged strict + kept, false-deletion / false-move rates, unknown rate, association, buckets, localisation error; the perception matching rule (label vs class, distance vs overlap, one-to-one); bootstrap determinism. |
| `test_oracle.py` | `OracleSource` on synthetic boxes: masks only where depth is valid, hidden objects not seen, points in the reference frame, `min_points`, deterministic token shuffle; `PackedMasks`. |
| `test_b0.py` | B0 states (unchanged / moved / removed / new), one-to-one matching, structure ignored, exact labels, threshold from the config; the result validates. |
| `test_pipeline.py` | Frame chunks; alignment cache hits and misses. |
| `test_visibility.py` | Frame votes on hand-made depth maps (seen, free, occluded, too few pixels, invalid depth, the 50% rule, erosion); decorrelation (same viewpoint 30× → 1 vote, 30 cm / 20° limits, independent keys, off switch); vote tallies. |
| `test_online.py` | A synthetic room with real ray casting: a removed box → free votes → `removed`; removed but occluded / out of view → `unknown` (B0 deletes all three); one viewpoint repeated → 1 vote; tentative snapshots switch from `unknown` to `removed` mid-walk; works with `OracleSource`. |
| `test_perception.py` | Synthetic room with real ray casting: a box added → exactly one candidate, centroid within 5 cm (of its visible surface); unchanged room → none; one viewpoint repeated → none (decorrelation); object flat against a wall filtered, 30 cm off the wall kept; swung door dropped by the swing region (kept with the filter off); crops reach the labeller; CLIP top-k order, ties and cache; nearer-pixel rules; voxel view counting; clustering; door regions; train vocabulary excludes dev/relchange rooms and val. Shared scene helpers live in `synth.py`. |
| `test_b1.py` | B1's rule on hand-made votes (2 counted free → removed, 1 → unknown, matched nodes as in B0). |
| `test_association.py` | Label terms (oracle gate + class fallback, perception soft term), size term, Hungarian beats greedy, gate, determinism, absorption (own instance; mesh-hole candidates many-to-one; a different-label new object is kept). |
| `test_classify.py` | ICP recovers a move; a half-seen unchanged box is not moved although its centroid offset is > 10 cm; low-fit fallback; final states. |
| `test_b2.py` | B2 on a point-box room: confirmed + absorbed, moved via ICP (partial view), removed, unknown, new; no far "moves"; perception mode. |
| `test_update.py` | Cup moved from table to shelf gets the shelf; things on a moved or removed parent are recomputed; a new bottle gets the table; R0 keeps everything, R1 recomputes everything. |
| `test_texture.py` | OBJ reading (per-vertex and per-corner texture coordinates); a quad with a four-colour texture renders each colour in the right image corner; empty pixels are black. |
| `test_appearance.py` | Colour-feature similarity; patch ↔ pixel mapping; which pixels may be compared; the frame-relative distance; the thin-object vote rule; free-like decorrelation and evidence rules; cache. **Acceptance:** a 2 cm striped book removed from a table is missed by depth only (`unchanged`) and caught with appearance (`removed`); a book *added* gives exactly one appearance candidate (centroid < 5 cm), depth alone none. Slow variant with real DINOv2. |
| `test_contrast.py` | (M7) The surface behind an object is found through the object's own back faces and also when it touches the back face; the "seen needs contrast" vote rule on hand-made images; a pillow against a backrest removed: the old rule says "seen", the new one "no info"; a box in front of a wall is "seen" under both. |
| `test_m7_variants.py` | (M7) Immediate commitment removes what deferred commitment calls moved (and the observation then goes to another chair or becomes "new"); the commitment frame is recorded during the walk; a cup follows its moved table and keeps it as parent. |
| `test_ablations.py` | (M7) Config overrides (typos rejected), the experiment list, when a stored result may be reused, and the results tables / plots / acceptance on made-up results. |
| `test_viz.py` | (M8) Room outline keeps structure, drops the ceiling; the top-down PNG with two and three panels; a Rerun recording from made-up snapshots has the camera, trajectory, node groups and the final graph after the last frame. |
| `test_audit.py` | (M8) Per-object details do not change any count; which predictions land in which false-positive pool; the veto set; seeded sampling; the CSV round trip and the verdict summary (incl. a missing verdict being refused); Wilson interval; comparing two `metrics` blocks. |
| `test_confidence.py` | Class-prior counting and smoothing; the worked examples (chair 2 free → deleted, bed 2 free → kept, 3 → deleted, mixed evidence kept); cap; matched / new objects; priors off; FULL vs B2 on a small room (FULL keeps the bed and the mixed-evidence item). |
| `test_relations_eval.py` | Relation counts on a hand-made pair: subsets a / b / c / c-broad, floor split, GT vs associated universe, new-node parents, several GT parents; aggregation. |
| `test_real_data.py` | *(slow, needs `data/`)* Renders a real dev frame and compares with its depth; val GT counts reproduce the expected raw counts; one dev pair end-to-end with oracle + B0, twice, with identical `metrics`; one dev pair with B1: deterministic, no more false deletions than B0, snapshots valid; one dev pair in perception mode (B1) twice: identical `metrics`, candidate files, valid graphs; B2 on one dev pair in both modes twice (identical `metrics`, valid graphs, relation counts); real CLIP on two crops. |

Run with `uv run pytest -q` (195 tests, ~20 s); the slow ones with `uv run pytest -q -m slow` (10 tests, a few minutes; `test_real_data.py` also runs FULL on one dev pair twice and real DINOv2 on a reference frame).

## 8. What is not built yet

Everything planned is built. Optional items that were not done: the DINOv2 term in association (old B3), a door open / closed diagnostic (Erik: skipped).

Since spec v2 there are two **observation modes**. `perception` (the headline) finds rescan objects from the rescan's own depth and colour. `oracle` (upper bound) uses the rescan's ground-truth objects. Both feed the same change logic.
