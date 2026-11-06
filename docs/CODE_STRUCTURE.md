# Code structure — what is where and what it does

State: end of milestone M2 (2026-11-06). Covers everything in `src/lsg/`, `tests/` and `configs/`. For *why* things are done this way, see `DESIGN.md` (the spec) and `DECISIONS.md` (the log); this file only explains *what* the code does.

## 1. The project in one paragraph

3RScan contains rooms that were scanned several times. The first scan of a room is the **reference**; later visits are **rescans**. Between visits, furniture moves, objects disappear and new ones appear. The goal is to take the reference room's scene graph (objects + "cup is on the table" relations), look at a rescan's camera frames, and update the graph: which objects are unchanged, moved, removed, unknown (not seen) or new.

M0 built the foundation: reading the data, drawing ("rendering") a scanned room from any camera position, lining up the rescan with the reference, and checking on real data that all of this is correct. M1 added the reference scene graph (objects + support relations, from annotations or from geometry) and the ground truth of what changed between two visits. M2 added the observation interface (what the method sees of a rescan), the oracle version of it, the first (naive) change detector B0, and the scoring harness `lsg eval`.

## 2. Folder map

```
Task2/
├── configs/default.yaml     all tunable numbers (thresholds, ICP, relation rules, eval definitions, gates)
├── docs/                    DESIGN (spec), DECISIONS (log), STATUS (finished / open items), AI_USAGE, this file
├── splits/                  lists of scan ids / (reference, rescan) pairs to use
├── src/lsg/                 the Python package (run as `uv run lsg …`)
│   ├── cli.py               the `lsg` command
│   ├── config.py            loads the YAML config, finds data/outputs folders
│   ├── runinfo.py           run names, git commit, writes metrics.json
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
│   │   ├── render.py        draws a mesh as seen by a camera (depth + object ids)
│   │   └── align.py         fine-tunes how the rescan lines up with the reference
│   ├── obs/                 OBSERVATION INTERFACE (what the method sees of a rescan)
│   │   └── base.py          ObsInstance, ObsSource, RefContext, SeqInfo, PackedMasks
│   ├── change/              CHANGE LOGIC (method side)
│   │   ├── pipeline.py      frame selection, per-pair preparation, the M2 frame loop
│   │   └── baselines.py     B0: naive same-label matching
│   ├── eval/                CHECKING / SCORING (may read rescan ground truth)
│   │   ├── m0_checks.py     the four M0 acceptance checks
│   │   ├── gt.py            ground truth per pair: moved / removed / new / ...
│   │   ├── gt_stats.py      M1 checks: GT counts, displacement histogram, rule sanity
│   │   ├── metrics.py       scoring: matching rules, per-pair counts, averages + bootstrap CIs
│   │   └── harness.py       `lsg eval` / `lsg run`: method + observations + GT -> metrics.json
│   ├── oracle/              ground-truth rescan objects: the upper-bound mode
│   │   └── source.py        OracleSource
│   └── perception/          (stub; M4) finds rescan objects from depth/colour alone: the headline mode
├── tests/                   automated tests (`uv run pytest -q`)
├── data/      (git-ignored) the downloaded dataset
├── cache/     (git-ignored) refined alignments (cache/align/), later model features
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
| `open_sequence(root, id)` → `SequenceReader` | Access to the camera frames. Reads straight from `sequence.zip`, so nothing has to be unzipped. |
| `SequenceReader.pose/depth/color(i)` | Pose (4x4), depth in metres (0 = no measurement) and color image of frame `i`. Images stay rotated 90°, as stored. |
| `iter_frames(seq, stride)` | Loops over every n-th frame, skipping frames with a broken pose. |
| `read_ply`, `read_pgm16`, `parse_info` | Low-level file decoders (mesh file, depth image, camera settings). |

### `io/dssg.py` — reading 3DSSG
- `load_objects(root, id)`: per object, its label, class ids and attributes.
- `load_relations(root, id)`: relations such as `(12, 5, "standing on")` = object 12 stands on object 5.
- `load_predicates(root)`: the list of relation names.

The big JSON files are parsed only once per run and then reused.

### `geom/transforms.py` — coordinate maths
- `mat4_from_colmajor`: turns the 16 numbers in 3RScan.json into a 4x4 matrix (they are stored column by column).
- `apply(T, points)` moves points with a transform. `invert(T)` gives the reverse transform.
- `backproject(depth, K, pose)`: depth image → 3D points. `project(points, K)` goes the other way (3D → pixel).
- `rotation_angle_deg(T)`: how much a transform rotates.

### `geom/render.py` — drawing a mesh from a camera
- `Renderer(vertices, faces, labels)` prepares a mesh once, then `render(K, T_w2c, W, H)` returns, for a camera, a **depth image** and an **object-id image** (~5 ms per depth-sized frame, CPU only).
- `distance(points)` gives each point's distance to the nearest surface of the mesh (used by the checks).

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
- `Node`, `Edge`, `Graph`: exactly the fields of DESIGN §4. Ids: `r<objectId>` for reference objects, `n<k>` for new ones, `room` for the room. `floor` is only a parent name, not a node.
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
- Counts per split (must reproduce DESIGN §2.2), a histogram of how far objects moved (`outputs/<run>/gt_displacement_hist.png`), and how often the geometric relation rules agree with 3DSSG on the reference scans.

### `obs/base.py` — the observation interface
- `ObsInstance`: one object seen in the rescan. It has an opaque `token`, a label (+ `label_topk`), a class id, its observed 3D points in the **reference** frame, and a mask per frame showing where it was seen.
- `ObsSource`: anything that produces `ObsInstance`s frame by frame: `begin(...)`, then `observe(frame, ...)` for every processed frame in time order, then `finish()` returns the instances. The oracle implements it now; perception (M4) will too. The change logic only ever talks to this interface.
- `RefContext` (the reference scan + its graph) and `SeqInfo` (rescan id, camera parameters, the refined `T_align`, which frames are processed) bundle what the method is allowed to know.
- `PackedMasks` stores the per-frame masks bit-packed (8x smaller) but reads like a dict of boolean images.

### `oracle/source.py` — the upper-bound observations
- `OracleSource` renders the rescan's own **annotated** mesh from each rescan camera. Every object that shows up (where the sensor also has valid depth) becomes an `ObsInstance` with its real label and the mesh vertices that were seen. Objects with fewer than 50 seen vertices are dropped.
- Object ids are replaced by shuffled tokens (`o0`, `o1`, ...) so the method cannot cheat by id; `token_map()` undoes the shuffle and is only used by `eval/`.

### `change/pipeline.py` — preparing a pair and walking its frames
- `select_frames(...)`: every 3rd frame, or one contiguous chunk of them (for the "observe only 50 / 25 / 10%" experiments).
- `prepare_pair(...)`: builds the reference graph and the refined alignment. The refined alignment is cached in `cache/align/`, so ICP runs once per rescan (and chunk).
- `collect_observations(...)`: the simple M2 loop: feed every processed frame to the `ObsSource`, return its instances. M3 replaces it with the full online loop.

### `change/baselines.py` — B0, the naive change detector
- `b0(G_ref, obs, cfg)`: pairs each reference object with the nearest observed object of the **same label** (one-to-one, closest pairs first). Within 0.5 m it is `unchanged`, farther `moved` (and its position is updated); without a partner `removed`; observed objects nobody claimed become `new` nodes. It returns the updated graph.
- It ignores visibility, so anything simply not seen in the rescan is deleted. Later baselines fix exactly that.

### `eval/metrics.py` — scoring
- `match_new(...)`: which predicted new nodes are real new objects. Oracle: by token. Perception: same label (or class) and close (<= 30 cm) or overlapping (box IoU >= 0.25), best overlaps first, one-to-one.
- `score_pair(G_upd, gt, ...)`: counts for one pair: hits / false alarms / misses for moved, removed and new; unchanged right / kept; false deletions and false moves on objects the rescan annotation doesn't mention; how often the method paired an object with the right observation; moved recall per distance and size bucket; how far off the new position of a correctly detected move is.
- `aggregate(...)`: sums the counts over all pairs, turns them into percentages, and adds a 95% confidence interval by resampling pairs (bootstrap). Every number comes with its support `n`.

### `eval/harness.py` — `lsg eval` and `lsg run`
- For every pair: load the GT, prepare the pair, run the oracle over the frames, run the method, score, and save the updated graph to `outputs/<run>/graphs/<mode>/`.
- Pairs of the same reference run in one worker process (4 in parallel by default). Results are sorted back into a fixed order, so the `metrics` block of metrics.json is identical across reruns; timings go into a separate `run` block.

### `cli.py` — the command line
- `uv run lsg m0` runs the M0 checks. Exit code 0 means checks (1)–(3) passed.
- `uv run lsg m1` runs the M1 checks (about 70 s). Exit code 0 means all four passed.
- `uv run lsg splits textures` writes `splits/download_textures.txt`.
- `uv run lsg graph --scan <reference id> --mode gt|geometric [--out file.json]` exports one reference graph and validates it.
- `uv run lsg eval --split eval_val --obs oracle --method B0 [--workers N] [--frac F --seed S]` runs and scores a whole split (~80 s for val); `uv run lsg run --pair <ref> <rescan> --obs oracle --method B0` does one pair. Both print a results table and write `outputs/<run>/metrics.json`. `--obs perception` answers "arrives at M4" for now.
- Later milestones add `lsg ablate`.

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
| `test_real_data.py` | *(slow, needs `data/`)* Renders a real dev frame and compares with its depth; val GT counts reproduce DESIGN §2.2; one dev pair end-to-end with oracle + B0, twice, with identical `metrics`. |

Run with `uv run pytest -q` (81 tests, ~7 s); the slow ones with `uv run pytest -q -m slow` (~1.5 min).

## 8. What is not built yet

Planned in DESIGN §6 and §9 (spec v2), in order:
- M3: the online frame loop with visibility votes, decorrelation and deferred decisions (`change/`);
- M4: perception mode: finding new/moved objects from depth differences and naming them with CLIP (`perception/`);
- M5: object matching, moved test, graph update + relations (`change/`, `graph/`);
- M6: confidence and priors; texture rendering + appearance comparison for thin objects (`perception/appearance`);
- M7: all ablations; M8: visualisation (`viz/`) and write-up.

Since spec v2 there are two **observation modes**. `perception` (the headline) finds rescan objects from the rescan's own depth and colour. `oracle` (upper bound) uses the rescan's ground-truth objects. Both feed the same change logic.
