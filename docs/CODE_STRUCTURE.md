# Code structure — what is where and what it does

State: end of milestone M0 (2026-11-05). Covers everything in `src/lsg/`, `tests/` and `configs/`. For *why* things are done this way, see `DESIGN.md` (the spec) and `DECISIONS.md` (the log); this file only explains *what* the code does.

## 1. The project in one paragraph

3RScan contains rooms that were scanned several times. The first scan of a room is the **reference**; later visits are **rescans**. Between visits, furniture moves, objects disappear and new ones appear. The goal is to take the reference room's scene graph (objects + "cup is on the table" relations), look at a rescan's camera frames, and update the graph: which objects are unchanged, moved, removed, unknown (not seen) or new.

M0 does not do any of that yet. It builds the foundation: reading the data, drawing ("rendering") a scanned room from any camera position, lining up the rescan with the reference, and checking on real data that all of this is correct.

## 2. Folder map

```
Task2/
├── configs/default.yaml     all tunable numbers (thresholds, ICP settings, M0 gates)
├── docs/                    DESIGN (spec), DECISIONS (log), AI_USAGE, this file
├── splits/                  lists of scan ids / (reference, rescan) pairs to use
├── src/lsg/                 the Python package (run as `uv run lsg …`)
│   ├── cli.py               the `lsg` command
│   ├── config.py            loads the YAML config, finds data/outputs folders
│   ├── runinfo.py           run names, git commit, writes metrics.json
│   ├── classes.py           which object classes are "structure", "anchor", "thing"
│   ├── io/                  READING DATA
│   │   ├── scan3r.py        3RScan: metadata, meshes + labels, camera frames
│   │   └── dssg.py          3DSSG: object attributes and relations
│   ├── geom/                GEOMETRY
│   │   ├── transforms.py    moving points between coordinate frames, depth -> 3D points
│   │   ├── render.py        draws a mesh as seen by a camera (depth + object ids)
│   │   └── align.py         fine-tunes how the rescan lines up with the reference
│   ├── eval/                CHECKING / SCORING (may read rescan ground truth)
│   │   └── m0_checks.py     the four M0 acceptance checks
│   └── oracle/              (empty, M2+) ground-truth rescan objects for the method
├── tests/                   automated tests (`uv run pytest -q`)
├── data/      (git-ignored) the downloaded dataset
└── outputs/   (git-ignored) results of runs: outputs/<run_id>/metrics.json, PNGs
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

### `cli.py` — the command line
- `uv run lsg m0` runs the checks. Exit code 0 means checks (1)–(3) passed.
- Later milestones add more subcommands (`lsg run`, `lsg eval`, `lsg ablate`).

## 6. The one rule that shapes the layout

The method must not peek at the rescan's answers (CLAUDE.md, hard rule 1). Rescan **annotations** are:
- the rescan's object labels (PLY / semseg / 3DSSG);
- the lists of moved, removed and non-rigid objects in 3RScan.json.

These may only be read inside `eval/` (scoring, checks) and `oracle/` (the stand-in object detector, from M2). Everything else reads a rescan only through `open_sequence` (photos, depth, poses). That is why `load_scan` and `SequenceReader` are separate. It is also why `align.py` uses depth only, and why the M0 checks live in `eval/`.

## 7. Tests (`tests/`)

| File | What it checks (all on small synthetic data unless noted) |
|---|---|
| `test_config.py` | Config loads; hash ignores key order; `LSG_DATA` override; metrics.json contents. |
| `test_transforms.py` | Column-major parsing; inverse; depth → 3D → pixel round trip. |
| `test_io.py` | Depth images (big-endian, comments); zip and folder give identical frames; both kinds of `rigid` entries; PLY parsing; object summaries; 3DSSG; split files. |
| `test_render.py` | A flat wall renders at the right depth everywhere; a box in front gets the right label; empty pixels are 0; a posed camera's depth back-projects onto the mesh. |
| `test_align.py` | ICP recovers a known 5 cm / 1° misalignment; oversized corrections are rejected. |
| `test_real_data.py` | *(slow, needs `data/`)* Renders a real dev frame and compares with its depth. |

Run with `uv run pytest -q` (21 tests, ~2 s); the slow one with `uv run pytest -q -m slow`.

## 8. What is not built yet

Planned in DESIGN §6 and §9, in order:
- M1: reference graph + ground-truth change labels (`graph/`, `eval/`);
- M2: oracle instances, baseline B0, scoring (`oracle/`, `eval/`);
- M3: visibility votes (`change/visibility`);
- M4: object matching + moved test (`change/association`, `change/classify`);
- M5: graph update + relations (`change/update`, `graph/relations`);
- M6: confidence and priors (`change/confidence`), appearance;
- M7: visualisation (`viz/`).
