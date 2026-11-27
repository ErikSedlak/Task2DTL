# Living Scene Graphs — Subproject D: change detection and graph maintenance

Given the scene graph of a **reference** scan of a room (3RScan + 3DSSG) and the RGB-D frames of a later **rescan**,
this code updates the graph object by object. Each reference node gets a state `unchanged | moved | removed | unknown`,
new objects become `new` nodes, support relations ("cup on table") are recomputed, and every node carries a confidence.

- **`perception` mode** (headline): the method sees only the reference data (graph, mesh, texture) and the rescan's
  RGB-D frames, poses and `T_align`. It finds rescan objects itself from depth and appearance residuals and names them with CLIP.
- **`oracle` mode** (upper bound): rescan objects come from the rescan's ground-truth instance segmentation (IDs shuffled).

Only seeing *through* the place where a node was lowers its confidence (for thin objects: an appearance mismatch from several
viewpoints). Unseen nodes stay `unknown`, not `removed`.

Report: [docs/report/report.pdf](docs/report/report.pdf) · Code map: [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)

## 1. Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/). Developed on Windows 10; Linux is untested.

```bash
uv sync            # torch 2.5 (CUDA 12.1 wheels, runs on CPU too), open3d, open_clip, timm, transformers, rerun-sdk
uv run pytest -q   # fast tests, < 60 s, no data needed
```

- A GPU (4 GB is enough) is used only for CLIP ViT-B/32 and DINOv2 ViT-S/14 (and Grounding DINO + SAM 2 for the detector ablation).
  Without one they fall back to CPU. Rendering is CPU-only (Open3D).
- Model weights (~0.7 GB) download from the Hugging Face Hub on first use; afterwards `lsg` runs offline.

## 2. Data

The data root is `./data` or the directory in `LSG_DATA` (PowerShell: `$env:LSG_DATA = "D:\3rscan"`):

```
$LSG_DATA/
├── 3RScan.json              # http://campar.in.tum.de/public_datasets/3RScan/3RScan.json
├── 3DSSG/                   # https://3dssg.github.io: objects.json relationships.json relationships.txt classes.txt
└── <scan id>/               # one folder per scan
```

The 3RScan download script `download_files.py` is **not part of this repository** (it is the authors' script and not covered by
this licence). Get it from <https://github.com/WaldJohannaU/3RScan> after accepting the
[3RScan terms of use](http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf) and save it in the repo root (`.gitignore` lists it).

Demo pair (about 120 MB), a train-split pair from [splits/dev_train.txt](splits/dev_train.txt):

```bash
R=0ad2d39b-79e2-2212-99ae-830c292cd079   # reference
S=0ad2d3a1-79e2-2212-9b99-a96495d9f7fe   # rescan
for id in $R $S; do for t in semseg.v2.json labels.instances.annotated.v2.ply sequence.zip; do
  python download_files.py -o "$LSG_DATA" --id $id --type $t; done; done
for t in mesh.refined.v2.obj mesh.refined.mtl mesh.refined_0.png; do      # texture: reference only
  python download_files.py -o "$LSG_DATA" --id $R --type $t; done
```

The rescan's `semseg` / PLY are read only by the oracle and the evaluation, never by perception mode (`tests/test_no_leak.py`).
Class priors and the CLIP vocabulary come from train-split metadata, so no further downloads are needed.

## 3. Run one pair (both modes)

```bash
uv run lsg run --pair $R $S --obs perception,oracle --method FULL --run-id demo --snapshot-every 2
```

This prints a metrics table per mode (about 6 s per pair once `cache/` is warm; the first run also downloads models) and writes `outputs/demo/`:

| path | content |
|---|---|
| `graphs/<mode>/<ref>_<rescan>.json` | the updated scene graph (schema: `src/lsg/graph/model.py`) |
| `snapshots/<mode>/<ref>_<rescan>/` | tentative graph every 20 processed frames (every 2 with `--snapshot-every 2`) |
| `candidates/<ref>_<rescan>.json` | perception candidates (centroid, box, CLIP top-k) |
| `metrics.json` | all scores, config, config hash and git SHA |

Expected result for the demo pair (24 processed frames): oracle 7 unchanged, 1 moved, 6 unknown; perception 5 unchanged, 9 unknown,
no `new` node (the pair has none), 0 of 5 absent-unlisted objects deleted in both modes.

## 4. Visualize the graph

All figures read a saved run and never change results. Colours: moved blue, removed orange (hatched), new green, unchanged grey, unknown dotted.

| figure | shows |
|---|---|
| top-down before / after | boxes of the reference graph, the updated graph and (`--gt`) the ground truth; `--edges` adds support edges |
| node-edge graph | only the changed part: moved, removed and new nodes plus their parents; arrows point from a thing to its support parent (solid `on`, dashed `in`), same layout before and after |
| Rerun timeline | 3D: reference mesh, camera path, node boxes coloured by tentative state, live candidates and the final graph; scrub the `frame` timeline |
| one frame of the vote | reference render and depth, rescan photo and depth, and the verdict per pixel (seen, seen through, no contrast, occluded, no information) for one object |
| confidence over the walk | confidence of five objects after each processed frame, with the deletion threshold |

```bash
# demo pair (R, S from section 2; needs the run `demo` from section 3)
uv run lsg viz topdown --run demo --pair $R $S --mode perception --gt --edges   # -> outputs/demo/viz/
uv run lsg viz rerun   --run demo --pair $R $S --mode perception --spawn        # -> outputs/demo/viz/*.rrd

# the report's example room (validation pair 8eabc45f -> 5341b7b3; needs its data and outputs/chg3_final)
R=8eabc45f-5af7-2f32-8528-640861d2a135; S=5341b7b3-8a66-2cdd-856d-9d70e194568b
uv run python docs/report/graph_figure.py --run outputs/chg3_final/val/FULL --mode perception     # -> docs/figures/graph_before_after.png
uv run python docs/report/render_compare_figure.py --ref $R --rescan $S --node r5                  # lists frames where r5 is in view
uv run python docs/report/render_compare_figure.py --ref $R --rescan $S --node r5 --frame 117 --out docs/figures/render_compare_seen.png
uv run python docs/report/render_compare_figure.py --examples                                      # seven prepared situations
uv run python docs/report/confidence_figure.py --run outputs/chg3_final/val/FULL                   # -> docs/figures/confidence_walk.png
```

Only `--gt` reads rescan annotations (evaluation side, `src/lsg/eval/figures.py`). The replay figures take about a minute per pair on CPU.
The figures used in the report are in [docs/figures/](docs/figures/).

## 5. Evaluation and experiments

| command | what | time (16-core CPU + GTX 1650, caches warm) |
|---|---|---|
| `uv run lsg eval --split dev_train --obs perception,oracle --method FULL` | one method (`B0 B1 B2 FULL`) on a split | ~35 s |
| `uv run lsg eval --split eval_val_small --obs perception,oracle --method FULL` | 20-pair validation subset | ~1 min |
| `uv run lsg ablate --list` / `--names FULL-appearance,FULL+immediate` | single ablation rows | ~6–7 min per val row |
| `uv run lsg detect cache --split eval_val` | detections for the row `FULL+detector` (once) | ~2 h (GPU) |
| `uv run lsg m7 --run-id chg3_final` | every table, ablation and plot of the report | ≈ 4 h |
| `uv run lsg anytime --split eval_val` | the method at every snapshot: metrics over the walk | ~20 min |
| `uv run lsg audit sample` → label `audit.csv` → `uv run lsg audit summarize` | ground-truth audit | ~10 min + labelling |
| `uv run pytest -q -m slow` | slow tests (real data, models) | ~3 min |

Rules the code keeps: perception never reads rescan annotations; thresholds live in [configs/default.yaml](configs/default.yaml) and were
chosen on the 33 training pairs of `splits/dev_tune.txt`; every metric run writes `outputs/<run_id>/metrics.json` with config hash and
git SHA; the same config and seed give the same metrics.

## 6. Results, by assignment point

All numbers: 110 validation pairs, one run (`outputs/chg3_final`), in %. Full tables with 95% CIs are in the
[report](docs/report/report.pdf) and `outputs/chg3_final/results.md`.

**1. Detect unchanged, moved, removed and new objects** (report §2.1, code `src/lsg/change/`)

| change type | metric | oracle | perception | perception + detector (ablation) |
|---|---|---|---|---|
| removed | precision / recall | 58.5 / 32.9 | 22.3 / 34.2 | 28.9 / 32.9 |
| moved (≥ 10 cm) | precision / recall / F1 | 51.9 / 64.3 / 57.4 | 54.7 / 34.9 / 42.6 | F1 44.9 |
| new | precision / recall | 68.3 / 65.5 | 3.5 / 4.1 | 6.2 / 21.6 |
| unchanged | kept (`unchanged` or `unknown`) | 90.3 | 93.4 | 91.9 |
| absent but not listed as removed | *false deletion* ↓ | 8.7 | 7.3 | 6.9 |

- The annotation is incomplete (1,018 objects missing from the rescan, only 96 listed as removed), so absent-unlisted objects are scored
  separately. The naive baseline B0 deletes 95% of them.
- New objects are the weak spot in perception. The detector (Grounding DINO + SAM 2) lost its tuning-set rule and stays an ablation row (`FULL+detector`).
- Baselines: B0 matches by class and position and deletes everything unmatched, B1 adds free-space votes, B2 adds Hungarian matching and the ICP move test.

**2. Update the graph: move nodes, change anchor and relation, delete, add** (report §2.2, `src/lsg/graph/`, `src/lsg/change/classify.py`)
- Moved nodes get a new box and their support parent is re-checked; removed nodes stay in the file marked `removed`; new nodes get a CLIP label, box and parent.
- Parent accuracy against the rescan's 3DSSG, R0 (keep old relations) / R1 (recompute all) / update (re-check only what changed):
  all things perception 94.9 / 79.8 / 92.2, oracle 91.8 / 79.8 / 92.8; moved things perception 94.0 / 69.6 / 85.7, oracle 94.0 / 66.7 / 86.9.
  The type (`on` / `in`) is right in 95–97% of cases when the parent is right.
- Only 4 of 278 moved things change their parent, so R0 is hard to beat on validation. On the 40-pair relation set (things whose parent changed,
  oracle, n = 14) the update gets 35.7%, R1 42.9%, R0 0%.

**3. Update policy** (report §2.3, `src/lsg/change/confidence.py`, `visibility.py`)
- Confidence starts at p = 0.9. A counted *seen* vote adds 0.63 to the log-odds, a counted *free* vote (the camera saw through the node's place) subtracts 1.47.
- A node is deleted only with free evidence (2 free votes, or 3 appearance mismatches for thin objects) **and** p < 0.4.
- Occluded, out-of-view and invalid-depth pixels give no vote, so such nodes end as `unknown`. Votes count only from viewpoints 0.30 m / 20° apart,
  and `moved` / `removed` are final only at the end of the walk (deciding immediately drops oracle moved F1 from 57 to 24).

**4. Evaluation** — the tables above and in report §3.1–3.2.

**5. Ablation: geometry only vs. geometry + appearance** (report §3.3)

| version | oracle: removed R / false del. / moved F1 | perception: removed R / false del. / moved F1 | perception thin: removed / kept |
|---|---|---|---|
| geometry + appearance (FULL) | 32.9 / 8.7 / 57.4 | 34.2 / 7.3 / 42.6 | 1 of 4 / 91.9 |
| geometry only (depth) | 32.9 / 8.4 / 57.0 | 32.9 / 7.1 / 42.6 | 0 of 4 / 95.4 |

Appearance changes little overall, since only 4 of the 73 scored removals are thin (< 15 cm). It matters for what it was built for:
in a synthetic test a 2 cm book removed from a table is kept by depth alone and deleted with appearance.

**6. Ablation: how much of the room is observed** (rows `FULL_frac<f>_s<seed>`, plot `m7_fraction.png`). FULL on a contiguous chunk of the frames, mean over 3 chunk positions:

| frames | perception moved F1 | perception removed R | perception false del. | perception unknown | oracle moved F1 | oracle new R |
|---|---|---|---|---|---|---|
| 100% | 42.6 | 34.2 | 7.3 | 49.4 | 57.4 | 65.5 |
| 50% | 30.9 | 20.5 | 4.3 | 65.6 | 53.0 | 51.9 |
| 25% | 21.6 | 13.7 | 2.2 | 80.4 | 42.9 | 37.0 |
| 10% | 12.4 | 5.5 | 1.4 | 91.7 | 31.8 | 18.3 |

Seeing less costs recall but not safety: false deletions fall because unseen nodes stay `unknown`. This table is in `outputs/chg3_final/results.md`, not in the report.
Quick version: `uv run lsg ablate --names group:frac --split eval_val_small --run-id frac_small`.

**7. Harder case**
- *Few images of the rescan: done.* It is the experiment of point 6 (10% of the frames); `lsg anytime` also measures the graph after the first k frames.
- *Estimating the poses: not done.* The method uses the given camera poses. It refines the rough rescan→reference alignment itself (online ICP of the accumulated
  depth against the reference mesh, `src/lsg/geom/align.py`); with no refinement perception moved F1 is 41.6 vs 42.6 (within the CIs).

**8. Deliverables:** code in `src/lsg/` (about 270 tests), evaluation in sections 5–6, graph visualization in section 4.

## License and data

- **Code:** [MIT License](LICENSE), © 2026 Erik Sedlák.
- **Data:** 3RScan and 3DSSG are not included and not covered by this licence. 3RScan is
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) with its own terms of use; 3DSSG has its own terms (<https://3dssg.github.io>).
- **Figures:** `docs/figures/` and the report PDF show renderings, photos and annotations of 3RScan scans. Treat them as CC BY-NC-SA 4.0:
  attribute 3RScan and 3DSSG, no commercial use, share alike.
- **Models:** CLIP, DINOv2, Grounding DINO and SAM 2 come from the Hugging Face Hub and keep their own licences.
