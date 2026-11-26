# Living Scene Graphs — Subproject D: change detection and graph maintenance

Given the scene graph of a **reference** scan of a room (3RScan + 3DSSG) and the RGB-D frames of a later
**rescan**, this code updates the graph object by object. Each reference node gets a state:
`unchanged | moved | removed | unknown`. New objects become `new` nodes, and the support relations
("cup on table") are recomputed. Every node carries a confidence and `last_seen`.

- **`perception` mode** (the headline): the method sees only the reference data (graph, mesh, texture) and the
  rescan's RGB-D frames, poses and `T_align`. It finds rescan objects itself, from depth and appearance
  residuals against the rendered reference, and labels them with CLIP.
- **`oracle` mode** (upper bound): rescan objects come from the rescan's GT instance segmentation (IDs shuffled).

Only seeing *through* the place where a node was (or, for thin objects, an appearance mismatch seen from
several viewpoints) lowers its confidence. Unseen nodes stay `unknown`, not `removed`.

- Write-up: [docs/REPORT.md](docs/REPORT.md); slides: [docs/slides.pptx](docs/slides.pptx).
- Spec: [docs/DESIGN.md](docs/DESIGN.md); decision log: [docs/DECISIONS.md](docs/DECISIONS.md).
- Code map: [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).

**Headline results** (full method FULL, 110 validation pairs, one run: `outputs/chg3_final`; 95% bootstrap CIs and
all ablations in [docs/REPORT.md](docs/REPORT.md)):

| mode | removed P / R | false deletion ↓ | unchanged kept | moved F1 | new P / R | new P / R, any label |
|---|---|---|---|---|---|---|
| oracle | 58.5 / 32.9 | 8.7 | 90.3 | 57.4 | 68.3 / 65.5 | 68.3 / 65.5 |
| perception | 22.3 / 34.2 | 7.3 | 93.4 | 42.6 | 3.5 / 4.1 | 5.6 / 6.4 |
| perception + detector (ablation) | 28.9 / 32.9 | 6.9 | 91.9 | 44.9 | 6.2 / 21.6 | 10.1 / 35.1 |

- *False deletion* = share of reference objects that are missing from the rescan annotation but not listed as
  removed, which the method deletes.
- The naive baseline B0 deletes 95% of them (oracle).
- The method is the final one: CHANGE 1, 2 and 3 of `docs/` (relaxed candidates, one confidence policy, moves
  need old-place evidence, measured vote weights, co-segmented candidates, the graph maintained during the walk
  with online alignment). The open-vocabulary detector (Grounding DINO + SAM 2) lost its tuning-set rule and is
  the ablation row `FULL+detector`.

## 1. Setup

- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- Developed and tested on Windows 10. Nothing is Windows-specific, but Linux is untested.

```bash
uv sync                      # installs torch 2.5 (CUDA 12.1 wheels; runs on CPU without a GPU), open3d, open_clip, timm, transformers, rerun-sdk
uv run pytest -q             # fast tests, < 60 s, no data needed
```

**GPU.**
- An NVIDIA GPU (4 GB is enough) is used only for the frozen CLIP ViT-B/32 and DINOv2 ViT-S/14 (and, for the
  detector ablation only, Grounding DINO tiny + SAM 2.1 small in `lsg detect cache`).
- Without one, both fall back to CPU automatically.
- Rendering is CPU-only (Open3D `RaycastingScene`, no OpenGL).

**Model weights** (~0.7 GB) are downloaded from the Hugging Face Hub on first use. Once both are cached, `lsg`
sets `HF_HUB_OFFLINE=1` itself (`models.hub_offline: auto` in [configs/default.yaml](configs/default.yaml)).

## 2. Data for one pair

The data root is `./data`, or the directory in the environment variable `LSG_DATA`:

```bash
export LSG_DATA=/path/to/3rscan          # PowerShell: $env:LSG_DATA = "D:\3rscan"
```

It must contain:

```
$LSG_DATA/
├── 3RScan.json                       # http://campar.in.tum.de/public_datasets/3RScan/3RScan.json
├── 3DSSG/                            # from the 3DSSG project page, https://3dssg.github.io
│   ├── objects.json  relationships.json  relationships.txt  classes.txt
└── <scan id>/                        # one folder per scan (download_files.py)
```

`download_files.py` is the official 3RScan download script. You get it after accepting the 3RScan
terms of use, and it asks you to confirm them on every call.

The demo pair is the reference `0ad2d39b-79e2-2212-99ae-830c292cd079` and its rescan
`0ad2d3a1-79e2-2212-9b99-a96495d9f7fe` (a train-split pair, from [splits/dev_train.txt](splits/dev_train.txt)).
Download it with:

```bash
R=0ad2d39b-79e2-2212-99ae-830c292cd079
S=0ad2d3a1-79e2-2212-9b99-a96495d9f7fe
for id in $R $S; do for t in semseg.v2.json labels.instances.annotated.v2.ply sequence.zip; do
  python download_files.py -o "$LSG_DATA" --id $id --type $t; done; done
for t in mesh.refined.v2.obj mesh.refined.mtl mesh.refined_0.png; do        # texture: reference only
  python download_files.py -o "$LSG_DATA" --id $R --type $t; done
```

- Per scan this is about 50 MB, plus about 17 MB of texture for the reference.
- The rescan's `semseg` / PLY are read **only** by the oracle and the evaluation, never by perception mode
  (`tests/test_no_leak.py`).
- The class priors and the CLIP vocabulary come from train-split *metadata* (3RScan.json + 3DSSG `objects.json`).
  They need no further downloads.

## 3. Run one pair end-to-end (both modes)

```bash
uv run lsg run --pair 0ad2d39b-79e2-2212-99ae-830c292cd079 0ad2d3a1-79e2-2212-9b99-a96495d9f7fe \
    --obs perception,oracle --method FULL --run-id demo
```

This prints a metrics table per mode and writes `outputs/demo/`:

| path | content |
|---|---|
| `graphs/<mode>/<ref>_<rescan>.json` | the updated scene graph G_upd (schema: DESIGN §4) |
| `snapshots/<mode>/<ref>_<rescan>/frame_XXXXXX.json` | tentative graphs during the walk (every 20 processed frames, plus the last) + `seq_info.json`; with `lsg anytime` the full method runs at every snapshot |
| `candidates/<ref>_<rescan>.json` | perception candidates (centroid, box, CLIP top-k) |
| `metrics.json` | all scores, config, config hash and git SHA |

**Expected result** (24 processed frames; the same graphs as the reported dev run `chg3_final/dev/FULL`):

| mode | node states (things + anchors) | scored |
|---|---|---|
| oracle | 7 unchanged, 1 moved, 6 unknown | moved 1 TP / 1 FN, unchanged kept 6 / 6, 0 of 5 absent-unlisted deleted |
| perception | 5 unchanged, 9 unknown | moved 0 TP / 0 FP / 2 FN, no `new` node (this pair has no GT-new object), unchanged kept 6 / 6, 0 of 5 absent-unlisted deleted or moved |

With the M7 method (the overrides of the ablation row `FULL-m7`, `eval/ablations.py` `M7_OVERRIDES`) the graphs are
those of `m7_final/dev/FULL`. Perception runs also write `snapshots/perception/<pair>/candidates_frame_XXXXXX.json`;
`lsg viz rerun` draws them as `world/candidates`, grey → green by confidence.

**Runtime:** about 6 s per pair once `cache/` is warm (DINOv2 similarities, CLIP labels; the alignment is estimated
during the walk). The first run also computes these and downloads the models.

### Look at it

```bash
# before / after top-down figure (+ GT panel, computed on the evaluation side)
uv run lsg viz topdown --run demo --pair <REF> <RESCAN> --gt

# Rerun timeline: reference mesh, camera path, node boxes coloured by tentative state, final graph
uv run lsg run --pair <REF> <RESCAN> --obs perception,oracle --method FULL --run-id demo --snapshot-every 2
uv run lsg viz rerun --run demo --pair <REF> <RESCAN> --mode perception --spawn   # or: uv run rerun outputs/demo/viz/<file>.rrd
```

- Top-down colours: moved blue, removed orange (hatched), new aqua, unchanged grey, unknown dotted.
- `--snapshot-every 2` gives a denser timeline on this short sequence. It only changes the snapshot interval; the
  results stay the same.
- In Rerun, scrub the `frame` timeline.

## 4. Evaluation and experiments

| command | what | time (16-core CPU + GTX 1650, caches warm) |
|---|---|---|
| `uv run lsg eval --split dev_train --obs perception,oracle --method FULL` | one method on a split; methods `B0 B1 B2 FULL` | ~35 s |
| `uv run lsg eval --split eval_val_small --obs perception,oracle --method FULL` | the 20-pair milestone subset of val | ~1 min |
| `uv run lsg ablate --list` / `--names FULL-appearance,FULL+immediate` | single ablation rows (DESIGN §8) | ~6–7 min per val row |
| `uv run lsg detect cache --split eval_val` | Grounding DINO + SAM 2 detections for the row `FULL+detector` (once) | ~2 h (GPU) |
| `uv run lsg m7 --run-id chg3_final` | every results table, ablation and plot of the report (after the detect cache) | ≈ 4 h |
| `uv run lsg anytime --split eval_val` | the method at every snapshot: metrics over the walk, state flips | ~20 min |
| `uv run lsg audit sample` → label `audit.csv` → `uv run lsg audit summarize` | §7.4 GT audit, sampled from `chg3_final/val/FULL` | ~10 min + labelling |
| `uv run pytest -q -m slow` | slow tests (real data, models) | ~3 min |

**Rules the code keeps** (CLAUDE.md):
- perception never reads rescan annotations;
- thresholds live in [configs/default.yaml](configs/default.yaml) and were chosen on the 33 training pairs of
  `splits/dev_tune.txt`, each with a rule written before the run (REPORT §8 lists what val influenced earlier);
- every metric run writes `outputs/<run_id>/metrics.json` with the config hash and git SHA;
- the same config and seed give the same `metrics`.
