# CLAUDE.md — Living Scene Graphs, Subproject D (change detection + graph maintenance)

Source of truth: `docs/DESIGN.md` (spec), `docs/DECISIONS.md` (dated log). If code and spec disagree, stop and ask; don't silently change the spec. Log every non-trivial choice in `DECISIONS.md` (date, decision, why, alternatives).

## Workflow
- Work **one milestone at a time** (M0…M7 in DESIGN §9). A milestone is done only when its acceptance check passes; paste the check's output into the PR/commit message.
- Small commits. No new dependencies without a line in DECISIONS.md.
- Log AI use and compute cost per milestone in `docs/AI_USAGE.md` (what was generated, what was hand-checked, time on GPU).

## Environment
- Python 3.11, **uv**: `uv sync`, run everything with `uv run …`. Tests: `uv run pytest -q` (must pass before every commit; fast tests < 60 s, slow ones marked `@pytest.mark.slow`).
- Data root: env `LSG_DATA` (default `./data`). `data/`, `outputs/`, `cache/` are git-ignored.
- Rendering = Open3D `RaycastingScene` on CPU (≈5 ms per 224×172 depth frame; no GPU, no OpenGL). GPU only for DINOv2 embeddings (`--device cuda`, falls back to CPU).

## Conventions (verified on real data, see DECISIONS 2026-11-05)
- Units: **metres** everywhere in code. Raw depth `.depth.pgm` = uint16 **mm**, 0 = invalid → divide by `m_depthShift` (1000).
- `frame-XXXXXX.pose.txt` = 4×4 **camera→world** (RGB camera), world = that scan's mesh frame.
- 3RScan.json transforms are 16 floats, **column-major** → `np.array(t).reshape(4,4).T`. Translations look like metres (FAQ says mm — **M0 verifies**; never hard-code a ×0.001).
- Reference frame = the **reference scan's** mesh frame. Rescan → reference: `T_align = M(scan["transform"])`; poses: `T_align @ pose`.
- **+Z is up**; floor is at z ≈ −1.4 m, not 0. Heights are always relative to the floor estimate.
- Color 960×540, depth 224×172; same FOV, intrinsics per resolution in `_info.txt` (use `m_calibrationDepthIntrinsic` for depth). Images are stored **rotated 90°** (rotate crops upright only for appearance models).
- Some `rigid` entries in 3RScan.json are bare ints (no transform). Handle both forms.
- 3DSSG relationship tuple = `[subject_id, object_id, rel_id, name]`, subject is the child ("cup standing on table"); ids = `semseg` objectIds; `rel_id` indexes `relationships.txt` with `none` = 0.
- Graph node ids: reference nodes `"r<objectId>"`, new nodes `"n<k>"`. JSON schema in DESIGN §4.

## Hard rules
1. **Rescan annotations** (rescan `semseg`, rescan PLY labels, rescan 3DSSG, `rigid/removed/nonrigid` lists) may be read **only** in `src/lsg/oracle/` and `src/lsg/eval/`. The method sees rescan RGB-D + poses + oracle instances (IDs shuffled with a seeded permutation).
2. Only **free-space evidence** (seeing *through* where a node was) lowers confidence. Occluded / out-of-view / invalid depth = no information.
3. Never tune on `splits/eval_*.txt`. Tune on `splits/dev_train.txt` only; thresholds live in `configs/default.yaml`, not in code.
4. Every metric script writes a JSON to `outputs/<run_id>/metrics.json` with the config hash and git SHA.
5. Don't download or load the whole dataset; use `splits/download_tier*.txt`.
