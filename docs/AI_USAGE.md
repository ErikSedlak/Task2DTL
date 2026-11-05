# AI usage and compute cost

Required by the brief: state where AI tools were used and roughly what runs cost. Add one row per session or milestone.

| Date | Tool | What it did | What I checked by hand | Compute / cost |
|---|---|---|---|---|
| 2026-11-04/05 | Claude (chat agent) | Literature search; data-format verification on local scans (pose convention, depth units, up axis, transform layout, GT statistics); split selection; drafted CLAUDE.md, DESIGN.md, DECISIONS.md | (fill in) | CPU only, a few minutes |
| 2026-11-05 | Claude Code (Opus 5.5) | M0: uv scaffold, loaders (3RScan meta, PLY, zip sequences, 3DSSG), Open3D renderer, label-free ICP alignment refinement, `lsg m0` checks + overlays, tests; diagnosed the `T_align` residual and the `rigid` frame semantics; DESIGN §2.1/§5 updates | Overlay PNGs inspected by Claude (labels on pillows/sofa/walls line up; moved pillows/chair show as red residuals); Erik: (fill in) | CPU only; `lsg m0` ≈ 25 s for 3 dev pairs; all probes < 5 min total |
| | Claude Code | (per milestone) | | |

GPU runs (DINOv2 embeddings, etc.): log device, wall time and number of pairs.
