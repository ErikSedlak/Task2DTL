"""`lsg` command line. Subcommands are added per milestone."""

from __future__ import annotations

import argparse
import sys

from lsg.config import load_config


def _cmd_m0(args: argparse.Namespace) -> int:
    """`lsg m0`: run the M0 checks, print the summary.

    Returns:
        Exit code 0 if checks (1)-(3) pass, else 1. Check (4) is visual and never fails the run.
    """
    from lsg.eval.m0_checks import format_summary, run_m0

    res = run_m0(load_config(args.config), run_id=args.run_id)
    print(format_summary(res))
    acc = res["acceptance"]
    ok = all(a["pass"] is True for k, a in acc.items() if k != "4_overlays")
    return 0 if ok else 1


def _cmd_m1(args: argparse.Namespace) -> int:
    """`lsg m1`: GT statistics, displacement histogram, reference-rule sanity, texture list.

    Returns:
        Exit code 0 if the count, histogram and texture-list checks pass, else 1.
    """
    from lsg.eval.gt_stats import format_summary, run_m1

    res = run_m1(load_config(args.config), run_id=args.run_id)
    print(format_summary(res))
    return 0 if all(a["pass"] is True for a in res["acceptance"].values()) else 1


def _cmd_m4(args: argparse.Namespace) -> int:
    """`lsg m4`: perception on dev (B0 / B1, both modes), candidate recall, overlays, sweep.

    Returns:
        Exit code 0 if the overlay check passes (recall is reported, without a threshold).
    """
    from lsg.eval.m4_checks import format_summary, run_m4

    res = run_m4(load_config(args.config), run_id=args.run_id, sweep=not args.no_sweep)
    print(format_summary(res))
    return 0 if res["acceptance"]["overlays_3_dev_frames"]["pass"] else 1


def _cmd_m5(args: argparse.Namespace) -> int:
    """`lsg m5`: B0 / B1 / B2 on dev (both modes), optional sweep, B2 on val with relations.

    Returns:
        Exit code 0 if every acceptance check passes.
    """
    from lsg.eval.m5_checks import format_summary, run_m5

    res = run_m5(load_config(args.config), run_id=args.run_id, sweep=not args.no_sweep, val=not args.no_val,
                 full_val=args.full_val)  # fmt: skip
    print(format_summary(res))
    return 0 if all(a["pass"] for a in res["acceptance"].values()) else 1


def _cmd_m6(args: argparse.Namespace) -> int:
    """`lsg m6`: OBJ/PLY check, texture render check, B2 / FULL variants on dev, sweep, FULL on val.

    Returns:
        Exit code 0 if every acceptance check passes.
    """
    from lsg.eval.m6_checks import format_summary, run_m6

    res = run_m6(load_config(args.config), run_id=args.run_id, sweep=not args.no_sweep, val=not args.no_val,
                 objply=not args.no_objply, full_val=args.full_val)  # fmt: skip
    print(format_summary(res))
    return 0 if all(a["pass"] for a in res["acceptance"].values()) else 1


def _cmd_m7(args: argparse.Namespace) -> int:
    """`lsg m7`: dev checks, all ablations on val + relchange, tables, plots, determinism.

    Returns:
        Exit code 0 if every acceptance check passes.
    """
    from lsg.eval.m7_checks import format_summary, run_m7

    res = run_m7(load_config(args.config), run_id=args.run_id, dev_only=args.dev_only, sweep=not args.no_sweep,
                 frac=not args.no_frac, relchange=not args.no_relchange, determinism=not args.no_determinism,
                 resume=not args.no_resume, workers=args.workers)  # fmt: skip
    print(format_summary(res))
    return 0 if all(a["pass"] for a in res["acceptance"].values()) else 1


def _cmd_ablate(args: argparse.Namespace) -> int:
    """`lsg ablate --names a,b --split S`: run ablation rows (DESIGN §8), one metrics.json each."""
    from lsg.eval.ablations import run_ablation, select
    from lsg.eval.harness import format_summary, split_path
    from lsg.io.scan3r import read_pairs
    from lsg.runinfo import new_run_id

    cfg = load_config(args.config)
    if args.list:
        from lsg.eval.ablations import registry

        for a in registry(cfg).values():
            print(f"{a.name:22s} {a.group:10s} {a.method:5s} {dict(a.overrides) or ''} frac {a.frac:g} seed {a.seed}"
                  f"  {a.desc}")  # fmt: skip
        return 0
    sp = split_path(args.split)
    pairs = read_pairs(sp)
    rid = args.run_id or new_run_id("ablate")
    for ab in select(cfg, args.names):
        r = run_ablation(cfg, ab, pairs, f"{rid}/{ab.name}", sp.stem, _modes(args.obs), args.workers,
                         resume=not args.no_resume)  # fmt: skip
        print(f"### {ab.name}: {ab.desc} ({'reused' if r['reused'] else 'run'})")
        print(format_summary(r))
    return 0


def _cmd_splits_textures(args: argparse.Namespace) -> int:
    """`lsg splits textures`: write the reference scans that need textures (DESIGN §2.3)."""
    from lsg.config import REPO_ROOT
    from lsg.io.splits import write_texture_list

    cfg = load_config(args.config)
    sc = cfg["splits"]
    out = REPO_ROOT / (args.out or sc["texture_list"])
    ids = write_texture_list([REPO_ROOT / f for f in sc["texture_sources"]], out)
    print(f"wrote {len(ids)} reference ids to {out}")
    return 0


def _cmd_splits_subset(args: argparse.Namespace) -> int:
    """`lsg splits subset`: seeded one-rescan-per-room subset of a pair list (milestone val subset)."""
    from lsg.config import REPO_ROOT
    from lsg.io.scan3r import read_pairs
    from lsg.io.splits import subset_pairs, write_pairs

    cfg = load_config(args.config)
    sc = cfg["splits"]["milestone_val"]
    src = REPO_ROOT / (args.split or sc["source"])
    n, seed = int(args.n if args.n is not None else sc["n_refs"]), int(args.seed if args.seed is not None else sc["seed"])
    out = REPO_ROOT / (args.out or sc["file"])
    pairs = subset_pairs(read_pairs(src), n, seed)
    write_pairs(pairs, out, f"{len(pairs)} pairs: {n} references of {src.name}, one rescan each, seed {seed}\n"
                            "written by `lsg splits subset`; milestone val checks only (DECISIONS M6)")  # fmt: skip
    print(f"wrote {len(pairs)} pairs to {out}")
    return 0


def _cmd_splits_devtune(args: argparse.Namespace) -> int:
    """`lsg splits devtune`: write the tuning set and its download list (CHANGE 3 Part 0; metadata only)."""
    from lsg.config import REPO_ROOT, data_root
    from lsg.eval.devtune import download_ids, pair_counts, select_dev_tune, totals
    from lsg.io.dssg import load_objects
    from lsg.io.scan3r import load_meta, read_pairs
    from lsg.io.splits import write_pairs
    from lsg.train_meta import excluded_references

    cfg = load_config(args.config)
    dc = cfg["splits"]["dev_tune"]
    root = data_root(cfg)
    rows = pair_counts(load_meta(root), lambda s: load_objects(root, s),
                       excluded_references([REPO_ROOT / f for f in dc["exclude_splits"]]))  # fmt: skip
    w = dc["weights"]
    heavy, rand = select_dev_tune(rows, int(dc["n_heavy"]), int(dc["n_random"]), int(dc["seed"]), w)
    include = [p for f in dc["include"] for p in read_pairs(REPO_ROOT / f)]
    pairs = [(r.ref, r.rescan) for r in heavy + rand] + [p for p in include if p not in {(r.ref, r.rescan) for r in heavy + rand}]
    print(f"eligible train pairs: {len(rows)} in {len({r.ref for r in rows})} rooms; totals {totals(rows)}")
    print(f"{'part':<7} {'reference':<38} {'rescan':<38} rem rigid new score")
    for part, rs in (("heavy", heavy), ("random", rand)):
        for r in rs:
            print(f"{part:<7} {r.ref:<38} {r.rescan:<38} {r.removed:>3} {r.rigid:>5} {r.new:>3} {r.score(w):>5.1f}")
    th, tr = totals(heavy), totals(rand)
    print(f"heavy {th}\nrandom {tr}\nselected (without include) {totals(heavy + rand)}; include: {len(include)} pairs")
    already = {line.strip() for f in dc["already_downloaded"] for line in (REPO_ROOT / f).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")}  # fmt: skip
    ids = download_ids(pairs, already)
    print(f"{len(pairs)} pairs; {len(ids)} scans to download")
    if args.dry_run:
        return 0
    out = REPO_ROOT / (args.out or dc["file"])
    t = totals(heavy + rand)
    write_pairs(pairs, out, f"{len(pairs)} pairs: {len(heavy)} change-heavy + {len(rand)} random train rooms (seed {dc['seed']}), "
                            f"one rescan each, + {len(include)} include pairs ({', '.join(dc['include'])})\n"
                            f"metadata counts of the {len(heavy) + len(rand)} selected pairs (things/anchors): removed {t['removed']}, "
                            f"rigid {t['rigid']} (incl. < 10 cm), new {t['new']}\n"
                            "written by `lsg splits devtune`; tuning only (CLAUDE.md hard rule 4, CHANGE 3)")  # fmt: skip
    dl = REPO_ROOT / dc["download_list"]
    dl.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
    print(f"wrote {out} and {dl}\nnext: `uv run lsg splits textures`, then download (DESIGN §2.3 commands with {dl.name} "
          f"and {cfg['splits']['texture_list']})")  # fmt: skip
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    """`lsg graph --scan R --mode gt|geometric`: export one reference graph as JSON."""
    from lsg.config import data_root
    from lsg.graph.build import build_reference_graph
    from lsg.graph.model import validate
    from lsg.io.dssg import load_objects, load_relations
    from lsg.io.scan3r import load_meta, load_scan

    cfg = load_config(args.config)
    root = data_root(cfg)
    if not load_meta(root)[args.scan].is_reference:
        print(f"{args.scan} is not a reference scan (hard rule 1: no rescan annotations here)")
        return 1
    scan = load_scan(root, args.scan, with_intrinsics=False)
    rels = load_relations(root, args.scan) if args.mode == "gt" else None
    g = build_reference_graph(scan, load_objects(root, args.scan), rels, args.mode, cfg)
    problems = validate(g)
    out = args.out or f"{args.scan}_{args.mode}.graph.json"
    g.save(out)
    n_sup = sum(e.type in ("on", "in") for e in g.edges)
    print(f"wrote {out}: {len(g.nodes)} nodes, {len(g.edges)} edges ({n_sup} support)")
    for p in problems:
        print(f"  INVALID: {p}")
    return 0 if not problems else 1


def _run_dir(run: str):
    """`--run`: a directory, or a run id under the outputs root."""
    from pathlib import Path

    from lsg.config import outputs_root

    p = Path(run)
    return p if p.is_dir() else outputs_root(load_config(None)) / run


def _cmd_viz_topdown(args: argparse.Namespace) -> int:
    """`lsg viz topdown --run DIR --pair R S --mode M [--gt]`: before/after PNG of one pair (M8)."""
    from lsg.eval.figures import topdown_for_pair

    cfg = load_config(args.config)
    run = _run_dir(args.run)
    ref, s = args.pair
    for mode in _modes(args.mode):
        g = run / "graphs" / mode / f"{ref}_{s}.json"
        if not g.is_file():
            print(f"no graph {g}")
            return 1
        out = args.out or run / "viz" / f"{ref}_{s}_{mode}_topdown.png"
        p = topdown_for_pair(cfg, g, ref, s, out, with_gt=args.gt, edges=args.edges)
        print(f"wrote {p}")
    return 0


def _cmd_viz_rerun(args: argparse.Namespace) -> int:
    """`lsg viz rerun --run DIR --pair R S --mode M`: Rerun recording of the online walk (M8)."""
    from lsg.config import data_root
    from lsg.viz.rerun_timeline import rerun_timeline

    cfg = load_config(args.config)
    run = _run_dir(args.run)
    ref, s = args.pair
    for mode in _modes(args.mode):
        snap = run / "snapshots" / mode / f"{ref}_{s}"
        if not (snap / "seq_info.json").is_file():
            print(f"no snapshots with seq_info.json in {snap} (rerun `lsg run` with snapshots)")
            return 1
        g = run / "graphs" / mode / f"{ref}_{s}.json"
        out = args.out or run / "viz" / f"{ref}_{s}_{mode}.rrd"
        p = rerun_timeline(data_root(cfg), snap, out, graph_final=g if g.is_file() else None, images=args.images,
                           spawn=args.spawn)  # fmt: skip
        print(f"wrote {p}" + ("" if args.spawn else f"  (view: uv run rerun {p})"))
    return 0


def _cmd_newcurve(args: argparse.Namespace) -> int:
    """`lsg newcurve`: new P / R over the candidate-confidence threshold (CHANGE 2)."""
    from pathlib import Path

    from lsg.config import outputs_root
    from lsg.eval.newcurve import format_curve, new_curve

    cfg = load_config(args.config)
    run = Path(args.run)
    if not run.is_dir():
        run = outputs_root(cfg) / args.run
    ths = [float(t) for t in args.thresholds.split(",")] if args.thresholds else None
    print(format_curve(new_curve(cfg, run, ths)))
    return 0


def _cmd_sam_cache(args: argparse.Namespace) -> int:
    """`lsg sam cache --split <split>`: SAM label maps of every processed rescan frame (CHANGE 3 Part N)."""
    import time
    from pathlib import Path

    from lsg.change.pipeline import select_frames
    from lsg.config import REPO_ROOT, data_root
    from lsg.io.scan3r import open_sequence, read_pairs
    from lsg.perception.sam_cache import SamRunner, cache_key, cache_rescan

    cfg = load_config(args.config)
    root = data_root(cfg)
    sp = Path(args.split) if Path(args.split).suffix else REPO_ROOT / "splits" / f"{args.split}.txt"
    pairs = read_pairs(sp)[: args.pairs] if args.pairs else read_pairs(sp)
    runner = SamRunner(cfg, device=args.device)
    print(f"SAM cache key {cache_key(cfg)}; {len(pairs)} rescans of {sp.name}", flush=True)
    t0, n = time.time(), 0
    for k, (_, s) in enumerate(pairs):
        with open_sequence(root, s) as seq:
            fids = select_frames(seq.frame_ids, int(cfg["frames"]["stride"]))
        r = cache_rescan(cfg, root, s, fids, runner)
        n += r["done"]
        rate = (time.time() - t0) / max(n, 1)
        print(f"[{k + 1}/{len(pairs)}] {s}: {r['done']} new, {r['skipped']} cached, {r['s']} s "
              f"({rate:.2f} s/frame, device {runner.device})", flush=True)  # fmt: skip
    return 0


def _cmd_detect_cache(args: argparse.Namespace) -> int:
    """`lsg detect cache --split <split>`: Grounding DINO + SAM 2 detections of every 2nd processed frame (CHANGE 3 Part E)."""
    import time
    from pathlib import Path

    from lsg.config import REPO_ROOT, data_root
    from lsg.eval.harness import pair_prompts
    from lsg.io.scan3r import open_sequence, read_pairs
    from lsg.perception.detect_cache import DetectorRunner, cache_key, cache_rescan, detect_frames

    cfg = load_config(args.config)
    root = data_root(cfg)
    d = cfg["perception"]["detector"]
    pairs = []
    for sp_name in args.split.split(","):
        sp = Path(sp_name) if Path(sp_name).suffix else REPO_ROOT / "splits" / f"{sp_name}.txt"
        pairs += read_pairs(sp)
    pairs = pairs[: args.pairs] if args.pairs else pairs
    runner = DetectorRunner(cfg, device=args.device)
    print(f"detect cache key {cache_key(cfg)}; {len(pairs)} rescans of {args.split}", flush=True)
    t0, n = time.time(), 0
    prompts_of: dict[str, list[str]] = {}
    for k, (ref, s) in enumerate(pairs):
        if ref not in prompts_of:
            prompts_of[ref] = pair_prompts(cfg, root, ref)
        with open_sequence(root, s) as seq:
            fids = detect_frames(seq.frame_ids, int(cfg["frames"]["stride"]), int(d.get("frame_every", 2)))
        r = cache_rescan(cfg, root, s, prompts_of[ref], fids, runner)
        n += r["done"]
        rate = (time.time() - t0) / max(n, 1)
        print(f"[{k + 1}/{len(pairs)}] {s}: {r['done']} new, {r['skipped']} cached, {r['n_det']} detections, "
              f"{r['n_prompts']} prompts, {r['s']} s ({rate:.2f} s/frame, {runner.device}, {runner.sam_kind})",
              flush=True)  # fmt: skip
    return 0


def _cmd_anytime(args: argparse.Namespace) -> int:
    """`lsg anytime`: FULL with snapshot method runs; anytime curve + stability (CHANGE 3 Part B)."""
    from lsg.eval.anytime import format_anytime
    from lsg.eval.harness import run_split

    cfg = load_config(args.config)
    res = run_split(cfg, args.split, obs_modes=args.modes.split(","), method=args.method, run_id=args.run_id,
                    workers=args.workers, snapshots=True)  # fmt: skip
    print(format_anytime(res["anytime"]))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_movediag(args: argparse.Namespace) -> int:
    """`lsg movediag --run <dir>`: predicted moves by old-place evidence (CHANGE 3 Part A)."""
    import json
    from pathlib import Path

    from lsg.config import outputs_root
    from lsg.eval.movediag import format_breakdown, move_breakdown

    cfg = load_config(args.config)
    run = Path(args.run)
    if not run.is_dir():
        run = outputs_root(cfg) / args.run
    res = move_breakdown(run, cfg)
    if not res:
        print(f"no details/ in {run} (run with details=True)")
        return 1
    print(format_breakdown(res))
    if args.json:
        (run / "movediag.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def _cmd_chg3_parta(args: argparse.Namespace) -> int:
    """`lsg chg3 parta`: CHANGE 3 Part A variants on dev_tune (both modes) + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_a, run_part_a

    res = run_part_a(load_config(args.config), run_id=args.run_id, split=args.split, workers=args.workers)
    print(format_part_a(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_pdel(args: argparse.Namespace) -> int:
    """`lsg chg3 pdel`: p_del sweep on dev_tune at the learned weights + the pre-written selection (Part C)."""
    from lsg.eval.chg3_checks import format_part_c_pdel, run_part_c_pdel

    res = run_part_c_pdel(load_config(args.config), run_id=args.run_id, split=args.split, workers=args.workers)
    print(format_part_c_pdel(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_votestats(args: argparse.Namespace) -> int:
    """`lsg chg3 votestats --run <dir>`: vote weights as likelihood ratios from a details run (CHANGE 3 Part C)."""
    from pathlib import Path

    from lsg.config import outputs_root
    from lsg.eval.votestats import candidate_vote_weights, format_candidate_weights, format_weights, node_vote_weights
    from lsg.runinfo import write_metrics

    cfg = load_config(args.config)
    run = Path(args.run)
    if not run.is_dir():
        run = outputs_root(cfg) / args.run
    res = {m: node_vote_weights(run, cfg, m) for m in args.modes.split(",")}
    for r in res.values():
        print(format_weights(r))
    cand = candidate_vote_weights(run, cfg, "perception") if (run / "details" / "perception").is_dir() else None
    if cand is not None and cand["units"]["present"] + cand["units"]["gone"]:
        print(format_candidate_weights(cand))
    print(f"metrics: {write_metrics(run / 'votestats', {'run': str(run), 'nodes': res, 'candidates': cand}, cfg)}")
    return 0


def _cmd_chg3_partn_seg(args: argparse.Namespace) -> int:
    """`lsg chg3 partn-seg`: CHANGE 3 Part N (N1) blob-splitting variants on dev_tune + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_n_segment, run_part_n_segment

    only = args.only.split(",") if args.only else None
    res = run_part_n_segment(load_config(args.config), args.run_id, args.split, args.workers, only)
    print(format_part_n_segment(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_partn_ref(args: argparse.Namespace) -> int:
    """`lsg chg3 partn-ref --seg <N1 variant>`: CHANGE 3 Part N (N2) explain-away variants + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_n_segment, run_part_n_explain

    res = run_part_n_explain(load_config(args.config), args.seg, args.run_id, args.split, args.workers)
    print(format_part_n_segment(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_partn_noise(args: argparse.Namespace) -> int:
    """`lsg chg3 partn-noise`: CHANGE 3 Part N (N5) noise-rejection variants + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_n_noise, run_part_n_noise

    res = run_part_n_noise(load_config(args.config), args.run_id, args.split, args.workers)
    print(format_part_n_noise(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_partb_align(args: argparse.Namespace) -> int:
    """`lsg chg3 partb-align`: CHANGE 3 Part B (B3) online-alignment variants + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_b_align, run_part_b_align

    res = run_part_b_align(load_config(args.config), args.run_id, args.split, args.workers)
    print(format_part_b_align(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_chg3_parte(args: argparse.Namespace) -> int:
    """`lsg chg3 parte`: CHANGE 3 Part E detector thresholds + variants on dev_tune + the pre-written selection."""
    from lsg.eval.chg3_checks import format_part_e, run_part_e

    res = run_part_e(load_config(args.config), args.run_id, args.split, args.workers)
    print(format_part_e(res))
    print(f"metrics: {res['metrics_path']}")
    return 0


def _cmd_newfunnel(args: argparse.Namespace) -> int:
    """`lsg newfunnel --run <dir>`: where GT new objects are lost in perception (needs details/ + candidates/)."""
    from pathlib import Path

    from lsg.config import outputs_root
    from lsg.eval.newfunnel import format_funnel, new_funnel

    run = Path(args.run) if Path(args.run).is_dir() else outputs_root(load_config(args.config)) / args.run
    print(format_funnel(new_funnel(run)))
    return 0


def _cmd_chg3_idstats(args: argparse.Namespace) -> int:
    """`lsg chg3 idstats --run <dir>`: identity-check similarity of same / different pairs (CHANGE 3 Part N)."""
    from pathlib import Path

    from lsg.config import outputs_root
    from lsg.eval.identitystats import format_stats, identity_stats
    from lsg.runinfo import write_metrics

    cfg = load_config(args.config)
    run = Path(args.run)
    if not run.is_dir():
        run = outputs_root(cfg) / args.run
    res = identity_stats(run, tuple(args.modes.split(",")))
    print(format_stats(res))
    print(f"metrics: {write_metrics(run / 'idstats', {'run': str(run), **res}, cfg)}")
    return 0


def _cmd_audit_sample(args: argparse.Namespace) -> int:
    """`lsg audit sample`: FULL on val with per-object details, FP pools, seeded sample, panels, CSVs (§7.4)."""
    from lsg.eval.audit import run_audit_sample

    from_run = None if str(args.from_run).lower() == "none" else args.from_run
    res = run_audit_sample(load_config(args.config), args.run_id, split=args.split, from_run=from_run,
                           compare=None if args.no_compare else args.compare, b2_run=args.b2_run,
                           n_per_cat=args.n, seed=args.seed, workers=args.workers, max_pairs=args.max_pairs,
                           m7_method=args.m7_method)  # fmt: skip
    for k, v in res["pool_sizes"].items():
        print(f"  pool {k:22s} {v}")
    c = res["consistency"]
    if c:
        print(f"  metrics identical to {c['compared_to']}: {c['identical']}")
    print(f"wrote {res['n_rows']} sampled rows + {res['n_veto']} veto rows -> {res['out_dir']}/audit.csv, "
          f"audit_veto.csv, panels/  (answer the question in the 'answer' column: yes | no | unsure)")
    return 0 if not c or c["identical"] else 1


def _cmd_audit_summarize(args: argparse.Namespace) -> int:
    """`lsg audit summarize`: share of GT gaps among the labelled false positives."""
    from lsg.eval.audit import run_audit_summarize

    res = run_audit_summarize(load_config(args.config), args.run_id)
    print(res["markdown"])
    return 0


def _modes(s: str) -> list[str]:
    """`"perception,oracle"` -> `["perception", "oracle"]`."""
    return [m.strip() for m in s.split(",") if m.strip()]


def _cmd_eval(args: argparse.Namespace) -> int:
    """`lsg eval --split S --obs oracle --method B0`: run + score a split, write metrics.json."""
    from lsg.eval.harness import format_summary, run_split

    try:
        res = run_split(load_config(args.config), args.split, obs_modes=_modes(args.obs), method=args.method,
                        frac=args.frac, seed=args.seed, workers=args.workers, run_id=args.run_id,
                        snapshots=args.snapshots)  # fmt: skip
    except NotImplementedError as e:
        print(f"not available yet: {e}")
        return 2
    print(format_summary(res))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """`lsg run --pair R S --obs oracle --method B0`: one pair, same outputs as `lsg eval`."""
    from lsg.eval.harness import format_summary, run_eval

    cfg = load_config(args.config)
    if args.snapshot_every is not None:  # snapshot interval only; votes and states do not depend on it
        cfg["online"]["snapshot_every"] = int(args.snapshot_every)
    try:
        res = run_eval(cfg, [tuple(args.pair)], obs_modes=_modes(args.obs),
                       method=args.method, frac=args.frac, seed=args.seed, workers=1,
                       run_id=args.run_id, split_name=f"pair {args.pair[0]} {args.pair[1]}",
                       snapshots=not args.no_snapshots)  # fmt: skip
    except NotImplementedError as e:
        print(f"not available yet: {e}")
        return 2
    print(format_summary(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `lsg` command (see `[project.scripts]` in pyproject.toml).

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code of the chosen subcommand.
    """
    p = argparse.ArgumentParser(prog="lsg")
    sub = p.add_subparsers(dest="cmd", required=True)
    m0 = sub.add_parser("m0", help="M0 acceptance checks on the dev pairs")
    m0.add_argument("--config", default=None)
    m0.add_argument("--run-id", default=None)
    m0.set_defaults(func=_cmd_m0)

    m1 = sub.add_parser("m1", help="M1: GT statistics, displacement histogram, rule sanity")
    m1.add_argument("--config", default=None)
    m1.add_argument("--run-id", default=None)
    m1.set_defaults(func=_cmd_m1)

    m4 = sub.add_parser("m4", help="M4: perception on dev, candidate recall, overlays")
    m4.add_argument("--config", default=None)
    m4.add_argument("--run-id", default=None)
    m4.add_argument("--no-sweep", action="store_true", help="skip the door-filter / min_voxels sweep")
    m4.set_defaults(func=_cmd_m4)

    m5 = sub.add_parser("m5", help="M5: B0/B1/B2 on dev, B2 on val with the relation table")
    m5.add_argument("--config", default=None)
    m5.add_argument("--run-id", default=None)
    m5.add_argument("--no-sweep", action="store_true", help="skip the tau_assoc x w_p dev sweep")
    m5.add_argument("--no-val", action="store_true", help="skip the val run (dev checks only)")
    m5.add_argument("--full-val", action="store_true", help="all of eval_val instead of the milestone subset")
    m5.set_defaults(func=_cmd_m5)
    m6 = sub.add_parser("m6", help="M6: texture render check, FULL (confidence + appearance) on dev and val")
    m6.add_argument("--config", default=None)
    m6.add_argument("--run-id", default=None)
    m6.add_argument("--no-sweep", action="store_true", help="skip the L_hit dev sweep")
    m6.add_argument("--no-val", action="store_true", help="skip the val run (dev checks only)")
    m6.add_argument("--full-val", action="store_true", help="all of eval_val instead of the milestone subset")
    m6.add_argument("--no-objply", action="store_true", help="skip the OBJ/PLY check over the 72 references")
    m6.set_defaults(func=_cmd_m6)

    m7 = sub.add_parser("m7", help="M7: all ablations on val + relchange (both modes), results tables, plots")
    m7.add_argument("--config", default=None)
    m7.add_argument("--run-id", default=None, help="fixed id (e.g. m7_final) lets a rerun resume finished rows")
    m7.add_argument("--dev-only", action="store_true", help="dev rows + regressions only (no val)")
    m7.add_argument("--no-sweep", action="store_true", help="skip the tau_free sweep")
    m7.add_argument("--no-frac", action="store_true", help="skip the observed-fraction runs")
    m7.add_argument("--no-relchange", action="store_true", help="skip the relchange relation runs")
    m7.add_argument("--no-determinism", action="store_true", help="skip the two-run determinism check")
    m7.add_argument("--no-resume", action="store_true", help="rerun rows even if a matching metrics.json exists")
    m7.add_argument("--workers", type=int, default=None, help="processes per val run (default: config)")
    m7.set_defaults(func=_cmd_m7)

    ab = sub.add_parser("ablate", help="run ablation rows (DESIGN §8) on a split")
    ab.add_argument("--names", default="FULL", help="comma list, 'all', or 'group:<method|component|variant|tau_free|frac>'")
    ab.add_argument("--split", default="eval_val", help="split name or file")
    ab.add_argument("--obs", default="perception,oracle")
    ab.add_argument("--workers", type=int, default=None)
    ab.add_argument("--run-id", default=None)
    ab.add_argument("--no-resume", action="store_true")
    ab.add_argument("--list", action="store_true", help="list the registry and exit")
    ab.add_argument("--config", default=None)
    ab.set_defaults(func=_cmd_ablate)

    nc = sub.add_parser("newcurve", help="new-object P / R over p_new_min from a run's saved perception graphs (CHANGE 2)")
    nc.add_argument("--run", required=True, help="run directory (or id under outputs/) with graphs/perception/")
    nc.add_argument("--thresholds", default=None, help="comma list (default: p_new_min + m7.p_new_sweep)")
    nc.add_argument("--config", default=None)
    nc.set_defaults(func=_cmd_newcurve)

    c3 = sub.add_parser("chg3", help="CHANGE 3 dev_tune sweeps")
    c3s = c3.add_subparsers(dest="chg3_cmd", required=True)
    pa = c3s.add_parser("parta", help="Part A (move needs old-place evidence): variants + selection")
    pa.add_argument("--run-id", default="chg3_partA_sweep")
    pa.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pa.add_argument("--workers", type=int, default=None)
    pa.add_argument("--config", default=None)
    pa.set_defaults(func=_cmd_chg3_parta)
    pd = c3s.add_parser("pdel", help="Part C: p_del sweep at the configured vote weights + selection")
    pd.add_argument("--run-id", default="chg3_partC_pdel")
    pd.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pd.add_argument("--workers", type=int, default=None)
    pd.add_argument("--config", default=None)
    pd.set_defaults(func=_cmd_chg3_pdel)
    vs = c3s.add_parser("votestats", help="Part C: vote weights (likelihood ratios) from a run's details/")
    vs.add_argument("--run", required=True, help="run directory (or id under outputs/) with details/<mode>/")
    vs.add_argument("--modes", default="oracle,perception")
    vs.add_argument("--config", default=None)
    vs.set_defaults(func=_cmd_chg3_votestats)

    sm = sub.add_parser("sam", help="SAM segment cache for co-segmentation (CHANGE 3 Part N)")
    sms = sm.add_subparsers(dest="sam_cmd", required=True)
    smc = sms.add_parser("cache", help="SAM label maps of every processed frame of a split's rescans")
    smc.add_argument("--split", default="dev_tune", help="split name under splits/ or a pair file")
    smc.add_argument("--pairs", type=int, default=None, help="only the first N pairs (timing)")
    smc.add_argument("--device", default="cuda")
    smc.add_argument("--config", default=None)
    smc.set_defaults(func=_cmd_sam_cache)

    nf = sub.add_parser("newfunnel", help="where GT new objects are lost in perception (run with details)")
    nf.add_argument("--run", required=True, help="run dir or id under outputs/ (with details/ and candidates/)")
    nf.add_argument("--config", default=None)
    nf.set_defaults(func=_cmd_newfunnel)

    dt = sub.add_parser("detect", help="open-vocabulary detection cache: Grounding DINO + SAM 2 (CHANGE 3 Part E)")
    dts = dt.add_subparsers(dest="detect_cmd", required=True)
    dtc = dts.add_parser("cache", help="detections of every 2nd processed frame of a split's rescans")
    dtc.add_argument("--split", default="dev_tune", help="split name(s) under splits/ (comma list) or a pair file")
    dtc.add_argument("--pairs", type=int, default=None, help="only the first N pairs (timing)")
    dtc.add_argument("--device", default="cuda")
    dtc.add_argument("--config", default=None)
    dtc.set_defaults(func=_cmd_detect_cache)

    pns = c3s.add_parser("partn-seg", help="Part N (N1): blob-splitting variants + selection (resumable)")
    pns.add_argument("--run-id", default="chg3_partN_seg")
    pns.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pns.add_argument("--only", default=None, help="comma list of variants to run now (others read back if present)")
    pns.add_argument("--workers", type=int, default=None)
    pns.add_argument("--config", default=None)
    pns.set_defaults(func=_cmd_chg3_partn_seg)
    pnr = c3s.add_parser("partn-ref", help="Part N (N2): reference explain-away variants on an N1 variant + selection")
    pnr.add_argument("--seg", required=True, help="N1 variant name (chg3.part_n.segment.variants)")
    pnr.add_argument("--run-id", default="chg3_partN_ref")
    pnr.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pnr.add_argument("--workers", type=int, default=None)
    pnr.add_argument("--config", default=None)
    pnr.set_defaults(func=_cmd_chg3_partn_ref)
    pnn = c3s.add_parser("partn-noise", help="Part N (N5): noise rejection / thin-object confidence variants + selection")
    pnn.add_argument("--run-id", default="chg3_partN_noise")
    pnn.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pnn.add_argument("--workers", type=int, default=None)
    pnn.add_argument("--config", default=None)
    pnn.set_defaults(func=_cmd_chg3_partn_noise)
    pba = c3s.add_parser("partb-align", help="Part B (B3): offline vs online alignment variants + selection")
    pba.add_argument("--run-id", default="chg3_partB_align")
    pba.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pba.add_argument("--workers", type=int, default=None)
    pba.add_argument("--config", default=None)
    pba.set_defaults(func=_cmd_chg3_partb_align)
    pe_ = c3s.add_parser("parte", help="Part E: detector thresholds (step 1) + variants (step 2) + selection (resumable)")
    pe_.add_argument("--run-id", default="chg3_partE")
    pe_.add_argument("--split", default="dev_tune", help="tuning split only (hard rule 4)")
    pe_.add_argument("--workers", type=int, default=None)
    pe_.add_argument("--config", default=None)
    pe_.set_defaults(func=_cmd_chg3_parte)
    ids = c3s.add_parser("idstats", help="Part N: identity-check similarity of same / different pairs from details/")
    ids.add_argument("--run", required=True, help="run directory (or id under outputs/) with details/<mode>/")
    ids.add_argument("--modes", default="oracle,perception")
    ids.add_argument("--config", default=None)
    ids.set_defaults(func=_cmd_chg3_idstats)

    at = sub.add_parser("anytime", help="method runs on every snapshot during the walk; anytime curve (CHANGE 3 Part B)")
    at.add_argument("--split", default="dev_tune")
    at.add_argument("--modes", default="oracle,perception")
    at.add_argument("--method", default="FULL")
    at.add_argument("--run-id", default=None)
    at.add_argument("--workers", type=int, default=None)
    at.add_argument("--config", default=None)
    at.set_defaults(func=_cmd_anytime)

    md = sub.add_parser("movediag", help="predicted moves by old-place evidence from a run's details/ (CHANGE 3)")
    md.add_argument("--run", required=True, help="run directory (or id under outputs/) with details/<mode>/")
    md.add_argument("--json", action="store_true", help="also write <run>/movediag.json")
    md.add_argument("--config", default=None)
    md.set_defaults(func=_cmd_movediag)

    sp = sub.add_parser("splits", help="split-file helpers")
    sps = sp.add_subparsers(dest="splits_cmd", required=True)
    tx = sps.add_parser("textures", help="write splits/download_textures.txt")
    tx.add_argument("--config", default=None)
    tx.add_argument("--out", default=None, help="relative to the repo root")
    tx.set_defaults(func=_cmd_splits_textures)
    ss = sps.add_parser("subset", help="write the seeded milestone val subset (splits.milestone_val)")
    ss.add_argument("--config", default=None)
    ss.add_argument("--split", default=None, help="source pair list (default splits.milestone_val.source)")
    ss.add_argument("--n", type=int, default=None, help="number of references (one rescan each)")
    ss.add_argument("--seed", type=int, default=None)
    ss.add_argument("--out", default=None, help="relative to the repo root")
    ss.set_defaults(func=_cmd_splits_subset)
    dt = sps.add_parser("devtune", help="write the tuning set splits/dev_tune.txt + download list (splits.dev_tune)")
    dt.add_argument("--config", default=None)
    dt.add_argument("--out", default=None, help="pair list, relative to the repo root (default splits.dev_tune.file)")
    dt.add_argument("--dry-run", action="store_true", help="print the selection, write nothing")
    dt.set_defaults(func=_cmd_splits_devtune)

    gr = sub.add_parser("graph", help="export a reference scene graph as JSON")
    gr.add_argument("--scan", required=True, help="reference scan id")
    gr.add_argument("--mode", choices=["gt", "geometric"], default="gt")
    gr.add_argument("--out", default=None)
    gr.add_argument("--config", default=None)
    gr.set_defaults(func=_cmd_graph)

    au = sub.add_parser("audit", help="§7.4 GT audit: sample false positives, summarise verdicts (M8)")
    aus = au.add_subparsers(dest="audit_cmd", required=True)
    a1 = aus.add_parser("sample", help="FULL on val with details, FP pools, sample, panels, CSVs")
    a1.add_argument("--run-id", default="chg3_audit")
    a1.add_argument("--split", default="eval_val")
    a1.add_argument("--from-run", default="chg3_final/val/FULL",
                    help="run with details/ to sample from (the final `lsg m7` FULL row); 'none' = run FULL here")
    a1.add_argument("--compare", default=None, help="run whose metrics a fresh FULL run must reproduce")
    a1.add_argument("--no-compare", action="store_true")
    a1.add_argument("--m7-method", action="store_true",
                    help="audit the M7 FULL (CHANGE 1-3 off; use --from-run none --compare m7_final/val/FULL)")
    a1.add_argument("--b2-run", default="chg3_final/val/B2", help="run with B2 graphs (veto set)")
    a1.add_argument("--n", type=int, default=10, help="sample size per mode and category")
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--workers", type=int, default=None)
    a1.add_argument("--max-pairs", type=int, default=None, help="first N pairs only (dry run, no compare)")
    a1.add_argument("--config", default=None)
    a1.set_defaults(func=_cmd_audit_sample)
    a2 = aus.add_parser("summarize", help="verdict shares from the labelled audit.csv / audit_veto.csv")
    a2.add_argument("--run-id", default="chg3_audit")
    a2.add_argument("--config", default=None)
    a2.set_defaults(func=_cmd_audit_summarize)

    vz = sub.add_parser("viz", help="figures: top-down before/after, Rerun timeline (M8)")
    vzs = vz.add_subparsers(dest="viz_cmd", required=True)
    for vname, vhelp in (("topdown", "before/after PNG of one pair"), ("rerun", "Rerun .rrd of the online walk")):
        v = vzs.add_parser(vname, help=vhelp)
        v.add_argument("--run", required=True, help="run directory or run id (e.g. outputs/m7_final/val/FULL)")
        v.add_argument("--pair", nargs=2, required=True, metavar=("REF", "RESCAN"))
        v.add_argument("--mode", default="perception,oracle", help="comma list of oracle,perception")
        v.add_argument("--out", default=None, help="output file (one mode only)")
        v.add_argument("--config", default=None)
        if vname == "topdown":
            v.add_argument("--gt", action="store_true", help="add the GT panel (reads rescan annotations, eval side)")
            v.add_argument("--edges", action="store_true", help="draw support edges")
            v.set_defaults(func=_cmd_viz_topdown)
        else:
            v.add_argument("--images", action="store_true", help="log the colour images too")
            v.add_argument("--spawn", action="store_true", help="open the Rerun viewer")
            v.set_defaults(func=_cmd_viz_rerun)

    for name, helptext in (("eval", "run a method on a split and score it (DESIGN §7)"),
                           ("run", "run a method on one (reference, rescan) pair and score it")):
        sp_ = sub.add_parser(name, help=helptext)
        if name == "eval":
            sp_.add_argument("--split", required=True, help="split name (eval_val, dev_train) or file")
            sp_.add_argument("--workers", type=int, default=None, help="processes (default: config)")
            sp_.add_argument("--snapshots", action="store_true", help="write tentative snapshots (DESIGN §4)")
            sp_.set_defaults(func=_cmd_eval)
        else:
            sp_.add_argument("--pair", nargs=2, required=True, metavar=("REF", "RESCAN"))
            sp_.add_argument("--no-snapshots", action="store_true", help="skip tentative snapshots")
            sp_.add_argument("--snapshot-every", type=int, default=None,
                             help="snapshot every N processed frames (default: config online.snapshot_every)")
            sp_.set_defaults(func=_cmd_run)
        sp_.add_argument("--obs", default="oracle", help="comma list of oracle,perception (e.g. perception,oracle)")
        sp_.add_argument("--method", default="B0", help="B0 | B1 | B2 | FULL")
        sp_.add_argument("--frac", type=float, default=1.0, help="observed fraction (contiguous chunk)")
        sp_.add_argument("--seed", type=int, default=0)
        sp_.add_argument("--config", default=None)
        sp_.add_argument("--run-id", default=None)

    args = p.parse_args(argv)
    from lsg.config import setup_hub_offline

    hub_cfg = load_config(getattr(args, "config", None))
    if getattr(args, "func", None) is _cmd_detect_cache:  # the detector models must be cached too
        hub_cfg["perception"]["detector"]["enabled"] = True
    setup_hub_offline(hub_cfg)  # before torch / timm / open_clip / transformers load
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
