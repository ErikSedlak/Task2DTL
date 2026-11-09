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

    try:
        res = run_eval(load_config(args.config), [tuple(args.pair)], obs_modes=_modes(args.obs),
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

    gr = sub.add_parser("graph", help="export a reference scene graph as JSON")
    gr.add_argument("--scan", required=True, help="reference scan id")
    gr.add_argument("--mode", choices=["gt", "geometric"], default="gt")
    gr.add_argument("--out", default=None)
    gr.add_argument("--config", default=None)
    gr.set_defaults(func=_cmd_graph)

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
            sp_.set_defaults(func=_cmd_run)
        sp_.add_argument("--obs", default="oracle", help="comma list of oracle,perception (e.g. perception,oracle)")
        sp_.add_argument("--method", default="B0", help="B0 | B1 | B2 | FULL")
        sp_.add_argument("--frac", type=float, default=1.0, help="observed fraction (contiguous chunk)")
        sp_.add_argument("--seed", type=int, default=0)
        sp_.add_argument("--config", default=None)
        sp_.add_argument("--run-id", default=None)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
