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

    sp = sub.add_parser("splits", help="split-file helpers")
    sps = sp.add_subparsers(dest="splits_cmd", required=True)
    tx = sps.add_parser("textures", help="write splits/download_textures.txt")
    tx.add_argument("--config", default=None)
    tx.add_argument("--out", default=None, help="relative to the repo root")
    tx.set_defaults(func=_cmd_splits_textures)

    gr = sub.add_parser("graph", help="export a reference scene graph as JSON")
    gr.add_argument("--scan", required=True, help="reference scan id")
    gr.add_argument("--mode", choices=["gt", "geometric"], default="gt")
    gr.add_argument("--out", default=None)
    gr.add_argument("--config", default=None)
    gr.set_defaults(func=_cmd_graph)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
