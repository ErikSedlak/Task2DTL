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
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
