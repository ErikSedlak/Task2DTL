"""Run bookkeeping: run ids, git SHA, metrics.json (CLAUDE.md hard rule 4)."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

from lsg.config import REPO_ROOT, config_hash


def git_sha() -> dict[str, Any]:
    """Current commit, so every result can be traced back to the code that produced it.

    Returns:
        `{"sha": <commit hash>, "dirty": <True if tracked files have uncommitted changes>}`,
        or both None if git is unavailable.
    """

    def _git(*args: str) -> str:
        """Run a git command in the repo and return its trimmed stdout."""
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def new_run_id(prefix: str) -> str:
    """Timestamped run name, e.g. `m0_20261105-143012`."""
    return f"{prefix}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def write_metrics(
    run_dir: Path, metrics: dict[str, Any], cfg: dict[str, Any], extra: dict[str, Any] | None = None
) -> Path:
    """Write `run_dir/metrics.json` with the results plus everything needed to reproduce them.

    Args:
        run_dir: Output folder of this run (created if missing).
        metrics: Results; numpy scalars/arrays and Paths are converted automatically. This
            block must be identical for the same config + seed (DESIGN §7.5).
        cfg: The config used, stored in full together with its hash.
        extra: Run information that may differ between reruns (wall time, cache hits),
            stored next to `metrics` under `"run"`.

    Returns:
        Path of the written file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_dir.name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_sha(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "metrics": metrics,
    }
    if extra is not None:
        payload["run"] = extra
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return out


def _json_default(o: Any) -> Any:
    """`json.dumps` fallback: numpy scalars/arrays -> Python values/lists, Path -> str."""
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
