"""Config loading. Thresholds live in YAML (configs/default.yaml), never in code."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read a YAML config file into a nested dict.

    Args:
        path: YAML file to read. Defaults to `configs/default.yaml`.

    Returns:
        The parsed config (empty dict for an empty file).
    """
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_hash(cfg: dict[str, Any]) -> str:
    """Short fingerprint of a config, stored in every metrics.json (CLAUDE.md hard rule 4).

    The hash is the sha256 of the config serialised as JSON with sorted keys, so two configs
    with the same content but different key order get the same hash.

    Returns:
        First 12 hex characters of the sha256.
    """
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _resolve(p: str | os.PathLike) -> Path:
    """Absolute path; relative paths are taken relative to the repo root, not the CWD."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def data_root(cfg: dict[str, Any] | None = None) -> Path:
    """Folder holding 3RScan.json, 3DSSG/ and one folder per scan.

    Priority: environment variable `LSG_DATA`, then `cfg["data_root"]`, then `./data`.
    """
    env = os.environ.get("LSG_DATA")
    if env:
        return _resolve(env)
    return _resolve((cfg or {}).get("data_root", "./data"))


def outputs_root(cfg: dict[str, Any] | None = None) -> Path:
    """Folder where runs write `outputs/<run_id>/` (git-ignored). Default `./outputs`."""
    return _resolve((cfg or {}).get("outputs_root", "./outputs"))


def cache_root(cfg: dict[str, Any] | None = None) -> Path:
    """Folder for cached intermediate results (git-ignored). Default `./cache`."""
    return _resolve((cfg or {}).get("cache_root", "./cache"))
