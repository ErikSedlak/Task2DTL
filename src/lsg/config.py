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
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_hash(cfg: dict[str, Any]) -> str:
    """sha256 over canonical JSON (sorted keys), first 12 hex chars."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def data_root(cfg: dict[str, Any] | None = None) -> Path:
    """Env `LSG_DATA` wins, then cfg['data_root'], then ./data."""
    env = os.environ.get("LSG_DATA")
    if env:
        return _resolve(env)
    return _resolve((cfg or {}).get("data_root", "./data"))


def outputs_root(cfg: dict[str, Any] | None = None) -> Path:
    return _resolve((cfg or {}).get("outputs_root", "./outputs"))
