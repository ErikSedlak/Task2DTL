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


def hub_repos(cfg: dict[str, Any]) -> list[str]:
    """Hugging Face Hub repos of the frozen models: DINOv2 via timm (`timm/<appearance.model>`),
    CLIP via open_clip (`perception.clip.hf_repo`), and with `perception.detector.enabled` the detector's
    Grounding DINO + SAM 2 (`lsg detect cache` enables it for the check)."""
    out = []
    if cfg.get("appearance", {}).get("model"):
        out.append(f"timm/{cfg['appearance']['model']}")
    if cfg.get("perception", {}).get("clip", {}).get("hf_repo"):
        out.append(str(cfg["perception"]["clip"]["hf_repo"]))
    det = cfg.get("perception", {}).get("detector") or {}  # CHANGE 3 Part E: needed by `lsg detect cache` only
    if det.get("enabled", False):
        out += [str(det[k]) for k in ("gdino_model", "sam_model") if det.get(k)]
    return out


def hub_cache_dir() -> Path:
    """Local Hugging Face Hub cache: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hub_cached(repo: str, cache: Path | None = None) -> bool:
    """Whether a model repo has a downloaded snapshot (at least one file) in the local cache."""
    d = (cache or hub_cache_dir()) / ("models--" + repo.replace("/", "--")) / "snapshots"
    return d.is_dir() and any(f.is_file() for f in d.rglob("*"))


def setup_hub_offline(cfg: dict[str, Any]) -> str:
    """Set `HF_HUB_OFFLINE=1` (no Hub requests) when allowed, before any model library is imported.

    `models.hub_offline`: `true` -> always offline; `false` -> never set; `auto` (default) ->
    offline only if every repo of `hub_repos` is cached, so a fresh machine still downloads the
    weights once. An `HF_HUB_OFFLINE` already in the environment always wins. Worker processes
    inherit the variable. Returns what was decided (`env`, `offline`, `online`).
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return "env"
    mode = str(cfg.get("models", {}).get("hub_offline", "auto")).lower()
    if mode == "true" or (mode == "auto" and all(hub_cached(r) for r in hub_repos(cfg))):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return "offline"
    return "online"


def milestone_val_split(cfg: dict[str, Any], full: bool = False) -> Path:
    """Val pair list for milestone checks: the seeded subset (`splits.milestone_val.file`), or the
    full `eval_val` with `full=True` (final tables, M7); Erik's choice at M6."""
    mv = cfg["splits"]["milestone_val"]
    return REPO_ROOT / (mv["source"] if full else mv["file"])


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
