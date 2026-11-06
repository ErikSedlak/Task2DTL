"""Per-pair preparation and the minimal M2 frame loop (method side).

Uses only allowed inputs (hard rule 1): reference scan + 3DSSG, the rescan's RGB-D frames +
poses, and 3RScan's raw `T_align`. It never sees the rescan's `ScanMeta` (which carries the
`rigid/removed/nonrigid` GT lists), only its `T_align` matrix.

M3 replaces `collect_observations` with `change.online.run_online` (render, votes, snapshots).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lsg.config import cache_root
from lsg.geom.align import refine_from_sequence
from lsg.graph.build import build_reference_graph
from lsg.io.scan3r import Scan, SequenceReader, iter_frames
from lsg.obs.base import ObsInstance, ObsSource, RefContext, SeqInfo


def select_frames(frame_ids: list[int], stride: int, frac: float = 1.0, seed: int = 0, key: str = "") -> list[int]:
    """Processed frames: every `stride`-th frame, then one contiguous chunk (DESIGN §7.3).

    Args:
        frame_ids: All frame ids of the sequence, in time order.
        stride: Frame stride (config `frames.stride`).
        frac: Share of the strided frames to keep (1.0 = all).
        seed: Seed of the random chunk start.
        key: Mixed into the seed (the rescan id), so pairs get different starts.

    Returns:
        Frame ids in time order. For frac >= 1 the whole strided sequence.
    """
    ids = list(frame_ids)[:: int(stride)]
    if frac >= 1.0 or not ids:
        return ids
    n = max(1, int(round(frac * len(ids))))
    rng = np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])
    start = int(rng.integers(0, len(ids) - n + 1))
    return ids[start : start + n]


def _align_key(ref_id: str, rescan_id: str, T_raw: np.ndarray, acfg: Mapping[str, Any], frame_ids: Iterable[int]) -> str:
    """Cache key of a refined alignment: inputs that change the ICP result."""
    blob = json.dumps(
        {"ref": ref_id, "rescan": rescan_id, "T": np.round(np.asarray(T_raw), 9).tolist(),
         "cfg": dict(acfg), "frames": list(map(int, frame_ids))},
        sort_keys=True, default=str,
    )  # fmt: skip
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def refined_alignment(
    seq: SequenceReader,
    ref_scan: Scan,
    rescan_id: str,
    T_raw: np.ndarray,
    cfg: Mapping[str, Any],
    frame_ids: list[int],
    use_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The method's `T_align`: label-free ICP refinement (M0), cached in `cache/align/`.

    Returns:
        (4x4 transform, info). `info["cached"]` tells whether it came from the cache. With
        `alignment.refine: false` the raw transform is returned.
    """
    acfg = cfg["alignment"]
    if not acfg.get("refine", True):
        return np.asarray(T_raw, float).copy(), {"accepted": False, "disabled": True, "cached": False}
    path = cache_root(dict(cfg)) / "align" / f"{rescan_id}_{_align_key(ref_scan.scan_id, rescan_id, T_raw, acfg, frame_ids)}.json"
    if use_cache and path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        return np.array(d["T"], float), {**d["info"], "cached": True}
    T, info = refine_from_sequence(seq, ref_scan.vertices, ref_scan.faces, T_raw, acfg, frame_ids)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"T": T.tolist(), "info": info}), encoding="utf-8")
    return T, {**info, "cached": False}


def prepare_pair(
    root: str | Path,
    ref_scan: Scan,
    objs_R: Mapping[int, Mapping[str, Any]],
    rels_R: list[tuple[int, int, str]],
    seq: SequenceReader,
    rescan_id: str,
    T_raw: np.ndarray,
    cfg: Mapping[str, Any],
    frame_ids: list[int],
    G_ref=None,
) -> tuple[RefContext, SeqInfo, dict[str, Any]]:
    """Everything the method gets for one (reference, rescan) pair.

    Args:
        root: Data root.
        ref_scan, objs_R, rels_R: Reference scan, its 3DSSG objects and relations.
        seq: The rescan's sequence (RGB-D + poses).
        rescan_id: Rescan id.
        T_raw: 3RScan's raw `T_align` (rescan -> reference).
        cfg: Full config.
        frame_ids: Processed frames (`select_frames`).
        G_ref: Prebuilt reference graph (GT relations); built here if None.

    Returns:
        (RefContext, SeqInfo with the refined `T_align`, alignment info).
    """
    if G_ref is None:
        G_ref = build_reference_graph(ref_scan, objs_R, rels_R, "gt", cfg)
    T, ainfo = refined_alignment(seq, ref_scan, rescan_id, T_raw, cfg, frame_ids)
    ctx = RefContext(ref_scan.scan_id, ref_scan, G_ref, float(G_ref.meta["floor_z"]))
    info = SeqInfo(rescan_id, Path(root), seq.info, T, list(frame_ids))
    return ctx, info, ainfo


def collect_observations(
    seq: SequenceReader, source: ObsSource, ref_ctx: RefContext, seq_info: SeqInfo
) -> list[ObsInstance]:
    """Minimal M2 frame loop: `begin`, `observe` per processed frame in time order, `finish`.

    `ref_render` is None here; M3's online loop renders the reference and passes it.
    """
    source.begin(ref_ctx, seq_info)
    for frame in iter_frames(seq, idx=seq_info.frame_ids):
        source.observe(frame, None)
    return source.finish()
