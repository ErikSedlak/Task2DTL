"""The online frame loop (DESIGN §6.7): render -> votes -> decorrelation -> observe -> snapshot.

Rescan frames are processed in time order. Per frame the reference mesh is rendered at the
aligned rescan pose (`T_align @ pose`), frame votes are cast for every anchor / thing node,
counted through the `Decorrelator`, and the frame plus the reference render go to the
`ObsSource`. States during the walk are **tentative** (snapshots only); `moved` / `removed` are
decided once, after the last frame, by the method (`change.baselines`) from the final votes and
observations (deferred commitment, §3).

Method side: imports only `geom`, `graph`, `io`, `obs` (no-leak test).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lsg.change.visibility import (
    COUNTED_KINDS,
    Decorrelator,
    NodeVotes,
    confirmed_in_place,
    frame_votes,
    has_free_evidence,
)
from lsg.geom.render import Renderer
from lsg.geom.transforms import invert
from lsg.graph.model import Graph, object_id_of
from lsg.io.scan3r import iter_frames
from lsg.obs.base import ObsInstance, ObsSource, RefContext, RefRender, SeqInfo

VOTING_LEVELS = ("anchor", "thing")


@dataclass
class OnlineResult:
    """What the online loop hands to the method at the end of the sequence."""

    obs: list[ObsInstance]
    votes: dict[str, NodeVotes]  # node id -> tally
    n_frames: int
    snapshots: list[Path] = field(default_factory=list)
    time_s: dict[str, float] = field(default_factory=dict)


def tentative_state(v: NodeVotes | None, cfg: Mapping[str, Any]) -> str:
    """Vote-only state during the walk (no association yet, so no `moved`).

    `removed` with >= k_free counted free votes, `unchanged` if confirmed in place, else
    `unknown`. Tentative: the final state is decided at the end of the sequence.
    """
    if has_free_evidence(v, cfg):
        return "removed"
    if confirmed_in_place(v, cfg):
        return "unchanged"
    return "unknown"


def tentative_graph(
    G_ref: Graph,
    votes: Mapping[str, NodeVotes],
    frame_idx: int,
    cfg: Mapping[str, Any],
    rescan_id: str | None = None,
    obs_mode: str | None = None,
) -> Graph:
    """Snapshot graph (§4): G_ref with tentative vote-only states, `meta.tentative = True`."""
    G = copy.deepcopy(G_ref)
    G.meta.update(scan_obs=rescan_id, obs_mode=obs_mode, frame_idx=int(frame_idx), tentative=True)
    for n in G.nodes:
        if n.level in VOTING_LEVELS:
            n.state = tentative_state(votes.get(n.id), cfg)
    return G


def run_online(
    ref_ctx: RefContext,
    seq_info: SeqInfo,
    seq,
    obs_source: ObsSource,
    cfg: Mapping[str, Any],
    snapshot_dir: str | Path | None = None,
    obs_mode: str | None = None,
    renderer: Renderer | None = None,
) -> OnlineResult:
    """Process the rescan's frames in time order (DESIGN §6.7).

    Args:
        ref_ctx: Reference scan (mesh + objectId face labels) and G_ref.
        seq_info: Rescan id, intrinsics, refined `T_align`, processed frame ids.
        seq: Open sequence (`SequenceReader` or anything with `frame(idx)` returning a `Frame`).
        obs_source: Observation source; gets `begin`, `observe(frame, RefRender)`, `finish`.
        cfg: Full config (`visibility`, `decorrelation`, `states`, `online`).
        snapshot_dir: If given, tentative snapshots go to `snapshot_dir/frame_XXXXXX.json`
            every `online.snapshot_every` processed frames, plus one after the last frame.
        obs_mode: Written to the snapshots' `meta.obs_mode`.
        renderer: Prebuilt reference renderer (built here if None).

    Returns:
        `OnlineResult` with the observations, per-node vote tallies and snapshot paths.
    """
    t0 = time.time()
    scan, G_ref = ref_ctx.scan, ref_ctx.graph
    r = renderer or Renderer(scan.vertices, scan.faces, scan.face_labels)
    node_of: dict[int, str] = {}
    for n in G_ref.nodes:
        oid = object_id_of(n.id)
        if n.level in VOTING_LEVELS and oid is not None:
            node_of[oid] = n.id
    votes = {nid: NodeVotes() for nid in node_of.values()}
    dec = Decorrelator.from_config(cfg)
    K = seq_info.intrinsics.K_depth
    W, H = seq_info.intrinsics.depth_wh
    T_align = np.asarray(seq_info.T_align, float)
    every = int(cfg.get("online", {}).get("snapshot_every", 0) or 0)
    snap_dir = Path(snapshot_dir) if snapshot_dir is not None else None
    snaps: list[Path] = []
    t_render = t_obs = 0.0

    obs_source.begin(ref_ctx, seq_info)
    k = 0
    last_idx = None
    for frame in iter_frames(seq, idx=seq_info.frame_ids):
        t1 = time.time()
        pose = T_align @ frame.pose_c2w
        d_exp, lab = r.render(K, invert(pose), W, H)
        fv = frame_votes(node_of, d_exp, lab, frame.depth, cfg)
        for oid, v in fv.items():
            nid = node_of[oid]
            counted = v.kind in COUNTED_KINDS and dec.counts((nid, v.kind), pose)
            votes[nid].add(v.kind, counted, frame.idx)
        t2 = time.time()
        obs_source.observe(frame, RefRender(d_exp, lab))
        t_obs += time.time() - t2
        t_render += t2 - t1
        k += 1
        last_idx = frame.idx
        if snap_dir is not None and every > 0 and k % every == 0:
            snaps.append(_snapshot(G_ref, votes, frame.idx, cfg, seq_info, obs_mode, snap_dir))
    t3 = time.time()
    obs = obs_source.finish()
    t_finish = time.time() - t3
    if snap_dir is not None and last_idx is not None and (every <= 0 or k % every != 0):
        snaps.append(_snapshot(G_ref, votes, last_idx, cfg, seq_info, obs_mode, snap_dir))
    times = {"render_votes": t_render, "observe": t_obs, "finish": t_finish, "total": time.time() - t0}
    return OnlineResult(obs, votes, k, snaps, times)


def _snapshot(G_ref, votes, frame_idx, cfg, seq_info, obs_mode, snap_dir: Path) -> Path:
    """Write one tentative snapshot and return its path."""
    G = tentative_graph(G_ref, votes, frame_idx, cfg, seq_info.rescan_id, obs_mode)
    return G.save(snap_dir / f"frame_{int(frame_idx):06d}.json")
