"""The online frame loop (DESIGN §6.7): render -> votes -> decorrelation -> observe -> snapshot.

Rescan frames are processed in time order. Per frame the reference mesh is rendered at the
aligned rescan pose (`T_align @ pose`), frame votes are cast for every anchor / thing node,
counted through the `Decorrelator`, and the frame plus the reference render go to the
`ObsSource`. States during the walk are **tentative** (snapshots only); `moved` / `removed` are
decided once, after the last frame, by the method (`change.baselines`) from the final votes and
observations (deferred commitment, §3).

With an `appearance` runner (FULL, M6) every frame that shows a thin node (or every frame, if
the observation source wants appearance for thin-object candidates) also gets an appearance
map: thin nodes' depth seen / noinfo votes are replaced by the appearance verdict
(`visibility.thin_votes`), and the map rides along in `RefRender` to the source. After
`finish()`, `obs_views` counts the decorrelated viewpoints of every observation (§6.11).

With `contrast` (FULL, M7, DECISIONS 2026-11-15) the reference ray of every pixel on a voting
node is continued to the next surface with another label (`Renderer.depth_behind`), and only
contrastive pixels can support a `seen` vote (`visibility.frame_votes`).

Immediate commitment (ablation, M7): every node records the first frame at which its tentative
state became `removed` (`NodeVotes.commit_removed_frame`; with priors also tentative p < p_del).
Methods use it only with `states.commitment: immediate`.

Method side: imports only `geom`, `graph`, `io`, `obs`, `change`, `perception`, `train_meta` (no-leak test).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from lsg.change.confidence import below_p_del, node_logodds, prior_logodds, probability
from lsg.change.visibility import (
    COUNTED_KINDS,
    NOINFO,
    SEEN,
    Decorrelator,
    NodeVotes,
    confirmed_in_place,
    frame_votes,
    has_free_evidence,
    thin_votes,
)
from lsg.geom.render import Renderer
from lsg.geom.transforms import invert
from lsg.graph.model import Graph, object_id_of
from lsg.io.scan3r import iter_frames
from lsg.obs.base import ObsInstance, ObsSource, RefContext, RefRender, SeqInfo
from lsg.perception.appearance import tau_node

VOTING_LEVELS = ("anchor", "thing")


@dataclass
class OnlineResult:
    """What the online loop hands to the method at the end of the sequence."""

    obs: list[ObsInstance]
    votes: dict[str, NodeVotes]  # node id -> tally
    n_frames: int
    snapshots: list[Path] = field(default_factory=list)
    time_s: dict[str, float] = field(default_factory=dict)
    obs_views: dict[str, int] = field(default_factory=dict)  # token -> decorrelated views (M6)
    stats: dict[str, Any] = field(default_factory=dict)  # appearance frames, thin nodes, ...
    T_align: np.ndarray | None = None  # the alignment at the end of the walk (online estimate or the given one)


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
    priors: Any = None,
) -> Graph:
    """Snapshot graph (§4): G_ref with tentative vote-only states, `meta.tentative = True`.

    With `priors` (FULL, M6) nodes also carry their tentative confidence (unmatched formula).
    """
    G = copy.deepcopy(G_ref)
    G.meta.update(scan_obs=rescan_id, obs_mode=obs_mode, frame_idx=int(frame_idx), tentative=True)
    for n in G.nodes:
        if n.level in VOTING_LEVELS:
            v = votes.get(n.id)
            n.state = tentative_state(v, cfg)
            if priors is not None:
                n.confidence = probability(node_logodds(prior_logodds(n.label, n.nyu40, priors, cfg), v, cfg))
    return G


def observation_views(
    obs: list[ObsInstance], poses: Mapping[int, np.ndarray], cfg: Mapping[str, Any]
) -> dict[str, int]:
    """Decorrelated viewpoints per observation: frames with a mask of >= `min_mask_px` pixels,
    counted through a fresh `Decorrelator` in time order (§6.11 "seen at the new location")."""
    min_px = int(cfg["visibility"]["min_mask_px"])
    out: dict[str, int] = {}
    for o in obs:
        dec = Decorrelator.from_config(cfg)
        n = 0
        for f in sorted(o.masks):
            if f not in poses:
                continue
            area = o.masks.area(f) if hasattr(o.masks, "area") else int(np.asarray(o.masks[f]).sum())
            if area >= min_px and dec.counts(o.token, poses[f]):
                n += 1
        out[o.token] = n
    return out


def run_online(
    ref_ctx: RefContext,
    seq_info: SeqInfo,
    seq,
    obs_source: ObsSource,
    cfg: Mapping[str, Any],
    snapshot_dir: str | Path | None = None,
    obs_mode: str | None = None,
    renderer: Renderer | None = None,
    appearance: Any = None,
    priors: Any = None,
    contrast: bool = False,
    on_snapshot: Any = None,
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
        appearance: `perception.appearance.AppearanceRunner` (FULL with appearance on, M6) or None.
            Frames that show a thin node with a depth seen / noinfo vote (all frames if the source
            has `wants_appearance`) get an `AppearanceMap`; thin votes are replaced by its verdict.
        priors: Class prior table (`train_meta.PriorTable`); snapshots then carry confidence, and
            the immediate-commitment frame also needs tentative p < p_del.
        contrast: Seen votes need contrast (FULL with `visibility.seen_needs_contrast`, M7).
        on_snapshot: CHANGE 3 Part B: called every `online.snapshot_every` processed frames as
            `on_snapshot(obs=obs_source.peek(), votes=copy, obs_views, frame_idx, k)` (the caller runs the
            method on the current state, writes and scores the snapshot); replaces the vote-only
            snapshots, and the caller writes the final graph as the last snapshot. Opaque here.

    Returns:
        `OnlineResult` with the observations, per-node vote tallies and snapshot paths.
    """
    t0 = time.time()
    # The walk works on its own SeqInfo: the online aligner replaces `T_align` during the walk, and the caller's
    # object (shared by both observation modes of a pair in the eval harness) must keep the raw transform, or the
    # second mode would start from an estimate made from all frames of the first walk (fixed 2026-11-22).
    seq_info = replace(seq_info, T_align=np.array(seq_info.T_align, dtype=float, copy=True))
    scan, G_ref = ref_ctx.scan, ref_ctx.graph
    r = renderer or Renderer(scan.vertices, scan.faces, scan.face_labels)
    node_of: dict[int, str] = {}
    for n in G_ref.nodes:
        oid = object_id_of(n.id)
        if n.level in VOTING_LEVELS and oid is not None:
            node_of[oid] = n.id
    votes = {nid: NodeVotes() for nid in node_of.values()}
    l0_of: dict[str, float] = {}
    if priors is not None:
        l0_of = {n.id: prior_logodds(n.label, n.nyu40, priors, cfg) for n in G_ref.nodes if n.id in votes}
    vc = cfg["visibility"]
    node_lab = np.fromiter(node_of, np.int64) if node_of else np.zeros(0, np.int64)
    t_contrast = 0.0
    dec = Decorrelator.from_config(cfg)
    K = seq_info.intrinsics.K_depth
    W, H = seq_info.intrinsics.depth_wh
    aligner = online_aligner(ref_ctx, seq_info, cfg)  # CHANGE 3 Part B: cumulative online alignment (or None)
    a_step = int(cfg["alignment"].get("frame_step", 10))
    a_every = int((cfg["alignment"].get("online") or {}).get("every", 20))
    every = int(cfg.get("online", {}).get("snapshot_every", 0) or 0)
    thin = ref_ctx.thin_oids(float(cfg["appearance"]["thin_max_extent"])) & set(node_of) if appearance else set()
    want_all = bool(getattr(obs_source, "wants_appearance", False)) and appearance is not None
    t_node = tau_node(cfg) if appearance is not None else 0.0
    if appearance is not None:
        appearance.begin(seq_info, seq)
    poses: dict[int, np.ndarray] = {}
    n_app = 0
    snap_dir = Path(snapshot_dir) if snapshot_dir is not None else None
    snaps: list[Path] = []
    t_render = t_obs = 0.0

    if snap_dir is not None:
        _write_seq_info(snap_dir, ref_ctx.scan_id, seq_info)
    obs_source.begin(ref_ctx, seq_info)
    k = 0
    last_idx = None
    for frame in iter_frames(seq, idx=seq_info.frame_ids):
        t1 = time.time()
        pose = np.asarray(seq_info.T_align, float) @ frame.pose_c2w
        poses[int(frame.idx)] = pose
        amap = up = None
        T_w2c = invert(pose)
        if appearance is None:
            d_exp, lab = r.render(K, T_w2c, W, H)
        else:
            d_exp, lab, prim = r.render(K, T_w2c, W, H, return_prim=True)
        d_behind = None
        if contrast:
            tc = time.time()
            d_behind = r.depth_behind(K, T_w2c, W, H, d_exp, lab, np.isin(lab, node_lab),
                                      float(vc.get("contrast_eps", 1e-3)))  # fmt: skip
            t_contrast += time.time() - tc
        fv = frame_votes(node_of, d_exp, lab, frame.depth, cfg, d_behind=d_behind)
        if appearance is not None:
            if want_all or any(fv[o].kind in (SEEN, NOINFO) for o in thin):
                amap = appearance.frame(frame, pose, d_exp)
                n_app += 1
                fv = thin_votes(fv, thin, lab, amap.comparable_px, amap.rel_px, cfg, t_node)
                up = np.where(prim >= 0, r.face_normal_z()[np.maximum(prim, 0)], 0.0).astype(np.float32)
        for oid, v in fv.items():
            nid = node_of[oid]
            counted = v.kind in COUNTED_KINDS and dec.counts((nid, v.kind), pose)
            nv = votes[nid]
            nv.add(v.kind, counted, frame.idx)
            if counted and v.kind != SEEN and nv.commit_removed_frame is None and has_free_evidence(nv, cfg):
                if priors is None or below_p_del(node_logodds(l0_of[nid], nv, cfg), cfg):
                    nv.commit_removed_frame = int(frame.idx)
        t2 = time.time()
        obs_source.observe(frame, RefRender(d_exp, lab, amap, up))
        t_obs += time.time() - t2
        t_render += t2 - t1
        k += 1
        last_idx = frame.idx
        if aligner is not None:  # only frames seen so far; the new estimate applies from the next frame on
            if (k - 1) % a_step == 0:
                aligner.add_frame(frame.depth, K, frame.pose_c2w)
            if k % a_every == 0:
                changed, _ = aligner.update()
                if changed:  # reassign (the sources hold this walk's SeqInfo and read T_align every frame)
                    seq_info.T_align = aligner.T.copy()
        if every > 0 and k % every == 0:
            if on_snapshot is not None:  # CHANGE 3 Part B: the method on the current state (eval-side callback)
                o_now = obs_source.peek()
                r_snap = on_snapshot(o_now, copy.deepcopy(votes), observation_views(o_now, poses, cfg), int(frame.idx), k)
                if r_snap is not None:
                    snaps.append(r_snap)
            elif snap_dir is not None:
                snaps.append(_snapshot(G_ref, votes, frame.idx, cfg, seq_info, obs_mode, snap_dir, priors, obs_source))
    # the last snapshot: vote-only here; with `on_snapshot` the caller writes the final graph itself
    last_snap = (on_snapshot is None and snap_dir is not None and last_idx is not None
                 and (every <= 0 or k % every != 0))  # fmt: skip
    if last_snap:  # before `finish`, so the sidecar shows the candidates of the walk
        snaps.append(_snapshot(G_ref, votes, last_idx, cfg, seq_info, obs_mode, snap_dir, priors, obs_source))
    t3 = time.time()
    obs = obs_source.finish()
    t_finish = time.time() - t3
    times = {"render_votes": t_render, "observe": t_obs, "finish": t_finish, "total": time.time() - t0}
    if contrast:
        times["contrast"] = t_contrast
    stats: dict[str, Any] = {"thin_nodes": len(thin), "appearance_frames": n_app}
    if aligner is not None:
        stats["align_online"] = aligner.updates
    if appearance is not None:
        times["appearance"] = appearance.time_s
        stats.update(appearance_device=getattr(appearance.ext, "device", None),
                     appearance_cache_hits=appearance.cache.hits, appearance_cache_misses=appearance.cache.misses)  # fmt: skip
    return OnlineResult(obs, votes, k, snaps, times, observation_views(obs, poses, cfg), stats,
                        np.asarray(seq_info.T_align, float).copy())


def online_aligner(ref_ctx: RefContext, seq_info: SeqInfo, cfg: Mapping[str, Any]):
    """`geom.align.OnlineAligner` when `alignment.online.enabled` (and `alignment.refine`), else None.

    `seq_info.T_align` must hold the raw 3RScan transform at the start (`change.pipeline.prepare_pair` skips the
    pre-pass then); it is updated in place whenever an estimate is accepted. Target `structure` = the
    reference wall / floor / ceiling faces (reference labels), falling back to all surfaces below
    `min_target_points` vertices.
    """
    from lsg.geom.align import OnlineAligner, mesh_target

    ac = cfg["alignment"]
    oc = ac.get("online") or {}
    if not (ac.get("refine", True) and oc.get("enabled", False)):
        return None
    scan = ref_ctx.scan
    V, F = scan.vertices, scan.faces
    if oc.get("target", "all") == "structure":
        oids = [o for o in scan.instances if ref_ctx.object_nyu40(o) in STRUCT_ALIGN_NYU40]
        Vs, Fs = ref_ctx.faces_of(oids)
        used, inv = np.unique(Fs, return_inverse=True)  # compact: only the structure's own vertices
        if len(used) >= int(oc.get("min_target_points", 500)):
            V, F = Vs[used], inv.reshape(Fs.shape)
    return OnlineAligner(mesh_target(V, F), np.asarray(seq_info.T_align, float), dict(ac))


STRUCT_ALIGN_NYU40 = (1, 2, 22)  # wall, floor, ceiling: never moved between visits


def _snapshot(G_ref, votes, frame_idx, cfg, seq_info, obs_mode, snap_dir: Path, priors=None, source=None) -> Path:
    """Write one tentative snapshot and return its path.

    If the observation source has `live_candidates` (perception, CHANGE 2), its current
    candidates go to the sidecar `candidates_frame_XXXXXX.json` (box, confidence, size, tier).
    """
    G = tentative_graph(G_ref, votes, frame_idx, cfg, seq_info.rescan_id, obs_mode, priors)
    p = G.save(snap_dir / f"frame_{int(frame_idx):06d}.json")
    live = getattr(source, "live_candidates", None)
    if live is not None:
        import json

        rows = live()
        (snap_dir / f"candidates_frame_{int(frame_idx):06d}.json").write_text(
            json.dumps({"frame_idx": int(frame_idx), "candidates": rows}), encoding="utf-8")  # fmt: skip
    return p


def _write_seq_info(snap_dir: Path, ref_id: str, seq_info: SeqInfo) -> Path:
    """`seq_info.json` next to the snapshots: refined `T_align` + processed frames (for `lsg viz rerun`)."""
    import json

    snap_dir.mkdir(parents=True, exist_ok=True)
    p = snap_dir / "seq_info.json"
    p.write_text(json.dumps({"ref": ref_id, "rescan": seq_info.rescan_id,
                             "T_align": np.asarray(seq_info.T_align, float).tolist(),
                             "frame_ids": [int(i) for i in seq_info.frame_ids]}), encoding="utf-8")  # fmt: skip
    return p
