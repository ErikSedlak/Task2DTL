"""Frame votes from render-and-compare, and viewpoint decorrelation.

Per processed frame, the reference mesh is rendered at the rescan camera's aligned pose. For a
node i, P_i = pixels whose first hit is i. Comparing observed depth d_obs with the rendered
depth d_exp on P_i gives one frame vote:

- `seen`: >= `vote_frac` of the valid pixels are present (|d_obs - d_exp| <= tau_near);
- `free`: >= `vote_frac` of the valid pixels are seen through (d_obs > d_exp + tau_free);
- `noinfo`: anything else (occluded / nearer pixels, mixed evidence, too few valid pixels);
- `unobserved`: |P_i| < `min_mask_px` (out of frustum or hidden behind other reference geometry).

Seen votes need contrast (FULL): with `d_behind` (depth of the next
reference surface behind the pixel's first hit with another label), only *contrastive* pixels,
d_behind > d_exp + tau_free, can support `seen`: there, depth would show the difference if the
node were gone. `seen` then needs >= `min_mask_px` valid contrastive pixels, >= `vote_frac` of
them present; `free` is unchanged; anything else is `noinfo`.

Thin nodes (M6, appearance on): depth cannot tell whether a thing thinner than tau_free is
still there, so their depth `seen` / `noinfo` vote is replaced by the appearance verdict on the
comparable pixels (`thin_votes`): mean frame-relative appearance distance > `tau_node` ->
`free_like`, else `seen`; too few comparable pixels -> `noinfo`. A depth `free` vote stays.

Only free-space evidence (and, for thin nodes, appearance mismatch from >= `k_free_like`
viewpoints) may lower confidence (hard rule 3); occlusion, invalid depth and out-of-view are
no information. Votes count only after viewpoint decorrelation (`Decorrelator`). Method side:
no rescan annotations are involved.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

SEEN, FREE, NOINFO, UNOBSERVED, FREE_LIKE = "seen", "free", "noinfo", "unobserved", "free_like"
COUNTED_KINDS = (SEEN, FREE, FREE_LIKE)  # kinds that go through the decorrelator


@dataclass(frozen=True)
class FrameVote:
    """One node's vote in one frame, with the pixel counts it was decided from."""

    kind: str
    n_px: int  # |P_i|: first-hit pixels (after erosion)
    n_valid: int  # of those, valid observed and rendered depth
    n_present: int
    n_free: int
    n_comparable: int = 0  # thin nodes: appearance-comparable pixels (M6)
    n_mismatch: int = 0  # thin nodes: of those, mismatching
    n_contrast: int = 0  # seen needs contrast (M7): valid contrastive pixels
    n_present_contrast: int = 0  # of those, present


def erode_labels(labels: np.ndarray, r: int) -> np.ndarray:
    """Per-label erosion: keep a pixel only if its (2r+1)^2 neighbourhood has the same label.

    Removes silhouette pixels of every node at once (others become 0). Image borders are
    padded with the nearest value, so they do not erode.
    """
    if r <= 0:
        return labels
    from scipy.ndimage import maximum_filter, minimum_filter

    size = 2 * int(r) + 1
    lo = minimum_filter(labels, size=size, mode="nearest")
    hi = maximum_filter(labels, size=size, mode="nearest")
    return np.where((lo == labels) & (hi == labels), labels, 0)


def frame_votes(
    node_oids: Iterable[int],
    ref_depth: np.ndarray,
    ref_labels: np.ndarray,
    obs_depth: np.ndarray,
    cfg: Mapping[str, Any],
    d_behind: np.ndarray | None = None,
) -> dict[int, FrameVote]:
    """Frame votes of the given reference objects in one frame.

    Args:
        node_oids: Reference objectIds to vote on (anchor / thing nodes).
        ref_depth: (H, W) rendered reference depth [m], 0 = no hit.
        ref_labels: (H, W) reference objectId of the first hit, 0 = none.
        obs_depth: (H, W) observed sensor depth [m], 0 = invalid.
        cfg: Full config; uses `cfg["visibility"]`.
        d_behind: (H, W) depth of the next reference surface behind the first hit with another
            label (`Renderer.depth_behind`, inf = none), or None for the M3 rule. Given: `seen`
            only from contrastive pixels.

    Returns:
        `{objectId: FrameVote}` for every id in `node_oids` (`unobserved` included).
    """
    vc = cfg["visibility"]
    tau_near, tau_free = float(vc["tau_near"]), float(vc["tau_free"])
    min_px, frac = int(vc["min_mask_px"]), float(vc["vote_frac"])
    lab = erode_labels(np.asarray(ref_labels, np.int64), int(vc.get("erode_px", 0)))
    lab = np.where(lab > 0, lab, 0).ravel()
    d_exp = np.asarray(ref_depth, np.float64).ravel()
    d_obs = np.asarray(obs_depth, np.float64).ravel()
    valid = (d_obs > 0) & (d_exp > 0)
    diff = d_obs - d_exp
    present = valid & (np.abs(diff) <= tau_near)
    free = valid & (diff > tau_free)

    oids = [int(o) for o in node_oids]
    n = max([int(lab.max()) if lab.size else 0, *oids, 0]) + 1
    c_px = np.bincount(lab, minlength=n)
    c_val = np.bincount(lab, weights=valid, minlength=n)
    c_pre = np.bincount(lab, weights=present, minlength=n)
    c_fre = np.bincount(lab, weights=free, minlength=n)
    contrast = d_behind is not None
    if contrast:
        db = np.asarray(d_behind, np.float64).ravel()
        con = valid & (db > d_exp + tau_free)
        c_con = np.bincount(lab, weights=con, minlength=n)
        c_pcon = np.bincount(lab, weights=con & present, minlength=n)

    out: dict[int, FrameVote] = {}
    for o in oids:
        npx, nv, npre, nfr = int(c_px[o]), int(c_val[o]), int(c_pre[o]), int(c_fre[o])
        nc, npc = (int(c_con[o]), int(c_pcon[o])) if contrast else (0, 0)
        if o <= 0 or npx < min_px:
            kind = UNOBSERVED
        elif nv < min_px:
            kind = NOINFO
        elif (nc >= min_px and npc >= frac * nc) if contrast else (npre >= frac * nv):
            kind = SEEN
        elif nfr >= frac * nv:
            kind = FREE
        else:
            kind = NOINFO
        out[o] = FrameVote(kind, npx, nv, npre, nfr, n_contrast=nc, n_present_contrast=npc)
    return out


def thin_votes(
    depth_votes: Mapping[int, FrameVote],
    thin_oids: Iterable[int],
    ref_labels: np.ndarray,
    comparable_px: np.ndarray,
    rel_px: np.ndarray,
    cfg: Mapping[str, Any],
    tau_node: float,
) -> dict[int, FrameVote]:
    """Replace thin nodes' depth `seen` / `noinfo` votes by the appearance verdict (M6, Erik).

    Over the node's first-hit pixels that are appearance-comparable: mean frame-relative patch
    distance > `tau_node` -> `free_like`, else `seen`; < `min_mask_px` comparable pixels ->
    `noinfo`. Depth `free` and `unobserved` votes, and non-thin nodes, are unchanged.
    (`n_mismatch` records the comparable pixels whose own distance exceeds `tau_node`.)

    Args:
        depth_votes: `frame_votes` output of the frame.
        thin_oids: objectIds of thin nodes.
        ref_labels: (H, W) reference first-hit objectIds (as given to `frame_votes`).
        comparable_px, rel_px: (H, W) appearance maps (`perception.appearance.AppearanceMap`).
        cfg: Full config (`visibility`).
        tau_node: Threshold on the node's mean relative distance (`appearance.tau_node`).

    Returns:
        A copy of `depth_votes` with the thin nodes' votes replaced.
    """
    vc = cfg["visibility"]
    min_px = int(vc["min_mask_px"])
    out = dict(depth_votes)
    thin = [int(o) for o in thin_oids if int(o) in depth_votes and depth_votes[int(o)].kind in (SEEN, NOINFO)]
    if not thin:
        return out
    lab = erode_labels(np.asarray(ref_labels, np.int64), int(vc.get("erode_px", 0)))
    lab = np.where(lab > 0, lab, 0).ravel()
    n = max(int(lab.max()) if lab.size else 0, *thin) + 1
    comp = np.asarray(comparable_px, bool).ravel()
    rel = np.where(comp, np.asarray(rel_px, np.float64).ravel(), 0.0)
    c_comp = np.bincount(lab, weights=comp, minlength=n)
    c_sum = np.bincount(lab, weights=rel, minlength=n)
    c_mis = np.bincount(lab, weights=comp & (rel > tau_node), minlength=n)
    for o in thin:
        v = depth_votes[o]
        nc, nm = int(c_comp[o]), int(c_mis[o])
        if nc < min_px:
            kind = NOINFO
        elif round(c_sum[o] / nc, 6) > tau_node:
            kind = FREE_LIKE
        else:
            kind = SEEN
        out[o] = FrameVote(kind, v.n_px, v.n_valid, v.n_present, v.n_free, nc, nm)
    return out


class Decorrelator:
    """Counts an event only from a new enough viewpoint.

    Per key (e.g. `(node_id, "free")` or a voxel), the camera pose of the last *counted* event
    is stored. A new event counts if the camera centre moved >= `d_dec` or the viewing
    direction (camera +z axis in the world) turned >= `theta_dec_deg` since then. The first
    event of a key always counts. Repeating one viewpoint therefore adds one counted event.
    """

    def __init__(self, d_dec: float, theta_dec_deg: float, enabled: bool = True):
        """
        Args:
            d_dec: Minimum camera-centre distance [m].
            theta_dec_deg: Minimum viewing-direction change [deg].
            enabled: False = every event counts (decorrelation-off ablation).
        """
        self.d_dec = float(d_dec)
        self.cos_dec = float(np.cos(np.radians(theta_dec_deg)))
        self.enabled = bool(enabled)
        self._last: dict[Hashable, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Decorrelator:
        """Built from `cfg["decorrelation"]`."""
        d = cfg["decorrelation"]
        return cls(float(d["d_dec"]), float(d["theta_dec_deg"]), bool(d.get("enabled", True)))

    def counts(self, key: Hashable, cam_pose: np.ndarray) -> bool:
        """Whether an event of `key` seen from `cam_pose` (4x4 camera->world) counts.

        A counted event becomes the new reference viewpoint of its key.
        """
        c = np.asarray(cam_pose[:3, 3], float)
        z = np.asarray(cam_pose[:3, 2], float)
        z = z / (np.linalg.norm(z) or 1.0)
        prev = self._last.get(key)
        if self.enabled and prev is not None:
            pc, pz = prev
            if np.linalg.norm(c - pc) < self.d_dec and float(np.dot(z, pz)) > self.cos_dec:
                return False
        self._last[key] = (c, z)
        return True


@dataclass
class NodeVotes:
    """Vote tally of one reference node over the processed frames."""

    raw_seen: int = 0
    raw_free: int = 0
    noinfo: int = 0
    seen: int = 0  # counted (decorrelated)
    free: int = 0  # counted (decorrelated)
    raw_free_like: int = 0  # thin nodes: appearance mismatch votes (M6)
    free_like: int = 0  # counted (decorrelated)
    n_observed_frames: int = 0  # frames with a vote other than unobserved
    first_free_frame: int | None = None  # first counted free vote
    first_free_like_frame: int | None = None  # first counted free-like vote
    last_seen_frame: int | None = None  # last raw seen vote
    commit_removed_frame: int | None = None  # first frame with a tentative `removed` (immediate commitment, M7)

    def add(self, kind: str, counted: bool, frame_idx: int) -> None:
        """Record one frame vote (`counted` = passed the decorrelator; seen / free / free-like)."""
        if kind == UNOBSERVED:
            return
        self.n_observed_frames += 1
        if kind == NOINFO:
            self.noinfo += 1
        elif kind == SEEN:
            self.raw_seen += 1
            self.seen += bool(counted)
            self.last_seen_frame = int(frame_idx)
        elif kind == FREE:
            self.raw_free += 1
            if counted:
                self.free += 1
                if self.first_free_frame is None:
                    self.first_free_frame = int(frame_idx)
        elif kind == FREE_LIKE:
            self.raw_free_like += 1
            if counted:
                self.free_like += 1
                if self.first_free_like_frame is None:
                    self.first_free_like_frame = int(frame_idx)

    def to_json(self) -> dict[str, Any]:
        """Plain dict (for diagnostics)."""
        return dict(self.__dict__)


def has_free_evidence(v: NodeVotes | None, cfg: Mapping[str, Any]) -> bool:
    """Removed / unknown split: >= k_free counted free + free-like votes, where appearance
    alone (no counted depth free vote) needs >= `k_free_like` free-like votes (M6).

    Without free-like votes (appearance off) this is the M3 rule `free >= k_free`.
    """
    if v is None:
        return False
    st = cfg["states"]
    if v.free + v.free_like < int(st["k_free"]):
        return False
    return v.free > 0 or v.free_like >= int(st.get("k_free_like", 3))


def old_place_class(v: NodeVotes | None, cfg: Mapping[str, Any]) -> str:
    """Evidence at a node's reference place (CHANGE 3 Part A), from counted votes only.

    - `unobserved`: no counted seen / free / free-like vote;
    - `net_seen`: L_hit·seen > L_miss·free + L_free_like·free_like with the node weights of
      `confidence` (CHANGE 3 Part C, Erik: the same weighted evidence as FULL's confidence, so a node
      the removal veto keeps is never moved away and vice versa); checked first;
    - `free_ev`: free evidence (`has_free_evidence`);
    - `mixed`: everything else.
    """
    if v is None or v.seen == v.free == v.free_like == 0:
        return "unobserved"
    cc = cfg.get("confidence") or {}
    w_s, w_f, w_fl = (float(cc.get(k, 1.0)) for k in ("L_hit", "L_miss", "L_free_like"))
    if w_s * v.seen > w_f * v.free + w_fl * v.free_like + 1e-12:
        return "net_seen"
    if has_free_evidence(v, cfg):
        return "free_ev"
    return "mixed"


def confirmed_in_place(v: NodeVotes | None, cfg: Mapping[str, Any]) -> bool:
    """Confirmed: >= k_seen counted seen votes and no counted free or free-like vote."""
    return v is not None and v.seen >= int(cfg["states"]["k_seen"]) and v.free == 0 and v.free_like == 0
