"""The observation interface (DESIGN §6.4): what the change logic sees of a rescan.

Two implementations exist behind `ObsSource`: `lsg.oracle.OracleSource` (upper bound, rescan GT
instances) and `lsg.perception.PerceptionSource` (headline, from M4). The change logic depends
only on this module, so both modes share association, classification, update and eval.

This module is method side: it must not import `lsg.oracle` or `lsg.eval` (no-leak test).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from lsg.graph.model import Graph
from lsg.io.scan3r import Frame, Intrinsics, Scan


class PackedMasks(Mapping[int, np.ndarray]):
    """Per-frame boolean masks stored bit-packed (`np.packbits`, 8x smaller).

    Behaves like a read-only `{frame_idx: bool[H, W]}` dict; masks are unpacked on access.
    Raw bool masks of every instance in every processed frame would need ~230 MB per pair.
    """

    def __init__(self, shape: tuple[int, int]):
        """
        Args:
            shape: (H, W) of every mask (depth resolution, 172 x 224).
        """
        self.shape = (int(shape[0]), int(shape[1]))
        self._data: dict[int, np.ndarray] = {}
        self._area: dict[int, int] = {}

    def add(self, frame_idx: int, mask: np.ndarray) -> None:
        """Store (or OR into an existing) mask for `frame_idx`."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != self.shape:
            raise ValueError(f"mask shape {m.shape} != {self.shape}")
        if frame_idx in self._data:
            m = m | self[frame_idx]
        self._data[int(frame_idx)] = np.packbits(m.ravel())
        self._area[int(frame_idx)] = int(m.sum())

    def area(self, frame_idx: int) -> int:
        """Number of True pixels of one mask (without unpacking)."""
        return self._area[frame_idx]

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Unpacked (H, W) bool mask of one frame."""
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self._data[frame_idx], count=n).astype(bool).reshape(self.shape)

    def __iter__(self) -> Iterator[int]:
        """Frame indices in insertion (time) order."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of frames with a mask."""
        return len(self._data)


@dataclass
class ObsInstance:
    """One observed rescan object (DESIGN §6.4). Geometry in the **reference** frame.

    `token` is opaque to the method: oracle `o<k>` (shuffled id), perception `c<k>`.
    """

    token: str
    label: str
    label_topk: list[tuple[str, float]]
    nyu40: int | None
    points: np.ndarray  # (N, 3) reference frame
    masks: Mapping[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> bool[172, 224]
    crops: list[np.ndarray] = field(default_factory=list)  # upright RGB crops

    @property
    def centroid(self) -> np.ndarray:
        """Mean of the observed points."""
        return self.points.mean(axis=0)

    @property
    def aabb_min(self) -> np.ndarray:
        """Lower AABB corner of the observed points."""
        return self.points.min(axis=0)

    @property
    def aabb_max(self) -> np.ndarray:
        """Upper AABB corner of the observed points."""
        return self.points.max(axis=0)


@dataclass
class RefContext:
    """What the method knows about the reference: its scan (mesh + labels) and G_ref."""

    scan_id: str
    scan: Scan
    graph: Graph
    floor_z: float


@dataclass
class SeqInfo:
    """What the method knows about the rescan besides its frames (DESIGN §1)."""

    rescan_id: str
    root: Path
    intrinsics: Intrinsics
    T_align: np.ndarray  # rescan -> reference, ICP-refined (the method's alignment)
    frame_ids: list[int]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefRender:
    """The reference mesh rendered at a rescan frame's aligned pose (depth resolution)."""

    depth: np.ndarray  # (H, W) metres, 0 = no hit
    labels: np.ndarray  # (H, W) reference objectId of the first hit, 0 = none


@runtime_checkable
class ObsSource(Protocol):
    """Produces `ObsInstance`s from a rescan, frame by frame in time order."""

    def begin(self, ref_ctx: RefContext, seq_info: SeqInfo) -> None:
        """Called once before the first frame."""

    def observe(self, frame: Frame, ref_render: RefRender | None) -> None:
        """Called per processed frame, in time order, with the reference rendered at the
        frame's aligned pose (`change.online.run_online`; None if called without the loop)."""

    def finish(self) -> list[ObsInstance]:
        """Called once after the last frame; returns the observed instances."""
