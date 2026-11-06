"""Observation interface (DESIGN §6.4). Method side: never imports `lsg.oracle` / `lsg.eval`."""

from lsg.obs.base import ObsInstance, ObsSource, PackedMasks, RefContext, RefRender, SeqInfo

__all__ = ["ObsInstance", "ObsSource", "PackedMasks", "RefContext", "RefRender", "SeqInfo"]
