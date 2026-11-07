"""Perception mode (headline, DESIGN §6.6): rescan objects from depth residuals (M4).

Modules: `residual` (nearer points + surface / structure filter), `voxels` (decorrelated
multi-view voxel evidence), `cluster` (connected components + door swing filter), `labeler`
(CLIP zero-shot labels), `source` (`PerceptionSource`, the `ObsSource`).

Hard rule 2 (CLAUDE.md): this package must never import `lsg.oracle` or `lsg.eval`, and never
read rescan semseg, rescan PLY labels, rescan 3DSSG or the `rigid/removed/nonrigid` lists.
`tests/test_no_leak.py` enforces this.
"""

from lsg.perception.labeler import CLIPLabeler, StubLabeler
from lsg.perception.source import PerceptionSource

__all__ = ["CLIPLabeler", "PerceptionSource", "StubLabeler"]
