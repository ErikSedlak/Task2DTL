"""Perception mode (headline, DESIGN §6.6): rescan objects from depth/appearance residuals.

Implemented at M4. Hard rule 2 (CLAUDE.md): this package must never import `lsg.oracle` or
`lsg.eval`, and never read rescan semseg, rescan PLY labels, rescan 3DSSG or the
`rigid/removed/nonrigid` lists. `tests/test_no_leak.py` enforces this.
"""
