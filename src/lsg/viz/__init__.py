"""Visualisation (DESIGN §6.14, M8): top-down before/after figures and the Rerun timeline.

Reads only method inputs and outputs (reference scan, rescan frames + poses, graphs, snapshots).
GT overlays are computed in `lsg.eval` and passed in (hard rule 1); `viz` never imports
`lsg.oracle` / `lsg.eval` (checked by `tests/test_no_leak.py`).
"""
