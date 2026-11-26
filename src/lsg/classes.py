"""Node classes by 3DSSG / PLY nyu40 id. Spec definitions, not tunables."""

from __future__ import annotations

ANCHOR_NYU40 = frozenset({3, 4, 6, 7, 10, 12, 14, 15, 17, 32})
STRUCTURE_NYU40 = frozenset({1, 2, 8, 9, 11, 13, 16, 19, 20, 22, 28, 30, 38})


def node_level(nyu40: int) -> str:
    """Graph level of an object from its nyu40 class id.

    Returns:
        "structure" (walls, floor, doors, ...; not a graph node), "anchor" (large furniture
        that other objects stand on or in), or "thing" (everything else).
    """
    if nyu40 in STRUCTURE_NYU40:
        return "structure"
    if nyu40 in ANCHOR_NYU40:
        return "anchor"
    return "thing"
