"""3DSSG loaders (objects.json, relationships.json). Files are parsed once per process.

Relationship tuples are `[subject_id, object_id, rel_id, name]`; the subject is the child
("cup standing on table"). Ids are semseg objectIds.

Hard rule 1: reading 3DSSG of a **rescan** is allowed only from `lsg.oracle` / `lsg.eval`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=4)
def _index(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, Any] = {}
    for sc in data["scans"]:
        out.setdefault(sc["scan"], []).append(sc)
    return out


def load_predicates(root: str | Path) -> list[str]:
    """`relationships.txt`; index = rel_id, `none` = 0."""
    return (Path(root) / "3DSSG" / "relationships.txt").read_text(encoding="utf-8").splitlines()


def load_objects(root: str | Path, scan_id: str) -> dict[int, dict[str, Any]]:
    """objectId -> {label, nyu40 (int), eigen13, rio27, ...}. Empty if the scan is not in 3DSSG."""
    out: dict[int, dict[str, Any]] = {}
    for sc in _index(str(Path(root) / "3DSSG" / "objects.json")).get(scan_id, []):
        for o in sc["objects"]:
            d = dict(o)
            for k in ("nyu40", "eigen13", "rio27"):
                if k in d:
                    d[k] = int(d[k])
            out[int(o["id"])] = d
    return out


def load_relations(root: str | Path, scan_id: str) -> list[tuple[int, int, str]]:
    """[(child, parent, name)], de-duplicated, in file order."""
    seen: set[tuple[int, int, str]] = set()
    out: list[tuple[int, int, str]] = []
    for sc in _index(str(Path(root) / "3DSSG" / "relationships.json")).get(scan_id, []):
        for subj, obj, _rid, name in sc["relationships"]:
            t = (int(subj), int(obj), str(name))
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
