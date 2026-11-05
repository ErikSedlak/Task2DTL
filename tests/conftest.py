import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def random_rigid(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(size=3)
    return T


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def real_data_root():
    root = Path(os.environ.get("LSG_DATA", REPO / "data"))
    if not (root / "3RScan.json").is_file():
        pytest.skip("local 3RScan data not available")
    return root


def box_points(lo, hi, step=0.02):
    """Point grid filling an axis-aligned box (synthetic object geometry)."""
    axes = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def make_scan(objs, scan_id="REF"):
    """Synthetic `Scan` from {objectId: (points, nyu40, label)} (no faces needed)."""
    from lsg.io.scan3r import Instance, Scan

    V = np.concatenate([p for p, _, _ in objs.values()])
    oid = np.concatenate([np.full(len(p), i) for i, (p, _, _) in objs.items()])
    nyu = np.concatenate([np.full(len(p), n) for p, n, _ in objs.values()])
    inst = {
        i: Instance(i, lab, n, p.mean(0), p.min(0), p.max(0), len(p))
        for i, (p, n, lab) in objs.items()
    }
    F = np.zeros((0, 3), np.int64)
    e = np.zeros(0, np.int64)
    return Scan(scan_id, V, F, oid, nyu, e, e, inst, None, {i: lab for i, (_, _, lab) in objs.items()})
