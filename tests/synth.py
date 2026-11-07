"""Synthetic scenes shared by the online-loop and perception tests (real ray casting).

Scenes are built from labelled boxes / quads as `Scan`s; rescan depth is rendered from the
rescan mesh, so unchanged surfaces agree exactly with the reference render.
"""

import numpy as np

from lsg.geom.render import Renderer
from lsg.geom.transforms import apply, invert
from lsg.io.scan3r import Frame, Instance, Intrinsics, Scan

K = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]])
W, H = 224, 172
INTR = Intrinsics(K * 4, (960, 540), K, (W, H), 1000.0, 1)

T_ALIGN = np.eye(4)  # rescan -> reference
_c, _s = np.cos(np.radians(10)), np.sin(np.radians(10))
T_ALIGN[:3, :3] = [[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]]
T_ALIGN[:3, 3] = [0.4, -0.7, 0.05]


def box(cmin, cmax):
    """Closed axis-aligned box mesh (8 vertices, 12 triangles)."""
    x0, y0, z0 = cmin
    x1, y1, z1 = cmax
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)  # fmt: skip
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])  # fmt: skip
    return V, F


def quad(p0, p1, p2, p3):
    """Two-triangle quad."""
    return np.array([p0, p1, p2, p3], float), np.array([[0, 1, 2], [0, 2, 3]])


def make_scan(parts, scan_id, T=None):
    """`Scan` from {objectId: (V, F, nyu40, label)}; vertices mapped with T (default I)."""
    V, F, vo, vn, fl = [], [], [], [], []
    off = 0
    for oid, (v, f, nyu, _) in parts.items():
        V.append(v)
        F.append(f + off)
        vo.append(np.full(len(v), oid))
        vn.append(np.full(len(v), nyu))
        fl.append(np.full(len(f), oid))
        off += len(v)
    V, F = np.concatenate(V), np.concatenate(F)
    if T is not None:
        V = apply(T, V)
    vo, vn, fl = np.concatenate(vo), np.concatenate(vn), np.concatenate(fl)
    inst = {i: Instance(i, lab, n, V[vo == i].mean(0), V[vo == i].min(0), V[vo == i].max(0), int((vo == i).sum()))
            for i, (_, _, n, lab) in parts.items()}  # fmt: skip
    return Scan(scan_id, V, F, vo, vn, fl, vn[F[:, 0]], inst, INTR, {i: p[3] for i, p in parts.items()})


def look_at(eye, target):
    """Camera->world pose (camera x right, y down, z forward), world +z up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


class FakeSeq:
    """Duck-typed `SequenceReader`: depth rendered from the rescan mesh at each pose.

    `poses_ref` are reference-frame camera poses; frames store the rescan-frame pose
    `T_align^-1 @ P`, as a real rescan sequence would.
    """

    def __init__(self, rescan: Scan, poses_ref, T_align=T_ALIGN):
        r = Renderer(rescan.vertices, rescan.faces, rescan.face_labels)
        self.info = INTR
        self.frame_ids = [3 * i for i in range(len(poses_ref))]
        self._frames = {}
        for fid, P in zip(self.frame_ids, poses_ref):
            P_s = invert(T_align) @ P
            d, _ = r.render(K, invert(P_s), W, H)
            self._frames[fid] = Frame(fid, P_s, d.astype(np.float32), f"frame-{fid:06d}.color.jpg")

    def frame(self, idx):
        return self._frames[idx]
