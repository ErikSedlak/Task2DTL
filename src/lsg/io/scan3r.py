"""3RScan loaders: metadata (3RScan.json), scan meshes + instances, RGB-D sequences.

Hard rule 1 (CLAUDE.md): `load_scan` reads annotations (PLY objectIds, semseg). Calling it on a
**rescan** is allowed only from `lsg.oracle` and `lsg.eval`. The method side reads rescans only
through `open_sequence` / `iter_frames` (RGB-D + poses).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lsg.geom.transforms import mat4_from_colmajor

# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidEntry:
    """A moved instance. `T` is None for the bare-int form (moved, transform unknown)."""

    id: int
    T: np.ndarray | None


@dataclass
class ScanMeta:
    """Everything 3RScan.json says about one scan.

    For a reference scan, `reference_id == scan_id`, `T_align` is the identity and the change
    lists are empty. For a rescan, the change lists (`rigid`, `removed`, `nonrigid`) are
    ground truth: hard rule 1 allows reading them only in `lsg.oracle` / `lsg.eval`.
    """

    scan_id: str
    reference_id: str
    is_reference: bool
    split: str | None
    T_align: np.ndarray  # rescan mesh frame -> reference mesh frame (identity for references)
    rigid: list[RigidEntry] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    nonrigid: list[int] = field(default_factory=list)


@dataclass
class Meta:
    """All of 3RScan.json, indexed by scan id: `meta["<scan id>"]` gives its `ScanMeta`."""

    scans: dict[str, ScanMeta]

    def __getitem__(self, scan_id: str) -> ScanMeta:
        """Metadata of one scan (reference or rescan). Raises KeyError if unknown."""
        return self.scans[scan_id]

    def __contains__(self, scan_id: str) -> bool:
        """`scan_id in meta`."""
        return scan_id in self.scans


def _parse_rigid(entry) -> RigidEntry:
    """One item of a rescan's `rigid` list: a dict with a transform, or a bare object id.

    The transform maps reference-frame instance points into the rescan frame (verified at M0).
    """
    if isinstance(entry, dict):
        ir, ires = int(entry["instance_reference"]), int(entry["instance_rescan"])
        if ir != ires:  # never observed; fail loudly if it happens
            raise ValueError(f"rigid entry with instance_reference {ir} != instance_rescan {ires}")
        t = entry.get("transform")
        return RigidEntry(ir, mat4_from_colmajor(t) if t is not None else None)
    return RigidEntry(int(entry), None)


def parse_meta(records: list[dict]) -> Meta:
    """Build a `Meta` from the already-loaded JSON list of 3RScan.json.

    Each record is one reference scan with a `scans` list of its rescans. Rescans without a
    `transform` get the identity.
    """
    scans: dict[str, ScanMeta] = {}
    for rec in records:
        ref = rec["reference"]
        scans[ref] = ScanMeta(ref, ref, True, rec.get("type"), np.eye(4))
        for s in rec.get("scans", []):
            sid = s["reference"]
            t = s.get("transform")
            scans[sid] = ScanMeta(
                scan_id=sid,
                reference_id=ref,
                is_reference=False,
                split=rec.get("type"),
                T_align=mat4_from_colmajor(t) if t is not None else np.eye(4),
                rigid=[_parse_rigid(e) for e in s.get("rigid", [])],
                removed=[int(i) for i in s.get("removed", [])],
                nonrigid=[int(i) for i in s.get("nonrigid", [])],
            )
    return Meta(scans)


def load_meta(root: str | Path) -> Meta:
    """Read `<root>/3RScan.json` (~3 MB, 0.04 s) into a `Meta`."""
    with open(Path(root) / "3RScan.json", encoding="utf-8") as f:
        return parse_meta(json.load(f))


def read_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Read a pair list from `splits/` (e.g. `dev_train.txt`).

    Format: one `reference_scan rescan` pair per line; text after `#` is ignored.

    Returns:
        `[(reference_id, rescan_id), ...]` in file order.
    """
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            r, s = line.split()[:2]
            pairs.append((r, s))
    return pairs


# --------------------------------------------------------------------------------------------
# PLY (ASCII, 3RScan layout)
# --------------------------------------------------------------------------------------------


@dataclass
class PlyData:
    """Raw content of a PLY file: one array per vertex property, plus the triangles."""

    vertex: dict[str, np.ndarray]  # property name -> (N,) array
    faces: np.ndarray  # (M, 3) int64


def read_ply(path: str | Path) -> PlyData:
    """Read an ASCII PLY such as `labels.instances.annotated.v2.ply`.

    Supports what 3RScan uses: one vertex element with scalar properties (x, y, z, red, green,
    blue, objectId, globalId, NYU40, Eigen13, RIO27) and triangle faces. Anything else raises
    ValueError rather than being silently misread.

    Returns:
        `PlyData` with `vertex["x"]`, `vertex["objectId"]`, ... as float64 arrays and
        `faces` as (M, 3) int64.
    """
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    body_start = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    if not any(h.strip() == "format ascii 1.0" for h in header):
        raise ValueError(f"{path}: only ASCII PLY is supported")
    n_vert = n_face = 0
    vprops: list[str] = []
    current = None
    for h in header:
        tok = h.split()
        if not tok:
            continue
        if tok[0] == "element":
            current = tok[1]
            if current == "vertex":
                n_vert = int(tok[2])
            elif current == "face":
                n_face = int(tok[2])
        elif tok[0] == "property" and current == "vertex":
            vprops.append(tok[-1])
    vals = np.fromstring(raw[body_start:].decode("ascii"), sep=" ")  # noqa: NPY201 (text mode)
    nv = n_vert * len(vprops)
    v = vals[:nv].reshape(n_vert, len(vprops))
    f = vals[nv:]
    if f.size != n_face * 4:
        raise ValueError(f"{path}: only triangle faces supported ({f.size} vs {n_face * 4})")
    f = f.reshape(n_face, 4)
    if not np.all(f[:, 0] == 3):
        raise ValueError(f"{path}: non-triangle face")
    return PlyData({p: v[:, i] for i, p in enumerate(vprops)}, f[:, 1:].astype(np.int64))


def face_majority(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face label = majority of its 3 vertex labels; with 3 different labels, vertex 0 wins."""
    a, b, c = (vertex_labels[faces[:, k]] for k in range(3))
    return np.where((b == c) & (a != b), b, a)


# --------------------------------------------------------------------------------------------
# Sequences (RGB-D + poses), read straight from sequence.zip or from an extracted folder
# --------------------------------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Camera parameters of a scan's sequence, from `_info.txt`.

    Color (960x540) and depth (224x172) see the same field of view but have separate 3x3
    intrinsics. Sizes are (width, height) of the images as stored (rotated 90°).
    """

    K_color: np.ndarray  # 3x3
    color_wh: tuple[int, int]
    K_depth: np.ndarray  # 3x3, use for depth (CLAUDE.md)
    depth_wh: tuple[int, int]
    depth_shift: float
    n_frames: int


def parse_info(text: str) -> Intrinsics:
    """Parse the `key = value` lines of a sequence's `_info.txt` into `Intrinsics`."""
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    def K(name: str) -> np.ndarray:
        """3x3 intrinsics from a 16-number `m_calibration...Intrinsic` line (stored 4x4)."""
        return np.array([float(x) for x in kv[name].split()]).reshape(4, 4)[:3, :3]

    return Intrinsics(
        K_color=K("m_calibrationColorIntrinsic"),
        color_wh=(int(kv["m_colorWidth"]), int(kv["m_colorHeight"])),
        K_depth=K("m_calibrationDepthIntrinsic"),
        depth_wh=(int(kv["m_depthWidth"]), int(kv["m_depthHeight"])),
        depth_shift=float(kv["m_depthShift"]),
        n_frames=int(kv.get("m_frames.size", "0")),
    )


def read_pgm16(data: bytes) -> np.ndarray:
    """Decode a binary (P5) PGM image, the format of `frame-XXXXXX.depth.pgm`.

    16-bit samples are big-endian (PGM spec; verified on 3RScan, M0). Header comments are
    skipped.

    Args:
        data: The raw file bytes.

    Returns:
        (H, W) uint16 array of raw values (for depth: millimetres, 0 = invalid).
    """
    tokens: list[bytes] = []
    pos = 0
    while len(tokens) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        start = pos
        while not data[pos : pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1  # single whitespace after maxval
    if tokens[0] != b"P5":
        raise ValueError("not a P5 PGM")
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    dt = ">u2" if maxval > 255 else "u1"
    n = w * h
    return np.frombuffer(data, dtype=dt, count=n, offset=pos).reshape(h, w).astype(np.uint16)


_FRAME_RE = re.compile(r"frame-(\d{6})\.pose\.txt$")


@dataclass
class Frame:
    """One RGB-D frame. The color image is loaded on demand via `SequenceReader.color(idx)`."""

    idx: int
    pose_c2w: np.ndarray  # 4x4 camera -> that scan's mesh frame
    depth: np.ndarray  # (H, W) float32 metres, 0 = invalid, stored (rotated) orientation
    color_name: str


class SequenceReader:
    """Reads a scan's sequence from `sequence/` if extracted, else directly from `sequence.zip`.

    Holds the zip file open; use it as a context manager (`with open_sequence(...) as seq:`)
    or call `close()`. `frame_ids` lists the available frames, `info` the intrinsics.
    """

    def __init__(self, scan_dir: str | Path):
        """
        Args:
            scan_dir: The scan's folder, `<data root>/<scan id>`.

        Raises:
            FileNotFoundError: If neither `sequence/_info.txt` nor `sequence.zip` exists.
        """
        self.scan_dir = Path(scan_dir)
        folder = self.scan_dir / "sequence"
        self._zip: zipfile.ZipFile | None = None
        if (folder / "_info.txt").is_file():
            self._folder: Path | None = folder
            names = [p.name for p in folder.iterdir()]
        elif (self.scan_dir / "sequence.zip").is_file():
            self._folder = None
            self._zip = zipfile.ZipFile(self.scan_dir / "sequence.zip")
            self._zip_names = {Path(n).name: n for n in self._zip.namelist()}
            names = list(self._zip_names)
        else:
            raise FileNotFoundError(f"no sequence/ or sequence.zip in {self.scan_dir}")
        self.frame_ids = sorted(int(m.group(1)) for n in names if (m := _FRAME_RE.search(n)))
        self.info = parse_info(self.read_bytes("_info.txt").decode("ascii"))

    @property
    def source(self) -> str:
        """Where frames come from: "folder" or "zip"."""
        return "folder" if self._folder is not None else "zip"

    def read_bytes(self, name: str) -> bytes:
        """Raw bytes of one sequence file, e.g. "frame-000010.pose.txt"."""
        if self._folder is not None:
            return (self._folder / name).read_bytes()
        assert self._zip is not None
        return self._zip.read(self._zip_names[name])

    def pose(self, idx: int) -> np.ndarray:
        """4x4 camera->world pose of frame `idx`, world = this scan's own mesh frame.

        To place a rescan camera in the reference frame use `T_align @ pose`.
        """
        txt = self.read_bytes(f"frame-{idx:06d}.pose.txt").decode("ascii")
        return np.array([float(x) for x in txt.split()], dtype=np.float64).reshape(4, 4)

    def depth(self, idx: int) -> np.ndarray:
        """(172, 224) float32 depth of frame `idx` in metres (stored orientation), 0 = invalid."""
        raw = read_pgm16(self.read_bytes(f"frame-{idx:06d}.depth.pgm"))
        return (raw.astype(np.float32) / np.float32(self.info.depth_shift))

    def color(self, idx: int) -> np.ndarray:
        """RGB image of frame `idx`: (540, 960, 3) uint8 in stored (rotated 90°) orientation.

        Rotate with `np.rot90(img, k=-1)` to see it upright (only needed for viewing and
        appearance models; geometry always uses the stored orientation).
        """
        from PIL import Image

        with Image.open(io.BytesIO(self.read_bytes(f"frame-{idx:06d}.color.jpg"))) as im:
            return np.asarray(im.convert("RGB"))

    def frame(self, idx: int) -> Frame:
        """Pose + depth of frame `idx` bundled as a `Frame`."""
        return Frame(idx, self.pose(idx), self.depth(idx), f"frame-{idx:06d}.color.jpg")

    def close(self) -> None:
        """Close the zip file (no-op for an extracted folder)."""
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> SequenceReader:
        """Support `with SequenceReader(...) as seq:`."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the zip when the `with` block ends."""
        self.close()


def open_sequence(root: str | Path, scan_id: str) -> SequenceReader:
    """Open the RGB-D sequence of a scan. Safe for rescans on the method side (no labels)."""
    return SequenceReader(Path(root) / scan_id)


def iter_frames(
    seq: SequenceReader, stride: int = 1, idx: Sequence[int] | None = None
) -> Iterator[Frame]:
    """Loop over a sequence's frames, skipping frames whose pose is not finite.

    Args:
        seq: An open sequence.
        stride: Take every `stride`-th frame (the method uses 3).
        idx: Explicit frame ids; overrides `stride` (used for contiguous chunks).

    Yields:
        `Frame` objects in order.
    """
    ids = list(idx) if idx is not None else seq.frame_ids[::stride]
    for i in ids:
        f = seq.frame(i)
        if np.all(np.isfinite(f.pose_c2w)):
            yield f


# --------------------------------------------------------------------------------------------
# Scan mesh + instances (annotation reader, see module docstring)
# --------------------------------------------------------------------------------------------


@dataclass
class Instance:
    """One annotated object of a scan, summarised from its mesh vertices (scan's own frame)."""

    id: int
    label: str
    nyu40: int
    centroid: np.ndarray  # mean of instance vertices
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    n_vertices: int


@dataclass
class Scan:
    """A scan's labelled mesh: geometry, per-vertex and per-face labels, and its objects."""

    scan_id: str
    vertices: np.ndarray  # (N, 3) float64, scan's own mesh frame
    faces: np.ndarray  # (M, 3) int64
    vertex_object_id: np.ndarray  # (N,) int64, 0 = unlabelled
    vertex_nyu40: np.ndarray  # (N,) int64
    face_labels: np.ndarray  # (M,) int64 objectId by majority
    face_nyu40: np.ndarray  # (M,) int64 by majority
    instances: dict[int, Instance]
    intrinsics: Intrinsics | None
    semseg_labels: dict[int, str] = field(default_factory=dict)  # objectId -> label, semseg.v2.json

    def instance_points(self, object_id: int) -> np.ndarray:
        """(K, 3) mesh vertices labelled `object_id` (empty if the object is not in this scan)."""
        return self.vertices[self.vertex_object_id == object_id]


def load_scan(root: str | Path, scan_id: str, with_intrinsics: bool = True) -> Scan:
    """Load a scan's labelled mesh and its object list (~2 s for a large room).

    Annotation reader: on a rescan, only `lsg.oracle` / `lsg.eval` may call this (module
    docstring, hard rule 1).

    Args:
        root: Data root.
        scan_id: Scan to load; reads `labels.instances.annotated.v2.ply` and `semseg.v2.json`.
        with_intrinsics: Also read `_info.txt` from the sequence (None if there is none).

    Returns:
        `Scan` with vertices/faces, objectId and NYU40 per vertex and per face (face = majority
        of its 3 vertices), `instances` for every objectId > 0 with vertices, and
        `semseg_labels` = all objectIds of `semseg.v2.json` (O_R / O_S).
    """
    d = Path(root) / scan_id
    ply = read_ply(d / "labels.instances.annotated.v2.ply")
    V = np.stack([ply.vertex["x"], ply.vertex["y"], ply.vertex["z"]], axis=1)
    oid = ply.vertex["objectId"].astype(np.int64)
    nyu = ply.vertex["NYU40"].astype(np.int64)
    F = ply.faces

    labels: dict[int, str] = {}
    semseg = d / "semseg.v2.json"
    if semseg.is_file():
        for g in json.loads(semseg.read_text(encoding="utf-8"))["segGroups"]:
            labels[int(g["objectId"])] = g["label"]

    instances: dict[int, Instance] = {}
    order = np.argsort(oid, kind="stable")
    ids, starts = np.unique(oid[order], return_index=True)
    bounds = list(starts[1:]) + [len(order)]
    for i, s, e in zip(ids, starts, bounds):
        if i == 0:
            continue
        sel = order[s:e]
        P = V[sel]
        vals, counts = np.unique(nyu[sel], return_counts=True)
        instances[int(i)] = Instance(
            id=int(i),
            label=labels.get(int(i), ""),
            nyu40=int(vals[np.argmax(counts)]),
            centroid=P.mean(axis=0),
            aabb_min=P.min(axis=0),
            aabb_max=P.max(axis=0),
            n_vertices=len(sel),
        )

    intr = None
    if with_intrinsics:
        try:
            with SequenceReader(d) as seq:
                intr = seq.info
        except FileNotFoundError:
            intr = None

    return Scan(
        scan_id=scan_id,
        vertices=V,
        faces=F,
        vertex_object_id=oid,
        vertex_nyu40=nyu,
        face_labels=face_majority(oid, F),
        face_nyu40=face_majority(nyu, F),
        instances=instances,
        intrinsics=intr,
        semseg_labels=labels,
    )


# --------------------------------------------------------------------------------------------
# Textured reference mesh (OBJ + texture atlas; reference scans only, M6)
# --------------------------------------------------------------------------------------------


@dataclass
class TexturedMesh:
    """A reference scan's textured mesh (`mesh.refined.v2.obj` + `mesh.refined_0.png`).

    Geometry is in the scan's own mesh frame (as the PLY). `corner_uv[k, j]` is the texture
    coordinate (u, v in [0, 1], v = 0 at the image bottom, OBJ convention) of corner j of face k.
    """

    scan_id: str
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64
    corner_uv: np.ndarray  # (M, 3, 2) float32
    texture: np.ndarray  # (H, W, 3) uint8


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the triangles of a Wavefront OBJ with texture coordinates.

    Supports `v`, `vt` and triangle `f` lines of the forms `a/b/c`, `a/b` and `a//c`
    (1-based indices; negative indices are not used by 3RScan and raise). Other lines
    (`vn`, `mtllib`, `usemtl`, comments) are ignored; one material is assumed.

    Returns:
        vertices (N, 3) float64, faces (M, 3) int64 (0-based vertex indices), corner_uv
        (M, 3, 2) float32 (zeros where a face corner has no `vt`).
    """
    V: list[str] = []
    VT: list[str] = []
    F: list[str] = []
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append(line[2:])
            elif line.startswith("vt "):
                VT.append(line[3:])
            elif line.startswith("f "):
                F.append(line[2:])
    verts = np.array(" ".join(V).split(), dtype=np.float64).reshape(-1, 3) if V else np.zeros((0, 3))
    uv = np.array([s.split()[:2] for s in VT], dtype=np.float32) if VT else np.zeros((0, 2), np.float32)
    toks = [t for s in F for t in s.split()]
    if len(toks) != 3 * len(F):
        raise ValueError(f"{path}: only triangle faces are supported")
    vi = np.empty(len(toks), np.int64)
    ti = np.full(len(toks), -1, np.int64)
    for k, t in enumerate(toks):
        parts = t.split("/")
        vi[k] = int(parts[0])
        if len(parts) > 1 and parts[1]:
            ti[k] = int(parts[1])
    if (vi <= 0).any() or (ti == 0).any() or (ti < -1).any():
        raise ValueError(f"{path}: negative or zero OBJ indices are not supported")
    faces = (vi - 1).reshape(-1, 3)
    corner = np.zeros((len(toks), 2), np.float32)
    has = ti > 0
    if has.any():
        corner[has] = uv[ti[has] - 1]
    return verts, faces, corner.reshape(-1, 3, 2)


def load_textured_mesh(root: str | Path, scan_id: str) -> TexturedMesh:
    """Load `mesh.refined.v2.obj` + `mesh.refined_0.png` of a **reference** scan (~3 s).

    The texture is reference-scan data (method input); rescans have no textures here.
    """
    from PIL import Image

    d = Path(root) / scan_id
    V, F, UV = read_obj(d / "mesh.refined.v2.obj")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(d / "mesh.refined_0.png") as im:
        tex = np.asarray(im.convert("RGB"))
    return TexturedMesh(scan_id, V, F, UV, tex)
