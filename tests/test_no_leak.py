"""No-leak test (CLAUDE.md hard rules 1-2, DESIGN §9 M2).

Method-side code must not import `lsg.oracle` / `lsg.eval` (absolute or relative imports), and
`perception/` must not read rescan annotations. The method side is every package except
`oracle/`, `eval/` and the CLI (which wires everything together).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lsg"
FORBIDDEN_PKGS = ("lsg.oracle", "lsg.eval")
METHOD_SIDE = ["perception", "obs", "change", "graph", "geom", "io", "classes.py", "config.py", "runinfo.py",
               "train_meta.py"]
# rescan-annotation access that perception must never contain
FORBIDDEN_CALLS = {"load_scan", "load_relations", "load_objects", "gt_changes"}
FORBIDDEN_ATTRS = {"rigid", "removed", "nonrigid", "semseg_labels", "vertex_object_id", "face_labels"}
FORBIDDEN_STRINGS = ("semseg", "labels.instances", "3RScan.json")


def _module_name(path: Path, root: Path) -> str:
    """`src/lsg/graph/model.py` -> `lsg.graph.model` (packages: `lsg.graph`)."""
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_modules(path: Path, root: Path = SRC) -> list[tuple[int, str]]:
    """All modules a file imports, relative imports resolved. Returns (line, module)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mod = _module_name(path, root)
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                full = ".".join(base + ([node.module] if node.module else []))
            else:
                full = node.module or ""
            out.append((node.lineno, full))
            out += [(node.lineno, f"{full}.{a.name}") for a in node.names]
    return out


def leaks(path: Path, root: Path = SRC) -> list[str]:
    """Imports of `lsg.oracle` / `lsg.eval` in one file (one finding per import line)."""
    bad: dict[int, str] = {}
    for ln, m in imported_modules(path, root):
        if ln not in bad and any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PKGS):
            bad[ln] = f"{path}:{ln} imports {m}"
    return list(bad.values())


def annotation_reads(path: Path) -> list[str]:
    """Calls / attributes / strings in a file that would read rescan annotations."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in FORBIDDEN_CALLS:
                bad.append(f"{path}:{node.lineno} calls {name}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            bad.append(f"{path}:{node.lineno} uses .{node.attr}")
        elif isinstance(node, ast.ImportFrom):
            bad += [f"{path}:{node.lineno} imports {a.name}" for a in node.names if a.name in FORBIDDEN_CALLS]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(s in node.value for s in FORBIDDEN_STRINGS) and not _is_docstring(node, tree):
                bad.append(f"{path}:{node.lineno} string mentions {node.value[:40]!r}")
    return bad


def _is_docstring(node: ast.Constant, tree: ast.Module) -> bool:
    """True if the constant is a module/class/function docstring (prose may name the rules)."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(n, "body", [])
            if body and isinstance(body[0], ast.Expr) and body[0].value is node:
                return True
    return False


def method_side_files() -> list[Path]:
    """Every .py file of the method side."""
    files = []
    for entry in METHOD_SIDE:
        p = SRC / entry
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    return files


def test_method_side_exists():
    assert (SRC / "perception" / "__init__.py").is_file()
    assert len(method_side_files()) >= 20
    # the real perception code is scanned, not just the package stub (M4)
    names = {f.name for f in (SRC / "perception").glob("*.py")}
    assert {"residual.py", "voxels.py", "cluster.py", "labeler.py", "source.py"} <= names


def test_method_side_never_imports_oracle_or_eval():
    problems = [p for f in method_side_files() for p in leaks(f)]
    assert not problems, "\n".join(problems)


def test_perception_reads_no_rescan_annotations():
    problems = [p for f in sorted((SRC / "perception").rglob("*.py")) for p in annotation_reads(f)]
    assert not problems, "\n".join(problems)


def test_runtime_import_closure_is_clean():
    """Importing the method side must not pull in oracle/eval transitively."""
    code = (
        "import sys, lsg.perception, lsg.perception.source, lsg.perception.residual, lsg.perception.voxels,"
        " lsg.perception.cluster, lsg.perception.labeler, lsg.train_meta, lsg.obs, lsg.change.pipeline, lsg.change.baselines, lsg.change.online,"
        " lsg.change.visibility, lsg.change.association, lsg.change.classify, lsg.change.update, lsg.graph.build;"
        "bad=[m for m in sys.modules if m.startswith(('lsg.oracle','lsg.eval'))];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"leaked modules: {r.stdout} {r.stderr}"


@pytest.mark.parametrize(
    "src, n",
    [
        ("from lsg.eval.gt import gt_changes\n", 1),
        ("import lsg.oracle\n", 1),
        ("from ..oracle import source\n", 1),  # relative, from lsg/perception/x.py
        ("from .. import eval\n", 1),
        ("from lsg.graph import model\nimport numpy\n", 0),
        ("from lsg.evaluation import x\n", 0),  # prefix match must respect the dot
    ],
)
def test_checker_catches_planted_imports(tmp_path, src, n):
    root = tmp_path / "lsg"
    f = root / "perception" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text(src, encoding="utf-8")
    assert len(leaks(f, root)) == n


def test_checker_catches_planted_annotation_reads(tmp_path):
    f = tmp_path / "x.py"
    f.write_text(
        '"""Docstrings may mention semseg."""\n'
        "from lsg.io.scan3r import load_scan\n"
        "def g(meta, s):\n"
        "    S = load_scan(root, s)\n"
        "    return meta[s].rigid, open(s + '/semseg.v2.json')\n",
        encoding="utf-8",
    )
    bad = annotation_reads(f)
    assert any("imports load_scan" in b for b in bad)
    assert any("calls load_scan" in b for b in bad)
    assert any(".rigid" in b for b in bad)
    assert any("semseg" in b for b in bad)
    assert not any(":1 " in b for b in bad)  # the docstring is fine
