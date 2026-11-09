"""Split-file helpers (DESIGN §2.3). `lsg splits textures` writes `splits/download_textures.txt`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from lsg.io.scan3r import read_pairs


def reference_ids(split_files: Iterable[str | Path]) -> list[str]:
    """Unique reference scan ids of one or more pair lists, sorted.

    Args:
        split_files: Pair lists such as `splits/eval_val.txt` (`reference rescan` per line).

    Returns:
        Sorted list of distinct reference ids.
    """
    refs: set[str] = set()
    for f in split_files:
        refs.update(r for r, _ in read_pairs(f))
    return sorted(refs)


def write_texture_list(split_files: Iterable[str | Path], out: str | Path) -> list[str]:
    """Write the reference scans whose textures are needed (one id per line).

    Only reference scans need textures (appearance residual renders the reference texture,
    DESIGN §6.6); rescans never do.

    Args:
        split_files: Pair lists to collect references from (dev + val + relchange).
        out: Output file, normally `splits/download_textures.txt`.

    Returns:
        The written ids.
    """
    ids = reference_ids(split_files)
    Path(out).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
    return ids


def subset_pairs(pairs: Iterable[tuple[str, str]], n_refs: int, seed: int = 0) -> list[tuple[str, str]]:
    """Seeded subset: `n_refs` references, one rescan each (DECISIONS M6, Erik: milestone val subset).

    References are drawn without replacement from the sorted reference list, then one rescan
    per drawn reference, both with `numpy.random.default_rng(seed)`; the result keeps the
    original pair order. One pair per room keeps rooms with many rescans from dominating.
    """
    import numpy as np

    pairs = list(pairs)
    by_ref: dict[str, list[str]] = {}
    for r, s in pairs:
        by_ref.setdefault(r, []).append(s)
    refs = sorted(by_ref)
    rng = np.random.default_rng(int(seed))
    pick = sorted(rng.choice(len(refs), size=min(int(n_refs), len(refs)), replace=False).tolist())
    keep = {(refs[i], by_ref[refs[i]][int(rng.integers(len(by_ref[refs[i]])))]) for i in pick}
    return [p for p in pairs if p in keep]


def write_pairs(pairs: Iterable[tuple[str, str]], out: str | Path, header: str = "") -> None:
    """Write a pair list (`reference rescan` per line) with optional `#` header lines."""
    lines = [f"# {h}\n" for h in header.splitlines() if h] + ["# reference_scan rescan\n"]
    lines += [f"{r} {s}\n" for r, s in pairs]
    Path(out).write_text("".join(lines), encoding="utf-8", newline="\n")
