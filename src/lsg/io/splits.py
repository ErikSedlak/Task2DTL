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
