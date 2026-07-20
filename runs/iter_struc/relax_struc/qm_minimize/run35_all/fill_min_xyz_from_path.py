#!/usr/bin/env python3
"""For each immediate subdirectory of run35_all: if min.xyz is missing, write the last
frame of path.xyz as min.xyz. Prints the names of directories that were updated.

Safe to re-run: only affects subdirs where min.xyz does not exist."""

from __future__ import annotations

import sys
from pathlib import Path


def last_xyz_frame_lines(lines: list[str]) -> list[str]:
    """Extract the last XYZ frame from a list of lines."""
    frames: list[list[str]] = []
    i, n = 0, len(lines)
    while i < n:
        # Skip empty lines
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            natoms = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(f"Invalid atom count at line {i + 1}: {lines[i]!r}") from e
        need = i + natoms + 2
        if need > n:
            raise ValueError(
                f"Incomplete frame starting at line {i + 1}: "
                f"need {natoms + 2} lines, only {n - i} left"
            )
        frames.append(lines[i:need])
        i = need
    if not frames:
        raise ValueError("No XYZ frames found")
    return frames[-1]


def main() -> int:
    base = Path(__file__).resolve().parent
    copied: list[str] = []
    skipped_no_path: list[str] = []
    errors: list[tuple[str, str]] = []

    for sub in sorted(base.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        if sub.name.startswith("."):
            continue
        min_xyz = sub / "min.xyz"
        path_xyz = sub / "path.xyz"
        if min_xyz.exists():
            continue
        if not path_xyz.is_file():
            skipped_no_path.append(sub.name)
            continue
        try:
            text = path_xyz.read_text(encoding="utf-8", errors="replace")
            last = last_xyz_frame_lines(text.splitlines())
            min_xyz.write_text("\n".join(last) + "\n", encoding="utf-8")
        except (OSError, ValueError) as e:
            errors.append((sub.name, str(e)))
            continue
        copied.append(sub.name)

    if copied:
        print("Wrote min.xyz from the last frame of path.xyz in directories:")
        for name in copied:
            print(f"  {name}")
        print(f"Total: {len(copied)}.")
    else:
        print("No directories needed min.xyz to be written (either min.xyz already exists, or path.xyz is missing).")

    if skipped_no_path:
        print("Skipped the following directories (min.xyz missing and no path.xyz found):", file=sys.stderr)
        for name in skipped_no_path:
            print(f"  {name}", file=sys.stderr)

    if errors:
        print("Failed to process the following directories:", file=sys.stderr)
        for name, msg in errors:
            print(f"  {name}: {msg}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
