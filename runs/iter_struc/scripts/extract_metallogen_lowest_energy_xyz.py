#!/usr/bin/env python3
"""
For each ``*/save_dir/gen_struc_before_clustering.xyz`` under a root directory,
read the multi-frame XYZ and write a single output XYZ containing one frame per
id: the structure with the lowest energy ``E=...`` parsed from the comment line
(tie: earliest frame in the file).

Root may be (1) a directory with subdirs that each contain ``save_dir/gen_struc_before_clustering.xyz``,
(2) a directory that directly contains ``save_dir/gen_struc_before_clustering.xyz``, or
(3) a path to a ``gen_struc_before_clustering.xyz`` file.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ATOM_LINE_RE = re.compile(
    r"^\s*([A-Za-z]{1,3})\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
# Total energy in extended XYZ comment, e.g. "chg=2 mult=1 E=-49.05610510532 scale=0.7"
COMMENT_ENERGY_RE = re.compile(
    r"\bE\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\b"
)

REL_SAVE_PATH = Path("save_dir") / "gen_struc_before_clustering.xyz"


@dataclass(frozen=True)
class XYZFrame:
    atoms: list[tuple[str, float, float, float]]
    comment: str

    def to_xyz(self) -> str:
        lines = [str(len(self.atoms)), self.comment]
        for el, x, y, z in self.atoms:
            lines.append(f"{el:<3} {x: .8f} {y: .8f} {z: .8f}")
        return "\n".join(lines) + "\n"


def _parse_atom_line(line: str) -> Optional[tuple[str, float, float, float]]:
    m = ATOM_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    el = m.group(1)
    x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
    return el, x, y, z


def _energy_from_comment(comment: str) -> Optional[float]:
    m = COMMENT_ENERGY_RE.search(comment)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_multi_frame_xyz(path: Path) -> list[tuple[Optional[float], int, XYZFrame]]:
    """
    Parse extended XYZ. Returns list of (energy or None, frame_index, frame)
    for each frame in file order.
    """
    raw = path.read_text(errors="replace").splitlines()
    out: list[tuple[Optional[float], int, XYZFrame]] = []
    i = 0
    frame_idx = 0
    n = len(raw)

    while i < n:
        while i < n and not raw[i].strip():
            i += 1
        if i >= n:
            break
        try:
            natoms = int(raw[i].strip())
        except ValueError:
            break
        if natoms < 0 or i + 1 >= n:
            break
        comment = raw[i + 1].rstrip("\n")
        e = _energy_from_comment(comment)
        atoms: list[tuple[str, float, float, float]] = []
        j = i + 2
        end = min(j + natoms, n)
        while j < end:
            parsed = _parse_atom_line(raw[j])
            if parsed is None:
                break
            atoms.append(parsed)
            j += 1
        if len(atoms) != natoms:
            # Skip malformed tail; stop to avoid misalignment.
            break
        out.append((e, frame_idx, XYZFrame(atoms=atoms, comment=comment)))
        frame_idx += 1
        i = j

    return out


def _id_from_xyz_path(p: Path) -> str:
    # .../<id>/save_dir/gen_struc_before_clustering.xyz
    return p.parent.parent.name


def _iter_before_clustering_xyz(root: Path) -> Iterable[Path]:
    root = root.resolve()
    target_name = "gen_struc_before_clustering.xyz"

    if root.is_file() and root.name == target_name:
        yield root
        return

    cand = root / REL_SAVE_PATH
    if cand.is_file():
        yield cand
        return

    def sort_key(p: Path):
        name = p.parent.parent.name
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    pattern = f"*/{REL_SAVE_PATH.as_posix()}"
    for p in sorted(root.glob(pattern), key=sort_key):
        if p.is_file():
            yield p


def _best_frame_for_xyz(xyz_path: Path) -> Optional[XYZFrame]:
    gid = _id_from_xyz_path(xyz_path)
    frames = _parse_multi_frame_xyz(xyz_path)
    if not frames:
        return None

    with_e: list[tuple[float, int, XYZFrame]] = []
    for e, idx, fr in frames:
        if e is not None and e == e:  # not NaN
            with_e.append((e, idx, fr))

    if with_e:
        with_e.sort(key=lambda t: (t[0], t[1]))
        best_e, best_idx, fr = with_e[0]
        return XYZFrame(
            atoms=fr.atoms,
            comment=f"id={gid} lowest_E={best_e} frame_index={best_idx} | {fr.comment}",
        )

    # No parseable energies: keep first geometry
    _, idx, fr = frames[0]
    return XYZFrame(
        atoms=fr.atoms,
        comment=f"id={gid} frame_index={idx} (no E= in comments) | {fr.comment}",
    )


def write_xyz(frames: list[XYZFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(fr.to_xyz())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Extract lowest-energy frame from each save_dir/gen_struc_before_clustering.xyz "
            "under root (energy from comment line E=...)."
        )
    )
    ap.add_argument(
        "root",
        type=Path,
        help=(
            "Either (1) directory with */save_dir/gen_struc_before_clustering.xyz, "
            "(2) a directory containing save_dir/gen_struc_before_clustering.xyz, "
            "or (3) path to gen_struc_before_clustering.xyz."
        ),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output multi-frame XYZ (default: <root>/lowest_energy.xyz).",
    )
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    if args.output is not None:
        out_path = args.output
    elif root.is_file():
        out_path = root.parent / "lowest_energy.xyz"
    else:
        out_path = root / "lowest_energy.xyz"

    frames: list[XYZFrame] = []
    missing = 0
    for xyz_path in _iter_before_clustering_xyz(root):
        fr = _best_frame_for_xyz(xyz_path)
        if fr is not None:
            frames.append(fr)
        else:
            missing += 1

    if not frames:
        raise SystemExit(f"No parsable structures under: {root}")

    write_xyz(frames, out_path)
    print(f"Wrote {len(frames)} frames to {out_path} (missing={missing}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
