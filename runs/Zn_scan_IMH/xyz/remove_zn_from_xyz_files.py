#!/usr/bin/env python3
"""Create copies of generated XYZ files with the Zn atom removed."""

from pathlib import Path
from typing import List, Tuple


XYZ_DIR = Path(__file__).resolve().parent / "xyz_files"
SOURCE_PREFIX = "1Zn_1ImH_6Wat"
OUTPUT_PREFIX = "1Zn_6Wat"

Atom = Tuple[str, str]


def read_xyz(path: Path) -> Tuple[str, List[Atom]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must be atom count") from exc

    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, found {len(atom_lines)}")

    atoms: List[Atom] = []
    for idx, line in enumerate(atom_lines, start=1):
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Bad atom line in {path}, line {idx + 2}: {line}")
        atoms.append((parts[0], parts[1]))

    return lines[1], atoms


def output_name(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(SOURCE_PREFIX):
        raise ValueError(f"Unexpected source XYZ name: {path.name}")
    return OUTPUT_PREFIX + stem[len(SOURCE_PREFIX) :] + path.suffix


def remove_zn(path: Path) -> None:
    source_comment, atoms = read_xyz(path)
    zn_indices = [idx for idx, (elem, _coords) in enumerate(atoms) if elem.upper() == "ZN"]
    if len(zn_indices) != 1:
        raise ValueError(f"Expected exactly one Zn atom in {path}, found {len(zn_indices)}")

    out_atoms = [atom for idx, atom in enumerate(atoms) if idx != zn_indices[0]]
    out_path = path.with_name(output_name(path))
    comment = f"{out_path.name}: Zn removed from {path.name}; source comment: {source_comment}"
    lines = [str(len(out_atoms)), comment]
    lines.extend(f"{elem} {coords}" for elem, coords in out_atoms)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not XYZ_DIR.is_dir():
        raise SystemExit(f"XYZ directory not found: {XYZ_DIR}")

    for path in sorted(XYZ_DIR.glob(f"{SOURCE_PREFIX}*.xyz")):
        remove_zn(path)


if __name__ == "__main__":
    main()
