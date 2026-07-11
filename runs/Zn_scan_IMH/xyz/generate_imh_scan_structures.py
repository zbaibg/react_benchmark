#!/usr/bin/env python3
"""
Generate IMH translation scans for the two Zn/ImH/water XYZ structures.

For each input XYZ:
- Locate Zn.
- Identify the IMH block as the contiguous atoms from the first N after Zn up
  to the first O atom.
- Use the IMH N closest to Zn as the Zn-N donor.
- Translate the whole IMH block along the Zn -> N direction by each delta.
  Positive deltas increase the Zn-N distance; negative deltas shorten it.
- Write new XYZs next to the inputs named:
  <basename>_IMH_<delta>.xyz, e.g. 1Zn_1ImH_6Wat_1Hbond_IMH_m0.2.xyz.
  The XYZ comment records the source file and applied IMH displacement.
"""

import math
from pathlib import Path
from typing import List, Tuple


XYZ_DIR = Path(__file__).resolve().parent / "xyz_files"

SOURCE_FILES = (
    "1Zn_1ImH_6Wat_1Hbond.xyz",
    "1Zn_1ImH_6Wat_2Hbond.xyz",
)

DELTAS = [
    -0.1,
    0.1,
    0.2,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
]


Atom = Tuple[str, float, float, float]


def read_xyz(path: Path) -> Tuple[List[str], List[Atom]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must be atom count") from exc

    header = lines[:2]
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, found {len(atom_lines)}")

    atoms: List[Atom] = []
    for idx, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line in {path}, line {idx + 2}: {line}")
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, x, y, z))

    return header, atoms


def write_xyz(path: Path, comment: str, atoms: List[Atom]) -> None:
    lines = [str(len(atoms)), comment]
    for elem, x, y, z in atoms:
        lines.append(f"{elem:2s} {x:18.10f} {y:18.10f} {z:18.10f}")
    path.write_text("\n".join(lines) + "\n")


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def find_zn_index(atoms: List[Atom]) -> int:
    for i, (elem, *_rest) in enumerate(atoms):
        if elem.upper() == "ZN":
            return i
    raise ValueError("No Zn atom found in structure")


def find_imh_indices(atoms: List[Atom], zn_idx: int) -> List[int]:
    start_idx = -1
    for i in range(zn_idx + 1, len(atoms)):
        if atoms[i][0].upper() == "N":
            start_idx = i
            break
    if start_idx < 0:
        raise ValueError("No IMH N atom found after Zn")

    end_idx = len(atoms)
    for i in range(start_idx + 1, len(atoms)):
        if atoms[i][0].upper() == "O":
            end_idx = i
            break

    return list(range(start_idx, end_idx))


def find_zn_bound_n(atoms: List[Atom], zn_idx: int, imh_indices: List[int]) -> int:
    _, xz, yz, zz = atoms[zn_idx]
    n_indices = [i for i in imh_indices if atoms[i][0].upper() == "N"]
    if not n_indices:
        raise ValueError("No N atom found in IMH block")

    return min(
        n_indices,
        key=lambda i: distance((xz, yz, zz), (atoms[i][1], atoms[i][2], atoms[i][3])),
    )


def normalized_vector(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return dx / norm, dy / norm, dz / norm


def move_imh(
    atoms: List[Atom], zn_idx: int, donor_idx: int, imh_indices: List[int], delta: float
) -> List[Atom]:
    _, xz, yz, zz = atoms[zn_idx]
    _, xd, yd, zd = atoms[donor_idx]
    vx, vy, vz = normalized_vector((xz, yz, zz), (xd, yd, zd))

    new_atoms = list(atoms)
    for i in imh_indices:
        elem, x, y, z = atoms[i]
        new_atoms[i] = (
            elem,
            x + vx * delta,
            y + vy * delta,
            z + vz * delta,
        )
    return new_atoms


def format_delta(delta: float) -> str:
    """Filename-friendly delta: -0.2 -> m0.2, 0.0 -> 0.0."""
    if delta < 0:
        return "m" + f"{abs(delta):.1f}"
    return f"{delta:.1f}"


def process_xyz_file(path: Path) -> None:
    _header, atoms = read_xyz(path)
    zn_idx = find_zn_index(atoms)
    imh_indices = find_imh_indices(atoms, zn_idx)
    donor_idx = find_zn_bound_n(atoms, zn_idx, imh_indices)

    base = path.stem
    for delta in DELTAS:
        shifted_atoms = move_imh(atoms, zn_idx, donor_idx, imh_indices, delta)
        out_name = f"{base}_IMH_{format_delta(delta)}.xyz"
        comment = f"{out_name}: generated from {path.name}; IMH translated by {delta:+.1f} A along Zn-N"
        write_xyz(path.with_name(out_name), comment, shifted_atoms)


def main() -> None:
    if not XYZ_DIR.is_dir():
        raise SystemExit(f"XYZ directory not found: {XYZ_DIR}")

    for filename in SOURCE_FILES:
        path = XYZ_DIR / filename
        if not path.is_file():
            raise SystemExit(f"Source XYZ file not found: {path}")
        process_xyz_file(path)


if __name__ == "__main__":
    main()
