#!/usr/bin/env python3
"""
Generate Zn–N / Zn–O bond-scan structures for all xyz files in ./xyz_files.

For each input XYZ:
- Locate the Zn atom.
- Find the closest N or O atom to Zn (the coordinating donor).
- For each delta in DELTAS (negative => bond compression), move ONLY the Zn
  atom along the donor -> Zn direction by `delta` Å (so a negative delta
  shortens the Zn–donor distance).
- Write out new XYZs named: <basename>_<delta>.xyz, e.g.
  1Zn_1MIm_0MeOH_m0.2.xyz  ("m" denotes a negative delta).
"""

import math
from pathlib import Path
from typing import List, Tuple


XYZ_DIR = Path(__file__).resolve().parent / "xyz_files"

ZN_DONOR_CUTOFF = 3.0

DELTAS = [-0.1, -0.2, -0.3, -0.4]


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

    atoms: List[Atom] = []
    for idx, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line in {path}, line {idx+2}: {line}")
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, x, y, z))

    return header, atoms


def write_xyz(path: Path, header: List[str], atoms: List[Atom]) -> None:
    natoms = len(atoms)
    lines = [str(natoms), header[1] if len(header) > 1 else ""]
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


def find_closest_donor(atoms: List[Atom], zn_idx: int) -> int:
    _, xz, yz, zz = atoms[zn_idx]
    best_idx = -1
    best_d = float("inf")
    for i, (elem, x, y, z) in enumerate(atoms):
        if i == zn_idx:
            continue
        if elem.upper() not in {"N", "O"}:
            continue
        d = distance((xz, yz, zz), (x, y, z))
        if d <= ZN_DONOR_CUTOFF and d < best_d:
            best_d = d
            best_idx = i
    if best_idx < 0:
        raise ValueError("No N or O donor found within cutoff of Zn")
    return best_idx


def normalized_vector(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        return 0.0, 0.0, 0.0
    return dx / norm, dy / norm, dz / norm


def move_zn(
    atoms: List[Atom], zn_idx: int, donor_idx: int, delta: float
) -> List[Atom]:
    """
    Return a new atoms list where Zn is shifted along the donor->Zn direction
    by `delta` Å. A negative delta therefore shortens the Zn–donor bond.
    """
    _, xz, yz, zz = atoms[zn_idx]
    _, xd, yd, zd = atoms[donor_idx]
    vx, vy, vz = normalized_vector((xd, yd, zd), (xz, yz, zz))

    new_atoms = list(atoms)
    elem_zn = atoms[zn_idx][0]
    new_atoms[zn_idx] = (
        elem_zn,
        xz + vx * delta,
        yz + vy * delta,
        zz + vz * delta,
    )
    return new_atoms


def format_delta(delta: float) -> str:
    """Filename-friendly delta: -0.2 -> m0.2, 0.3 -> 0.3."""
    if delta < 0:
        return "m" + f"{abs(delta):.1f}"
    return f"{delta:.1f}"


def process_xyz_file(path: Path) -> None:
    header, atoms = read_xyz(path)
    zn_idx = find_zn_index(atoms)
    donor_idx = find_closest_donor(atoms, zn_idx)

    base = path.stem
    for delta in DELTAS:
        scaled_atoms = move_zn(atoms, zn_idx, donor_idx, delta)
        delta_str = format_delta(delta)
        out_name = f"{base}_{delta_str}.xyz"
        out_path = path.with_name(out_name)
        write_xyz(out_path, header, scaled_atoms)


def main() -> None:
    if not XYZ_DIR.is_dir():
        raise SystemExit(f"XYZ directory not found: {XYZ_DIR}")

    xyz_files = sorted(
        p for p in XYZ_DIR.glob("*.xyz")
        if "_m" not in p.stem
    )
    if not xyz_files:
        raise SystemExit(f"No source xyz files found in {XYZ_DIR}")

    for xyz in xyz_files:
        process_xyz_file(xyz)


if __name__ == "__main__":
    main()
