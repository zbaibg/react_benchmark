#!/usr/bin/env python3
"""
Generate H-scan structures for all xyz files in ./xyz_files.

For each input XYZ:
- Detect N–H and O–H bonds by a simple distance cutoff.
- For each specified delta (in Å), adjust the N–H / O–H bond length by that amount.
- Move the H atom along the bond direction, keeping the heavy atom fixed.
- Write out new XYZs named: <basename>_<delta>.xyz (e.g. H3O_monomer_0.2.xyz).

You can customize:
- BOND_DETECTION_CUTOFF: max distance to consider N–H / O–H as a bond
- DELTAS: the scan grid in Å
"""

import math
from pathlib import Path
from typing import List, Tuple


# Directory that holds source xyz files (created by copy.sh)
XYZ_DIR = Path(__file__).resolve().parent / "xyz_files"

# Distance cutoff (Å) to consider N–H or O–H as a bond
BOND_DETECTION_CUTOFF = 1.25

# Scan grid (in Å) to add to each N–H / O–H bond length
DELTAS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def read_xyz(path: Path) -> Tuple[List[str], List[Tuple[str, float, float, float]]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must be atom count") from exc

    header = lines[:2]
    atom_lines = lines[2 : 2 + natoms]

    atoms = []
    for idx, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line in {path}, line {idx+2}: {line}")
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, x, y, z))

    return header, atoms


def write_xyz(path: Path, header: List[str], atoms: List[Tuple[str, float, float, float]]) -> None:
    natoms = len(atoms)
    lines = [str(natoms), header[1] if len(header) > 1 else ""]
    for elem, x, y, z in atoms:
        lines.append(f"{elem:2s} {x:18.10f} {y:18.10f} {z:18.10f}")
    path.write_text("\n".join(lines) + "\n")


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def find_nh_oh_bonds(
    atoms: List[Tuple[str, float, float, float]],
) -> List[Tuple[int, int]]:
    """
    Return a list of (heavy_idx, h_idx) for N–H or O–H bonds.
    Indices are 0-based indices into the atoms list.

    For each heavy atom (N or O), only one bonded H is selected.
    This means that for, e.g., H2O (one O, two H) or H3O+ (one O, three H), only one O–H bond will be stretched.
    """
    bonds: List[Tuple[int, int]] = []
    chosen_h_for_heavy = {}
    for i, (elem_i, xi, yi, zi) in enumerate(atoms):
        if elem_i.upper() not in {"N", "O"}:
            continue
        for j, (elem_j, xj, yj, zj) in enumerate(atoms):
            if elem_j.upper() != "H":
                continue
            d = distance((xi, yi, zi), (xj, yj, zj))
            if d <= BOND_DETECTION_CUTOFF:
                # Only record the first H bonded to each heavy atom
                if i not in chosen_h_for_heavy:
                    chosen_h_for_heavy[i] = j

    for heavy_idx, h_idx in chosen_h_for_heavy.items():
        bonds.append((heavy_idx, h_idx))

    return bonds


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


def apply_delta_to_bonds(
    atoms: List[Tuple[str, float, float, float]],
    bonds: List[Tuple[int, int]],
    delta: float,
) -> List[Tuple[str, float, float, float]]:
    """
    Return new atoms list with all N–H / O–H bonds adjusted by `delta` (Å).
    Heavy atom stays fixed; H is moved along the bond direction.
    """
    new_atoms = list(atoms)

    for heavy_idx, h_idx in bonds:
        elem_h, xh, yh, zh = new_atoms[h_idx]
        elem_heavy, xhe, yhe, zhe = new_atoms[heavy_idx]

        v = normalized_vector((xhe, yhe, zhe), (xh, yh, zh))
        # Move hydrogen by delta along the heavy->H direction
        xh_new = xh + v[0] * delta
        yh_new = yh + v[1] * delta
        zh_new = zh + v[2] * delta

        new_atoms[h_idx] = (elem_h, xh_new, yh_new, zh_new)

    return new_atoms


def format_delta(delta: float) -> str:
    """
    Convert delta to a string suitable for filenames.
    Example: -0.2 -> m0.2, 0.3 -> 0.3, 0.0 -> 0.0
    """
    if delta < 0:
        return "m" + str(abs(delta))
    return str(delta)


def process_xyz_file(path: Path) -> None:
    header, atoms = read_xyz(path)
    bonds = find_nh_oh_bonds(atoms)
    if not bonds:
        return
    base = path.stem

    for delta in DELTAS:
        scaled_atoms = apply_delta_to_bonds(atoms, bonds, delta)
        delta_str = format_delta(delta)
        # If the basename ends with "_monomer", insert the delta before "_monomer"
        # so that names look like: <basename_without_monomer>_<delta>_monomer.xyz
        if base.endswith("_monomer"):
            prefix = base[: -len("_monomer")]
            out_name = f"{prefix}_{delta_str}_monomer.xyz"
        else:
            out_name = f"{base}_{delta_str}.xyz"
        out_path = path.with_name(out_name)
        write_xyz(out_path, header, scaled_atoms)


def main() -> None:
    if not XYZ_DIR.is_dir():
        raise SystemExit(f"XYZ directory not found: {XYZ_DIR}")

    xyz_files = sorted(XYZ_DIR.glob("*.xyz"))
    if not xyz_files:
        raise SystemExit(f"No xyz files found in {XYZ_DIR}")

    for xyz in xyz_files:
        process_xyz_file(xyz)


if __name__ == "__main__":
    main()

