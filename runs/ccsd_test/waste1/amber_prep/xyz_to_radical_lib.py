#!/usr/bin/env python3
"""
Build an Amber .lib for an unsupported_mol/small ion from an XYZ file.

Reads geometry from XYZ, infers bonds (distance-based or MDAnalysis guess),
computes equilibrium bond lengths, angles, and dihedrals, defines custom atom
types, and writes frcmod + tleap script then runs tleap to produce RESNAME.lib.

Intended for species that are awkward with antechamber (e.g. H3O+). Force
constants are placeholders; geometry is taken from the XYZ. Use with QM later.

Example:
  python xyz_to_radical_lib.py xyz/xyz_files/H3O_monomer.xyz -r H3O -q 1 -o .
  # If tleap not in PATH: add --no-run-tleap and run tleap manually in amber env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Optional: use MDAnalysis for XYZ + bond guessing
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

# Covalent radii (single bond, Angstrom) for bond guessing if not using MDAnalysis
COV_R = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39}
# Default for unknown
COV_R_DEFAULT = 0.77

# Element to mass (approx)
MASS = {"H": 1.008, "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "S": 32.07, "Cl": 35.45, "Br": 79.90, "I": 126.9}
MASS_DEFAULT = 12.0

# Placeholder LJ for "topology only" (small so they don't dominate if used in MM)
NONBON_DEFAULT = (1.5, 0.1)   # Rmin/2, epsilon
NONBON_H = (0.0, 0.0)


def parse_xyz_simple(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse XYZ without MDAnalysis: return (elements, positions)."""
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    # lines[1] = comment; lines[2:2+n] = sym x y z
    pos = []
    elts = []
    for i in range(n):
        parts = lines[2 + i].split()
        elts.append(parts[0].capitalize())
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elts, np.array(pos)


def guess_bonds_simple(elements: list[str], pos: np.ndarray, fudge: float = 1.3) -> list[tuple[int, int]]:
    """Guess bonds from distances: (i,j) if d < fudge * (r_i + r_j)."""
    n = len(elements)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            r_i = COV_R.get(elements[i], COV_R_DEFAULT)
            r_j = COV_R.get(elements[j], COV_R_DEFAULT)
            d = np.linalg.norm(pos[i] - pos[j])
            if d < fudge * (r_i + r_j):
                bonds.append((i, j))
    return bonds


def load_xyz_and_bonds(path: Path, bond_fudge: float = 1.3) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    """Load XYZ and bond list. Prefer MDAnalysis if available and use its guess_bonds."""
    if HAS_MDA:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                u = mda.Universe(str(path), guess_bonds=True)
            pos = u.atoms.positions.copy()
            if hasattr(u.atoms, "elements") and u.atoms.elements is not None:
                elements = list(u.atoms.elements)
            else:
                elements = [a.name for a in u.atoms]
            if hasattr(u.atoms, "bonds") and u.atoms.bonds is not None:
                bonds = [(int(b[0].index), int(b[1].index)) for b in u.atoms.bonds]
            else:
                u.atoms.guess_bonds()
                bonds = [(int(b[0].index), int(b[1].index)) for b in u.atoms.bonds]
            return elements, pos, bonds
        except Exception:
            pass
    elements, pos = parse_xyz_simple(path)
    bonds = guess_bonds_simple(elements, pos, fudge=bond_fudge)
    return elements, pos, bonds


def bond_length(pos: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pos[i] - pos[j]))


def angle_rad(pos: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle i-j-k in radians."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def angle_deg(pos: np.ndarray, i: int, j: int, k: int) -> float:
    return np.degrees(angle_rad(pos, i, j, k))


def dihedral_rad(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral i-j-k-l in radians."""
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    m = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    return float(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


def dihedral_deg(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    return np.degrees(dihedral_rad(pos, i, j, k, l))


def build_angles_dihedrals(bonds: list[tuple[int, int]], n_atoms: int) -> tuple[list, list]:
    """From bond list, build (angle_list, dihedral_list). Each item is tuple of indices."""
    bond_set = set()
    for i, j in bonds:
        bond_set.add((min(i, j), max(i, j)))
    # Neighbors per atom
    neighbors: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    # Angles: (a, c, b) where c is central, (a,c) and (c,b) are bonds; each angle once
    angles_seen: set[tuple[int, int, int]] = set()
    angles = []
    for c in range(n_atoms):
        adj = neighbors[c]
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                a, b = adj[i], adj[j]
                key = (min(a, b), c, max(a, b))
                if key not in angles_seen:
                    angles_seen.add(key)
                    angles.append((a, c, b))
    # Dihedrals: (a, b, c, d) where a-b, b-c, c-d are bonds; avoid duplicates
    dihedrals_seen: set[tuple[int, int, int, int]] = set()
    dihedrals = []
    for a, b in bonds:
        for c in neighbors[b]:
            if c == a:
                continue
            for d in neighbors[c]:
                if d == b:
                    continue
                key = (a, b, c, d) if a < d else (d, c, b, a)
                if key not in dihedrals_seen:
                    dihedrals_seen.add(key)
                    dihedrals.append((a, b, c, d))
    return angles, dihedrals


def assign_atom_types(elements: list[str]) -> list[str]:
    """Assign custom type per atom: E0, E1, ... (E = element symbol, index per atom)."""
    return [f"{e}{i}" for i, e in enumerate(elements)]


def write_frcmod(
    path: Path,
    type_names: list[str],
    elements: list[str],
    bonds: list[tuple[int, int]],
    angles: list,
    dihedrals: list,
    pos: np.ndarray,
    k_bond: float = 400.0,
    k_angle: float = 55.0,
    k_dihe: float = 1.0,
    n_dihe: int = 3,
    comment: str = "",
) -> None:
    """Write Amber frcmod with geometry from xyz; force constants are placeholders."""
    lines = [f"# {comment or 'Generated from XYZ'}", "MASS"]
    seen_mass = set()
    for t, e in zip(type_names, elements):
        if t in seen_mass:
            continue
        seen_mass.add(t)
        m = MASS.get(e, MASS_DEFAULT)
        lines.append(f"{t:6s}  {m:.3f}  0.0")
    lines.append("")
    if bonds:
        lines.append("BOND")
        for i, j in bonds:
            ti, tj = type_names[i], type_names[j]
            req = bond_length(pos, i, j)
            lines.append(f"{ti}-{tj}  {k_bond:.1f}  {req:.4f}")
        lines.append("")
    if angles:
        lines.append("ANGLE")
        for a, b, c in angles:
            ta, tb, tc = type_names[a], type_names[b], type_names[c]
            theq = angle_deg(pos, a, b, c)
            lines.append(f"{ta}-{tb}-{tc}  {k_angle:.1f}  {theq:.2f}")
        lines.append("")
    if dihedrals:
        lines.append("DIHE")
        for a, b, c, d in dihedrals:
            ta, tb, tc, td = type_names[a], type_names[b], type_names[c], type_names[d]
            phi_eq = np.degrees(dihedral_rad(pos, a, b, c, d))
            # Amber: type1-type2-type3-type4  idiv  pk  phase  period
            lines.append(f"{ta}-{tb}-{tc}-{td}  1  {k_dihe:.2f}  {phi_eq:.1f}  {n_dihe}")
        lines.append("")
    lines.append("NONBON")
    seen_nb = set()
    for t, e in zip(type_names, elements):
        if t in seen_nb:
            continue
        seen_nb.add(t)
        r, eps = NONBON_H if e == "H" else NONBON_DEFAULT
        lines.append(f"{t:6s}  {r:.4f}  {eps:.4f}")
    path.write_text("\n".join(lines) + "\n")


def write_leap(
    path: Path,
    resname: str,
    type_names: list[str],
    elements: list[str],
    pos: np.ndarray,
    bonds: list[tuple[int, int]],
    total_charge: float = 0.0,
) -> None:
    """Write tleap script: load params, create residue with positions from xyz, saveOff."""
    n = len(elements)
    charge_per_atom = total_charge / n if n else 0.0
    lines = [
        f"logFile create_{resname}.log",
        f"loadamberparams frcmod.{resname}",
        "",
        "addAtomTypes {",
    ]
    for t, e in zip(type_names, elements):
        lines.append(f'    {{ "{t}" "{e}" "sp3" }}')
    lines.append("}")
    lines.append("")
    # createAtom: atom name = element + per-element index from 1 (e.g. O1, H1, H2, H3)
    elem_count: dict[str, int] = {}
    atom_names = []
    for e in elements:
        elem_count[e] = elem_count.get(e, 0) + 1
        atom_names.append(f"{e}{elem_count[e]}")
    atom_vars = []
    for i in range(n):
        v = f"a{i}"
        atom_vars.append(v)
        lines.append(f"{v} = createAtom  {atom_names[i]}  {type_names[i]}  {charge_per_atom:.5f}")
    lines.append("")
    for i, v in enumerate(atom_vars):
        lines.append(f"set {v} element \"{elements[i]}\"")
    lines.append("")
    for i, v in enumerate(atom_vars):
        x, y, z = pos[i]
        lines.append(f"set {v} position {{  {x:.5f}  {y:.5f}  {z:.5f} }}")
    lines.append("")
    lines.append(f"r = createResidue {resname}")
    for v in atom_vars:
        lines.append(f"add r {v}")
    lines.append("")
    for i, j in bonds:
        lines.append(f"bond {atom_vars[i]} {atom_vars[j]}")
    lines.append("")
    lines.append(f"{resname} = createUnit {resname}")
    lines.append(f"add {resname} r")
    lines.append("")
    lines.append(f"saveOff {resname} {resname}.lib")
    lines.append("quit")
    path.write_text("\n".join(lines) + "\n")


def run_tleap(work_dir: Path, resname: str, tleap_cmd: str = "tleap") -> bool:
    """Run tleap -f create_RESNAME.leap in work_dir; return True if RESNAME.lib exists.
    Ensure tleap is in PATH (e.g. conda activate amber) or use --no-run-tleap and run tleap manually."""
    leap_file = work_dir / f"create_{resname}.leap"
    if not leap_file.exists():
        return False
    out = subprocess.run(
        [tleap_cmd, "-f", str(leap_file)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr or out.stdout or "")
        return False
    return (work_dir / f"{resname}.lib").exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Amber .lib for an unsupported_mol from XYZ (geometry from file; FF params placeholder)."
    )
    parser.add_argument("xyz", type=Path, help="Input XYZ file")
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for frcmod, leap script, and .lib (default: current)",
    )
    parser.add_argument(
        "-r", "--resname",
        default=None,
        help="Residue/unit name (default: stem of xyz, e.g. H3O_monomer -> H3O)",
    )
    parser.add_argument(
        "-q", "--charge",
        type=float,
        default=0.0,
        help="Total charge (default 0); distributed evenly over atoms",
    )
    parser.add_argument(
        "--no-run-tleap",
        action="store_true",
        help="Only write frcmod and leap script; do not run tleap",
    )
    parser.add_argument(
        "--tleap-cmd",
        default="tleap",
        help="tleap executable (default: tleap)",
    )
    parser.add_argument(
        "--bond-fudge",
        type=float,
        default=1.3,
        help="Bond guess distance = fudge * (r_i + r_j) (default 1.3)",
    )
    args = parser.parse_args()

    xyz_path = args.xyz.resolve()
    if not xyz_path.exists():
        print(f"Error: XYZ not found: {xyz_path}", file=sys.stderr)
        return 1

    resname = args.resname
    if resname is None:
        stem = xyz_path.stem
        # e.g. H3O_monomer -> H3O
        resname = stem.replace("_monomer", "").replace("_", "").upper() or "MOL"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    elements, pos, bonds = load_xyz_and_bonds(xyz_path, bond_fudge=args.bond_fudge)
    n = len(elements)
    if n == 0:
        print("Error: no atoms in XYZ", file=sys.stderr)
        return 1

    angles, dihedrals = build_angles_dihedrals(bonds, n)
    type_names = assign_atom_types(elements)

    frcmod_path = out_dir / f"frcmod.{resname}"
    leap_path = out_dir / f"create_{resname}.leap"
    comment = f"From {xyz_path.name}; topology only (QM later)"
    write_frcmod(
        frcmod_path,
        type_names,
        elements,
        bonds,
        angles,
        dihedrals,
        pos,
        comment=comment,
    )
    write_leap(leap_path, resname, type_names, elements, pos, bonds, total_charge=args.charge)

    print(f"Wrote {frcmod_path} and {leap_path}")
    if args.no_run_tleap:
        print("Skipping tleap (--no-run-tleap). Run manually:")
        print(f"  cd {out_dir} && {args.tleap_cmd} -f create_{resname}.leap")
        return 0

    if not run_tleap(out_dir, resname, tleap_cmd=args.tleap_cmd):
        print("tleap failed or did not produce .lib", file=sys.stderr)
        return 1
    print(f"Created {out_dir / (resname + '.lib')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
