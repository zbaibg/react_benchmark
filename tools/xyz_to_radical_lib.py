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
import importlib.util
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
    # Dihedrals: (a, b, c, d) where a-b, b-c, c-d are bonds; avoid duplicates.
    # Iterate over each bond as the *middle* bond (b-c) so we enumerate all
    # a in neighbors(b) and d in neighbors(c); this is direction-agnostic.
    dihedrals_seen: set[tuple[int, int, int, int]] = set()
    dihedrals = []
    for b, c in bonds:
        for a in neighbors[b]:
            if a == c:
                continue
            for d in neighbors[c]:
                if d == b or d == a:
                    continue
                key = (a, b, c, d) if (a, b) < (d, c) else (d, c, b, a)
                if key not in dihedrals_seen:
                    dihedrals_seen.add(key)
                    dihedrals.append((a, b, c, d))
    return angles, dihedrals


_SUFFIXES = "123456789"


def _idx_to_suffix_char(i: int) -> str:
    if i < 0 or i >= len(_SUFFIXES):
        raise ValueError(
            f"Atom count for a single element exceeds capacity (max 9). "
            "Only suffixes 1-9 are allowed to keep atom types at 2 characters."
        )
    return _SUFFIXES[i]


def assign_atom_types(elements: list[str]) -> list[str]:
    """Assign 2-character atom types: <Element><suffix 1-9>.

    Amber frcmod BOND / ANGLE / DIHE lines are parsed with fixed 2-character
    fields for each atom type, separated by '-'. If a type name is 3+ chars
    (e.g. "H10"), tleap silently misreads the line and later emits
    "No torsion terms" for that quadruple. Using suffixes 1-9 keeps 
    each type at most 2 characters for up to 9 atoms of the SAME element.
    """
    counts = {}
    types = []
    for e in elements:
        c = counts.get(e, 0)
        counts[e] = c + 1
        types.append(f"{e}{_idx_to_suffix_char(c)}")
    return types


def _import_module_from_path(script_path: Path, module_name: str):
    """Dynamically import a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_atom_names_from_zif(
    xyz_path: Path,
    name_script: Path,
    pos: np.ndarray,
    expand_nh_oh_radius: bool = False,
    delete_wrong_bonds: bool = True,
    match_tol: float = 1e-3,
) -> list[str]:
    """Call zif_meoh_assign_name.xyz_to_mda() to get semantic atom names.

    Uses position matching to remap the returned names back to the original XYZ
    atom order, so the result is a list aligned with `pos`/`elements`.
    """
    mod = _import_module_from_path(name_script, "zif_meoh_assign_name_loaded")
    if not hasattr(mod, "xyz_to_mda"):
        raise AttributeError(
            f"{name_script} has no function 'xyz_to_mda'; cannot assign names."
        )
    u = mod.xyz_to_mda(
        str(xyz_path),
        expand_nh_oh_radius=expand_nh_oh_radius,
        delete_wrong_bonds=delete_wrong_bonds,
    )
    zif_pos = np.asarray(u.atoms.positions)
    zif_names = [str(n) for n in u.atoms.names]
    n = len(pos)
    if len(zif_names) != n:
        raise ValueError(
            f"zif naming returned {len(zif_names)} atoms but XYZ has {n}"
        )
    names = [None] * n
    used = [False] * n
    for i in range(n):
        best_j = -1
        best_d = float("inf")
        for j in range(n):
            if used[j]:
                continue
            d = float(np.linalg.norm(zif_pos[j] - pos[i]))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j < 0 or best_d > match_tol:
            raise ValueError(
                f"Cannot match XYZ atom {i} (pos {pos[i]}) to any zif atom "
                f"(best d={best_d:.4g})"
            )
        used[best_j] = True
        names[i] = zif_names[best_j]
    return names


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
    atom_names: list[str] | None = None,
) -> None:
    """Write tleap script: load params, create residue with positions from xyz, saveOff.

    If `atom_names` is provided, use those names verbatim (length must match
    len(elements)). Otherwise fall back to per-element sequential names
    (e.g. O1, H1, H2, H3). Matching `atom_names` with the names used in the
    downstream PDB (e.g. produced by zif_meoh_assign_name.py) is required
    for tleap to find the residue template, otherwise atoms get created
    without types and loading fails.
    """
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
    if atom_names is not None:
        if len(atom_names) != n:
            raise ValueError(
                f"atom_names length {len(atom_names)} != n_atoms {n}"
            )
    else:
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
    default_name_script = Path(__file__).resolve().parent / "zif_meoh_assign_name.py"
    parser.add_argument(
        "--name-script",
        type=Path,
        default=default_name_script,
        help=(
            "Path to zif_meoh_assign_name.py; atom names in the .lib are "
            "taken from its xyz_to_mda() output (by position matching to "
            "the XYZ) so they agree with MOL_clean.pdb produced by the same "
            f"script. Default: {default_name_script}. "
            "Pass --no-name-script to disable and fall back to per-element "
            "sequential names (e.g. O1, H1, H2, H3)."
        ),
    )
    parser.add_argument(
        "--no-name-script",
        action="store_true",
        help="Disable --name-script; use per-element sequential atom names.",
    )
    parser.add_argument(
        "--atom-names",
        type=str,
        default=None,
        help=(
            "Comma-separated atom names in XYZ order, overriding both the "
            "default per-element names and --name-script. Example for MO+ "
            "(H C H H O H H): 'HC1,C1,HC2,HC3,O1,HO1,HO2'."
        ),
    )
    parser.add_argument(
        "--zif-expand-nh-oh-bond-radius",
        action="store_true",
        help="Passed through to zif_meoh_assign_name.xyz_to_mda().",
    )
    parser.add_argument(
        "--zif-no-delete-wrong-bonds",
        action="store_true",
        help=(
            "By default we call xyz_to_mda(delete_wrong_bonds=True); pass this "
            "flag to disable that."
        ),
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

    atom_names: list[str] | None = None
    if args.atom_names is not None:
        atom_names = [s.strip() for s in args.atom_names.split(",")]
        if len(atom_names) != n:
            print(
                f"Error: --atom-names has {len(atom_names)} entries but XYZ has {n} atoms",
                file=sys.stderr,
            )
            return 1
    elif not args.no_name_script and args.name_script is not None:
        name_script = args.name_script.resolve()
        if not name_script.exists():
            print(f"Error: --name-script not found: {name_script}", file=sys.stderr)
            return 1
        try:
            atom_names = get_atom_names_from_zif(
                xyz_path,
                name_script,
                pos,
                expand_nh_oh_radius=args.zif_expand_nh_oh_bond_radius,
                delete_wrong_bonds=not args.zif_no_delete_wrong_bonds,
            )
            print(f"Atom names (from {name_script.name}): {atom_names}")
        except Exception as exc:
            print(f"Error while getting names via name-script: {exc}", file=sys.stderr)
            return 1

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
    write_leap(
        leap_path,
        resname,
        type_names,
        elements,
        pos,
        bonds,
        total_charge=args.charge,
        atom_names=atom_names,
    )

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
