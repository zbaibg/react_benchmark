#!/usr/bin/env python3
"""
Analyze atomic charges for MIm_monomer, MImH_monomer, 1Zn_1MIm_3MeOH_full, 1Zn_1MImH_3MeOH_full
across run8 (mDFTB3_prime, SP_opt), run13 (M06L-D3 def2-TZVPPD DEFGRID3, SP_init), and run16 (PBE-D3(BJ) def2-TZVPPD, SP_init).

- run8: Charges read from SP_opt/run8/{system}/detailed.out under 'Atomic gross charges' (DFTB)
- run13/run16: Charges read from SP_init/run{13|16}/{system}/old.orc_job.dat under 'MULLIKEN' / 'LOEWDIN ATOMIC CHARGES' (DFT)
"""

import re
import os
import sys
from pathlib import Path

# Set working directory to script location
BASE = Path(__file__).resolve().parent
# Allow importing zif_meoh_assign_name from repo python_scripts
_QMMM_ROOT = BASE.parent.parent.parent
if str(_QMMM_ROOT) not in sys.path:
    sys.path.insert(0, str(_QMMM_ROOT))
try:
    from python_scripts.zif_meoh_assign_name import xyz_to_mda
    import MDAnalysis as mda
    import numpy as np
    _HAS_ZIF_ASSIGN = True
except Exception:
    _HAS_ZIF_ASSIGN = False
    xyz_to_mda = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
    plt = None

# Covalent radii (Å) for bond drawing
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "ZN": 1.22}
BOND_TOLERANCE = 1.35  # bond if distance < (ra + rb) * this

# Which two data sets to use for plotting (first label, second label on each atom in 2D plot).
# Allowed values: "run8", "run13M", "run13L", "run16M", "run16L"
PLOT_DATA_1 = "run8"    # first charge shown in 2D plot
PLOT_DATA_2 = "run13M"  # second charge shown in 2D plot (run13M/run13L = Mulliken/Loewdin, run16M/run16L = Mulliken/Loewdin)

PLOT_DATA_OPTIONS = ("run8", "run13M", "run13L", "run16M", "run16L")


def resolve_plot_data_key(key):
    """
    Map PLOT_DATA_1/PLOT_DATA_2 to (run_key, dft_charge_type, label).
    run_key for get_charges(); dft_charge_type only used for run13/run16; label for plot title.
    """
    if key == "run8":
        return "run8", "mulliken", "run8"
    if key == "run13M":
        return "run13", "mulliken", "run13 Mulliken"
    if key == "run13L":
        return "run13", "loewdin", "run13 Loewdin"
    if key == "run16M":
        return "run16", "mulliken", "run16 Mulliken"
    if key == "run16L":
        return "run16", "loewdin", "run16 Loewdin"
    raise ValueError(f"Invalid plot data key: {key!r}. Allowed: {PLOT_DATA_OPTIONS}")


SYSTEMS = [
    "MIm_monomer",
    "MImH_monomer",
    "1Zn_1MIm_3MeOH_full",
    "1Zn_1MImH_3MeOH_full",
]

# run8 uses SP_opt, run13/run16 use SP_init
RUN_CONFIG = {
    "run8": {"stage": "SP_opt", "label": "run8 (mDFTB3_prime)", "file": "detailed.out", "parser": "dftb"},
    "run13": {"stage": "SP_init", "label": "run13 (M06L-D3 def2-TZVPPD)", "file": "old.orc_job.dat", "parser": "orca"},
    "run16": {"stage": "SP_init", "label": "run16 (PBE-D3(BJ) def2-TZVPPD)", "file": "old.orc_job.dat", "parser": "orca"},
}


def parse_dftb_charges(filepath):
    """Parse 'Atomic gross charges (e)' from detailed.out. Returns list of (atom_index_0based, charge)."""
    with open(filepath) as f:
        text = f.read()
    charges = []
    start = text.find("Atomic gross charges (e)")
    if start == -1:
        return charges
    rest = text[start:]
    lines = rest.split("\n")
    in_table = False
    for line in lines:
        if "Atom" in line and "Charge" in line and not in_table:
            in_table = True
            continue
        if in_table:
            # Data line: "    1       0.63147824" or "   10      -0.36467365"
            m = re.match(r"\s*(\d+)\s+([-\d.]+)\s*$", line.strip())
            if m:
                idx = int(m.group(1))
                q = float(m.group(2))
                charges.append((idx, q))
            else:
                # Break upon encountering a non-data line (e.g., "Nr. of electrons")
                if line.strip() and not re.match(r"^\s*\d+\s+[-\d.]+\s*$", line):
                    break
    return charges


def parse_orca_mulliken(filepath):
    """Parse 'MULLIKEN ATOMIC CHARGES' from old.orc_job.dat. Returns list of (atom_index, element, charge)."""
    with open(filepath) as f:
        text = f.read()
    charges = []
    start = text.find("MULLIKEN ATOMIC CHARGES")
    if start == -1:
        return charges
    rest = text[start:]
    lines = rest.split("\n")
    # Skip the header and separator lines: "MULLIKEN ATOMIC CHARGES", "--------"
    for i in range(2, len(lines)):
        line = lines[i]
        # Format: "   0 Zn:   -0.208129" or "  24 C :   -0.496752"
        m = re.match(r"\s*(\d+)\s+(\w+)\s*:\s*([-\d.]+)\s*$", line.strip())
        if m:
            idx = int(m.group(1))
            elem = m.group(2).strip()
            q = float(m.group(3))
            charges.append((idx, elem, q))
        elif "Sum of atomic charges" in line or "MULLIKEN REDUCED" in line or "LOEWDIN" in line:
            break
    return charges


def parse_orca_loewdin(filepath):
    """Parse 'LOEWDIN ATOMIC CHARGES' from old.orc_job.dat. Returns list of (atom_index, element, charge)."""
    with open(filepath) as f:
        text = f.read()
    charges = []
    start = text.find("LOEWDIN ATOMIC CHARGES")
    if start == -1:
        return charges
    rest = text[start:]
    lines = rest.split("\n")
    # Skip the header and separator lines: "LOEWDIN ATOMIC CHARGES", "-------"
    for i in range(2, len(lines)):
        line = lines[i]
        # Same format as Mulliken: "   0 Zn:    0.855902" or "  24 C :   -0.009446"
        m = re.match(r"\s*(\d+)\s+(\w+)\s*:\s*([-\d.]+)\s*$", line.strip())
        if m:
            idx = int(m.group(1))
            elem = m.group(2).strip()
            q = float(m.group(3))
            charges.append((idx, elem, q))
        elif "LOEWDIN REDUCED" in line or "Sum of atomic" in line:
            break
    return charges


def parse_xyz(filepath):
    """Parse xyz file. Returns list of (element, x, y, z) in order; element is uppercase (e.g. ZN)."""
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines()]
    if not lines:
        return []
    n = int(lines[0])
    # Second line may be comment; data starts at line 2 (0-indexed)
    start = 2
    atoms = []
    for i in range(n):
        if start + i >= len(lines):
            break
        parts = lines[start + i].split()
        if len(parts) >= 4:
            elem = parts[0].upper()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((elem, x, y, z))
    return atoms


def coords_3d_to_2d_pca(atoms_xyz):
    """Project 3D coordinates to 2D using PCA (two largest variance axes). atoms_xyz: list of (elem, x, y, z). Returns (n, 2) array."""
    if not atoms_xyz:
        return None
    try:
        import numpy as _np
    except Exception:
        _np = None
    X = [[a[1], a[2], a[3]] for a in atoms_xyz]
    if _np is not None:
        X = _np.array(X)
        X_centered = X - X.mean(axis=0)
        cov = _np.cov(X_centered.T)
        w, v = _np.linalg.eigh(cov)
        idx = _np.argsort(w)[::-1]
        V = v[:, idx[:2]]
        return X_centered @ V
    # Fallback: use x, y
    return [[a[1], a[2]] for a in atoms_xyz]


def _canonical_orientation_imidazole(coords_2d, elements, system):
    """
    Unify 2D orientation for MIm_monomer and MImH_monomer so they show the same face.
    coords_2d: list of [x, y] or array (n, 2). elements: list of element symbols.
    Returns coords_2d as list of [x, y] with columns possibly flipped.
    """
    if system not in ("MIm_monomer", "MImH_monomer"):
        return coords_2d
    try:
        import numpy as _np
    except Exception:
        return coords_2d
    arr = _np.asarray(coords_2d).astype(float).copy()
    if arr.size == 0:
        return coords_2d
    n = len(arr)
    n_idx = [i for i in range(n) if elements[i] == "N"]
    if not n_idx:
        return coords_2d
    centroid = arr.mean(axis=0)
    # 1) Flip y so N centroid is above molecule centroid (same "top" for both)
    n_centroid = arr[n_idx].mean(axis=0)
    if n_centroid[1] < centroid[1]:
        arr[:, 1] = -arr[:, 1]
    # 2) Flip x so first N is on the right (consistent left-right)
    centroid = arr.mean(axis=0)
    if arr[n_idx[0], 0] < centroid[0]:
        arr[:, 0] = -arr[:, 0]
    return arr.tolist()


def _dist(a, b):
    """Euclidean distance between 3D points a, b (each [x,y,z])."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def plot_structure_2d(system, atoms_xyz, charges_1, charges_2, label_1, label_2, out_path):
    """
    Draw 2D structure and label each atom with "charge_1 (charge_2)".
    atoms_xyz: list of (element, x, y, z); charges_1, charges_2: dict idx -> charge.
    label_1, label_2: short names for plot title.
    """
    if not _HAS_MATPLOTLIB or not atoms_xyz:
        return
    n = len(atoms_xyz)
    elements = [a[0] for a in atoms_xyz]
    coords_2d = coords_3d_to_2d_pca(atoms_xyz)
    if coords_2d is None:
        return
    coords_2d = _canonical_orientation_imidazole(coords_2d, elements, system)
    # Support both list-of-lists and array for coords_2d
    def xy(i):
        row = coords_2d[i]
        return (row[0], row[1])

    coords_3d = [[a[1], a[2], a[3]] for a in atoms_xyz]
    bonds = []
    for i in range(n):
        ra = COVALENT_RADII.get(elements[i], 0.77)
        for j in range(i + 1, n):
            rb = COVALENT_RADII.get(elements[j], 0.77)
            d = _dist(coords_3d[i], coords_3d[j])
            if d < (ra + rb) * BOND_TOLERANCE:
                bonds.append((i, j))
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, j in bonds:
        ax.plot([xy(i)[0], xy(j)[0]], [xy(i)[1], xy(j)[1]], "k-", lw=1.5, zorder=0)
    # Atom colors and text color (white on dark circles for readability)
    color_map = {"H": "gray", "C": "black", "N": "blue", "O": "red", "ZN": "green"}
    text_color_map = {"H": "black", "C": "white", "N": "white", "O": "white", "ZN": "white"}
    for i in range(n):
        c = color_map.get(elements[i], "black")
        txt_color = text_color_map.get(elements[i], "black")
        xi, yi = xy(i)
        # Large circle so two-line charge label fits inside
        ax.scatter(xi, yi, c=c, s=900, zorder=1, edgecolors="black", linewidths=1)
        q1 = charges_1.get(i, float("nan"))
        q2 = charges_2.get(i, float("nan"))
        s1 = "–" if q1 != q1 else f"{q1:.1f}"
        s2 = "–" if q2 != q2 else f"{q2:.1f}"
        label = f"{s1}\n({s2})"
        ax.annotate(
            label,
            (xi, yi),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=10,
            ha="center",
            va="center",
            color=txt_color,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{system}\n{label_1} ({label_2})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_structure_path(system):
    """Use structure from SP_init/run16/{system}/min.xyz"""
    p = BASE / "SP_init" / "run16" / system / "min.xyz"
    if p.exists():
        return p
    return None


def get_resname_atomname_list(system):
    """
    Use zif_meoh_assign_name to assign resname and atom name per atom in original file order.
    Returns list of (resname, atom_name) with length = n_atoms, or None on failure.
    """
    if not _HAS_ZIF_ASSIGN:
        return None
    struct_path = get_structure_path(system)
    if not struct_path:
        return None
    try:
        orig_u = mda.Universe(str(struct_path))
        u_labeled = xyz_to_mda(str(struct_path))
    except Exception:
        return None
    n_orig = len(orig_u.atoms)
    n_labeled = len(u_labeled.atoms)
    if n_orig != n_labeled:
        return None
    # Match by (element, position): for each orig atom, find labeled atom with same element and closest position
    def elem_of(a):
        if hasattr(a, "element") and a.element:
            return str(a.element).upper()
        return (str(a.name)[:1].upper() if a.name else "")

    used = set()
    result = [None] * n_orig
    for i, a_orig in enumerate(orig_u.atoms):
        pos = a_orig.position
        elem = elem_of(a_orig)
        best_j = None
        best_d = float("inf")
        for j, a_lab in enumerate(u_labeled.atoms):
            if j in used:
                continue
            if elem_of(a_lab) != elem:
                continue
            d = np.linalg.norm(a_lab.position - pos)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d < 0.05:
            used.add(best_j)
            a_lab = u_labeled.atoms[best_j]
            resname = (a_lab.residue.resname if hasattr(a_lab, "residue") and hasattr(a_lab.residue, "resname") else "") or ""
            result[i] = (resname, a_lab.name or "")
    if any(r is None for r in result):
        return None
    return result


def get_charges(system, run_key, dft_charge_type="mulliken"):
    """
    Get atomic charges for a given system and run. 
    For run13/run16, 'dft_charge_type' can be 'mulliken' or 'loewdin'.
    Returns a list of (position, element, charge) or None on failure.
    """
    cfg = RUN_CONFIG[run_key]
    stage, fname = cfg["stage"], cfg["file"]
    path = BASE / stage / run_key / system / fname
    if not path.exists():
        return None, None  # (charges or (idx, elem, q)), error_msg
    if cfg["parser"] == "dftb":
        raw = parse_dftb_charges(path)
        # Unify to (position_0based, element, charge), elements not available for DFTB ("")
        return [(i, "", q) for i, (_, q) in enumerate(raw)], None
    else:
        if dft_charge_type == "loewdin":
            raw = parse_orca_loewdin(path)
        else:
            raw = parse_orca_mulliken(path)
        return [(i, r[1], r[2]) for i, r in enumerate(raw)], None


def main():
    for system in SYSTEMS:
        data_run8, _ = get_charges(system, "run8")
        data_run13, _ = get_charges(system, "run13")
        data_run16, _ = get_charges(system, "run16")
        data_run13_loewdin, _ = get_charges(system, "run13", dft_charge_type="loewdin")
        data_run16_loewdin, _ = get_charges(system, "run16", dft_charge_type="loewdin")

        data_run8 = data_run8 or []
        data_run13 = data_run13 or []
        data_run16 = data_run16 or []
        data_run13_loewdin = data_run13_loewdin or []
        data_run16_loewdin = data_run16_loewdin or []

        n_atoms = max(len(data_run8), len(data_run13), len(data_run16))
        if n_atoms == 0:
            continue

        run8_by_pos = {r[0]: r[2] for r in data_run8}
        run13_by_pos = {r[0]: r[2] for r in data_run13}
        run16_by_pos = {r[0]: r[2] for r in data_run16}
        run13_loewdin_by_pos = {r[0]: r[2] for r in data_run13_loewdin}
        run16_loewdin_by_pos = {r[0]: r[2] for r in data_run16_loewdin}
        ref = data_run13 if data_run13 else (data_run16 if data_run16 else data_run8)
        elements = [r[1] for r in ref] if ref else []
        has_elem = bool(elements)

        # Resname and atom name from zif_meoh_assign_name (same atom order as charges)
        labels = get_resname_atomname_list(system) if _HAS_ZIF_ASSIGN else None
        has_labels = bool(labels and len(labels) >= n_atoms)

        def fmt(x):
            if x != x:
                return "    --"
            return f"{x:8.4f}"

        header = "Atom"
        if has_labels:
            header = "Resname  Name"
        elif has_elem:
            header = "Atom  Elem"
        first_col_width = 22
        print(f"\n### {system} ###")
        print(f"  {header:<{first_col_width}}   run8 charge   run13_Loewdin  run16_Loewdin   run13_Mulliken  run16_Mulliken")
        print("  " + "-" * 95)
        # Build rows: (sort_key, label_str, charge_values...)
        rows = []
        for i in range(n_atoms):
            q8 = run8_by_pos.get(i, float("nan"))
            q13 = run13_by_pos.get(i, float("nan"))
            q13L = run13_loewdin_by_pos.get(i, float("nan"))
            q16 = run16_by_pos.get(i, float("nan"))
            q16L = run16_loewdin_by_pos.get(i, float("nan"))
            if has_labels and i < len(labels):
                resname, aname = labels[i]
                label_str = f"  {resname:8}  {aname:8}"
                sort_key = (resname, aname)
            else:
                elem = elements[i] if i < len(elements) else ""
                label_str = f"  {i+1:4}"
                if elem:
                    label_str += f"  {elem:4}"
                sort_key = (i,)  # keep original order when no labels
            label_str = (label_str + " " * first_col_width)[:first_col_width]
            rows.append((sort_key, label_str, q8, q13L, q16L, q13, q16))
        rows.sort(key=lambda r: r[0])
        for _, label_str, q8, q13L, q16L, q13, q16 in rows:
            print(label_str + f"   {fmt(q8)}        {fmt(q13L)}        {fmt(q16L)}        {fmt(q13)}        {fmt(q16)}")

        # 2D structure plot: use PLOT_DATA_1 and PLOT_DATA_2
        if _HAS_MATPLOTLIB:
            run_key_1, ctype_1, label_1 = resolve_plot_data_key(PLOT_DATA_1)
            run_key_2, ctype_2, label_2 = resolve_plot_data_key(PLOT_DATA_2)
            data_1, _ = get_charges(system, run_key_1, dft_charge_type=ctype_1)
            data_2, _ = get_charges(system, run_key_2, dft_charge_type=ctype_2)
            charges_1_by_pos = {r[0]: r[2] for r in (data_1 or [])}
            charges_2_by_pos = {r[0]: r[2] for r in (data_2 or [])}
            struct_path = get_structure_path(system)
            if struct_path and charges_1_by_pos and charges_2_by_pos:
                atoms_xyz = parse_xyz(struct_path)
                if len(atoms_xyz) >= n_atoms:
                    out_dir = BASE / "charge_2d"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{system}_{PLOT_DATA_1}_{PLOT_DATA_2}.png"
                    plot_structure_2d(system, atoms_xyz, charges_1_by_pos, charges_2_by_pos, label_1, label_2, out_path)
                    print(f"  -> 2D figure saved: {out_path}")


if __name__ == "__main__":
    main()
