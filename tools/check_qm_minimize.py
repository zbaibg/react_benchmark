import os
import math
import tempfile
from glob import glob
from pathlib import Path

import numpy as np

import sys

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from paths import REPO_ROOT  # noqa: E402
import zif_meoh_assign_name as zma  # noqa: E402

BASE_DIRS = [
    str(REPO_ROOT / "runs" / "lig_exchange" / "MIMH_PBE_STRUC_6COORD"),
    str(REPO_ROOT / "runs" / "lig_exchange" / "MIM_PBE_STRUC_6COORD"),
    str(REPO_ROOT / "runs" / "lig_exchange" / "MIMH_PBE_STRUC"),
    str(REPO_ROOT / "runs" / "lig_exchange" / "MIM_PBE_STRUC"),
    str(REPO_ROOT / "runs" / "B3LYP_struc"),
    str(REPO_ROOT / "runs" / "M052X_struc"),
    str(REPO_ROOT / "runs" / "M052X_struc_less"),
]

# These two thresholds can be adjusted for screening "suspicious ligands" more easily
ZN_N_SUSPECT = 2.8  # Å, mark if the maximum Zn–Nmin for imidazole is greater than this
ZN_O_SUSPECT = 2.8  # Å, mark if the maximum Zn–O for MeOH is greater than this


def extract_last_frame_to_temp_xyz(path_xyz: str) -> str:
    """
    Extract the last frame from a multi-frame XYZ (path.xyz), write it to a temporary xyz file, and return the temp file path.
    """
    with open(path_xyz, "r") as f:
        lines = f.readlines()

    # Search from the end for the last atom number line
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1

    while i >= 0:
        stripped = lines[i].strip()
        if stripped and stripped.isdigit():
            natoms = int(stripped)
            start = i + 2  # skip the comment line
            end = start + natoms
            frame_lines = lines[start:end]
            if len(frame_lines) < natoms:
                raise ValueError(f"Last frame in {path_xyz} is incomplete.")
            break
        i -= 1
    else:
        raise ValueError(f"Cannot find last frame in {path_xyz}")

    # Write temporary xyz
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xyz", text=True)
    with os.fdopen(tmp_fd, "w") as f:
        f.write(f"{natoms}\n")
        f.write("last frame from path.xyz\n")
        f.writelines(frame_lines)
    return tmp_path


def distance(vec_a, vec_b):
    return float(np.linalg.norm(vec_a - vec_b))


def analyze_complex(path_xyz: str):
    """
    Analyze a complex's path.xyz and return:
    - max_zn_nmin_across_mim: the maximum Zn–Nmin value among all imidazoles (float or None)
    - max_zn_o_across_moh: the maximum Zn–O value among all MeOH (float or None)
    - mim_details: list of (resname, resid, atom_name, distance) for each relevant N atom
    - moh_details: list of (resname, resid, atom_name, distance) for each relevant O atom
    """
    tmp_xyz = extract_last_frame_to_temp_xyz(path_xyz)
    try:
        u = zma.xyz_to_mda(tmp_xyz)
    finally:
        if os.path.exists(tmp_xyz):
            os.unlink(tmp_xyz)

    # Zn
    zn_atoms = u.select_atoms("resname ZN and name ZN")
    if len(zn_atoms) != 1:
        # If there is no Zn or more than one Zn, return None for easier post-checks
        return None, None
    zn_pos = zn_atoms.positions[0]

    mim_like_resnames = {"MIM", "IMH", "MIH"}
    mim_residues = [r for r in u.residues if r.resname in mim_like_resnames]
    moh_residues = [r for r in u.residues if r.resname == "MOH"]

    # For each imidazole residue: take the minimum Zn–N1/N2 value for that residue
    # and record only that closest N for printing
    mim_zn_nmins = []
    mim_details = []
    for res in mim_residues:
        # First try finding N1/N2 by name, fallback to all N if not found
        Ns = res.atoms.select_atoms("name N1 or name N2")
        if len(Ns) == 0:
            Ns = res.atoms.select_atoms("element N")
        if len(Ns) == 0:
            continue
        min_d = None
        min_atom_name = None
        for n in Ns:
            d = distance(zn_pos, n.position)
            if min_d is None or d < min_d:
                min_d = d
                min_atom_name = n.name
        if min_d is not None:
            mim_zn_nmins.append(min_d)
            mim_details.append((res.resname, getattr(res, "resid", None), min_atom_name, min_d))

    # For each MeOH residue: take O1 / O atom to Zn distance
    moh_zn_os = []
    moh_details = []
    for res in moh_residues:
        O = res.atoms.select_atoms("name O1")
        if len(O) == 0:
            O = res.atoms.select_atoms("element O")
        if len(O) == 0:
            continue
        d = distance(zn_pos, O.positions[0])
        moh_zn_os.append(d)
        moh_details.append((res.resname, getattr(res, "resid", None), O.names[0], d))

    max_mim_zn_nmin = max(mim_zn_nmins) if mim_zn_nmins else None
    max_moh_zn_o = max(moh_zn_os) if moh_zn_os else None

    return max_mim_zn_nmin, max_moh_zn_o, mim_details, moh_details


def main():
    xyz_files = []
    for base in BASE_DIRS:
        pattern = os.path.join(base, "qm_minimize", "run*", "*", "path.xyz")
        xyz_files.extend(glob(pattern, recursive=True))

    print(f"Found {len(xyz_files)} path.xyz files\n")
    print("File\tmax_Zn-Nmin_MIM(Å)\tmax_Zn-O_MeOH(Å)\tFlag")
    print("-" * 100)

    for path_xyz in sorted(xyz_files):
        # Skip monomer structures
        if "_monomer" in os.path.basename(os.path.dirname(path_xyz)):
            continue

        try:
            max_mim_zn_nmin, max_moh_zn_o, mim_details, moh_details = analyze_complex(path_xyz)
        except Exception as e:
            print(f"{path_xyz}\tERROR\tERROR\t{e}")
            continue

        if max_mim_zn_nmin is None and max_moh_zn_o is None:
            print(f"{path_xyz}\tNone\tNone\t(no Zn/MIM/MOH)")
            continue

        flags = []
        if max_mim_zn_nmin is not None and max_mim_zn_nmin > ZN_N_SUSPECT:
            flags.append("MIM_far")
        if max_moh_zn_o is not None and max_moh_zn_o > ZN_O_SUSPECT:
            flags.append("MeOH_far")

        flag_str = ",".join(flags)

        # Format numbers and color those above cutoff in red (ANSI)
        if max_mim_zn_nmin is not None:
            mim_str_plain = f"{max_mim_zn_nmin:6.3f}"
            if max_mim_zn_nmin > ZN_N_SUSPECT:
                mim_str = f"\033[31m{mim_str_plain}\033[0m"
            else:
                mim_str = mim_str_plain
        else:
            mim_str = "None"

        if max_moh_zn_o is not None:
            moh_str_plain = f"{max_moh_zn_o:6.3f}"
            if max_moh_zn_o > ZN_O_SUSPECT:
                moh_str = f"\033[31m{moh_str_plain}\033[0m"
            else:
                moh_str = moh_str_plain
        else:
            moh_str = "None"

        # One line per structure
        print(f"{path_xyz}\t{mim_str}\t{moh_str}\t{flag_str}")


if __name__ == "__main__":
    main()