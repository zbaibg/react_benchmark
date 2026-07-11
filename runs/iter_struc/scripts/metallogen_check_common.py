"""Shared helpers for ``check_metallogen_short_bonds.py`` driver scripts (iter_struc / NO3).

Extracted verbatim from the iter_struc driver logic so behavior stays identical.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from MetalloGen import chem, om, process  # type: ignore[import-not-found]
from MetalloGen.clean_geometry import TMCOptimizer  # type: ignore[import-not-found]

# Reuse thresholds as relax_struc_lbfgs_5/check_minimize.py and MetalloGen clean_geometry.py.
_TMC_DEFAULTS = TMCOptimizer(calculator=object())
RATIO_CRITERIA = _TMC_DEFAULTS.ratio_criteria
ATOM_D_CRITERIA = _TMC_DEFAULTS.atom_d_criteria
BOND_CRITERIA = _TMC_DEFAULTS.bond_criteria

_RELAXING_CONFORMER_RE = re.compile(r"Relaxing conformer\s+(?P<idx>\d+)/\d+")
_ENERGY_RE = re.compile(r"\bE=(?P<e>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")
_CLUSTER_ID_RE = re.compile(r"\bcluster_id=(?P<cid>\d+)\b")
_ATOM_LINE_RE = re.compile(
    r"^\s*([A-Za-z]{1,3})\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def dftbplus_relax_failed(log_text: str) -> bool:
    """Failure detection for DFTB+ relaxation logs.

    Policy:
    - "ERROR STOP" => failed
    - otherwise, require that the log contains "Geometry converged"
      (if the log exists but doesn't contain that string, treat as failed)
    """
    if not log_text:
        # If the log file exists but is empty/unreadable, treat as failed.
        return True
    # Use strict, case-sensitive markers (no lowercasing).
    if "ERROR STOP" in log_text:
        return True
    return "Geometry converged" not in log_text


def load_enumeration(
    tsv_path: Path,
) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    """Return (id -> formula_id, formula_id -> formula string, id -> msmiles)."""
    id_to_fid: dict[int, int] = {}
    fid_to_formula: dict[int, str] = {}
    id_to_msmiles: dict[int, str] = {}
    with tsv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            oid = int(row["id"])
            fid = int(row["formula_id"])
            formula = row["formula"].strip()
            msmiles = row["msmiles"].strip()
            id_to_fid[oid] = fid
            fid_to_formula[fid] = formula
            id_to_msmiles[oid] = msmiles
    return id_to_fid, fid_to_formula, id_to_msmiles


_METAL_DIST_RE = re.compile(
    r"Zn-(?P<elem>[NO])\[\d+\]:\s*(?P<dist>[0-9.]+)"
)


def _parse_atom_line(line: str) -> tuple[str, float, float, float] | None:
    m = _ATOM_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    return (
        m.group(1),
        float(m.group(2)),
        float(m.group(3)),
        float(m.group(4)),
    )


def parse_multi_xyz_frames(
    xyz_path: Path,
) -> list[list[tuple[str, float, float, float]]]:
    if not xyz_path.exists():
        return []
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[list[tuple[str, float, float, float]]] = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            break
        if i + 1 >= n:
            break
        j = i + 2
        end = min(j + natoms, n)
        atoms: list[tuple[str, float, float, float]] = []
        while j < end:
            atom = _parse_atom_line(lines[j])
            if atom is None:
                break
            atoms.append(atom)
            j += 1
        if len(atoms) != natoms:
            break
        out.append(atoms)
        i = j
    return out


def _elements_coords(
    atoms: list[tuple[str, float, float, float]],
) -> tuple[list[str], list[tuple[float, float, float]]]:
    return [a[0] for a in atoms], [(a[1], a[2], a[3]) for a in atoms]


def _atoms_to_metallogen_molecule(
    atoms: list[tuple[str, float, float, float]],
) -> chem.Molecule:
    mol = chem.Molecule()
    atom_list: list[chem.Atom] = []
    for element, x, y, z in atoms:
        atom = chem.Atom(element)
        atom.x = x
        atom.y = y
        atom.z = z
        atom_list.append(atom)
    mol.atom_list = atom_list
    return mol


def _first_zn_index(elements: list[str]) -> int | None:
    for i, e in enumerate(elements):
        if e == "Zn":
            return i
    return None


def max_zn_n_zn_o_distances_angstrom(
    atoms: list[tuple[str, float, float, float]],
) -> tuple[float | None, float | None]:
    """Longest Zn–N and Zn–O distances (Å) from the first Zn to any N / O.

    Note: This is a whole-molecule metric; for metal-ligand-only distances use
    `max_zn_n_zn_o_from_metallogen_log`.
    """
    elems, coords = _elements_coords(atoms)
    zn_idx = _first_zn_index(elems)
    if zn_idx is None:
        return None, None
    zn = np.asarray(coords[zn_idx], dtype=float)
    max_n: float | None = None
    max_o: float | None = None
    for i, e in enumerate(elems):
        if i == zn_idx:
            continue
        if e == "N":
            d = float(np.linalg.norm(np.asarray(coords[i], dtype=float) - zn))
            max_n = d if max_n is None else max(max_n, d)
        elif e == "O":
            d = float(np.linalg.norm(np.asarray(coords[i], dtype=float) - zn))
            max_o = d if max_o is None else max(max_o, d)
    return max_n, max_o


def expected_topology_from_msmiles(msmiles: str) -> tuple[list[str], np.ndarray]:
    metal_complex = om.get_om_from_modified_smiles(msmiles)
    expected_elements = [a.get_element() for a in metal_complex.get_atom_list()]
    expected_adj = np.array(metal_complex.get_adj_matrix(), dtype=int)
    return expected_elements, expected_adj


def validate_geometry_and_topology(
    new_atoms: list[tuple[str, float, float, float]],
    expected_elements: list[str],
    expected_adj: np.ndarray,
    ratio_criteria: float,
    atom_d_criteria: float,
    bond_criteria: float,
) -> tuple[bool, str]:
    if len(expected_elements) != len(new_atoms):
        return False, "natoms_mismatch"
    new_elements, new_coords = _elements_coords(new_atoms)
    if expected_elements != new_elements:
        return False, "element_order_mismatch"

    # Reuse MetalloGen geometry check logic first.
    if not process.check_geometry(new_coords, criteria=atom_d_criteria):
        return False, "bad_geometry"

    new_mol = _atoms_to_metallogen_molecule(new_atoms)
    new_adj = process.get_adj_matrix_from_distance(
        new_mol, coeff=bond_criteria, criteria=atom_d_criteria
    )
    distance_matrix = process.spatial.distance_matrix(np.array(new_coords), np.array(new_coords))
    radius_list = new_mol.get_radius_list()
    radius_matrix = np.repeat(radius_list, len(radius_list)).reshape((len(radius_list), len(radius_list)))
    ratio_matrix = distance_matrix / (radius_matrix + radius_matrix.T)
    np.fill_diagonal(ratio_matrix, 1e6)
    if float(np.min(ratio_matrix)) < ratio_criteria:
        return False, "bad_geometry"

    n = len(new_elements)
    zn_idx = _first_zn_index(new_elements)
    for i in range(n):
        for j in range(i + 1, n):
            if zn_idx is not None and (i == zn_idx or j == zn_idx):
                # Follow clean_geometry.py idea: ignore metal-involving topology.
                continue
            if int(expected_adj[i][j]) != int(new_adj[i][j]):
                return False, "topology_changed"
    return True, "ok"


def extract_single_xyz_atoms(
    xyz_path: Path,
) -> tuple[list[tuple[str, float, float, float]], list[str]]:
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected xyz format: {xyz_path}")
    natoms = int(lines[0].strip())
    if len(lines) < 2 + natoms:
        raise ValueError(
            f"Unexpected xyz length for {xyz_path}: natoms={natoms} lines={len(lines)}"
        )
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[2 : 2 + natoms]:
        atom = _parse_atom_line(line)
        if atom is None:
            raise ValueError(f"Malformed atom line in {xyz_path}: {line}")
        atoms.append(atom)
    return atoms, lines[: 2 + natoms]


def parse_metallogen_log(
    log_path: Path, zn_n_thresh: float, zn_o_thresh: float
) -> tuple[int, int, list[int]]:
    """Parse one metallogen.log.

    Returns:
        total_conf: total number of relaxed conformers found
        n_conf_stable: number of conformers where
            all Zn-N < zn_n_thresh and all Zn-O < zn_o_thresh
        stable_conformer_nums: conformer indices k (from "Relaxing conformer k/...") that
            satisfy the stability criteria
    """
    total_conf = 0
    n_conf_stable = 0
    stable_conformer_nums: list[int] = []

    in_block = False
    max_n: float | None = None
    max_o: float | None = None
    current_conf_num: int | None = None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0, 0, []

    for line in lines:
        m_relaxing = _RELAXING_CONFORMER_RE.search(line)
        if m_relaxing:
            current_conf_num = int(m_relaxing.group("idx"))

        if "Metal-ligand distances (Angstrom), M = Zn[0]" in line:
            # Start a new distance block for a conformer
            in_block = True
            max_n = None
            max_o = None
            continue

        if not in_block:
            continue

        if "Relaxation success!" in line:
            # End of current conformer block
            total_conf += 1
            # Stable if all Zn-N and all Zn-O are below their thresholds
            n_ok = (max_n is None) or (max_n < zn_n_thresh)
            o_ok = (max_o is None) or (max_o < zn_o_thresh)
            if n_ok and o_ok:
                n_conf_stable += 1
                if current_conf_num is not None:
                    stable_conformer_nums.append(current_conf_num)
            in_block = False
            continue

        m = _METAL_DIST_RE.search(line)
        if not m:
            continue
        elem = m.group("elem")
        dist = float(m.group("dist"))
        if elem == "N":
            if max_n is None or dist > max_n:
                max_n = dist
        elif elem == "O":
            if max_o is None or dist > max_o:
                max_o = dist

    return total_conf, n_conf_stable, stable_conformer_nums


def parse_metallogen_log_with_max_dists(
    log_path: Path, zn_n_thresh: float, zn_o_thresh: float
) -> tuple[
    int,  # total_conf
    int,  # n_conf_stable
    list[int],  # stable_conformer_nums
    list[int],  # relaxed_conformer_nums (Relaxation success!, before Zn-distance stability filter)
    dict[int, tuple[float | None, float | None]],  # conformer_k -> (max_Zn-N, max_Zn-O)
]:
    """Parse one metallogen.log with per-conformer max Zn-N / Zn-O.

    This extends `parse_metallogen_log` by also returning, for each
    "Relaxing conformer k/..." conformer, the corresponding maxima extracted from
    the "Metal-ligand distances (Angstrom), M = Zn[0]" block.
    """
    total_conf = 0
    n_conf_stable = 0
    stable_conformer_nums: list[int] = []
    relaxed_conformer_nums: list[int] = []
    max_dists_by_k: dict[int, tuple[float | None, float | None]] = {}

    in_block = False
    max_n: float | None = None
    max_o: float | None = None
    current_conf_num: int | None = None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0, 0, [], {}

    for line in lines:
        m_relaxing = _RELAXING_CONFORMER_RE.search(line)
        if m_relaxing:
            current_conf_num = int(m_relaxing.group("idx"))

        if "Metal-ligand distances (Angstrom), M = Zn[0]" in line:
            # Start a new distance block for a conformer
            in_block = True
            max_n = None
            max_o = None
            continue

        if not in_block:
            continue

        if "Relaxation success!" in line:
            total_conf += 1
            if current_conf_num is not None:
                max_dists_by_k[current_conf_num] = (max_n, max_o)
                relaxed_conformer_nums.append(current_conf_num)
            n_ok = (max_n is None) or (max_n < zn_n_thresh)
            o_ok = (max_o is None) or (max_o < zn_o_thresh)
            if n_ok and o_ok and current_conf_num is not None:
                n_conf_stable += 1
                stable_conformer_nums.append(current_conf_num)
            in_block = False
            continue

        m = _METAL_DIST_RE.search(line)
        if not m:
            continue
        elem = m.group("elem")
        dist = float(m.group("dist"))
        if elem == "N":
            if max_n is None or dist > max_n:
                max_n = dist
        elif elem == "O":
            if max_o is None or dist > max_o:
                max_o = dist

    return (
        total_conf,
        n_conf_stable,
        stable_conformer_nums,
        relaxed_conformer_nums,
        max_dists_by_k,
    )


def max_zn_n_zn_o_from_metallogen_log(
    log_path: Path, conformer_k: int
) -> tuple[float | None, float | None]:
    """Return (max_Zn-N, max_Zn-O) from metallogen.log for a given conformer k.

    Distances are taken only from the "Metal-ligand distances (Angstrom), M = Zn[0]" block
    corresponding to that conformer, i.e. the same set of distances used for stability screening.
    """
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None, None

    current_conf_num: int | None = None
    in_block = False
    max_n: float | None = None
    max_o: float | None = None

    for line in lines:
        m_relaxing = _RELAXING_CONFORMER_RE.search(line)
        if m_relaxing:
            current_conf_num = int(m_relaxing.group("idx"))

        if "Metal-ligand distances (Angstrom), M = Zn[0]" in line:
            in_block = True
            max_n = None
            max_o = None
            continue

        if not in_block:
            continue

        if "Relaxation success!" in line:
            if current_conf_num == conformer_k:
                return max_n, max_o
            in_block = False
            continue

        if current_conf_num != conformer_k:
            continue

        m = _METAL_DIST_RE.search(line)
        if not m:
            continue
        elem = m.group("elem")
        dist = float(m.group("dist"))
        if elem == "N":
            max_n = dist if max_n is None else max(max_n, dist)
        elif elem == "O":
            max_o = dist if max_o is None else max(max_o, dist)

    return None, None


_ML_ZN_PAIR_LINE_RE = re.compile(
    r"^\s*Zn-(?P<elem>[NO])\[(?P<aidx>\d+)\]:\s*(?P<dist>[0-9.]+)\s*$"
)


def metal_ligand_zn_o_n_by_atom_index_for_conformer_k(
    log_path: Path, conformer_k: int
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-atom Zn–O / Zn–N distances (Å) from the M–L block for one conformer.

    Atom indices match ``print_metal_ligand_distances`` / ``atom_list`` order (same as xyz).

    Returns:
        (zn_o_dist_by_atom_index, zn_n_dist_by_atom_index)
    """
    zn_o: dict[int, float] = {}
    zn_n: dict[int, float] = {}
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return zn_o, zn_n

    current_conf_num: int | None = None
    in_block = False
    block_owner_k: int | None = None

    for line in lines:
        m_relaxing = _RELAXING_CONFORMER_RE.search(line)
        if m_relaxing:
            current_conf_num = int(m_relaxing.group("idx"))

        if "Metal-ligand distances (Angstrom), M = Zn[0]" in line:
            in_block = True
            block_owner_k = current_conf_num
            if block_owner_k == conformer_k:
                zn_o, zn_n = {}, {}
            continue

        if not in_block:
            continue

        if "Relaxation success!" in line:
            in_block = False
            continue

        if block_owner_k != conformer_k:
            continue

        m = _ML_ZN_PAIR_LINE_RE.match(line)
        if not m:
            continue
        elem = m.group("elem")
        aidx = int(m.group("aidx"))
        dist = float(m.group("dist"))
        if elem == "O":
            zn_o[aidx] = dist
        else:
            zn_n[aidx] = dist

    return zn_o, zn_n


def extract_xyz_blocks_by_cluster_ids(
    xyz_path: Path, wanted_cluster_ids: set[int]
) -> dict[int, tuple[float, str]]:
    """Extract xyz blocks from a multi-structure xyz by cluster_id.

    The input xyz format is expected to be:
      natoms
      chg=... mult=... E=... cluster_id=...
      <atom lines...>
    repeated for each conformer.

    Returns:
      cluster_id -> (energy, xyz_block_string)
    """
    if not xyz_path.exists():
        return {}

    blocks: dict[int, tuple[float, str]] = {}
    lines = xyz_path.read_text(encoding="utf-8").splitlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            atom_count = int(line)
        except ValueError:
            # Unexpected; stop to avoid infinite loops.
            break

        if i + 1 >= n:
            break
        if i + 2 + atom_count > n:
            break

        comment = lines[i + 1].strip()
        m_e = _ENERGY_RE.search(comment)
        m_cid = _CLUSTER_ID_RE.search(comment)
        if not (m_e and m_cid):
            i += 2 + atom_count
            continue

        cluster_id = int(m_cid.group("cid"))
        if cluster_id not in wanted_cluster_ids:
            i += 2 + atom_count
            continue

        energy = float(m_e.group("e"))
        block_lines = lines[i : i + 2 + atom_count]
        blocks[cluster_id] = (energy, "\n".join(block_lines))
        i += 2 + atom_count

    return blocks


def extract_single_xyz_with_energy(xyz_path: Path) -> tuple[float, list[str]]:
    """Extract (energy, xyz_lines) from a single-structure xyz.

    Expected format:
      natoms
      comment line containing E=...
      <atom lines...>
    """
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected xyz format: {xyz_path}")

    natoms = int(lines[0].strip())
    # comment is on line 2 (index 1); atoms follow.
    if len(lines) < 2 + natoms:
        raise ValueError(
            f"Unexpected xyz length for {xyz_path}: natoms={natoms} lines={len(lines)}"
        )

    comment = lines[1]
    m_e = _ENERGY_RE.search(comment)
    if not m_e:
        raise ValueError(f"Missing E=... in comment line for {xyz_path}: {comment}")
    energy = float(m_e.group("e"))

    block_lines = lines[: 2 + natoms]
    return energy, block_lines
