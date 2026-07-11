#!/usr/bin/env python3
import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np

# Default xyz directory (used by scripts that analyze xyz files)
XYZ_DIR = os.path.join('xyz', 'xyz_files')

# Conversion factor
HARTREE_TO_KCAL = 627.5094740631

_RE_MOLPRO_SUMMARY_ENERGY = re.compile(
    r'energy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
)
_RE_MOLPRO_BANG_TOTAL_ENERGY = re.compile(
    r'!\s*CCSD\(T\).*total energy\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',
    re.IGNORECASE,
)


def detect_ligand_name(xyz_dir: Optional[str] = None) -> str:
    """
    Auto-detect whether complexes use 'MIm' or 'MImH' from xyz filenames.

    Returns 'MIm' if detection fails or directory is missing.
    """
    if xyz_dir is None:
        xyz_dir = XYZ_DIR
    if not os.path.exists(xyz_dir):
        return 'MIm'
    for f in os.listdir(xyz_dir):
        if f.endswith('.xyz') and '_monomer' not in f:
            if 'MImH' in f:
                return 'MImH'
            if 'MIm' in f:
                return 'MIm'
    return 'MIm'


LIGAND_NAME = detect_ligand_name()


def get_Etot_orca(filepath: str) -> Optional[float]:
    """
    Read ORCA text output (e.g. orc_job.dat) and return the last
    'FINAL SINGLE POINT ENERGY' in kcal/mol.

    Returns None if the file is missing or no such line is found.
    """
    if not os.path.exists(filepath):
        return None

    final_energy_hartree: Optional[float] = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'FINAL SINGLE POINT ENERGY' in line:
                    parts = line.split()
                    # Energy is typically the last token on the line
                    try:
                        final_energy_hartree = float(parts[-1])
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"  WARNING: Error reading {filepath}: {e}")
        return None

    if final_energy_hartree is not None:
        return final_energy_hartree * HARTREE_TO_KCAL
    return None


def get_Etot_molpro_hartree(filepath: str) -> Optional[float]:
    """
    Read Molpro stdout log (typically ``molpro.log``) and return the final
    total energy in Hartree.

    Parsing strategy:
      - Require ``Molpro calculation terminated`` near the end of the file.
      - Prefer the final summary line containing ``energy=``.
      - Fall back to the last ``!CCSD(T)... total energy`` line if needed.
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', errors='replace') as f:
            text = f.read()
    except Exception as e:
        print(f"  WARNING: Error reading {filepath}: {e}")
        return None

    lines = text.splitlines()
    if not lines:
        return None

    tail = '\n'.join(lines[-10:])
    if 'Molpro calculation terminated' not in tail:
        return None

    tail_search = '\n'.join(lines[-80:])
    matches = list(_RE_MOLPRO_SUMMARY_ENERGY.finditer(tail_search))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    matches = list(_RE_MOLPRO_BANG_TOTAL_ENERGY.finditer(tail_search))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    matches = list(_RE_MOLPRO_BANG_TOTAL_ENERGY.finditer(text))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    return None


def get_Etot_molpro(filepath: str) -> Optional[float]:
    """
    Read Molpro stdout log (typically ``molpro.log``) and return the final
    total energy in kcal/mol.
    """
    energy_hartree = get_Etot_molpro_hartree(filepath)
    if energy_hartree is None:
        return None
    return energy_hartree * HARTREE_TO_KCAL


def get_Etot_qxtb_energy(filepath: str) -> Optional[float]:
    """
    Read q-xtb `energy` file and return the last-step energy in kcal/mol.

    Example format:
        $energy
          1     -2391.28658225207528    -3170.71368101119833  99.9 99.9 99.9
        $end

    Assumptions:
      - energy is in Hartree
      - the total energy is the 2nd column (first float after step index)
    """
    if not os.path.exists(filepath):
        return None

    last_energy_hartree: Optional[float] = None
    in_block = False
    try:
        with open(filepath, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.lower().startswith("$energy"):
                    in_block = True
                    continue
                if line.lower().startswith("$end"):
                    in_block = False
                    continue
                if not in_block:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    # parts[0] is step index, parts[1] is energy in eV
                    last_energy_hartree = float(parts[1])
                except ValueError:
                    continue
    except Exception as e:
        print(f"  WARNING: Error reading {filepath}: {e}")
        return None

    if last_energy_hartree is None:
        return None
    return last_energy_hartree * HARTREE_TO_KCAL


def get_Etot_amber(filepath: str) -> Optional[float]:
    """
    Reads Amber-like min.out and returns the final energy in kcal/mol.

    Handles two main formats:
      - DL-FIND style lines:        'Etot   = <value>'
      - Single point style lines:   'EXTERNESCF = <value>' or similar keys
      - FINAL RESULTS section with ENERGY column (MM or mixed runs)

    If the file does not exist, or no energy can be parsed from min.out,
    this function will fall back to reading:

        <dir_of_min.out>/orc_job.dat
        <dir_of_min.out>/molpro.log
        <dir_of_min.out>/energy

    Returns:
        float or None if no energy information could be obtained.
    """
    final_energy: Optional[float] = None

    if not os.path.exists(filepath):
        # Prefer ORCA fallback if present; then Molpro; otherwise try q-xtb.
        workdir = os.path.dirname(filepath)
        orca_path = os.path.join(workdir, 'orc_job.dat')
        orca_e = get_Etot_orca(orca_path)
        if orca_e is not None:
            return orca_e
        molpro_e = get_Etot_molpro(os.path.join(workdir, 'molpro.log'))
        if molpro_e is not None:
            return molpro_e
        return get_Etot_qxtb_energy(os.path.join(workdir, "energy"))

    energy_keywords = ['Etot', 'EXTERNESCF', 'DFTBPLUSESCF', 'XTBESCF']

    try:
        with open(filepath, 'r') as f:
            for line in f:
                for kw in energy_keywords:
                    if kw in line and '=' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == '=' and i + 1 < len(parts):
                                try:
                                    final_energy = float(parts[i + 1])
                                except ValueError:
                                    pass
                                break
                        break
    except Exception as e:
        print(f"  WARNING: Error reading {filepath}: {e}")
        return None

    # Try FINAL RESULTS / ENERGY table if still None
    if final_energy is None:
        try:
            in_final = False
            with open(filepath, 'r') as f:
                for line in f:
                    if 'FINAL RESULTS' in line:
                        in_final = True
                    if in_final and 'NSTEP' in line and 'ENERGY' in line:
                        next_line = next(f, None)
                        if next_line:
                            parts = next_line.split()
                            if len(parts) >= 2:
                                try:
                                    final_energy = float(parts[1])
                                except ValueError:
                                    pass
        except Exception:
            pass

    # If still nothing, fall back to ORCA, Molpro, then q-xtb if present.
    if final_energy is None:
        workdir = os.path.dirname(filepath)
        orca_path = os.path.join(workdir, 'orc_job.dat')
        orca_energy = get_Etot_orca(orca_path)
        if orca_energy is not None:
            print(f"  INFO: Using ORCA energy from {orca_path} (min.out had no result)")
            return orca_energy
        molpro_path = os.path.join(workdir, 'molpro.log')
        molpro_energy = get_Etot_molpro(molpro_path)
        if molpro_energy is not None:
            print(f"  INFO: Using Molpro energy from {molpro_path} (min.out had no result)")
            return molpro_energy
        qxtb_energy = get_Etot_qxtb_energy(os.path.join(workdir, "energy"))
        if qxtb_energy is not None:
            print(f"  INFO: Using q-xtb energy from {os.path.join(workdir, 'energy')} (min.out had no result)")
            return qxtb_energy

    return final_energy


def get_complex_energies(
    run_dir: str,
    complex_name: str,
    prefix: str = '',
    suffix: str = '',
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Generic helper to compute complex energies with optional BSSE correction.

    The directory prefix used for lookup is built as:
        ``<prefix><complex_name><suffix>``

    Examples:
        - ``get_complex_energies(run_dir, '1Zn_1MIm_0MeOH')``
          -> ``1Zn_1MIm_0MeOH_full``
        - ``get_complex_energies(run_dir, '1Zn_1MIm_0MeOH', suffix='_m0.4')``
          -> ``1Zn_1MIm_0MeOH_m0.4_full``
        - ``get_complex_energies(run_dir, '1Zn_1MIm_0MeOH', prefix='alt_')``
          -> ``alt_1Zn_1MIm_0MeOH_full``

    Returns:
        (e_full, e_bsse_corrected, has_bsse):
            e_full           : raw full complex energy from min.out / ORCA
            e_bsse_corrected : E_full - sum(E_ghost_n - E_monomer_n), or None
            has_bsse         : True if BSSE data was complete for all monomers
    """
    label = f"{prefix}{complex_name}{suffix}"

    full_dir = os.path.join(run_dir, f"{label}_full")
    e_full = get_Etot_amber(os.path.join(full_dir, 'min.out'))

    if e_full is None:
        return None, None, False

    monomer_pattern = re.compile(re.escape(label) + r'_monomer_(\d+)$')
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]

    monomer_indices: List[int] = []
    for d in subdirs:
        m = monomer_pattern.match(d)
        if m:
            monomer_indices.append(int(m.group(1)))

    if not monomer_indices:
        return e_full, None, False

    all_ghosts_present = True
    bsse_sum = 0.0
    failed_bsse = []

    for n in sorted(monomer_indices):
        monomer_dir = os.path.join(run_dir, f"{label}_monomer_{n}")
        ghost_dir = os.path.join(run_dir, f"{label}_monomer_{n}_ghost")

        if not os.path.exists(ghost_dir):
            failed_bsse.append(f"{label}_monomer_{n}_ghost (dir missing)")
            all_ghosts_present = False
            continue

        e_monomer = get_Etot_amber(os.path.join(monomer_dir, 'min.out'))
        e_ghost = get_Etot_amber(os.path.join(ghost_dir, 'min.out'))

        if e_monomer is None:
            failed_bsse.append(f"{label}_monomer_{n}")
            all_ghosts_present = False
        if e_ghost is None:
            failed_bsse.append(f"{label}_monomer_{n}_ghost")
            all_ghosts_present = False

        if e_monomer is not None and e_ghost is not None:
            bsse_sum += (e_ghost - e_monomer)

    if failed_bsse:
        print(f"  BSSE failed for {run_dir}/{label}: {', '.join(failed_bsse)}")

    if all_ghosts_present and len(monomer_indices) > 0:
        e_corrected = e_full - bsse_sum
        return e_full, e_corrected, True
    else:
        return e_full, None, False


def parse_complex_composition(complex_name: str) -> Optional[Dict[str, int]]:
    """
    Parse complex labels such as:
        '1Zn_2MIm_3MeOH'
        '1Zn_2MImH_3MeOH'

    into a composition dictionary:
        {'Zn': 1, 'MIm': 2, 'MeOH': 3}
    (the 'MIm' count is used for both MIm and MImH style labels).
    """
    m = re.match(r'(\d+)Zn_(\d+)MImH?_(\d+)MeOH', complex_name)
    if m:
        return {
            'Zn': int(m.group(1)),
            'MIm': int(m.group(2)),
            'MeOH': int(m.group(3)),
        }
    return None


def discover_complexes(xyz_dir: str = XYZ_DIR) -> List[str]:
    """
    Auto-discover complexes from xyz files, excluding monomer xyz files,
    and sort them primarily by ligand count (MIm/MImH) and secondarily
    by decreasing MeOH count.
    """
    complexes: List[str] = []
    if not os.path.exists(xyz_dir):
        return complexes

    for f in os.listdir(xyz_dir):
        if not f.endswith('.xyz'):
            continue
        name = f.replace('.xyz', '')
        if '_monomer' in name:
            continue
        comp = parse_complex_composition(name)
        if comp is not None:
            complexes.append(name)

    complexes.sort(key=lambda c: (parse_complex_composition(c)['MIm'],
                                  -parse_complex_composition(c)['MeOH']))
    return complexes


def calc_ebind(
    e_complex: Optional[float],
    composition: Dict[str, int],
    e_zn: Optional[float],
    e_mim: Optional[float],
    e_meoh: Optional[float],
) -> Optional[float]:
    """
    Binding energy:
        Ebind = E_complex - (n_Zn*E_Zn + n_lig*E_lig + n_MeOH*E_MeOH)
    """
    if e_complex is None or e_zn is None:
        return None
    e_ref = composition['Zn'] * e_zn + composition['MIm'] * e_mim + composition['MeOH'] * e_meoh
    return e_complex - e_ref


def calc_dE(
    e_reactant: Optional[float],
    e_product: Optional[float],
    comp_r: Dict[str, int],
    comp_p: Dict[str, int],
    e_mim: Optional[float],
    e_meoh: Optional[float],
) -> float:
    """
    Generic dE for ligand exchange reactions.

    dn_lig  = lig_P - lig_R (gained)
    dn_MeOH = MeOH_R - MeOH_P (lost)

    Returns np.nan if reactant or product energy is missing.
    """
    if e_reactant is None or e_product is None:
        return np.nan
    dn_mim = comp_p['MIm'] - comp_r['MIm']
    dn_meoh = comp_r['MeOH'] - comp_p['MeOH']
    return e_product + dn_meoh * e_meoh - e_reactant - dn_mim * e_mim


def reaction_label(comp_r: Dict[str, int], comp_p: Dict[str, int]) -> str:
    """
    Human-readable reaction label:
        e.g. '+1MIm-2MeOH' or '+1MImH-2MeOH'

    Uses the globally detected LIGAND_NAME ('MIm' or 'MImH').
    """
    dn_mim = comp_p['MIm'] - comp_r['MIm']
    dn_meoh = comp_r['MeOH'] - comp_p['MeOH']
    parts: List[str] = []
    if dn_mim != 0:
        parts.append(f"+{dn_mim}{LIGAND_NAME}")
    if dn_meoh != 0:
        parts.append(f"-{dn_meoh}MeOH")
    return ''.join(parts) if parts else '(none)'


__all__ = [
    'XYZ_DIR',
    'HARTREE_TO_KCAL',
    'LIGAND_NAME',
    'detect_ligand_name',
    'get_Etot_orca',
    'get_Etot_molpro_hartree',
    'get_Etot_molpro',
    'get_Etot_qxtb_energy',
    'get_Etot_amber',
    'get_complex_energies',
    'parse_complex_composition',
    'discover_complexes',
    'calc_ebind',
    'calc_dE',
    'reaction_label',
]

