#!/usr/bin/env python3
"""
Combined dE analysis for ccsd_test3: ligand exchange (Zn·MeOH -> Zn·MIm or Zn·MImH) and deprotonation.
Also reports Ebind_dist_Zn_MeOH from 1Zn_0MIm_1MeOH full/monomer/ghost fragment energies (no separate Zn/MeOH monomers).

Complex naming matches ORCA single-point runs under SP_init/run*/ etc.:
  - Lig exchange: 1Zn_0MIm_1MeOH -> 1Zn_1MIm_0MeOH (MIm) and -> 1Zn_1MImH_0MeOH (MImH)
  - Deprotonation: 1Zn_1MImH_0MeOH vs 1Zn_1MIm_0MeOH (+ monomer references)
"""

# --- repo path bootstrap (auto) ---
from pathlib import Path as _Path
import sys as _sys
_REPO_CAND = _Path(__file__).resolve().parent
while _REPO_CAND != _REPO_CAND.parent and not (_REPO_CAND / "software.yaml").exists():
    _REPO_CAND = _REPO_CAND.parent
if not (_REPO_CAND / "software.yaml").exists():
    raise RuntimeError("Could not locate repo root (software.yaml)")
REPO_ROOT = _REPO_CAND
TOOLS_DIR = REPO_ROOT / "tools"
_sys.path.insert(0, str(TOOLS_DIR))
try:
    from paths import load_software as _load_software
    _SW = _load_software()
except Exception:
    _SW = {}
# --- end bootstrap ---
import glob
import os
import re
import sys
import yaml

import numpy as np
import pandas as pd

# Repo root and tools (analib)
_BASE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_BASE, "..", ".."))
_PY = os.path.join(ROOT_DIR, "python_scripts")
for _p in (_PY, ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analib import (  # type: ignore
    get_Etot_amber,
    get_Etot_molpro,
    get_complex_energies,
    parse_complex_composition,
    reaction_label,
)

# ORCA energies in 'orc_job.dat' are in Hartree
HARTREE_TO_KCAL = 627.5094740631

_RE_MRCC_CORR_ENERGY = re.compile(
    r"^\s*(CCSD\(F12\*\)\(T\+\)|CCSD\(F12\*\)\(T\*\)|CCSD\(F12\*\)\(T\))\s+correlation energy\s+\[au\]:\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
_RE_MRCC_TOTAL_ENERGY = re.compile(
    r"^\s*Total\s+(CCSD\(F12\*\)\(T\+\)|CCSD\(F12\*\)\(T\*\)|CCSD\(F12\*\)\(T\))\s+energy\s+\[au\]:\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
MRCC_LEVELS = [
    "CCSD(F12*)(T)",
    "CCSD(F12*)(T*)",
    "CCSD(F12*)(T+)",
]

# Deprotonation: Zn·MImH·MeOH vs Zn·MIm·MeOH
COMPLEX_MIMH = "1Zn_1MImH_0MeOH"
COMPLEX_MIM = "1Zn_1MIm_0MeOH"

# Ligand exchange: Zn·(MeOH) -> Zn·(MIm) or Zn·(MImH); third entry is monomer reference for dE
LIG_COMPLEXES = [
    "1Zn_0MIm_1MeOH",
    "1Zn_1MIm_0MeOH",
    "1Zn_1MImH_0MeOH",
]
LIG_REACTIONS = [
    ("1Zn_0MIm_1MeOH", "1Zn_1MIm_0MeOH", "MIm"),
    ("1Zn_0MIm_1MeOH", "1Zn_1MImH_0MeOH", "MImH"),
]

SP_DIRS = ["SP_init", "SP_opt"]
MIN_DIRS = ["minimize", "qm_minimize"]


def _uses_mimh_ligand(complex_name: str) -> bool:
    """True if the complex coordinates MImH (not neutral MIm)."""
    return "_MImH_" in complex_name


def _calc_dE_meoh_swap(
    e_r,
    e_p,
    comp_r,
    comp_p,
    e_lig_ref,
    e_meoh,
):
    """MeOH <-> MIm/MImH exchange; e_lig_ref is E(MIm) or E(MImH) as appropriate."""
    if any(pd.isna(x) for x in (e_r, e_p, e_lig_ref, e_meoh)):
        return np.nan
    dn_mim = comp_p["MIm"] - comp_r["MIm"]
    dn_meoh = comp_r["MeOH"] - comp_p["MeOH"]
    return e_p + dn_meoh * e_meoh - e_r - dn_mim * e_lig_ref


def _calc_ebind(
    complex_name: str,
    e_complex,
    composition,
    e_zn,
    e_mim,
    e_mimh,
    e_meoh,
):
    """Binding energy; MImH complexes use E(MImH), others E(MIm) for the imidazole fragment."""
    if pd.isna(e_complex):
        return None
    e_lig = e_mimh if _uses_mimh_ligand(complex_name) else e_mim
    # Only require monomer energies that are actually used by this composition.
    if composition["Zn"] > 0 and pd.isna(e_zn):
        return None
    if composition["MIm"] > 0 and pd.isna(e_lig):
        return None
    if composition["MeOH"] > 0 and pd.isna(e_meoh):
        return None

    e_ref = 0.0
    if composition["Zn"] > 0:
        e_ref += composition["Zn"] * e_zn
    if composition["MIm"] > 0:
        e_ref += composition["MIm"] * e_lig
    if composition["MeOH"] > 0:
        e_ref += composition["MeOH"] * e_meoh
    return e_complex - e_ref


def _rxn_col(r, p, compositions, lig_kind: str) -> str:
    if lig_kind == "MIm":
        lbl = reaction_label(compositions[r], compositions[p])
        return f"dE({r}->{p}) {lbl}"
    return f"dE({r}->{p}) +1MImH-1MeOH"


def _energy_path(run_path, subdir):
    """Path to min.out; get_Etot_amber falls back to orc_job.dat in the same directory."""
    return os.path.join(run_path, subdir, "min.out")


def _parse_orca_before_and_final_hartree(dat_path: str):
    """
    Parse ORCA text output (typically 'orc_job.dat') for:
      - E(TOT)-before F12 corrections  (Hartree)  [may be absent]
      - FINAL SINGLE POINT ENERGY     (Hartree)
    Returns (before_hartree_or_None, final_hartree_or_None, triples_hartree_or_None)
    """
    if not os.path.exists(dat_path):
        return None, None, None
    before = None
    final = None
    triples = None
    try:
        with open(dat_path, "r") as f:
            for line in f:
                if "E(TOT)-before F12 corrections" in line:
                    parts = line.split()
                    try:
                        before = float(parts[-1])
                    except Exception:
                        pass
                elif "FINAL SINGLE POINT ENERGY" in line:
                    parts = line.split()
                    try:
                        final = float(parts[-1])
                    except Exception:
                        pass
                elif "Triples Correction (T)" in line:
                    parts = line.split()
                    try:
                        triples = float(parts[-1])
                    except Exception:
                        pass
    except Exception:
        return None, None, None
    return before, final, triples


def _get_orca_energy_kcal(dat_path: str, f12corr: str):
    """
    f12corr:
      - "yes": use FINAL SINGLE POINT ENERGY
      - "no" : if before-F12 exists use it, else fall back to FINAL
    Returns kcal/mol or None.
    """
    before_h, final_h, triples_h = _parse_orca_before_and_final_hartree(dat_path)
    if final_h is None and before_h is None:
        return None
    if f12corr == "yes":
        return None if final_h is None else final_h * HARTREE_TO_KCAL
    # f12corr == "no"
    # For F12 jobs, ORCA's "E(TOT)-before F12 corrections" is typically at CCSD level;
    # include unscaled triples when available to get no-F12 CCSD(T)-like energy.
    if before_h is not None and triples_h is not None:
        return (before_h + triples_h) * HARTREE_TO_KCAL
    if before_h is not None:
        return before_h * HARTREE_TO_KCAL
    return None if final_h is None else final_h * HARTREE_TO_KCAL

def _parse_mrcc_level_energies_hartree(log_path: str):
    """
    Parse MRCC level energies from mrcc.log.
    Prefer TOTAL energies (consistent with other methods in this script).
    Fall back to correlation energies only when total lines are absent.
    Expected lines like:
      Total CCSD(F12*)(T)  energy [au]:       -341.962...
      Total CCSD(F12*)(T*) energy [au]:       -341.968...
      Total CCSD(F12*)(T+) energy [au]:       -341.968...
    Returns dict {level: value_hartree}.
    """
    if not os.path.exists(log_path):
        return {}
    out_total = {}
    out_corr = {}
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                mt = _RE_MRCC_TOTAL_ENERGY.match(line)
                if mt:
                    level = mt.group(1)
                    try:
                        out_total[level] = float(mt.group(2))
                    except Exception:
                        pass
                    continue
                mc = _RE_MRCC_CORR_ENERGY.match(line)
                if mc:
                    level = mc.group(1)
                    try:
                        out_corr[level] = float(mc.group(2))
                    except Exception:
                        pass
    except Exception:
        return {}
    return out_total if out_total else out_corr


def _get_mrcc_energy_kcal(log_path: str, mrcc_level: str):
    e_map = _parse_mrcc_level_energies_hartree(log_path)
    if mrcc_level not in e_map:
        return None
    return e_map[mrcc_level] * HARTREE_TO_KCAL


def _detect_run_mrcc_levels(run_path: str):
    """
    Detect which MRCC levels are available for this run.
    Probe the ligand-reference full complex first.
    """
    probe = os.path.join(run_path, f"{LIG_COMPLEXES[0]}_full", "mrcc.log")
    e_map = _parse_mrcc_level_energies_hartree(probe)
    return [lvl for lvl in MRCC_LEVELS if lvl in e_map]


def _run_has_f12_before(run_path: str) -> bool:
    """
    Heuristic: detect F12 runs by checking for 'before F12' energy in ORCA output.

    Prefer Zn_monomer/orc_job.dat (historical convention). If that file is missing
    (job not run yet) but ligand-exchange complex SPs are present, fall back to
    the first ligand complex full calculation (same probe idea as MRCC detection).
    """
    for rel in (
        os.path.join("Zn_monomer", "orc_job.dat"),
        os.path.join(f"{LIG_COMPLEXES[0]}_full", "orc_job.dat"),
    ):
        probe = os.path.join(run_path, rel)
        before_h, _, _ = _parse_orca_before_and_final_hartree(probe)
        if before_h is not None:
            return True
    return False


def _complex_energies_for_f12corr(run_dir: str, complex_name: str, f12corr: str):
    """
    Like analib.get_complex_energies, but for a specific F12corr variant and reading ORCA outputs directly.
    Energies are returned in kcal/mol.
    """
    full_dir = os.path.join(run_dir, f"{complex_name}_full")
    e_full = _get_orca_energy_kcal(os.path.join(full_dir, "orc_job.dat"), f12corr)
    if e_full is None:
        return None, None, False

    # Detect monomer indices present
    monomer_indices = []
    monomer_prefix = f"{complex_name}_monomer_"
    try:
        for d in os.listdir(run_dir):
            if not os.path.isdir(os.path.join(run_dir, d)):
                continue
            if d.startswith(monomer_prefix) and d[len(monomer_prefix) :].isdigit():
                monomer_indices.append(int(d[len(monomer_prefix) :]))
    except Exception:
        monomer_indices = []

    if not monomer_indices:
        return e_full, None, False

    all_ghosts_present = True
    bsse_sum = 0.0
    for n in sorted(monomer_indices):
        monomer_dir = os.path.join(run_dir, f"{complex_name}_monomer_{n}")
        ghost_dir = os.path.join(run_dir, f"{complex_name}_monomer_{n}_ghost")
        if not os.path.exists(ghost_dir):
            all_ghosts_present = False
            continue
        e_monomer = _get_orca_energy_kcal(os.path.join(monomer_dir, "orc_job.dat"), f12corr)
        e_ghost = _get_orca_energy_kcal(os.path.join(ghost_dir, "orc_job.dat"), f12corr)
        if e_monomer is None or e_ghost is None:
            all_ghosts_present = False
            continue
        bsse_sum += (e_ghost - e_monomer)

    if all_ghosts_present and len(monomer_indices) > 0:
        return e_full, e_full - bsse_sum, True
    return e_full, None, False


def _fragment_energy_kcal(
    run_path: str,
    rel_subdir: str,
    is_sp: bool,
    run_has_f12: bool,
    f12corr: str,
    mrcc_level: str,
):
    """Energy (kcal/mol) for a single-job subdirectory; F12 SP runs use orc_job.dat + f12corr variant."""
    sub = os.path.join(run_path, rel_subdir)
    dat = os.path.join(sub, "orc_job.dat")
    mrcclog = os.path.join(sub, "mrcc.log")
    mlog = os.path.join(sub, "molpro.log")
    mout = os.path.join(sub, "min.out")
    if is_sp and run_has_f12:
        return _get_orca_energy_kcal(dat, f12corr)
    if is_sp:
        # MRCC: use selected correlation level if requested.
        if mrcc_level != "none" and os.path.exists(mrcclog):
            e_mrcc = _get_mrcc_energy_kcal(mrcclog, mrcc_level)
            if e_mrcc is not None:
                return e_mrcc
        # Non-F12 SP: prefer ORCA dat if present, else Molpro stdout log.
        if os.path.exists(dat):
            # 'yes' (final) is appropriate when there is no explicit before-F12 energy.
            return _get_orca_energy_kcal(dat, "yes")
        if os.path.exists(mlog):
            return get_Etot_molpro(mlog)
    return get_Etot_amber(mout)


def _ebind_dist_zn_meoh(
    run_path: str,
    is_sp: bool,
    run_has_f12: bool,
    f12corr: str,
    bsse_flag: str,
    mrcc_level: str,
):
    """
    Zn-MeOH binding from 1Zn_0MIm_1MeOH fragment calculations only (no Zn_monomer / MeOH_monomer).
    BSSE no: E(full) - E(monomer_0) - E(monomer_1)
    BSSE yes: counterpoise E(full) - E(monomer_0_ghost) - E(monomer_1_ghost)
    """
    c = LIG_COMPLEXES[0]  # 1Zn_0MIm_1MeOH
    e_full = _fragment_energy_kcal(
        run_path, f"{c}_full", is_sp, run_has_f12, f12corr, mrcc_level
    )
    if bsse_flag == "no":
        e0 = _fragment_energy_kcal(
            run_path, f"{c}_monomer_0", is_sp, run_has_f12, f12corr, mrcc_level
        )
        e1 = _fragment_energy_kcal(
            run_path, f"{c}_monomer_1", is_sp, run_has_f12, f12corr, mrcc_level
        )
    else:
        e0 = _fragment_energy_kcal(
            run_path, f"{c}_monomer_0_ghost", is_sp, run_has_f12, f12corr, mrcc_level
        )
        e1 = _fragment_energy_kcal(
            run_path, f"{c}_monomer_1_ghost", is_sp, run_has_f12, f12corr, mrcc_level
        )
    if e_full is None or e0 is None or e1 is None:
        return np.nan
    return e_full - e0 - e1


def _complex_energies_for_mrcc(run_dir: str, complex_name: str, mrcc_level: str):
    """
    MRCC analogue of get_complex_energies using selected correlation level from mrcc.log.
    """
    full_dir = os.path.join(run_dir, f"{complex_name}_full")
    e_full = _get_mrcc_energy_kcal(os.path.join(full_dir, "mrcc.log"), mrcc_level)
    if e_full is None:
        return None, None, False

    monomer_indices = []
    monomer_prefix = f"{complex_name}_monomer_"
    try:
        for d in os.listdir(run_dir):
            if not os.path.isdir(os.path.join(run_dir, d)):
                continue
            if d.startswith(monomer_prefix) and d[len(monomer_prefix) :].isdigit():
                monomer_indices.append(int(d[len(monomer_prefix) :]))
    except Exception:
        monomer_indices = []

    if not monomer_indices:
        return e_full, None, False

    all_ghosts_present = True
    bsse_sum = 0.0
    for n in sorted(monomer_indices):
        monomer_dir = os.path.join(run_dir, f"{complex_name}_monomer_{n}")
        ghost_dir = os.path.join(run_dir, f"{complex_name}_monomer_{n}_ghost")
        if not os.path.exists(ghost_dir):
            all_ghosts_present = False
            continue
        e_monomer = _get_mrcc_energy_kcal(
            os.path.join(monomer_dir, "mrcc.log"), mrcc_level
        )
        e_ghost = _get_mrcc_energy_kcal(
            os.path.join(ghost_dir, "mrcc.log"), mrcc_level
        )
        if e_monomer is None or e_ghost is None:
            all_ghosts_present = False
            continue
        bsse_sum += (e_ghost - e_monomer)

    if all_ghosts_present and len(monomer_indices) > 0:
        return e_full, e_full - bsse_sum, True
    return e_full, None, False


def analyze_runs(base_dirs=None):
    if base_dirs is None:
        base_dirs = SP_DIRS + MIN_DIRS

    compositions = {c: parse_complex_composition(c) for c in LIG_COMPLEXES}
    for c in LIG_COMPLEXES:
        if compositions[c] is None:
            raise ValueError(f"Bad complex label: {c}")
    rxn_cols = [
        _rxn_col(r, p, compositions, lk) for r, p, lk in LIG_REACTIONS
    ]

    results = []

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        is_sp = base_dir in SP_DIRS
        run_dirs = sorted(glob.glob(os.path.join(base_dir, "run*")))

        for run_path in run_dirs:
            run_name = os.path.basename(run_path)

            method_name = run_name
            notes_path = os.path.join(run_path, "notes.yaml")
            if os.path.exists(notes_path):
                with open(notes_path, "r") as f:
                    try:
                        notes = yaml.safe_load(f)
                        method_name = notes.get("name", run_name)
                    except Exception:
                        pass

            run_has_before_f12 = bool(is_sp and _run_has_f12_before(run_path))

            # For F12 runs, output two rows per BSSE flag:
            #   - F12corr=no  (before F12 corrections)
            #   - F12corr=yes (final energy with F12 corrections)
            # For non-F12 runs, output only F12corr=no (existing behavior).
            f12corr_variants = ["no", "yes"] if run_has_before_f12 else ["no"]

            mrcc_levels = (
                _detect_run_mrcc_levels(run_path)
                if (is_sp and not run_has_before_f12)
                else []
            )
            mrcc_variants = mrcc_levels if mrcc_levels else ["none"]

            for f12corr, mrcc_level in (
                (f, m) for f in f12corr_variants for m in mrcc_variants
            ):
                if run_has_before_f12:
                    # Monomers (kcal/mol), split into before/after F12
                    e_zn = _get_orca_energy_kcal(
                        os.path.join(run_path, "Zn_monomer", "orc_job.dat"), f12corr
                    )
                    e_mim = _get_orca_energy_kcal(
                        os.path.join(run_path, "MIm_monomer", "orc_job.dat"), f12corr
                    )
                    e_meoh = _get_orca_energy_kcal(
                        os.path.join(run_path, "MeOH_monomer", "orc_job.dat"), f12corr
                    )
                    e_mimh = _get_orca_energy_kcal(
                        os.path.join(run_path, "MImH_monomer", "orc_job.dat"), f12corr
                    )
                else:
                    # Non-F12 runs: could be minimization (min.out) or SP (orc_job.dat / molpro.log).
                    e_zn = _fragment_energy_kcal(
                        run_path, "Zn_monomer", is_sp, False, f12corr, mrcc_level
                    )
                    e_mim = _fragment_energy_kcal(
                        run_path, "MIm_monomer", is_sp, False, f12corr, mrcc_level
                    )
                    e_meoh = _fragment_energy_kcal(
                        run_path, "MeOH_monomer", is_sp, False, f12corr, mrcc_level
                    )
                    e_mimh = _fragment_energy_kcal(
                        run_path, "MImH_monomer", is_sp, False, f12corr, mrcc_level
                    )

                if e_mim is None or e_meoh is None or e_mimh is None:
                    missing = []
                    if e_mim is None:
                        missing.append("MIm_monomer")
                    if e_meoh is None:
                        missing.append("MeOH_monomer")
                    if e_mimh is None:
                        missing.append("MImH_monomer")
                    print(
                        f"Warning: {base_dir}/{run_name} (F12corr={f12corr}): missing {', '.join(missing)}, related terms will be NaN"
                    )
                # Keep this run and let downstream formulas naturally produce NaN when inputs are missing.
                e_mim = np.nan if e_mim is None else e_mim
                e_meoh = np.nan if e_meoh is None else e_meoh
                e_mimh = np.nan if e_mimh is None else e_mimh
                if e_zn is None:
                    print(
                        f"Warning: {base_dir}/{run_name} (F12corr={f12corr}): missing Zn_monomer, Ebind will be NaN"
                    )
                    e_zn = np.nan

                # H monomer: optional
                if run_has_before_f12:
                    e_h = _get_orca_energy_kcal(
                        os.path.join(run_path, "H_monomer", "orc_job.dat"), f12corr
                    )
                else:
                    e_h = _fragment_energy_kcal(
                        run_path, "H_monomer", is_sp, False, f12corr, mrcc_level
                    )
                if e_h is None:
                    e_h = 0.0
                    print(
                        f"  WARNING: H_monomer energy missing in {run_path} (F12corr={f12corr}), using E_H = 0"
                    )

                raw_lig = {}
                bsse_lig = {}
                has_bsse_map = {}

                for comp in LIG_COMPLEXES:
                    if is_sp and run_has_before_f12:
                        e_raw, e_corr, bsse_found = _complex_energies_for_f12corr(
                            run_path, comp, f12corr
                        )
                    elif is_sp and mrcc_level != "none":
                        e_raw, e_corr, bsse_found = _complex_energies_for_mrcc(
                            run_path, comp, mrcc_level
                        )
                    elif is_sp:
                        e_raw, e_corr, bsse_found = get_complex_energies(run_path, comp)
                    else:
                        e_raw = get_Etot_amber(_energy_path(run_path, comp))
                        e_corr, bsse_found = None, False

                    if e_raw is None:
                        print(
                            f"Warning: {base_dir}/{run_name} (F12corr={f12corr}): missing energy for {comp}, related terms will be NaN"
                        )
                        raw_lig[comp] = np.nan
                        bsse_lig[comp] = None
                        has_bsse_map[comp] = False
                        continue
                    raw_lig[comp] = e_raw
                    bsse_lig[comp] = e_corr
                    has_bsse_map[comp] = bsse_found

                e_mimh_full = raw_lig[COMPLEX_MIMH]
                e_mim_full = raw_lig[COMPLEX_MIM]
                e_mimh_bsse = bsse_lig[COMPLEX_MIMH]
                e_mim_bsse = bsse_lig[COMPLEX_MIM]
                has_bsse_mimh = has_bsse_map[COMPLEX_MIMH]
                has_bsse_mim = has_bsse_map[COMPLEX_MIM]

                has_bsse_deproton = has_bsse_mimh and has_bsse_mim
                has_bsse_lig_any = any(has_bsse_map.values())

                def build_deproton_row(e_zn_mimh_meoh, e_zn_mim_meoh):
                    if e_mimh is not None and e_mim is not None:
                        dE_deproton = e_mim + e_h - e_mimh
                    else:
                        dE_deproton = np.nan
                    if e_zn_mimh_meoh is not None and e_zn_mim_meoh is not None:
                        dE_complex_deproton = e_zn_mim_meoh + e_h - e_zn_mimh_meoh
                    else:
                        dE_complex_deproton = np.nan
                    return dE_deproton, dE_complex_deproton

                def build_row(bsse_flag, energies_dep_mimh, energies_dep_mim, lig_e):
                    """lig_e: dict complex_name -> energy for LIG_COMPLEXES."""
                    d1, d2 = build_deproton_row(energies_dep_mimh, energies_dep_mim)
                    row = {
                        "Phase": base_dir,
                        "Run": run_name,
                        "Method": method_name,
                        "BSSE": bsse_flag,
                        "F12corr": f12corr,
                        "MRCCLevel": mrcc_level,
                        "E_Zn": e_zn,
                        "E_MImH": e_mimh,
                        "E_MIm": e_mim,
                        "E_H": e_h,
                        "E_MeOH": e_meoh,
                        f"E_{COMPLEX_MIMH}": energies_dep_mimh,
                        "dE_MImH_MIm_H": d1,
                        f"dE_{COMPLEX_MIMH}_{COMPLEX_MIM}_H": d2,
                    }
                    for comp in LIG_COMPLEXES:
                        row[f"E_{comp}"] = lig_e[comp]
                        eb = _calc_ebind(
                            comp,
                            lig_e[comp],
                            compositions[comp],
                            e_zn,
                            e_mim,
                            e_mimh,
                            e_meoh,
                        )
                        row[f"Ebind_{comp}"] = eb if eb is not None else np.nan
                    row["Ebind_dist_Zn_MeOH"] = _ebind_dist_zn_meoh(
                        run_path,
                        is_sp,
                        run_has_before_f12,
                        f12corr,
                        bsse_flag,
                        mrcc_level,
                    )
                    for col, (r, p, lk) in zip(rxn_cols, LIG_REACTIONS):
                        e_ref = e_mim if lk == "MIm" else e_mimh
                        row[col] = _calc_dE_meoh_swap(
                            lig_e[r],
                            lig_e[p],
                            compositions[r],
                            compositions[p],
                            e_ref,
                            e_meoh,
                        )
                    return row

                results.append(
                    build_row("no", e_mimh_full, e_mim_full, raw_lig.copy())
                )

                # Always emit BSSE=yes row for SP runs.
                # Missing BSSE terms stay NaN so partially available data is still visible.
                has_any_bsse_value = any(bsse_lig[c] is not None for c in LIG_COMPLEXES)
                if is_sp:
                    bsse_lig_map = {
                        c: (bsse_lig[c] if bsse_lig[c] is not None else np.nan)
                        for c in LIG_COMPLEXES
                    }
                    results.append(
                        build_row(
                            "yes",
                            e_mimh_bsse if e_mimh_bsse is not None else np.nan,
                            e_mim_bsse if e_mim_bsse is not None else np.nan,
                            bsse_lig_map,
                        )
                    )
                    if (has_bsse_mimh or has_bsse_mim) and not (
                        has_bsse_deproton and e_mimh_bsse is not None and e_mim_bsse is not None
                    ):
                        print(
                            f"  {base_dir}/{run_name} (F12corr={f12corr}): partial BSSE (deprotonation complexes), BSSE row kept with NaN"
                        )
                    if has_bsse_lig_any and not all(bsse_lig[c] is not None for c in LIG_COMPLEXES):
                        print(
                            f"  {base_dir}/{run_name} (F12corr={f12corr}): partial BSSE (lig exchange complexes), BSSE row kept with NaN"
                        )
                    if not has_any_bsse_value:
                        print(
                            f"  {base_dir}/{run_name} (F12corr={f12corr}): no BSSE-corrected complex energies, BSSE row kept with NaN"
                        )

                # done for current mrcc_level variant
            # done for all f12corr variants for this run_path
            continue

            e_zn = get_Etot_amber(_energy_path(run_path, "Zn_monomer"))
            e_mim = get_Etot_amber(_energy_path(run_path, "MIm_monomer"))
            e_meoh = get_Etot_amber(_energy_path(run_path, "MeOH_monomer"))
            e_mimh = get_Etot_amber(_energy_path(run_path, "MImH_monomer"))

            if e_mim is None or e_meoh is None or e_mimh is None:
                missing = []
                if e_mim is None:
                    missing.append("MIm_monomer")
                if e_meoh is None:
                    missing.append("MeOH_monomer")
                if e_mimh is None:
                    missing.append("MImH_monomer")
                print(f"Skipping {base_dir}/{run_name}: missing {', '.join(missing)}")
                continue
            if e_zn is None:
                print(f"Warning: {base_dir}/{run_name}: missing Zn_monomer, Ebind will be NaN")

            e_h_path = _energy_path(run_path, "H_monomer")
            e_h = get_Etot_amber(e_h_path)
            if e_h is None:
                e_h = 0.0
                print(f"  WARNING: H_monomer energy missing in {run_path}, using E_H = 0")

            raw_lig = {}
            bsse_lig = {}
            has_bsse_map = {}

            skip_run = False
            for comp in LIG_COMPLEXES:
                if is_sp:
                    e_raw, e_corr, bsse_found = get_complex_energies(run_path, comp)
                else:
                    e_raw = get_Etot_amber(_energy_path(run_path, comp))
                    e_corr, bsse_found = None, False

                if e_raw is None:
                    print(f"Skipping {base_dir}/{run_name}: missing energy for {comp}")
                    skip_run = True
                    break
                raw_lig[comp] = e_raw
                bsse_lig[comp] = e_corr
                has_bsse_map[comp] = bsse_found

            if skip_run:
                continue

            e_mimh_full = raw_lig[COMPLEX_MIMH]
            e_mim_full = raw_lig[COMPLEX_MIM]
            e_mimh_bsse = bsse_lig[COMPLEX_MIMH]
            e_mim_bsse = bsse_lig[COMPLEX_MIM]
            has_bsse_mimh = has_bsse_map[COMPLEX_MIMH]
            has_bsse_mim = has_bsse_map[COMPLEX_MIM]

            has_bsse_deproton = has_bsse_mimh and has_bsse_mim
            has_bsse_lig_any = any(has_bsse_map.values())

            def build_deproton_row(e_zn_mimh_meoh, e_zn_mim_meoh):
                if e_mimh is not None and e_mim is not None:
                    dE_deproton = e_mim + e_h - e_mimh
                else:
                    dE_deproton = np.nan
                if e_zn_mimh_meoh is not None and e_zn_mim_meoh is not None:
                    dE_complex_deproton = e_zn_mim_meoh + e_h - e_zn_mimh_meoh
                else:
                    dE_complex_deproton = np.nan
                return dE_deproton, dE_complex_deproton

            def build_row(bsse_flag, energies_dep_mimh, energies_dep_mim, lig_e):
                """lig_e: dict complex_name -> energy for LIG_COMPLEXES."""
                d1, d2 = build_deproton_row(energies_dep_mimh, energies_dep_mim)
                row = {
                    "Phase": base_dir,
                    "Run": run_name,
                    "Method": method_name,
                    "BSSE": bsse_flag,
                    "F12corr": "no",
                    "E_Zn": e_zn,
                    "E_MImH": e_mimh,
                    "E_MIm": e_mim,
                    "E_H": e_h,
                    "E_MeOH": e_meoh,
                    f"E_{COMPLEX_MIMH}": energies_dep_mimh,
                    "dE_MImH_MIm_H": d1,
                    f"dE_{COMPLEX_MIMH}_{COMPLEX_MIM}_H": d2,
                }
                for comp in LIG_COMPLEXES:
                    row[f"E_{comp}"] = lig_e[comp]
                    eb = _calc_ebind(
                        comp,
                        lig_e[comp],
                        compositions[comp],
                        e_zn,
                        e_mim,
                        e_mimh,
                        e_meoh,
                    )
                    row[f"Ebind_{comp}"] = eb if eb is not None else np.nan
                for col, (r, p, lk) in zip(rxn_cols, LIG_REACTIONS):
                    e_ref = e_mim if lk == "MIm" else e_mimh
                    row[col] = _calc_dE_meoh_swap(
                        lig_e[r],
                        lig_e[p],
                        compositions[r],
                        compositions[p],
                        e_ref,
                        e_meoh,
                    )
                return row

            results.append(
                build_row("no", e_mimh_full, e_mim_full, raw_lig.copy())
            )

            all_lig_bsse_ok = all(bsse_lig[c] is not None for c in LIG_COMPLEXES)
            can_bsse_row = (
                is_sp
                and has_bsse_deproton
                and e_mimh_bsse is not None
                and e_mim_bsse is not None
                and has_bsse_lig_any
                and all_lig_bsse_ok
            )
            if can_bsse_row:
                bsse_lig_map = {c: bsse_lig[c] for c in LIG_COMPLEXES}
                results.append(
                    build_row("yes", e_mimh_bsse, e_mim_bsse, bsse_lig_map)
                )
            elif is_sp and (
                (has_bsse_mimh or has_bsse_mim)
                and not (has_bsse_deproton and e_mimh_bsse is not None and e_mim_bsse is not None)
            ):
                print(
                    f"  {base_dir}/{run_name}: partial BSSE (deprotonation complexes), skipping BSSE row"
                )
            elif is_sp and has_bsse_lig_any and not all_lig_bsse_ok:
                print(
                    f"  {base_dir}/{run_name}: partial BSSE (lig exchange complexes), skipping BSSE row"
                )

    return pd.DataFrame(results)


def _run_id_as_int(run_name: str):
    """Extract numeric run id from names like 'run95' for numeric sorting."""
    m = re.search(r"(\d+)$", str(run_name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


if __name__ == "__main__":
    df = analyze_runs()

    if df.empty:
        print("No data found.")
        sys.exit(0)

    rxn_cols = [
        _rxn_col(r, p, {c: parse_complex_composition(c) for c in LIG_COMPLEXES}, lk)
        for r, p, lk in LIG_REACTIONS
    ]
    ebind_cols = [c for c in df.columns if c.startswith("Ebind_")]
    depro_cols = [
        "dE_MImH_MIm_H",
        f"dE_{COMPLEX_MIMH}_{COMPLEX_MIM}_H",
    ]
    depro_cols = [c for c in depro_cols if c in df.columns]

    # CSV keeps Ebind_*; screen summary is dE-only
    cols = ["Phase", "Run", "Method", "BSSE", "F12corr", "MRCCLevel"]
    cols += [c for c in df.columns if c.startswith("E_") and c not in cols]
    cols += ebind_cols
    cols += rxn_cols
    cols += depro_cols
    seen = set()
    ordered = []
    for c in cols:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    for c in df.columns:
        if c not in seen:
            ordered.append(c)
    df = df[ordered]

    # Sort rows by numeric run id (e.g. run2 < run10), then keep a stable
    # deterministic order for variants inside each run.
    df["_RunID"] = pd.to_numeric(df["Run"].map(_run_id_as_int), errors="coerce")
    sort_cols = [c for c in ("Phase", "_RunID", "Run", "BSSE", "F12corr", "MRCCLevel") if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, kind="mergesort", inplace=True, ignore_index=True)
    if "_RunID" in df.columns:
        df.drop(columns=["_RunID"], inplace=True)

    output_file = "dE.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    print("\nLigand exchange reaction columns:")
    for c in rxn_cols:
        print(f"  {c}")

    print("\nDeprotonation column legend:")
    print("  dE_MImH_MIm_H: MImH = MIm + H  (monomer)")
    print(
        f"  dE_{COMPLEX_MIMH}_{COMPLEX_MIM}_H: {COMPLEX_MIMH} = {COMPLEX_MIM} + H"
    )

    show_de = ["Phase", "Run", "Method", "BSSE", "F12corr", "MRCCLevel"] + rxn_cols + depro_cols
    show_de = [c for c in show_de if c in df.columns]
    print("\nSummary — dE only (kcal/mol); full table including Ebind_* is in dE.csv:")
    print(df[show_de].to_string())
