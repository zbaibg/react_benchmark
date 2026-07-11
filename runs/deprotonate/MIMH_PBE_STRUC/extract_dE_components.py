#!/usr/bin/env python3
"""
Extract the four thermodynamic cycle reactions from CSV files.

Reactions:
  1. [Zn·MeOH4]²⁺ + MImH → [Zn·MImH·MeOH3]²⁺ + MeOH  (ligand exchange, MImH)
  2. [Zn·MImH·MeOH3]²⁺ → [Zn·MIm·MeOH3]⁺ + H⁺       (deprotonation of complex)
  3. [Zn·MeOH4]²⁺ + MIm⁻ → [Zn·MIm·MeOH3]⁺ + MeOH   (ligand exchange, MIm⁻)
  4. MImH → MIm⁻ + H⁺                                (free ligand deprotonation)

dE: PBE-D3(BJ) def2-TZVPPD (mDFTB3D3/3ob_prime)
  - mDFTB3D3/3ob_prime: use SP_opt
  - PBE-D3(BJ) def2: use SP_init
Error = E(mDFTB3D3/3ob_prime) - E(PBE-D3(BJ) def2)
dE.csv has BSSE column (yes/no): PBE values and cycle closure use BSSE yes / BSSE no separately.
"""

import os
import re

import pandas as pd


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARTREE_TO_KCAL = 627.509474

# Mapping from symbolic species keys to directory info.
# For most keys the species live in this deprotonation directory.
# Some ligand-exchange species live under ../lig_exchange/{MIMH_PBE_STRUC|MIM_PBE_STRUC}.
#
# For deprotonation species we keep backwards-compatible keys matching dE.csv:
SPECIES_DIR_MAP = {
    # Deprotonation benchmark (this directory)
    'E_MImH': ('deprot', 'MImH_monomer'),
    'E_MIm': ('deprot', 'MIm_monomer'),
    'E_H': ('deprot', 'H_monomer'),
    'E_H2O': ('deprot', 'Wat_monomer'),
    'E_H3O': ('deprot', 'H3O_monomer'),
    'E_1Zn_1MImH_3MeOH': ('deprot', '1Zn_1MImH_3MeOH_full'),
    'E_1Zn_1MIm_3MeOH': ('deprot', '1Zn_1MIm_3MeOH_full'),
    # Ligand-exchange systems, MImH branch (folder: lig_exchange/MIMH_PBE_STRUC)
    'LE_MIMH_ZnMeOH4': ('lig_mimh', '1Zn_0MImH_4MeOH_full'),
    'LE_MIMH_ZnMImH': ('lig_mimh', '1Zn_1MImH_3MeOH_full'),
    'LE_MIMH_MImH': ('lig_mimh', 'MImH_monomer'),
    'LE_MIMH_MeOH': ('lig_mimh', 'MeOH_monomer'),
    # Ligand-exchange systems, MIm- branch (folder: lig_exchange/MIM_PBE_STRUC)
    'LE_MIM_ZnMeOH4': ('lig_mim', '1Zn_0MIm_4MeOH_full'),
    'LE_MIM_ZnMIm': ('lig_mim', '1Zn_1MIm_3MeOH_full'),
    'LE_MIM_MIm': ('lig_mim', 'MIm_monomer'),
    'LE_MIM_MeOH': ('lig_mim', 'MeOH_monomer'),
}

# For energy-component analysis we首先关注 dE.csv 中的酸碱反应，
# 然后再扩展到配体交换反应。化学计量用上面的符号键表示。
REACTION_COMPONENT_SPECS = [
    (
        '[Zn·MImH·MeOH3]²⁺ → [Zn·MIm·MeOH3]⁺ + H⁺',
        {
            'E_1Zn_1MImH_3MeOH': -1,
            'E_1Zn_1MIm_3MeOH': 1,
            'E_H': 1,
        },
    ),
    (
        'MImH → MIm⁻ + H⁺',
        {
            'E_MImH': -1,
            'E_MIm': 1,
            'E_H': 1,
        },
    ),
    (
        '[Zn·MImH·MeOH3]²⁺ + H2O → [Zn·MIm·MeOH3]⁺ + H3O⁺',
        {
            'E_1Zn_1MImH_3MeOH': -1,
            'E_H2O': -1,
            'E_1Zn_1MIm_3MeOH': 1,
            'E_H3O': 1,
        },
    ),
    (
        'MImH + H2O → MIm⁻ + H3O⁺',
        {
            'E_MImH': -1,
            'E_H2O': -1,
            'E_MIm': 1,
            'E_H3O': 1,
        },
    ),
    # Ligand-exchange reactions: use ligand_exchange directories
    (
        '[Zn·MeOH4]²⁺ + MImH → [Zn·MImH·MeOH3]²⁺ + MeOH',
        {
            'LE_MIMH_ZnMeOH4': -1,
            'LE_MIMH_MImH': -1,
            'LE_MIMH_ZnMImH': 1,
            'LE_MIMH_MeOH': 1,
        },
    ),
    (
        '[Zn·MeOH4]²⁺ + MIm⁻ → [Zn·MIm·MeOH3]⁺ + MeOH',
        {
            'LE_MIM_ZnMeOH4': -1,
            'LE_MIM_MIm': -1,
            'LE_MIM_ZnMIm': 1,
            'LE_MIM_MeOH': 1,
        },
    ),
]


def parse_dftb_energy_components(filepath):
    """
    Parse all energy components from DFTB3 detailed.out.
    Returns dict: {component_label -> energy_in_Hartree}.

    DFTB energy hierarchy (in detailed.out):
      Total energy = Total Electronic energy + Repulsive energy + Dispersion energy
      Total Electronic energy = Energy H0 + Energy SCC + Energy 3rd + Energy Multipole
      Energy Multipole = sum of the six multipole terms (Monopole-Dipole, etc.)
    Band energy / Band free energy are eigenvalue sums, not part of Total energy.
    """
    comps = {}
    if not os.path.exists(filepath):
        return comps
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            name, rest = line.split(':', 1)
            name = name.strip()
            lower = name.lower()
            # Collect various "energy" lines plus total free energy variants
            if (
                'energy' in lower
                or lower.startswith('total mermin free energy')
                or lower.startswith('extrapolated to 0')
                or lower.startswith('force related energy')
            ):
                m = re.search(r'([-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*H', rest)
                if m:
                    comps[name] = float(m.group(1))
    return comps


def parse_orca_energy_components(filepath):
    """
    Parse energy components from ORCA old.orc_job.dat.
    Returns dict: {component_label -> energy_in_Hartree}.
    """
    comps = {}
    if not os.path.exists(filepath):
        return comps

    with open(filepath, 'r') as f:
        text = f.read()

    lines = text.splitlines()

    # Locate 'TOTAL SCF ENERGY' block
    start = None
    for i, line in enumerate(lines):
        if 'TOTAL SCF ENERGY' in line:
            start = i
            break
    if start is None:
        return comps

    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith('SCF CONVERGENCE'):
            break
        if not stripped:
            # Skip blank lines inside the block
            continue
        if ':' not in stripped:
            continue

        name, rest = stripped.split(':', 1)
        name = name.strip()

        # Lines we care about have an Eh value
        m = re.search(r'([-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*Eh', rest)
        if m:
            comps[name] = float(m.group(1))

    # Dispersion correction (e.g., "Dispersion correction           -0.029063720")
    m_disp = re.search(
        r'^\s*Dispersion correction\s+([-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$',
        text,
        re.MULTILINE,
    )
    if m_disp:
        comps['Dispersion correction'] = float(m_disp.group(1))

    # FINAL SINGLE POINT ENERGY (scalar, no units printed, but in Eh)
    m_final = re.search(
        r'FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)',
        text,
    )
    if m_final:
        comps['FINAL SINGLE POINT ENERGY'] = float(m_final.group(1))

    return comps


def get_species_energy_components(species_key, method):
    """
    Get energy components for one species and one method.
    method: 'DFTB' or 'PBE'.
    Returns dict {component_label -> energy_in_Hartree}.
    """
    # Decode where this species lives
    cfg = SPECIES_DIR_MAP.get(species_key)
    if cfg is None:
        return {}
    where, local_name = cfg

    if method == 'DFTB':
        filename = 'detailed.out'
        parser = parse_dftb_energy_components
        if where == 'deprot':
            base = os.path.join(SCRIPT_DIR, PHASE_DFTB, 'run8')
        elif where == 'lig_mimh':
            base = os.path.join(SCRIPT_DIR, '..', 'lig_exchange', 'MIMH_PBE_STRUC', 'SP_opt', 'run8')
        elif where == 'lig_mim':
            base = os.path.join(SCRIPT_DIR, '..', 'lig_exchange', 'MIM_PBE_STRUC', 'SP_opt', 'run8')
        else:
            return {}
    else:
        filename = 'old.orc_job.dat'
        parser = parse_orca_energy_components
        if where == 'deprot':
            base = os.path.join(SCRIPT_DIR, PHASE_PBE, 'run16')
        elif where == 'lig_mimh':
            base = os.path.join(SCRIPT_DIR, '..', 'lig_exchange', 'MIMH_PBE_STRUC', 'SP_init', 'run16')
        elif where == 'lig_mim':
            base = os.path.join(SCRIPT_DIR, '..', 'lig_exchange', 'MIM_PBE_STRUC', 'SP_init', 'run16')
        else:
            return {}

    path = os.path.join(base, local_name, filename)
    # Special case: H reference for PBE is taken as zero in dE.csv; if file is
    # missing we simply treat all components as zero.
    if not os.path.exists(path):
        if method == 'PBE' and local_name == 'H_monomer':
            return {}
        return {}
    return parser(path)


def compute_reaction_component_deltas():
    """
    For each reaction in REACTION_COMPONENT_SPECS, compute Δ(energy component)
    for DFTB3 and PBE.

    Returns:
        dict[reaction_label][method] -> {component_label -> (delta_E_H, delta_E_kcal)}
    """
    results = {}
    for label, stoich in REACTION_COMPONENT_SPECS:
        method_results = {}
        for method in ('DFTB', 'PBE'):
            comp_sums = {}
            for species_key, coeff in stoich.items():
                if species_key not in SPECIES_DIR_MAP:
                    continue
                comps = get_species_energy_components(species_key, method)
                for comp_name, value_h in comps.items():
                    comp_sums[comp_name] = comp_sums.get(comp_name, 0.0) + coeff * value_h
            if comp_sums:
                method_results[method] = {
                    k: (v, v * HARTREE_TO_KCAL) for k, v in comp_sums.items()
                }
        if method_results:
            results[label] = method_results
    return results


# DFTB: components that sum to Total energy (for verification)
_DFTB_TOTAL_COMPONENTS = [
    'Total Electronic energy',
    'Repulsive energy',
    'Dispersion energy',
]


def _dftb_sum_check(comps, tol_h=1e-6):
    """Check Total energy = Total Electronic + Repulsive + Dispersion for a component dict (in Eh)."""
    total = comps.get('Total energy')
    if total is None:
        return None, None
    parts = [comps.get(c) for c in _DFTB_TOTAL_COMPONENTS]
    if any(p is None for p in parts):
        return total, None
    summed = sum(parts)
    return total, summed if abs(total - summed) <= tol_h else summed


def print_reaction_component_deltas(results=None):
    """Pretty-print reaction energy-component changes for DFTB3 and PBE."""
    if results is None:
        results = compute_reaction_component_deltas()
    if not results:
        return

    print("\n\nEnergy component changes for deprotonation reactions (ΔE in Hartree / kcal/mol)")
    print("  DFTB: Total energy = Total Electronic energy + Repulsive energy + Dispersion energy")
    print("  (Band energy / Band free energy are eigenvalue sums, not part of Total energy.)\n")
    for label, methods in results.items():
        print(f"\n--- {label} ---")
        for method in ('DFTB', 'PBE'):
            if method not in methods:
                continue
            print(f"  {method}:")
            comps = methods[method]
            # Sort components by descending |ΔE_kcal|
            items = sorted(comps.items(), key=lambda kv: abs(kv[1][1]), reverse=True)
            print("    {:35s} {:>15s} {:>15s}".format("Component", "ΔE (Eh)", "ΔE (kcal/mol)"))
            for comp_name, (dE_h, dE_kcal) in items:
                print(f"    {comp_name:35s} {dE_h:15.6f} {dE_kcal:15.3f}")
            # DFTB: verify that the three components sum to Total energy (for this reaction's delta)
            if method == 'DFTB':
                total_d, sum_d = _dftb_sum_check({k: v[0] for k, v in comps.items()})
                if total_d is not None and sum_d is not None:
                    print("    [Check: Total energy = Total Electronic + Repulsive + Dispersion: "
                          "Δ(Total) = {:.6f} Eh, sum(Δ) = {:.6f} Eh, diff = {:.2e} {}]".format(
                              total_d, sum_d, total_d - sum_d,
                              "OK" if abs(total_d - sum_d) < 1e-6 else "MISMATCH"))
                elif total_d is not None:
                    print("    [Check: missing component(s) for sum; cannot verify.]")


def check_total_energies_vs_table(out_df, results, tol_kcal=1e-3):
    """
    Explicitly check that total reaction energies from components match
    the table values without BSSE (within tolerance).
    """
    if results is None:
        return
    if out_df is None or out_df.empty:
        return

    print("\n\nCheck: Total reaction energies from components vs table (noBSSE)")
    print("  (tolerance = {:.3e} kcal/mol)".format(tol_kcal))

    for _, row in out_df.iterrows():
        label = row.get('Reaction')
        if not label or label not in results:
            continue
        methods = results[label]

        def fmt_name(s):
            return s if len(s) <= 60 else s[:57] + "..."

        # DFTB: use 'Total energy' component
        if 'DFTB' in methods and 'dE_mDFTB3D3' in row and pd.notna(row['dE_mDFTB3D3']):
            comps_dftb = methods['DFTB']
            if 'Total energy' in comps_dftb:
                dE_comp_kcal = comps_dftb['Total energy'][1]
                dE_tab_kcal = float(row['dE_mDFTB3D3'])
                diff = dE_comp_kcal - dE_tab_kcal
                status = "OK" if abs(diff) <= tol_kcal else "MISMATCH"
                print(
                    f"  DFTB  {fmt_name(label)}:\n"
                    f"    components = {dE_comp_kcal:10.3f}  table(noBSSE) = {dE_tab_kcal:10.3f}  "
                    f"diff = {diff:8.3f}  [{status}]"
                )

        # PBE: use FINAL SINGLE POINT ENERGY (if present) or 'Total Energy'
        if 'PBE' in methods and 'dE_PBE_noBSSE' in row and pd.notna(row['dE_PBE_noBSSE']):
            comps_pbe = methods['PBE']
            comp_name = None
            if 'FINAL SINGLE POINT ENERGY' in comps_pbe:
                comp_name = 'FINAL SINGLE POINT ENERGY'
            elif 'Total Energy' in comps_pbe:
                comp_name = 'Total Energy'
            if comp_name is not None:
                dE_comp_kcal = comps_pbe[comp_name][1]
                dE_tab_kcal = float(row['dE_PBE_noBSSE'])
                diff = dE_comp_kcal - dE_tab_kcal
                status = "OK" if abs(diff) <= tol_kcal else "MISMATCH"
                print(
                    f"  PBE   {fmt_name(label)}:\n"
                    f"    components ({comp_name}) = {dE_comp_kcal:10.3f}  table(noBSSE) = {dE_tab_kcal:10.3f}  "
                    f"diff = {diff:8.3f}  [{status}]"
                )


def main():
    # Read pre-computed cycle energies (generated by extract_cycle_energies.py)
    cycle_csv = os.path.join(SCRIPT_DIR, 'cycle_reactions.csv')
    if os.path.exists(cycle_csv):
        out_df = pd.read_csv(cycle_csv)
    else:
        print(f"WARNING: {cycle_csv} not found; component totals will not be checked "
              "against table values.")
        out_df = None

    # Print detailed energy-component changes (from detailed.out / old.orc_job.dat)
    comp_results = compute_reaction_component_deltas()
    print_reaction_component_deltas(comp_results)
    # Explicitly check that Total energy / Total Energy from components reproduces
    # the noBSSE dE values in the summary table.
    check_total_energies_vs_table(out_df, comp_results)

    return out_df


if __name__ == '__main__':
    main()
