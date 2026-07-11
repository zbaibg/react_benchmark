#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
import yaml
import sys

# Make project root (where analib.py lives) importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import (  # type: ignore
    get_Etot_amber,
    get_complex_energies,
)

# Complex names used for deprotonation analysis
COMPLEX_MIMH = '1Zn_1MImH_3MeOH'
COMPLEX_MIM = '1Zn_1MIm_3MeOH'

def analyze_runs(base_dirs=['SP_init', 'SP_opt']):
    results = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        run_dirs = sorted(glob.glob(os.path.join(base_dir, 'run*')))
        
        for run_path in run_dirs:
            run_name = os.path.basename(run_path)
            
            method_name = run_name
            notes_path = os.path.join(run_path, 'notes.yaml')
            if os.path.exists(notes_path):
                with open(notes_path, 'r') as f:
                    try:
                        notes = yaml.safe_load(f)
                        method_name = notes.get('name', run_name)
                    except Exception:
                        pass

            e_mimh = get_Etot_amber(os.path.join(run_path, 'MImH_monomer', 'min.out'))
            e_mim = get_Etot_amber(os.path.join(run_path, 'MIm_monomer', 'min.out'))

            # Complexes: 1Zn_1MImH_3MeOH and 1Zn_1MIm_3MeOH (with optional BSSE)
            e_mimh_full, e_mimh_bsse, has_bsse_mimh = get_complex_energies(run_path, COMPLEX_MIMH)
            e_mim_full, e_mim_bsse, has_bsse_mim = get_complex_energies(run_path, COMPLEX_MIM)
            has_bsse = has_bsse_mimh and has_bsse_mim

            # H_monomer for MImH = MIm + H; default 0 if parse fails
            e_h_path = os.path.join(run_path, 'H_monomer', 'min.out')
            e_h = get_Etot_amber(e_h_path)
            if e_h is None:
                e_h = 0.0
                print(f"  WARNING: H_monomer energy parse failed in {run_path}, using E_H = 0")

            # Wat_monomer (H2O) and H3O_monomer for MImH + H2O = MIm + H3O
            e_h2o = get_Etot_amber(os.path.join(run_path, 'Wat_monomer', 'min.out'))
            e_h3o = get_Etot_amber(os.path.join(run_path, 'H3O_monomer', 'min.out'))

            def build_dE_row(e_zn_mimh_meoh3, e_zn_mim_meoh3):
                """Build dE values from given complex energies (raw or BSSE-corrected)."""
                if e_mimh is not None and e_mim is not None:
                    dE_deproton = e_mim + e_h - e_mimh  # MImH -> MIm + H
                else:
                    dE_deproton = np.nan
                if e_zn_mimh_meoh3 is not None and e_zn_mim_meoh3 is not None:
                    dE_complex_deproton = e_zn_mim_meoh3 + e_h - e_zn_mimh_meoh3
                else:
                    dE_complex_deproton = np.nan
                if all(x is not None for x in (e_mimh, e_mim, e_h2o, e_h3o)):
                    dE_h2o_transfer = e_mim + e_h3o - e_mimh - e_h2o
                else:
                    dE_h2o_transfer = np.nan
                if all(x is not None for x in (e_zn_mimh_meoh3, e_zn_mim_meoh3, e_h2o, e_h3o)):
                    dE_complex_h2o_transfer = e_zn_mim_meoh3 + e_h3o - e_zn_mimh_meoh3 - e_h2o
                else:
                    dE_complex_h2o_transfer = np.nan
                # H2O + H -> H3O+
                if all(x is not None for x in (e_h2o, e_h, e_h3o)):
                    dE_H2O_H_H3O = e_h3o - e_h2o - e_h
                else:
                    dE_H2O_H_H3O = np.nan
                return dE_deproton, dE_complex_deproton, dE_h2o_transfer, dE_complex_h2o_transfer, dE_H2O_H_H3O

            # Row with raw complex energies (BSSE = no)
            dE_deproton, dE_complex_deproton, dE_h2o_transfer, dE_complex_h2o_transfer, dE_H2O_H_H3O = build_dE_row(
                e_mimh_full, e_mim_full
            )
            row_raw = {
                'Phase': base_dir,
                'Run': run_name,
                'Method': method_name,
                'BSSE': 'no',
                'E_MImH': e_mimh,
                'E_MIm': e_mim,
                'E_H': e_h,
                'E_H2O': e_h2o,
                'E_H3O': e_h3o,
                'E_1Zn_1MImH_3MeOH': e_mimh_full,
                'E_1Zn_1MIm_3MeOH': e_mim_full,
                'dE_MImH_MIm_H': dE_deproton,
                'dE_1Zn_1MImH_3MeOH_1Zn_1MIm_3MeOH_H': dE_complex_deproton,
                'dE_MImH_H2O_MIm_H3O': dE_h2o_transfer,
                'dE_1Zn_1MImH_3MeOH_H2O_1Zn_1MIm_3MeOH_H3O': dE_complex_h2o_transfer,
                'dE_H2O_H_H3O': dE_H2O_H_H3O,
            }
            results.append(row_raw)

            # Row with BSSE-corrected complex energies when both complexes have BSSE data
            if has_bsse and e_mimh_bsse is not None and e_mim_bsse is not None:
                dE_deproton_b, dE_complex_deproton_b, dE_h2o_transfer_b, dE_complex_h2o_transfer_b, dE_H2O_H_H3O_b = build_dE_row(
                    e_mimh_bsse, e_mim_bsse
                )
                row_bsse = {
                    'Phase': base_dir,
                    'Run': run_name,
                    'Method': method_name,
                    'BSSE': 'yes',
                    'E_MImH': e_mimh,
                    'E_MIm': e_mim,
                    'E_H': e_h,
                    'E_H2O': e_h2o,
                    'E_H3O': e_h3o,
                    'E_1Zn_1MImH_3MeOH': e_mimh_bsse,
                    'E_1Zn_1MIm_3MeOH': e_mim_bsse,
                    'dE_MImH_MIm_H': dE_deproton_b,
                    'dE_1Zn_1MImH_3MeOH_1Zn_1MIm_3MeOH_H': dE_complex_deproton_b,
                    'dE_MImH_H2O_MIm_H3O': dE_h2o_transfer_b,
                    'dE_1Zn_1MImH_3MeOH_H2O_1Zn_1MIm_3MeOH_H3O': dE_complex_h2o_transfer_b,
                    'dE_H2O_H_H3O': dE_H2O_H_H3O_b,
                }
                results.append(row_bsse)
            elif has_bsse_mimh or has_bsse_mim:
                print(f"  {base_dir}/{run_name}: partial BSSE data (only one complex), skipping BSSE row")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = analyze_runs()

    if df.empty:
        print("No data found.")
    else:
        cols = [
            'Phase',
            'Run',
            'Method',
            'BSSE',
            'E_MImH',
            'E_MIm',
            'E_H',
            'E_H2O',
            'E_H3O',
            'E_1Zn_1MImH_3MeOH',
            'E_1Zn_1MIm_3MeOH',
            'dE_MImH_MIm_H',
            'dE_1Zn_1MImH_3MeOH_1Zn_1MIm_3MeOH_H',
            'dE_MImH_H2O_MIm_H3O',
            'dE_1Zn_1MImH_3MeOH_H2O_1Zn_1MIm_3MeOH_H3O',
            'dE_H2O_H_H3O',
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        if 'dE_MImH_MIm_H' in df.columns:
            df.sort_values(by=['dE_MImH_MIm_H'], inplace=True, ignore_index=True)

        output_file = 'dE.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        print("\nReaction column legend:")
        print("  dE_MImH_MIm_H: MImH = MIm + H  (monomer deprotonation)")
        print("  dE_1Zn_1MImH_3MeOH_1Zn_1MIm_3MeOH_H: 1Zn_1MImH_3MeOH = 1Zn_1MIm_3MeOH + H")
        print("  dE_MImH_H2O_MIm_H3O: MImH + H2O = MIm + H3O")
        print("  dE_1Zn_1MImH_3MeOH_H2O_1Zn_1MIm_3MeOH_H3O: [Zn·MImH·MeOH3]²⁺ + H2O = [Zn·MIm·MeOH3]⁺ + H3O⁺")
        print("  dE_H2O_H_H3O: H2O + H = H3O⁺")

        display_cols = cols
        print("\nDeprotonation energies (kcal/mol):")
        print(df[display_cols].to_string())
