#! /usr/bin/env python3
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
    parse_complex_composition,
    calc_ebind,
)

def analyze_runs(base_dirs=['SP_init', 'SP_opt']):
    complexes = [
        '1Zn_0MIm_4MeOH',
        '1Zn_1MIm_3MeOH',
        '1Zn_2MIm_2MeOH',
        '1Zn_3MIm_1MeOH',
        '1Zn_4MIm_0MeOH'
    ]
    compositions = {c: parse_complex_composition(c) for c in complexes}
    
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
            
            e_zn = get_Etot_amber(os.path.join(run_path, 'Zn_monomer', 'min.out'))
            e_mim = get_Etot_amber(os.path.join(run_path, 'MIm_monomer', 'min.out'))
            e_meoh = get_Etot_amber(os.path.join(run_path, 'MeOH_monomer', 'min.out'))
            
            if e_mim is None or e_meoh is None:
                missing = []
                if e_mim is None:
                    missing.append('MIm_monomer')
                if e_meoh is None:
                    missing.append('MeOH_monomer')
                print(f"Skipping {base_dir}/{run_name}: Missing energies for {', '.join(missing)}")
                continue
            if e_zn is None:
                print(f"Warning: {base_dir}/{run_name}: Missing Zn_monomer energy, Ebind will be NaN")
                
            raw_energies = {}
            bsse_energies = {}
            has_bsse = False
            
            skip_run = False
            for comp in complexes:
                e_raw, e_corr, bsse_found = get_complex_energies(run_path, comp)
                if e_raw is None:
                    print(f"Skipping {base_dir}/{run_name}: Missing energy for {comp}")
                    skip_run = True
                    break
                raw_energies[comp] = e_raw
                bsse_energies[comp] = e_corr
                if bsse_found:
                    has_bsse = True
            
            if skip_run:
                continue
            
            # Row without BSSE (always present)
            row_raw = {
                'Phase': base_dir,
                'Run': run_name,
                'Method': method_name,
                'BSSE': 'no',
                'E_Zn': e_zn,
                'E_MIm': e_mim,
                'E_MeOH': e_meoh,
            }
            for comp in complexes:
                row_raw[f'E_{comp}'] = raw_energies[comp]
                eb = calc_ebind(raw_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                row_raw[f'Ebind_{comp}'] = eb if eb is not None else np.nan
            for i in range(len(complexes) - 1):
                r, p = complexes[i], complexes[i+1]
                dE = (raw_energies[p] + e_meoh) - (raw_energies[r] + e_mim)
                row_raw[f'dE_{i}_{i+1}'] = dE
            results.append(row_raw)
            
            # Row with BSSE (only for QM runs that have ghost data)
            if has_bsse:
                all_bsse_ok = all(bsse_energies[c] is not None for c in complexes)
                if all_bsse_ok:
                    row_bsse = {
                        'Phase': base_dir,
                        'Run': run_name,
                        'Method': method_name,
                        'BSSE': 'yes',
                        'E_Zn': e_zn,
                        'E_MIm': e_mim,
                        'E_MeOH': e_meoh,
                    }
                    for comp in complexes:
                        row_bsse[f'E_{comp}'] = bsse_energies[comp]
                        eb = calc_ebind(bsse_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                        row_bsse[f'Ebind_{comp}'] = eb if eb is not None else np.nan
                    for i in range(len(complexes) - 1):
                        r, p = complexes[i], complexes[i+1]
                        dE = (bsse_energies[p] + e_meoh) - (bsse_energies[r] + e_mim)
                        row_bsse[f'dE_{i}_{i+1}'] = dE
                    results.append(row_bsse)
                else:
                    print(f"Warning: {base_dir}/{run_name} has partial BSSE data, skipping BSSE row")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = analyze_runs()
    
    cols = ['Phase', 'Run', 'Method', 'BSSE']
    ebind_cols = [c for c in df.columns if c.startswith('Ebind_')]
    de_cols = [c for c in df.columns if c.startswith('dE_')]
    cols.extend(ebind_cols)
    cols.extend(de_cols)
    other_cols = [c for c in df.columns if c not in cols]
    cols.extend(other_cols)
    
    df = df[cols]
    df.sort_values(by=[ 'dE_0_1'], inplace=True, ignore_index=True)
    
    output_file = 'dE.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")
    
    display_cols = ['Phase', 'Run', 'Method', 'BSSE'] + ebind_cols + de_cols
    print("\nBinding Energies & Reaction Energies (kcal/mol):")
    print(df[display_cols].to_string())
