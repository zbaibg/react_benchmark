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
    XYZ_DIR,
    LIGAND_NAME,
    detect_ligand_name,
    get_Etot_amber,
    get_complex_energies,
    discover_complexes,
    parse_complex_composition,
    calc_ebind,
    calc_dE,
    reaction_label,
)

def analyze_runs(base_dirs=['SP_init', 'SP_opt']):
    complexes = discover_complexes()
    compositions = {c: parse_complex_composition(c) for c in complexes}

    print(f"Discovered {len(complexes)} complexes (sorted by {LIGAND_NAME} count):")
    for c in complexes:
        comp = compositions[c]
        cn = comp['Zn'] + comp['MIm'] + comp['MeOH']
        print(f"  {c}  (CN={cn - comp['Zn']}, Zn={comp['Zn']} {LIGAND_NAME}={comp['MIm']} MeOH={comp['MeOH']})")

    print("\nExchange reactions:")
    for i in range(len(complexes) - 1):
        r, p = complexes[i], complexes[i+1]
        lbl = reaction_label(compositions[r], compositions[p])
        print(f"  {r} -> {p}  ({lbl})")
    print()

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
            mim_monomer_name = f'{LIGAND_NAME}_monomer'
            e_mim = get_Etot_amber(os.path.join(run_path, mim_monomer_name, 'min.out'))
            e_meoh = get_Etot_amber(os.path.join(run_path, 'MeOH_monomer', 'min.out'))
            
            if e_mim is None or e_meoh is None:
                missing = []
                if e_mim is None:
                    missing.append(mim_monomer_name)
                if e_meoh is None:
                    missing.append('MeOH_monomer')
                print(f"Skipping {base_dir}/{run_name}: Missing energies for {', '.join(missing)}")
                continue
                
            raw_energies = {}
            bsse_energies = {}
            has_bsse = False
            missing_complexes = []
            
            for comp in complexes:
                e_raw, e_corr, bsse_found = get_complex_energies(run_path, comp)
                raw_energies[comp] = e_raw
                bsse_energies[comp] = e_corr
                if bsse_found:
                    has_bsse = True
                if e_raw is None:
                    missing_complexes.append(comp)
            
            if len(missing_complexes) == len(complexes):
                print(f"Skipping {base_dir}/{run_name}: No complex energies found")
                continue
            if missing_complexes:
                print(f"  {base_dir}/{run_name}: Missing energies for {', '.join(missing_complexes)}")

            row_raw = {
                'Phase': base_dir,
                'Run': run_name,
                'Method': method_name,
                'BSSE': 'no',
                'E_Zn': e_zn,
                f'E_{LIGAND_NAME}': e_mim,
                'E_MeOH': e_meoh,
            }
            for comp in complexes:
                row_raw[f'E_{comp}'] = raw_energies[comp]
                eb = calc_ebind(raw_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                row_raw[f'Ebind_{comp}'] = eb if eb is not None else np.nan
            for i in range(len(complexes) - 1):
                r, p = complexes[i], complexes[i+1]
                dE = calc_dE(raw_energies[r], raw_energies[p],
                             compositions[r], compositions[p], e_mim, e_meoh)
                row_raw[f'dE_{i}_{i+1}'] = dE
            results.append(row_raw)
            
            if has_bsse:
                all_bsse_ok = all(bsse_energies[c] is not None for c in complexes)
                if all_bsse_ok:
                    row_bsse = {
                        'Phase': base_dir,
                        'Run': run_name,
                        'Method': method_name,
                        'BSSE': 'yes',
                        'E_Zn': e_zn,
                        f'E_{LIGAND_NAME}': e_mim,
                        'E_MeOH': e_meoh,
                    }
                    for comp in complexes:
                        row_bsse[f'E_{comp}'] = bsse_energies[comp]
                        eb = calc_ebind(bsse_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                        row_bsse[f'Ebind_{comp}'] = eb if eb is not None else np.nan
                    for i in range(len(complexes) - 1):
                        r, p = complexes[i], complexes[i+1]
                        dE = calc_dE(bsse_energies[r], bsse_energies[p],
                                     compositions[r], compositions[p], e_mim, e_meoh)
                        row_bsse[f'dE_{i}_{i+1}'] = dE
                    results.append(row_bsse)
                else:
                    print(f"  {base_dir}/{run_name}: partial BSSE data, skipping BSSE row")
            
    return pd.DataFrame(results), complexes

if __name__ == "__main__":
    df, complexes = analyze_runs()
    compositions = {c: parse_complex_composition(c) for c in complexes}
    
    cols = ['Phase', 'Run', 'Method', 'BSSE']
    ebind_cols = [c for c in df.columns if c.startswith('Ebind_')]
    de_cols = [c for c in df.columns if c.startswith('dE_')]
    cols.extend(ebind_cols)
    cols.extend(de_cols)
    other_cols = [c for c in df.columns if c not in cols]
    cols.extend(other_cols)
    
    df = df[cols]
    first_de = de_cols[0] if de_cols else None
    if first_de and first_de in df.columns:
        df.sort_values(by=[ first_de], inplace=True, ignore_index=True)
    
    output_file = 'dE.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    print("\nReaction column legend:")
    for i in range(len(complexes) - 1):
        r, p = complexes[i], complexes[i+1]
        lbl = reaction_label(compositions[r], compositions[p])
        print(f"  dE_{i}_{i+1}: {r} -> {p}  ({lbl})")
    
    display_cols = ['Phase', 'Run', 'Method', 'BSSE'] + ebind_cols + de_cols
    display_cols = [c for c in display_cols if c in df.columns]
    print("\nBinding Energies & Reaction Energies (kcal/mol):")
    print(df[display_cols].to_string())
