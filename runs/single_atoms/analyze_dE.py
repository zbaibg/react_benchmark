#!/usr/bin/env python3
import pandas as pd
import os
import glob
import sys

# Make project root (where analib.py lives) importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import get_Etot_amber  # type: ignore

def analyze_single_atoms(base_dirs=['SP_init', 'SP_opt']):
    results = []

    atom_names = ['C', 'H', 'N', 'O', 'Zn']
    monomer_dirs = [f'{atom}_monomer' for atom in atom_names]

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        run_dirs = sorted(glob.glob(os.path.join(base_dir, 'run*')))
        for run_path in run_dirs:
            run_name = os.path.basename(run_path)
            row = {'Phase': base_dir, 'Run': run_name}
            
            found_any = False
            for atom, monomer_dir in zip(atom_names, monomer_dirs):
                min_out = os.path.join(run_path, monomer_dir, 'min.out')
                energy = get_Etot_amber(min_out)
                row[f'E_{atom}'] = energy
                if energy is not None:
                    found_any = True

            if found_any:
                results.append(row)

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = analyze_single_atoms()

    if df.empty:
        print("No single atom monomer energy data found (C, H, N, O, Zn).")
    else:
        cols = ['Phase', 'Run'] + [f'E_{atom}' for atom in ['C', 'H', 'N', 'O', 'Zn']]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        output_file = 'single_atom_energies.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}\n")

        print("Energies of single atom monomers (in kcal/mol where appropriate):")
        print(df.to_string(index=False))
