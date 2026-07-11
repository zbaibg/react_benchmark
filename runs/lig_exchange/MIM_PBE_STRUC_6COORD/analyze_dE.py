#! /usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
import yaml
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import (  # type: ignore
    get_Etot_amber,
    get_complex_energies,
    parse_complex_composition,
    calc_ebind,
    calc_dE,
    reaction_label,
)

COMPLEXES = [
    '1Zn_0MIm_5MeOH',
    '1Zn_0MIm_6MeOH',
    '1Zn_1MIm_5MeOH',
    '1Zn_2MIm_3MeOH',
    '1Zn_5MIm_0MeOH',
]

REACTIONS = [
    ('1Zn_0MIm_6MeOH', '1Zn_1MIm_5MeOH'),
    ('1Zn_1MIm_5MeOH', '1Zn_2MIm_3MeOH'),
    ('1Zn_2MIm_3MeOH', '1Zn_5MIm_0MeOH'),
]

SP_DIRS = ['SP_init', 'SP_opt']
MIN_DIRS = ['minimize', 'qm_minimize']


def _rxn_col(comp_r, comp_p, compositions):
    lbl = reaction_label(compositions[comp_r], compositions[comp_p])
    return f'dE({comp_r}->{comp_p}) {lbl}'


def analyze_runs():
    compositions = {c: parse_complex_composition(c) for c in COMPLEXES}
    rxn_cols = [_rxn_col(r, p, compositions) for r, p in REACTIONS]

    results = []

    for base_dir in SP_DIRS + MIN_DIRS:
        if not os.path.exists(base_dir):
            continue

        is_sp = base_dir in SP_DIRS
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
            for comp in COMPLEXES:
                if is_sp:
                    e_raw, e_corr, bsse_found = get_complex_energies(run_path, comp)
                else:
                    e_raw = get_Etot_amber(os.path.join(run_path, comp, 'min.out'))
                    e_corr, bsse_found = None, False

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

            row_raw = {
                'Phase': base_dir,
                'Run': run_name,
                'Method': method_name,
                'BSSE': 'no',
                'E_Zn': e_zn,
                'E_MIm': e_mim,
                'E_MeOH': e_meoh,
            }
            for comp in COMPLEXES:
                row_raw[f'E_{comp}'] = raw_energies[comp]
                eb = calc_ebind(raw_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                row_raw[f'Ebind_{comp}'] = eb if eb is not None else np.nan
            for col, (r, p) in zip(rxn_cols, REACTIONS):
                row_raw[col] = calc_dE(raw_energies[r], raw_energies[p],
                                       compositions[r], compositions[p], e_mim, e_meoh)
            results.append(row_raw)

            if has_bsse:
                all_bsse_ok = all(bsse_energies[c] is not None for c in COMPLEXES)
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
                    for comp in COMPLEXES:
                        row_bsse[f'E_{comp}'] = bsse_energies[comp]
                        eb = calc_ebind(bsse_energies[comp], compositions[comp], e_zn, e_mim, e_meoh)
                        row_bsse[f'Ebind_{comp}'] = eb if eb is not None else np.nan
                    for col, (r, p) in zip(rxn_cols, REACTIONS):
                        row_bsse[col] = calc_dE(bsse_energies[r], bsse_energies[p],
                                                compositions[r], compositions[p], e_mim, e_meoh)
                    results.append(row_bsse)
                else:
                    print(f"Warning: {base_dir}/{run_name} has partial BSSE data, skipping BSSE row")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = analyze_runs()

    if df.empty:
        print("No data found.")
        sys.exit(0)

    cols = ['Phase', 'Run', 'Method', 'BSSE']
    ebind_cols = [c for c in df.columns if c.startswith('Ebind_')]
    de_cols = [c for c in df.columns if c.startswith('dE(')]
    cols.extend(ebind_cols)
    cols.extend(de_cols)
    other_cols = [c for c in df.columns if c not in cols]
    cols.extend(other_cols)

    df = df[cols]
    if de_cols:
        df.sort_values(by=[de_cols[0]], inplace=True, ignore_index=True)

    output_file = 'dE.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

    display_cols = ['Phase', 'Run', 'Method', 'BSSE'] + ebind_cols + de_cols
    print("\nBinding Energies & Reaction Energies (kcal/mol):")
    print(df[display_cols].to_string())
