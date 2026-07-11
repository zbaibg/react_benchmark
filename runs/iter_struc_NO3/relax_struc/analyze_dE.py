#!/usr/bin/env python3
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Make repo root and python_scripts/ importable.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))
PYTHON_SCRIPTS_DIR = os.path.join(ROOT_DIR, "python_scripts")
for _path in (PYTHON_SCRIPTS_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analib import get_Etot_amber, get_complex_energies  # type: ignore

FULL_COMPLEX_DIR_RE = re.compile(r"^(1Zn_\d+NO3_\d+MeOH)_full$")
BARE_COMPLEX_DIR_RE = re.compile(r"^(1Zn_\d+NO3_\d+MeOH)$")
FORMULA_RE = re.compile(r"^1Zn_(\d+)NO3_(\d+)MeOH$")
RUN_RE = re.compile(r"^run(\d+)$")

REACTION_TYPE_ORDER = {
    "Add MeOH": 0,
    "Add NO3(-)": 1,
    "Exchange (-MeOH, +NO3(-))": 2,
}

DEFAULT_BASE_DIR = "SP_init"
RUN_SOURCE_OVERRIDES = {
    "run41": "qm_minimize",
}


def _run_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    match = RUN_RE.match(name)
    if match is None:
        return (10**9, name)
    return (int(match.group(1)), name)


def _load_method_name(run_path: str) -> str:
    run_name = os.path.basename(run_path)
    notes_path = os.path.join(run_path, "notes.yaml")
    if not os.path.exists(notes_path):
        return run_name
    try:
        with open(notes_path, "r") as f:
            notes = yaml.safe_load(f) or {}
    except Exception:
        return run_name
    return str(notes.get("name", run_name))


def _parse_formula_counts(formula: str) -> Optional[Tuple[int, int]]:
    match = FORMULA_RE.fullmatch(formula)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _complex_sort_key(formula: str) -> Tuple[int, int, str]:
    counts = _parse_formula_counts(formula)
    if counts is None:
        return (10**9, 10**9, formula)
    n_no3, n_meoh = counts
    return (n_no3, n_meoh, formula)


def _iter_complex_names(run_path: str, source_base_dir: str) -> List[str]:
    complex_names: List[str] = []
    for entry in os.listdir(run_path):
        full_path = os.path.join(run_path, entry)
        if not os.path.isdir(full_path):
            continue

        if source_base_dir == "qm_minimize":
            if BARE_COMPLEX_DIR_RE.fullmatch(entry):
                complex_names.append(entry)
            continue

        match = FULL_COMPLEX_DIR_RE.fullmatch(entry)
        if match is not None:
            complex_names.append(match.group(1))

    return sorted(set(complex_names), key=_complex_sort_key)


def _reaction_type_from_counts(
    reactant_counts: Tuple[int, int],
    product_counts: Tuple[int, int],
) -> Optional[str]:
    d_no3 = product_counts[0] - reactant_counts[0]
    d_meoh = product_counts[1] - reactant_counts[1]

    if (d_no3, d_meoh) == (0, 1):
        return "Add MeOH"
    if (d_no3, d_meoh) == (1, 0):
        return "Add NO3(-)"
    if (d_no3, d_meoh) == (1, -1):
        return "Exchange (-MeOH, +NO3(-))"
    return None


def _balanced_reaction_plain(edge_type: str, from_formula: str, to_formula: str) -> str:
    if edge_type == "Add MeOH":
        return f"{from_formula} + MeOH -> {to_formula}"
    if edge_type == "Add NO3(-)":
        return f"{from_formula} + NO3- -> {to_formula}"
    if edge_type == "Exchange (-MeOH, +NO3(-))":
        return f"{from_formula} + NO3- -> {to_formula} + MeOH"
    return f"{from_formula} -> {to_formula}"


def _compute_de_rxn_kcal(
    edge_type: str,
    e_from: float,
    e_to: float,
    monomer_refs: Dict[str, float],
) -> Optional[float]:
    d_cluster = e_to - e_from

    if edge_type == "Add NO3(-)":
        e_no3 = monomer_refs.get("NO3")
        return None if e_no3 is None else d_cluster - e_no3
    if edge_type == "Add MeOH":
        e_meoh = monomer_refs.get("MeOH")
        return None if e_meoh is None else d_cluster - e_meoh
    if edge_type == "Exchange (-MeOH, +NO3(-))":
        e_no3 = monomer_refs.get("NO3")
        e_meoh = monomer_refs.get("MeOH")
        if e_no3 is None or e_meoh is None:
            return None
        return d_cluster - e_no3 + e_meoh
    return None


def _collect_monomer_refs(run_path: str) -> Dict[str, float]:
    monomer_dirs = {
        "NO3": "NO3_monomer",
        "MeOH": "MeOH_monomer",
        "Zn": "Zn_monomer",
    }
    refs: Dict[str, float] = {}
    for label, subdir in monomer_dirs.items():
        energy = get_Etot_amber(os.path.join(run_path, subdir, "min.out"))
        if energy is not None:
            refs[label] = energy
    return refs


def _collect_complex_energies(
    run_path: str,
    source_base_dir: str,
    use_bsse: bool,
) -> Dict[str, float]:
    energies: Dict[str, float] = {}
    for complex_name in _iter_complex_names(run_path, source_base_dir):
        if source_base_dir == "qm_minimize":
            if use_bsse:
                continue
            e_raw = get_Etot_amber(os.path.join(run_path, complex_name, "min.out"))
            if e_raw is not None:
                energies[complex_name] = e_raw
            continue

        e_raw, e_bsse, has_bsse = get_complex_energies(run_path, complex_name)
        if use_bsse:
            if has_bsse and e_bsse is not None:
                energies[complex_name] = e_bsse
        else:
            if e_raw is not None:
                energies[complex_name] = e_raw
    return energies


def _collect_reaction_rows_for_run(
    run_path: str,
    source_base_dir: str,
    use_bsse: bool,
) -> Dict[Tuple[str, str, str, str], float]:
    monomer_refs = _collect_monomer_refs(run_path)
    complex_energies = _collect_complex_energies(
        run_path,
        source_base_dir=source_base_dir,
        use_bsse=use_bsse,
    )
    if use_bsse and not complex_energies:
        return {}

    formulas = sorted(complex_energies.keys(), key=_complex_sort_key)
    rows: Dict[Tuple[str, str, str, str], float] = {}

    for from_formula in formulas:
        from_counts = _parse_formula_counts(from_formula)
        if from_counts is None:
            continue
        e_from = complex_energies[from_formula]

        for to_formula in formulas:
            if to_formula == from_formula:
                continue

            to_counts = _parse_formula_counts(to_formula)
            if to_counts is None:
                continue

            edge_type = _reaction_type_from_counts(from_counts, to_counts)
            if edge_type is None:
                continue

            e_to = complex_energies[to_formula]
            dE_rxn = _compute_de_rxn_kcal(edge_type, e_from, e_to, monomer_refs)
            if dE_rxn is None:
                continue

            balanced = _balanced_reaction_plain(edge_type, from_formula, to_formula)
            rows[(edge_type, from_formula, to_formula, balanced)] = dE_rxn

    return rows


def _collect_run_sources(default_base_dir: str) -> List[Tuple[str, str, str]]:
    run_sources: Dict[str, Tuple[str, str]] = {}

    if os.path.exists(default_base_dir):
        for run_path in sorted(glob.glob(os.path.join(default_base_dir, "run*")), key=_run_sort_key):
            if not os.path.isdir(run_path):
                continue
            run_name = os.path.basename(run_path)
            run_sources[run_name] = (default_base_dir, run_path)

    for run_name, source_base_dir in RUN_SOURCE_OVERRIDES.items():
        run_path = os.path.join(source_base_dir, run_name)
        if os.path.isdir(run_path):
            run_sources[run_name] = (source_base_dir, run_path)

    return [
        (run_name, source_base_dir, run_path)
        for run_name, (source_base_dir, run_path) in sorted(
            run_sources.items(),
            key=lambda item: _run_sort_key(item[0]),
        )
    ]


def analyze_runs(base_dir: str = DEFAULT_BASE_DIR) -> pd.DataFrame:
    run_sources = _collect_run_sources(base_dir)
    if not run_sources:
        return pd.DataFrame()

    all_rows: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    column_order: List[str] = []
    method_map: Dict[str, str] = {}
    source_map: Dict[str, str] = {}

    for run_name, source_base_dir, run_path in run_sources:
        method_map[run_name] = _load_method_name(run_path)
        source_map[run_name] = source_base_dir

        for use_bsse in (False, True):
            row_values = _collect_reaction_rows_for_run(
                run_path,
                source_base_dir=source_base_dir,
                use_bsse=use_bsse,
            )
            if not row_values:
                continue

            column_name = f"{run_name}|BSSE={'yes' if use_bsse else 'no'}"
            column_order.append(column_name)

            for row_key, dE_value in row_values.items():
                all_rows.setdefault(row_key, {})
                all_rows[row_key][column_name] = dE_value

    if not all_rows:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for (reaction_type, from_formula, to_formula, balanced_reaction), values in all_rows.items():
        record: Dict[str, object] = {
            "reaction_type": reaction_type,
            "from_formula": from_formula,
            "to_formula": to_formula,
            "balanced_reaction": balanced_reaction,
        }
        for column_name in column_order:
            record[column_name] = values.get(column_name, np.nan)
        records.append(record)

    def _record_sort_key(record: Dict[str, object]) -> Tuple[int, str, str, str]:
        reaction_type = str(record["reaction_type"])
        return (
            REACTION_TYPE_ORDER.get(reaction_type, 10**9),
            str(record["from_formula"]),
            str(record["to_formula"]),
            str(record["balanced_reaction"]),
        )

    df = pd.DataFrame(sorted(records, key=_record_sort_key))
    df.attrs["method_map"] = method_map
    df.attrs["source_map"] = source_map
    return df


if __name__ == "__main__":
    df = analyze_runs(base_dir=DEFAULT_BASE_DIR)

    if df.empty:
        print("No reaction energies found under configured run sources.")
    else:
        output_file = "dE.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        method_map = df.attrs.get("method_map", {})
        source_map = df.attrs.get("source_map", {})
        if method_map:
            print("\nRun legend:")
            for run_name in sorted(method_map.keys(), key=lambda x: _run_sort_key(x)):
                source_desc = source_map.get(run_name, DEFAULT_BASE_DIR)
                print(f"  {run_name} [{source_desc}]: {method_map[run_name]}")

        value_cols = [
            col for col in df.columns
            if col not in ("reaction_type", "from_formula", "to_formula", "balanced_reaction")
        ]
        print("\nReaction energies (kcal/mol):")
        print(df[["reaction_type", "balanced_reaction"] + value_cols].to_string(index=False))
