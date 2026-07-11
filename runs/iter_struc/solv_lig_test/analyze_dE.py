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

FORMULA_RE = re.compile(r"^1Zn_(\d+)(ImH|Im-)_(\d+)Wat$")
RUN_RE = re.compile(r"^run(\d+)$")
MONOMER_H_REACTION_TYPE = "monomer Protonation (+H)"
MONOMER_H_REACTIONS = (
    ("Im-_monomer", "ImH_monomer", "Im-", "ImH"),
)
REACTION_TYPE_ORDER = {
    "Add Wat": 0,
    "Add ImH": 1,
    "Add Im(-)": 2,
    "Deprotonation (-H+)": 3,
    "Exchange (-Wat, +Im(-))": 4,
    "Exchange (-Wat, +ImH)": 5,
    MONOMER_H_REACTION_TYPE: 6,
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


def _parse_formula_counts(formula: str) -> Optional[Tuple[int, int, int]]:
    match = FORMULA_RE.fullmatch(formula)
    if match is None:
        return None
    n_ligand = int(match.group(1))
    ligand = match.group(2)
    n_wat = int(match.group(3))
    n_im_minus = n_ligand if ligand == "Im-" else 0
    n_imh = n_ligand if ligand == "ImH" else 0
    return (n_im_minus, n_imh, n_wat)


def _complex_sort_key(formula: str) -> Tuple[int, int, int, str]:
    counts = _parse_formula_counts(formula)
    if counts is None:
        return (10**9, 10**9, 10**9, formula)
    n_im_minus, n_imh, n_wat = counts
    return (n_im_minus + n_imh, n_wat, n_im_minus, formula)


def _iter_complex_names(run_path: str, source_base_dir: str) -> List[str]:
    complex_names: List[str] = []
    for entry in os.listdir(run_path):
        entry_path = os.path.join(run_path, entry)
        if not os.path.isdir(entry_path):
            continue

        if source_base_dir == "qm_minimize":
            if _parse_formula_counts(entry) is not None:
                complex_names.append(entry)
            continue

        if entry.endswith("_full"):
            complex_name = entry[:-len("_full")]
            if _parse_formula_counts(complex_name) is not None:
                complex_names.append(complex_name)
    return sorted(set(complex_names), key=_complex_sort_key)


def _reaction_type_from_counts(
    reactant_counts: Tuple[int, int, int],
    product_counts: Tuple[int, int, int],
) -> Optional[str]:
    d_im_minus = product_counts[0] - reactant_counts[0]
    d_imh = product_counts[1] - reactant_counts[1]
    d_wat = product_counts[2] - reactant_counts[2]

    if (d_im_minus, d_imh, d_wat) == (0, 0, 1):
        return "Add Wat"
    if (d_im_minus, d_imh, d_wat) == (0, 1, 0):
        return "Add ImH"
    if (d_im_minus, d_imh, d_wat) == (1, 0, 0):
        return "Add Im(-)"
    if (d_im_minus, d_imh, d_wat) == (1, -1, 0):
        return "Deprotonation (-H+)"
    if (d_im_minus, d_imh, d_wat) == (1, 0, -1):
        return "Exchange (-Wat, +Im(-))"
    if (d_im_minus, d_imh, d_wat) == (0, 1, -1):
        return "Exchange (-Wat, +ImH)"
    return None


def _balanced_reaction_plain(edge_type: str, from_formula: str, to_formula: str) -> str:
    if edge_type == MONOMER_H_REACTION_TYPE:
        return f"{from_formula} + H -> {to_formula}"
    if edge_type == "Add Wat":
        return f"{from_formula} + Wat -> {to_formula}"
    if edge_type == "Add ImH":
        return f"{from_formula} + ImH -> {to_formula}"
    if edge_type == "Add Im(-)":
        return f"{from_formula} + Im- -> {to_formula}"
    if edge_type == "Deprotonation (-H+)":
        return f"{from_formula} -> {to_formula} + H+"
    if edge_type == "Exchange (-Wat, +Im(-))":
        return f"{from_formula} + Im- -> {to_formula} + Wat"
    if edge_type == "Exchange (-Wat, +ImH)":
        return f"{from_formula} + ImH -> {to_formula} + Wat"
    return f"{from_formula} -> {to_formula}"


def _compute_de_rxn_kcal(
    edge_type: str,
    e_from: float,
    e_to: float,
    monomer_refs: Dict[str, float],
) -> Optional[float]:
    d_cluster = e_to - e_from

    if edge_type == "Add Im(-)":
        e_im_minus = monomer_refs.get("Im-")
        return None if e_im_minus is None else d_cluster - e_im_minus
    if edge_type == "Add ImH":
        e_imh = monomer_refs.get("ImH")
        return None if e_imh is None else d_cluster - e_imh
    if edge_type == "Deprotonation (-H+)":
        e_h = monomer_refs.get("H")
        return None if e_h is None else d_cluster + e_h
    if edge_type == "Add Wat":
        e_wat = monomer_refs.get("Wat")
        return None if e_wat is None else d_cluster - e_wat
    if edge_type == "Exchange (-Wat, +Im(-))":
        e_im_minus = monomer_refs.get("Im-")
        e_wat = monomer_refs.get("Wat")
        if e_im_minus is None or e_wat is None:
            return None
        return d_cluster - e_im_minus + e_wat
    if edge_type == "Exchange (-Wat, +ImH)":
        e_imh = monomer_refs.get("ImH")
        e_wat = monomer_refs.get("Wat")
        if e_imh is None or e_wat is None:
            return None
        return d_cluster - e_imh + e_wat
    return None


def _collect_monomer_refs(run_path: str, warn_missing_h: bool = True) -> Dict[str, float]:
    monomer_dirs = {
        "Im-": "Im-_monomer",
        "ImH": "ImH_monomer",
        "Wat": "Wat_monomer",
        "Zn": "Zn_monomer",
    }
    refs: Dict[str, float] = {}
    for label, subdir in monomer_dirs.items():
        energy = get_Etot_amber(os.path.join(run_path, subdir, "min.out"))
        if energy is not None:
            refs[label] = energy

    e_h = get_Etot_amber(os.path.join(run_path, "H_monomer", "min.out"))
    if e_h is not None:
        refs["H"] = e_h
    else:
        refs["H"] = 0.0
        if warn_missing_h:
            h_dir = os.path.join(run_path, "H_monomer")
            reason = "H_monomer energy parse failed" if os.path.isdir(h_dir) else "missing H_monomer"
            print(f"  WARNING: {os.path.basename(run_path)} {reason}, using E(H)=0.0")

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
    monomer_refs = _collect_monomer_refs(run_path, warn_missing_h=not use_bsse)
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

    e_h = monomer_refs.get("H")
    if e_h is not None:
        for from_formula, to_formula, from_key, to_key in MONOMER_H_REACTIONS:
            e_from = monomer_refs.get(from_key)
            e_to = monomer_refs.get(to_key)
            if e_from is None or e_to is None:
                continue
            balanced = _balanced_reaction_plain(
                MONOMER_H_REACTION_TYPE,
                from_formula,
                to_formula,
            )
            rows[(MONOMER_H_REACTION_TYPE, from_formula, to_formula, balanced)] = (
                e_to - e_from - e_h
            )

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
    df = analyze_runs(base_dir="SP_init")

    if df.empty:
        print("No reaction energies found under SP_init/run*.")
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
