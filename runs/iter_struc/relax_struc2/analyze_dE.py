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

FULL_COMPLEX_DIR_RE = re.compile(r"^(1Zn_\d+MIm_\d+MImH_\d+MeOH)_full$")
BARE_COMPLEX_DIR_RE = re.compile(r"^(1Zn_\d+MIm_\d+MImH_\d+MeOH)$")
FORMULA_RE = re.compile(r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$")
RUN_RE = re.compile(r"^run(\d+)$")
MONOMER_H_REACTION_TYPE = "monomer Deprotonation H"
MONOMER_H_REACTIONS = (
    ("MImH_monomer", "MImH2_monomer", "MImH", "MImH2"),
    ("MIm_monomer", "MImH_monomer", "MIm", "MImH"),
    ("MeOH_monomer", "MeOH2_monomer", "MeOH", "MeOH2"),
)
REACTION_TYPE_ORDER = {
    "Add MeOH": 0,
    "Add MImH": 1,
    "Add MIm(-)": 2,
    "Deprotonation (-H+)": 3,
    "Exchange (-MeOH, +MIm(-))": 4,
    "Exchange (-MeOH, +MImH)": 5,
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
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _complex_sort_key(formula: str) -> Tuple[int, int, int, str]:
    counts = _parse_formula_counts(formula)
    if counts is None:
        return (10**9, 10**9, 10**9, formula)
    n_mim, n_mimh, n_meoh = counts
    return (n_mim + n_mimh, n_meoh, n_mim, formula)


def _iter_complex_names(run_path: str, source_base_dir: str) -> List[str]:
    complex_names: List[str] = []
    for entry in os.listdir(run_path):
        if source_base_dir == "qm_minimize":
            if BARE_COMPLEX_DIR_RE.fullmatch(entry) and os.path.isdir(os.path.join(run_path, entry)):
                complex_names.append(entry)
            continue

        match = FULL_COMPLEX_DIR_RE.fullmatch(entry)
        if match is not None and os.path.isdir(os.path.join(run_path, entry)):
            complex_names.append(match.group(1))
    return sorted(set(complex_names), key=_complex_sort_key)


def _reaction_type_from_counts(
    reactant_counts: Tuple[int, int, int],
    product_counts: Tuple[int, int, int],
) -> Optional[str]:
    d_mim = product_counts[0] - reactant_counts[0]
    d_mimh = product_counts[1] - reactant_counts[1]
    d_meoh = product_counts[2] - reactant_counts[2]

    if (d_mim, d_mimh, d_meoh) == (0, 0, 1):
        return "Add MeOH"
    if (d_mim, d_mimh, d_meoh) == (0, 1, 0):
        return "Add MImH"
    if (d_mim, d_mimh, d_meoh) == (1, 0, 0):
        return "Add MIm(-)"
    if (d_mim, d_mimh, d_meoh) == (1, -1, 0):
        return "Deprotonation (-H+)"
    if (d_mim, d_mimh, d_meoh) == (1, 0, -1):
        return "Exchange (-MeOH, +MIm(-))"
    if (d_mim, d_mimh, d_meoh) == (0, 1, -1):
        return "Exchange (-MeOH, +MImH)"
    return None


def _balanced_reaction_plain(edge_type: str, from_formula: str, to_formula: str) -> str:
    if edge_type == MONOMER_H_REACTION_TYPE:
        return f"{from_formula} + H -> {to_formula}"
    if edge_type == "Add MeOH":
        return f"{from_formula} + MeOH -> {to_formula}"
    if edge_type == "Add MImH":
        return f"{from_formula} + MImH -> {to_formula}"
    if edge_type == "Add MIm(-)":
        return f"{from_formula} + MIm- -> {to_formula}"
    if edge_type == "Deprotonation (-H+)":
        return f"{from_formula} -> {to_formula} + H+"
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        return f"{from_formula} + MIm- -> {to_formula} + MeOH"
    if edge_type == "Exchange (-MeOH, +MImH)":
        return f"{from_formula} + MImH -> {to_formula} + MeOH"
    return f"{from_formula} -> {to_formula}"


def _compute_de_rxn_kcal(
    edge_type: str,
    e_from: float,
    e_to: float,
    monomer_refs: Dict[str, float],
) -> Optional[float]:
    d_cluster = e_to - e_from

    if edge_type == "Add MIm(-)":
        e_mim = monomer_refs.get("MIm")
        return None if e_mim is None else d_cluster - e_mim
    if edge_type == "Add MImH":
        e_mimh = monomer_refs.get("MImH")
        return None if e_mimh is None else d_cluster - e_mimh
    if edge_type == "Deprotonation (-H+)":
        e_h = monomer_refs.get("H")
        return None if e_h is None else d_cluster + e_h
    if edge_type == "Add MeOH":
        e_meoh = monomer_refs.get("MeOH")
        return None if e_meoh is None else d_cluster - e_meoh
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        e_mim = monomer_refs.get("MIm")
        e_meoh = monomer_refs.get("MeOH")
        if e_mim is None or e_meoh is None:
            return None
        return d_cluster - e_mim + e_meoh
    if edge_type == "Exchange (-MeOH, +MImH)":
        e_mimh = monomer_refs.get("MImH")
        e_meoh = monomer_refs.get("MeOH")
        if e_mimh is None or e_meoh is None:
            return None
        return d_cluster - e_mimh + e_meoh
    return None


def _collect_monomer_refs(run_path: str, warn_missing_h: bool = True) -> Dict[str, float]:
    monomer_dirs = {
        "MIm": "MIm_monomer",
        "MImH": "MImH_monomer",
        "MImH2": "MImH2_monomer",
        "MeOH": "MeOH_monomer",
        "MeOH2": "MeOH2_monomer",
        "H": "H_monomer",
    }
    refs: Dict[str, float] = {}
    for label, subdir in monomer_dirs.items():
        energy = get_Etot_amber(os.path.join(run_path, subdir, "min.out"))
        if energy is not None:
            refs[label] = energy

    if "H" not in refs:
        refs["H"] = 0.0
        if warn_missing_h:
            print(f"  WARNING: {os.path.basename(run_path)} missing H_monomer, using E(H)=0.0")

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
