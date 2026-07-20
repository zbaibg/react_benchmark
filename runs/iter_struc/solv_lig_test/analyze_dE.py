#!/usr/bin/env python3
"""Ligand-exchange energies: ZnLSolv5 + Solv -> ZnSolv6 + L."""

import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Make repo tools/ importable (analib lives there).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
for _path in (TOOLS_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analib import get_Etot_amber, get_complex_energies  # type: ignore

# Legacy: 1Zn_1ImH_5Wat / 1Zn_1Im-_5Wat / 1Zn_0ImH_6Wat
LEGACY_RE = re.compile(r"^1Zn_(\d+)(ImH|Im-)_(\d+)Wat$")
# Explicit: 1Zn_1Im-_0ImH_5MeOH / 1Zn_1MIm_0MImH_5Wat / ...
EXPLICIT_RE = re.compile(
    r"^1Zn_(\d+)(Im-|MIm)_(\d+)(ImH|MImH)_(\d+)(Wat|MeOH)$"
)
RUN_RE = re.compile(r"^run(\d+)$")

REACTION_TYPE = "Exchange (-L, +Solv)"
DEFAULT_BASE_DIR = "SP_init"
RUN_SOURCE_OVERRIDES = {
    "run41": "qm_minimize",
}

MONOMER_DIRS = {
    "Im-": "Im-_monomer",
    "ImH": "ImH_monomer",
    "MIm": "MIm_monomer",
    "MImH": "MImH_monomer",
    "Wat": "Wat_monomer",
    "MeOH": "MeOH_monomer",
}


@dataclass(frozen=True)
class ComplexInfo:
    formula: str
    ligand: Optional[str]  # None for ZnSolv6
    n_ligand: int
    solvent: str
    n_solvent: int

    @property
    def is_zn_l_solv5(self) -> bool:
        return self.n_ligand == 1 and self.n_solvent == 5 and self.ligand is not None

    @property
    def is_zn_solv6(self) -> bool:
        return self.n_ligand == 0 and self.n_solvent == 6


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


def _parse_complex(formula: str) -> Optional[ComplexInfo]:
    match = LEGACY_RE.fullmatch(formula)
    if match is not None:
        n_ligand = int(match.group(1))
        ligand = match.group(2)
        n_solvent = int(match.group(3))
        return ComplexInfo(
            formula=formula,
            ligand=None if n_ligand == 0 else ligand,
            n_ligand=n_ligand,
            solvent="Wat",
            n_solvent=n_solvent,
        )

    match = EXPLICIT_RE.fullmatch(formula)
    if match is not None:
        n_a = int(match.group(1))
        lig_a = match.group(2)
        n_b = int(match.group(3))
        lig_b = match.group(4)
        n_solvent = int(match.group(5))
        solvent = match.group(6)
        n_ligand = n_a + n_b
        if n_ligand == 0:
            ligand = None
        elif n_a == 1 and n_b == 0:
            ligand = lig_a
        elif n_a == 0 and n_b == 1:
            ligand = lig_b
        else:
            return None
        return ComplexInfo(
            formula=formula,
            ligand=ligand,
            n_ligand=n_ligand,
            solvent=solvent,
            n_solvent=n_solvent,
        )

    return None


def _pretty_formula(formula: str) -> str:
    """Drop zero-count species for display, e.g. 1Zn_0MIm_1MImH_5Wat -> 1Zn_1MImH_5Wat."""
    match = LEGACY_RE.fullmatch(formula)
    if match is not None:
        n_ligand = int(match.group(1))
        ligand = match.group(2)
        n_solvent = int(match.group(3))
        parts = ["1Zn"]
        if n_ligand > 0:
            parts.append(f"{n_ligand}{ligand}")
        if n_solvent > 0:
            parts.append(f"{n_solvent}Wat")
        return "_".join(parts)

    match = EXPLICIT_RE.fullmatch(formula)
    if match is not None:
        n_a = int(match.group(1))
        lig_a = match.group(2)
        n_b = int(match.group(3))
        lig_b = match.group(4)
        n_solvent = int(match.group(5))
        solvent = match.group(6)
        parts = ["1Zn"]
        if n_a > 0:
            parts.append(f"{n_a}{lig_a}")
        if n_b > 0:
            parts.append(f"{n_b}{lig_b}")
        if n_solvent > 0:
            parts.append(f"{n_solvent}{solvent}")
        return "_".join(parts)

    return formula


def _complex_sort_key(formula: str) -> Tuple[str, int, str, str]:
    info = _parse_complex(formula)
    if info is None:
        return ("", 10**9, "", formula)
    return (info.solvent, -info.n_solvent, info.ligand or "", formula)


def _iter_complex_names(run_path: str, source_base_dir: str) -> List[str]:
    complex_names: List[str] = []
    for entry in os.listdir(run_path):
        entry_path = os.path.join(run_path, entry)
        if not os.path.isdir(entry_path):
            continue

        if source_base_dir == "qm_minimize":
            if _parse_complex(entry) is not None:
                complex_names.append(entry)
            continue

        if entry.endswith("_full"):
            complex_name = entry[: -len("_full")]
            if _parse_complex(complex_name) is not None:
                complex_names.append(complex_name)
    return sorted(set(complex_names), key=_complex_sort_key)


def _collect_monomer_refs(run_path: str) -> Dict[str, float]:
    refs: Dict[str, float] = {}
    for label, subdir in MONOMER_DIRS.items():
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


def _find_zn_solv6(
    solvent: str,
    complex_infos: Dict[str, ComplexInfo],
) -> Optional[str]:
    candidates = [
        formula
        for formula, info in complex_infos.items()
        if info.is_zn_solv6 and info.solvent == solvent
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_complex_sort_key)[0]


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
    if not complex_energies:
        return {}

    complex_infos = {
        formula: info
        for formula in complex_energies
        for info in [_parse_complex(formula)]
        if info is not None
    }
    zn_solv6_by_solvent: Dict[str, str] = {}
    for solvent in {info.solvent for info in complex_infos.values()}:
        zn6 = _find_zn_solv6(solvent, complex_infos)
        if zn6 is not None:
            zn_solv6_by_solvent[solvent] = zn6

    rows: Dict[Tuple[str, str, str, str], float] = {}
    for formula, info in complex_infos.items():
        if not info.is_zn_l_solv5:
            continue

        ligand = info.ligand
        solvent = info.solvent
        assert ligand is not None

        zn6 = zn_solv6_by_solvent.get(solvent)
        e_l = monomer_refs.get(ligand)
        e_solv = monomer_refs.get(solvent)
        if zn6 is None or e_l is None or e_solv is None:
            continue

        e_from = complex_energies[formula]
        e_to = complex_energies[zn6]
        # ZnLSolv5 + Solv -> ZnSolv6 + L
        dE_rxn = e_to + e_l - e_from - e_solv
        from_disp = _pretty_formula(formula)
        to_disp = _pretty_formula(zn6)
        balanced = f"{from_disp} + {solvent} -> {to_disp} + {ligand}"
        rows[(REACTION_TYPE, from_disp, to_disp, balanced)] = dE_rxn

    return rows


def _collect_run_sources(default_base_dir: str) -> List[Tuple[str, str, str]]:
    run_sources: Dict[str, Tuple[str, str]] = {}

    if os.path.exists(default_base_dir):
        for run_path in sorted(
            glob.glob(os.path.join(default_base_dir, "run*")),
            key=_run_sort_key,
        ):
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

    def _record_sort_key(record: Dict[str, object]) -> Tuple[str, str, str]:
        return (
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
        print(
            "No ligand-exchange energies found "
            "(need ZnLSolv5, ZnSolv6, and monomer refs)."
        )
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
            col
            for col in df.columns
            if col
            not in (
                "reaction_type",
                "from_formula",
                "to_formula",
                "balanced_reaction",
            )
        ]
        print("\nLigand exchange: ZnLSolv5 + Solv -> ZnSolv6 + L (kcal/mol):")
        print(
            df[["reaction_type", "balanced_reaction"] + value_cols].to_string(
                index=False
            )
        )
