#!/usr/bin/env python3
"""
Zn–MeOH dissociation binding from Molpro single-points under each run*/:

  run*/Zn/sbatch.log
  run*/MeOH/sbatch.log
  run*/ZnMeOH/sbatch.log   (also accepts ZnMeoH)

Ebind_dist_Zn_MeOH (BSSE=no) = E(ZnMeOH) − E(Zn) − E(MeOH), same as ORCA fragment
supermolecule minus monomers without counterpoise, in kcal/mol.

BSSE=yes uses counterpoise monomer jobs in the same run directory (when present):
  E(ZnMeOH) − E(Zn_dummyMeOH) − E(dummyZn_MeOH)
i.e. supermolecule minus Zn with ghost MeOH minus MeOH with ghost Zn.
Runs without both dummy jobs still emit a BSSE=yes row with NaN binding terms
(see react_benchmark/ccsd_test/analyze_dE.py pattern).

Energies (Hartree) are read from each sbatch.log using the same tail convention
as a typical Molpro stdout capture (file often ends with a newline, so splitting
on ``\\n`` leaves an empty last element):

  - Second-to-last line must contain ``Molpro calculation terminated``.
  - The total energy is parsed from the 7th line from the end: a line containing
    ``energy=`` (e.g. ``CCSD(T)/... energy=   -115.55...``).
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))

HARTREE_TO_KCAL = 627.5094740631

_RE_SUMMARY_ENERGY = re.compile(
    r"energy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
)
_RE_RUN_ID = re.compile(r"^run(\d+)$")


def _run_sort_key(run_name: str) -> tuple[int, int | str]:
    """
    Natural run ordering:
      run0, run1, ..., run9, run10, ...
    Non-matching names are placed after canonical run IDs.
    """
    m = _RE_RUN_ID.match(run_name)
    if m:
        return (0, int(m.group(1)))
    return (1, run_name)


def _parse_molpro_sbatch_energy_hartree(log_path: str) -> float | None:
    """
    Total energy in Hartree from Molpro sbatch.log.

    Validates ``Molpro calculation terminated`` on the second-to-last line when
    the log is split on ``\\n`` (matches editors that show a blank line after the
    final newline). Parses ``energy=`` from the 7th line from the end.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    lines = text.split("\n")
    if len(lines) < 7:
        return None
    if "Molpro calculation terminated" not in lines[-2]:
        return None
    m = _RE_SUMMARY_ENERGY.search(lines[-7])
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _resolve_znmeoh_dir(run_path: str) -> str | None:
    for name in ("ZnMeOH", "ZnMeoH"):
        if os.path.isdir(os.path.join(run_path, name)):
            return name
    return None


def _cp_monomer_logs(run_path: str) -> tuple[str, str] | None:
    """
    Counterpoise monomer jobs (full complex basis, partner as dummy):
      Zn_dummyMeOH  -> E(Zn; ghost MeOH)
      dummyZn_MeOH  -> E(MeOH; ghost Zn)
    Returns (path_Zn_dummyMeOH_log, path_dummyZn_MeOH_log) if both dirs exist.
    """
    zn_d = os.path.join(run_path, "Zn_dummyMeOH", "sbatch.log")
    me_d = os.path.join(run_path, "dummyZn_MeOH", "sbatch.log")
    if os.path.isfile(zn_d) and os.path.isfile(me_d):
        return zn_d, me_d
    return None


def _row_for_bsse(
    run_name: str,
    method_name: str,
    znm_dir: str | None,
    znm_log: str,
    zn_log: str,
    meoh_log: str,
    bsse: str,
) -> dict:
    """One result row: bsse 'no' uses Zn/MeOH monomer logs; 'yes' uses CP dummy monomer logs when paths differ."""
    e_zn_h = _parse_molpro_sbatch_energy_hartree(zn_log)
    e_meoh_h = _parse_molpro_sbatch_energy_hartree(meoh_log)
    e_znm_h = _parse_molpro_sbatch_energy_hartree(znm_log) if znm_log else None

    missing = []
    if e_zn_h is None:
        missing.append(os.path.dirname(zn_log) + "/sbatch.log")
    if e_meoh_h is None:
        missing.append(os.path.dirname(meoh_log) + "/sbatch.log")
    if znm_dir is None:
        missing.append("ZnMeOH/")
    elif e_znm_h is None:
        missing.append(f"{znm_dir}/sbatch.log")

    if missing:
        print(f"Warning: {run_name} (BSSE={bsse}): missing or unparsed energy in {', '.join(missing)}")

    e_zn = np.nan if e_zn_h is None else e_zn_h * HARTREE_TO_KCAL
    e_meoh = np.nan if e_meoh_h is None else e_meoh_h * HARTREE_TO_KCAL
    e_znm = np.nan if e_znm_h is None else e_znm_h * HARTREE_TO_KCAL

    if e_zn_h is not None and e_meoh_h is not None and e_znm_h is not None:
        ebind = (e_znm_h - e_zn_h - e_meoh_h) * HARTREE_TO_KCAL
    else:
        ebind = np.nan

    return {
        "Run": run_name,
        "Method": method_name,
        "BSSE": bsse,
        "ZnMeOH_dir": znm_dir if znm_dir else "",
        "E_Zn": e_zn,
        "E_MeOH": e_meoh,
        "E_ZnMeOH": e_znm,
        "Ebind_dist_Zn_MeOH": ebind,
    }


def analyze_runs(base_dir: str | None = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = _BASE

    results: list[dict] = []
    run_paths = sorted(
        glob.glob(os.path.join(base_dir, "run*")),
        key=lambda p: _run_sort_key(os.path.basename(p)),
    )
    for run_path in run_paths:
        if not os.path.isdir(run_path):
            continue
        run_name = os.path.basename(run_path)

        method_name = run_name
        notes_path = os.path.join(run_path, "notes.yaml")
        if os.path.isfile(notes_path):
            with open(notes_path, "r") as f:
                try:
                    notes = yaml.safe_load(f)
                    if isinstance(notes, dict) and notes.get("name"):
                        method_name = str(notes["name"])
                except Exception:
                    pass

        zn_log = os.path.join(run_path, "Zn", "sbatch.log")
        meoh_log = os.path.join(run_path, "MeOH", "sbatch.log")
        znm_dir = _resolve_znmeoh_dir(run_path)
        znm_log = os.path.join(run_path, znm_dir, "sbatch.log") if znm_dir else ""

        results.append(
            _row_for_bsse(run_name, method_name, znm_dir, znm_log, zn_log, meoh_log, "no")
        )

        cp = _cp_monomer_logs(run_path)
        if cp is not None:
            zn_cp_log, meoh_cp_log = cp
            results.append(
                _row_for_bsse(
                    run_name, method_name, znm_dir, znm_log, zn_cp_log, meoh_cp_log, "yes"
                )
            )
        else:
            print(
                f"  {run_name} (BSSE=yes): no Zn_dummyMeOH/dummyZn_MeOH pair, BSSE row kept with NaN"
            )
            e_znm_h_ph = _parse_molpro_sbatch_energy_hartree(znm_log) if znm_log else None
            e_znm_ph = np.nan if e_znm_h_ph is None else e_znm_h_ph * HARTREE_TO_KCAL
            results.append(
                {
                    "Run": run_name,
                    "Method": method_name,
                    "BSSE": "yes",
                    "ZnMeOH_dir": znm_dir if znm_dir else "",
                    "E_Zn": np.nan,
                    "E_MeOH": np.nan,
                    "E_ZnMeOH": e_znm_ph,
                    "Ebind_dist_Zn_MeOH": np.nan,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    os.chdir(_BASE)
    df = analyze_runs()

    if df.empty:
        print("No run* directories found.")
        sys.exit(0)

    df["_run_ord"] = df["Run"].map(_run_sort_key)
    df["_bsse_ord"] = df["BSSE"].map({"no": 0, "yes": 1})
    df.sort_values(
        by=["_run_ord", "_bsse_ord"],
        inplace=True,
        ignore_index=True,
        na_position="last",
    )
    df.drop(columns=["_run_ord", "_bsse_ord"], inplace=True)

    col_order = [
        "Run",
        "Method",
        "BSSE",
        "ZnMeOH_dir",
        "E_Zn",
        "E_MeOH",
        "E_ZnMeOH",
        "Ebind_dist_Zn_MeOH",
    ]
    col_order = [c for c in col_order if c in df.columns]
    rest = [c for c in df.columns if c not in col_order]
    df = df[col_order + rest]

    output_file = os.path.join(_BASE, "dE.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(
        "\nEbind_dist_Zn_MeOH: BSSE=no → E(ZnMeOH)−E(Zn)−E(MeOH); "
        "BSSE=yes → E(ZnMeOH)−E(Zn_dummyMeOH)−E(dummyZn_MeOH)  (kcal/mol). "
        "E_* columns are the monomer energies used in that binding formula."
    )
    print(
        "Energies: sbatch.log second-to-last line (split on newline) must contain "
        "'Molpro calculation terminated'; total energy from 'energy=' on the 7th line from the end.\n"
    )
    print("Full table (kcal/mol); dE.csv has the same columns:")
    print(df.to_string(index=False))
