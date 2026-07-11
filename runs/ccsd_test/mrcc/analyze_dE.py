#!/usr/bin/env python3
"""
Zn–MeOH binding analysis from MRCC single-point outputs under each run*/:

  run*/Zn/out
  run*/MeOH/out
  run*/ZnMeOH/out   (also accepts ZnMeoH)

Counterpoise (BSSE=yes) uses ghost-basis monomer jobs (when present):

  E(ZnMeOH) - E(dummyZnMeOH) - E(ZndummyMeOH)

For each BSSE mode, this script reports energies and Ebind at three MRCC levels:
  (T), (T*), (T+)
in kcal/mol.
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
LEVELS = ("T", "T*", "T+")

_RE_TOTAL = re.compile(
    r"Total\s+CCSD\(F12\*\)\((T\+?|T\*)\)\s+energy\s+\[au\]:\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
_RE_TOTAL_CCSDT = re.compile(
    r"Total\s+CCSD\(T\)\s+energy\s+\[au\]:\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def _parse_mrcc_total_energies_hartree(out_path: str) -> dict[str, float] | None:
    """Parse final Total CCSD(F12*)(T/T*/T+) energies [au] from MRCC out."""
    if not os.path.isfile(out_path):
        return None
    try:
        with open(out_path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    parsed: dict[str, float] = {}
    for level, value in _RE_TOTAL.findall(text):
        try:
            parsed[level] = float(value)
        except ValueError:
            continue

    if all(level in parsed for level in LEVELS):
        return parsed
    return None


def _parse_mrcc_total_ccsdt_energy_hartree(out_path: str) -> float | None:
    """Parse non-F12 Total CCSD(T) energy [au] from MRCC out."""
    if not os.path.isfile(out_path):
        return None
    try:
        with open(out_path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    matches = _RE_TOTAL_CCSDT.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _resolve_znmeoh_dir(run_path: str) -> str | None:
    for name in ("ZnMeOH", "ZnMeoH"):
        if os.path.isdir(os.path.join(run_path, name)):
            return name
    return None


def _resolve_cp_monomer_outs(run_path: str) -> tuple[str, str] | None:
    """
    Counterpoise monomer jobs in MRCC run dir:
      dummyZnMeOH -> MeOH with ghost Zn basis
      ZndummyMeOH -> Zn with ghost MeOH basis
    """
    meoh_cp = os.path.join(run_path, "dummyZnMeOH", "out")
    zn_cp = os.path.join(run_path, "ZndummyMeOH", "out")
    if os.path.isfile(meoh_cp) and os.path.isfile(zn_cp):
        return zn_cp, meoh_cp
    return None


def _safe_kcal(energy_h: float | None) -> float:
    return np.nan if energy_h is None else energy_h * HARTREE_TO_KCAL


def _build_level_columns(
    e_zn: dict[str, float] | None,
    e_meoh: dict[str, float] | None,
    e_znmeoh: dict[str, float] | None,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for level in LEVELS:
        ez = None if e_zn is None else e_zn.get(level)
        em = None if e_meoh is None else e_meoh.get(level)
        ec = None if e_znmeoh is None else e_znmeoh.get(level)

        row[f"E_Zn_{level}"] = _safe_kcal(ez)
        row[f"E_MeOH_{level}"] = _safe_kcal(em)
        row[f"E_ZnMeOH_{level}"] = _safe_kcal(ec)

        if ez is None or em is None or ec is None:
            row[f"Ebind_{level}"] = np.nan
        else:
            row[f"Ebind_{level}"] = (ec - ez - em) * HARTREE_TO_KCAL
    return row


def _row_for_bsse(
    run_name: str,
    method_name: str,
    znmeoh_dir: str | None,
    znmeoh_out: str,
    zn_out: str,
    meoh_out: str,
    bsse: str,
) -> dict:
    e_zn = _parse_mrcc_total_energies_hartree(zn_out)
    e_meoh = _parse_mrcc_total_energies_hartree(meoh_out)
    e_znmeoh = _parse_mrcc_total_energies_hartree(znmeoh_out) if znmeoh_out else None
    e_zn_ccsdt = _parse_mrcc_total_ccsdt_energy_hartree(zn_out)
    e_meoh_ccsdt = _parse_mrcc_total_ccsdt_energy_hartree(meoh_out)
    e_znmeoh_ccsdt = (
        _parse_mrcc_total_ccsdt_energy_hartree(znmeoh_out) if znmeoh_out else None
    )

    missing = []
    if e_zn is None:
        missing.append(os.path.dirname(zn_out) + "/out")
    if e_meoh is None:
        missing.append(os.path.dirname(meoh_out) + "/out")
    if znmeoh_dir is None:
        missing.append("ZnMeOH/")
    elif e_znmeoh is None:
        missing.append(f"{znmeoh_dir}/out")
    if missing:
        print(f"Warning: {run_name} (BSSE={bsse}): missing or unparsed {', '.join(missing)}")

    row = {
        "Run": run_name,
        "Method": method_name,
        "BSSE": bsse,
        "ZnMeOH_dir": znmeoh_dir if znmeoh_dir else "",
    }
    row.update(_build_level_columns(e_zn, e_meoh, e_znmeoh))
    if e_zn_ccsdt is None or e_meoh_ccsdt is None or e_znmeoh_ccsdt is None:
        row["Ebind"] = np.nan
    else:
        row["Ebind"] = (e_znmeoh_ccsdt - e_zn_ccsdt - e_meoh_ccsdt) * HARTREE_TO_KCAL
    return row


def analyze_runs(base_dir: str | None = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = _BASE

    rows: list[dict] = []
    for run_path in sorted(glob.glob(os.path.join(base_dir, "run*"))):
        if not os.path.isdir(run_path):
            continue
        run_name = os.path.basename(run_path)

        method_name = run_name
        notes_path = os.path.join(run_path, "notes.yaml")
        if os.path.isfile(notes_path):
            with open(notes_path, "r", errors="replace") as f:
                try:
                    notes = yaml.safe_load(f)
                    if isinstance(notes, dict) and notes.get("name"):
                        method_name = str(notes["name"])
                except Exception:
                    pass

        zn_out = os.path.join(run_path, "Zn", "out")
        meoh_out = os.path.join(run_path, "MeOH", "out")
        znmeoh_dir = _resolve_znmeoh_dir(run_path)
        znmeoh_out = os.path.join(run_path, znmeoh_dir, "out") if znmeoh_dir else ""

        rows.append(
            _row_for_bsse(run_name, method_name, znmeoh_dir, znmeoh_out, zn_out, meoh_out, "no")
        )

        cp = _resolve_cp_monomer_outs(run_path)
        if cp is not None:
            zn_cp_out, meoh_cp_out = cp
            rows.append(
                _row_for_bsse(
                    run_name,
                    method_name,
                    znmeoh_dir,
                    znmeoh_out,
                    zn_cp_out,
                    meoh_cp_out,
                    "yes",
                )
            )
        else:
            print(
                f"  {run_name} (BSSE=yes): no dummyZnMeOH/ZndummyMeOH pair, "
                "BSSE row kept with NaN."
            )
            e_znmeoh = _parse_mrcc_total_energies_hartree(znmeoh_out) if znmeoh_out else None
            row = {
                "Run": run_name,
                "Method": method_name,
                "BSSE": "yes",
                "ZnMeOH_dir": znmeoh_dir if znmeoh_dir else "",
            }
            row.update(_build_level_columns(None, None, e_znmeoh))
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.chdir(_BASE)
    df = analyze_runs()
    if df.empty:
        print("No run* directories found.")
        sys.exit(0)

    df["_bsse_ord"] = df["BSSE"].map({"no": 0, "yes": 1})
    df.sort_values(
        by=["Run", "_bsse_ord"],
        inplace=True,
        ignore_index=True,
        na_position="last",
    )
    df.drop(columns="_bsse_ord", inplace=True)

    prefix_cols = ["Run", "Method", "BSSE", "ZnMeOH_dir"]
    dynamic_cols: list[str] = []
    for level in LEVELS:
        dynamic_cols.extend(
            [f"E_Zn_{level}", f"E_MeOH_{level}", f"E_ZnMeOH_{level}", f"Ebind_{level}"]
        )
    col_order = [c for c in prefix_cols + dynamic_cols if c in df.columns]
    rest = [c for c in df.columns if c not in col_order]
    df = df[col_order + rest]

    out_csv = os.path.join(_BASE, "dE.csv")
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    print(
        "\nEbind_(T/T*/T+): BSSE=no -> E(ZnMeOH)-E(Zn)-E(MeOH); "
        "BSSE=yes -> E(ZnMeOH)-E(dummyZnMeOH)-E(ZndummyMeOH). "
        "All energies are in kcal/mol."
    )
    ebind_view_cols = ["Run", "Method", "BSSE"] + [
        f"Ebind_{level}" for level in LEVELS if f"Ebind_{level}" in df.columns
    ]
    if "Ebind" in df.columns:
        ebind_view_cols.append("Ebind")
    print("\nEbind table:")
    print(df[ebind_view_cols].to_string(index=False))
