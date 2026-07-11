#!/usr/bin/env python3
import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Make project root (where analib.py lives) importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import get_Etot_amber  # type: ignore

# Runs to include in dEabs curve plotting.
# Set to None to include all runs in dEabs_df.
PLOT_RUNS: Optional[List[str]] = None
# Reference run used by delta plot. Change this manually as needed.
# If this run is absent from the selected data, the script will raise an error.
REF_RUN = "run102"
PREFERRED_MONOMER_ORDER = ["MImH", "MImH2", "H3O", "Wat", "MeOH", "MeOH2"]
SCAN_DIR_PATTERN = re.compile(r"^(?P<monomer>.+)_(?P<scan>m?\d+(?:\.\d+)?)_monomer$")


def _parse_scan_label(label: str) -> float:
    if label.startswith("m"):
        return -float(label[1:])
    return float(label)


def _parse_struct_dir(dirname: str) -> Tuple[str, float]:
    """
    dirname examples: MImH_0.2_monomer, MImH2_m0.3_monomer
    Return (monomer_name, numeric_scan_displacement).
    """
    match = SCAN_DIR_PATTERN.fullmatch(dirname)
    if match is None:
        raise ValueError(f"Unexpected directory name for scan: {dirname}")

    monomer = match.group("monomer")
    scan_label = match.group("scan")
    return monomer, _parse_scan_label(scan_label)


def _ordered_monomers(monomers: List[str]) -> List[str]:
    preferred = [m for m in PREFERRED_MONOMER_ORDER if m in monomers]
    remaining = sorted(m for m in monomers if m not in PREFERRED_MONOMER_ORDER)
    return preferred + remaining

def collect_monomer_energies(base_dir: str = "SP_init") -> pd.DataFrame:
    """
    Process monomer scans under the current H_scan directory only:
    - Traverse all *monomer directories under base_dir/run*/
    - Auto-detect monomer names from directory names like
      <monomer>_<scan>_monomer
    - For each monomer and each run, collect total energy for each scan value.
    """
    results: List[Dict] = []

    if not os.path.exists(base_dir):
        return pd.DataFrame()

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run*")))

    for run_path in run_dirs:
        run_name = os.path.basename(run_path)

        # Optional: read method name from notes.yaml if present
        method_name = run_name
        notes_path = os.path.join(run_path, "notes.yaml")
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r") as f:
                    notes = yaml.safe_load(f) or {}
                method_name = str(notes.get("name", run_name))
            except Exception:
                method_name = run_name

        for struct_dir in sorted(glob.glob(os.path.join(run_path, "*_monomer"))):
            if not os.path.isdir(struct_dir):
                continue

            dirname = os.path.basename(struct_dir)
            try:
                monomer, scan_value = _parse_struct_dir(dirname)
            except ValueError:
                continue

            # Always pass the min.out path to the parser.
            # If min.out is missing or unparseable, get_Etot_amber() can
            # still fall back to orc_job.dat (or q-xtb energy) in struct_dir.
            min_out = os.path.join(struct_dir, "min.out")
            e_tot = get_Etot_amber(min_out)
            if e_tot is None:
                continue

            results.append(
                {
                    "Phase": base_dir,
                    "Run": run_name,
                    "Method": method_name,
                    "Monomer": monomer,
                    "StructDir": dirname,
                    "scan_value": scan_value,
                    "E_tot": e_tot,
                }
            )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.sort_values(by=["Monomer", "Run", "scan_value"], inplace=True, ignore_index=True)
    return df

def _format_scan_label(value: float) -> str:
    """
    Format a scan value to match directory-style labels:
    - 0.0  -> "0.0"
    - -0.2 -> "m0.2"
    - 0.3  -> "0.3"
    """
    if np.isclose(value, 0.0):
        return "0.0"
    if value < 0:
        return f"m{abs(value):.1f}"
    return f"{value:.1f}"


def _get_target_values_from_energy_df(df: pd.DataFrame) -> List[float]:
    if df.empty or "scan_value" not in df.columns:
        return []

    scan_values = sorted(
        float(v) for v in df["scan_value"].dropna().unique() if not np.isclose(v, 0.0)
    )
    return scan_values

def _get_target_values_from_dEabs_df(dEabs_df: pd.DataFrame) -> List[float]:
    target_values: List[float] = []
    for column in dEabs_df.columns:
        if not column.startswith("dE_0.0_"):
            continue
        label = column[len("dE_0.0_") :]
        try:
            value = _parse_scan_label(label)
        except ValueError:
            continue
        if np.isclose(value, 0.0):
            continue
        target_values.append(value)
    return sorted(set(target_values))


def _make_axes_grid(n_panels: int) -> Tuple[plt.Figure, np.ndarray]:
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )
    return fig, np.atleast_1d(axes).ravel()


def _select_reference_row(sub: pd.DataFrame, ref_run: str) -> Tuple[pd.Series, str]:
    if sub.empty:
        raise ValueError("Reference selection requires non-empty data.")

    run_sub = sub[sub["Run"] == ref_run]
    if run_sub.empty:
        monomer = str(sub["Monomer"].iloc[0]) if "Monomer" in sub.columns else "unknown"
        raise ValueError(f"Reference run '{ref_run}' not found for monomer '{monomer}'.")

    row = run_sub.iloc[0]
    return row, f"{row['Run']} | {row['Method']}"

def build_dEabs_table(
    df: pd.DataFrame,
    target_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Build an "absolute" dE table relative to the 0.0 scan point:
    dEabs columns are named like:
      dE_0.0_m0.2, dE_0.0_m0.1, dE_0.0_0.1, ... dE_0.0_0.6
    for each (Phase, Run, Method, Monomer).
    """
    if df.empty:
        return df

    if target_values is None:
        target_values = _get_target_values_from_energy_df(df)

    rows: List[Dict] = []

    for (phase, run_name, method_name, monomer), g in df.groupby(
        ["Phase", "Run", "Method", "Monomer"]
    ):
        g = g.sort_values("scan_value").reset_index(drop=True)
        scan_vals = g["scan_value"].to_numpy()
        energies = g["E_tot"].to_numpy()

        # Reference energy: first scan point with scan_value == 0.0
        ref_idx_arr = np.where(np.isclose(scan_vals, 0.0))[0]
        ref_energy = None
        if len(ref_idx_arr) > 0:
            ref_energy = float(energies[int(ref_idx_arr[0])])

        row: Dict[str, float] = {
            "Phase": phase,
            "Run": run_name,
            "Method": method_name,
            "Monomer": monomer,
        }

        for v in target_values:
            label = _format_scan_label(v)
            col_name = f"dE_0.0_{label}"

            if ref_energy is None:
                row[col_name] = np.nan
                continue

            target_idx_arr = np.where(np.isclose(scan_vals, v))[0]
            if len(target_idx_arr) == 0:
                row[col_name] = np.nan
            else:
                e_target = float(energies[int(target_idx_arr[0])])
                row[col_name] = e_target - ref_energy

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    wide_df = pd.DataFrame(rows)
    id_cols = ["Phase", "Run", "Method", "Monomer"]
    dE_cols = [f"dE_0.0_{_format_scan_label(v)}" for v in target_values]
    # Enforce requested column order.
    return wide_df[id_cols + dE_cols]


def plot_dEabs_curves(
    dEabs_df: pd.DataFrame,
    target_values: Optional[List[float]] = None,
    runs_to_plot: Optional[List[str]] = None,
    output_file: str = "dEabs_curves.png",
) -> None:
    """
    Plot dEabs curves for each monomer, with one line per method.
    """
    if dEabs_df.empty:
        print("Skip plotting dEabs curves: empty dEabs table.")
        return

    if target_values is None:
        target_values = _get_target_values_from_dEabs_df(dEabs_df)

    plot_df = dEabs_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting dEabs curves: no rows after Run filter.")
            return

    x_vals = sorted(set([0.0] + target_values))

    monomers = _ordered_monomers(plot_df["Monomer"].dropna().unique().tolist())
    if not monomers:
        print("Skip plotting dEabs curves: no monomer rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(monomers))

    for i, monomer in enumerate(monomers):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Monomer"] == monomer].copy()
        if sub.empty:
            ax.set_title(f"{monomer} (no data)")
            ax.grid(alpha=0.3)
            continue

        for _, row in sub.iterrows():
            y_vals = []
            for x in x_vals:
                if np.isclose(x, 0.0):
                    y_vals.append(0.0)
                    continue
                c = f"dE_0.0_{_format_scan_label(x)}"
                y_vals.append(float(row[c]) if pd.notna(row[c]) else np.nan)
            label = f"{row['Run']} | {row['Method']}"
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, markersize=4, label=label)

        ax.set_title(monomer)
        ax.set_xlabel("scan value")
        ax.set_ylabel("dEabs (kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(monomers), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("dEabs Curves by Monomer and Method", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"dEabs curve figure saved to {output_file}")


def plot_dEabs_delta_vs_reference(
    dEabs_df: pd.DataFrame,
    target_values: Optional[List[float]] = None,
    runs_to_plot: Optional[List[str]] = None,
    ref_run: str = REF_RUN,
    output_file: str = "dEabs_delta_vs_reference.png",
) -> None:
    """
    Plot delta curves: dEabs(method) - dEabs(reference), per monomer.
    """
    if dEabs_df.empty:
        print("Skip plotting delta dEabs curves: empty dEabs table.")
        return

    if target_values is None:
        target_values = _get_target_values_from_dEabs_df(dEabs_df)

    plot_df = dEabs_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting delta dEabs curves: no rows after Run filter.")
            return

    if ref_run not in plot_df["Run"].unique():
        raise ValueError(
            f"Reference run '{ref_run}' not found in dEabs data. "
            "Please update REF_RUN at the top of the script."
        )

    x_vals = sorted(set([0.0] + target_values))
    monomers = _ordered_monomers(plot_df["Monomer"].dropna().unique().tolist())
    if not monomers:
        print("Skip plotting delta dEabs curves: no monomer rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(monomers))

    for i, monomer in enumerate(monomers):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Monomer"] == monomer].copy()
        if sub.empty:
            ax.set_title(f"{monomer} (no data)")
            ax.grid(alpha=0.3)
            continue

        ref_row, ref_label = _select_reference_row(sub, ref_run)

        ref_y_vals = []
        for x in x_vals:
            if np.isclose(x, 0.0):
                ref_y_vals.append(0.0)
                continue
            c = f"dE_0.0_{_format_scan_label(x)}"
            ref_y_vals.append(float(ref_row[c]) if pd.notna(ref_row[c]) else np.nan)
        ref_y = np.array(ref_y_vals, dtype=float)

        for _, row in sub.iterrows():
            y_vals = []
            for x in x_vals:
                if np.isclose(x, 0.0):
                    y_vals.append(0.0)
                    continue
                c = f"dE_0.0_{_format_scan_label(x)}"
                y_vals.append(float(row[c]) if pd.notna(row[c]) else np.nan)
            y = np.array(y_vals, dtype=float)
            delta = y - ref_y
            label = f"{row['Run']} | {row['Method']}"
            ax.plot(x_vals, delta, marker="o", linewidth=1.8, markersize=4, label=label)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{monomer} (ref: {ref_label})")
        ax.set_xlabel("scan value")
        ax.set_ylabel("delta dEabs vs reference (kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(monomers), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Delta dEabs Curves Relative to Reference", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Delta dEabs curve figure saved to {output_file}")

def main() -> None:
    df = collect_monomer_energies(base_dir="SP_init")

    if df.empty:
        print("No monomer scan data found.")
        return

    target_values = _get_target_values_from_energy_df(df)

    # Absolute dE relative to scan_value == 0.0
    dEabs_df = build_dEabs_table(df, target_values=target_values)
    if not dEabs_df.empty:
        output_file_abs = "dEabs.csv"
        dEabs_df.to_csv(output_file_abs, index=False)
        print(f"\nAbsolute dE table saved to {output_file_abs}")
        print("Columns: Phase, Run, Method, Monomer, then dE_0.0_* columns")
        print("\nAbsolute dE relative to scan_value == 0.0 (wide format):")
        print(dEabs_df.to_string(index=False))
        plot_dEabs_curves(dEabs_df, target_values=target_values, runs_to_plot=PLOT_RUNS)
        plot_dEabs_delta_vs_reference(
            dEabs_df,
            target_values=target_values,
            runs_to_plot=PLOT_RUNS,
        )

if __name__ == "__main__":
    main()
