#!/usr/bin/env python3
import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Make project root (where analib.py lives) importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import get_complex_energies, parse_complex_composition  # type: ignore

# Runs to include in dEabs curve plotting.
# Set to None to include all runs in dEabs_df.
PLOT_RUNS: Optional[List[str]] = None
# Reference run used by delta plot. Change this manually as needed.
# If this run is absent from the selected data, the script will raise an error.
REF_RUN = "run102"
FULL_DIR_PATTERN = re.compile(
    r"^(?P<complex>\d+Zn_\d+MImH?_\d+MeOH)(?:_(?P<scan>m?\d+(?:\.\d+)?))?_full$"
)


def _parse_scan_label(label: Optional[str]) -> float:
    if label is None:
        return 0.0
    if label.startswith("m"):
        return -float(label[1:])
    return float(label)


def _parse_full_dir(dirname: str) -> Tuple[str, float, str]:
    """
    dirname examples:
      1Zn_1MIm_0MeOH_full
      1Zn_1MIm_0MeOH_m0.3_full
      1Zn_1MImH_0MeOH_0.2_full

    Return (complex_name, numeric_scan_displacement, suffix_for_lookup).
    """
    match = FULL_DIR_PATTERN.fullmatch(dirname)
    if match is None:
        raise ValueError(f"Unexpected directory name for Zn scan: {dirname}")

    complex_name = match.group("complex")
    scan_label = match.group("scan")
    scan_value = _parse_scan_label(scan_label)
    suffix = "" if scan_label is None else f"_{scan_label}"
    return complex_name, scan_value, suffix


def _complex_sort_key(complex_name: str) -> Tuple[int, int, str]:
    comp = parse_complex_composition(complex_name)
    if comp is None:
        return (10**9, 10**9, complex_name)
    return (comp["MIm"], -comp["MeOH"], complex_name)


def _ordered_complexes(complexes: List[str]) -> List[str]:
    return sorted(complexes, key=_complex_sort_key)


def collect_scan_energies(base_dir: str = "SP_init") -> pd.DataFrame:
    """
    Process Zn scans under the current directory:
    - Traverse all ``*_full`` directories under ``base_dir/run*``
    - Auto-detect complex names from directory names like
      ``<complex>_full`` or ``<complex>_<scan>_full``
    - For each complex, scan point, and run, collect both raw and
      BSSE-corrected energies when BSSE data is complete.
    """
    results: List[Dict[str, object]] = []

    if not os.path.exists(base_dir):
        return pd.DataFrame()

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run*")))

    for run_path in run_dirs:
        run_name = os.path.basename(run_path)

        method_name = run_name
        notes_path = os.path.join(run_path, "notes.yaml")
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r") as f:
                    notes = yaml.safe_load(f) or {}
                method_name = str(notes.get("name", run_name))
            except Exception:
                method_name = run_name

        for full_dir in sorted(glob.glob(os.path.join(run_path, "*_full"))):
            if not os.path.isdir(full_dir):
                continue

            dirname = os.path.basename(full_dir)
            try:
                complex_name, scan_value, suffix = _parse_full_dir(dirname)
            except ValueError:
                continue

            e_raw, e_bsse, has_bsse = get_complex_energies(
                run_path,
                complex_name,
                suffix=suffix,
            )
            if e_raw is None:
                continue

            common = {
                "Phase": base_dir,
                "Run": run_name,
                "Method": method_name,
                "Complex": complex_name,
                "StructDir": dirname,
                "scan_value": scan_value,
            }

            results.append(
                {
                    **common,
                    "BSSE": "no",
                    "E_tot": e_raw,
                }
            )

            if has_bsse and e_bsse is not None:
                results.append(
                    {
                        **common,
                        "BSSE": "yes",
                        "E_tot": e_bsse,
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.sort_values(
        by=["Complex", "Run", "BSSE", "scan_value"], inplace=True, ignore_index=True
    )
    return df

def add_relative_energies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Complex, Run, BSSE):
    - Use the structure with scan_value == 0.0 as the energy reference (if exists),
      otherwise use the minimum energy point as reference,
      and calculate E_rel_0 = E_i - E_ref
    - Calculate energy difference to previous scan point: dE_prev = E_i - E_{i-1}
      (sorted by scan_value in ascending order, within each monomer and run)
    """
    if df.empty:
        return df

    df = df.copy()

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy without the group-by index attached,
        # to avoid pandas FutureWarning about including grouping columns.
        g = g.copy()
        g = g.sort_values("scan_value").reset_index(drop=True)

        # reference for E_rel_0
        zero_rows = g.index[np.isclose(g["scan_value"].values, 0.0)]
        if len(zero_rows) > 0:
            ref_idx = zero_rows[0]
        else:
            ref_idx = g["E_tot"].idxmin()

        ref_energy = g.loc[ref_idx, "E_tot"]
        g["E_rel_0"] = g["E_tot"] - ref_energy

        # neighbor differences
        g["dE_prev"] = g["E_tot"].diff()

        return g

    df = (
        df.groupby(["Complex", "Run", "BSSE"], group_keys=False, sort=False)
        .apply(_per_group, include_groups=True)
        .reset_index(drop=True)
    )
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
        complex_name = str(sub["Complex"].iloc[0]) if "Complex" in sub.columns else "unknown"
        raise ValueError(f"Reference run '{ref_run}' not found for complex '{complex_name}'.")

    if "BSSE" in run_sub.columns:
        no_bsse = run_sub[run_sub["BSSE"] == "no"]
        if not no_bsse.empty:
            run_sub = no_bsse

    row = run_sub.iloc[0]
    label = f"{row['Run']} | {row['Method']}"
    if "BSSE" in row.index:
        label += f" | BSSE={row['BSSE']}"
    return row, label


def build_pairwise_dE_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Phase, Run, BSSE, Complex), create columns like:
    - dE_m0.2_m0.1, dE_m0.1_0.0, dE_0.0_0.1, ...
      where each value is E(scan_j) - E(scan_i) between neighboring scan points.

    The returned DataFrame has one row per (Phase, Run, BSSE, Complex),
    with many dE_*_* columns.
    """
    if df.empty:
        return df

    rows: List[Dict] = []

    for (phase, run_name, method_name, bsse, complex_name), g in df.groupby(
        ["Phase", "Run", "Method", "BSSE", "Complex"]
    ):
        g_sorted = g.sort_values("scan_value")
        scan_vals = g_sorted["scan_value"].to_numpy()
        energies = g_sorted["E_tot"].to_numpy()

        row: Dict[str, float] = {
            "Phase": phase,
            "Run": run_name,
            "Method": method_name,
            "BSSE": bsse,
            "Complex": complex_name,
        }

        # Neighbor differences, one column per consecutive pair
        for i in range(1, len(scan_vals)):
            v_prev = float(scan_vals[i - 1])
            v_curr = float(scan_vals[i])
            label_prev = _format_scan_label(v_prev)
            label_curr = _format_scan_label(v_curr)
            col_name = f"dE_{label_prev}_{label_curr}"
            row[col_name] = float(energies[i] - energies[i - 1])

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    wide_df = pd.DataFrame(rows)
    # Order columns: identifiers first, then all dE_* columns sorted by name
    id_cols = ["Phase", "Run", "Method", "BSSE", "Complex"]
    dE_cols = sorted([c for c in wide_df.columns if c not in id_cols])
    return wide_df[id_cols + dE_cols]

def build_dEabs_table(
    df: pd.DataFrame,
    target_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Build an "absolute" dE table relative to the 0.0 scan point:
    dEabs columns are named like:
      dE_0.0_m0.2, dE_0.0_m0.1, dE_0.0_0.1, ... dE_0.0_0.6
    for each (Phase, Run, Method, BSSE, Complex).
    """
    if df.empty:
        return df

    if target_values is None:
        target_values = _get_target_values_from_energy_df(df)

    rows: List[Dict] = []

    for (phase, run_name, method_name, bsse, complex_name), g in df.groupby(
        ["Phase", "Run", "Method", "BSSE", "Complex"]
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
            "BSSE": bsse,
            "Complex": complex_name,
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
    id_cols = ["Phase", "Run", "Method", "BSSE", "Complex"]
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
    Plot dEabs curves for each complex, with one line per method / BSSE choice.
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

    complexes = _ordered_complexes(plot_df["Complex"].dropna().unique().tolist())
    if not complexes:
        print("Skip plotting dEabs curves: no complex rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(complexes))

    for i, complex_name in enumerate(complexes):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Complex"] == complex_name].copy()
        if sub.empty:
            ax.set_title(f"{complex_name} (no data)")
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
            label = f"{row['Run']} | {row['Method']} | BSSE={row['BSSE']}"
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, markersize=4, label=label)

        ax.set_title(complex_name)
        ax.set_xlabel("scan value")
        ax.set_ylabel("dEabs (kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(complexes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("dEabs Curves by Complex and Method", fontsize=14)
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
    Plot delta curves: dEabs(method) - dEabs(reference), per complex.
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
    complexes = _ordered_complexes(plot_df["Complex"].dropna().unique().tolist())
    if not complexes:
        print("Skip plotting delta dEabs curves: no complex rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(complexes))

    for i, complex_name in enumerate(complexes):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Complex"] == complex_name].copy()
        if sub.empty:
            ax.set_title(f"{complex_name} (no data)")
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
            label = f"{row['Run']} | {row['Method']} | BSSE={row['BSSE']}"
            ax.plot(x_vals, delta, marker="o", linewidth=1.8, markersize=4, label=label)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{complex_name} (ref: {ref_label})")
        ax.set_xlabel("scan value")
        ax.set_ylabel("delta dEabs vs reference (kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(complexes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Delta dEabs Curves Relative to Reference", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Delta dEabs curve figure saved to {output_file}")

def main() -> None:
    df = collect_scan_energies(base_dir="SP_init")

    if df.empty:
        print("No Zn scan data found.")
        return

    target_values = _get_target_values_from_energy_df(df)

    # Absolute dE relative to scan_value == 0.0
    dEabs_df = build_dEabs_table(df, target_values=target_values)
    if not dEabs_df.empty:
        output_file_abs = "dEabs.csv"
        dEabs_df.to_csv(output_file_abs, index=False)
        print(f"\nAbsolute dE table saved to {output_file_abs}")
        print("Columns: Phase, Run, Method, BSSE, Complex, then dE_0.0_* columns")
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
