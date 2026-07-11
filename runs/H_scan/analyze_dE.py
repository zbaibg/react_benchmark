#!/usr/bin/env python3
import os
import glob
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Make project root (where analib.py lives) importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analib import get_Etot_amber  # type: ignore

MONOMERS = ["MImH", "H3O", "Wat", "MeOH"]
# Runs to include in dEabs curve plotting.
# Set to None to include all runs in dEabs_df.
PLOT_RUNS: Optional[List[str]] = ["run23", "run24", "run33", "run8",'run16','run28','run34','run35']
# Reference used by delta plot: each curve minus run24 DLPNO-CCSD(T).
REF_RUN = "run24"
REF_METHOD_KEYWORD = "DLPNO-CCSD(T)"

def _parse_scan_value(monomer: str, dirname: str) -> float:
    """
    dirname examples: MImH_0.2_monomer, MImH_m0.2_monomer
    Return the numeric scan displacement (e.g. 0.2, -0.2).
    """
    prefix = f"{monomer}_"
    suffix = "_monomer"
    if not (dirname.startswith(prefix) and dirname.endswith(suffix)):
        raise ValueError(f"Unexpected directory name for scan: {dirname}")
    middle = dirname[len(prefix): -len(suffix)]
    if middle.startswith("m"):
        middle = "-" + middle[1:]
    return float(middle)

def collect_monomer_energies(base_dir: str = "SP_init") -> pd.DataFrame:
    """
    Process monomer scans under the current H_scan directory only:
    - Traverse all *monomer directories under base_dir/run*/
    - For each monomer (MImH, H3O, Wat, MeOH) and each run,
      collect total energy for each scan value.
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

        for monomer in MONOMERS:
            pattern = os.path.join(run_path, f"{monomer}_*_monomer")
            for struct_dir in sorted(glob.glob(pattern)):
                dirname = os.path.basename(struct_dir)

                try:
                    scan_value = _parse_scan_value(monomer, dirname)
                except Exception:
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

def add_relative_energies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Monomer, Run):
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
        df.groupby(["Monomer", "Run"], group_keys=False, sort=False)
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


def build_pairwise_dE_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Phase, Run, Monomer), create columns like:
    - dE_m0.2_m0.1, dE_m0.1_0.0, dE_0.0_0.1, ...
      where each value is E(scan_j) - E(scan_i) between neighboring scan points.

    The returned DataFrame has one row per (Phase, Run, Monomer),
    with many dE_*_* columns.
    """
    if df.empty:
        return df

    rows: List[Dict] = []

    for (phase, run_name, method_name, monomer), g in df.groupby(
        ["Phase", "Run", "Method", "Monomer"]
    ):
        g_sorted = g.sort_values("scan_value")
        scan_vals = g_sorted["scan_value"].to_numpy()
        energies = g_sorted["E_tot"].to_numpy()

        row: Dict[str, float] = {
            "Phase": phase,
            "Run": run_name,
            "Method": method_name,
            "Monomer": monomer,
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
    id_cols = ["Phase", "Run", "Method", "Monomer"]
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
    for each (Phase, Run, Method, Monomer).
    """
    if df.empty:
        return df

    if target_values is None:
        # Match the values requested by the user.
        target_values = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

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
        target_values = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    plot_df = dEabs_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting dEabs curves: no rows after Run filter.")
            return

    x_vals = sorted([0.0] + target_values)

    monomers = [m for m in MONOMERS if m in plot_df["Monomer"].unique()]
    if not monomers:
        print("Skip plotting dEabs curves: no monomer rows found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

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
    ref_method_keyword: str = REF_METHOD_KEYWORD,
    output_file: str = "dEabs_delta_vs_run24_ccsdt.png",
) -> None:
    """
    Plot delta curves: dEabs(method) - dEabs(reference), per monomer.
    """
    if dEabs_df.empty:
        print("Skip plotting delta dEabs curves: empty dEabs table.")
        return

    if target_values is None:
        target_values = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    plot_df = dEabs_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting delta dEabs curves: no rows after Run filter.")
            return

    x_vals = sorted([0.0] + target_values)
    monomers = [m for m in MONOMERS if m in plot_df["Monomer"].unique()]
    if not monomers:
        print("Skip plotting delta dEabs curves: no monomer rows found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, monomer in enumerate(monomers):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Monomer"] == monomer].copy()
        if sub.empty:
            ax.set_title(f"{monomer} (no data)")
            ax.grid(alpha=0.3)
            continue

        ref_sub = sub[(sub["Run"] == ref_run) & (sub["Method"].str.contains(ref_method_keyword, regex=False))]
        if ref_sub.empty:
            ax.set_title(f"{monomer} (missing reference)")
            ax.grid(alpha=0.3)
            continue

        ref_row = ref_sub.iloc[0]
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
        ax.set_title(monomer)
        ax.set_xlabel("scan value")
        ax.set_ylabel("delta dEabs vs run24 CCSD(T) (kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(monomers), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Delta dEabs Curves Relative to run24 DLPNO-CCSD(T)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Delta dEabs curve figure saved to {output_file}")

def main() -> None:
    df = collect_monomer_energies(base_dir="SP_init")

    if df.empty:
        print("No monomer scan data found.")
        return

    # Wide-format table: one row per (Phase, Run, Monomer),
    # with columns like dE_m0.2_m0.1, dE_m0.1_0.0, ...
    wide_df = build_pairwise_dE_table(df)
    if not wide_df.empty:
        output_file_wide = "dE.csv"
        wide_df.to_csv(output_file_wide, index=False)
        print(f"\nPairwise dE table saved to {output_file_wide}")
        print("Columns:")
        print("  Phase, Run, Method, Monomer, then many dE_xxx_yyy columns")
        print("\nPairwise neighbor dE (wide format):")
        print(wide_df.to_string(index=False))

    # Absolute dE relative to scan_value == 0.0
    dEabs_df = build_dEabs_table(df)
    if not dEabs_df.empty:
        output_file_abs = "dEabs.csv"
        dEabs_df.to_csv(output_file_abs, index=False)
        print(f"\nAbsolute dE table saved to {output_file_abs}")
        print("Columns: Phase, Run, Method, Monomer, then dE_0.0_* columns")
        print("\nAbsolute dE relative to scan_value == 0.0 (wide format):")
        print(dEabs_df.to_string(index=False))
        plot_dEabs_curves(dEabs_df, runs_to_plot=PLOT_RUNS)
        plot_dEabs_delta_vs_reference(dEabs_df, runs_to_plot=PLOT_RUNS)

if __name__ == "__main__":
    main()
