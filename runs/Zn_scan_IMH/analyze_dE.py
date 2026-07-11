#!/usr/bin/env python3
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Make the shared analysis helpers importable.
SCRIPT_DIR = Path(__file__).resolve().parent
QMMM_ROOT = SCRIPT_DIR.parents[1]
PYTHON_SCRIPTS_DIR = QMMM_ROOT / "python_scripts"
for import_dir in (PYTHON_SCRIPTS_DIR, QMMM_ROOT):
    import_dir_str = str(import_dir)
    if import_dir_str not in sys.path:
        sys.path.insert(0, import_dir_str)

from analib import get_complex_energies, parse_complex_composition  # type: ignore

# Runs to include in the analysis.
# Set to None to include all available runs.
# Example: RUNS_TO_ANALYZE = ["run0", "run110", "run112"]
#RUNS_TO_ANALYZE: Optional[List[str]] = ['run0','run113','run115','run110']
#RUNS_TO_ANALYZE: Optional[List[str]] = ['run0','run118','run117','run41']
RUNS_TO_ANALYZE: Optional[List[str]] = ['run8','run110','run41']

# Reference run used by delta plot. Change this manually as needed.
# If this run is absent from the selected data, the script will raise an error.
REF_RUN = "run0"

# Set to False to omit "BSSE corr=..." from plot legends.
SHOW_BSSE_CORR_IN_LEGEND = False

FULL_DIR_PATTERN = re.compile(
    r"^(?P<complex>(?:\d+Zn_)?(?:\d+ImH_)?\d+Wat_\d+Hbond)(?:_IMH_(?P<scan>m?\d+(?:\.\d+)?))?_full$"
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
      1Zn_1ImH_6Wat_1Hbond_full
      1Zn_1ImH_6Wat_1Hbond_IMH_m0.2_full
      1Zn_1ImH_6Wat_2Hbond_IMH_0.5_full
      1ImH_6Wat_1Hbond_IMH_0.5_full

    Return (complex_name, numeric_scan_displacement, suffix_for_lookup).
    """
    match = FULL_DIR_PATTERN.fullmatch(dirname)
    if match is None:
        raise ValueError(f"Unexpected directory name for Zn scan: {dirname}")

    complex_name = match.group("complex")
    scan_label = match.group("scan")
    scan_value = _parse_scan_label(scan_label)
    suffix = "" if scan_label is None else f"_IMH_{scan_label}"
    return complex_name, scan_value, suffix


def _xyz_filename(complex_name: str, scan_value: float) -> str:
    if np.isclose(scan_value, 0.0):
        return f"{complex_name}.xyz"
    return f"{complex_name}_IMH_{_format_scan_label(scan_value)}.xyz"


def _with_zn_complex_for_distance(complex_name: str) -> str:
    if re.match(r"\d+Zn_", complex_name):
        return complex_name
    if re.match(r"\d+ImH_\d+Wat_\d+Hbond$", complex_name):
        return f"1Zn_{complex_name}"
    return complex_name


def _read_xyz_atoms(path: Path) -> List[Tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must be atom count") from exc

    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, found {len(atom_lines)}")

    atoms: List[Tuple[str, float, float, float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line in {path}: {line}")
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _nearest_zn_n_distance_from_xyz(path: Path) -> float:
    atoms = _read_xyz_atoms(path)
    zn_atoms = [(x, y, z) for elem, x, y, z in atoms if elem.upper() == "ZN"]
    n_atoms = [(x, y, z) for elem, x, y, z in atoms if elem.upper() == "N"]
    if not zn_atoms:
        raise ValueError(f"No Zn atom found in {path}")
    if not n_atoms:
        raise ValueError(f"No N atom found in {path}")

    zn = zn_atoms[0]
    return min(_distance(zn, n) for n in n_atoms)


def build_zn_n_distance_map(
    complexes: List[str],
    scan_values: List[float],
    xyz_dir: Path = SCRIPT_DIR / "xyz" / "xyz_files",
) -> Dict[Tuple[str, float], float]:
    """
    Map each complex/scan point to the nearest Zn-N distance from xyz/xyz_files.

    Complexes without Zn reuse the matching pre-deletion Zn structure, e.g.
    1ImH_6Wat_1Hbond uses 1Zn_1ImH_6Wat_1Hbond at the same scan value.
    """
    distances: Dict[Tuple[str, float], float] = {}
    cache: Dict[Tuple[str, float], float] = {}

    for complex_name in complexes:
        with_zn_complex = _with_zn_complex_for_distance(complex_name)
        for scan_value in scan_values:
            cache_key = (with_zn_complex, float(scan_value))
            if cache_key not in cache:
                xyz_path = xyz_dir / _xyz_filename(with_zn_complex, float(scan_value))
                if not xyz_path.exists():
                    continue
                cache[cache_key] = _nearest_zn_n_distance_from_xyz(xyz_path)
            distances[(complex_name, float(scan_value))] = cache[cache_key]

    return distances


def add_zn_n_distances(
    df: pd.DataFrame,
    distance_map: Dict[Tuple[str, float], float],
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["Zn_N_distance"] = [
        distance_map.get((str(complex_name), float(scan_value)), np.nan)
        for complex_name, scan_value in zip(df["Complex"], df["scan_value"])
    ]
    return df


def _complex_sort_key(complex_name: str) -> Tuple[object, ...]:
    hbond_match = re.match(
        r"(?:(\d+)Zn_)?(?:(\d+)ImH_)?(\d+)Wat_(\d+)Hbond$",
        complex_name,
    )
    if hbond_match:
        return (
            int(hbond_match.group(1) or 0),
            int(hbond_match.group(2) or 0),
            int(hbond_match.group(4)),
            complex_name,
        )

    comp = parse_complex_composition(complex_name)
    if comp is None:
        return (10**9, 10**9, complex_name)
    return (comp["MIm"], -comp["MeOH"], complex_name)


def _ordered_complexes(complexes: List[str]) -> List[str]:
    return sorted(complexes, key=_complex_sort_key)


def _run_sort_key(run: str, requested_order: Optional[List[str]] = None) -> Tuple[int, object]:
    requested_order = RUNS_TO_ANALYZE if requested_order is None else requested_order
    requested_index = {
        str(requested_run): i for i, requested_run in enumerate(requested_order or [])
    }
    run_str = str(run)
    if run_str in requested_index:
        return (0, requested_index[run_str])
    if re.fullmatch(r"run\d+", run_str):
        return (1, int(run_str[3:]))
    return (2, run_str)


def _ordered_runs(
    runs: List[str],
    requested_order: Optional[List[str]] = None,
) -> List[str]:
    unique_runs = list(dict.fromkeys(str(run) for run in runs))
    return sorted(unique_runs, key=lambda run: _run_sort_key(run, requested_order))


def _run_style_order(
    runs: List[str],
    requested_order: Optional[List[str]] = None,
) -> List[str]:
    requested_order = RUNS_TO_ANALYZE if requested_order is None else requested_order
    if requested_order is None:
        return _ordered_runs(runs, requested_order=None)

    requested_runs = list(dict.fromkeys(str(run) for run in requested_order))
    requested_set = set(requested_runs)
    extras = _ordered_runs(
        [str(run) for run in runs if str(run) not in requested_set],
        requested_order=[],
    )
    return requested_runs + extras


def _sort_by_run_order(
    df: pd.DataFrame,
    run_order: List[str],
) -> pd.DataFrame:
    if df.empty or "Run" not in df.columns:
        return df

    order_index = {run: i for i, run in enumerate(run_order)}
    out = df.copy()
    out["_run_order"] = out["Run"].astype(str).map(order_index)
    out.sort_values("_run_order", kind="stable", inplace=True)
    return out.drop(columns="_run_order")


def _load_run_names(config_path: Path = SCRIPT_DIR / "run_configs.yaml") -> Dict[str, str]:
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return {}

    run_names: Dict[str, str] = {}
    for key, value in config.items():
        if not re.fullmatch(r"run\d+", str(key)):
            continue
        if isinstance(value, dict) and value.get("name"):
            run_names[str(key)] = str(value["name"])
    return run_names


def collect_scan_energies(
    base_dir: str = "SP_init",
    runs_to_analyze: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process IMH translation scans under the current directory:
    - Traverse all ``*_full`` directories under ``base_dir/run*``
    - Auto-detect complex names from directory names like
      ``<complex>_full`` or ``<complex>_IMH_<scan>_full``
    - For each complex, scan point, and run, collect both raw and
      BSSE-corrected energies when BSSE data is complete.
    """
    results: List[Dict[str, object]] = []

    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = SCRIPT_DIR / base_path
    phase_name = base_path.name

    if not base_path.exists():
        return pd.DataFrame()

    run_names = _load_run_names()
    run_dirs = sorted(glob.glob(str(base_path / "run*")))
    if runs_to_analyze is not None:
        selected_runs = set(runs_to_analyze)
        run_dirs = [
            run_dir
            for run_dir in run_dirs
            if os.path.basename(run_dir) in selected_runs
        ]

    for run_path in run_dirs:
        run_name = os.path.basename(run_path)

        method_name = run_names.get(run_name, run_name)
        notes_path = os.path.join(run_path, "notes.yaml")
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r") as f:
                    notes = yaml.safe_load(f) or {}
                method_name = str(notes.get("name", method_name))
            except Exception:
                pass

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
                "Phase": phase_name,
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


def _make_axes_grid(n_panels: int, sharey: bool = True) -> Tuple[plt.Figure, np.ndarray]:
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5 * nrows),
        sharex=True,
        sharey=sharey,
    )
    return fig, np.atleast_1d(axes).ravel()


def _add_figure_legend(fig: plt.Figure, axes: np.ndarray, ncols: int = 2) -> None:
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=8,
            ncols=ncols,
            frameon=False,
        )


def _plot_legend_label(parts: List[object], bsse: Optional[object] = None) -> str:
    label_parts = [str(part) for part in parts]
    if SHOW_BSSE_CORR_IN_LEGEND and bsse is not None:
        label_parts.append(f"BSSE corr={bsse}")
    return " | ".join(label_parts)


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
    label = _plot_legend_label(
        [row["Run"], row["Method"]],
        row["BSSE"] if "BSSE" in row.index else None,
    )
    return row, label


def _shift_curve_to_xmax_zero(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    """
    Shift a curve so the finite point at the largest x value has y == 0.
    """
    shifted = y_vals.astype(float, copy=True)
    finite_mask = np.isfinite(x_vals) & np.isfinite(shifted)
    if not finite_mask.any():
        return shifted

    finite_indices = np.where(finite_mask)[0]
    ref_idx = finite_indices[np.argmax(x_vals[finite_indices])]
    return shifted - shifted[ref_idx]


def _short_hbond_complex_label(complex_name: str) -> str:
    hbond_match = re.search(r"(\d+Hbond)$", complex_name)
    if not hbond_match:
        return complex_name
    zn_label = "with Zn" if re.match(r"\d+Zn_", complex_name) else "without Zn"
    return f"{zn_label} {hbond_match.group(1)}"


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
    distance_map: Optional[Dict[Tuple[str, float], float]] = None,
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

    scan_vals = sorted(set([0.0] + target_values))
    run_order = _ordered_runs(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    run_style_order = _run_style_order(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    plot_df = _sort_by_run_order(plot_df, run_order)

    complexes = _ordered_complexes(plot_df["Complex"].dropna().unique().tolist())
    if not complexes:
        print("Skip plotting dEabs curves: no complex rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(complexes), sharey=False)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_run = {
        run: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, run in enumerate(run_style_order)
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_run = {
        run: marker_cycle[i % len(marker_cycle)] for i, run in enumerate(run_style_order)
    }

    for i, complex_name in enumerate(complexes):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Complex"] == complex_name].copy()
        if sub.empty:
            ax.set_title(f"{complex_name} (no data)")
            ax.grid(alpha=0.3)
            continue

        x_vals = np.array(
            [
                (
                    distance_map.get((complex_name, float(scan_value)), np.nan)
                    if distance_map is not None
                    else float(scan_value)
                )
                for scan_value in scan_vals
            ],
            dtype=float,
        )
        order = np.argsort(x_vals)

        for _, row in sub.iterrows():
            y_vals = []
            for scan_value in scan_vals:
                if np.isclose(scan_value, 0.0):
                    y_vals.append(0.0)
                    continue
                c = f"dE_0.0_{_format_scan_label(scan_value)}"
                y_vals.append(float(row[c]) if pd.notna(row[c]) else np.nan)
            y = _shift_curve_to_xmax_zero(x_vals, np.array(y_vals, dtype=float))
            label = _plot_legend_label([row["Run"], row["Method"]], row["BSSE"])
            ax.plot(
                x_vals[order],
                y[order],
                marker=marker_by_run.get(row["Run"], "o"),
                color=color_by_run.get(row["Run"]),
                linestyle="--" if row["BSSE"] == "yes" else "-",
                alpha=0.85,
                linewidth=1.8,
                markersize=4,
                label=label,
            )

        ax.set_title(_short_hbond_complex_label(complex_name), fontsize=10)
        ax.set_xlabel("nearest Zn-N distance (Angstrom)")
        ax.set_ylabel("dEabs - dEabs(max x) (kcal/mol)")
        ax.grid(alpha=0.3)

    for j in range(len(complexes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("dEabs Curves by Complex and Method, Shifted to Max x", fontsize=14)
    _add_figure_legend(fig, axes_flat, ncols=1)
    fig.tight_layout(rect=[0, 0.24, 1, 0.95])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"dEabs curve figure saved to {output_file}")


def plot_dEabs_delta_vs_reference(
    dEabs_df: pd.DataFrame,
    target_values: Optional[List[float]] = None,
    runs_to_plot: Optional[List[str]] = None,
    ref_run: str = REF_RUN,
    distance_map: Optional[Dict[Tuple[str, float], float]] = None,
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
        print(
            f"Skip plotting delta dEabs curves: reference run '{ref_run}' "
            "not found in parsed dEabs data."
        )
        print("The reference plot will be generated once that run has completed energies.")
        return

    scan_vals = sorted(set([0.0] + target_values))
    run_order = _ordered_runs(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    run_style_order = _run_style_order(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    plot_df = _sort_by_run_order(plot_df, run_order)
    complexes = _ordered_complexes(plot_df["Complex"].dropna().unique().tolist())
    if not complexes:
        print("Skip plotting delta dEabs curves: no complex rows found.")
        return

    fig, axes_flat = _make_axes_grid(len(complexes))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_run = {
        run: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, run in enumerate(run_style_order)
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_run = {
        run: marker_cycle[i % len(marker_cycle)] for i, run in enumerate(run_style_order)
    }

    for i, complex_name in enumerate(complexes):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Complex"] == complex_name].copy()
        if sub.empty:
            ax.set_title(f"{complex_name} (no data)")
            ax.grid(alpha=0.3)
            continue

        if ref_run not in sub["Run"].unique():
            ax.set_title(f"{complex_name} (no {ref_run} reference)")
            ax.grid(alpha=0.3)
            print(
                f"Skip delta panel for {complex_name}: reference run '{ref_run}' "
                "is absent for this complex."
            )
            continue

        ref_row, _ = _select_reference_row(sub, ref_run)

        x_vals = np.array(
            [
                (
                    distance_map.get((complex_name, float(scan_value)), np.nan)
                    if distance_map is not None
                    else float(scan_value)
                )
                for scan_value in scan_vals
            ],
            dtype=float,
        )
        order = np.argsort(x_vals)

        ref_y_vals = []
        for scan_value in scan_vals:
            if np.isclose(scan_value, 0.0):
                ref_y_vals.append(0.0)
                continue
            c = f"dE_0.0_{_format_scan_label(scan_value)}"
            ref_y_vals.append(float(ref_row[c]) if pd.notna(ref_row[c]) else np.nan)
        ref_y = np.array(ref_y_vals, dtype=float)
        ref_y = _shift_curve_to_xmax_zero(x_vals, ref_y)

        nonzero_mask = np.array([not np.isclose(x, 0.0) for x in scan_vals], dtype=bool)
        if not np.isfinite(ref_y[nonzero_mask]).any():
            ax.set_title(f"{complex_name} (no usable {ref_run} reference)")
            ax.grid(alpha=0.3)
            print(
                f"Skip delta panel for {complex_name}: reference run '{ref_run}' "
                "has no finite dE values relative to scan 0.0."
            )
            continue

        for _, row in sub.iterrows():
            y_vals = []
            for scan_value in scan_vals:
                if np.isclose(scan_value, 0.0):
                    y_vals.append(0.0)
                    continue
                c = f"dE_0.0_{_format_scan_label(scan_value)}"
                y_vals.append(float(row[c]) if pd.notna(row[c]) else np.nan)
            y = np.array(y_vals, dtype=float)
            y = _shift_curve_to_xmax_zero(x_vals, y)
            delta = y - ref_y
            label = _plot_legend_label([row["Run"], row["Method"]], row["BSSE"])
            ax.plot(
                x_vals[order],
                delta[order],
                marker=marker_by_run.get(row["Run"], "o"),
                color=color_by_run.get(row["Run"]),
                linestyle="--" if row["BSSE"] == "yes" else "-",
                alpha=0.85,
                linewidth=1.8,
                markersize=4,
                label=label,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(
            f"{_short_hbond_complex_label(complex_name)} (ref: {ref_run})",
            fontsize=10,
        )
        ax.set_xlabel("nearest Zn-N distance (Angstrom)")
        ax.set_ylabel("delta shifted dEabs vs reference (kcal/mol)")
        ax.grid(alpha=0.3)

    for j in range(len(complexes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Delta dEabs Curves Shifted to Max x, Relative to Reference", fontsize=14)
    _add_figure_legend(fig, axes_flat, ncols=1)
    fig.tight_layout(rect=[0, 0.24, 1, 0.95])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Delta dEabs curve figure saved to {output_file}")


def plot_hbond_comparison_relative_to_2hbond_min(
    df: pd.DataFrame,
    runs_to_plot: Optional[List[str]] = None,
    output_file: str = "hbond_comparison_ref_2hbond_min.png",
) -> None:
    """
    Plot 1Hbond and 2Hbond scans on one figure.

    For each (Phase, Run, Method, BSSE), energies are shifted by the minimum
    E_tot among that method's 2Hbond scan points.
    """
    if df.empty:
        print("Skip plotting Hbond comparison: empty energy table.")
        return

    plot_df = df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting Hbond comparison: no rows after Run filter.")
            return

    run_order = _ordered_runs(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    run_style_order = _run_style_order(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    plot_df = _sort_by_run_order(plot_df, run_order)
    complexes = _ordered_complexes(plot_df["Complex"].dropna().unique().tolist())
    hbond_complexes = [c for c in complexes if c.endswith(("1Hbond", "2Hbond"))]
    if not hbond_complexes:
        print("Skip plotting Hbond comparison: no 1Hbond/2Hbond complexes found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_run = {
        run: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, run in enumerate(run_style_order)
    }
    plotted = False

    for (phase, run_name, method_name, bsse), g in plot_df.groupby(
        ["Phase", "Run", "Method", "BSSE"], sort=False
    ):
        ref_rows = g[g["Complex"].str.endswith("2Hbond", na=False)]
        ref_energies = ref_rows["E_tot"].dropna()
        if ref_energies.empty:
            print(
                f"Skip Hbond comparison for {run_name} | BSSE corr={bsse}: "
                "no finite 2Hbond reference energies."
            )
            continue

        ref_energy = float(ref_energies.min())
        color = color_by_run.get(run_name)

        for complex_name in hbond_complexes:
            sort_col = "Zn_N_distance" if "Zn_N_distance" in g.columns else "scan_value"
            sub = g[g["Complex"] == complex_name].sort_values(sort_col)
            if sub.empty:
                continue

            x_col = "Zn_N_distance" if "Zn_N_distance" in sub.columns else "scan_value"
            x_vals = sub[x_col].to_numpy(dtype=float)
            y_vals = sub["E_tot"].to_numpy(dtype=float) - ref_energy
            hbond_label = complex_name.rsplit("_", 1)[-1]
            linestyle = "-" if hbond_label == "2Hbond" else "--"
            marker = "o" if hbond_label == "2Hbond" else "s"
            label = _plot_legend_label([run_name, hbond_label], bsse)
            ax.plot(
                x_vals,
                y_vals,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("Skip plotting Hbond comparison: no plottable curves.")
        return

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("nearest Zn-N distance (Angstrom)")
    ax.set_ylabel("E - min(E 2Hbond for same method) (kcal/mol)")
    ax.set_title("1Hbond vs 2Hbond Scans, Referenced to Each Method's 2Hbond Minimum")
    ax.grid(alpha=0.3)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
        ncols=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Hbond comparison figure saved to {output_file}")


def _without_zn_complex_name(complex_name: str) -> Optional[str]:
    match = re.fullmatch(
        r"\d+Zn_(?P<rest>\d+ImH_\d+Wat_\d+Hbond)",
        complex_name,
    )
    if match is None:
        return None
    return match.group("rest")


def build_zn_minus_no_zn_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match structures such as:
      1Zn_1ImH_6Wat_1Hbond_IMH_0.5_full
      1ImH_6Wat_1Hbond_IMH_0.5_full

    and calculate E(with Zn) - E(without Zn) for the same
    (Phase, Run, Method, BSSE, Hbond, scan_value).
    """
    if df.empty:
        return df

    complex_names = set(df["Complex"].dropna().astype(str).unique())
    rows: List[Dict[str, object]] = []

    for complex_with in _ordered_complexes(list(complex_names)):
        complex_without = _without_zn_complex_name(complex_with)
        if complex_without is None or complex_without not in complex_names:
            continue

        with_df = df[df["Complex"] == complex_with]
        without_df = df[df["Complex"] == complex_without]
        if with_df.empty or without_df.empty:
            continue

        key_cols = ["Phase", "Run", "Method", "BSSE", "scan_value"]
        merged = with_df.merge(
            without_df,
            on=key_cols,
            suffixes=("_with_zn", "_without_zn"),
        )
        if merged.empty:
            continue

        for _, row in merged.iterrows():
            rows.append(
                {
                    "Phase": row["Phase"],
                    "Run": row["Run"],
                    "Method": row["Method"],
                    "BSSE": row["BSSE"],
                    "Complex_with_zn": complex_with,
                    "Complex_without_zn": complex_without,
                    "scan_value": float(row["scan_value"]),
                    "Zn_N_distance": float(row["Zn_N_distance_with_zn"]),
                    "E_with_zn": float(row["E_tot_with_zn"]),
                    "E_without_zn": float(row["E_tot_without_zn"]),
                    "dE_with_zn_minus_without_zn": float(
                        row["E_tot_with_zn"] - row["E_tot_without_zn"]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    curve_cols = [
        "Phase",
        "Run",
        "Method",
        "BSSE",
        "Complex_with_zn",
        "Complex_without_zn",
    ]
    curve_min = out.groupby(curve_cols)["dE_with_zn_minus_without_zn"].transform("min")
    out["dE_shifted_to_min"] = out["dE_with_zn_minus_without_zn"] - curve_min
    curve_max = out.groupby(curve_cols)["dE_with_zn_minus_without_zn"].transform("max")
    out["dE_shifted_to_max"] = out["dE_with_zn_minus_without_zn"] - curve_max
    out.sort_values(
        by=["Complex_with_zn", "Run", "BSSE", "Zn_N_distance"],
        inplace=True,
        ignore_index=True,
    )
    return out


def plot_zn_minus_no_zn_curves(
    diff_df: pd.DataFrame,
    runs_to_plot: Optional[List[str]] = None,
    output_file: str = "with_zn_minus_without_zn_scan.png",
) -> None:
    """
    Plot E(with Zn) - E(without Zn), shifted so each curve's maximum is 0.

    Each Hbond structure is written to a separate figure.
    """
    if diff_df.empty:
        print("Skip plotting with-Zn minus without-Zn curves: empty table.")
        return

    plot_df = diff_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print(
                "Skip plotting with-Zn minus without-Zn curves: "
                "no rows after Run filter."
            )
            return

    complexes = _ordered_complexes(
        plot_df["Complex_with_zn"].dropna().unique().tolist()
    )
    if not complexes:
        print("Skip plotting with-Zn minus without-Zn curves: no paired complexes.")
        return

    run_order = _ordered_runs(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    run_style_order = _run_style_order(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    plot_df = _sort_by_run_order(plot_df, run_order)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_run = {
        run: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, run in enumerate(run_style_order)
    }
    any_plotted = False
    output_path = Path(output_file)
    output_suffix = output_path.suffix or ".png"

    for complex_name in complexes:
        fig, ax = plt.subplots(figsize=(5, 5))
        sub = plot_df[plot_df["Complex_with_zn"] == complex_name].copy()
        plotted = False

        for (_, run_name, method_name, bsse), g in sub.groupby(
            ["Phase", "Run", "Method", "BSSE"],
            sort=False,
        ):
            g = g.sort_values("Zn_N_distance")
            if len(g) < 2:
                print(
                    f"Skip {complex_name} {run_name} | BSSE corr={bsse}: "
                    "fewer than two paired scan points."
                )
                continue
            label = _plot_legend_label([run_name, method_name], bsse)
            ax.plot(
                g["Zn_N_distance"].to_numpy(dtype=float),
                g["dE_shifted_to_max"].to_numpy(dtype=float),
                marker="o",
                color=color_by_run.get(run_name),
                linestyle="--" if bsse == "yes" else "-",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
            plotted = True

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        hbond_match = re.search(r"(\d+Hbond)$", complex_name)
        hbond_label = hbond_match.group(1) if hbond_match else complex_name
        ax.set_title(hbond_label, fontsize=10)
        ax.set_xlabel("Zn-N distance (Angstrom)")
        ax.set_ylabel("dE - max (kcal/mol)")
        ax.grid(alpha=0.3)

        if not plotted:
            plt.close(fig)
            print(f"Skip plotting {complex_name}: no plottable curves.")
            continue

        fig.suptitle(
            "With Zn - Without Zn, Shifted to Max",
            fontsize=12,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            fontsize=8,
            ncols=1,
            frameon=False,
        )

        if len(complexes) > 1:
            complex_output = output_path.with_name(
                f"{output_path.stem}_{hbond_label}{output_suffix}"
            )
        else:
            complex_output = output_path
        fig.tight_layout(rect=[0, 0.24, 1, 0.95])
        fig.savefig(complex_output, dpi=200)
        plt.close(fig)
        any_plotted = True
        print(f"With-Zn minus without-Zn scan figure saved to {complex_output}")

    if not any_plotted:
        print("Skip plotting with-Zn minus without-Zn curves: no plottable curves.")


def build_scan_two_end_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate E(with Zn at x) - E(without Zn at reversed x).

    Within each matched with-Zn / without-Zn complex and each
    (Phase, Run, Method, BSSE), the with-Zn points are sorted by increasing
    Zn-N distance and the without-Zn points are sorted by decreasing Zn-N
    distance. The paired difference curve is then shifted so the final
    maximum-x point is zero.
    """
    if df.empty:
        return df

    complex_names = set(df["Complex"].dropna().astype(str).unique())
    rows: List[Dict[str, object]] = []
    key_cols = ["Phase", "Run", "Method", "BSSE"]

    for complex_with in _ordered_complexes(list(complex_names)):
        complex_without = _without_zn_complex_name(complex_with)
        if complex_without is None or complex_without not in complex_names:
            continue

        with_df = df[df["Complex"] == complex_with]
        without_df = df[df["Complex"] == complex_without]
        if with_df.empty or without_df.empty:
            continue

        for key, with_group in with_df.groupby(key_cols, sort=False):
            without_mask = np.ones(len(without_df), dtype=bool)
            for col, value in zip(key_cols, key):
                without_mask &= without_df[col].to_numpy() == value
            without_group = without_df[without_mask]
            if without_group.empty:
                continue

            with_sorted = (
                with_group.dropna(subset=["Zn_N_distance", "E_tot"])
                .sort_values("Zn_N_distance")
                .reset_index(drop=True)
            )
            without_reversed = (
                without_group.dropna(subset=["Zn_N_distance", "E_tot"])
                .sort_values("Zn_N_distance", ascending=False)
                .reset_index(drop=True)
            )
            n_pairs = min(len(with_sorted), len(without_reversed))
            if n_pairs < 2:
                continue
            if len(with_sorted) != len(without_reversed):
                phase, run_name, method_name, bsse = key
                print(
                    f"Warning: scan_two_end truncates {complex_with} {run_name} "
                    f"| BSSE corr={bsse} from {len(with_sorted)} and "
                    f"{len(without_reversed)} points to {n_pairs} pairs."
                )

            for pair_idx in range(n_pairs):
                with_row = with_sorted.iloc[pair_idx]
                without_row = without_reversed.iloc[pair_idx]
                rows.append(
                    {
                        "Phase": with_row["Phase"],
                        "Run": with_row["Run"],
                        "Method": with_row["Method"],
                        "BSSE": with_row["BSSE"],
                        "Complex_with_zn": complex_with,
                        "Complex_without_zn": complex_without,
                        "scan_value_with_zn": float(with_row["scan_value"]),
                        "scan_value_without_zn_reversed": float(without_row["scan_value"]),
                        "Zn_N_distance": float(with_row["Zn_N_distance"]),
                        "Zn_N_distance_without_zn_reversed": float(
                            without_row["Zn_N_distance"]
                        ),
                        "E_with_zn": float(with_row["E_tot"]),
                        "E_without_zn_reversed": float(without_row["E_tot"]),
                        "dE_with_zn_minus_without_zn_reversed": float(
                            with_row["E_tot"] - without_row["E_tot"]
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    curve_cols = [
        "Phase",
        "Run",
        "Method",
        "BSSE",
        "Complex_with_zn",
        "Complex_without_zn",
    ]

    def _per_curve(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Zn_N_distance").copy()
        y = g["dE_with_zn_minus_without_zn_reversed"].to_numpy(dtype=float)
        x = g["Zn_N_distance"].to_numpy(dtype=float)
        g["dE_shifted_to_xmax"] = _shift_curve_to_xmax_zero(x, y)
        return g

    out = (
        out.groupby(curve_cols, group_keys=False, sort=False)
        .apply(_per_curve, include_groups=True)
        .reset_index(drop=True)
    )
    out.sort_values(
        by=["Complex_with_zn", "Run", "BSSE", "Zn_N_distance"],
        inplace=True,
        ignore_index=True,
    )
    return out


def plot_scan_two_end_curves(
    scan_two_end_df: pd.DataFrame,
    runs_to_plot: Optional[List[str]] = None,
    output_file: str = "scan_two_end.png",
) -> None:
    """
    Plot E(with Zn at x) - E(without Zn at reversed x), shifted to max x.
    """
    if scan_two_end_df.empty:
        print("Skip plotting scan_two_end curves: empty table.")
        return

    plot_df = scan_two_end_df.copy()
    if runs_to_plot is not None:
        plot_df = plot_df[plot_df["Run"].isin(runs_to_plot)].copy()
        if plot_df.empty:
            print("Skip plotting scan_two_end curves: no rows after Run filter.")
            return

    complexes = _ordered_complexes(
        plot_df["Complex_with_zn"].dropna().unique().tolist()
    )
    if not complexes:
        print("Skip plotting scan_two_end curves: no paired complexes.")
        return

    ncols = 2 if len(complexes) > 1 else 1
    nrows = int(math.ceil(len(complexes) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    run_order = _ordered_runs(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    run_style_order = _run_style_order(
        plot_df["Run"].dropna().unique().tolist(),
        requested_order=runs_to_plot,
    )
    plot_df = _sort_by_run_order(plot_df, run_order)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_run = {
        run: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, run in enumerate(run_style_order)
    }
    plotted = False

    for i, complex_name in enumerate(complexes):
        ax = axes_flat[i]
        sub = plot_df[plot_df["Complex_with_zn"] == complex_name].copy()

        for (_, run_name, method_name, bsse), g in sub.groupby(
            ["Phase", "Run", "Method", "BSSE"],
            sort=False,
        ):
            g = g.sort_values("Zn_N_distance")
            if len(g) < 2:
                print(
                    f"Skip scan_two_end {complex_name} {run_name} | BSSE corr={bsse}: "
                    "fewer than two paired scan points."
                )
                continue
            label = _plot_legend_label([run_name, method_name], bsse)
            ax.plot(
                g["Zn_N_distance"].to_numpy(dtype=float),
                g["dE_shifted_to_xmax"].to_numpy(dtype=float),
                marker="o",
                color=color_by_run.get(run_name),
                linestyle="--" if bsse == "yes" else "-",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
            plotted = True

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        hbond_match = re.search(r"(\d+Hbond)$", complex_name)
        ax.set_title(hbond_match.group(1) if hbond_match else complex_name, fontsize=10)
        ax.set_xlabel("with-Zn Zn-N distance (Angstrom)")
        if i % ncols == 0:
            ax.set_ylabel("E(with Zn) - E(without Zn reversed x), shifted (kcal/mol)")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.3)

    for j in range(len(complexes), len(axes_flat)):
        axes_flat[j].axis("off")

    if not plotted:
        plt.close(fig)
        print("Skip plotting scan_two_end curves: no plottable curves.")
        return

    fig.suptitle(
        "With Zn - Without Zn Reversed Along x, Shifted to Max x",
        fontsize=12,
    )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=8,
        ncols=1,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.24, 1, 0.95])
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"scan_two_end figure saved to {output_file}")


def main() -> None:
    df = collect_scan_energies(base_dir="SP_init", runs_to_analyze=RUNS_TO_ANALYZE)

    if df.empty:
        print("No Zn scan data found.")
        return

    target_values = _get_target_values_from_energy_df(df)
    scan_values = sorted(float(v) for v in df["scan_value"].dropna().unique())
    complexes = df["Complex"].dropna().astype(str).unique().tolist()
    zn_n_distance_map = build_zn_n_distance_map(complexes, scan_values)
    df = add_zn_n_distances(df, zn_n_distance_map)
    if df["Zn_N_distance"].isna().any():
        missing = (
            df[df["Zn_N_distance"].isna()][["Complex", "scan_value"]]
            .drop_duplicates()
            .sort_values(["Complex", "scan_value"])
        )
        print("\nWarning: missing Zn-N distances for these complex/scan points:")
        print(missing.to_string(index=False))

    plot_hbond_comparison_relative_to_2hbond_min(df)
    zn_minus_no_zn_df = build_zn_minus_no_zn_table(df)
    if not zn_minus_no_zn_df.empty:
        output_file_zn = "with_zn_minus_without_zn.csv"
        zn_minus_no_zn_df.to_csv(output_file_zn, index=False)
        print(f"\nWith-Zn minus without-Zn table saved to {output_file_zn}")
        print(
            "\nMatched E(1Zn_1ImH_6Wat_*) - E(1ImH_6Wat_*) scan energies "
            "(same run, BSSE, and scan_value):"
        )
        print(zn_minus_no_zn_df.to_string(index=False))
        plot_zn_minus_no_zn_curves(
            zn_minus_no_zn_df,
        )
    scan_two_end_df = build_scan_two_end_table(df)
    if not scan_two_end_df.empty:
        output_file_scan_two_end = "scan_two_end.csv"
        scan_two_end_df.to_csv(output_file_scan_two_end, index=False)
        print(f"\nscan_two_end table saved to {output_file_scan_two_end}")
        print(
            "\nMatched E(with Zn at x) - E(without Zn at reversed x), "
            "shifted to the maximum-x point:"
        )
        print(scan_two_end_df.to_string(index=False))
        plot_scan_two_end_curves(scan_two_end_df)

    # Absolute dE relative to scan_value == 0.0
    dEabs_df = build_dEabs_table(df, target_values=target_values)
    if not dEabs_df.empty:
        output_file_abs = "dEabs.csv"
        dEabs_df.to_csv(output_file_abs, index=False)
        print(f"\nAbsolute dE table saved to {output_file_abs}")
        print("Columns: Phase, Run, Method, BSSE, Complex, then dE_0.0_* columns")
        print("\nAbsolute dE relative to scan_value == 0.0 (wide format):")
        print(dEabs_df.to_string(index=False))
        plot_dEabs_curves(
            dEabs_df,
            target_values=target_values,
            distance_map=zn_n_distance_map,
        )
        plot_dEabs_delta_vs_reference(
            dEabs_df,
            target_values=target_values,
            distance_map=zn_n_distance_map,
        )

if __name__ == "__main__":
    main()
