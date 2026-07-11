#!/usr/bin/env python3
"""Analyze IMH-water binding energies.

Ebind convention: each structure uses its own per-structure monomer
directories. For BSSE=no, Ebind = E_full - sum(E_monomer_i). For BSSE=yes,
Ebind = E_full - sum(E_monomer_i_ghost).

The script writes both the deformed-monomer reference and relaxed-monomer
reference variants. The relaxed-monomer variant uses the shared ImH_monomer
and Wat_monomer references for each run.
"""

# --- repo path bootstrap (auto) ---
from pathlib import Path as _Path
import sys as _sys
_REPO_CAND = _Path(__file__).resolve().parent
while _REPO_CAND != _REPO_CAND.parent and not (_REPO_CAND / "software.yaml").exists():
    _REPO_CAND = _REPO_CAND.parent
if not (_REPO_CAND / "software.yaml").exists():
    raise RuntimeError("Could not locate repo root (software.yaml)")
REPO_ROOT = _REPO_CAND
TOOLS_DIR = REPO_ROOT / "tools"
_sys.path.insert(0, str(TOOLS_DIR))
try:
    from paths import load_software as _load_software
    _SW = _load_software()
except Exception:
    _SW = {}
# --- end bootstrap ---

import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = str(REPO_ROOT)
PYTHON_SCRIPTS_DIR = str(TOOLS_DIR)
IMH_WATER_CONTACTS_DIR = (
    str(TOOLS_DIR)
)
XYZ_DIR = os.path.join(_BASE_DIR, "xyz", "xyz_files")
IMH_WATER_CLASS_CUTOFFS = {
    "NH...O": {"distance_A": 2.5, "min_angle_deg": 130.0},
    "OH...N3": {"distance_A": 2.5, "min_angle_deg": 140.0},
    "CH...O(C1)": {"distance_A": 3.5, "min_angle_deg": 90.0},
    "CH...O(C2)": {"distance_A": 3.5, "min_angle_deg": 90.0},
    "CH...O(C3)": {"distance_A": 3.5, "min_angle_deg": 90.0},
    "OH...pi(C1)": {"distance_A": 2.8, "min_angle_deg": 130.0},
    "OH...pi(C2)": {"distance_A": 3.0, "min_angle_deg": 120.0},
    "OH...pi(C3)": {"distance_A": 3.5, "min_angle_deg": 90.0},
    "OH...pi(N9)": {"distance_A": 4.0, "min_angle_deg": 90.0},
}
IMH_WATER_DISTANCE_CUTOFF_A = max(
    cutoff["distance_A"] for cutoff in IMH_WATER_CLASS_CUTOFFS.values()
)
IMH_WATER_MIN_ANGLE_DEG = min(
    cutoff["min_angle_deg"] for cutoff in IMH_WATER_CLASS_CUTOFFS.values()
)
IMH_WATER_NO_HBOND_CLOSE_CUTOFF_A = 4.0
IMH_WATER_IMIDAZOLE_ATOMS = 9
IMH_WATER_WATER_ATOMS = 3
IMH_WATER_N3_INDEX = 8
IMH_WATER_NH_H_INDEX = 6
IMH_WATER_NH_N_INDEX = 9

for _path in (PYTHON_SCRIPTS_DIR, ROOT_DIR, IMH_WATER_CONTACTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analib import get_Etot_amber  # type: ignore
from analyze_imh_water_contacts import classify_imh_water_xyz  # type: ignore

STRUCTURE_RE = re.compile(r"^IMZW(\d+)(?:_(\d+))?_WAT(\d+)$")
RUN_RE = re.compile(r"^run(\d+)$")
DEFAULT_BASE_DIR = "SP_init"
REF_COLUMN = "run102|BSSE=yes"
COMPARE_COLUMN = "run110|BSSE=no"
REF_DEFORMED_MONOMER_TAG = "ref_deformed_monomer"
REF_RELAXED_MONOMER_TAG = "ref_relaxed_monomer"
EBIND_OUTPUT_TEMPLATE = "Ebind_{tag}.csv"
SUMMARY_OUTPUT_TEMPLATE = "class_Ebind_{tag}.csv"
CORRELATION_PLOT_DIR_TEMPLATE = "correlation_plots_{tag}"
DEFORMATION_OUTPUT_FILE = "deformation_energy_ref_deformed_minus_relaxed_monomer.csv"
DEFORMATION_PLOT_DIR = (
    "deformation_energy_correlation_plots_ref_deformed_minus_relaxed_monomer"
)
DEFORMATION_REF_RUN = "run102"
DEFORMATION_IMH_ONLY_RUNS = ("run0", "run110.mmsolvent")
RELAXED_MONOMER_DIRS = {
    "ImH": "ImH_monomer",
    "Wat": "Wat_monomer",
}
MONOMER_SPECIES_BY_INDEX = {
    0: "ImH",
    1: "Wat",
}
CONTACT_COLUMNS = [
    "contact_class",
    "contact_type",
    "contact_detail",
    "contact_distance_A",
    "contact_angle_deg",
    "contact_angle_atoms",
    "contact_matched",
    "contact_status",
]


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


def _parse_structure_name(structure: str) -> Optional[Tuple[int, int, int]]:
    match = STRUCTURE_RE.fullmatch(structure)
    if match is None:
        return None
    n_imzw = int(match.group(1))
    variant = int(match.group(2) or 0)
    n_wat = int(match.group(3))
    return (n_imzw, variant, n_wat)


def _structure_sort_key(structure: str) -> Tuple[int, int, int, str]:
    parsed = _parse_structure_name(structure)
    if parsed is None:
        return (10**9, 10**9, 10**9, structure)
    n_imzw, variant, n_wat = parsed
    return (n_imzw, variant, n_wat, structure)


def _iter_structure_names(run_path: str) -> List[str]:
    structures: List[str] = []
    for entry in os.listdir(run_path):
        if not entry.endswith("_full"):
            continue
        structure = entry[: -len("_full")]
        if _parse_structure_name(structure) is not None:
            structures.append(structure)
    return sorted(set(structures), key=_structure_sort_key)


def _has_bsse_ghost_dirs(run_path: str) -> bool:
    for entry in os.listdir(run_path):
        if not entry.endswith("_ghost"):
            continue
        if os.path.isdir(os.path.join(run_path, entry)):
            return True
    return False


def _monomer_species(monomer_index: int) -> Optional[str]:
    return MONOMER_SPECIES_BY_INDEX.get(monomer_index)


def _relaxed_monomer_energy(run_path: str, species: str) -> Optional[float]:
    monomer_dir = RELAXED_MONOMER_DIRS.get(species)
    if monomer_dir is None:
        return None

    return get_Etot_amber(os.path.join(run_path, monomer_dir, "min.out"))


def _collect_relaxed_monomer_refs(run_path: str) -> Dict[str, Optional[float]]:
    return {
        species: _relaxed_monomer_energy(run_path, species)
        for species in RELAXED_MONOMER_DIRS
    }


def _get_monomer_indices(run_dir: str, label: str) -> List[int]:
    monomer_pattern = re.compile(re.escape(label) + r"_monomer_(\d+)$")
    monomer_indices: List[int] = []

    for entry in os.listdir(run_dir):
        path = os.path.join(run_dir, entry)
        if not os.path.isdir(path):
            continue
        match = monomer_pattern.match(entry)
        if match:
            monomer_indices.append(int(match.group(1)))

    return sorted(monomer_indices)


def _sum_own_monomer_energies(
    run_dir: str,
    label: str,
    monomer_indices: List[int],
    use_ghost: bool,
) -> Optional[float]:
    if not monomer_indices:
        print(f"  WARNING: {run_dir}/{label} has no own monomer directories")
        return None

    total = 0.0
    missing: List[str] = []

    for n in monomer_indices:
        monomer_name = f"{label}_monomer_{n}"
        if use_ghost:
            monomer_name = f"{monomer_name}_ghost"

        monomer_path = os.path.join(run_dir, monomer_name, "min.out")
        e_monomer = get_Etot_amber(monomer_path)
        if e_monomer is None:
            missing.append(monomer_name)
            continue
        total += e_monomer

    if missing:
        label_type = "BSSE ghost monomers" if use_ghost else "own monomers"
        print(
            f"  WARNING: {run_dir}/{label} missing {label_type}: "
            f"{', '.join(missing)}"
        )
        return None

    return total


def _sum_relaxed_monomer_energies(
    run_dir: str,
    label: str,
    monomer_indices: List[int],
    relaxed_refs: Dict[str, Optional[float]],
) -> Optional[float]:
    if not monomer_indices:
        print(f"  WARNING: {run_dir}/{label} has no own monomer directories")
        return None

    total = 0.0
    missing: List[str] = []

    for n in monomer_indices:
        species = _monomer_species(n)
        if species is None:
            missing.append(f"monomer_{n}:unknown_species")
            continue

        energy = relaxed_refs.get(species)
        if energy is None:
            missing.append(f"{species}:{RELAXED_MONOMER_DIRS.get(species, 'unknown')}")
            continue
        total += energy

    if missing:
        print(
            f"  WARNING: {run_dir}/{label} missing relaxed monomers: "
            f"{', '.join(missing)}"
        )
        return None

    return total


def _compute_structure_ebind(
    run_dir: str,
    structure: str,
    use_bsse: bool,
    ref_mode: str,
    relaxed_refs: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[float]:
    full_dir = os.path.join(run_dir, f"{structure}_full")
    e_full = get_Etot_amber(os.path.join(full_dir, "min.out"))
    if e_full is None:
        return None

    monomer_indices = _get_monomer_indices(run_dir, structure)
    if ref_mode == REF_DEFORMED_MONOMER_TAG:
        monomer_sum = _sum_own_monomer_energies(
            run_dir,
            structure,
            monomer_indices,
            use_ghost=use_bsse,
        )
        if monomer_sum is None:
            return None
        return e_full - monomer_sum

    if ref_mode == REF_RELAXED_MONOMER_TAG:
        if relaxed_refs is None:
            relaxed_refs = _collect_relaxed_monomer_refs(run_dir)
        relaxed_sum = _sum_relaxed_monomer_energies(
            run_dir,
            structure,
            monomer_indices,
            relaxed_refs,
        )
        if relaxed_sum is None:
            return None

        bsse_correction = 0.0
        if use_bsse:
            deformed_sum = _sum_own_monomer_energies(
                run_dir,
                structure,
                monomer_indices,
                use_ghost=False,
            )
            ghost_sum = _sum_own_monomer_energies(
                run_dir,
                structure,
                monomer_indices,
                use_ghost=True,
            )
            if deformed_sum is None or ghost_sum is None:
                return None
            bsse_correction = ghost_sum - deformed_sum

        return e_full - bsse_correction - relaxed_sum

    raise ValueError(f"Unknown Ebind reference mode: {ref_mode}")


def _empty_contact_record(status: str) -> Dict[str, object]:
    return {
        "contact_class": np.nan,
        "contact_type": np.nan,
        "contact_detail": np.nan,
        "contact_distance_A": np.nan,
        "contact_angle_deg": np.nan,
        "contact_angle_atoms": np.nan,
        "contact_matched": np.nan,
        "contact_status": status,
    }


def _classify_structure_contact(structure: str) -> Dict[str, object]:
    xyz_path = os.path.join(XYZ_DIR, f"{structure}.xyz")
    if not os.path.exists(xyz_path):
        return _empty_contact_record("missing_xyz")

    try:
        rows = classify_imh_water_xyz(
            xyz_path,
            imidazole_atoms_count=IMH_WATER_IMIDAZOLE_ATOMS,
            water_atoms_count=IMH_WATER_WATER_ATOMS,
            cutoff=IMH_WATER_DISTANCE_CUTOFF_A,
            min_angle=IMH_WATER_MIN_ANGLE_DEG,
            n3_position=IMH_WATER_N3_INDEX,
            nh_h_position=IMH_WATER_NH_H_INDEX,
            nh_n_position=IMH_WATER_NH_N_INDEX,
            include_far=True,
            class_cutoffs=IMH_WATER_CLASS_CUTOFFS,
            no_hbond_close_cutoff=IMH_WATER_NO_HBOND_CLOSE_CUTOFF_A,
        )
    except Exception as exc:
        return _empty_contact_record(f"error: {exc}")

    if not rows:
        return _empty_contact_record("no_water_contact")

    row = min(rows, key=lambda item: item["distance_A"])
    return {
        "contact_class": row.get("interaction_label", row.get("interaction_type")),
        "contact_type": row.get("interaction_type"),
        "contact_detail": row.get("interaction_detail"),
        "contact_distance_A": row.get("distance_A", np.nan),
        "contact_angle_deg": row.get("angle_deg", np.nan),
        "contact_angle_atoms": row.get("angle_atoms", ""),
        "contact_matched": row.get("matched_interactions", ""),
        "contact_status": "ok",
    }


def _collect_ebind_rows_for_run(
    run_path: str,
    use_bsse: bool,
    ref_mode: str,
    relaxed_refs: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, float]:
    if use_bsse and not _has_bsse_ghost_dirs(run_path):
        return {}

    rows: Dict[str, float] = {}
    for structure in _iter_structure_names(run_path):
        ebind = _compute_structure_ebind(
            run_path,
            structure,
            use_bsse=use_bsse,
            ref_mode=ref_mode,
            relaxed_refs=relaxed_refs,
        )
        if ebind is not None:
            rows[structure] = ebind

    return rows


def _collect_run_sources(default_base_dir: str) -> List[Tuple[str, str]]:
    run_sources: List[Tuple[str, str]] = []
    if not os.path.exists(default_base_dir):
        return run_sources

    for run_path in sorted(
        glob.glob(os.path.join(default_base_dir, "run*")),
        key=_run_sort_key,
    ):
        if os.path.isdir(run_path):
            run_sources.append((os.path.basename(run_path), run_path))
    return run_sources


def analyze_runs(
    base_dir: str = DEFAULT_BASE_DIR,
    ref_mode: str = REF_DEFORMED_MONOMER_TAG,
) -> pd.DataFrame:
    run_sources = _collect_run_sources(base_dir)
    if not run_sources:
        return pd.DataFrame()

    all_rows: Dict[str, Dict[str, float]] = {}
    column_order: List[str] = []
    method_map: Dict[str, str] = {}

    for run_name, run_path in run_sources:
        method_map[run_name] = _load_method_name(run_path)
        relaxed_refs = (
            _collect_relaxed_monomer_refs(run_path)
            if ref_mode == REF_RELAXED_MONOMER_TAG
            else None
        )

        for use_bsse in (False, True):
            row_values = _collect_ebind_rows_for_run(
                run_path,
                use_bsse=use_bsse,
                ref_mode=ref_mode,
                relaxed_refs=relaxed_refs,
            )
            if not row_values:
                continue

            column_name = f"{run_name}|BSSE={'yes' if use_bsse else 'no'}"
            column_order.append(column_name)

            for structure, ebind in row_values.items():
                all_rows.setdefault(structure, {})
                all_rows[structure][column_name] = ebind

    if not all_rows:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for structure in sorted(all_rows.keys(), key=_structure_sort_key):
        parsed = _parse_structure_name(structure)
        n_imzw, variant, n_wat = parsed if parsed is not None else (np.nan, np.nan, np.nan)
        record: Dict[str, object] = {
            "structure": structure,
            "n_imzw": n_imzw,
            "variant": variant,
            "n_wat": n_wat,
        }
        record.update(_classify_structure_contact(structure))
        for column_name in column_order:
            record[column_name] = all_rows[structure].get(column_name, np.nan)
        records.append(record)

    df = pd.DataFrame(records)
    df.attrs["method_map"] = method_map
    return df


def _value_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in ("structure", "n_imzw", "variant", "n_wat", *CONTACT_COLUMNS)
    ]


def _format_mean_std(mean: float, std: Optional[float]) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def summarize_run_vs_ref(
    df: pd.DataFrame,
    ref_col: str = REF_COLUMN,
    compare_col: str = COMPARE_COLUMN,
) -> pd.DataFrame:
    if df.empty or ref_col not in df.columns or compare_col not in df.columns:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for contact_class in df["contact_class"].dropna().unique():
        sub = df[df["contact_class"] == contact_class]
        ref = sub[ref_col]
        diff = sub[compare_col] - ref
        n = len(sub)
        ref_std = ref.std(ddof=1) if n > 1 else np.nan
        diff_std = diff.std(ddof=1) if n > 1 else np.nan
        ref_mean = ref.mean()
        diff_mean = diff.mean()
        records.append(
            {
                "contact_class": contact_class,
                "N": n,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "ref_mean_std": _format_mean_std(ref_mean, ref_std),
                "diff_mean": diff_mean,
                "diff_std": diff_std,
                "diff_mean_std": _format_mean_std(diff_mean, diff_std),
                "RMSE": float(np.sqrt(np.mean(diff**2))),
                "_sort_key": abs(ref_mean),
            }
        )

    records.sort(key=lambda row: row["_sort_key"], reverse=True)
    for row in records:
        del row["_sort_key"]

    ref_all = df[ref_col]
    diff_all = df[compare_col] - ref_all
    n_all = len(df)
    ref_mean_all = ref_all.mean()
    diff_mean_all = diff_all.mean()
    records.append(
        {
            "contact_class": "Overall",
            "N": n_all,
            "ref_mean": ref_mean_all,
            "ref_std": ref_all.std(ddof=1),
            "ref_mean_std": _format_mean_std(ref_mean_all, ref_all.std(ddof=1)),
            "diff_mean": diff_mean_all,
            "diff_std": diff_all.std(ddof=1),
            "diff_mean_std": _format_mean_std(diff_mean_all, diff_all.std(ddof=1)),
            "RMSE": float(np.sqrt(np.mean(diff_all**2))),
        }
    )

    return pd.DataFrame(records)


def summarize_all_runs_vs_ref(
    df: pd.DataFrame,
    ref_col: str = REF_COLUMN,
) -> pd.DataFrame:
    if df.empty or ref_col not in df.columns:
        return pd.DataFrame()

    compare_cols = [col for col in _value_columns(df) if col != ref_col]
    if not compare_cols:
        return pd.DataFrame()

    wide: Optional[pd.DataFrame] = None
    for compare_col in compare_cols:
        summary = summarize_run_vs_ref(
            df,
            ref_col=ref_col,
            compare_col=compare_col,
        )
        if summary.empty:
            continue

        run_cols = summary[
            ["contact_class", "diff_mean", "diff_std", "diff_mean_std", "RMSE"]
        ].rename(
            columns={
                "diff_mean": f"{compare_col}|diff_mean",
                "diff_std": f"{compare_col}|diff_std",
                "diff_mean_std": f"{compare_col}|diff_mean_std",
                "RMSE": f"{compare_col}|RMSE",
            }
        )

        if wide is None:
            wide = summary[
                ["contact_class", "N", "ref_mean", "ref_std", "ref_mean_std"]
            ].merge(run_cols, on="contact_class")
        else:
            wide = wide.merge(run_cols, on="contact_class")

    if wide is None:
        return pd.DataFrame()
    return _order_class_ebind_columns(wide, compare_cols)


def _order_class_ebind_columns(
    wide: pd.DataFrame,
    compare_cols: List[str],
) -> pd.DataFrame:
    front_cols = ["contact_class", "N", "ref_mean_std"]
    back_cols: List[str] = []
    for compare_col in compare_cols:
        diff_mean_std_col = f"{compare_col}|diff_mean_std"
        rmse_col = f"{compare_col}|RMSE"
        if diff_mean_std_col in wide.columns:
            front_cols.append(diff_mean_std_col)
        if rmse_col in wide.columns:
            front_cols.append(rmse_col)
    for compare_col in compare_cols:
        diff_mean_col = f"{compare_col}|diff_mean"
        diff_std_col = f"{compare_col}|diff_std"
        if diff_mean_col in wide.columns:
            back_cols.append(diff_mean_col)
        if diff_std_col in wide.columns:
            back_cols.append(diff_std_col)
    back_cols.extend(["ref_mean", "ref_std"])
    ordered_cols = front_cols + back_cols
    return wide[ordered_cols]


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "plot"


def _prepare_matplotlib_cache() -> None:
    cache_dir = os.path.join("/tmp", "analyze_dE_matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)


def plot_run_vs_ref_correlation(
    df: pd.DataFrame,
    ref_col: str = REF_COLUMN,
    compare_col: str = COMPARE_COLUMN,
    output_file: Optional[str] = None,
) -> bool:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    if df.empty or ref_col not in df.columns or compare_col not in df.columns:
        return False

    if output_file is None:
        default_dir = CORRELATION_PLOT_DIR_TEMPLATE.format(
            tag=REF_DEFORMED_MONOMER_TAG
        )
        os.makedirs(default_dir, exist_ok=True)
        output_file = os.path.join(
            default_dir,
            f"{_safe_filename_part(compare_col)}_vs_"
            f"{_safe_filename_part(ref_col)}.png",
        )

    plot_df = df[[ref_col, compare_col, "contact_class"]].dropna()
    if plot_df.empty:
        return False

    x = plot_df[ref_col].to_numpy()
    y = plot_df[compare_col].to_numpy()
    diff = y - x
    rmse = float(np.sqrt(np.mean(diff**2)))
    diff_std = diff.std(ddof=1) if len(diff) > 1 else np.nan
    pearson_r = (
        float(np.corrcoef(x, y)[0, 1])
        if len(plot_df) > 1 and np.std(x) > 0.0 and np.std(y) > 0.0
        else np.nan
    )
    can_fit = len(plot_df) > 1 and np.ptp(x) > 0.0
    slope = intercept = np.nan
    if can_fit:
        slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    contact_classes = sorted(plot_df["contact_class"].dropna().unique())
    cmap = plt.get_cmap("tab20")

    for idx, contact_class in enumerate(contact_classes):
        sub = plot_df[plot_df["contact_class"] == contact_class]
        ax.scatter(
            sub[ref_col],
            sub[compare_col],
            s=48,
            alpha=0.85,
            color=cmap(idx % 20),
            edgecolors="white",
            linewidths=0.5,
            label=str(contact_class),
        )

    lim_min = min(x.min(), y.min()) - 0.5
    lim_max = max(x.max(), y.max()) + 0.5
    lims = [lim_min, lim_max]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="y = x")

    fit_x = np.array(lims)
    if can_fit:
        fit_y = slope * fit_x + intercept
        ax.plot(
            fit_x,
            fit_y,
            color="crimson",
            linewidth=1.5,
            label=f"fit: y = {slope:.3f}x {intercept:+.3f}",
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{ref_col} (kcal/mol)")
    ax.set_ylabel(f"{compare_col} (kcal/mol)")
    ax.set_title(f"{compare_col} vs {ref_col}")
    ax.grid(True, alpha=0.25)

    stats_text = (
        f"N = {len(plot_df)}\n"
        f"R = {pearson_r:.3f}\n"
        f"RMSE = {rmse:.3f} kcal/mol\n"
        f"diff = {diff.mean():+.3f} +/- {diff_std:.3f} kcal/mol"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.35,
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_all_runs_vs_ref_correlations(
    df: pd.DataFrame,
    ref_col: str = REF_COLUMN,
    output_dir: Optional[str] = None,
) -> List[str]:
    if df.empty or ref_col not in df.columns:
        return []

    if output_dir is None:
        output_dir = CORRELATION_PLOT_DIR_TEMPLATE.format(
            tag=REF_DEFORMED_MONOMER_TAG
        )

    output_files: List[str] = []
    os.makedirs(output_dir, exist_ok=True)
    ref_name = _safe_filename_part(ref_col)

    for compare_col in _value_columns(df):
        if compare_col == ref_col:
            continue
        output_file = os.path.join(
            output_dir,
            f"{_safe_filename_part(compare_col)}_vs_{ref_name}.png",
        )
        if plot_run_vs_ref_correlation(
            df,
            ref_col=ref_col,
            compare_col=compare_col,
            output_file=output_file,
        ):
            output_files.append(output_file)

    return output_files


def collect_deformation_energy_rows(
    base_dir: str = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    run_sources = _collect_run_sources(base_dir)
    if not run_sources:
        return pd.DataFrame()

    all_rows: Dict[Tuple[str, int], Dict[str, float]] = {}
    column_order: List[str] = []
    method_map: Dict[str, str] = {}

    for run_name, run_path in run_sources:
        method_map[run_name] = _load_method_name(run_path)
        relaxed_refs = _collect_relaxed_monomer_refs(run_path)
        has_values = False

        for structure in _iter_structure_names(run_path):
            for monomer_index in _get_monomer_indices(run_path, structure):
                species = _monomer_species(monomer_index)
                if species is None:
                    continue

                relaxed_energy = relaxed_refs.get(species)
                if relaxed_energy is None:
                    continue

                monomer_path = os.path.join(
                    run_path,
                    f"{structure}_monomer_{monomer_index}",
                    "min.out",
                )
                deformed_energy = get_Etot_amber(monomer_path)
                if deformed_energy is None:
                    continue

                all_rows.setdefault((structure, monomer_index), {})
                all_rows[(structure, monomer_index)][run_name] = (
                    deformed_energy - relaxed_energy
                )
                has_values = True

        if has_values:
            column_order.append(run_name)

    if not all_rows:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for structure, monomer_index in sorted(
        all_rows.keys(),
        key=lambda item: (_structure_sort_key(item[0]), item[1]),
    ):
        parsed = _parse_structure_name(structure)
        n_imzw, variant, n_wat = parsed if parsed is not None else (np.nan, np.nan, np.nan)
        record: Dict[str, object] = {
            "structure": structure,
            "n_imzw": n_imzw,
            "variant": variant,
            "n_wat": n_wat,
            "monomer_index": monomer_index,
            "monomer_species": _monomer_species(monomer_index) or "unknown",
        }
        record.update(_classify_structure_contact(structure))
        for column_name in column_order:
            record[column_name] = all_rows[(structure, monomer_index)].get(
                column_name,
                np.nan,
            )
        records.append(record)

    df = pd.DataFrame(records)
    df.attrs["method_map"] = method_map
    return df


def _deformation_value_columns(df: pd.DataFrame) -> List[str]:
    excluded = (
        "structure",
        "n_imzw",
        "variant",
        "n_wat",
        "monomer_index",
        "monomer_species",
        *CONTACT_COLUMNS,
    )
    return [col for col in df.columns if col not in excluded]


def _format_deformation_stats(
    plot_df: pd.DataFrame,
    ref_col: str,
    compare_col: str,
) -> str:
    blocks: List[str] = []
    for species in sorted(plot_df["monomer_species"].dropna().unique()):
        sub = plot_df[plot_df["monomer_species"] == species]
        x = sub[ref_col].to_numpy()
        y = sub[compare_col].to_numpy()
        diff = y - x
        rmse = float(np.sqrt(np.mean(diff**2)))
        diff_std = diff.std(ddof=1) if len(diff) > 1 else np.nan
        pearson_r = (
            float(np.corrcoef(x, y)[0, 1])
            if len(sub) > 1 and np.std(x) > 0.0 and np.std(y) > 0.0
            else np.nan
        )
        blocks.append(
            f"{species}: N = {len(sub)}\n"
            f"  R = {pearson_r:.3f}\n"
            f"  RMSE = {rmse:.3f} kcal/mol\n"
            f"  diff = {diff.mean():+.3f} +/- {diff_std:.3f} kcal/mol"
        )
    return "\n".join(blocks)


def plot_deformation_energy_correlation(
    df: pd.DataFrame,
    ref_col: str = DEFORMATION_REF_RUN,
    compare_col: str = "run110",
    output_file: Optional[str] = None,
    species_filter: Optional[str] = None,
) -> bool:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    if df.empty or ref_col not in df.columns or compare_col not in df.columns:
        return False

    if output_file is None:
        os.makedirs(DEFORMATION_PLOT_DIR, exist_ok=True)
        species_suffix = (
            f"_{_safe_filename_part(species_filter)}_only"
            if species_filter is not None
            else ""
        )
        output_file = os.path.join(
            DEFORMATION_PLOT_DIR,
            f"{_safe_filename_part(compare_col)}_vs_"
            f"{_safe_filename_part(ref_col)}{species_suffix}.png",
        )

    plot_df = df[[ref_col, compare_col, "monomer_species"]].dropna()
    if species_filter is not None:
        plot_df = plot_df[plot_df["monomer_species"] == species_filter]
    if plot_df.empty:
        return False

    x = plot_df[ref_col].to_numpy()
    y = plot_df[compare_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    species_styles = {
        "ImH": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
        "Wat": {"color": "tab:orange", "marker": "s", "linestyle": "-."},
    }

    for species in sorted(plot_df["monomer_species"].dropna().unique()):
        sub = plot_df[plot_df["monomer_species"] == species]
        style = species_styles.get(
            str(species),
            {"color": "tab:gray", "marker": "o", "linestyle": ":"},
        )
        ax.scatter(
            sub[ref_col],
            sub[compare_col],
            s=42,
            alpha=0.85,
            color=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.5,
            label=str(species),
        )

        sub_x = sub[ref_col].to_numpy()
        sub_y = sub[compare_col].to_numpy()
        if len(sub) > 1 and np.ptp(sub_x) > 0.0:
            slope, intercept = np.polyfit(sub_x, sub_y, 1)
            fit_x = np.array([sub_x.min(), sub_x.max()])
            ax.plot(
                fit_x,
                slope * fit_x + intercept,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                label=f"{species} fit: y = {slope:.3f}x {intercept:+.3f}",
            )

    lim_min = min(x.min(), y.min()) - 0.5
    lim_max = max(x.max(), y.max()) + 0.5
    lims = [lim_min, lim_max]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="y = x")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{ref_col} deformation energy (kcal/mol)")
    ax.set_ylabel(f"{compare_col} deformation energy (kcal/mol)")
    title = f"Deformation energy: {compare_col} vs {ref_col}"
    if species_filter is not None:
        title = f"{title} ({species_filter} only)"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    stats_text = _format_deformation_stats(plot_df, ref_col, compare_col)
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.35,
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_all_deformation_energy_correlations(
    df: pd.DataFrame,
    ref_col: str = DEFORMATION_REF_RUN,
    output_dir: str = DEFORMATION_PLOT_DIR,
) -> List[str]:
    if df.empty or ref_col not in df.columns:
        return []

    output_files: List[str] = []
    os.makedirs(output_dir, exist_ok=True)
    ref_name = _safe_filename_part(ref_col)

    for compare_col in _deformation_value_columns(df):
        if compare_col == ref_col:
            continue
        output_file = os.path.join(
            output_dir,
            f"{_safe_filename_part(compare_col)}_vs_{ref_name}.png",
        )
        if plot_deformation_energy_correlation(
            df,
            ref_col=ref_col,
            compare_col=compare_col,
            output_file=output_file,
        ):
            output_files.append(output_file)

    for compare_col in DEFORMATION_IMH_ONLY_RUNS:
        if compare_col == ref_col or compare_col not in df.columns:
            continue
        output_file = os.path.join(
            output_dir,
            f"{_safe_filename_part(compare_col)}_vs_{ref_name}_ImH_only.png",
        )
        if plot_deformation_energy_correlation(
            df,
            ref_col=ref_col,
            compare_col=compare_col,
            output_file=output_file,
            species_filter="ImH",
        ):
            output_files.append(output_file)

    return output_files


def _write_ebind_outputs(ref_mode: str) -> Optional[pd.DataFrame]:
    df = analyze_runs(base_dir=DEFAULT_BASE_DIR, ref_mode=ref_mode)

    if df.empty:
        print(f"No binding energies found for {ref_mode} under {DEFAULT_BASE_DIR}/run*.")
        return None

    output_file = EBIND_OUTPUT_TEMPLATE.format(tag=ref_mode)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    summary_file = SUMMARY_OUTPUT_TEMPLATE.format(tag=ref_mode)
    summary_df = summarize_all_runs_vs_ref(df)
    if not summary_df.empty:
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
        compare_cols = [
            col
            for col in _value_columns(df)
            if col != REF_COLUMN and f"{col}|diff_mean_std" in summary_df.columns
        ]
        print(
            f"\nAll runs vs {REF_COLUMN} for {ref_mode} "
            f"(sorted by |ref_mean|, kcal/mol):"
        )
        display_cols = ["contact_class", "N", "ref_mean_std"]
        for compare_col in compare_cols:
            display_cols.extend(
                [f"{compare_col}|diff_mean_std", f"{compare_col}|RMSE"]
            )
        print(summary_df[display_cols].to_string(index=False))
    else:
        print(
            f"\nWARNING: summary skipped for {ref_mode}; missing {REF_COLUMN} "
            f"or compare columns in results"
        )

    plot_dir = CORRELATION_PLOT_DIR_TEMPLATE.format(tag=ref_mode)
    plot_files = plot_all_runs_vs_ref_correlations(df, output_dir=plot_dir)
    if plot_files:
        print(f"Correlation plots saved to {plot_dir}/ ({len(plot_files)} files)")

    return df


def _write_deformation_outputs() -> Optional[pd.DataFrame]:
    df = collect_deformation_energy_rows(base_dir=DEFAULT_BASE_DIR)

    if df.empty:
        print(f"No deformation energies found under {DEFAULT_BASE_DIR}/run*.")
        return None

    df.to_csv(DEFORMATION_OUTPUT_FILE, index=False)
    print(f"\nDeformation energies saved to {DEFORMATION_OUTPUT_FILE}")

    plot_files = plot_all_deformation_energy_correlations(df)
    if plot_files:
        print(
            f"Deformation energy correlation plots saved to {DEFORMATION_PLOT_DIR}/ "
            f"({len(plot_files)} files)"
        )
    else:
        print(
            f"WARNING: deformation energy plots skipped; missing "
            f"{DEFORMATION_REF_RUN} or compare columns"
        )

    return df


if __name__ == "__main__":
    result_dfs = [
        _write_ebind_outputs(REF_DEFORMED_MONOMER_TAG),
        _write_ebind_outputs(REF_RELAXED_MONOMER_TAG),
    ]
    deformation_df = _write_deformation_outputs()

    method_map: Dict[str, str] = {}
    for df in result_dfs:
        if df is not None:
            method_map.update(df.attrs.get("method_map", {}))
    if deformation_df is not None:
        method_map.update(deformation_df.attrs.get("method_map", {}))

    if method_map:
        print("\nRun legend:")
        for run_name in sorted(method_map.keys(), key=_run_sort_key):
            print(f"  {run_name}: {method_map[run_name]}")
