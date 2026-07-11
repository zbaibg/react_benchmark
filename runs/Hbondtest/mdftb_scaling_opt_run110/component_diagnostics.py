#!/usr/bin/env python3
"""Run and parse DFTB+ component diagnostics for mDFTB scaling tests."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


HARTREE_TO_KCAL_MOL = 627.5094740631
WORK_DIR = Path(__file__).resolve().parent
ROOT_DIR = WORK_DIR.parent
RUN110_DIR = ROOT_DIR / "SP_init" / "run110"
TARGET_CSV = ROOT_DIR / "Ebind.csv"

COMPONENT_PATTERNS = {
    "band": r"Band energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "energy_h0": r"Energy H0:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "scc": r"Energy SCC:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "third": r"Energy 3rd:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "monopole_dipole": r"Energy Monopole-Dipole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "dipole_dipole": r"Energy Dipole-Dipole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "monopole_quadrupole": r"Energy Monopole-Quadrupole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "dipole_quadrupole": r"Energy Dipole-Quadrupole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "quadrupole_quadrupole": r"Energy Quadrupole-Quadrupole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "multipole": r"Energy Multipole:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "electronic_total": r"Total Electronic energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "repulsive": r"Repulsive energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "dispersion": r"Dispersion energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
    "total": r"Total energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
}


def target_structures() -> list[str]:
    df = pd.read_csv(TARGET_CSV)
    return list(df["structure"].dropna())


def run_one(name: str, input_path: Path, output_root: Path, dftb_bin: str, timeout: int, threads: int) -> str:
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, run_dir / "dftb_in.hsd")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [dftb_bin],
        cwd=run_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    (run_dir / "dftb_stdout.log").write_text(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-60:])
        raise RuntimeError(f"{name} failed with {proc.returncode}\n{tail}")
    if not (run_dir / "detailed.out").exists():
        raise RuntimeError(f"{name} completed but detailed.out is missing")
    return name


def run_dftb_inputs(input_root: Path, output_root: Path, workers: int, dftb_bin: str, timeout: int, threads: int) -> None:
    names = ["ImH_monomer", "Wat_monomer"] + [f"{structure}_full" for structure in target_structures()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for name in names:
            input_path = input_root / name / "dftb_in.hsd"
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            futures.append(pool.submit(run_one, name, input_path, output_root, dftb_bin, timeout, threads))
        for future in as_completed(futures):
            print(f"finished {future.result()}", flush=True)


def parse_components(path: Path) -> Dict[str, float]:
    text = path.read_text(errors="replace")
    row: Dict[str, float] = {}
    for name, pattern in COMPONENT_PATTERNS.items():
        match = re.search(pattern, text)
        if match is not None:
            row[name] = float(match.group(1))
    missing = [name for name in ("total", "repulsive", "dispersion") if name not in row]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return row


def read_variant_outputs(variant: str, detailed_root: Path, structures: Iterable[str]) -> pd.DataFrame:
    names = ["ImH_monomer", "Wat_monomer"] + [f"{structure}_full" for structure in structures]
    rows = []
    for name in names:
        path = detailed_root / name / "detailed.out"
        if not path.exists():
            raise FileNotFoundError(path)
        row = {"variant": variant, "system": name}
        row.update(parse_components(path))
        rows.append(row)
    return pd.DataFrame(rows)


def binding_components(system_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in system_df.groupby("variant"):
        by_system = {row["system"]: row for _, row in sub.iterrows()}
        imh = by_system["ImH_monomer"]
        wat = by_system["Wat_monomer"]
        for _, target in target_df.iterrows():
            structure = target["structure"]
            comp = by_system[f"{structure}_full"]
            row = {
                "variant": variant,
                "structure": structure,
                "contact_class": target["contact_class"],
                "ref": target["run102|BSSE=yes"],
            }
            for component in COMPONENT_PATTERNS:
                if component not in comp or component not in imh or component not in wat:
                    continue
                row[f"{component}_bind"] = (
                    float(comp[component]) - float(imh[component]) - float(wat[component])
                ) * HARTREE_TO_KCAL_MOL
            row["diff_vs_ref"] = row["total_bind"] - row["ref"]
            rows.append(row)
    return pd.DataFrame(rows)


def component_delta_summary(bind_df: pd.DataFrame) -> pd.DataFrame:
    baseline = bind_df[bind_df["variant"] == "baseline"].set_index("structure")
    rows = []
    for variant in sorted(set(bind_df["variant"]) - {"baseline"}):
        other = bind_df[bind_df["variant"] == variant].set_index("structure")
        common = baseline.index.intersection(other.index)
        for col in sorted(c for c in bind_df.columns if c.endswith("_bind")):
            delta = other.loc[common, col] - baseline.loc[common, col]
            rows.append(
                {
                    "variant": variant,
                    "component": col.removesuffix("_bind"),
                    "mean_delta_kcal_mol": float(delta.mean()),
                    "std_delta_kcal_mol": float(delta.std(ddof=1)),
                    "min_delta_kcal_mol": float(delta.min()),
                    "max_delta_kcal_mol": float(delta.max()),
                }
            )
        diff_delta = other.loc[common, "diff_vs_ref"] - baseline.loc[common, "diff_vs_ref"]
        rows.append(
            {
                "variant": variant,
                "component": "diff_vs_ref",
                "mean_delta_kcal_mol": float(diff_delta.mean()),
                "std_delta_kcal_mol": float(diff_delta.std(ddof=1)),
                "min_delta_kcal_mol": float(diff_delta.min()),
                "max_delta_kcal_mol": float(diff_delta.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "mean_delta_kcal_mol"])


def fit_summary(bind_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in bind_df.groupby("variant"):
        x = sub["ref"].to_numpy(dtype=float)
        y = sub["total_bind"].to_numpy(dtype=float)
        diff = y - x
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "variant": variant,
                "N": len(sub),
                "RMSE": float(np.sqrt(np.mean(diff**2))),
                "diff_mean": float(diff.mean()),
                "diff_std": float(diff.std(ddof=1)),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-slope", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dftb-bin", default=shutil.which("dftb+") or "dftb+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_df = pd.read_csv(TARGET_CSV).dropna(subset=["run102|BSSE=yes"])
    structures = list(target_df["structure"])

    slope_input = WORK_DIR / "results_slope" / "best_hsd_inputs"
    slope_output = WORK_DIR / "results_slope" / "component_runs"
    if args.run_slope:
        run_dftb_inputs(slope_input, slope_output, args.workers, args.dftb_bin, args.timeout, args.threads)

    system_frames = [
        read_variant_outputs("baseline", RUN110_DIR, structures),
        read_variant_outputs("slope", slope_output, structures),
    ]
    system_df = pd.concat(system_frames, ignore_index=True)
    bind_df = binding_components(system_df, target_df)
    out_dir = WORK_DIR / "component_diagnostics"
    out_dir.mkdir(exist_ok=True)
    system_df.to_csv(out_dir / "component_system_energies.csv", index=False)
    bind_df.to_csv(out_dir / "component_binding_energies.csv", index=False)
    component_delta_summary(bind_df).to_csv(out_dir / "slope_minus_baseline_component_summary.csv", index=False)
    fit_summary(bind_df).to_csv(out_dir / "variant_fit_summary.csv", index=False)
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
