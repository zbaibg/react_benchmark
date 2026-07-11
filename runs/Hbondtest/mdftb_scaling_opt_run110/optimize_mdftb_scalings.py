#!/usr/bin/env python3
"""Optimize mDFTB multipole scaling parameters for run110 vs run102 reference.

All DFTB+ jobs are launched from temporary directories under this folder. The
original SP_init/run110 directories are read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, differential_evolution, minimize


HARTREE_TO_KCAL_MOL = 627.5094740631

WORK_DIR = Path(__file__).resolve().parent
ROOT_DIR = WORK_DIR.parent
RUN110_DIR = ROOT_DIR / "SP_init" / "run110"
TARGET_CSV = ROOT_DIR / "Ebind.csv"
RESULTS_DIR = Path(os.environ.get("MDFTB_OPT_RESULTS_DIR", WORK_DIR / "results")).resolve()
TMP_DIR = Path(os.environ.get("MDFTB_OPT_TMP_DIR", WORK_DIR / "tmp")).resolve()

ATOM_ORDER = ("H", "C", "N", "O")
PARAM_NAMES = (
    "D_H",
    "D_C",
    "D_N",
    "D_O",
    "Q_H",
    "Q_C",
    "Q_N",
    "Q_O",
)
BASE_VALUES = {
    "D": {"H": 1.0, "C": 0.8, "N": 0.6, "O": 0.2},
    "Q": {"H": 1.0, "C": 2.6, "N": 3.4, "O": 3.0},
}
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=float)
UPPER_BOUNDS = np.array([2.5, 2.5, 2.5, 2.5, 4.0, 5.0, 5.0, 5.0], dtype=float)

TOTAL_ENERGY_RE = re.compile(
    r"Total\s+Energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H"
)


def base_vector() -> np.ndarray:
    return np.array(
        [*(BASE_VALUES["D"][atom] for atom in ATOM_ORDER), *(BASE_VALUES["Q"][atom] for atom in ATOM_ORDER)],
        dtype=float,
    )


def unpack_params(x: Iterable[float]) -> Dict[str, Dict[str, float]]:
    values = list(float(v) for v in x)
    if len(values) != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} parameters, got {len(values)}")
    return {
        "D": dict(zip(ATOM_ORDER, values[:4])),
        "Q": dict(zip(ATOM_ORDER, values[4:])),
    }


def param_record(x: Iterable[float]) -> Dict[str, float]:
    return dict(zip(PARAM_NAMES, (float(v) for v in x)))


def params_key(x: Iterable[float]) -> str:
    rounded = ",".join(f"{float(v):.8f}" for v in x)
    return hashlib.sha1(rounded.encode("ascii")).hexdigest()[:12]


def replace_scaling_block(text: str, block_name: str, values: Dict[str, float]) -> str:
    block_re = re.compile(
        rf"({re.escape(block_name)}\s*=\s*\{{)(.*?)(\n\s*\}})",
        flags=re.DOTALL,
    )
    match = block_re.search(text)
    if match is None:
        raise ValueError(f"missing {block_name} block")

    body = match.group(2)
    for atom, value in values.items():
        line_re = re.compile(
            rf"(^\s*{re.escape(atom)}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?",
            flags=re.MULTILINE,
        )
        body, count = line_re.subn(lambda m, v=value: f"{m.group(1)}{v:.16g}", body)
        if count != 1:
            raise ValueError(f"expected one {atom} line in {block_name}, found {count}")

    return text[: match.start()] + match.group(1) + body + match.group(3) + text[match.end() :]


def parse_type_names(text: str) -> set[str]:
    match = re.search(r"TypeNames\s*=\s*\{(.*?)\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("missing Geometry/TypeNames block")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def filter_block_lines(text: str, block_name: str, keep_line) -> str:
    block_re = re.compile(
        rf"({re.escape(block_name)}\s*=\s*\{{)(.*?)(\n\s*\}})",
        flags=re.DOTALL,
    )
    match = block_re.search(text)
    if match is None:
        return text
    body = match.group(2)
    new_lines = [line for line in body.splitlines(keepends=True) if keep_line(line)]
    return (
        text[: match.start()]
        + match.group(1)
        + "".join(new_lines)
        + match.group(3)
        + text[match.end() :]
    )


def prune_mdftb_entries_to_type_names(text: str) -> str:
    type_names = parse_type_names(text)

    def keep_one_center_line(line: str) -> bool:
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*:", line)
        return match is None or match.group(1) in type_names

    def keep_scaling_line(line: str) -> bool:
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*=", line)
        return match is None or match.group(1) in type_names

    text = filter_block_lines(text, "OneCenterAtomIntegrals", keep_one_center_line)
    text = filter_block_lines(text, "AtomDIntegralScalings", keep_scaling_line)
    text = filter_block_lines(text, "AtomQIntegralScalings", keep_scaling_line)
    return text


def make_hsd(template: str, x: Iterable[float]) -> str:
    values = unpack_params(x)
    text = replace_scaling_block(template, "AtomDIntegralScalings", values["D"])
    text = replace_scaling_block(text, "AtomQIntegralScalings", values["Q"])
    return prune_mdftb_entries_to_type_names(text)


def load_targets() -> pd.DataFrame:
    df = pd.read_csv(TARGET_CSV)
    required = {"structure", "run102|BSSE=yes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{TARGET_CSV} missing columns: {sorted(missing)}")
    return df.dropna(subset=["run102|BSSE=yes"]).copy()


def load_templates(structures: Iterable[str]) -> Dict[str, str]:
    names = ["ImH_monomer", "Wat_monomer"] + [f"{structure}_full" for structure in structures]
    templates: Dict[str, str] = {}
    for name in names:
        path = RUN110_DIR / name / "dftb_pin.hsd"
        if not path.exists():
            raise FileNotFoundError(path)
        templates[name] = path.read_text()
    return templates


def parse_total_energy(output: str, detailed_out: Path) -> float:
    match = TOTAL_ENERGY_RE.search(output)
    if match is not None:
        return float(match.group(1))
    if detailed_out.exists():
        text = detailed_out.read_text(errors="replace")
        match = re.search(
            r"Total energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+H",
            text,
        )
        if match is not None:
            return float(match.group(1))
    raise ValueError("could not parse DFTB+ total energy")


def run_dftb_energy(
    name: str,
    hsd_text: str,
    dftb_bin: str,
    timeout_s: int,
    threads: int,
) -> Tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix=f"{name}.", dir=TMP_DIR) as scratch:
        scratch_path = Path(scratch)
        (scratch_path / "dftb_in.hsd").write_text(hsd_text)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [dftb_bin],
            cwd=scratch_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-40:])
            raise RuntimeError(f"{name}: dftb+ failed with return code {proc.returncode}\n{tail}")
        return name, parse_total_energy(proc.stdout, scratch_path / "detailed.out")


class Objective:
    def __init__(
        self,
        target_df: pd.DataFrame,
        templates: Dict[str, str],
        dftb_bin: str,
        workers: int,
        timeout_s: int,
        threads: int,
        objective_name: str,
        rmse_scale: float,
        slope_scale: float,
        mean_scale: float,
    ) -> None:
        self.target_df = target_df
        self.templates = templates
        self.dftb_bin = dftb_bin
        self.workers = workers
        self.timeout_s = timeout_s
        self.threads = threads
        self.objective_name = objective_name
        self.rmse_scale = rmse_scale
        self.slope_scale = slope_scale
        self.mean_scale = mean_scale
        self.cache: Dict[str, Tuple[float, Dict[str, object]]] = {}
        self.best: Tuple[float, np.ndarray, Dict[str, object]] | None = None
        self.history_path = RESULTS_DIR / "objective_history.csv"
        self.eval_count = 0

    def __call__(self, x: np.ndarray) -> float:
        if np.any(x < LOWER_BOUNDS) or np.any(x > UPPER_BOUNDS):
            return 1.0e6
        key = params_key(x)
        if key in self.cache:
            return self.cache[key][0]

        self.eval_count += 1
        started = time.time()
        try:
            metrics = self.evaluate(x)
            objective = self.objective_value(metrics)
        except Exception as exc:
            metrics = {
                "objective": 1.0e6,
                "RMSE": 1.0e6,
                "diff_mean": math.nan,
                "mean_error": math.nan,
                "diff_std": math.nan,
                "pearson_r": math.nan,
                "slope": math.nan,
                "slope_error": math.nan,
                "intercept": math.nan,
                "error": str(exc).replace("\n", " | "),
            }
            objective = 1.0e6

        metrics["objective"] = objective
        metrics["elapsed_s"] = time.time() - started
        metrics["eval"] = self.eval_count
        metrics["key"] = key
        metrics.update(param_record(x))
        self.append_history(metrics)
        self.cache[key] = (objective, metrics)

        if self.best is None or objective < self.best[0]:
            self.best = (objective, np.array(x, dtype=float), metrics)
            print(
                f"eval {self.eval_count:03d}: new best {self.objective_name}={objective:.6f} "
                f"RMSE={metrics.get('RMSE', math.nan):.6f} "
                f"slope={metrics.get('slope', math.nan):.6f} "
                f"diff={metrics.get('diff_mean', math.nan):+.6f} "
                f"key={key}",
                flush=True,
            )
        else:
            print(
                f"eval {self.eval_count:03d}: {self.objective_name}={objective:.6f} "
                f"RMSE={metrics.get('RMSE', math.nan):.6f} "
                f"slope={metrics.get('slope', math.nan):.6f} "
                f"best={self.best[0]:.6f}",
                flush=True,
            )
        return objective

    def objective_value(self, metrics: Dict[str, object]) -> float:
        if self.objective_name == "rmse":
            return float(metrics["RMSE"])
        if self.objective_name == "slope":
            return abs(float(metrics["slope"]) - 1.0)
        if self.objective_name == "mean":
            return abs(float(metrics["diff_mean"]))
        if self.objective_name == "slope_mean":
            slope_term = abs(float(metrics["slope"]) - 1.0) / self.slope_scale
            mean_term = abs(float(metrics["diff_mean"])) / self.mean_scale
            return float(np.sqrt(slope_term**2 + mean_term**2))
        if self.objective_name == "rmse_slope_mean":
            rmse_term = float(metrics["RMSE"]) / self.rmse_scale
            slope_term = abs(float(metrics["slope"]) - 1.0) / self.slope_scale
            mean_term = abs(float(metrics["diff_mean"])) / self.mean_scale
            return float(np.sqrt(rmse_term**2 + slope_term**2 + mean_term**2))
        raise ValueError(f"unsupported objective: {self.objective_name}")

    def evaluate(self, x: np.ndarray) -> Dict[str, object]:
        hsd_by_name = {name: make_hsd(template, x) for name, template in self.templates.items()}
        energies: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(
                    run_dftb_energy,
                    name,
                    hsd_text,
                    self.dftb_bin,
                    self.timeout_s,
                    self.threads,
                )
                for name, hsd_text in hsd_by_name.items()
            ]
            for future in as_completed(futures):
                name, energy = future.result()
                energies[name] = energy

        e_imh = energies["ImH_monomer"]
        e_wat = energies["Wat_monomer"]
        predicted = []
        for structure in self.target_df["structure"]:
            e_complex = energies[f"{structure}_full"]
            predicted.append((e_complex - e_imh - e_wat) * HARTREE_TO_KCAL_MOL)

        target = self.target_df["run102|BSSE=yes"].to_numpy(dtype=float)
        pred = np.array(predicted, dtype=float)
        diff = pred - target
        slope, intercept = np.polyfit(target, pred, 1)
        return {
            "N": len(diff),
            "RMSE": float(np.sqrt(np.mean(diff**2))),
            "diff_mean": float(np.mean(diff)),
            "mean_error": float(abs(np.mean(diff))),
            "diff_std": float(np.std(diff, ddof=1)),
            "pearson_r": float(np.corrcoef(target, pred)[0, 1]),
            "slope": float(slope),
            "slope_error": float(abs(slope - 1.0)),
            "intercept": float(intercept),
            "error": "",
        }

    def append_history(self, row: Dict[str, object]) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        exists = self.history_path.exists()
        fieldnames = [
            "eval",
            "key",
            "objective",
            "RMSE",
            "diff_mean",
            "mean_error",
            "diff_std",
            "pearson_r",
            "slope",
            "slope_error",
            "intercept",
            "elapsed_s",
            *PARAM_NAMES,
            "error",
        ]
        with self.history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    def ebind_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        hsd_by_name = {name: make_hsd(template, x) for name, template in self.templates.items()}
        energies: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(
                    run_dftb_energy,
                    name,
                    hsd_text,
                    self.dftb_bin,
                    self.timeout_s,
                    self.threads,
                )
                for name, hsd_text in hsd_by_name.items()
            ]
            for future in as_completed(futures):
                name, energy = future.result()
                energies[name] = energy

        e_imh = energies["ImH_monomer"]
        e_wat = energies["Wat_monomer"]
        out = self.target_df.copy()
        pred = []
        for structure in out["structure"]:
            e_complex = energies[f"{structure}_full"]
            pred.append((e_complex - e_imh - e_wat) * HARTREE_TO_KCAL_MOL)
        out["optimized|BSSE=no"] = pred
        out["optimized_minus_ref"] = out["optimized|BSSE=no"] - out["run102|BSSE=yes"]
        return out


def summarize_by_contact(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    records = []
    for contact_class, sub in df.groupby("contact_class", dropna=False):
        diff = sub[pred_col] - sub["run102|BSSE=yes"]
        records.append(
            {
                "contact_class": contact_class,
                "N": len(sub),
                "diff_mean": float(diff.mean()),
                "diff_std": float(diff.std(ddof=1)) if len(sub) > 1 else math.nan,
                "RMSE": float(np.sqrt(np.mean(diff.to_numpy(dtype=float) ** 2))),
            }
        )
    records.sort(key=lambda row: row["RMSE"], reverse=True)
    all_diff = df[pred_col] - df["run102|BSSE=yes"]
    records.append(
        {
            "contact_class": "Overall",
            "N": len(df),
            "diff_mean": float(all_diff.mean()),
            "diff_std": float(all_diff.std(ddof=1)),
            "RMSE": float(np.sqrt(np.mean(all_diff.to_numpy(dtype=float) ** 2))),
        }
    )
    return pd.DataFrame(records)


def write_plot(df: pd.DataFrame, pred_col: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plot_df = df[["run102|BSSE=yes", pred_col, "contact_class"]].dropna()
    x = plot_df["run102|BSSE=yes"].to_numpy(dtype=float)
    y = plot_df[pred_col].to_numpy(dtype=float)
    diff = y - x
    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = np.corrcoef(x, y)[0, 1]
    rmse = np.sqrt(np.mean(diff**2))

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab20")
    for idx, contact_class in enumerate(sorted(plot_df["contact_class"].dropna().unique())):
        sub = plot_df[plot_df["contact_class"] == contact_class]
        ax.scatter(
            sub["run102|BSSE=yes"],
            sub[pred_col],
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
    ax.plot(
        fit_x,
        slope * fit_x + intercept,
        color="crimson",
        linewidth=1.5,
        label=f"fit: y = {slope:.3f}x {intercept:+.3f}",
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("run102|BSSE=yes (kcal/mol)")
    ax.set_ylabel(f"{pred_col} (kcal/mol)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03,
        0.97,
        (
            f"N = {len(plot_df)}\n"
            f"R = {pearson_r:.3f}\n"
            f"RMSE = {rmse:.3f} kcal/mol\n"
            f"diff = {diff.mean():+.3f} +/- {diff.std(ddof=1):.3f} kcal/mol"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, borderpad=0.4, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_best_inputs(x: np.ndarray, templates: Dict[str, str]) -> None:
    out_dir = RESULTS_DIR / "best_hsd_inputs"
    out_dir.mkdir(exist_ok=True)
    for name, template in templates.items():
        subdir = out_dir / name
        subdir.mkdir(exist_ok=True)
        (subdir / "dftb_in.hsd").write_text(make_hsd(template, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "powell", "de"), default="powell")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--maxfev", type=int, default=80)
    parser.add_argument("--de-maxiter", type=int, default=20)
    parser.add_argument("--de-popsize", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--dftb-bin", default=shutil.which("dftb+") or "dftb+")
    parser.add_argument(
        "--objective",
        choices=("rmse", "slope", "mean", "slope_mean", "rmse_slope_mean"),
        default="rmse",
    )
    parser.add_argument("--rmse-scale", type=float, default=0.4)
    parser.add_argument("--slope-scale", type=float, default=0.05)
    parser.add_argument("--mean-scale", type=float, default=0.25)
    parser.add_argument("--append-history", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    history_path = RESULTS_DIR / "objective_history.csv"
    if history_path.exists() and not args.append_history:
        history_path.unlink()

    target_df = load_targets()
    templates = load_templates(target_df["structure"])
    objective = Objective(
        target_df=target_df,
        templates=templates,
        dftb_bin=args.dftb_bin,
        workers=args.workers,
        timeout_s=args.timeout,
        threads=args.threads,
        objective_name=args.objective,
        rmse_scale=args.rmse_scale,
        slope_scale=args.slope_scale,
        mean_scale=args.mean_scale,
    )

    x0 = base_vector()
    print(f"Loaded {len(target_df)} structures plus 2 monomers.")
    print(f"Using dftb+ at {args.dftb_bin}")
    print("Baseline parameters:")
    print(json.dumps(param_record(x0), indent=2, sort_keys=True))

    baseline_objective = objective(x0)
    baseline_metrics = objective.cache[params_key(x0)][1]
    print(
        "Baseline from direct dftb+: "
        f"objective={baseline_objective:.6f}, "
        f"RMSE={baseline_metrics['RMSE']:.6f} kcal/mol, "
        f"slope={baseline_metrics['slope']:.6f}, "
        f"diff={baseline_metrics['diff_mean']:+.6f} kcal/mol"
    )

    if args.mode == "baseline":
        best_x = x0
    elif args.mode == "powell":
        result = minimize(
            objective,
            x0,
            method="Powell",
            bounds=Bounds(LOWER_BOUNDS, UPPER_BOUNDS),
            options={"maxfev": args.maxfev, "xtol": 1.0e-3, "ftol": 1.0e-4, "disp": True},
        )
        print(result)
        best_x = result.x
        if objective.best is not None and objective.best[0] < objective(best_x):
            best_x = objective.best[1]
    else:
        result = differential_evolution(
            objective,
            bounds=list(zip(LOWER_BOUNDS, UPPER_BOUNDS)),
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            seed=args.seed,
            polish=True,
            updating="immediate",
            workers=1,
            tol=0.01,
            atol=0.0,
            disp=True,
        )
        print(result)
        best_x = result.x
        if objective.best is not None and objective.best[0] < objective(best_x):
            best_x = objective.best[1]

    best_df = objective.ebind_dataframe(best_x)
    best_summary = summarize_by_contact(best_df, "optimized|BSSE=no")
    best_df.to_csv(RESULTS_DIR / "best_ebind.csv", index=False)
    best_summary.to_csv(RESULTS_DIR / "best_summary_by_contact.csv", index=False)
    write_plot(
        best_df,
        "optimized|BSSE=no",
        RESULTS_DIR / "best_correlation.png",
        "Optimized mDFTB scalings vs run102|BSSE=yes",
    )
    write_best_inputs(best_x, templates)

    best_payload = {
        "objective": args.objective,
        "mode": args.mode,
        "rmse_scale_kcal_mol": args.rmse_scale,
        "slope_scale": args.slope_scale,
        "mean_scale_kcal_mol": args.mean_scale,
        "baseline_objective": baseline_objective,
        "baseline_metrics": baseline_metrics,
        "best_parameters": param_record(best_x),
        "bounds": {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS)
        },
        "best_summary": best_summary.to_dict(orient="records"),
    }
    (RESULTS_DIR / "best_scalings.json").write_text(json.dumps(best_payload, indent=2))
    print("Best parameters:")
    print(json.dumps(param_record(best_x), indent=2, sort_keys=True))
    print("Summary:")
    print(best_summary.to_string(index=False))
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
