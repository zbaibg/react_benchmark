#!/usr/bin/env python3
"""Plot 1Hbond Mulliken charge scans for selected QM/MM partitions."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parent
DISTANCE_CSV = ROOT / "with_zn_minus_without_zn.csv"
RUN_CONFIGS = ROOT / "run_configs.yaml"
BASE_DIR = ROOT / "SP_init"
COMPLEX_WITH_ZN = "1Zn_1ImH_6Wat_1Hbond"
COMPLEX_WITHOUT_ZN = "1ImH_6Wat_1Hbond"
MULLIKEN_ROW_RE = re.compile(
    r"^\s*(?P<index>\d+)\s+(?P<element>[A-Za-z][A-Za-z]?)\s*:"
    r"\s*(?P<charge>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
DFTB_CHARGE_ROW_RE = re.compile(
    r"^\s*(?P<atom>\d+)\s+"
    r"(?P<charge>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*$"
)
SCAN_RE = re.compile(r"_IMH_(?P<label>m?\d+(?:\.\d+)?)_full$")
AMBER_CHARGE_SCALE = 18.2223
FULL_QM_RUNS = {"run41", "run110"}
COMPONENT_COLORS = {
    "Mulliken Charge of IMH": "tab:blue",
    "Mulliken Charge of Zn": "tab:orange",
    "Mulliken Charge of 6Wat": "tab:green",
    "Mulliken Charge of Zn+6Wat": "tab:red",
}


@dataclass(frozen=True)
class AtomCharge:
    index: int
    element: str
    charge: float


@dataclass(frozen=True)
class ScanPoint:
    run: str
    scan_value: float
    zn_n_distance: float
    imh_total_charge: Optional[float] = None
    zn_charge: Optional[float] = None
    wat6_charge: Optional[float] = None
    zn_6wat_charge: Optional[float] = None
    output_path: Optional[Path] = None


def parse_scan_label(label: str) -> float:
    if label.startswith("m"):
        return -float(label[1:])
    return float(label)


def scan_value_from_dirname(dirname: str) -> float:
    match = SCAN_RE.search(dirname)
    if match is None:
        return 0.0
    return parse_scan_label(match.group("label"))


def load_distances(path: Path = DISTANCE_CSV) -> Dict[float, float]:
    distances: Dict[float, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row.get("Run") == "run41"
                and row.get("Complex_with_zn") == COMPLEX_WITH_ZN
                and row.get("BSSE") == "no"
            ):
                distances[float(row["scan_value"])] = float(row["Zn_N_distance"])
    if not distances:
        raise ValueError(f"No 1Hbond Zn-N distances found in {path}")
    return distances


def load_run_names(path: Path = RUN_CONFIGS) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    return {
        str(run): str(value["name"])
        for run, value in config.items()
        if str(run).startswith("run") and isinstance(value, dict) and value.get("name")
    }


def iter_scan_dirs(
    run: str,
    complex_name: str = COMPLEX_WITH_ZN,
    base_dir: Path = BASE_DIR,
) -> Iterable[Path]:
    run_dir = base_dir / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    yield from sorted(run_dir.glob(f"{complex_name}*_full"))


def parse_mulliken_charges(path: Path) -> List[AtomCharge]:
    charges: List[AtomCharge] = []
    in_block = False
    saw_row = False

    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip().startswith("MULLIKEN ATOMIC CHARGES"):
                charges = []
                in_block = True
                saw_row = False
                continue

            if not in_block:
                continue

            if "Sum of atomic charges" in line:
                return charges

            match = MULLIKEN_ROW_RE.match(line)
            if match:
                saw_row = True
                charges.append(
                    AtomCharge(
                        index=int(match.group("index")),
                        element=match.group("element"),
                        charge=float(match.group("charge")),
                    )
                )
                continue

            if saw_row and not line.strip():
                in_block = False

    raise ValueError(f"No complete Mulliken charge block found in {path}")


def parse_dftb_charges(path: Path) -> List[AtomCharge]:
    charges: List[AtomCharge] = []
    in_block = False
    saw_row = False

    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip().startswith("Atomic gross charges"):
                charges = []
                in_block = True
                saw_row = False
                continue

            if not in_block:
                continue

            match = DFTB_CHARGE_ROW_RE.match(line)
            if match:
                saw_row = True
                charges.append(
                    AtomCharge(
                        index=int(match.group("atom")) - 1,
                        element="",
                        charge=float(match.group("charge")),
                    )
                )
                continue

            if saw_row and not line.strip():
                return charges

    if charges:
        return charges
    raise ValueError(f"No DFTB Atomic gross charges block found in {path}")


def parse_mm_charges(path: Path) -> List[AtomCharge]:
    raw_charges: List[float] = []
    in_charge_block = False

    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("%FLAG "):
                if in_charge_block:
                    break
                in_charge_block = line.split()[1] == "CHARGE"
                continue

            if not in_charge_block or line.startswith("%FORMAT"):
                continue

            raw_charges.extend(float(token) for token in line.split())

    if not raw_charges:
        raise ValueError(f"No AMBER CHARGE block found in {path}")

    return [
        AtomCharge(index=index, element="", charge=raw_charge / AMBER_CHARGE_SCALE)
        for index, raw_charge in enumerate(raw_charges)
    ]


def parse_charges_for_dir(scan_dir: Path, preferred: str) -> Tuple[List[AtomCharge], Path]:
    if preferred == "orca":
        for name in ("old.orc_job.dat", "orc_job.dat"):
            output_path = scan_dir / name
            if output_path.exists():
                return parse_mulliken_charges(output_path), output_path
        raise FileNotFoundError(f"No ORCA output found in {scan_dir}")

    if preferred == "dftb_or_mm":
        output_path = scan_dir / "detailed.out"
        if output_path.exists():
            return parse_dftb_charges(output_path), output_path
        output_path = scan_dir / "box.prmtop"
        if output_path.exists():
            return parse_mm_charges(output_path), output_path
        raise FileNotFoundError(f"No detailed.out or box.prmtop found in {scan_dir}")

    raise ValueError(f"Unknown charge parser preference: {preferred}")


def imh_indices_for_run(
    run: str,
    charges: Sequence[AtomCharge],
    has_zn: bool = True,
) -> List[int]:
    if not has_zn:
        if run in {"run41", "run110"}:
            # Without-Zn full-system ordering: atoms 0-8 are IMH, then 6 waters.
            return list(range(0, 9))
        if run in {"run113", "run118"}:
            # QM-IMH/MM-Wat contains only IMH atoms.
            return list(range(0, 9))
        raise ValueError(f"Without-Zn IMH indices are not defined for {run}")

    if run in {"run0", "run41", "run110"}:
        # Full-system ordering: atom 0 is Zn and atoms 1-9 are IMH.
        return list(range(1, 10))
    if run in {"run113", "run118"}:
        # In QM-IMH/MM-Zn,Wat, the quantum system contains only IMH atoms.
        return list(range(0, 9))
    raise ValueError(f"IMH indices are not defined for {run}")


def wat6_indices_for_run(
    run: str,
    charges: Sequence[AtomCharge],
    has_zn: bool = True,
) -> List[int]:
    if not has_zn:
        if run in {"run41", "run110"}:
            # Without-Zn full-system ordering: IMH first, then 6 waters.
            return list(range(9, len(charges)))
        if run in {"run115", "run117"}:
            # QM-Wat/MM-IMH contains only the six waters.
            return list(range(0, len(charges)))
        raise ValueError(f"Without-Zn 6Wat indices are not defined for {run}")

    if run in {"run0", "run41", "run110"}:
        # Full-system ordering: Zn, 9 IMH atoms, then 6 waters.
        return list(range(10, len(charges)))
    if run in {"run115", "run117"}:
        # QM-Zn,Wat/MM-IMH contains Zn followed by the six waters.
        return list(range(1, len(charges)))
    raise ValueError(f"6Wat indices are not defined for {run}")


def sum_charge(charges: Sequence[AtomCharge], indices: Sequence[int]) -> float:
    charge_by_index = {atom.index: atom.charge for atom in charges}
    missing = [index for index in indices if index not in charge_by_index]
    if missing:
        raise ValueError(f"Missing Mulliken charges for atom indices: {missing}")
    return sum(charge_by_index[index] for index in indices)


def collect_imh_points(
    runs: Sequence[str],
    distances: Dict[float, float],
    preferred: str,
    complex_name: str = COMPLEX_WITH_ZN,
    has_zn: bool = True,
) -> List[ScanPoint]:
    points: List[ScanPoint] = []
    for run in runs:
        for scan_dir in iter_scan_dirs(run, complex_name=complex_name):
            scan_value = scan_value_from_dirname(scan_dir.name)
            if scan_value not in distances:
                continue
            charges, output_path = parse_charges_for_dir(scan_dir, preferred)
            imh_indices = imh_indices_for_run(run, charges, has_zn=has_zn)
            points.append(
                ScanPoint(
                    run=run,
                    scan_value=scan_value,
                    zn_n_distance=distances[scan_value],
                    imh_total_charge=sum_charge(charges, imh_indices),
                    output_path=output_path,
                )
            )
    return sorted(points, key=lambda point: (point.run, point.zn_n_distance))


def collect_zn_points(
    runs: Sequence[str],
    distances: Dict[float, float],
    preferred: str,
    complex_name: str = COMPLEX_WITH_ZN,
    has_zn: bool = True,
) -> List[ScanPoint]:
    points: List[ScanPoint] = []
    for run in runs:
        for scan_dir in iter_scan_dirs(run, complex_name=complex_name):
            scan_value = scan_value_from_dirname(scan_dir.name)
            if scan_value not in distances:
                continue
            charges, output_path = parse_charges_for_dir(scan_dir, preferred)
            wat6_indices = wat6_indices_for_run(run, charges, has_zn=has_zn)
            points.append(
                ScanPoint(
                    run=run,
                    scan_value=scan_value,
                    zn_n_distance=distances[scan_value],
                    zn_charge=charges[0].charge if has_zn else None,
                    wat6_charge=sum_charge(charges, wat6_indices),
                    zn_6wat_charge=(
                        charges[0].charge + sum_charge(charges, wat6_indices) if has_zn else None
                    ),
                    output_path=output_path,
                )
            )
    return sorted(points, key=lambda point: (point.run, point.zn_n_distance))


def write_csv(path: Path, points: Sequence[ScanPoint]) -> None:
    fieldnames = [
        "run",
        "scan_value",
        "zn_n_distance",
        "imh_total_charge",
        "zn_charge",
        "wat6_charge",
        "zn_6wat_charge",
        "output_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "run": point.run,
                    "scan_value": point.scan_value,
                    "zn_n_distance": point.zn_n_distance,
                    "imh_total_charge": point.imh_total_charge,
                    "zn_charge": point.zn_charge,
                    "wat6_charge": point.wat6_charge,
                    "zn_6wat_charge": point.zn_6wat_charge,
                    "output_path": point.output_path,
                }
            )


def _plot_by_run(
    ax: plt.Axes,
    points: Sequence[ScanPoint],
    runs: Sequence[str],
    value_attr: str,
    label_suffix: str,
    marker: str,
    run_names: Dict[str, str],
) -> None:
    for run in runs:
        run_points = [point for point in points if point.run == run]
        run_label = f"{run} {run_names.get(run, run)}"
        linestyle = "--" if run in FULL_QM_RUNS else "-"
        ax.plot(
            [point.zn_n_distance for point in run_points],
            [getattr(point, value_attr) for point in run_points],
            color=COMPONENT_COLORS[label_suffix],
            linestyle=linestyle,
            marker="^" if run in FULL_QM_RUNS else "o",
            linewidth=1.8,
            label=f"{label_suffix}, {run_label}",
        )


def set_padded_ylim(ax: plt.Axes, values: Sequence[Optional[float]], pad_fraction: float = 0.08) -> None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return

    ymin = min(numeric_values)
    ymax = max(numeric_values)
    if ymin == ymax:
        pad = max(abs(ymin) * pad_fraction, 0.05)
    else:
        pad = (ymax - ymin) * pad_fraction
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_combined_charges(
    zn_points: Sequence[ScanPoint],
    imh_points: Sequence[ScanPoint],
    imh_runs: Sequence[str],
    zn_runs: Sequence[str],
    output_path: Path,
    run_names: Dict[str, str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _plot_by_run(
        ax,
        imh_points,
        imh_runs,
        "imh_total_charge",
        "Mulliken Charge of IMH",
        "^",
        run_names,
    )
    _plot_by_run(ax, zn_points, zn_runs, "zn_charge", "Mulliken Charge of Zn", "o", run_names)
    _plot_by_run(
        ax,
        zn_points,
        zn_runs,
        "wat6_charge",
        "Mulliken Charge of 6Wat",
        "s",
        run_names,
    )
    _plot_by_run(
        ax,
        zn_points,
        zn_runs,
        "zn_6wat_charge",
        "Mulliken Charge of Zn+6Wat",
        "D",
        run_names,
    )
    set_padded_ylim(
        ax,
        [point.imh_total_charge for point in imh_points]
        + [point.zn_charge for point in zn_points]
        + [point.wat6_charge for point in zn_points]
        + [point.zn_6wat_charge for point in zn_points],
    )
    ax.set_xlabel("Zn-N distance (Angstrom)")
    ax.set_ylabel("Mulliken charge (e)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_without_zn_charges(
    wat_points: Sequence[ScanPoint],
    imh_points: Sequence[ScanPoint],
    imh_runs: Sequence[str],
    wat_runs: Sequence[str],
    output_path: Path,
    run_names: Dict[str, str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _plot_by_run(
        ax,
        imh_points,
        imh_runs,
        "imh_total_charge",
        "Mulliken Charge of IMH",
        "^",
        run_names,
    )
    _plot_by_run(
        ax,
        wat_points,
        wat_runs,
        "wat6_charge",
        "Mulliken Charge of 6Wat",
        "s",
        run_names,
    )
    set_padded_ylim(
        ax,
        [point.imh_total_charge for point in imh_points]
        + [point.wat6_charge for point in wat_points],
    )
    ax.set_xlabel("Original Zn-N distance before Zn deletion (Angstrom)")
    ax.set_ylabel("Mulliken charge (e)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 1Hbond Mulliken charges for run41/run118 and run41/run117."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT,
        help="Directory for generated PNG and CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    distances = load_distances()
    run_names = load_run_names()
    orca_imh_runs = ["run41", "run118"]
    orca_zn_runs = ["run41", "run117"]
    orca_imh_points = collect_imh_points(orca_imh_runs, distances, preferred="orca")
    orca_zn_points = collect_zn_points(orca_zn_runs, distances, preferred="orca")
    orca_csv = args.out_dir / "mulliken_1Hbond_combined_charge_scan.csv"
    orca_png = args.out_dir / "mulliken_1Hbond_zn_6wat_scan.png"
    write_csv(
        orca_csv,
        sorted([*orca_imh_points, *orca_zn_points], key=lambda point: (point.run, point.zn_n_distance)),
    )
    plot_combined_charges(
        orca_zn_points,
        orca_imh_points,
        orca_imh_runs,
        orca_zn_runs,
        orca_png,
        run_names,
        "1Hbond M05-2X Mulliken charge scan",
    )

    orca_without_zn_imh_points = collect_imh_points(
        orca_imh_runs,
        distances,
        preferred="orca",
        complex_name=COMPLEX_WITHOUT_ZN,
        has_zn=False,
    )
    orca_without_zn_wat_points = collect_zn_points(
        orca_zn_runs,
        distances,
        preferred="orca",
        complex_name=COMPLEX_WITHOUT_ZN,
        has_zn=False,
    )
    orca_without_zn_csv = args.out_dir / "mulliken_1Hbond_without_zn_charge_scan.csv"
    orca_without_zn_png = args.out_dir / "mulliken_1Hbond_without_zn_charge_scan.png"
    write_csv(
        orca_without_zn_csv,
        sorted(
            [*orca_without_zn_imh_points, *orca_without_zn_wat_points],
            key=lambda point: (point.run, point.zn_n_distance),
        ),
    )
    plot_without_zn_charges(
        orca_without_zn_wat_points,
        orca_without_zn_imh_points,
        orca_imh_runs,
        orca_zn_runs,
        orca_without_zn_png,
        run_names,
        "1Hbond M05-2X Mulliken charge scan without Zn",
    )

    dftb_imh_runs = ["run113", "run110"]
    dftb_zn_runs = ["run115", "run110"]
    dftb_imh_points = collect_imh_points(dftb_imh_runs, distances, preferred="dftb_or_mm")
    dftb_zn_points = collect_zn_points(dftb_zn_runs, distances, preferred="dftb_or_mm")
    dftb_csv = args.out_dir / "mulliken_1Hbond_dftb_charge_scan.csv"
    dftb_png = args.out_dir / "mulliken_1Hbond_dftb_charge_scan.png"
    write_csv(
        dftb_csv,
        sorted([*dftb_imh_points, *dftb_zn_points], key=lambda point: (point.run, point.zn_n_distance)),
    )
    plot_combined_charges(
        dftb_zn_points,
        dftb_imh_points,
        dftb_imh_runs,
        dftb_zn_runs,
        dftb_png,
        run_names,
        "1Hbond DFTB/MM charge scan",
    )

    dftb_without_zn_imh_points = collect_imh_points(
        dftb_imh_runs,
        distances,
        preferred="dftb_or_mm",
        complex_name=COMPLEX_WITHOUT_ZN,
        has_zn=False,
    )
    dftb_without_zn_wat_points = collect_zn_points(
        dftb_zn_runs,
        distances,
        preferred="dftb_or_mm",
        complex_name=COMPLEX_WITHOUT_ZN,
        has_zn=False,
    )
    dftb_without_zn_csv = args.out_dir / "mulliken_1Hbond_dftb_without_zn_charge_scan.csv"
    dftb_without_zn_png = args.out_dir / "mulliken_1Hbond_dftb_without_zn_charge_scan.png"
    write_csv(
        dftb_without_zn_csv,
        sorted(
            [*dftb_without_zn_imh_points, *dftb_without_zn_wat_points],
            key=lambda point: (point.run, point.zn_n_distance),
        ),
    )
    plot_without_zn_charges(
        dftb_without_zn_wat_points,
        dftb_without_zn_imh_points,
        dftb_imh_runs,
        dftb_zn_runs,
        dftb_without_zn_png,
        run_names,
        "1Hbond DFTB/MM charge scan without Zn",
    )

    print(f"Wrote {orca_png}")
    print(f"Wrote {orca_csv}")
    print(f"Wrote {orca_without_zn_png}")
    print(f"Wrote {orca_without_zn_csv}")
    print(f"Wrote {dftb_png}")
    print(f"Wrote {dftb_csv}")
    print(f"Wrote {dftb_without_zn_png}")
    print(f"Wrote {dftb_without_zn_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
