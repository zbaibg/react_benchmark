#!/usr/bin/env python3
"""Plot nearest water H distance to the Zn-bound IMH N in the 1Hbond scan."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
XYZ_DIR = ROOT / "xyz" / "xyz_files"
DISTANCE_CSV = ROOT / "with_zn_minus_without_zn.csv"
COMPLEX = "1Zn_1ImH_6Wat_1Hbond"
SCAN_RE = re.compile(r"_IMH_(?P<label>m?\d+(?:\.\d+)?)\.xyz$")


Atom = Tuple[str, float, float, float]


@dataclass(frozen=True)
class NHDistancePoint:
    scan_value: float
    zn_n_distance: float
    n_atom_index: int
    h_atom_index: int
    n_h_distance: float
    xyz_path: Path


def parse_scan_label(label: Optional[str]) -> float:
    if label is None:
        return 0.0
    if label.startswith("m"):
        return -float(label[1:])
    return float(label)


def scan_value_from_path(path: Path) -> float:
    match = SCAN_RE.search(path.name)
    if match is None:
        return 0.0
    return parse_scan_label(match.group("label"))


def read_xyz(path: Path) -> List[Atom]:
    lines = path.read_text().splitlines()
    natoms = int(lines[0].strip())
    atoms: List[Atom] = []
    for line in lines[2 : 2 + natoms]:
        elem, x, y, z = line.split()[:4]
        atoms.append((elem, float(x), float(y), float(z)))
    return atoms


def distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


def load_zn_n_distances() -> Dict[float, float]:
    distances: Dict[float, float] = {}
    with DISTANCE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row.get("Run") == "run41"
                and row.get("Complex_with_zn") == COMPLEX
                and row.get("BSSE") == "no"
            ):
                distances[float(row["scan_value"])] = float(row["Zn_N_distance"])
    return distances


def find_zn_index(atoms: List[Atom]) -> int:
    for index, atom in enumerate(atoms):
        if atom[0].upper() == "ZN":
            return index
    raise ValueError("No Zn atom found")


def find_imh_indices(atoms: List[Atom], zn_index: int) -> List[int]:
    start_index = next(
        index for index in range(zn_index + 1, len(atoms)) if atoms[index][0].upper() == "N"
    )
    end_index = len(atoms)
    for index in range(start_index + 1, len(atoms)):
        if atoms[index][0].upper() == "O":
            end_index = index
            break
    return list(range(start_index, end_index))


def collect_points() -> List[NHDistancePoint]:
    zn_n_distances = load_zn_n_distances()
    points: List[NHDistancePoint] = []

    for xyz_path in sorted(XYZ_DIR.glob(f"{COMPLEX}*.xyz")):
        scan_value = scan_value_from_path(xyz_path)
        if scan_value not in zn_n_distances:
            continue

        atoms = read_xyz(xyz_path)
        zn_index = find_zn_index(atoms)
        imh_indices = find_imh_indices(atoms, zn_index)
        imh_n_indices = [index for index in imh_indices if atoms[index][0].upper() == "N"]
        zn_bound_n_index = min(imh_n_indices, key=lambda index: distance(atoms[zn_index], atoms[index]))
        water_h_indices = [
            index
            for index in range(max(imh_indices) + 1, len(atoms))
            if atoms[index][0].upper() == "H"
        ]
        nearest_h_index = min(
            water_h_indices,
            key=lambda index: distance(atoms[zn_bound_n_index], atoms[index]),
        )

        points.append(
            NHDistancePoint(
                scan_value=scan_value,
                zn_n_distance=zn_n_distances[scan_value],
                n_atom_index=zn_bound_n_index + 1,
                h_atom_index=nearest_h_index + 1,
                n_h_distance=distance(atoms[zn_bound_n_index], atoms[nearest_h_index]),
                xyz_path=xyz_path,
            )
        )

    return sorted(points, key=lambda point: point.zn_n_distance)


def write_csv(points: List[NHDistancePoint], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scan_value",
                "zn_n_distance",
                "n_atom_index",
                "nearest_h_atom_index",
                "n_h_distance",
                "xyz_path",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "scan_value": point.scan_value,
                    "zn_n_distance": point.zn_n_distance,
                    "n_atom_index": point.n_atom_index,
                    "nearest_h_atom_index": point.h_atom_index,
                    "n_h_distance": point.n_h_distance,
                    "xyz_path": point.xyz_path,
                }
            )


def plot(points: List[NHDistancePoint], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    visible_points = [point for point in points if 1.0 <= point.n_h_distance <= 4.0]
    ax.plot(
        [point.zn_n_distance for point in points],
        [point.n_h_distance for point in points],
        marker="o",
        linewidth=1.8,
        color="tab:purple",
    )
    ax.set_xlabel("Zn-N distance (Angstrom)")
    ax.set_ylabel("Nearest N-H distance (Angstrom)")
    ax.set_ylim(1.0, 4.0)
    ax.set_yticks(np.arange(1.0, 4.0 + 0.1, 0.1))
    if visible_points:
        xmin = min(point.zn_n_distance for point in visible_points)
        xmax = max(point.zn_n_distance for point in visible_points)
        pad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
        ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_title("1Hbond nearest water H to Zn-bound IMH N")
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    points = collect_points()
    csv_path = ROOT / "1Hbond_nearest_n_h_distance_scan.csv"
    png_path = ROOT / "1Hbond_nearest_n_h_distance_scan.png"
    write_csv(points, csv_path)
    plot(points, png_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
