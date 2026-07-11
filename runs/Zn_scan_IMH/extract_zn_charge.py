#!/usr/bin/env python3
"""Extract Zn charges from mixed ORCA and DFTB SP_init runs."""

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple


SCAN_LABEL_RE = re.compile(r"^m?\d+(?:\.\d+)?$")
ORCA_CHARGE_ROW_RE = re.compile(
    r"^\s*(?P<index>\d+)\s+(?P<element>[A-Za-z][A-Za-z]?)\s*:"
    r"\s*(?P<charge>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
DFTB_CHARGE_ROW_RE = re.compile(
    r"^\s*(?P<atom>\d+)\s+"
    r"(?P<charge>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*$"
)
ORCA_FINAL_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+"
    r"(?P<energy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
ORCA_SCF_CONVERGED_RE = re.compile(r"SCF CONVERGED AFTER\s+(?P<cycles>\d+)\s+CYCLES")
DFTB_TOTAL_CHARGE_RE = re.compile(
    r"Total charge:\s+"
    r"(?P<charge>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
DFTB_TOTAL_ENERGY_RE = re.compile(
    r"Total energy:\s+"
    r"(?P<energy_hartree>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+H"
)
AMBER_CHARGE_SCALE = 18.2223
E_ANGSTROM_TO_DEBYE = 4.803204712570263


@dataclass
class ChargeRow:
    run: str
    method: str
    struct_dir: str
    complex_name: str
    scan_label: str
    scan_value: float
    target: str
    charge: Optional[float]
    charge_details: str
    total_charge: Optional[float]
    converged: bool
    cycles: Optional[int]
    energy_hartree: Optional[float]
    imh_total_charge: Optional[float]
    imh_dipole_debye: Optional[float]
    status: str
    output_path: str


def parse_scan_label(label: Optional[str]) -> Tuple[str, float]:
    if label is None:
        return "", 0.0
    if label.startswith("m"):
        return label, -float(label[1:])
    return label, float(label)


def parse_full_dir_name(dirname: str) -> Tuple[str, str, float]:
    stem = dirname[:-5] if dirname.endswith("_full") else dirname
    prefix, sep, last_token = stem.rpartition("_")
    if sep and SCAN_LABEL_RE.fullmatch(last_token):
        scan_label, scan_value = parse_scan_label(last_token)
        return prefix, scan_label, scan_value
    return stem, "", 0.0


def iter_run_dirs(base_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if name.startswith("run") and os.path.isdir(path):
            yield os.path.abspath(path)


def resolve_run_dir(run: str, base_dir: str) -> str:
    if os.path.isdir(run):
        return os.path.abspath(run)

    candidate = os.path.join(base_dir, run)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Could not find run directory '{run}' or '{candidate}'."
    )


def resolve_run_dirs(runs: Sequence[str], base_dir: str) -> List[str]:
    if runs:
        return [resolve_run_dir(run, base_dir) for run in runs]

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Could not find base directory '{base_dir}'.")

    return list(iter_run_dirs(base_dir))


def iter_structure_dirs(run_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if name.endswith("_full") and os.path.isdir(path):
            yield path


def parse_atom_numbers(values: Sequence[str]) -> List[int]:
    atom_numbers: List[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError(f"Invalid atom range '{part}'.")
                atom_numbers.extend(range(start, end + 1))
            else:
                atom_numbers.append(int(part))

    if not atom_numbers:
        raise ValueError("At least one DFTB atom number is required.")
    if any(atom_number < 1 for atom_number in atom_numbers):
        raise ValueError("DFTB atom numbers are 1-based and must be >= 1.")

    return sorted(dict.fromkeys(atom_numbers))


def _parse_float_from_line(line: str) -> Optional[float]:
    tokens = line.replace(":", " ").split()
    for token in reversed(tokens):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def parse_xyz_coordinates(path: str) -> Dict[int, Tuple[float, float, float]]:
    coordinates: Dict[int, Tuple[float, float, float]] = {}
    if not os.path.exists(path):
        return coordinates

    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return coordinates

    for atom_number, line in enumerate(lines[2:], start=1):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coordinates[atom_number] = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        except ValueError:
            continue
    return coordinates


def compute_fragment_dipole_debye(
    charges: Dict[int, float],
    coordinates: Dict[int, Tuple[float, float, float]],
    atom_numbers: Sequence[int],
) -> Optional[float]:
    if any(atom_number not in charges or atom_number not in coordinates for atom_number in atom_numbers):
        return None

    origin = (
        sum(coordinates[atom_number][0] for atom_number in atom_numbers) / len(atom_numbers),
        sum(coordinates[atom_number][1] for atom_number in atom_numbers) / len(atom_numbers),
        sum(coordinates[atom_number][2] for atom_number in atom_numbers) / len(atom_numbers),
    )
    dipole = [0.0, 0.0, 0.0]
    for atom_number in atom_numbers:
        charge = charges[atom_number]
        x, y, z = coordinates[atom_number]
        dipole[0] += charge * (x - origin[0])
        dipole[1] += charge * (y - origin[1])
        dipole[2] += charge * (z - origin[2])

    magnitude_e_angstrom = math.sqrt(sum(component * component for component in dipole))
    return magnitude_e_angstrom * E_ANGSTROM_TO_DEBYE


def compute_fragment_charge(
    charges: Dict[int, float],
    atom_numbers: Sequence[int],
) -> Optional[float]:
    if any(atom_number not in charges for atom_number in atom_numbers):
        return None
    return sum(charges[atom_number] for atom_number in atom_numbers)


def resolve_orca_output_path(struct_dir: str, dat_names: Sequence[str]) -> str:
    for dat_name in dat_names:
        candidate = os.path.join(struct_dir, dat_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(struct_dir, dat_names[0])


def detect_method(struct_dir: str, dat_names: Sequence[str]) -> Tuple[str, str]:
    detailed_path = os.path.join(struct_dir, "detailed.out")
    orca_path = resolve_orca_output_path(struct_dir, dat_names)
    prmtop_path = os.path.join(struct_dir, "box.prmtop")
    if os.path.exists(detailed_path):
        return "dftb", detailed_path
    if os.path.exists(orca_path):
        return "orca", orca_path
    if os.path.exists(prmtop_path):
        return "mm", prmtop_path
    return "missing", orca_path


def parse_orca_dat(path: str, atom: str, atom_index: int) -> Tuple[
    Optional[float],
    Dict[int, float],
    Optional[float],
    bool,
    Optional[int],
    Optional[float],
    str,
]:
    if not os.path.exists(path):
        return None, {}, None, False, None, None, "missing_output"

    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return None, {}, None, False, None, None, f"read_error:{exc}"

    mulliken_charge: Optional[float] = None
    all_charges: Dict[int, float] = {}
    sum_atomic_charges: Optional[float] = None
    scf_converged = False
    scf_cycles: Optional[int] = None
    final_energy_hartree: Optional[float] = None
    in_mulliken_block = False
    saw_mulliken_row = False
    atom_upper = atom.upper()

    for line in lines:
        cycle_match = ORCA_SCF_CONVERGED_RE.search(line)
        if cycle_match:
            scf_converged = True
            scf_cycles = int(cycle_match.group("cycles"))

        energy_match = ORCA_FINAL_ENERGY_RE.search(line)
        if energy_match:
            final_energy_hartree = float(energy_match.group("energy"))

        if line.strip().startswith("MULLIKEN ATOMIC CHARGES"):
            in_mulliken_block = True
            saw_mulliken_row = False
            continue

        if not in_mulliken_block:
            continue

        if "Sum of atomic charges" in line:
            sum_atomic_charges = _parse_float_from_line(line)
            in_mulliken_block = False
            continue

        row_match = ORCA_CHARGE_ROW_RE.match(line)
        if row_match:
            saw_mulliken_row = True
            element = row_match.group("element").upper()
            index = int(row_match.group("index"))
            charge = float(row_match.group("charge"))
            all_charges[index + 1] = charge
            if element == atom_upper and index == atom_index:
                mulliken_charge = charge
            continue

        if saw_mulliken_row and not line.strip():
            in_mulliken_block = False

    if mulliken_charge is None:
        if scf_converged:
            status = "missing_mulliken"
        else:
            status = "not_finished_or_no_mulliken"
    else:
        status = "ok"

    return (
        mulliken_charge,
        all_charges,
        sum_atomic_charges,
        scf_converged,
        scf_cycles,
        final_energy_hartree,
        status,
    )


def parse_dftb_detailed(path: str, atom_numbers: Sequence[int]) -> Tuple[
    Optional[float],
    Dict[int, float],
    str,
    Optional[float],
    bool,
    Optional[float],
    str,
]:
    if not os.path.exists(path):
        return None, {}, "", None, False, None, "missing_output"

    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return None, {}, "", None, False, None, f"read_error:{exc}"

    charges: Dict[int, float] = {}
    total_charge: Optional[float] = None
    total_energy_hartree: Optional[float] = None
    scc_converged = False
    in_charge_block = False
    saw_charge_row = False

    for line in lines:
        total_charge_match = DFTB_TOTAL_CHARGE_RE.search(line)
        if total_charge_match:
            total_charge = float(total_charge_match.group("charge"))

        total_energy_match = DFTB_TOTAL_ENERGY_RE.search(line)
        if total_energy_match:
            total_energy_hartree = float(total_energy_match.group("energy_hartree"))

        if line.strip() == "SCC converged":
            scc_converged = True

        if line.strip().startswith("Atomic gross charges"):
            in_charge_block = True
            saw_charge_row = False
            continue

        if not in_charge_block:
            continue

        row_match = DFTB_CHARGE_ROW_RE.match(line)
        if row_match:
            saw_charge_row = True
            atom_number = int(row_match.group("atom"))
            charges[atom_number] = float(row_match.group("charge"))
            continue

        if saw_charge_row and not line.strip():
            in_charge_block = False

    missing_atoms = [atom_number for atom_number in atom_numbers if atom_number not in charges]
    if not charges:
        if scc_converged:
            status = "missing_charges"
        else:
            status = "not_finished_or_no_charges"
    elif missing_atoms:
        status = "missing_atoms:" + ",".join(str(atom) for atom in missing_atoms)
    else:
        status = "ok"

    charge = None
    if not missing_atoms:
        charge = sum(charges[atom_number] for atom_number in atom_numbers)
    charge_details = ";".join(
        f"{atom_number}:{charges[atom_number]:.8f}"
        for atom_number in atom_numbers
        if atom_number in charges
    )

    return (
        charge,
        charges,
        charge_details,
        total_charge,
        scc_converged,
        total_energy_hartree,
        status,
    )


def parse_mm_prmtop(path: str, atom_numbers: Sequence[int]) -> Tuple[
    Optional[float],
    Dict[int, float],
    str,
    Optional[float],
    bool,
    Optional[float],
    str,
]:
    if not os.path.exists(path):
        return None, {}, "", None, False, None, "missing_output"

    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return None, {}, "", None, False, None, f"read_error:{exc}"

    raw_charges: List[float] = []
    in_charge_block = False
    for line in lines:
        if line.startswith("%FLAG "):
            if in_charge_block:
                break
            in_charge_block = line.split()[1] == "CHARGE"
            continue

        if not in_charge_block or line.startswith("%FORMAT"):
            continue

        for token in line.split():
            try:
                raw_charges.append(float(token))
            except ValueError:
                continue

    charges = {
        atom_number: raw_charge / AMBER_CHARGE_SCALE
        for atom_number, raw_charge in enumerate(raw_charges, start=1)
    }
    missing_atoms = [atom_number for atom_number in atom_numbers if atom_number not in charges]
    if not charges:
        status = "missing_charges"
    elif missing_atoms:
        status = "missing_atoms:" + ",".join(str(atom) for atom in missing_atoms)
    else:
        status = "ok"

    charge = None
    if not missing_atoms:
        charge = sum(charges[atom_number] for atom_number in atom_numbers)
    charge_details = ";".join(
        f"{atom_number}:{charges[atom_number]:.8f}"
        for atom_number in atom_numbers
        if atom_number in charges
    )
    total_charge = sum(charges.values()) if charges else None

    return charge, charges, charge_details, total_charge, bool(charges), None, status


def collect_rows(
    run_dirs: Sequence[str],
    orca_atom: str,
    orca_atom_index: int,
    dftb_atom_numbers: Sequence[int],
    mm_atom_numbers: Sequence[int],
    imh_atom_numbers: Sequence[int],
    dat_names: Sequence[str],
) -> List[ChargeRow]:
    rows: List[ChargeRow] = []
    dftb_target = ",".join(str(atom_number) for atom_number in dftb_atom_numbers)
    mm_target = ",".join(str(atom_number) for atom_number in mm_atom_numbers)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        for struct_dir in iter_structure_dirs(run_dir):
            dirname = os.path.basename(struct_dir)
            complex_name, scan_label, scan_value = parse_full_dir_name(dirname)
            method, output_path = detect_method(struct_dir, dat_names)
            coordinates = parse_xyz_coordinates(os.path.join(struct_dir, "min.xyz"))

            if method == "orca":
                (
                    charge,
                    all_charges,
                    total_charge,
                    converged,
                    cycles,
                    energy,
                    status,
                ) = parse_orca_dat(
                    output_path,
                    atom=orca_atom,
                    atom_index=orca_atom_index,
                )
                target = f"{orca_atom}:{orca_atom_index}"
                charge_details = ""
            elif method == "dftb":
                (
                    charge,
                    all_charges,
                    charge_details,
                    total_charge,
                    converged,
                    energy,
                    status,
                ) = parse_dftb_detailed(
                    output_path,
                    atom_numbers=dftb_atom_numbers,
                )
                cycles = None
                target = dftb_target
            elif method == "mm":
                (
                    charge,
                    all_charges,
                    charge_details,
                    total_charge,
                    converged,
                    energy,
                    status,
                ) = parse_mm_prmtop(
                    output_path,
                    atom_numbers=mm_atom_numbers,
                )
                cycles = None
                target = mm_target
            else:
                charge = None
                total_charge = None
                converged = False
                cycles = None
                energy = None
                status = "missing_output"
                target = f"{orca_atom}:{orca_atom_index}|{dftb_target}|{mm_target}"
                charge_details = ""
                all_charges = {}

            imh_dipole_debye = compute_fragment_dipole_debye(
                all_charges,
                coordinates,
                imh_atom_numbers,
            )
            imh_total_charge = compute_fragment_charge(
                all_charges,
                imh_atom_numbers,
            )

            rows.append(
                ChargeRow(
                    run=run_name,
                    method=method,
                    struct_dir=dirname,
                    complex_name=complex_name,
                    scan_label=scan_label,
                    scan_value=scan_value,
                    target=target,
                    charge=charge,
                    charge_details=charge_details,
                    total_charge=total_charge,
                    converged=converged,
                    cycles=cycles,
                    energy_hartree=energy,
                    imh_total_charge=imh_total_charge,
                    imh_dipole_debye=imh_dipole_debye,
                    status=status,
                    output_path=output_path,
                )
            )

    rows.sort(key=lambda row: (row.run, row.complex_name, row.scan_value, row.struct_dir))
    return rows


def _format_optional_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_table(rows: List[ChargeRow], handle: TextIO) -> None:
    header = (
        f"{'run':<12}  {'method':<7}  {'scan':>8}  {'struct_dir':<36}  "
        f"{'target':>8}  {'charge':>12}  {'IMH_charge':>12}  {'IMH_mu_D':>10}  "
        f"{'conv':>5}  {'cycles':>6}  {'status'}"
    )
    print(header, file=handle)
    print("-" * len(header), file=handle)
    for row in rows:
        scan = row.scan_label if row.scan_label else "0.0"
        conv = "yes" if row.converged else "no"
        cycles = "" if row.cycles is None else str(row.cycles)
        print(
            f"{row.run:<12}  {row.method:<7}  {scan:>8}  {row.struct_dir:<36}  "
            f"{row.target:>8}  {_format_optional_float(row.charge, 8):>12}  "
            f"{_format_optional_float(row.imh_total_charge, 8):>12}  "
            f"{_format_optional_float(row.imh_dipole_debye, 6):>10}  "
            f"{conv:>5}  {cycles:>6}  {row.status}",
            file=handle,
        )


def write_csv(rows: List[ChargeRow], handle: TextIO) -> None:
    fieldnames = [
        "run",
        "method",
        "struct_dir",
        "complex_name",
        "scan_label",
        "scan_value",
        "target",
        "charge",
        "charge_details",
        "total_charge",
        "converged",
        "cycles",
        "energy_hartree",
        "imh_total_charge",
        "imh_dipole_debye",
        "status",
        "output_path",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "run": row.run,
                "method": row.method,
                "struct_dir": row.struct_dir,
                "complex_name": row.complex_name,
                "scan_label": row.scan_label,
                "scan_value": row.scan_value,
                "target": row.target,
                "charge": row.charge,
                "charge_details": row.charge_details,
                "total_charge": row.total_charge,
                "converged": row.converged,
                "cycles": row.cycles,
                "energy_hartree": row.energy_hartree,
                "imh_total_charge": row.imh_total_charge,
                "imh_dipole_debye": row.imh_dipole_debye,
                "status": row.status,
                "output_path": row.output_path,
            }
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse mixed ORCA, DFTB, and MM *_full outputs in SP_init runs and "
            "extract Zn charges. ORCA runs use Mulliken charges from "
            "old.orc_job.dat/orc_job.dat; DFTB runs use Atomic gross charges "
            "from detailed.out; MM runs use charges from box.prmtop."
        )
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help=(
            "Run directories or run names. If omitted, all run* directories "
            "under --base are scanned."
        ),
    )
    parser.add_argument(
        "--base",
        default="SP_init",
        help="Base directory used when RUN is a run name (default: SP_init).",
    )
    parser.add_argument(
        "--orca-atom",
        default="Zn",
        help="Element symbol to extract from ORCA Mulliken block (default: Zn).",
    )
    parser.add_argument(
        "--orca-atom-index",
        type=int,
        default=0,
        help="Zero-based atom index to extract from ORCA output (default: 0).",
    )
    parser.add_argument(
        "--dftb-atom-number",
        action="append",
        help=(
            "1-based DFTB atom number, comma list, or range to sum. "
            "Can be given multiple times (default: 1)."
        ),
    )
    parser.add_argument(
        "--mm-atom-number",
        action="append",
        help=(
            "1-based MM atom number, comma list, or range to sum from "
            "box.prmtop charges. Can be given multiple times (default: 1)."
        ),
    )
    parser.add_argument(
        "--imh-atom-number",
        action="append",
        help=(
            "1-based atom number, comma list, or range used to calculate "
            "the IMH dipole from charges and min.xyz coordinates "
            "(default: 2-10)."
        ),
    )
    parser.add_argument(
        "--orca-dat-name",
        action="append",
        dest="orca_dat_names",
        help=(
            "ORCA output filename to try inside each *_full directory. Can be "
            "given multiple times. Default: old.orc_job.dat, then orc_job.dat."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write output to this file instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        dftb_atom_numbers = parse_atom_numbers(args.dftb_atom_number or ["1"])
        mm_atom_numbers = parse_atom_numbers(args.mm_atom_number or ["1"])
        imh_atom_numbers = parse_atom_numbers(args.imh_atom_number or ["2-10"])
        run_dirs = resolve_run_dirs(args.runs, args.base)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not run_dirs:
        print(f"ERROR: No run* directories found under {args.base}", file=sys.stderr)
        return 1

    rows = collect_rows(
        run_dirs,
        orca_atom=args.orca_atom,
        orca_atom_index=args.orca_atom_index,
        dftb_atom_numbers=dftb_atom_numbers,
        mm_atom_numbers=mm_atom_numbers,
        imh_atom_numbers=imh_atom_numbers,
        dat_names=args.orca_dat_names or ["old.orc_job.dat", "orc_job.dat"],
    )
    if not rows:
        names = ", ".join(os.path.basename(run_dir) for run_dir in run_dirs)
        print(f"ERROR: No *_full directories found under {names}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", newline="") as handle:
            if args.format == "csv":
                write_csv(rows, handle)
            else:
                write_table(rows, handle)
    elif args.format == "csv":
        write_csv(rows, sys.stdout)
    else:
        write_table(rows, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
