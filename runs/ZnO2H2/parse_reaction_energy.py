#!/usr/bin/env python3
"""Parse Zn(OH)2 -> ZnO + H2O reaction energies from SP_init runs.

run77  : DLPNO-CCSD(T)-F12,  Zn = cc-pVTZ-F12-wis (all-electron)
run78  : DLPNO-CCSD(T)-F12,  Zn = cc-pVTZ-PP-F12  (pseudopotential)
run103 : DLPNO-CCSD(T),      Zn = cc-pVTZ-F12-wis (all-electron)
run104 : DLPNO-CCSD(T),      Zn = cc-pVTZ-PP-F12  (pseudopotential)
run105 : DLPNO-CCSD(T),      Zn = cc-pVTZ-PP      (pseudopotential)
H/O use cc-pVTZ-F12 in all CCSD runs.

Prints a table to stdout and writes it to reaction_energy.csv / .md.
Reaction energy reported in kcal/mol.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631

HERE = Path(__file__).resolve().parent
ROOT = HERE / "SP_init"

RUNS = {
    "run16":  ("PBE-D3(BJ)",        "def2-TZVPPD",     "no"),
    "run77":  ("DLPNO-CCSD(T)-F12", "cc-pVTZ-F12-wis", "no"),
    "run78":  ("DLPNO-CCSD(T)-F12", "cc-pVTZ-PP-F12",  "yes"),
    "run103": ("DLPNO-CCSD(T)",     "cc-pVTZ-F12-wis", "no"),
    "run104": ("DLPNO-CCSD(T)",     "cc-pVTZ-PP-F12",  "yes"),
    "run105": ("DLPNO-CCSD(T)",     "cc-pVTZ-PP",      "yes"),
}

SPECIES = ("ZnO2H2_full", "ZnO_full", "H2O_full")
SPECIES_LABEL = {
    "ZnO2H2_full": "Zn(OH)2",
    "ZnO_full":    "ZnO",
    "H2O_full":    "H2O",
}

FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def parse_final_energy(dat_file: Path) -> float | None:
    if not dat_file.is_file():
        return None
    energy = None
    with dat_file.open("r", errors="ignore") as fh:
        for line in fh:
            m = FSPE_RE.search(line)
            if m:
                energy = float(m.group(1))
    return energy


def terminated_normally(dat_file: Path) -> bool:
    if not dat_file.is_file():
        return False
    with dat_file.open("rb") as fh:
        try:
            fh.seek(-4000, 2)
        except OSError:
            fh.seek(0)
        tail = fh.read().decode("utf-8", errors="ignore")
    return "ORCA TERMINATED NORMALLY" in tail


def collect(run: str) -> dict[str, dict]:
    out = {}
    for sp in SPECIES:
        dat = ROOT / run / sp / "orc_job.dat"
        out[sp] = {
            "path": dat,
            "energy_Eh": parse_final_energy(dat),
            "ok": terminated_normally(dat),
        }
    return out


def build_rows() -> list[dict]:
    rows = []
    for run, (method, zn_basis, ecp) in RUNS.items():
        data = collect(run)
        row = {"run": run, "method": method, "Zn_basis": zn_basis, "ECP": ecp}
        for sp in SPECIES:
            row[f"E_{SPECIES_LABEL[sp]}_Eh"] = data[sp]["energy_Eh"]
            row[f"status_{SPECIES_LABEL[sp]}"] = "OK" if data[sp]["ok"] else "FAILED"

        e_r  = data["ZnO2H2_full"]["energy_Eh"]
        e_p1 = data["ZnO_full"]["energy_Eh"]
        e_p2 = data["H2O_full"]["energy_Eh"]
        if None not in (e_r, e_p1, e_p2):
            dE_Eh   = (e_p1 + e_p2) - e_r
            row["dE_Eh"]       = dE_Eh
            row["dE_kcal_mol"] = dE_Eh * HARTREE_TO_KCAL
        else:
            row["dE_Eh"]       = None
            row["dE_kcal_mol"] = None
        rows.append(row)
    return rows


def _fmt(v, width, prec):
    if v is None:
        return f"{'N/A':>{width}}"
    return f"{v:>{width}.{prec}f}"


def print_table(rows: list[dict]) -> str:
    headers = [
        ("run",         "run",           6),
        ("method",      "method",       20),
        ("Zn_basis",    "Zn basis",     18),
        ("ECP",         "ECP",           4),
        ("dE_kcal_mol", "dE/kcal.mol-1",14),
    ]
    sep = "  "
    text_keys = ("run", "method", "Zn_basis", "ECP")
    head_line = sep.join(
        f"{h:<{w}}" if k in text_keys else f"{h:>{w}}"
        for k, h, w in headers
    )
    rule = "-" * len(head_line)

    lines = [rule, head_line, rule]
    for r in rows:
        parts = []
        for key, _h, w in headers:
            val = r.get(key)
            if key in text_keys:
                parts.append(f"{str(val):<{w}}")
            elif key == "dE_kcal_mol":
                parts.append(_fmt(val, w, 4))
            else:
                parts.append(_fmt(val, w, 10))
        lines.append(sep.join(parts))
    lines.append(rule)
    text = "\n".join(lines)
    print(text)
    return text


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["run", "method", "Zn_basis", "ECP", "dE_kcal_mol"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_md(rows: list[dict], path: Path) -> None:
    hdr   = ["run", "method", "Zn basis", "ECP", "dE / kcal.mol-1"]
    keys  = ["run", "method", "Zn_basis", "ECP", "dE_kcal_mol"]
    align = ["-",   "-",      "-",        "-",   "-:"]

    def cell(v, prec):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{prec}f}"
        return str(v)

    with path.open("w") as fh:
        fh.write("# Zn(OH)2 -> ZnO + H2O  reaction energies\n\n")
        fh.write("| " + " | ".join(hdr) + " |\n")
        fh.write("|" + "|".join(align) + "|\n")
        for r in rows:
            row_cells = []
            for k in keys:
                v = r.get(k)
                if k == "dE_kcal_mol":
                    row_cells.append(cell(v, 4))
                else:
                    row_cells.append(cell(v, 0))
            fh.write("| " + " | ".join(row_cells) + " |\n")


def main() -> None:
    title = "Zn(OH)2  ->  ZnO + H2O   reaction energies"
    print(title)
    print("=" * len(title))

    rows = build_rows()
    print_table(rows)

    csv_path = HERE / "reaction_energy.csv"
    md_path  = HERE / "reaction_energy.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)

    print(f"\nSaved: {csv_path.relative_to(HERE)}")
    print(f"Saved: {md_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
