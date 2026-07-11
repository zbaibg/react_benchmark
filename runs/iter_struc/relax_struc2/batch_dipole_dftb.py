#!/usr/bin/env python3
"""
Under SP_init/run8 and SP_init/run110: for each complex with dftb_pin.hsd,
create dipole/, copy dftb_pin.hsd -> dipole/dftb_in.hsd, strip Hamiltonian
lines for elements absent from Geometry/TypeNames, then run dftb+ in dipole/.

DFTB+ with IgnoreUnprocessedNodes=No aborts if unknown species appear not only
in OneCenterAtomIntegrals but also in AtomDIntegralScalings, AtomQIntegralScalings, SlaterKosterFiles, HubbardDerivs, and MaxAngularMomentum; those blocks are filtered the same way (species taken from TypeNames).

Usage:
  ./batch_dipole_dftb.py
  ./batch_dipole_dftb.py --dry-run
  ./batch_dipole_dftb.py --runs run8 --continue-on-error
  DFTBPLUS_COMMAND=/path/to/dftb+ ./batch_dipole_dftb.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Symbols used to classify LHS in atom blocks (exclude non-elements like Prefix).
PERIODIC_SYMBOLS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
    Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt
    Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
    Rg Cn Fl Lv Ts Og
    """.split()
)


def parse_type_names(hsd: str) -> set[str]:
    m = re.search(r"TypeNames\s*=\s*\{([^}]*)\}", hsd, re.DOTALL)
    if not m:
        return set()
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def filter_hsd_for_elements(text: str, allowed: set[str]) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    block: str | None = None

    def keep_atom_assign(line: str) -> bool:
        m = re.match(r"^(\s*)([A-Za-z][A-Za-z0-9]*)\s*=", line)
        if not m:
            return True
        sym = m.group(2)
        if sym not in PERIODIC_SYMBOLS:
            return True
        return sym in allowed

    def keep_ocai(line: str) -> bool:
        m = re.match(r"^(\s*)([^:]+):", line)
        if not m:
            return True
        sym = m.group(2)
        if sym not in PERIODIC_SYMBOLS:
            return True
        return sym in allowed

    def keep_skf(line: str) -> bool:
        m = re.match(r"^\s*([A-Z][a-z]*)-([A-Z][a-z]*)\s*=", line)
        if not m:
            return True
        a, b = m.group(1), m.group(2)
        if a not in PERIODIC_SYMBOLS or b not in PERIODIC_SYMBOLS:
            return True
        return a in allowed and b in allowed

    for line in lines:
        s = line.strip()
        if block is None:
            if re.search(r"OneCenterAtomIntegrals\s*=\s*\{", line):
                block = "ocai"
            elif re.search(r"AtomDIntegralScalings\s*=\s*\{", line):
                block = "dsc"
            elif re.search(r"AtomQIntegralScalings\s*=\s*\{", line):
                block = "qsc"
            elif re.search(r"HubbardDerivs\s*=\s*\{", line):
                block = "hub"
            elif re.search(r"MaxAngularMomentum\s*=\s*\{", line):
                block = "max"
            elif re.search(r"SlaterKosterFiles\s*=\s*\{", line):
                block = "skf"
            out.append(line)
            continue

        if s == "}" or (s.startswith("}") and block != "ocai"):
            # End of any of our blocks: closing brace only
            if re.match(r"^\s*\}\s*$", line):
                out.append(line)
                block = None
            else:
                out.append(line)
            continue

        if block == "ocai":
            if keep_ocai(line):
                out.append(line)
        elif block in ("dsc", "qsc", "hub", "max"):
            if keep_atom_assign(line):
                out.append(line)
        elif block == "skf":
            if keep_skf(line):
                out.append(line)
        else:
            out.append(line)

    return "".join(out)


def default_dftb_command() -> str:
    for cand in (
        os.environ.get("DFTBPLUS_COMMAND"),
        shutil.which("dftb+"),
        shutil.which("dftbplus"),
    ):
        if cand:
            return cand
    return "dftb+"


def should_skip_dir(p: Path) -> bool:
    n = p.name.lower()
    if n == "waste":
        return True
    if "ghost" in n:
        return True
    return False


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Prepare dipole/dftb_in.hsd and run DFTB+ for run8 & run110.")
    ap.add_argument(
        "--sp-init",
        type=Path,
        default=here / "SP_init",
        help="SP_init directory (default: ./SP_init beside this script)",
    )
    ap.add_argument(
        "--runs",
        default="run8,run110",
        help="Comma-separated run dirs under SP_init (default: run8,run110)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Prepare files only, do not run dftb+")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed dftb+; exit non-zero if any failed",
    )
    args = ap.parse_args()
    sp_init: Path = args.sp_init.resolve()
    dftb_cmd = default_dftb_command()
    run_names = [x.strip() for x in args.runs.split(",") if x.strip()]

    if not sp_init.is_dir():
        print(f"ERROR: SP_init not found: {sp_init}", file=sys.stderr)
        return 1

    failed: list[str] = []

    for run_id in run_names:
        run_dir = sp_init / run_id
        if not run_dir.is_dir():
            print(f"WARN: skip missing run dir: {run_dir}", file=sys.stderr)
            continue

        for cpx in sorted(run_dir.iterdir(), key=lambda q: q.name):
            if not cpx.is_dir() or cpx.name.startswith("."):
                continue
            if should_skip_dir(cpx):
                continue

            pin = cpx / "dftb_pin.hsd"
            if not pin.is_file():
                continue

            dipole_dir = cpx / "dipole"
            out_hsd = dipole_dir / "dftb_in.hsd"

            raw = pin.read_text(encoding="utf-8", errors="replace")
            allowed = parse_type_names(raw)
            if not allowed:
                print(f"WARN: no TypeNames in {pin}, skip", file=sys.stderr)
                continue

            filtered = filter_hsd_for_elements(raw, allowed)

            if args.dry_run:
                print(f"[dry-run] would write {out_hsd} and run dftb+ ({dftb_cmd})")
                continue

            dipole_dir.mkdir(parents=True, exist_ok=True)
            out_hsd.write_text(filtered, encoding="utf-8")

            print(f"==> {cpx.relative_to(sp_init)}  elements={sorted(allowed)}", flush=True)
            try:
                r = subprocess.run(
                    [dftb_cmd],
                    cwd=str(dipole_dir),
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except OSError as e:
                print(f"ERROR: could not execute {dftb_cmd}: {e}", file=sys.stderr)
                failed.append(str(cpx))
                if not args.continue_on_error:
                    return 1
                continue

            if r.returncode != 0:
                tail = (r.stdout or "")[-2000:]
                print(tail, file=sys.stderr)
                print(f"ERROR: dftb+ exit {r.returncode} in {dipole_dir}", file=sys.stderr)
                failed.append(str(cpx))
                if not args.continue_on_error:
                    return 1
            else:
                print(f"    OK ({dftb_cmd})")

    if failed:
        print(f"Failed ({len(failed)}):", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
