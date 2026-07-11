#!/usr/bin/env python3
"""
Scan SP_init/<run>/<complex>/ and summarize dipoles to CSV:

1. DFTB+ detailed.out — gross charges qᵢ × min.xyz; μ⃗_corr = Σ qᵢr⃗ᵢ − Q·R⃗_com with **Q from run0**
   prmtop (not Σ q_DFTB); R⃗_com from atomic masses derived from xyz symbols.
2. Amber MM — same formula; qᵢ from local ``box.prmtop``, masses from ``%FLAG MASS``.

**Total charge Q (electrons), all four runs:** sum of Amber ``CHARGE`` entries in
``SP_init/run0/<complex>/box.prmtop``, each multiplied by ``1/18.2223``. This is the
single source of truth; COM shift always uses this Q (never Σ Mulliken/DFTB gross
charges for Q). The CSV column **boxprmtop-charge** carries this value (five decimal places).

Which directory / method (CSV headers):
  run0(mm) → MM point charges + COM.
  run8(mdftb/3ob'), run110(mdftb/run59opt) → ``dipole/detailed.out`` full dipole (Debye) + COM.
  run102(RHF) → Molpro ``Dipole moment /Debye`` + COM from ``qm_minimize/…/min.xyz``.
  run41(M052X) → ORCA ``Total Dipole Moment`` (a.u.) from ``SP_init/run41/<complex>/``
    ``orc_job.dat`` or ``old.orc_job.dat``. ORCA uses **electric origin = center of mass**
    (see ``Choice of electric origin`` / ``Position of electric origin`` in ``orc_job.dat``), so
    we report |μ| in Debye **without** an extra COM translation (unlike run0/run8/run110/run102).
  run8(Mk Charge), run110(Mk Charge) → DFTB gross qᵢ × min.xyz + COM.

All methods other than run41 use mass-weighted COM in the correction described above.
Missing inputs → stderr + exit 1 (run41 only needs ORCA ``orc_job.dat`` / ``old.orc_job.dat``).

CSV: one row per complex; each ``run8(Mk Charge)`` / ``run110(Mk Charge)`` follows its
  paired mdftb column; ``run102(RHF)`` then ``run41(M052X)`` (then notes).
  If ``SP_init/run41/<complex>/`` lacks ORCA output or no ``Total Dipole Moment`` line parses,
  ``run41(M052X)`` is left blank and stderr lists skipped complexes.

Skipped complex directory names: ``ghost`` (case-insensitive); fragments
``*_monomer_*`` where something follows ``_monomer_`` (e.g. ``*_monomer_0``).
Standalone names ending in ``_monomer`` (e.g. ``MeOH_monomer``) are kept.

Default root is the relax_struc2 directory (parent of this script).

Usage:
  ./collect_dipoles_csv.py
  ./collect_dipoles_csv.py -o dipoles.csv --sp-init /path/to/SP_init \\
      --qm-minimize-dir ./qm_minimize/run41
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

DEBYE_PER_EANG = 4.803204514  # approximate e·Å → Debye
# ORCA prints Cartesian dipole components in atomic units (e·a₀); magnitude line says "(a.u.)".
DEBYE_PER_DIPOLE_AU = 2.541746451893896
AMBER_CHARGE_TO_ELECTRON = 1.0 / 18.2223

# |Σ q_MM − Q_run0| must stay below this when MM charges come from the same prmtop as Q.
MM_CHARGE_SUM_RTOL = 1e-5

# Element symbol (letters only, upper) → atomic number; xyz / Amber names like ZN1 → ZN
SYMBOL_TO_Z: dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "ZN": 30,
    "CU": 29,
    "FE": 26,
    "BR": 35,
    "I": 53,
}

# Average atomic weights (amu) for COM when only xyz symbols are available (DFTB path).
SYMBOL_TO_MASS_AMU: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "NA": 22.990,
    "MG": 24.305,
    "AL": 26.982,
    "SI": 28.085,
    "P": 30.974,
    "S": 32.06,
    "CL": 35.45,
    "K": 39.098,
    "CA": 40.078,
    "ZN": 65.38,
    "CU": 63.546,
    "FE": 55.845,
    "BR": 79.904,
    "I": 126.90,
}


def _elem_key_from_xyz(symbol: str) -> str:
    sym = symbol.strip()
    letters = re.match(r"^([A-Za-z]+)", sym)
    return letters.group(1).upper() if letters else sym.upper()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


_MOLPRO_DIPOLE_DEBYE_LINE = re.compile(
    r"^\s*Dipole moment /Debye\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$",
    re.MULTILINE,
)


def parse_molpro_dipole_debye_vector(text: str) -> tuple[float, float, float] | None:
    """Last ``Dipole moment /Debye`` row (components in Debye)."""
    matches = list(_MOLPRO_DIPOLE_DEBYE_LINE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


_ORCA_TOTAL_DIPOLE_AU = re.compile(
    r"^\s*Total Dipole Moment\s+:\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$",
    re.MULTILINE,
)


def parse_orca_total_dipole_au_vector(text: str) -> tuple[float, float, float] | None:
    """Last ORCA ``Total Dipole Moment`` row (components in atomic units)."""
    matches = list(_ORCA_TOTAL_DIPOLE_AU.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def resolve_orca_dat_path(stem_dir: Path) -> Path | None:
    """Prefer ``orc_job.dat``, then ``old.orc_job.dat`` (common rename after reruns)."""
    for name in ("orc_job.dat", "old.orc_job.dat"):
        p = stem_dir / name
        if p.is_file():
            return p
    return None


def resolve_orca_text_with_dipole(stem_dir: Path) -> tuple[Path, str] | None:
    """
    First ``orc_job.dat`` / ``old.orc_job.dat`` whose text contains a parseable
    ``Total Dipole Moment`` line (truncated ``orc_job.dat`` may be skipped).
    """
    for name in ("orc_job.dat", "old.orc_job.dat"):
        p = stem_dir / name
        if not p.is_file():
            continue
        t = _read_text(p)
        if not t:
            continue
        if parse_orca_total_dipole_au_vector(t):
            return (p, t)
    return None


def orca_dipole_magnitude_debye_from_au(
    mu_au: tuple[float, float, float],
) -> float:
    """
    |μ| in Debye from ORCA ``Total Dipole Moment`` components (atomic units).

    ORCA evaluates the dipole with **electric origin at center of mass** by default
    (printed just above ``DIPOLE MOMENT``); no extra COM correction is applied here.
    """
    mx = mu_au[0] * DEBYE_PER_DIPOLE_AU
    my = mu_au[1] * DEBYE_PER_DIPOLE_AU
    mz = mu_au[2] * DEBYE_PER_DIPOLE_AU
    return math.sqrt(mx * mx + my * my + mz * mz)


def molpro_dipole_COM_corrected_debye(
    mu_debye: tuple[float, float, float],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    q_total_e: float,
) -> float | None:
    """
    μ⃗ (Debye) from QC log → e·Å components, then μ_corr = μ_eÅ − Q·R_com, return |μ_corr| in Debye.
    """
    if len(coords) != len(masses):
        return None
    mx = mu_debye[0] / DEBYE_PER_EANG
    my = mu_debye[1] / DEBYE_PER_EANG
    mz = mu_debye[2] / DEBYE_PER_EANG
    com = _center_of_mass(coords, masses)
    if com is None:
        return None
    cx, cy, cz = com
    dx = mx - q_total_e * cx
    dy = my - q_total_e * cy
    dz = mz - q_total_e * cz
    return DEBYE_PER_EANG * math.sqrt(dx * dx + dy * dy + dz * dz)


def qm_minimize_xyz_path(qm_run_dir: Path, complex_name: str) -> Path:
    """
    ``qm_minimize/run41`` dirs omit the ``_full`` suffix used under SP_init.
    """
    key = complex_name[:-5] if complex_name.endswith("_full") else complex_name
    return qm_run_dir / key / "min.xyz"


def parse_detailed_out_gross_charges(text: str) -> list[float] | None:
    """
    DFTB+ xtb-style block:
      Atomic gross charges (e)
     Atom           Charge
        1       0.67269015
    """
    if "Atomic gross charges" not in text:
        return None
    idx = text.find("Atomic gross charges")
    sub = text[idx:]
    m = re.search(r"Atom\s+Charge", sub)
    if not m:
        return None
    body = sub[m.end() : m.end() + 200000]
    charges: list[float] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            if charges:
                break
            continue
        mm = re.match(r"^(\d+)\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$", line)
        if not mm:
            if charges:
                break
            continue
        charges.append(float(mm.group(2)))
    return charges or None


def parse_dftb_dipole_subdir_debye_vector(text: str) -> tuple[float, float, float] | None:
    """
    Last ``Dipole moment:  μx μy μz  Debye`` line in ``dipole/detailed.out``
    (includes atomic dipole / full DFTB dipole output).
    """
    pat = re.compile(
        r"^\s*Dipole moment:\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"Debye\s*$",
        re.MULTILINE,
    )
    matches = list(pat.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_min_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]] | None:
    raw = _read_text(path)
    if not raw:
        return None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    body = []
    elems: list[str] = []
    for ln in lines[2 : 2 + n]:
        tok = ln.split()
        if len(tok) < 4:
            return None
        elems.append(tok[0])
        x, y, z = float(tok[1]), float(tok[2]), float(tok[3])
        body.append((x, y, z))
    if len(body) != n:
        return None
    return elems, body


def _dipole_vector_e_angstrom(q: list[float], r: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    if len(q) != len(r) or not q:
        return None
    sx = sy = sz = 0.0
    for qi, (x, y, z) in zip(q, r):
        sx += qi * x
        sy += qi * y
        sz += qi * z
    return (sx, sy, sz)


def _center_of_mass(
    r: list[tuple[float, float, float]], masses: list[float]
) -> tuple[float, float, float] | None:
    if len(r) != len(masses) or not masses or sum(masses) <= 0:
        return None
    tw = 0.0
    sx = sy = sz = 0.0
    for mi, (x, y, z) in zip(masses, r):
        sx += mi * x
        sy += mi * y
        sz += mi * z
        tw += mi
    inv = 1.0 / tw
    return (sx * inv, sy * inv, sz * inv)


def dipole_mag_debye_COM_corrected(
    q: list[float],
    r: list[tuple[float, float, float]],
    masses: list[float],
    q_total_e: float,
) -> float | None:
    """
    μ⃗_corr = Σ qᵢr⃗ᵢ − Q·R⃗_com (e·Å); |μ_corr| in Debye.

    **Q** is the molecular total charge in electrons — always taken from
    ``SP_init/run0/<complex>/box.prmtop`` (sum of CHARGE/18.2223), not from Σ qᵢ
    for DFTB partial charges (SCC gross charges may not sum exactly to Q).
    """
    if len(q) != len(r) or len(masses) != len(r):
        return None
    mu = _dipole_vector_e_angstrom(q, r)
    if mu is None:
        return None
    com = _center_of_mass(r, masses)
    if com is None:
        return None
    cx, cy, cz = com
    dx = mu[0] - q_total_e * cx
    dy = mu[1] - q_total_e * cy
    dz = mu[2] - q_total_e * cz
    return DEBYE_PER_EANG * math.sqrt(dx * dx + dy * dy + dz * dz)


def masses_from_xyz_elements(elems: list[str]) -> tuple[list[float] | None, list[str]]:
    """Return (mass list, unknown element symbols if any)."""
    unknown: list[str] = []
    for e in elems:
        k = _elem_key_from_xyz(e)
        if k not in SYMBOL_TO_MASS_AMU:
            if k not in unknown:
                unknown.append(k)
    if unknown:
        return None, unknown
    return [
        SYMBOL_TO_MASS_AMU[_elem_key_from_xyz(e)] for e in elems
    ], []


def parse_prmtop_flag_numeric_block(text: str, flag: str) -> list[float] | None:
    m_flag = re.search(rf"%FLAG {re.escape(flag)}\s*\n%\S+\s*\n", text)
    if not m_flag:
        return None
    start = m_flag.end()
    m_next = re.search(r"\n%FLAG\s+", text[start:])
    end = start + (m_next.start() if m_next else len(text) - start)
    sec = text[start:end]
    out: list[float] = []
    for token in sec.split():
        token = token.strip()
        if token:
            out.append(float(token))
    return out or None


def parse_prmtop_mm_arrays(
    path: Path,
) -> tuple[list[float], list[float], list[int]] | None:
    """
    Same atom index for CHARGE, MASS, ATOMIC_NUMBER in Amber topology.
    Charges returned in electron units.
    """
    text = _read_text(path)
    if not text:
        return None
    raw_q = parse_prmtop_flag_numeric_block(text, "CHARGE")
    raw_m = parse_prmtop_flag_numeric_block(text, "MASS")
    raw_z = parse_prmtop_flag_numeric_block(text, "ATOMIC_NUMBER")
    if not raw_q or not raw_m or not raw_z:
        return None
    if not (len(raw_q) == len(raw_m) == len(raw_z)):
        return None
    q_e = [c * AMBER_CHARGE_TO_ELECTRON for c in raw_q]
    at_nums = [int(round(z)) for z in raw_z]
    return (q_e, raw_m, at_nums)


def mm_xyz_matches_prmtop(elems: list[str], at_nums: list[int]) -> list[str]:
    """Report mismatches between min.xyz element order and prmtop ATOMIC_NUMBER."""
    issues: list[str] = []
    n = min(len(elems), len(at_nums))
    for i in range(n):
        k = _elem_key_from_xyz(elems[i])
        ez = SYMBOL_TO_Z.get(k)
        if ez is None:
            issues.append(f"atom {i + 1}: unknown xyz symbol {k!r}")
            continue
        if ez != at_nums[i]:
            issues.append(
                f"atom {i + 1}: xyz {k} (Z={ez}) vs prmtop Z={at_nums[i]}"
            )
    if len(elems) != len(at_nums):
        issues.append(f"len xyz {len(elems)} vs prmtop {len(at_nums)}")
    return issues


def _skip_complex_name(name: str) -> bool:
    """Exclude ghost BSSE jobs and decomposed fragment dirs (*_monomer_<id>)."""
    if "ghost" in name.lower():
        return True
    # e.g. ..._monomer_0, ..._monomer_12 — not ..._monomer alone
    if re.search(r"_monomer_.+$", name):
        return True
    return False


def iter_complex_dirs(sp_init: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    if not sp_init.is_dir():
        return pairs
    for run_dir in sorted(sp_init.iterdir(), key=lambda p: (p.name,)):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        name = run_dir.name
        if not (name.startswith("run") or re.match(r"^run\d+$", name)):
            continue
        run_id = name
        for cpx in sorted(run_dir.iterdir(), key=lambda p: p.name):
            if not cpx.is_dir() or cpx.name.startswith(".") or cpx.name == "waste":
                continue
            if _skip_complex_name(cpx.name):
                continue
            pairs.append((run_id, cpx))
    return pairs


# Only these runs appear in the CSV; each uses a single dipole protocol.
RUN_DIPOLE_METHOD: dict[str, str] = {
    "run0": "mm",
    "run8": "dftb",
    "run110": "dftb",
    "run102": "molpro",
    "run41": "orca",
}

# CSV column titles (internal dirs stay run0, run8, …).
RUN_CSV_COLUMN: dict[str, str] = {
    "run0": "run0(mm)",
    "run8": "run8(Mk Charge)",
    "run8_atomdip": "run8(mdftb/3ob')",
    "run110": "run110(Mk Charge)",
    "run110_atomdip": "run110(mdftb/run59opt)",
    "run102": "run102(RHF)",
    "run41": "run41(M052X)",
}

# Each Mk Charge column immediately follows its detailed.out dipole column.
CSV_COLUMN_ORDER = [
    "run0",
    "run8_atomdip",
    "run8",
    "run110_atomdip",
    "run110",
    "run102",
    "run41",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Collect dipoles under SP_init to CSV.")
    ap.add_argument(
        "--sp-init",
        type=Path,
        default=here / "SP_init",
        help="Path to SP_init (default: ./SP_init next to this script)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=here / "dipoles_SP_init.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--qm-minimize-dir",
        type=Path,
        default=here / "qm_minimize" / "run41",
        help=(
            "Directory with <stem>/min.xyz for run102 COM only (run41 uses SP_init/run41); "
            "default: ./qm_minimize/run41"
        ),
    )
    args = ap.parse_args()
    sp_init = args.sp_init.resolve()
    qm_min_dir = args.qm_minimize_dir.resolve()

    pairs_raw = [
        (rid, cdir)
        for rid, cdir in iter_complex_dirs(sp_init)
        if rid in RUN_DIPOLE_METHOD
    ]
    # Charge Q comes only from run0 prmtop; ignore complexes that appear only under run41.
    complex_names = sorted(
        {cdir.name for rid, cdir in pairs_raw if rid != "run41"}
    )

    charge_e: dict[str, float] = {}
    errors: list[str] = []
    for cn in complex_names:
        p0 = sp_init / "run0" / cn / "box.prmtop"
        parsed0 = parse_prmtop_mm_arrays(p0)
        if not parsed0:
            errors.append(
                f"{cn}: need SP_init/run0/{cn}/box.prmtop to define total charge "
                "Q = sum(CHARGE)/18.2223 (electrons)"
            )
            continue
        q_e0, _, _ = parsed0
        charge_e[cn] = sum(q_e0)

    if errors:
        print("collect_dipoles_csv.py: charge preflight failed:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    pairs = [(rid, cdir) for rid, cdir in pairs_raw if cdir.name in charge_e]

    # complex -> run_id -> dipole magnitude (Debye) as string
    by_complex: dict[str, dict[str, str]] = defaultdict(dict)
    run41_missing_orca: list[str] = []
    run41_no_dipole_line: list[str] = []

    for run_id, cdir in pairs:
        cn = cdir.name

        def tag(run: str, msg: str) -> str:
            return f"{cn} {run}: {msg}"

        Q_tot = charge_e[cn]
        method = RUN_DIPOLE_METHOD[run_id]
        dip_str = ""
        err_here: list[str] = []

        xyz_p = cdir / "min.xyz"
        xyz_mm = parse_min_xyz(xyz_p) if xyz_p.is_file() else None

        if method == "molpro":
            log_p = cdir / "molpro.log"
            qm_xyz_p = qm_minimize_xyz_path(qm_min_dir, cn)
            qm_xyz = parse_min_xyz(qm_xyz_p) if qm_xyz_p.is_file() else None

            if not log_p.is_file():
                err_here.append(tag("run102", f"missing {log_p}"))
            elif not _read_text(log_p):
                err_here.append(tag("run102", f"empty {log_p}"))
            else:
                t = _read_text(log_p)
                mu_vec = parse_molpro_dipole_debye_vector(t or "")
                if mu_vec is None:
                    err_here.append(
                        tag(
                            "run102",
                            "no parseable 'Dipole moment /Debye' line in molpro.log",
                        )
                    )
                elif not qm_xyz_p.is_file():
                    err_here.append(tag("run102", f"missing qm_minimize geometry {qm_xyz_p}"))
                elif qm_xyz is None:
                    err_here.append(tag("run102", f"invalid qm_minimize min.xyz {qm_xyz_p}"))
                else:
                    elems_qm, coords_qm = qm_xyz
                    mlist, unk = masses_from_xyz_elements(elems_qm)
                    if unk:
                        err_here.append(
                            tag(
                                "run102",
                                f"unknown element {unk[0]} in qm_minimize min.xyz (need mass)",
                            )
                        )
                    elif len(mlist) != len(coords_qm):
                        err_here.append(tag("run102", "mass list length mismatch"))
                    else:
                        mu_c = molpro_dipole_COM_corrected_debye(
                            mu_vec, coords_qm, mlist, Q_tot
                        )
                        if mu_c is None:
                            err_here.append(tag("run102", "Molpro COM dipole computation failed"))
                        else:
                            dip_str = f"{mu_c:.1f}"

        elif method == "orca":
            resolved = resolve_orca_text_with_dipole(cdir)
            if resolved is None:
                if resolve_orca_dat_path(cdir) is None:
                    run41_missing_orca.append(cn)
                else:
                    run41_no_dipole_line.append(cn)
                dip_str = ""
            else:
                _orca_p, t_orca = resolved
                mu_au = parse_orca_total_dipole_au_vector(t_orca)
                assert mu_au is not None
                dip_str = f"{orca_dipole_magnitude_debye_from_au(mu_au):.1f}"

        elif method == "dftb":
            detail_p = cdir / "detailed.out"
            dip_sub_p = cdir / "dipole" / "detailed.out"
            detail_text = _read_text(detail_p) if detail_p.is_file() else None
            q_detail = (
                parse_detailed_out_gross_charges(detail_text) if detail_text else None
            )
            dip_atom_str = ""

            if not detail_p.is_file():
                err_here.append(tag(run_id, f"missing {detail_p}"))
            elif q_detail is None:
                err_here.append(
                    tag(run_id, "no Atomic gross charges block in detailed.out")
                )
            elif xyz_mm is None:
                err_here.append(tag(run_id, f"missing or invalid {cdir / 'min.xyz'}"))
            else:
                elems, coords = xyz_mm
                if len(q_detail) != len(coords):
                    err_here.append(
                        tag(
                            run_id,
                            f"gross charges N={len(q_detail)} vs min.xyz N={len(coords)}",
                        )
                    )
                else:
                    mlist, unk = masses_from_xyz_elements(elems)
                    if unk:
                        err_here.append(
                            tag(
                                run_id,
                                f"unknown element {unk[0]} in min.xyz for COM masses",
                            )
                        )
                    elif len(mlist) != len(coords):
                        err_here.append(tag(run_id, "mass list mismatch"))
                    else:
                        mu_d = dipole_mag_debye_COM_corrected(
                            q_detail, coords, mlist, Q_tot
                        )
                        if mu_d is None:
                            err_here.append(tag(run_id, "COM dipole computation failed"))
                        else:
                            dip_str = f"{mu_d:.1f}"

            # dipole/detailed.out — printed dipole (includes atomic dipole); same min.xyz + Q
            if not dip_sub_p.is_file():
                err_here.append(tag(run_id, f"missing {dip_sub_p}"))
            elif xyz_mm is None:
                err_here.append(
                    tag(run_id, "dipole/ COM needs same-directory min.xyz")
                )
            else:
                t_sub = _read_text(dip_sub_p)
                if not t_sub:
                    err_here.append(tag(run_id, f"empty {dip_sub_p}"))
                else:
                    mu_atom = parse_dftb_dipole_subdir_debye_vector(t_sub)
                    if mu_atom is None:
                        err_here.append(
                            tag(
                                run_id,
                                "dipole/detailed.out: no parseable "
                                "'Dipole moment … Debye' line",
                            )
                        )
                    else:
                        elems_a, coords_a = xyz_mm
                        mlist_a, unk_a = masses_from_xyz_elements(elems_a)
                        if unk_a:
                            err_here.append(
                                tag(
                                    run_id,
                                    f"dipole/ COM: unknown element {unk_a[0]} in min.xyz",
                                )
                            )
                        elif len(mlist_a) != len(coords_a):
                            err_here.append(
                                tag(run_id, "dipole/ COM: mass list mismatch")
                            )
                        else:
                            mu_ad = molpro_dipole_COM_corrected_debye(
                                mu_atom, coords_a, mlist_a, Q_tot
                            )
                            if mu_ad is None:
                                err_here.append(
                                    tag(run_id, "dipole/ COM dipole computation failed")
                                )
                            else:
                                dip_atom_str = f"{mu_ad:.1f}"

        elif method == "mm":
            prmtop_used = cdir / "box.prmtop"
            if not prmtop_used.is_file():
                err_here.append(tag("run0(mm)", f"missing {prmtop_used}"))
            elif xyz_mm is None:
                err_here.append(tag("run0(mm)", f"missing or invalid {cdir / 'min.xyz'}"))
            else:
                parsed = parse_prmtop_mm_arrays(prmtop_used)
                elems, coords = xyz_mm
                if parsed is None:
                    err_here.append(
                        tag("run0(mm)", "box.prmtop CHARGE/MASS/ATOMIC_NUMBER unreadable")
                    )
                else:
                    q_amber, masses_amu, at_nums = parsed
                    mm_issues = mm_xyz_matches_prmtop(elems, at_nums)
                    if mm_issues:
                        err_here.append(
                            tag(
                                "run0(mm)",
                                "xyz vs prmtop ATOMIC_NUMBER: " + "; ".join(mm_issues[:3]),
                            )
                        )
                    elif len(q_amber) != len(coords) or len(masses_amu) != len(coords):
                        err_here.append(
                            tag(
                                "run0(mm)",
                                f"prmtop natom {len(q_amber)} vs min.xyz {len(coords)}",
                            )
                        )
                    else:
                        q_sum = sum(q_amber)
                        if abs(q_sum - Q_tot) > MM_CHARGE_SUM_RTOL * max(abs(Q_tot), 1.0):
                            err_here.append(
                                tag(
                                    "run0(mm)",
                                    f"Σ partial charges {q_sum} ≠ Q(run0 ref) {Q_tot}",
                                )
                            )
                        else:
                            mu_m = dipole_mag_debye_COM_corrected(
                                q_amber, coords, masses_amu, Q_tot
                            )
                            if mu_m is None:
                                err_here.append(
                                    tag("run0(mm)", "COM dipole computation failed")
                                )
                            else:
                                dip_str = f"{mu_m:.1f}"

        errors.extend(err_here)
        if not err_here:
            by_complex[cn][run_id] = dip_str
            if method == "dftb":
                by_complex[cn][f"{run_id}_atomdip"] = dip_atom_str

    if run41_missing_orca:
        print(
            "collect_dipoles_csv.py: run41(M052X) skipped (no orc_job.dat / old.orc_job.dat): "
            + ", ".join(run41_missing_orca),
            file=sys.stderr,
        )
    if run41_no_dipole_line:
        print(
            "collect_dipoles_csv.py: run41(M052X) skipped (no parseable Total Dipole Moment): "
            + ", ".join(run41_no_dipole_line),
            file=sys.stderr,
        )

    if errors:
        print("collect_dipoles_csv.py failed:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    run_headers = [RUN_CSV_COLUMN[r] for r in CSV_COLUMN_ORDER]
    fieldnames = ["complex", "boxprmtop-charge", *run_headers, "notes"]

    rows_wide: list[dict[str, str]] = []
    for complex_name in sorted(by_complex.keys()):
        qv = charge_e[complex_name]
        row: dict[str, str] = {
            "complex": complex_name,
            "boxprmtop-charge": f"{qv:.5f}",
            "notes": "",
        }
        for run in CSV_COLUMN_ORDER:
            row[RUN_CSV_COLUMN[run]] = by_complex[complex_name].get(run, "")
        rows_wide.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_wide)

    print(
        f"Wrote {len(rows_wide)} complex rows (boxprmtop-charge + {len(CSV_COLUMN_ORDER)} "
        f"dipole columns) to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
