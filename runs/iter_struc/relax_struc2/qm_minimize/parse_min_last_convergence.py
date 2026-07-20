#!/usr/bin/env python3
"""Extract convergence info from each */min.out and write min_last_convergence.tsv.

Output columns:
  cycle, energy_change, max_step, rms_step, max_grad, rms_grad — from the *last*
  DL-FIND convergence cycle (full Energy…RMS grad block if present; otherwise the last
  partial block, e.g. one-step jobs that only print Max grad / RMS grad). Missing lines
  are written as N/A.
  cycle_to_reach_loose_criteria — the *first* cycle where Energy,
  Max step, RMS step, and Max grad are simultaneously <= targets (run35 defaults:
  5e-6, 1.8e-3, 1.2e-3, 4.5e-4). Cycles missing any of those four lines are skipped.
  Etot_last_cycle — total system energy (kcal/mol): AMBER ``Etot`` before the last DL-FIND
  cycle when present; otherwise the last ``EXTERNESCF`` line (e.g. single-atom minimization
  without DL-FIND). Folder ``H`` is forced to 0 (reference proton). N/A if unknown.
  Etot_loose_criteria_cycle — same Etot (kcal/mol) for the first loose-criteria cycle; N/A
  if that cycle is missing or loose criteria never met. For folder ``H``, both Etot columns
  are set to 0.
  converged_loose_criteria — yes if cycle_to_reach_loose_criteria is not N/A, else no.
  converged_signal — yes if the file contains the substring Convergence reached, else no
  (N/A if the file could not be read).

When converged_loose_criteria is yes but converged_signal is no, the script copies the frame in
path.xyz that corresponds to that cycle (1-based cycle → frame index cycle-1) to
min_loose_criteria_cyc<cycle>.xyz (e.g. min_loose_criteria_cyc65.xyz) next to min.out.
Use --no-copy-loose-xyz to skip copying. Set WRITE_MIN_LOOSE_CRITERIA_XYZ to False to
disable writing those files regardless of CLI.

Rows are sorted by ``max_grad`` from the last block (numeric, ascending).

Requires pandas (``pip install pandas``). Results are written with ``DataFrame.to_csv`` (tab-separated)
only; nothing is printed to the terminal. TSV column order is ``OUTPUT_TSV_COLUMNS`` at the bottom of this file.
In the TSV, ``energy_change`` … ``rms_grad`` use scientific notation with one decimal (``TSV_SCI_ONE_DECIMAL_COLUMNS``).

From each system's ``path.xyz``, the **last frame** is analyzed (same logic as ``check_qm_minimize.py``):
``max_Zn-Nmin_MIM(Å)`` and ``max_Zn-O_MeOH(Å)`` (Å, three decimals), plus ``Flag`` (``MIM_far`` / ``MeOH_far`` if
above ``ZN_N_SUSPECT`` / ``ZN_O_SUSPECT``, 2.5 Å by default). If ``min.out`` contains DL-FIND abort text
(``DL-Find encountered an error``, ``DL-FIND ERROR:``), ``dlfind error`` is appended
to ``Flag`` (comma-separated).
Requires ``python_scripts/zif_meoh_assign_name`` (MDAnalysis).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

# DL-FIND prints this when optimisation finishes successfully.
CONVERGENCE_REACHED_MARKER = "Convergence reached"

# Simultaneous four-criterion targets (run35 min.out, e.g. 1Zn_0MIm_0MImH_6MeOH).
FOUR_CRIT_TARGETS: dict[str, float] = {
    "energy_change": 5.0e-6,
    "max_step": 1.8e-3,
    "rms_step": 1.2e-3,
    "max_grad": 4.5e-4,
}

# When False, never write min_loose_criteria_cyc*.xyz (--no-copy-loose-xyz is ignored for this).
WRITE_MIN_LOOSE_CRITERIA_XYZ = False

# Geometry screening on path.xyz last frame (see check_qm_minimize.py).
ZN_N_SUSPECT = 2.5  # Å → Flag includes MIM_far if max Zn–Nmin(MIM) exceeds this
ZN_O_SUSPECT = 2.5  # Å → Flag includes MeOH_far if max Zn–O(MeOH) exceeds this

TSV_COL_MAX_ZN_NMIM = "max_Zn-Nmin_MIM(Å)"
TSV_COL_MAX_ZN_OMOH = "max_Zn-O_MeOH(Å)"
TSV_COL_FLAG = "Flag"

# If any of these appear in min.out, ``dlfind error`` is appended to ``Flag`` (comma-separated).
# Note: do not match ``ERROR: dlfind_module/dlf_update_sander is unimplemented`` — Amber prints that
# when restarting L-BFGS even when the run continues successfully.
MIN_OUT_DLFIND_ERROR_MARKERS: tuple[str, ...] = (
    "DL-Find encountered an error",
    "DL-FIND ERROR:",
)


def min_out_indicates_dlfind_error(text: str) -> bool:
    return any(m in text for m in MIN_OUT_DLFIND_ERROR_MARKERS)


def merge_dlfind_error_into_flag(rec: dict[str, object], min_out_text: str) -> None:
    if not min_out_indicates_dlfind_error(min_out_text):
        return
    tag = "dlfind error"
    cur = str(rec.get(TSV_COL_FLAG, "") or "").strip()
    parts = [p.strip() for p in cur.split(",") if p.strip()]
    if tag not in parts:
        parts.append(tag)
    rec[TSV_COL_FLAG] = ",".join(parts)


_zma_module: object | None = None
_zma_import_attempted = False


def _import_zif_meoh_assign_name():
    """Load zif_meoh_assign_name from repo ``python_scripts`` (one-time)."""
    global _zma_module, _zma_import_attempted
    if _zma_import_attempted:
        return _zma_module
    _zma_import_attempted = True
    try:
        repo_root = Path(__file__).resolve().parents[4]
        pscripts = repo_root / "python_scripts"
        if str(pscripts) not in sys.path:
            sys.path.insert(0, str(pscripts))
        import zif_meoh_assign_name as zm  # noqa: PLC0415

        _zma_module = zm
    except ImportError:
        _zma_module = None
    return _zma_module


def _dist3(a, b) -> float:
    ax, ay, az = (float(a[0]), float(a[1]), float(a[2]))
    bx, by, bz = (float(b[0]), float(b[1]), float(b[2]))
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def read_last_xyz_frame_text(path_xyz: Path) -> str:
    """Raw text of the last XYZ frame: ``natoms\\n`` + comment line + ``natoms`` coordinate lines.

    Matches ``check_qm_minimize.extract_last_frame_to_temp_xyz`` traversal (not ``parse_xyz_frames``,
    which skips blank lines and can desync comment/coords).
    """
    lines = path_xyz.read_text(errors="replace").splitlines(keepends=True)
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped and stripped.split()[0].isdigit():
            natoms = int(stripped.split()[0])
            start = i + 1
            if start >= len(lines):
                break
            comment = lines[start]
            start += 1
            end = start + natoms
            frame_body = lines[start:end]
            if len(frame_body) < natoms:
                break
            return f"{natoms}\n{comment}{''.join(frame_body)}"
        i -= 1
    raise ValueError(f"Cannot find a complete last frame in {path_xyz}")


def analyze_zn_ligand_last_frame(path_xyz: Path) -> tuple[float | None, float | None, str]:
    """Parse last path.xyz frame via zif_meoh_assign_name: (max Zn–Nmin MIM, max Zn–O MeOH, error tag).

    error tag is empty on success; otherwise a short reason (e.g. ``no_Zn``, ``Zn!=1``, ``no_zif_meoh_module``).
    """
    zma = _import_zif_meoh_assign_name()
    if zma is None:
        return None, None, "no_zif_meoh_module"

    try:
        frame_txt = read_last_xyz_frame_text(path_xyz)
    except (OSError, ValueError) as e:
        return None, None, f"path_xyz:{e!s}"[:120]

    fd, tmp = tempfile.mkstemp(suffix=".xyz", text=True)
    try:
        with os.fdopen(fd, "w") as tf:
            tf.write(frame_txt)
        u = zma.xyz_to_mda(tmp)
    except Exception as e:  # noqa: BLE001
        return None, None, f"xyz_to_mda:{e!s}"[:120]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    zn_atoms = u.select_atoms("resname ZN and name ZN")
    if len(zn_atoms) == 0:
        return None, None, "no_Zn"
    if len(zn_atoms) != 1:
        return None, None, "Zn!=1"

    zn_pos = zn_atoms.positions[0]
    mim_like_resnames = {"MIM", "IMH", "MIH"}
    mim_residues = [r for r in u.residues if r.resname in mim_like_resnames]
    moh_residues = [r for r in u.residues if r.resname == "MOH"]

    mim_zn_nmins: list[float] = []
    for res in mim_residues:
        ns = res.atoms.select_atoms("name N1 or name N2")
        if len(ns) == 0:
            ns = res.atoms.select_atoms("element N")
        if len(ns) == 0:
            continue
        min_d: float | None = None
        for n in ns:
            d = _dist3(zn_pos, n.position)
            if min_d is None or d < min_d:
                min_d = d
        if min_d is not None:
            mim_zn_nmins.append(min_d)

    moh_zn_os: list[float] = []
    for res in moh_residues:
        o = res.atoms.select_atoms("name O1")
        if len(o) == 0:
            o = res.atoms.select_atoms("element O")
        if len(o) == 0:
            continue
        moh_zn_os.append(_dist3(zn_pos, o.positions[0]))

    max_mim = max(mim_zn_nmins) if mim_zn_nmins else None
    max_moh = max(moh_zn_os) if moh_zn_os else None
    return max_mim, max_moh, ""


def geometry_tsv_fields(sys_dir: Path, system_rel: str, na: str) -> dict[str, str]:
    """Three TSV columns from path.xyz last frame: max Zn–Nmin(MIM), max Zn–O(MeOH), Flag."""
    if "_monomer" in system_rel:
        return {
            TSV_COL_MAX_ZN_NMIM: na,
            TSV_COL_MAX_ZN_OMOH: na,
            TSV_COL_FLAG: "skip_monomer",
        }

    path_xyz = sys_dir / "path.xyz"
    if not path_xyz.is_file():
        return {
            TSV_COL_MAX_ZN_NMIM: na,
            TSV_COL_MAX_ZN_OMOH: na,
            TSV_COL_FLAG: "no_path_xyz",
        }

    try:
        max_mim, max_moh, err = analyze_zn_ligand_last_frame(path_xyz)
    except Exception as e:  # noqa: BLE001
        return {
            TSV_COL_MAX_ZN_NMIM: na,
            TSV_COL_MAX_ZN_OMOH: na,
            TSV_COL_FLAG: f"ERROR:{e!s}"[:200],
        }

    if err:
        return {
            TSV_COL_MAX_ZN_NMIM: na,
            TSV_COL_MAX_ZN_OMOH: na,
            TSV_COL_FLAG: err[:200],
        }

    if max_mim is None and max_moh is None:
        return {
            TSV_COL_MAX_ZN_NMIM: na,
            TSV_COL_MAX_ZN_OMOH: na,
            TSV_COL_FLAG: "no_MIM_MeOH",
        }

    flags: list[str] = []
    if max_mim is not None and max_mim > ZN_N_SUSPECT:
        flags.append("MIM_far")
    if max_moh is not None and max_moh > ZN_O_SUSPECT:
        flags.append("MeOH_far")

    return {
        TSV_COL_MAX_ZN_NMIM: na if max_mim is None else f"{max_mim:.3f}",
        TSV_COL_MAX_ZN_OMOH: na if max_moh is None else f"{max_moh:.3f}",
        TSV_COL_FLAG: ",".join(flags),
    }


CYCLE_HEADER_RE = re.compile(r"Testing convergence\s+in cycle\s+(\d+)")
# Match convergence lines only (exclude "Energy calculation" etc.).
VAL_ENERGY = re.compile(r"(?m)^\s+Energy\s+([\d.E+-]+)\s+Target:")
VAL_MAX_STEP = re.compile(r"(?m)^\s*Max step\s+([\d.E+-]+)\s+Target:")
VAL_RMS_STEP = re.compile(r"(?m)^\s*RMS step\s+([\d.E+-]+)\s+Target:")
VAL_MAX_GRAD = re.compile(r"(?m)^\s*Max grad\s+([\d.E+-]+)\s+Target:")
VAL_RMS_GRAD = re.compile(r"(?m)^\s*RMS grad\s+([\d.E+-]+)\s+Target:")

# AMBER MD print: " Etot   =  -1688545.1730  EKtot   = ..." (kcal/mol).
# Do not use a bare "E" in the float class — it would match "EKtot" on the same line.
ETOT_RE = re.compile(
    r"(?m)^\s*Etot\s*=\s*(-?[\d.]+(?:[Ee][-+]?\d+)?)",
)

# Single-atom / no-DL-FIND QM/MM minimization often omits Etot but prints EXTERNESCF (kcal/mol).
EXTERNESCF_RE = re.compile(
    r"(?m)^\s*EXTERNESCF\s*=\s*(-?[\d.]+(?:[Ee][-+]?\d+)?)",
)

# Reference proton folder name: atomic H is taken as 0 kcal/mol (no reliable parse from failed min.out).
REFERENCE_H_SYSTEM_NAME = "H"

# Full metrics *within one cycle segment* (do not combine with DOTALL across headers:
# cycle 1 may be Max/RMS-grad-only while cycle 2 has Energy… — a global regex would pair
# "cycle 1" with cycle 2's numbers).
BLOCK_BODY_RE = re.compile(
    r"^\s+Energy\s+([\d.E+-]+)\s+Target:.*\n"
    r"\s*Max step\s+([\d.E+-]+)\s+Target:.*\n"
    r"\s*RMS step\s+([\d.E+-]+)\s+Target:.*\n"
    r"\s*Max grad\s+([\d.E+-]+)\s+Target:.*\n"
    r"\s*RMS grad\s+([\d.E+-]+)\s+Target:",
    re.MULTILINE,
)


def iter_convergence_segments(text: str):
    """Body text after each 'Testing convergence in cycle N' until the next such header."""
    for m in CYCLE_HEADER_RE.finditer(text):
        cycle = int(m.group(1))
        start = m.end()
        nxt = CYCLE_HEADER_RE.search(text, m.end())
        block = text[start : nxt.start() if nxt else len(text)]
        yield cycle, block


def iter_convergence_blocks(text: str):
    """Yield (cycle, metrics dict with float or None for each convergence line)."""
    keys = ("energy_change", "max_step", "rms_step", "max_grad", "rms_grad")
    patterns = (
        VAL_ENERGY,
        VAL_MAX_STEP,
        VAL_RMS_STEP,
        VAL_MAX_GRAD,
        VAL_RMS_GRAD,
    )
    for cycle, block in iter_convergence_segments(text):
        metrics: dict[str, float | None] = {}
        for key, pat in zip(keys, patterns, strict=True):
            mm = pat.search(block)
            metrics[key] = float(mm.group(1)) if mm else None
        yield cycle, metrics


def first_cycle_all_four_crit(text: str) -> int | None:
    """First cycle where energy, max_step, rms_step, max_grad are all present and <= targets."""
    for cycle, metrics in iter_convergence_blocks(text):
        ok = True
        for key, target in FOUR_CRIT_TARGETS.items():
            v = metrics.get(key)
            if v is None or v > target:
                ok = False
                break
        if ok:
            return cycle
    return None


def has_convergence_reached(text: str) -> bool:
    return CONVERGENCE_REACHED_MARKER in text


def etot_kcal_mol_by_convergence_cycle(text: str) -> dict[int, float]:
    """For each 'Testing convergence in cycle N', Etot = last AMBER Etot (kcal/mol) before that header."""
    out: dict[int, float] = {}
    for m in CYCLE_HEADER_RE.finditer(text):
        pos = m.start()
        cycle = int(m.group(1))
        chunk = text[:pos]
        last_e: re.Match[str] | None = None
        for em in ETOT_RE.finditer(chunk):
            last_e = em
        if last_e is not None:
            out[cycle] = float(last_e.group(1))
    return out


def format_etot_kcal_mol(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.12g}"


def last_etot_kcal_mol_from_min_out(text: str) -> float | None:
    """Last total-like energy in kcal/mol: prefer AMBER Etot lines, else last EXTERNESCF."""
    last_e: re.Match[str] | None = None
    for em in ETOT_RE.finditer(text):
        last_e = em
    if last_e is not None:
        return float(last_e.group(1))
    last_ex: re.Match[str] | None = None
    for em in EXTERNESCF_RE.finditer(text):
        last_ex = em
    if last_ex is not None:
        return float(last_ex.group(1))
    return None


def parse_xyz_frames(path: Path) -> list[str]:
    """Split path.xyz into frames; each frame is natoms line + n atom lines (kcal/mol coords)."""
    lines = path.read_text(errors="replace").splitlines()
    frames: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        try:
            n = int(s.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        block_lines = [lines[i]]
        i += 1
        count = 0
        while count < n and i < len(lines):
            if lines[i].strip():
                block_lines.append(lines[i])
                count += 1
            i += 1
        if count == n:
            frames.append("\n".join(block_lines) + "\n")
    return frames


def write_min_loose_criteria_xyz(sys_dir: Path, cycle_1based: int) -> None:
    """Write path.xyz frame (cycle_1based - 1) to min_loose_criteria_cyc{cycle}.xyz."""
    path_xyz = sys_dir / "path.xyz"
    if not path_xyz.is_file():
        print(f"parse_min_last_convergence: missing {path_xyz}", file=sys.stderr)
        return
    frames = parse_xyz_frames(path_xyz)
    idx = cycle_1based - 1
    if idx < 0 or idx >= len(frames):
        print(
            f"parse_min_last_convergence: {path_xyz} has {len(frames)} frames, "
            f"need index {idx} for cycle {cycle_1based}",
            file=sys.stderr,
        )
        return
    out = sys_dir / f"min_loose_criteria_cyc{cycle_1based}.xyz"
    out.write_text(frames[idx])
    print(
        f"parse_min_last_convergence: wrote {out.resolve()} "
        f"(path.xyz frame {idx}, cycle {cycle_1based})"
    )


def _fmt_convergence_metric(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.12g}"


# Written to TSV as scientific notation with one decimal (e.g. 3.8E-04).
TSV_SCI_ONE_DECIMAL_COLUMNS: tuple[str, ...] = (
    "energy_change",
    "max_step",
    "rms_step",
    "max_grad",
    "rms_grad",
)


def apply_reference_h_etot(rec: dict[str, object], system_rel: str) -> None:
    if system_rel != REFERENCE_H_SYSTEM_NAME:
        return
    z = format_etot_kcal_mol(0.0)
    rec["Etot_last_cycle_kcal_mol"] = z
    rec["Etot_loose_criteria_cycle_kcal_mol"] = z


def format_tsv_scientific_one_decimal(val: object) -> str:
    """N/A preserved; numeric values -> uppercase scientific with one decimal."""
    if val is None:
        return "N/A"
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return "N/A"
    try:
        x = float(s)
    except ValueError:
        return s
    return f"{x:.1E}"


def last_convergence_block(text: str) -> dict[str, str | int] | None:
    """Last convergence cycle: prefer a full Energy…RMS grad block; else last partial cycle.

    DL-FIND sometimes stops after one step and only prints Max grad / RMS grad (no Energy,
    Max step, or RMS step lines). Those runs used to yield no match and all-N/A rows.
    """
    last_full: dict[str, str | int] | None = None
    for cycle, block in iter_convergence_segments(text):
        m = BLOCK_BODY_RE.search(block)
        if m:
            last_full = {
                "cycle": cycle,
                "energy_change": m.group(1),
                "max_step": m.group(2),
                "rms_step": m.group(3),
                "max_grad": m.group(4),
                "rms_grad": m.group(5),
            }
    if last_full is not None:
        return last_full
    last: tuple[int, dict[str, float | None]] | None = None
    for cycle, metrics in iter_convergence_blocks(text):
        last = (cycle, metrics)
    if last is None:
        return None
    cycle, metrics = last
    if all(v is None for v in metrics.values()):
        return None
    return {
        "cycle": cycle,
        "energy_change": _fmt_convergence_metric(metrics.get("energy_change")),
        "max_step": _fmt_convergence_metric(metrics.get("max_step")),
        "rms_step": _fmt_convergence_metric(metrics.get("rms_step")),
        "max_grad": _fmt_convergence_metric(metrics.get("max_grad")),
        "rms_grad": _fmt_convergence_metric(metrics.get("rms_grad")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("min_last_convergence.tsv"),
        help="Output TSV path (default: ./min_last_convergence.tsv)",
    )
    ap.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory containing */min.out (default: current directory)",
    )
    ap.add_argument(
        "--no-copy-loose-xyz",
        action="store_true",
        help="Do not write min_loose_criteria_cyc*.xyz when loose criteria met but not converged_signal",
    )
    args = ap.parse_args()
    root = args.root.resolve()

    min_outs = sorted(root.glob("*/min.out"))
    if not min_outs:
        print(f"No */min.out under {root}", file=sys.stderr)
        return 1

    out_path = args.output
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    na = "N/A"
    records: list[dict[str, object]] = []
    for p in min_outs:
        rel = p.parent.name
        try:
            txt = p.read_text(errors="replace")
        except OSError as e:
            rec = {
                "system": rel,
                "cycle": "ERROR",
                "energy_change": str(e),
                "max_step": na,
                "rms_step": na,
                "max_grad": na,
                "rms_grad": na,
                "cycle_to_reach_loose_criteria": na,
                "converged_loose_criteria": na,
                "Etot_last_cycle_kcal_mol": na,
                "Etot_loose_criteria_cycle_kcal_mol": na,
                "converged_signal": na,
            }
            rec.update(geometry_tsv_fields(p.parent, rel, na))
            records.append(rec)
            continue
        etot_map = etot_kcal_mol_by_convergence_cycle(txt)
        conv_col = "yes" if has_convergence_reached(txt) else "no"
        # Last-block metrics vs first cycle meeting the four thresholds (independent).
        c4 = first_cycle_all_four_crit(txt)
        c4_col = str(c4) if c4 is not None else na
        loose_ok_col = "yes" if c4 is not None else "no"
        info = last_convergence_block(txt)
        sys_dir = p.parent
        if info is None:
            e_loose = etot_map.get(c4) if c4 is not None else None
            e_last_fb = last_etot_kcal_mol_from_min_out(txt)
            rec = {
                "system": rel,
                "cycle": na,
                "energy_change": na,
                "max_step": na,
                "rms_step": na,
                "max_grad": na,
                "rms_grad": na,
                "cycle_to_reach_loose_criteria": c4_col,
                "converged_loose_criteria": loose_ok_col,
                "Etot_last_cycle_kcal_mol": format_etot_kcal_mol(e_last_fb),
                "Etot_loose_criteria_cycle_kcal_mol": format_etot_kcal_mol(e_loose),
                "converged_signal": conv_col,
            }
            rec.update(geometry_tsv_fields(sys_dir, rel, na))
            merge_dlfind_error_into_flag(rec, txt)
            apply_reference_h_etot(rec, rel)
            records.append(rec)
            if (
                WRITE_MIN_LOOSE_CRITERIA_XYZ
                and c4 is not None
                and conv_col == "no"
                and not args.no_copy_loose_xyz
            ):
                write_min_loose_criteria_xyz(sys_dir, c4)
        else:
            i = info
            last_cy = int(i["cycle"])
            e_last = etot_map.get(last_cy)
            if e_last is None:
                e_last = last_etot_kcal_mol_from_min_out(txt)
            e_loose = etot_map.get(c4) if c4 is not None else None
            rec = {
                "system": rel,
                "cycle": i["cycle"],
                "energy_change": i["energy_change"],
                "max_step": i["max_step"],
                "rms_step": i["rms_step"],
                "max_grad": i["max_grad"],
                "rms_grad": i["rms_grad"],
                "cycle_to_reach_loose_criteria": c4_col,
                "converged_loose_criteria": loose_ok_col,
                "Etot_last_cycle_kcal_mol": format_etot_kcal_mol(e_last),
                "Etot_loose_criteria_cycle_kcal_mol": format_etot_kcal_mol(e_loose),
                "converged_signal": conv_col,
            }
            rec.update(geometry_tsv_fields(sys_dir, rel, na))
            merge_dlfind_error_into_flag(rec, txt)
            apply_reference_h_etot(rec, rel)
            records.append(rec)
            if (
                WRITE_MIN_LOOSE_CRITERIA_XYZ
                and c4 is not None
                and conv_col == "no"
                and not args.no_copy_loose_xyz
            ):
                write_min_loose_criteria_xyz(sys_dir, c4)

    df = pd.DataFrame.from_records(records).sort_values(
        "system",
        ascending=True,
        na_position="last",
        key=lambda s: pd.to_numeric(s, errors="coerce"),
    )

    for _col in TSV_SCI_ONE_DECIMAL_COLUMNS:
        df[_col] = df[_col].map(format_tsv_scientific_one_decimal)
    df['converged_signal_or_converged_loose_criteria'] = (
        (df['converged_signal'] == "yes") | (df['converged_loose_criteria'] == "yes")
    ).map({True: "yes", False: "no"})
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        columns=OUTPUT_TSV_COLUMNS,
    )
    return 0


# --- TSV column order: edit this list only (names must match record dict keys). ---
OUTPUT_TSV_COLUMNS: list[str] = [
    "system",
    "cycle",
    "converged_signal_or_converged_loose_criteria",
    "energy_change",
    "max_step",
    "rms_step",
    "max_grad",
    "rms_grad",
    "converged_signal",
    "converged_loose_criteria",
    "cycle_to_reach_loose_criteria",
    "Etot_last_cycle_kcal_mol",
    "Etot_loose_criteria_cycle_kcal_mol",
    TSV_COL_MAX_ZN_NMIM,
    TSV_COL_MAX_ZN_OMOH,
    TSV_COL_FLAG,
]


if __name__ == "__main__":
    raise SystemExit(main())
