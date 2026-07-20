#!/usr/bin/env python3
"""
For 1Zn + MIm / MImH / MeOH complexes with total ligand count 4, 5, or 6, create
missing structures under run35_new (sibling of run35 under qm_minimize) by transferring one proton between MIm <-> MImH
from the last frame of path.xyz of a neighboring stoichiometry in run35.

When several MIH (deprotonation) or several MIM (protonation) ligands exist, every
site is enumerated; DFTB+ single-point energies are run (see --dftb-template) and the
lowest-energy isomer is written to manually_created_<system>.xyz.
Each new system directory copies prepare.sh and orc_job.tpl from the same reference
folder as the chosen geometry (same path resolution as prepare.sh).

References are taken from --run35-root (path.xyz) and, in later waves, from
--out-root using path.xyz or manually_created_*.xyz so chains missing an initial
run35 parent can still be built iteratively.

Skips targets that already have a subdirectory under --check-dir (default: run35), i.e. only
builds stoichiometries missing from run35. --force only overwrites existing dirs under --out-root.
Use --no-dftb-screen to only use the
first ligand (no DFTB+). Use --only-systems to restrict which formulas are built.
Dry-run only resolves references under --run35-root (no simulated run35_new).
By default prints a full reference-selection trace on stderr; use -q / --quiet to
suppress it. Each successful build also prints one summary line (ref folder, geometry,
mode, wave).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_QM_MINIMIZE_ROOT = Path(__file__).resolve().parent  # .../qm_minimize

import numpy as np

# zif_meoh_assign_name adds nbZIFFF-km to path
sys.path.insert(0, "/home/zbai29/data/qmmm_test")

import MDAnalysis as mda

sys.path.append("/home/zbai29/JR/data/nbZIFFF-km")
import tools.xyz2ZIFFFlmp as xyz2ZIFFFlmp

_MAX_COVALENT_OH_DIST = 1.10

# DFTB+ 3ob template species; geometry may omit O (e.g. 0 MeOH) — Hamiltonian must match.
_DFTB_ALLOWED = frozenset({"Zn", "C", "N", "O", "H"})
_DFTB_SPECIES_ORDER = ("Zn", "C", "N", "O", "H")
_ELEM_LINE_RE = re.compile(r"^\s*(Zn|[CHNO])\s*=")
# Local copy of iter_struc/gen_struc_dftbplus/122/work_dir/sp_dftb_in.hsd
_DEFAULT_DFTB_TEMPLATE = _QM_MINIMIZE_ROOT / "sp_dftb_in.hsd"


def dftb_species_subset(elems: list[str]) -> list[str]:
    """Ordered list of element symbols present in the cluster (subset of template species)."""
    present = set(elems)
    unknown = present - _DFTB_ALLOWED
    if unknown:
        raise ValueError(f"Unsupported element(s) for DFTB+ template: {unknown}")
    return [s for s in _DFTB_SPECIES_ORDER if s in present]


def filter_dftb_hamiltonian_for_species(text: str, present: frozenset[str]) -> str:
    """
    Drop HubbardDerivs / MaxAngularMomentum / SlaterKosterFiles lines for species
    not in `present`. DFTB+ rejects inputs that list O parameters when no O atoms exist.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    section: str | None = None

    for line in lines:
        if "HubbardDerivs = {" in line:
            section = "hubbard"
            out.append(line)
            continue
        if "MaxAngularMomentum = {" in line:
            section = "maxang"
            out.append(line)
            continue
        if "SlaterKosterFiles = {" in line:
            section = "sk"
            out.append(line)
            continue

        if section:
            if line.strip() == "}":
                section = None
                out.append(line)
                continue
            if section == "hubbard":
                m = _ELEM_LINE_RE.match(line)
                if m and m.group(1) not in present:
                    continue
                out.append(line)
            elif section == "maxang":
                m = _ELEM_LINE_RE.match(line)
                if m and m.group(1) not in present:
                    continue
                out.append(line)
            elif section == "sk":
                s = line.strip()
                if s.startswith("Prefix"):
                    out.append(line)
                    continue
                if "-" in s and ".skf" in s and "=" in s:
                    key = s.split("=", 1)[0].strip()
                    if "-" in key:
                        a, b = key.split("-", 1)
                        if a in present and b in present:
                            out.append(line)
                    continue
                out.append(line)
            continue

        out.append(line)

    return "".join(out)


def formula_name(a: int, b: int, c: int) -> str:
    return f"1Zn_{a}MIm_{b}MImH_{c}MeOH"


def parse_formula_name(name: str) -> tuple[int, int, int] | None:
    m = re.match(r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def iter_targets(totals: tuple[int, ...]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for total in totals:
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                if c >= 0:
                    out.append((a, b, c))
    return out


def read_path_xyz_frames(path: Path) -> tuple[list[str], int]:
    """Return (list of frame blocks as single strings, 1-based index of last frame)."""
    text = path.read_text()
    lines = text.splitlines()
    i = 0
    frames: list[str] = []
    while i < len(lines):
        nat = int(lines[i].strip())
        block = lines[i : i + 2 + nat]
        frames.append("\n".join(block) + "\n")
        i += 2 + nat
    if not frames:
        raise ValueError(f"No frames in {path}")
    cycid = len(frames)
    return frames, cycid


def load_universe_guess_bonds(xyz_path: str) -> mda.Universe:
    u = mda.Universe(xyz_path)
    vdwradii_for_bondguess = xyz2ZIFFFlmp.vdwradii_for_bondguess.copy()
    vdwradii_for_bondguess["Zn"] = -999
    vdwradii_for_bondguess["ZN"] = -999
    u.atoms.guess_bonds(vdwradii=vdwradii_for_bondguess)
    hh_bonds = [b for b in u.bonds if b[0].element == "H" and b[1].element == "H"]
    if hh_bonds:
        u.delete_bonds(hh_bonds)
    oh_spurious = []
    for b in u.bonds:
        a1, a2 = b[0], b[1]
        e1, e2 = a1.element, a2.element
        if (e1 == "O" and e2 == "H") or (e1 == "H" and e2 == "O"):
            d = float(np.linalg.norm(a1.position - a2.position))
            if d > _MAX_COVALENT_OH_DIST:
                oh_spurious.append(b)
    if oh_spurious:
        u.delete_bonds(oh_spurious)
    u.atoms.fragments
    return u


def zn_atom(u: mda.Universe) -> mda.Atom:
    z = u.select_atoms("element Zn")
    if len(z) != 1:
        raise ValueError(f"Expected exactly one Zn, got {len(z)}")
    return z[0]


def mih_ligand_fragments(u: mda.Universe) -> list:
    """All 12-atom MIH (protonated imidazole) fragments."""
    return [fr for fr in u.atoms.fragments if len(fr.atoms) == 12]


def mim_ligand_fragments(u: mda.Universe) -> list:
    """All 11-atom MIM (deprotonated imidazole) fragments."""
    return [fr for fr in u.atoms.fragments if len(fr.atoms) == 11]


def first_mih_fragment(u: mda.Universe):
    frs = mih_ligand_fragments(u)
    return frs[0] if frs else None


def first_mim_fragment(u: mda.Universe):
    frs = mim_ligand_fragments(u)
    return frs[0] if frs else None


def _nh_on_mih(u: mda.Universe, fr) -> mda.Atom:
    for n in fr.select_atoms("element N"):
        for b in u.bonds:
            if n in b:
                other = b[0] if b[1] == n else b[1]
                if other.element == "H":
                    return other
    raise ValueError("No N-H found on MIH fragment")


def deprotonate_mih_fragment(u: mda.Universe, fr) -> mda.Universe:
    """Remove the acidic N-H from a specific MIH (12-atom) ligand."""
    if len(fr.atoms) != 12:
        raise ValueError("deprotonate_mih_fragment: expected 12-atom fragment")
    h_remove = _nh_on_mih(u, fr)
    keep = u.atoms - h_remove
    return mda.Merge(keep)


def deprotonate_one_mih(u: mda.Universe) -> mda.Universe:
    """Remove one acidic N-H from the first MIH (12-atom) ligand."""
    fr = first_mih_fragment(u)
    if fr is None:
        raise ValueError("No MIH (12-atom) fragment found for deprotonation")
    return deprotonate_mih_fragment(u, fr)


def pick_mim_n_for_protonation(fr, zn: mda.Atom) -> mda.Atom:
    ns = fr.select_atoms("element N")
    if len(ns) != 2:
        raise ValueError("MIM fragment should have 2 N atoms")
    n0, n1 = ns[0], ns[1]
    d0 = float(np.linalg.norm(n0.position - zn.position))
    d1 = float(np.linalg.norm(n1.position - zn.position))
    # Protonate the N farther from Zn (non-coordinating imidazolate N).
    return n0 if d0 >= d1 else n1


def protonate_mim_fragment_coords(u: mda.Universe, fr) -> tuple[list[str], np.ndarray]:
    """Add one H at the non-coordinating N of a specific MIM (11-atom) ligand."""
    if len(fr.atoms) != 11:
        raise ValueError("protonate_mim_fragment_coords: expected 11-atom fragment")
    zn = zn_atom(u)
    n = pick_mim_n_for_protonation(fr, zn)
    v = n.position - zn.position
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        v = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    h_pos = n.position + 1.01 * (v / norm)
    elems = [a.element for a in u.atoms] + ["H"]
    pos = np.vstack([u.atoms.positions, h_pos])
    return elems, pos


def protonate_one_mim_coords(u: mda.Universe) -> tuple[list[str], np.ndarray]:
    """Add one H to the non-coordinating N of the first MIM (11-atom) ligand."""
    fr = first_mim_fragment(u)
    if fr is None:
        raise ValueError("No MIM (11-atom) fragment found for protonation")
    return protonate_mim_fragment_coords(u, fr)


def universe_to_xyz_string(u: mda.Universe, title: str = "generated") -> str:
    elems = [a.element for a in u.atoms]
    pos = u.atoms.positions
    lines = [str(len(elems)), title]
    for sym, xyz in zip(elems, pos):
        lines.append(f"{sym:2s} {xyz[0]:13.7f} {xyz[1]:13.7f} {xyz[2]:13.7f}")
    return "\n".join(lines) + "\n"


def elems_pos_to_xyz_string(elems: list[str], pos: np.ndarray, title: str = "generated") -> str:
    lines = [str(len(elems)), title]
    for sym, xyz in zip(elems, pos):
        lines.append(f"{sym:2s} {xyz[0]:13.7f} {xyz[1]:13.7f} {xyz[2]:13.7f}")
    return "\n".join(lines) + "\n"


def _geometry_block_end_line(lines: list[str], start_i: int) -> int:
    depth = 0
    i = start_i
    while i < len(lines):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError("Unclosed Geometry block in DFTB+ template")


def replace_geometry_in_hsd(template: str, geometry_block: str) -> str:
    lines = template.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("Geometry = {"):
            end = _geometry_block_end_line(lines, i)
            return "".join(lines[:i]) + geometry_block.rstrip() + "\n" + "".join(lines[end + 1 :])
    raise ValueError("No Geometry = { block in DFTB+ template")


def elems_pos_to_dftb_geometry_block(elems: list[str], pos: np.ndarray) -> str:
    species = dftb_species_subset(elems)
    type_index = {s: i + 1 for i, s in enumerate(species)}
    typenames_inner = '" "'.join(species)
    lines = [
        "Geometry = {",
        "  TypeNames = {",
        f' "{typenames_inner}"',
        "  }",
        # MDAnalysis / path.xyz use Å; DFTB+ default for TypesAndCoordinates is Bohr — require Angstrom.
        "  TypesAndCoordinates [Angstrom] = {",
    ]
    for el, p in zip(elems, pos):
        ti = type_index[el]
        lines.append(
            f" {ti} {float(p[0]):.16e} {float(p[1]):.16e} {float(p[2]):.16e}"
        )
    lines.extend(
        [
            "  }",
            "  Periodic = No",
            "  Helical = No",
            "}",
        ]
    )
    return "\n".join(lines)


def patch_dftb_hamiltonian_options(template: str, charge: int) -> str:
    t = re.sub(
        r"^\s*Charge\s*=\s*[^\n]+",
        f"  Charge = {charge}",
        template,
        count=1,
        flags=re.MULTILINE,
    )
    t = re.sub(
        r"^\s*WriteDetailedOut\s*=\s*No",
        "  WriteDetailedOut = Yes",
        t,
        count=1,
        flags=re.MULTILINE,
    )
    return t


def build_dftb_in_hsd(
    template_path: Path,
    elems: list[str],
    pos: np.ndarray,
    charge: int,
) -> str:
    raw = template_path.read_text()
    raw = patch_dftb_hamiltonian_options(raw, charge)
    present = frozenset(dftb_species_subset(elems))
    raw = filter_dftb_hamiltonian_for_species(raw, present)
    geom = elems_pos_to_dftb_geometry_block(elems, pos)
    return replace_geometry_in_hsd(raw, geom)


def parse_dftb_total_energy_hartree(log_text: str) -> float | None:
    for line in log_text.splitlines():
        m = re.search(r"Total Energy:\s+([-\d.E+]+)\s+H", line)
        if m:
            return float(m.group(1))
    return None


def universe_to_elems_pos(u: mda.Universe) -> tuple[list[str], np.ndarray]:
    elems = [a.element for a in u.atoms]
    pos = u.atoms.positions.copy()
    return elems, pos


def enumerate_proton_candidates(
    u: mda.Universe, mode: str
) -> list[tuple[str, mda.Universe | None, tuple[list[str], np.ndarray] | None]]:
    """
    Each entry: (label, universe_or_none, elems_pos_or_none).
    Deprot: merged universe; prot: (elems, pos) tuple (no single Universe).
    """
    out: list[tuple[str, mda.Universe | None, tuple[list[str], np.ndarray] | None]] = []
    if mode == "deprot":
        frs = mih_ligand_fragments(u)
        if not frs:
            raise ValueError("No MIH fragments for deprotonation enumeration")
        for i, fr in enumerate(frs):
            u2 = deprotonate_mih_fragment(u, fr)
            out.append((f"deprot_mih_{i:02d}", u2, None))
    else:
        frs = mim_ligand_fragments(u)
        if not frs:
            raise ValueError("No MIM fragments for protonation enumeration")
        for i, fr in enumerate(frs):
            elems, pos = protonate_mim_fragment_coords(u, fr)
            out.append((f"prot_mim_{i:02d}", None, (elems, pos)))
    return out


def candidate_to_elems_pos(
    label: str,
    u_deprot: mda.Universe | None,
    elems_pos: tuple[list[str], np.ndarray] | None,
) -> tuple[list[str], np.ndarray]:
    if u_deprot is not None:
        return universe_to_elems_pos(u_deprot)
    assert elems_pos is not None
    return elems_pos


def candidate_to_xyz_string(
    label: str,
    u_deprot: mda.Universe | None,
    elems_pos: tuple[list[str], np.ndarray] | None,
    title: str,
) -> str:
    elems, pos = candidate_to_elems_pos(label, u_deprot, elems_pos)
    return elems_pos_to_xyz_string(elems, pos, title=title)


def run_dftb_plus_single(
    workdir: Path,
    hsd_content: str,
    dftbplus_cmd: str,
) -> tuple[float | None, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "dftb_in.hsd"
    log_path = workdir / "dftbplus.log"
    inp.write_text(hsd_content)
    try:
        proc = subprocess.run(
            [dftbplus_cmd],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except FileNotFoundError:
        return None, f"command not found: {dftbplus_cmd}"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    full_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(full_log)
    e = parse_dftb_total_energy_hartree(full_log)
    if e is None and proc.returncode != 0:
        return None, f"DFTB+ exit {proc.returncode}; see {log_path}"
    if e is None:
        return None, f"No Total Energy in log (exit {proc.returncode})"
    return e, full_log


def screen_candidates_dftb(
    candidates: list[tuple[str, mda.Universe | None, tuple[list[str], np.ndarray] | None]],
    *,
    template_path: Path,
    charge: int,
    out_screen_root: Path,
    dftbplus_cmd: str,
) -> tuple[str, float | None, list[tuple[str, float | None]]]:
    """
    Returns (best_label, best_energy_hartree, summary list of (label, energy or None)).
    """
    summary: list[tuple[str, float | None]] = []
    best_label: str | None = None
    best_e: float | None = None

    for lab, u_deprot, ep in candidates:
        elems, pos = candidate_to_elems_pos(lab, u_deprot, ep)
        try:
            hsd = build_dftb_in_hsd(template_path, elems, pos, charge)
        except Exception as ex:
            summary.append((lab, None))
            print(f"WARN build DFTB input {lab}: {ex}", file=sys.stderr)
            continue
        wd = out_screen_root / lab
        e, _log = run_dftb_plus_single(wd, hsd, dftbplus_cmd)
        summary.append((lab, e))
        (wd / "candidate.xyz").write_text(
            elems_pos_to_xyz_string(elems, pos, title=lab)
        )
        if e is not None and (best_e is None or e < best_e):
            best_e = e
            best_label = lab

    if best_label is None and candidates:
        best_label = candidates[0][0]
        print(
            f"WARN: DFTB+ failed for all isomers; using first candidate {best_label}. "
            f"See {out_screen_root}",
            file=sys.stderr,
        )
        return best_label, None, summary
    if best_label is None:
        raise RuntimeError(
            f"DFTB+ screening failed and no candidates; see {out_screen_root}"
        )
    return best_label, best_e, summary


def read_min_last_convergence(tsv_path: Path) -> dict[str, dict[str, str]]:
    if not tsv_path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with tsv_path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row.get("system", "").strip()
            if s:
                out[s] = row
    return out


def charge_mol_for_stoichiometry(a_mim: int) -> int:
    """Same rule as run35 prepare.sh: charge_MOL = 2 - (# MIm)."""
    return 2 - a_mim


def ref_candidates(target: tuple[int, int, int]) -> list[tuple[str, str]]:
    """Return list of (reference_name, mode) with mode 'deprot' or 'prot'."""
    a, b, c = target
    cand: list[tuple[str, str]] = []
    if a >= 1:
        cand.append((formula_name(a - 1, b + 1, c), "deprot"))
    if b >= 1:
        cand.append((formula_name(a + 1, b - 1, c), "prot"))
    return cand


def read_reference_geometry_block(geom_file: Path) -> tuple[str, int]:
    """
    Load one XYZ structure for proton transfer. If geom_file is path.xyz, use the
    last frame; otherwise treat as a single-structure .xyz (e.g. manually_created).
    Returns (xyz_block, cycid) with cycid = frame index (1-based) for path.xyz else 1.
    """
    if geom_file.name == "path.xyz":
        frames, cycid = read_path_xyz_frames(geom_file)
        return frames[-1], cycid
    text = geom_file.read_text()
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid xyz: {geom_file}")
    nat = int(lines[0].strip())
    block_lines = lines[: 2 + nat]
    return "\n".join(block_lines) + "\n", 1


def _geometry_score_for_ref(
    ref_name: str,
    root: Path,
    primary_root: Path,
    tsv: dict[str, dict[str, str]],
) -> tuple[tuple[int, int, int], Path] | None:
    """
    Best geometry file under root/ref_name/. Returns (sort_key, path) or None.
    sort_key larger is better: prefer primary run35, path.xyz over manual, converged, long path.xyz.
    """
    ref_dir = root / ref_name
    if not ref_dir.is_dir():
        return None
    row = tsv.get(ref_name)
    conv = 1 if row and row.get("converged_signal") == "yes" else 0
    is_primary = root.resolve() == primary_root.resolve()
    best: tuple[tuple[int, int, int], Path] | None = None

    px = ref_dir / "path.xyz"
    if px.is_file():
        try:
            _, ncyc = read_path_xyz_frames(px)
        except (ValueError, OSError):
            ncyc = 1
        tier = 3 if is_primary else 2
        key = (tier, conv, ncyc)
        best = (key, px)

    mx = ref_dir / f"manually_created_{ref_name}.xyz"
    if mx.is_file():
        tier = 1 if is_primary else 0
        key = (tier, conv, 1)
        if best is None or key > best[0]:
            best = (key, mx)

    return best


def pick_reference(
    target: tuple[int, int, int],
    search_roots: list[Path],
    primary_root: Path,
    tsv: dict[str, dict[str, str]],
) -> tuple[str, str, Path] | None:
    """
    Choose reference stoichiometry and geometry file among search_roots (run35, then run35_new, ...).
    Returns (ref_name, mode, path_to_path.xyz_or_manually_created.xyz) or None.
    """
    best: tuple[tuple[int, int, int], str, str, Path] | None = None
    for ref_name, mode in ref_candidates(target):
        for root in search_roots:
            got = _geometry_score_for_ref(ref_name, root, primary_root, tsv)
            if got is None:
                continue
            key, gpath = got
            tie = (key, ref_name)
            if best is None or tie > (best[0], best[1]):
                best = (key, ref_name, mode, gpath)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _root_tag(root: Path, primary_root: Path, out_root: Path | None) -> str:
    r = root.resolve()
    if r == primary_root.resolve():
        return "run35-root (primary)"
    if out_root is not None and r == out_root.resolve():
        return "out-root (run35_new)"
    return f"search-root ({r})"


def reference_selection_report(
    target: tuple[int, int, int],
    search_roots: list[Path],
    primary_root: Path,
    out_root: Path | None,
    tsv: dict[str, dict[str, str]],
    *,
    wave: int | None = None,
) -> list[str]:
    """
    Human-readable lines explaining which reference edges exist and which wins
    (same rules as pick_reference). sort_key = (tier, conv_flag, n_frames_or_1).
    tier: 3=primary+path.xyz, 2=extra+path.xyz, 1=primary+manual, 0=extra+manual.
    """
    lines: list[str] = []
    tgt = formula_name(*target)
    wv = f" wave={wave}" if wave is not None else ""
    lines.append(f"[ref-choice]{wv} target directory (to build): {tgt}")
    edges = ref_candidates(target)
    if not edges:
        lines.append("  (no valid deprot/prot edges: need a>=1 or b>=1)")
    for ref_name, mode in edges:
        role = (
            "deprot (remove N-H from one MIH → +1 MIm, -1 MImH)"
            if mode == "deprot"
            else "prot (add H on one MIM → -1 MIm, +1 MImH)"
        )
        lines.append(f"  candidate edge: mode={mode}  ({role})")
        lines.append(f"    reference stoichiometry (folder name): {ref_name}")
        for root in search_roots:
            tag = _root_tag(root, primary_root, out_root)
            ref_dir = root / ref_name
            if not ref_dir.is_dir():
                lines.append(f"    · {tag}: no directory → skip")
                lines.append(f"        path checked: {ref_dir}")
                continue
            px = ref_dir / "path.xyz"
            mx = ref_dir / f"manually_created_{ref_name}.xyz"
            parts: list[str] = []
            if px.is_file():
                try:
                    _, nf = read_path_xyz_frames(px)
                    parts.append(f"path.xyz with {nf} frame(s), use last → cycid={nf}")
                except (ValueError, OSError) as e:
                    parts.append(f"path.xyz present but unreadable ({e})")
            if mx.is_file():
                parts.append(f"manually_created_{ref_name}.xyz (single structure)")
            row = tsv.get(ref_name)
            conv = row.get("converged_signal") if row else None
            conv_s = f"min_last_convergence converged_signal={conv!r}" if conv is not None else "not in min_last_convergence.tsv"
            if not parts:
                lines.append(
                    f"    · {tag}: directory exists but no path.xyz or manually_created file; {conv_s}"
                )
                continue
            got = _geometry_score_for_ref(ref_name, root, primary_root, tsv)
            if got is None:
                lines.append(f"    · {tag}: {', '.join(parts)}; {conv_s} → no score")
                continue
            key, gpath = got
            lines.append(
                f"    · {tag}: {', '.join(parts)}; {conv_s}"
            )
            lines.append(
                f"        → competing geometry: {gpath.name}  sort_key={key}  (higher wins)"
            )

    picked = pick_reference(target, search_roots, primary_root, tsv)
    if picked is None:
        lines.append("  => CHOSEN: none (no geometry on any edge × root)")
    else:
        ref_name, mode, geom_path = picked
        pr = resolve_prepare_sh(ref_name, geom_path, search_roots)
        tpl = resolve_orc_job_tpl(ref_name, geom_path, search_roots)
        lines.append("  => CHOSEN (highest sort_key; tie-breaker: lexicographic ref name):")
        lines.append(f"      reference folder name: {ref_name}")
        lines.append(f"      reference directory (absolute): {geom_path.parent.resolve()}")
        lines.append(f"      geometry file: {geom_path.name}")
        lines.append(f"      geometry path (absolute): {geom_path.resolve()}")
        lines.append(f"      mode: {mode}")
        lines.append(f"      prepare.sh source: {pr.resolve() if pr else 'MISSING'}")
        lines.append(f"      orc_job.tpl source: {tpl.resolve() if tpl else 'MISSING'}")
    lines.append("")
    return lines


def format_build_summary_line(
    target_name: str,
    ref_name: str,
    mode: str,
    geom_path: Path,
    wave: int,
    *,
    dry_run: bool,
) -> str:
    prefix = "Would build" if dry_run else "Built"
    return (
        f"{prefix} {target_name}  ←  ref_dir={ref_name}  "
        f"ref_path={geom_path.parent.resolve()}  "
        f"geom={geom_path.name}  mode={mode}  wave={wave}"
    )


def resolve_prepare_sh(ref_name: str, geom_path: Path, search_roots: list[Path]) -> Path | None:
    """prepare.sh next to geometry, else first search_roots/ref_name/prepare.sh."""
    p = geom_path.parent / "prepare.sh"
    if p.is_file():
        return p
    for root in search_roots:
        q = root / ref_name / "prepare.sh"
        if q.is_file():
            return q
    return None


def resolve_orc_job_tpl(ref_name: str, geom_path: Path, search_roots: list[Path]) -> Path | None:
    """orc_job.tpl next to geometry, else first search_roots/ref_name/orc_job.tpl."""
    p = geom_path.parent / "orc_job.tpl"
    if p.is_file():
        return p
    for root in search_roots:
        q = root / ref_name / "orc_job.tpl"
        if q.is_file():
            return q
    return None


def should_skip_target(name: str, check_dir: Path) -> bool:
    """True if this stoichiometry already has a folder under check_dir (e.g. run35)."""
    return (check_dir / name).is_dir()


def patch_prepare_sh(
    prepare_path: Path,
    mol_xyz_rel: str,
    charge: int,
) -> None:
    text = prepare_path.read_text()
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if re.match(r"^\s*MOL_xyz\s*=", line):
            out.append(f'MOL_xyz={mol_xyz_rel}\n')
        elif re.match(r"^\s*charge_MOL\s*=", line):
            out.append(f"charge_MOL={charge}\n")
        else:
            out.append(line)
    prepare_path.write_text("".join(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run35-root",
        type=Path,
        default=_QM_MINIMIZE_ROOT / "run35",
        help="Directory containing 1Zn_* subfolders with path.xyz (default: qm_minimize/run35)",
    )
    ap.add_argument(
        "--check-dir",
        type=Path,
        default=_QM_MINIMIZE_ROOT / "run35",
        help="Skip if this dir already has the system subdir (default: run35)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_QM_MINIMIZE_ROOT / "run35_new",
        help="Output parent directory (default: qm_minimize/run35_new)",
    )
    ap.add_argument(
        "--min-last-convergence",
        type=Path,
        default=_QM_MINIMIZE_ROOT / "run35" / "min_last_convergence.tsv",
        help="ORCA min summary TSV (default: run35/min_last_convergence.tsv)",
    )
    ap.add_argument(
        "--totals",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="Total ligand counts (MIm+MImH+MeOH), default 4 5 6",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing directories under --out-root (does not rebuild if --check-dir already has the system)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    ap.add_argument(
        "--no-dftb-screen",
        action="store_true",
        help="Do not run DFTB+; use only the first MIH/MIM site (legacy behavior)",
    )
    ap.add_argument(
        "--dftb-template",
        type=Path,
        default=_DEFAULT_DFTB_TEMPLATE,
        help="DFTB+ input template (Geometry/Hamiltonian); default: sp_dftb_in.hsd next to this script",
    )
    ap.add_argument(
        "--dftbplus-cmd",
        default="dftb+",
        help="DFTB+ executable (default: dftb+ on PATH)",
    )
    ap.add_argument(
        "--only-systems",
        nargs="*",
        metavar="NAME",
        help="If set, only process these system directory names (e.g. 1Zn_0MIm_4MImH_1MeOH)",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Do not print the detailed reference-selection log (edges, roots, sort_key); summary lines remain",
    )
    args = ap.parse_args()

    tsv = read_min_last_convergence(args.min_last_convergence)
    targets = iter_targets(tuple(args.totals))
    if args.only_systems:
        want = set(args.only_systems)
        targets = [t for t in targets if formula_name(*t) in want]
    created: list[str] = []
    primary_root = args.run35_root.resolve()
    out_root = args.out_root.resolve()

    pending: list[tuple[int, int, int]] = []
    for a, b, c in targets:
        name = formula_name(a, b, c)
        if should_skip_target(name, args.check_dir):
            continue
        if (out_root / name).is_dir() and not args.force:
            continue
        pending.append((a, b, c))

    initial_n = len(pending)

    def skip_msg_no_ref(a: int, b: int, c: int) -> None:
        name = formula_name(a, b, c)
        print(
            f"SKIP {name}: no reference geometry in --run35-root / --out-root "
            f"(deprot {formula_name(a-1,b+1,c) if a>=1 else '—'} or "
            f"prot {formula_name(a+1,b-1,c) if b>=1 else '—'})",
            file=sys.stderr,
        )

    if args.dry_run:
        roots_dry = [primary_root]
        for a, b, c in pending:
            name = formula_name(a, b, c)
            if not args.quiet:
                for line in reference_selection_report(
                    (a, b, c),
                    roots_dry,
                    primary_root,
                    None,
                    tsv,
                    wave=None,
                ):
                    print(line, file=sys.stderr)
            picked = pick_reference((a, b, c), roots_dry, primary_root, tsv)
            if picked is None:
                skip_msg_no_ref(a, b, c)
                continue
            ref_name, mode, geom_path = picked
            print(
                format_build_summary_line(
                    name, ref_name, mode, geom_path, 0, dry_run=True
                ),
                file=sys.stderr,
            )
            last_block, cycid = read_reference_geometry_block(geom_path)
            fd, tmp_last = tempfile.mkstemp(suffix=".xyz")
            os.close(fd)
            Path(tmp_last).write_text(last_block)
            try:
                u = load_universe_guess_bonds(tmp_last)
                candidates = enumerate_proton_candidates(u, mode)
                if args.no_dftb_screen:
                    candidates = [candidates[0]]
                print(
                    f"Would create {out_root / name} from {ref_name} ({geom_path.name}) "
                    f"mode={mode} cycid={cycid} ({len(candidates)} proton site(s))"
                )
                for lab, _, _ in candidates:
                    print(f"    candidate {lab}")
                created.append(str((out_root / name).resolve()))
            finally:
                os.unlink(tmp_last)
    else:
        search_roots = [primary_root, out_root]
        max_waves = max(initial_n + 4, 24)
        wave = 0
        while pending and wave < max_waves:
            wave += 1
            n_built = 0
            next_pending: list[tuple[int, int, int]] = []
            for a, b, c in pending:
                name = formula_name(a, b, c)
                out_dir = out_root / name
                if out_dir.is_dir() and not args.force:
                    continue
                if not args.quiet:
                    for line in reference_selection_report(
                        (a, b, c),
                        search_roots,
                        primary_root,
                        out_root,
                        tsv,
                        wave=wave,
                    ):
                        print(line, file=sys.stderr)
                picked = pick_reference((a, b, c), search_roots, primary_root, tsv)
                if picked is None:
                    next_pending.append((a, b, c))
                    continue
                ref_name, mode, geom_path = picked
                print(
                    format_build_summary_line(
                        name, ref_name, mode, geom_path, wave, dry_run=False
                    ),
                    file=sys.stderr,
                )
                last_block, cycid = read_reference_geometry_block(geom_path)

                fd, tmp_last = tempfile.mkstemp(suffix=".xyz")
                os.close(fd)
                Path(tmp_last).write_text(last_block)
                try:
                    u = load_universe_guess_bonds(tmp_last)
                    candidates = enumerate_proton_candidates(u, mode)
                    if args.no_dftb_screen:
                        candidates = [candidates[0]]
                    charge = charge_mol_for_stoichiometry(a)
                    screen_root = out_dir / "dftb_screening"

                    out_dir.mkdir(parents=True, exist_ok=True)
                    ref_xyz_name = f"{ref_name}_cyc{cycid}.xyz"
                    if geom_path.name != "path.xyz":
                        ref_xyz_name = f"{ref_name}_from_{geom_path.stem}.xyz"
                    manual_name = f"manually_created_{name}.xyz"
                    mol_rel = f"./{manual_name}"

                    (out_dir / ref_xyz_name).write_text(last_block)

                    if args.no_dftb_screen or len(candidates) == 1:
                        best_label = candidates[0][0]
                        best_e = None
                        summary = [(best_label, None)]
                    else:
                        if not args.dftb_template.is_file():
                            print(
                                f"ERROR {name}: DFTB+ template not found: {args.dftb_template}",
                                file=sys.stderr,
                            )
                            shutil.rmtree(out_dir)
                            next_pending.append((a, b, c))
                            continue
                        best_label, best_e, summary = screen_candidates_dftb(
                            candidates,
                            template_path=args.dftb_template,
                            charge=charge,
                            out_screen_root=screen_root,
                            dftbplus_cmd=args.dftbplus_cmd,
                        )

                    lab_map = {c[0]: c for c in candidates}
                    best_c = lab_map[best_label]
                    modified_xyz = candidate_to_xyz_string(
                        best_c[0],
                        best_c[1],
                        best_c[2],
                        title=f"manually_created_{name}",
                    )
                    (out_dir / manual_name).write_text(modified_xyz)

                    sum_path = out_dir / "dftb_screening_summary.tsv"
                    with sum_path.open("w", newline="") as sf:
                        w = csv.writer(sf, delimiter="\t")
                        w.writerow(["candidate", "energy_Hartree", "selected"])
                        for lab, e in summary:
                            w.writerow(
                                [
                                    lab,
                                    f"{e:.10f}" if e is not None else "",
                                    "yes" if lab == best_label else "no",
                                ]
                            )
                    sel = out_dir / "dftb_selected.txt"
                    sel.write_text(
                        f"selected_candidate={best_label}\n"
                        f"energy_Hartree={best_e}\n"
                        f"reference={ref_name}\n"
                        f"reference_geometry={geom_path}\n"
                        f"mode={mode}\n"
                        f"cycid={cycid}\n"
                        f"iterative_wave={wave}\n"
                    )

                    src_prepare = resolve_prepare_sh(ref_name, geom_path, search_roots)
                    if src_prepare is None or not src_prepare.is_file():
                        print(
                            f"ERROR {name}: missing prepare.sh for reference {ref_name}",
                            file=sys.stderr,
                        )
                        shutil.rmtree(out_dir)
                        next_pending.append((a, b, c))
                        continue
                    src_tpl = resolve_orc_job_tpl(ref_name, geom_path, search_roots)
                    if src_tpl is None or not src_tpl.is_file():
                        print(
                            f"ERROR {name}: missing orc_job.tpl for reference {ref_name}",
                            file=sys.stderr,
                        )
                        shutil.rmtree(out_dir)
                        next_pending.append((a, b, c))
                        continue
                    shutil.copy2(src_prepare, out_dir / "prepare.sh")
                    patch_prepare_sh(out_dir / "prepare.sh", mol_rel, charge)
                    shutil.copy2(src_tpl, out_dir / "orc_job.tpl")
                    created.append(str(out_dir.resolve()))
                    n_built += 1
                finally:
                    os.unlink(tmp_last)

            pending = next_pending
            if n_built > 0:
                print(
                    f"create_run35_new_from_proton_transfer: wave {wave} "
                    f"built {n_built} (remaining {len(pending)})",
                    file=sys.stderr,
                )
            if n_built == 0:
                break

        for a, b, c in pending:
            skip_msg_no_ref(a, b, c)

    label = "Would create (dry-run):" if args.dry_run else "Created directories:"
    print(label)
    if created:
        for p in created:
            print(f"  {p}")
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
