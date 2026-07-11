#!/usr/bin/env python3
"""Scan ``save_dir/result_*.xyz`` for short Zn-O distances (NO3 / MeOH).

Targets Zn / NO3- / MeOH complexes (see zn_no3_meoh_msmiles_enumeration.tsv).
Every numeric ``id`` processed must have a parsable ``msmiles`` in the TSV; otherwise the script
exits with an error (no fallback screening without NO3/MeOH topology).

For each structure id (directory under gen_struc_dftbplus), this script:
- reads each relaxed structure from ``save_dir/result_N.xyz`` (coordinates + ``E=...`` on the
  comment line)
- counts how many conformers pass Zn-O screening (see below)

Zn-O **screening** for NO3/MeOH ligands uses ``metallogen.log`` lines
``Metal-ligand distances (Angstrom), M = Zn[0]`` (same block as the iter_struc driver): O atom
indices in ``Zn-O[i]:`` match the xyz. Conformer id ``N`` in ``save_dir/result_N.xyz`` matches
``Relaxing conformer N/...`` / that relaxation's M-L block.

Per msmiles topology (separate cutoffs ``ZN_O_SUSPECT_NO3`` and ``ZN_O_SUSPECT_MeOH``):
- Each NO3-: among oxygens **listed** as ``Zn-O[…]`` in that block,
  ``min(Zn-O) < ZN_O_SUSPECT_NO3``. If a nitrate oxygen is **missing** from the block but
  ``Zn-O`` from **xyz** is still ``< ZN_O_SUSPECT_NO3``, an ``[INFO]`` line is logged and that O is
  treated as an extra coordinated ``Zn-O`` (distance from xyz) for all NO3 screening and NO3
  histograms.
- MeOH: each alcohol oxygen must appear as ``Zn-O[i]:`` in the block with distance
  ``< ZN_O_SUSPECT_MeOH``.

``ana.log`` tables for the chosen lowest-E passed conformer include two distance columns from
the metallogen M-L block (and NO3 effective map): ``MeOH_max_Zn-O_ml_A`` (max over MeOH oxygens)
and ``NO3_max_of_mins_Zn-O_ml_A`` (max over nitrates of min coordinated Zn-O per nitrate).
Distribution plots (only for geometry/topology-passed relaxed
conformers with msmiles) are written as ``passed_hist_MeOH_maxZnO_ml.png`` (max of M-L log Zn-O
over MeOH oxygens; asserts each MeOH O is listed in the log when any MeOH is present) and
``passed_hist_NO3_maxMinZnO_ml.png`` (per nitrate: min Zn-O in the
effective NO3 map—log plus xyz backfill; then max over nitrates).
DFTB+ logs use ``work_dir/final_relax_{N-1}.dftbplus.log`` (MetalloGen).

Additionally, for each id it picks the lowest-energy stable conformer and writes its xyz block
from `save_dir/result_N.xyz` into `stable_lowE_per_id.xyz` (and aggregates per formula into
`stable_lowE_per_formula.xyz`). Energies come from the ``E=`` field in each file's comment line.

When choosing conformers, the script also checks `work_dir/final_relax_{N-1}.dftbplus.log`. If
that log indicates a failed/non-converged relaxation, the conformer is skipped from geometry/
topology screening and from the final selection.

Convergence rule used here: if `final_relax_*.dftbplus.log` exists, it **must** contain
"Geometry converged" to be considered converged. "ERROR STOP" always indicates failure.

Then it aggregates by formula_id using the enumeration TSV (default: zn_no3_meoh_msmiles_enumeration.tsv).

Implementation note: shared helpers are imported from
``react_benchmark/iter_struc/scripts/metallogen_check_common.py`` (same module used by the iter_struc
driver).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend for batch runs
import matplotlib.pyplot as plt

# Shared helpers (same module as iter_struc/scripts/check_metallogen_short_bonds.py imports).
_MCC_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "iter_struc"
    / "scripts"
    / "metallogen_check_common.py"
)
_spec = importlib.util.spec_from_file_location("_metallogen_check_common", _MCC_PATH)
_mcc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mcc)

dftbplus_relax_failed = _mcc.dftbplus_relax_failed
load_enumeration = _mcc.load_enumeration
expected_topology_from_msmiles = _mcc.expected_topology_from_msmiles
validate_geometry_and_topology = _mcc.validate_geometry_and_topology
extract_single_xyz_atoms = _mcc.extract_single_xyz_atoms
extract_single_xyz_with_energy = _mcc.extract_single_xyz_with_energy
_first_zn_index = _mcc._first_zn_index
RATIO_CRITERIA = _mcc.RATIO_CRITERIA
ATOM_D_CRITERIA = _mcc.ATOM_D_CRITERIA
BOND_CRITERIA = _mcc.BOND_CRITERIA
metal_ligand_zn_o_n_by_atom_index_for_conformer_k = (
    _mcc.metal_ligand_zn_o_n_by_atom_index_for_conformer_k
)

# NO3 / MeOH-specific thresholds (iter_struc script uses Zn-N + single Zn-O).
ZN_O_SUSPECT_NO3 = 2.5  # Å, min of three nitrate O-Zn distances vs this
ZN_O_SUSPECT_MeOH = 2.5  # Å, each MeOH alcohol O-Zn vs this

_ITER_STRUC = Path(__file__).resolve().parent.parent
_RESULT_XYZ_NAME_RE = re.compile(r"^result_(\d+)\.xyz$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tsv",
        type=Path,
        default=_ITER_STRUC / "zn_no3_meoh_msmiles_enumeration.tsv",
        help="Enumeration TSV with id / formula_id / formula / msmiles",
    )
    p.add_argument(
        "gen_dir",
        type=Path,
        help="Directory containing numeric id subdirs with save_dir/result_*.xyz",
    )
    p.add_argument(
        "--stable_out",
        type=Path,
        default=None,
        help="(Deprecated) Alias for --stable_out_per_id.",
    )
    p.add_argument(
        "--stable_out_per_id",
        type=Path,
        default=None,
        help="Collect lowest-energy stable conformers per id into this xyz file. "
        "Default: gen_dir/stable_lowE_per_id.xyz",
    )
    p.add_argument(
        "--stable_out_per_formula",
        type=Path,
        default=None,
        help="Collect lowest-energy stable conformer per formula_id into this xyz file. "
        "Default: gen_dir/stable_lowE_per_formula.xyz",
    )
    p.add_argument(
        "--cluster_id_offset",
        type=int,
        default=-1,
        help='Mapping from metallogen.log conformer index k ("Relaxing conformer k/...") to xyz cluster_id. Default: cluster_id = k-1.',
    )
    p.add_argument(
        "--zn_o_thresh_no3",
        type=float,
        default=ZN_O_SUSPECT_NO3,
        help="Threshold (Å): min Zn-O over effective coordinated NO3 oxygens (log lines plus "
        "any nitrate O missing from log but with xyz Zn-O<this, counted as coordinated) must be "
        "<this",
    )
    p.add_argument(
        "--zn_o_thresh_meoh",
        type=float,
        default=ZN_O_SUSPECT_MeOH,
        help="Threshold (Å): each MeOH alcohol O must appear in metallogen.log with Zn-O<this",
    )
    p.add_argument(
        "--zn_o_thresh",
        type=float,
        default=None,
        help="If set, overrides both --zn_o_thresh_no3 and --zn_o_thresh_meoh (Å).",
    )
    p.add_argument(
        "--ratio_criteria",
        type=float,
        default=RATIO_CRITERIA,
        help="Geometry collapse criterion on d/(r_i+r_j)",
    )
    p.add_argument(
        "--atom_d_criteria",
        type=float,
        default=ATOM_D_CRITERIA,
        help="Absolute minimum atom-atom distance criterion (Å)",
    )
    p.add_argument(
        "--bond_criteria",
        type=float,
        default=BOND_CRITERIA,
        help="Bond criterion on d/(r_i+r_j) for topology check",
    )
    return p.parse_args()


def o_ligand_groups_from_no3_meoh_topology(
    elements: list[str], adj: np.ndarray
) -> tuple[list[list[int]], list[int]]:
    """Partition ligand oxygens into NO3 vs MeOH using the msmiles reference graph.

    Each nitrate is N bonded to exactly three oxygens; those three indices form one group.
    Zn-O screening uses the minimum of the three Zn-O distances computed from full coordinates.

    Any oxygen not belonging to a nitrate's three O atoms is treated as MeOH (or other).
    """
    n = len(elements)
    nitrate_o_set: set[int] = set()
    nitrate_groups: list[list[int]] = []
    for i in range(n):
        if elements[i] != "N":
            continue
        neigh = [j for j in range(n) if i != j and int(adj[i, j]) != 0]
        o_neigh = [j for j in neigh if elements[j] == "O"]
        if len(o_neigh) != 3:
            continue
        nitrate_o_set.update(o_neigh)
        nitrate_groups.append(sorted(o_neigh))
    meoh_o: list[int] = []
    for i in range(n):
        if elements[i] == "O" and i not in nitrate_o_set:
            meoh_o.append(i)
    return nitrate_groups, meoh_o


def max_zn_o_from_atoms(
    atoms: list[tuple[str, float, float, float]],
) -> float | None:
    """Longest Zn-O (Å) from the first Zn to any oxygen in ``atoms``."""
    elems = [a[0] for a in atoms]
    coords = np.asarray([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    zn_idx = _first_zn_index(elems)
    if zn_idx is None:
        return None
    zn = coords[zn_idx]
    max_o: float | None = None
    for i, e in enumerate(elems):
        if i == zn_idx or e != "O":
            continue
        d = float(np.linalg.norm(coords[i] - zn))
        max_o = d if max_o is None else max(max_o, d)
    return max_o


def augment_zn_o_ml_with_short_nitrate_o_from_xyz(
    zn_o_ml: dict[int, float],
    nitrate_o_groups: list[list[int]],
    atoms: list[tuple[str, float, float, float]],
    zn_o_thresh_no3: float,
    *,
    oid: int,
    conformer_k: int,
    xyz_path: Path,
    warn: Callable[[str], None] | None = None,
) -> dict[int, float]:
    """Copy of M-L Zn-O distances, plus nitrate oxygens missing from the log but short in xyz.

    If a nitrate O is absent from ``zn_o_ml`` yet ``Zn-O`` from coordinates is
    ``< zn_o_thresh_no3``, it is treated as coordinated: we insert ``atom_index -> d_xyz`` and
    emit ``warn`` if given so downstream NO3 logic matches an extended coordination block.
    """
    out = dict(zn_o_ml)
    elems = [a[0] for a in atoms]
    coords = np.asarray([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    zn_idx = _first_zn_index(elems)
    if zn_idx is None:
        return out
    zn = coords[zn_idx]

    def _w(msg: str) -> None:
        if warn is not None:
            warn(msg)

    for group in nitrate_o_groups:
        for i in group:
            if i in out:
                continue
            if i >= len(coords):
                continue
            d_xyz = float(np.linalg.norm(coords[i] - zn))
            if d_xyz < zn_o_thresh_no3:
                out[i] = d_xyz
                _w(
                    f"[INFO] id={oid} conformer_k={conformer_k} xyz={xyz_path}: "
                    f"nitrate O atom_index={i} missing from metallogen Zn-O lines but "
                    f"Zn-O={d_xyz:.4f} Å < NO3 threshold {zn_o_thresh_no3:.4f} Å; "
                    f"treating as coordinated (xyz distance) for NO3 screening and histograms."
                )
    return out


def conformer_stable_no3_meoh_metallogen_and_xyz(
    atoms: list[tuple[str, float, float, float]],
    nitrate_o_groups: list[list[int]],
    meoh_o_indices: list[int],
    zn_o_ml: dict[int, float],
    zn_o_thresh_no3: float,
    zn_o_thresh_meoh: float,
    *,
    oid: int,
    conformer_k: int,
    xyz_path: Path,
    warn: Callable[[str], None] | None = None,
) -> tuple[bool, float | None]:
    """NO3/MeOH stability from metallogen M-L Zn-O lines, with xyz backfill for short NO3 O.

    Nitrate oxygens listed only in xyz (short Zn-O) are merged into the effective Zn-O map; see
    ``augment_zn_o_ml_with_short_nitrate_o_from_xyz``. MeOH still requires ``Zn-O[i]:`` in the
    log.
    """
    elems = [a[0] for a in atoms]
    coords = np.asarray([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    zn_idx = _first_zn_index(elems)
    max_o = max_zn_o_from_atoms(atoms)

    if zn_idx is None:
        return False, max_o

    zn_eff = augment_zn_o_ml_with_short_nitrate_o_from_xyz(
        zn_o_ml,
        nitrate_o_groups,
        atoms,
        zn_o_thresh_no3,
        oid=oid,
        conformer_k=conformer_k,
        xyz_path=xyz_path,
        warn=warn,
    )

    for group in nitrate_o_groups:
        coordinated = [i for i in group if i in zn_eff]
        if not coordinated:
            return False, max_o
        if min(zn_eff[i] for i in coordinated) >= zn_o_thresh_no3:
            return False, max_o

    for i in meoh_o_indices:
        if i >= len(coords):
            return False, max_o
        if i not in zn_o_ml or zn_o_ml[i] >= zn_o_thresh_meoh:
            return False, max_o

    return True, max_o


def _o_dists_ligand_from_atoms(
    atoms: list[tuple[str, float, float, float]],
    nitrate_groups: list[list[int]],
    meoh_o_indices: list[int],
) -> dict[int, float]:
    """Zn-O distances (Å) for ligand oxygens (all NO3 + MeOH indices) from coordinates."""
    elems = [a[0] for a in atoms]
    coords = np.asarray([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    zn_idx = _first_zn_index(elems)
    out: dict[int, float] = {}
    if zn_idx is None:
        return out
    zn = coords[zn_idx]
    seen: set[int] = set()
    for group in nitrate_groups:
        for i in group:
            if i not in seen and i < len(coords):
                seen.add(i)
                out[i] = float(np.linalg.norm(coords[i] - zn))
    for i in meoh_o_indices:
        if i not in seen and i < len(coords):
            out[i] = float(np.linalg.norm(coords[i] - zn))
    return out


def iter_sorted_result_xyz(save_dir: Path) -> list[tuple[int, Path]]:
    """``result_N.xyz`` paths sorted by integer ``N`` (MetalloGen success index)."""
    if not save_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in save_dir.iterdir():
        if not p.is_file():
            continue
        m = _RESULT_XYZ_NAME_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def parse_save_dir_result_xyz_with_max_dists(
    save_dir: Path,
    log_path: Path,
    zn_o_thresh_no3: float,
    zn_o_thresh_meoh: float,
    nitrate_o_groups: list[list[int]],
    meoh_o_indices: list[int],
    *,
    oid: int = -1,
    warn: Callable[[str], None] | None = None,
) -> tuple[
    int,
    int,
    list[int],
    list[int],
    dict[int, float | None],
    dict[int, dict[int, float]],
]:
    """Zn-O screening: metallogen.log M-L Zn-O for NO3/MeOH; short NO3 O in xyz backfill the map.

    Requires a parseable xyz with ``E=`` in the comment line (same format MetalloGen writes).
    ``log_path`` must be the sibling ``metallogen.log`` for that structure id (conformer ``N``
    matches the log block). ``nitrate_o_groups`` / ``meoh_o_indices`` come from msmiles topology
    (caller must ensure they exist).
    Optional ``warn`` receives messages when nitrate oxygens are promoted from xyz (see augment).
    """
    total_conf = 0
    n_conf_stable = 0
    stable_conformer_nums: list[int] = []
    relaxed_conformer_nums: list[int] = []
    max_dists_by_k: dict[int, float | None] = {}
    o_dists_by_k: dict[int, dict[int, float]] = {}

    for n, path in iter_sorted_result_xyz(save_dir):
        try:
            _e, _bl = extract_single_xyz_with_energy(path)
            atoms, _ = extract_single_xyz_atoms(path)
        except (ValueError, OSError):
            continue

        total_conf += 1
        relaxed_conformer_nums.append(n)
        max_o = max_zn_o_from_atoms(atoms)
        max_dists_by_k[n] = max_o
        o_dists_by_k[n] = _o_dists_ligand_from_atoms(
            atoms, nitrate_o_groups, meoh_o_indices
        )
        zn_o_ml, _zn_n_ml = metal_ligand_zn_o_n_by_atom_index_for_conformer_k(
            log_path, n
        )
        ok, _ = conformer_stable_no3_meoh_metallogen_and_xyz(
            atoms,
            nitrate_o_groups,
            meoh_o_indices,
            zn_o_ml,
            zn_o_thresh_no3,
            zn_o_thresh_meoh,
            oid=oid,
            conformer_k=n,
            xyz_path=path,
            warn=warn,
        )
        if ok:
            n_conf_stable += 1
            stable_conformer_nums.append(n)

    return (
        total_conf,
        n_conf_stable,
        stable_conformer_nums,
        relaxed_conformer_nums,
        max_dists_by_k,
        o_dists_by_k,
    )


def hist_metric_meoh_max_zn_o_ml(
    zn_o_ml: dict[int, float], meoh_o_indices: list[int]
) -> float | None:
    """Max Zn-O (Å) over MeOH alcohol oxygens in the M-L log.

    If there is at least one MeOH oxygen, every index must appear in ``zn_o_ml`` (asserted).
    If there are no MeOH oxygens in the topology, returns None.
    """
    if not meoh_o_indices:
        return None
    missing = [i for i in meoh_o_indices if i not in zn_o_ml]
    assert not missing, (
        "hist_metric_meoh_max_zn_o_ml: every MeOH alcohol O must be in metallogen Zn-O lines; "
        f"missing atom_index={missing}, meoh_o_indices={meoh_o_indices}"
    )
    return max(zn_o_ml[i] for i in meoh_o_indices)


def hist_metric_no3_max_of_mins_zn_o_ml(
    zn_o_ml: dict[int, float], nitrate_o_groups: list[list[int]]
) -> float | None:
    """Per NO3: min Zn-O among entries for that group's oxygens; then max over nitrate groups.

    ``zn_o_ml`` is typically the log map augmented with short nitrate O from xyz (see augment).
    """
    if not nitrate_o_groups:
        return None
    per_group: list[float] = []
    for group in nitrate_o_groups:
        ds = [zn_o_ml[i] for i in group if i in zn_o_ml]
        if not ds:
            return None
        per_group.append(min(ds))
    return max(per_group)


def ml_meoh_max_and_no3_maxmin_for_conformer(
    subdir: Path,
    conformer_k: int,
    oid: int,
    nitrate_o_groups: list[list[int]],
    meoh_o_indices: list[int],
    zn_o_no3: float,
) -> tuple[float | None, float | None]:
    """M-L metrics for one conformer: MeOH max Zn-O (log); NO3 max(min per NO3) effective map."""
    log_path = subdir / "metallogen.log"
    result_path = subdir / "save_dir" / f"result_{conformer_k}.xyz"
    zn_o_ml, _zn_n = metal_ligand_zn_o_n_by_atom_index_for_conformer_k(log_path, conformer_k)
    meoh_max = hist_metric_meoh_max_zn_o_ml(zn_o_ml, meoh_o_indices)
    atoms, _ = extract_single_xyz_atoms(result_path)
    zn_eff = augment_zn_o_ml_with_short_nitrate_o_from_xyz(
        zn_o_ml,
        nitrate_o_groups,
        atoms,
        zn_o_no3,
        oid=oid,
        conformer_k=conformer_k,
        xyz_path=result_path,
        warn=None,
    )
    no3_maxmin = hist_metric_no3_max_of_mins_zn_o_ml(zn_eff, nitrate_o_groups)
    return meoh_max, no3_maxmin


def _final_relax_allows_use(
    subdir: Path,
    k: int,
    oid: int,
    warned_missing_relax_logs: set[tuple[int, int]],
    emit: Callable[[str], None],
    *,
    fail_reasons: dict[str, int] | None = None,
) -> bool:
    """True if ``final_relax_{k-1}.dftbplus.log`` is absent (warn once), converged, or readable."""
    relax_log_path = subdir / "work_dir" / f"final_relax_{k - 1}.dftbplus.log"
    if relax_log_path.exists():
        try:
            relax_log_text = relax_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            relax_log_text = ""
        if dftbplus_relax_failed(relax_log_text):
            if fail_reasons is not None:
                fail_reasons["final_relax_error"] += 1
            return False
        return True
    warn_key = (oid, k)
    if warn_key not in warned_missing_relax_logs:
        warned_missing_relax_logs.add(warn_key)
        emit(
            f"[WARN] id={oid} k={k}: missing {relax_log_path.name}; "
            "optimization may be unconverged. Here we manually treat this "
            "case as geometry-converged and SCC-successful for downstream screening."
        )
    return True


def _write_passed_hist_zn_o(
    passed_values: list[float],
    zn_o_cutoff: float,
    *,
    title: str,
    xlabel: str,
    out_path: Path,
    emit: Callable[[str], None],
) -> None:
    passed_all_o = np.asarray(passed_values, dtype=float)
    if passed_all_o.size == 0:
        return
    bin_width = 0.1
    tick_step = 0.5
    eps = 1e-6
    passed_all_o_bins = passed_all_o.copy()
    passed_all_o_bins[np.isclose(passed_all_o_bins, zn_o_cutoff)] -= eps
    passed_o_over = passed_all_o[passed_all_o > zn_o_cutoff]
    xmin, xmax = float(np.min(passed_all_o_bins)), float(np.max(passed_all_o_bins))
    if xmin == xmax:
        xmin -= bin_width
        xmax += bin_width
    start_i = int(np.floor(xmin / bin_width))
    end_i = int(np.ceil(xmax / bin_width))
    bin_edges_o = np.arange(start_i * bin_width, (end_i + 1) * bin_width + 1e-9, bin_width)
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=150)
    ax.hist(
        passed_all_o_bins,
        bins=bin_edges_o,
        color="#4C78A8",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.6,
    )
    if passed_o_over.size:
        ax.hist(
            passed_o_over,
            bins=bin_edges_o,
            color="red",
            alpha=0.65,
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axvline(zn_o_cutoff, color="red", linestyle="--", linewidth=1.2)
    tick_start_i = int(np.floor(xmin / tick_step))
    tick_end_i = int(np.ceil(xmax / tick_step))
    ax.set_xticks([i * tick_step for i in range(tick_start_i, tick_end_i + 1)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    emit(f"[INFO] Wrote histogram to {out_path}")


def main() -> None:
    args = parse_args()
    if args.zn_o_thresh is not None:
        zn_o_no3 = zn_o_meoh = args.zn_o_thresh
    else:
        zn_o_no3 = args.zn_o_thresh_no3
        zn_o_meoh = args.zn_o_thresh_meoh

    id_to_fid, fid_to_formula, id_to_msmiles = load_enumeration(args.tsv)
    runtime_logs: list[str] = []
    # Histograms: M-L log Zn-O for geometry/topology-passed relaxed conformers (all ids).
    passed_all_meoh_max_ml: list[float] = []
    passed_all_no3_maxmin_ml: list[float] = []
    passed_hist_total_conformers: int = 0
    passed_hist_over_meoh: int = 0
    passed_hist_over_no3: int = 0
    warned_missing_relax_logs: set[tuple[int, int]] = set()

    def emit(msg: str) -> None:
        runtime_logs.append(msg)

    def _append_kv(comment: str, key: str, value: str) -> str:
        token = f"{key}={value}"
        if re.search(rf"\\b{re.escape(key)}=", comment):
            return comment
        return f"{comment} {token}"

    # Per-id statistics.
    # id -> {"total_conf": int, "n_conf_stable": int, "n_conf_passed": int, "passed_ids": list[int]}
    id_stats: dict[int, dict[str, object]] = {}
    # (id, best_passed_conformer_id, xyz_block)
    stable_lowE_per_id_blocks: list[tuple[int, int, str]] = []
    # id -> best passed conformer index k (1-based, from metallogen.log "Relaxing conformer k/..."),
    # or None if there is no passed conformer.
    id_best_passed_k: dict[int, int | None] = {}
    # id -> best passed energy corresponding to id_best_passed_k, or None if there is no passed conformer.
    id_best_passed_e: dict[int, float | None] = {}
    # Per passed lowest-E conformer: MeOH max Zn-O from log; NO3 max(min per NO3) effective ML map.
    id_best_meoh_max_zn_o_ml: dict[int, float | None] = {}
    id_best_no3_maxmin_zn_o_ml: dict[int, float | None] = {}
    # formula_id -> (E, oid, k, block_lines)
    best_per_formula: dict[int, tuple[float, int, int, list[str]]] = {}
    fid_best_meoh_max_zn_o_ml: dict[int, float | None] = {}
    fid_best_no3_maxmin_zn_o_ml: dict[int, float | None] = {}

    for subdir in sorted(args.gen_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            oid = int(subdir.name)
        except ValueError:
            continue

        save_dir = subdir / "save_dir"
        msmiles = id_to_msmiles.get(oid)
        if not msmiles:
            print(
                f"[FATAL] id={oid}: missing msmiles in TSV; "
                "NO3/MeOH screening requires msmiles topology.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            expected_elements, expected_adj = expected_topology_from_msmiles(msmiles)
            nitrate_o_groups, meoh_o_indices = o_ligand_groups_from_no3_meoh_topology(
                expected_elements, expected_adj
            )
        except Exception as e:
            print(
                f"[FATAL] id={oid}: failed to build topology from msmiles: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        (
            total_conf,
            n_conf_stable,
            stable_conformer_nums,
            relaxed_conformer_nums,
            _max_dists_by_k,
            _o_dists_by_k,
        ) = parse_save_dir_result_xyz_with_max_dists(
            save_dir,
            subdir / "metallogen.log",
            zn_o_no3,
            zn_o_meoh,
            nitrate_o_groups,
            meoh_o_indices,
            oid=oid,
            warn=emit,
        )
        if total_conf == 0:
            emit(
                f"[WARN] id={oid}: no parseable save_dir/result_*.xyz under {save_dir}; skipped"
            )
            continue
        id_stats[oid] = {
            "total_conf": total_conf,
            "n_conf_stable": n_conf_stable,
            "n_conf_passed": 0,
            "passed_ids": [],
        }
        id_best_passed_k[oid] = None
        id_best_passed_e[oid] = None

        # Histogram base ("passed"): relaxed conformers that also pass
        # (1) final DFTB+ relaxation convergence and (2) geometry/topology screening.
        # Zn-O distance screening is not required to enter the histogram; vertical lines use
        # --zn_o_thresh_meoh / --zn_o_thresh_no3 (red counts = values past that cutoff).
        log_path = subdir / "metallogen.log"
        for k in relaxed_conformer_nums:
            if not _final_relax_allows_use(
                subdir, k, oid, warned_missing_relax_logs, emit
            ):
                continue

            result_path = subdir / "save_dir" / f"result_{k}.xyz"
            if not result_path.exists():
                continue

            try:
                new_atoms, _ = extract_single_xyz_atoms(result_path)
            except ValueError:
                continue

            ok, _reason = validate_geometry_and_topology(
                new_atoms=new_atoms,
                expected_elements=expected_elements,
                expected_adj=expected_adj,
                ratio_criteria=args.ratio_criteria,
                atom_d_criteria=args.atom_d_criteria,
                bond_criteria=args.bond_criteria,
            )
            if not ok:
                continue

            passed_hist_total_conformers += 1
            zn_o_ml, _zn_n_ml = metal_ligand_zn_o_n_by_atom_index_for_conformer_k(
                log_path, k
            )
            meoh_h = hist_metric_meoh_max_zn_o_ml(zn_o_ml, meoh_o_indices)
            if meoh_h is not None:
                passed_all_meoh_max_ml.append(meoh_h)
                if meoh_h > zn_o_meoh:
                    passed_hist_over_meoh += 1
            # Same effective map as parse_save_dir (xyz backfill); warn=None avoids duplicate
            # [INFO] lines already emitted when that conformer was scanned there.
            zn_no3_hist = augment_zn_o_ml_with_short_nitrate_o_from_xyz(
                zn_o_ml,
                nitrate_o_groups,
                new_atoms,
                zn_o_no3,
                oid=oid,
                conformer_k=k,
                xyz_path=result_path,
                warn=None,
            )
            no3_h = hist_metric_no3_max_of_mins_zn_o_ml(zn_no3_hist, nitrate_o_groups)
            if no3_h is not None:
                passed_all_no3_maxmin_ml.append(no3_h)
                if no3_h > zn_o_no3:
                    passed_hist_over_no3 += 1

        # Collect lowest-energy conformer from fully passed (distance + geometry + topology).
        if not stable_conformer_nums:
            emit(
                f"[INFO] id={oid}: no stable conformers (n_conf_stable={n_conf_stable}); skip stable_lowE selection"
            )
            continue

        # Stable conformer id k matches save_dir/result_{k}.xyz (MetalloGen success ordinal).
        best: tuple[float, int, list[str]] | None = None  # (E, k, block_lines)
        missing_ks: list[int] = []
        fail_reasons: dict[str, int] = defaultdict(int)
        passed_conformer_ids: list[int] = []

        for k in stable_conformer_nums:
            if not _final_relax_allows_use(
                subdir, k, oid, warned_missing_relax_logs, emit, fail_reasons=fail_reasons
            ):
                continue

            result_path = subdir / "save_dir" / f"result_{k}.xyz"
            if not result_path.exists():
                missing_ks.append(k)
                continue
            try:
                energy, block_lines = extract_single_xyz_with_energy(result_path)
                new_atoms, _ = extract_single_xyz_atoms(result_path)
            except ValueError as e:
                emit(f"[WARN] id={oid}: failed to parse {result_path}: {e}")
                continue

            ok, reason = validate_geometry_and_topology(
                new_atoms=new_atoms,
                expected_elements=expected_elements,
                expected_adj=expected_adj,
                ratio_criteria=args.ratio_criteria,
                atom_d_criteria=args.atom_d_criteria,
                bond_criteria=args.bond_criteria,
            )
            if not ok:
                fail_reasons[reason] += 1
                continue

            passed_conformer_ids.append(k)
            # For stable_lowE selection we only need max distances of the chosen best conformer.
            if best is None or energy < best[0] or (energy == best[0] and k < best[1]):
                best = (energy, k, block_lines)

        id_stats[oid]["n_conf_passed"] = len(passed_conformer_ids)
        id_stats[oid]["passed_ids"] = passed_conformer_ids

        if missing_ks:
            emit(
                f"[WARN] id={oid}: missing result xyz for stable ks: "
                f"{missing_ks[:20]}{' ...' if len(missing_ks) > 20 else ''}"
            )
        if fail_reasons:
            emit(
                f"[INFO] id={oid}: failed geometry/topology counts: "
                + ", ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items()))
            )

        if best is None:
            continue

        best_e, best_k, best_lines = best
        id_best_passed_k[oid] = best_k
        id_best_passed_e[oid] = best_e
        meoh_ml, no3_mm = ml_meoh_max_and_no3_maxmin_for_conformer(
            subdir,
            best_k,
            oid,
            nitrate_o_groups,
            meoh_o_indices,
            zn_o_no3,
        )
        id_best_meoh_max_zn_o_ml[oid] = meoh_ml
        id_best_no3_maxmin_zn_o_ml[oid] = no3_mm

        fid = id_to_fid.get(oid)
        formula = fid_to_formula.get(fid, "") if fid is not None else ""
        def _fmt_d(d: float | None) -> str:
            return f"{d:.6f}" if d is not None else "none"

        emit(
            f"[INFO] id={oid} formula_id={fid if fid is not None else 'none'} "
            f"lowestE_conformer k={best_k} E={best_e:.12f} "
            f"MeOH_max_Zn-O_ml={_fmt_d(meoh_ml)} Å "
            f"NO3_max_of_mins_Zn-O_ml={_fmt_d(no3_mm)} Å"
        )
        if len(best_lines) >= 2:
            # result_{k}.xyz comment line typically doesn't include k,
            # so we attach it for traceability.
            best_lines[1] = _append_kv(best_lines[1], "id", str(oid))
            best_lines[1] = _append_kv(best_lines[1], "conformer_id", str(best_k))
            if fid is not None:
                best_lines[1] = _append_kv(best_lines[1], "formula_id", str(fid))
                if formula:
                    best_lines[1] = _append_kv(best_lines[1], "formula", formula)
        block_text = "\n".join(best_lines)
        stable_lowE_per_id_blocks.append((oid, best_k, block_text))

        # Track the lowest-energy stable conformer per formula_id.
        fid = id_to_fid.get(oid)
        if fid is not None:
            cur = best_per_formula.get(fid)
            # Tie-breakers: lower energy, then smaller id, then smaller k.
            if cur is None or best_e < cur[0] or (
                best_e == cur[0] and (oid < cur[1] or (oid == cur[1] and best_k < cur[2]))
            ):
                best_per_formula[fid] = (best_e, oid, best_k, best_lines)
                fid_best_meoh_max_zn_o_ml[fid] = meoh_ml
                fid_best_no3_maxmin_zn_o_ml[fid] = no3_mm

    if passed_all_meoh_max_ml:
        _write_passed_hist_zn_o(
            passed_all_meoh_max_ml,
            zn_o_meoh,
            title=(
                "Passed conformers: MeOH max Zn-O (M-L log); "
                "red = over MeOH suspect cutoff"
            ),
            xlabel="max Zn-O over MeOH oxygens in log (Angstrom)",
            out_path=args.gen_dir / "passed_hist_MeOH_maxZnO_ml.png",
            emit=emit,
        )
    if passed_all_no3_maxmin_ml:
        _write_passed_hist_zn_o(
            passed_all_no3_maxmin_ml,
            zn_o_no3,
            title=(
                "Passed conformers: NO3 max(min Zn-O per NO3 in M-L log); "
                "red = over NO3 suspect cutoff"
            ),
            xlabel="max over nitrates of (min Zn-O among listed O) (Angstrom)",
            out_path=args.gen_dir / "passed_hist_NO3_maxMinZnO_ml.png",
            emit=emit,
        )
    if passed_hist_total_conformers:
        emit(
            f"[INFO] Passed conformers (hist) geometry-passed total={passed_hist_total_conformers}; "
            f"MeOH-hist n={len(passed_all_meoh_max_ml)} "
            f"Zn-O>{zn_o_meoh:.3f} => {passed_hist_over_meoh}; "
            f"NO3-hist n={len(passed_all_no3_maxmin_ml)} "
            f"Zn-O>{zn_o_no3:.3f} => {passed_hist_over_no3}"
        )

    # Aggregate by formula_id
    fid_totals: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total_conf": 0, "n_conf_stable": 0, "n_conf_passed": 0}
    )

    for oid, stats in id_stats.items():
        if oid not in id_to_fid:
            continue
        fid = id_to_fid[oid]
        agg = fid_totals[fid]
        agg["total_conf"] += int(stats["total_conf"])
        agg["n_conf_stable"] += int(stats["n_conf_stable"])
        agg["n_conf_passed"] += int(stats["n_conf_passed"])
    # Prepare output text
    lines: list[str] = []
    lines.append(f"TSV: {args.tsv}")
    lines.append(f"gen_dir: {args.gen_dir}")
    lines.append(
        f"Thresholds: NO3 min(Zn-O) over effective coordinated O (log + xyz backfill if short) "
        f"<{zn_o_no3:.3f} Å; MeOH each log Zn-O<{zn_o_meoh:.3f} Å"
    )
    if args.zn_o_thresh is not None:
        lines.append(f"  (--zn_o_thresh {args.zn_o_thresh:.3f} Å applied to both NO3 and MeOH)")
    lines.append(
        f"Geometry/Topology criteria: ratio >= {args.ratio_criteria:.3f}, "
        f"min_dist >= {args.atom_d_criteria:.3f} Å, bond_ratio < {args.bond_criteria:.3f}"
    )
    lines.append("Topology reference: reconstructed from msmiles (TSV column `msmiles`).")
    lines.append("Definitions:")
    lines.append(
        "  - n_conf_stable: NO3/MeOH use metallogen.log M-L Zn-O lines (atom index = xyz). "
        "Each NO3-: min Zn-O over coordinated oxygens must be < NO3 threshold; oxygens missing "
        "from the log but short in xyz are promoted ([INFO] logged) and counted as coordinated. "
        "Each MeOH O must appear in the log with Zn-O < MeOH threshold."
    )
    lines.append(
        "  - n_conf_passed: subset of n_conf_stable that also passes geometry/topology checks, "
        "and whose corresponding final_relax_{k-1}.dftbplus.log does not indicate failure "
        '(requires "Geometry converged" and must not contain ERROR STOP).'
    )
    lines.append(
        "  - MeOH_max_Zn-O_ml_A: for passed lowest-E conformer, max of metallogen M-L Zn-O over "
        "MeOH alcohol oxygens (assert all listed in log when any MeOH); none if no MeOH."
    )
    lines.append(
        "  - NO3_max_of_mins_Zn-O_ml_A: same conformer, max over nitrates of min Zn-O among each "
        "nitrate's coordinated oxygens (effective map: log + xyz backfill); none if no NO3."
    )
    lines.append(
        "  - Histograms (geometry-passed relaxed conformers): passed_hist_MeOH_maxZnO_ml.png = "
        "max of metallogen M-L Zn-O over all MeOH oxygens (assert: each MeOH O in log if any MeOH); "
        "passed_hist_NO3_maxMinZnO_ml.png = same NO3 effective distances as n_conf_stable "
        "(log plus short nitrate O from xyz when missing from the log)."
    )
    lines.append("  - set relation: passed_conformer_ids ⊆ stable_conformer_ids.")
    lines.append("")

    header_id = (
        "id\tformula_id\ttotal_conf\tn_conf_stable\tn_conf_passed\tlowest_E\t"
        "MeOH_max_Zn-O_ml_A\tNO3_max_of_mins_Zn-O_ml_A\tpassed_conformer_ids"
    )
    lines.append(header_id)
    lines.append("-" * min(120, len(header_id) + 40))
    for oid in sorted(id_stats.keys()):
        stats = id_stats[oid]
        passed_ids = ",".join(str(k) for k in stats["passed_ids"])  # type: ignore[index]
        lowest_e = id_best_passed_e.get(oid)
        lowest_e_str = f"{lowest_e:.12f}" if lowest_e is not None else "none"
        mm = id_best_meoh_max_zn_o_ml.get(oid)
        nm = id_best_no3_maxmin_zn_o_ml.get(oid)
        mm_str = f"{mm:.6f}" if mm is not None else "none"
        nm_str = f"{nm:.6f}" if nm is not None else "none"
        fid = id_to_fid.get(oid)
        fid_str = str(fid) if fid is not None else "none"
        lines.append(
            f"{oid}\t{fid_str}\t{stats['total_conf']}\t{stats['n_conf_stable']}\t"
            f"{stats['n_conf_passed']}\t{lowest_e_str}\t{mm_str}\t{nm_str}\t{passed_ids}"
        )

    lines.append("")
    header_fid = (
        "formula_id\tformula\ttotal_conf\tn_conf_stable\tn_conf_passed\tlowest_E\t"
        "MeOH_max_Zn-O_ml_lowestE_A\tNO3_max_of_mins_Zn-O_ml_lowestE_A\tpassed_lowE_list_by_id"
    )
    lines.append(header_fid)
    lines.append("-" * min(120, len(header_fid) + 40))

    for fid in sorted(fid_totals.keys()):
        formula = fid_to_formula.get(fid, "")
        agg = fid_totals[fid]
        # For this formula_id, list (id, best_passed_conformer_id) for all processed ids.
        # If an id has no passed conformer: (id,none).
        oids_for_fid = sorted(oid for oid in id_stats.keys() if id_to_fid.get(oid) == fid)
        # Bracket the (id,conformerid) tuple with lowest passed energy among ids that have passed conformers.
        best_oid_for_fid: int | None = None
        best_triplet: tuple[float, int, int] | None = None  # (E, oid, k)
        for oid in oids_for_fid:
            k = id_best_passed_k.get(oid)
            e = id_best_passed_e.get(oid)
            if k is None or e is None:
                continue
            triplet = (e, oid, k)
            if best_triplet is None or triplet < best_triplet:
                best_triplet = triplet
                best_oid_for_fid = oid

        passed_items: list[str] = []
        for oid in oids_for_fid:
            k = id_best_passed_k.get(oid)
            if k is None:
                item = f"({oid},none)"
            else:
                item = f"({oid},{k})"
                if oid == best_oid_for_fid:
                    item = f"[{item}]"
            passed_items.append(item)

        passed_list = ",".join(passed_items)
        lowest_e_fid_str = f"{best_triplet[0]:.12f}" if best_triplet is not None else "none"
        f_mm = fid_best_meoh_max_zn_o_ml.get(fid)
        f_nm = fid_best_no3_maxmin_zn_o_ml.get(fid)
        f_mm_str = f"{f_mm:.6f}" if f_mm is not None else "none"
        f_nm_str = f"{f_nm:.6f}" if f_nm is not None else "none"
        lines.append(
            f"{fid}\t{formula}\t{agg['total_conf']}\t{agg['n_conf_stable']}\t"
            f"{agg['n_conf_passed']}\t{lowest_e_fid_str}\t{f_mm_str}\t{f_nm_str}\t{passed_list}"
        )

    stable_out_per_id_path = (
        args.stable_out_per_id
        if args.stable_out_per_id is not None
        else (args.stable_out if args.stable_out is not None else args.gen_dir / "stable_lowE_per_id.xyz")
    )
    stable_out_per_formula_path = (
        args.stable_out_per_formula
        if args.stable_out_per_formula is not None
        else args.gen_dir / "stable_lowE_per_formula.xyz"
    )

    stable_lowE_per_id_blocks.sort(key=lambda t: (t[0], t[1]))  # sort by (id, conformer_id)
    stable_out_per_id_path.write_text(
        "\n".join(block for _, _, block in stable_lowE_per_id_blocks)
        + ("\n" if stable_lowE_per_id_blocks else ""),
        encoding="utf-8",
    )
    if stable_lowE_per_id_blocks:
        emit(
            f"Wrote stable_lowE_per_id.xyz to {stable_out_per_id_path} "
            f"(n={len(stable_lowE_per_id_blocks)})"
        )
    else:
        emit(f"Wrote empty stable_lowE_per_id.xyz to {stable_out_per_id_path}")

    stable_lowE_per_formula_frames: list[tuple[int, int, str]] = []
    for fid, (best_e, oid, best_k, best_lines) in best_per_formula.items():
        _ = fid, best_e  # keep for clarity; sorting key uses (oid, best_k)
        formula = fid_to_formula.get(fid, "")
        if len(best_lines) >= 2:
            best_lines[1] = _append_kv(best_lines[1], "formula_id", str(fid))
            if formula:
                best_lines[1] = _append_kv(best_lines[1], "formula", formula)
        stable_lowE_per_formula_frames.append((oid, best_k, "\n".join(best_lines)))
    stable_lowE_per_formula_frames.sort(key=lambda t: (t[0], t[1]))  # sort by (id, conformer_id)

    stable_out_per_formula_text = "\n".join(
        block for _, _, block in stable_lowE_per_formula_frames
    ) + ("\n" if stable_lowE_per_formula_frames else "")

    stable_out_per_formula_path.write_text(
        stable_out_per_formula_text,
        encoding="utf-8",
    )
    if stable_lowE_per_formula_frames:
        emit(
            f"Wrote stable_lowE_per_formula.xyz to {stable_out_per_formula_path} "
            f"(n={len(stable_lowE_per_formula_frames)})"
        )
    else:
        emit(f"Wrote empty stable_lowE_per_formula.xyz to {stable_out_per_formula_path}")

    # Persist run-time info/warnings into ana.log together with summary tables.
    lines.append("")
    lines.append("runtime_messages")
    lines.append("-" * 80)
    lines.extend(runtime_logs)

    ana_path = args.gen_dir / "ana.log"
    lines.append(f"Wrote this file to: {ana_path}")
    ana_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

