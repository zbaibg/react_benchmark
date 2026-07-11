#!/usr/bin/env python3
"""Scan metallogen.log files for short Zn-N / Zn-O distances.

For each structure id (directory under gen_struc_dftbplus), this script:
- parses the corresponding metallogen.log
- finds each relaxed conformer (blocks with "Metal-ligand distances" followed by "Relaxation success!")
- counts how many conformers have at least one Zn–N or Zn–O bond shorter than thresholds

Additionally, for each id it picks the lowest-energy stable conformer and writes its xyz block
from `save_dir/result_x.xyz` (where x is 1-based index used in `metallogen.log`) into
`stable_lowE_per_id.xyz` (and also aggregates per formula into `stable_lowE_per_formula.xyz`).

When choosing the lowest-energy stable conformer, the script will also check the corresponding
DFTB+ relaxation log `work_dir/final_relax_{x-1}.dftbplus.log` (where x is the conformer_id from
`metallogen.log`). If that log indicates a failed/non-converged relaxation, the conformer is
treated as failed and skipped from geometry/topology screening and from the final selection.

Convergence rule used here: if `final_relax_*.dftbplus.log` exists, it **must** contain
"Geometry converged" to be considered converged. "ERROR STOP" always indicates failure.

Then it aggregates by formula_id using zn_msmiles_enumeration.tsv.

Shared parsers / geometry helpers live in ``metallogen_check_common.py`` (same directory).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend for batch runs
import matplotlib.pyplot as plt

from metallogen_check_common import (
    ATOM_D_CRITERIA,
    BOND_CRITERIA,
    RATIO_CRITERIA,
    dftbplus_relax_failed,
    _RELAXING_CONFORMER_RE,
    _ENERGY_RE,
    _CLUSTER_ID_RE,
    _ATOM_LINE_RE,
    load_enumeration,
    _parse_atom_line,
    parse_multi_xyz_frames,
    _elements_coords,
    _atoms_to_metallogen_molecule,
    _first_zn_index,
    max_zn_n_zn_o_distances_angstrom,
    expected_topology_from_msmiles,
    validate_geometry_and_topology,
    extract_single_xyz_atoms,
    parse_metallogen_log,
    parse_metallogen_log_with_max_dists,
    max_zn_n_zn_o_from_metallogen_log,
    extract_xyz_blocks_by_cluster_ids,
    extract_single_xyz_with_energy,
)

ZN_N_SUSPECT = 2.5  # Å, threshold for Zn-N
ZN_O_SUSPECT = 2.5  # Å, threshold for Zn-O

_ITER_STRUC = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tsv",
        type=Path,
        default=_ITER_STRUC / "zn_msmiles_enumeration.tsv",
        help="Enumeration TSV with id / formula_id / formula",
    )
    p.add_argument(
        "gen_dir",
        type=Path,
        help="Directory containing */metallogen.log",
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
        "--zn_n_thresh",
        type=float,
        default=ZN_N_SUSPECT,
        help="Threshold (Å) for Zn-N distances",
    )
    p.add_argument(
        "--zn_o_thresh",
        type=float,
        default=ZN_O_SUSPECT,
        help="Threshold (Å) for Zn-O distances",
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


def main() -> None:
    args = parse_args()
    id_to_fid, fid_to_formula, id_to_msmiles = load_enumeration(args.tsv)
    runtime_logs: list[str] = []
    # For plotting: collect max Zn distances (Å) for all passed conformers,
    # across all ids in gen_dir.
    passed_all_max_zn_n: list[float] = []
    passed_all_max_zn_o: list[float] = []
    passed_hist_total_conformers: int = 0
    passed_hist_over_zn_o: int = 0
    passed_hist_over_zn_n: int = 0
    passed_hist_over_any: int = 0  # Zn-O > ZN_O_SUSPECT or Zn-N > ZN_N_SUSPECT
    passed_hist_over_both: int = 0  # Zn-O > ZN_O_SUSPECT and Zn-N > ZN_N_SUSPECT
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
    # Longest Zn–N / Zn–O (Å) for the passed lowest-E conformer per id; None if no N/O or no Zn.
    id_best_max_zn_n: dict[int, float | None] = {}
    id_best_max_zn_o: dict[int, float | None] = {}
    # formula_id -> (E, oid, k, block_lines)
    best_per_formula: dict[int, tuple[float, int, int, list[str]]] = {}
    # Same metrics for the formula-level lowest-E passed structure (one id per formula_id).
    fid_best_max_zn_n: dict[int, float | None] = {}
    fid_best_max_zn_o: dict[int, float | None] = {}

    for subdir in sorted(args.gen_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            oid = int(subdir.name)
        except ValueError:
            continue

        log_path = subdir / "metallogen.log"
        (
            total_conf,
            n_conf_stable,
            stable_conformer_nums,
            relaxed_conformer_nums,
            max_dists_by_k,
        ) = parse_metallogen_log_with_max_dists(
            log_path, args.zn_n_thresh, args.zn_o_thresh
        )
        if total_conf == 0:
            continue
        id_stats[oid] = {
            "total_conf": total_conf,
            "n_conf_stable": n_conf_stable,
            "n_conf_passed": 0,
            "passed_ids": [],
        }
        id_best_passed_k[oid] = None
        id_best_passed_e[oid] = None

        msmiles = id_to_msmiles.get(oid)
        if not msmiles:
            emit(f"[WARN] id={oid}: missing msmiles in TSV; skip topology/geometry checks")
            continue
        try:
            expected_elements, expected_adj = expected_topology_from_msmiles(msmiles)
        except Exception as e:
            emit(f"[WARN] id={oid}: failed to reconstruct expected topology from msmiles: {e}")
            continue

        # Histogram base ("passed"): relaxed conformers that also pass
        # (1) final DFTB+ relaxation convergence and (2) geometry/topology screening.
        # Note: Zn-N/Zn-O suspect distance thresholds are NOT applied here.
        for k in relaxed_conformer_nums:
            relax_idx = k - 1
            relax_log_path = subdir / "work_dir" / f"final_relax_{relax_idx}.dftbplus.log"
            if relax_log_path.exists():
                try:
                    relax_log_text = relax_log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    relax_log_text = ""
                if dftbplus_relax_failed(relax_log_text):
                    continue
            else:
                warn_key = (oid, k)
                if warn_key not in warned_missing_relax_logs:
                    warned_missing_relax_logs.add(warn_key)
                    emit(
                        f"[WARN] id={oid} k={k}: missing {relax_log_path.name}; "
                        "optimization may be unconverged. Here we manually treat this "
                        "case as geometry-converged and SCC-successful for downstream screening."
                    )

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
            max_zn_n, max_zn_o = max_dists_by_k.get(k, (None, None))

            if max_zn_n is not None:
                passed_all_max_zn_n.append(float(max_zn_n))
            if max_zn_o is not None:
                passed_all_max_zn_o.append(float(max_zn_o))

            over_n = max_zn_n is not None and float(max_zn_n) > ZN_N_SUSPECT
            over_o = max_zn_o is not None and float(max_zn_o) > ZN_O_SUSPECT
            if over_n:
                passed_hist_over_zn_n += 1
            if over_o:
                passed_hist_over_zn_o += 1
            if over_n or over_o:
                passed_hist_over_any += 1
            if over_n and over_o:
                passed_hist_over_both += 1

        # Collect lowest-energy conformer from fully passed (distance + geometry + topology).
        if not stable_conformer_nums:
            emit(
                f"[INFO] id={oid}: no stable conformers (n_conf_stable={n_conf_stable}); skip stable_lowE selection"
            )
            continue

        # Now stable conformer index k is expected to map to save_dir/result_{k}.xyz
        # (k is 1-based index as shown in metallogen.log: "Relaxing conformer k/...").
        best: tuple[float, int, list[str]] | None = None  # (E, k, block_lines)
        missing_ks: list[int] = []
        fail_reasons: dict[str, int] = defaultdict(int)
        passed_conformer_ids: list[int] = []

        for k in stable_conformer_nums:
            # Optionally skip conformers whose final DFTB+ relaxation failed.
            # Convention: final_relax_x.dftbplus.log corresponds to conformer_id = x + 1.
            relax_idx = k - 1
            relax_log_path = subdir / "work_dir" / f"final_relax_{relax_idx}.dftbplus.log"
            if relax_log_path.exists():
                try:
                    relax_log_text = relax_log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    relax_log_text = ""
                if dftbplus_relax_failed(relax_log_text):
                    fail_reasons["final_relax_error"] += 1
                    continue
            else:
                warn_key = (oid, k)
                if warn_key not in warned_missing_relax_logs:
                    warned_missing_relax_logs.add(warn_key)
                    emit(
                        f"[WARN] id={oid} k={k}: missing {relax_log_path.name}; "
                        "optimization may be unconverged. Here we manually treat this "
                        "case as geometry-converged and SCC-successful for downstream screening."
                    )

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
        max_zn_n, max_zn_o = max_dists_by_k.get(best_k, (None, None))
        id_best_max_zn_n[oid] = max_zn_n
        id_best_max_zn_o[oid] = max_zn_o

        fid = id_to_fid.get(oid)
        formula = fid_to_formula.get(fid, "") if fid is not None else ""
        def _fmt_d(d: float | None) -> str:
            return f"{d:.6f}" if d is not None else "none"

        emit(
            f"[INFO] id={oid} formula_id={fid if fid is not None else 'none'} "
            f"lowestE_conformer k={best_k} E={best_e:.12f} "
            f"max_Zn-N={_fmt_d(max_zn_n)} Å max_Zn-O={_fmt_d(max_zn_o)} Å"
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
                fid_best_max_zn_n[fid] = max_zn_n
                fid_best_max_zn_o[fid] = max_zn_o

    # Plot histograms for all passed conformers (including those that exceed suspect thresholds).
    # Produce separate wide images for Zn-O and Zn-N.
    if passed_all_max_zn_o or passed_all_max_zn_n:
        passed_all_o = (
            np.asarray(passed_all_max_zn_o, dtype=float)
            if passed_all_max_zn_o
            else np.asarray([], dtype=float)
        )
        passed_all_n = (
            np.asarray(passed_all_max_zn_n, dtype=float)
            if passed_all_max_zn_n
            else np.asarray([], dtype=float)
        )

        bin_width = 0.1
        tick_step = 0.5

        if passed_all_o.size:
            # 为了让恰好等于 ZN_O_SUSPECT 的点落在左侧 bin，把它们轻微左移一个 eps。
            eps = 1e-6
            passed_all_o_bins = passed_all_o.copy()
            passed_all_o_bins[np.isclose(passed_all_o_bins, ZN_O_SUSPECT)] -= eps

            passed_o_over_suspect = passed_all_o[passed_all_o > ZN_O_SUSPECT]
            xmin, xmax = float(np.min(passed_all_o_bins)), float(np.max(passed_all_o_bins))
            if xmin == xmax:
                xmin -= bin_width
                xmax += bin_width
            start_i = int(np.floor(xmin / bin_width))
            end_i = int(np.ceil(xmax / bin_width))
            bin_edges_o = np.arange(
                start_i * bin_width, (end_i + 1) * bin_width + 1e-9, bin_width
            )
            fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=150)
            ax.hist(
                passed_all_o_bins,
                bins=bin_edges_o,
                color="#4C78A8",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.6,
            )
            if passed_o_over_suspect.size:
                ax.hist(
                    passed_o_over_suspect,
                    bins=bin_edges_o,
                    color="red",
                    alpha=0.65,
                    edgecolor="black",
                    linewidth=0.6,
                )
            ax.axvline(ZN_O_SUSPECT, color="red", linestyle="--", linewidth=1.2)
            tick_start_i = int(np.floor(xmin / tick_step))
            tick_end_i = int(np.ceil(xmax / tick_step))
            ax.set_xticks([i * tick_step for i in range(tick_start_i, tick_end_i + 1)])
            ax.set_title("Passed conformers: max Zn-O (all), red=over suspect")
            ax.set_xlabel("Zn-O max distance (Angstrom)")
            ax.set_ylabel("count")
            fig.tight_layout()
            hist_path_o = args.gen_dir / "passed_hist_maxZnO.png"
            fig.savefig(hist_path_o)
            plt.close(fig)
            emit(f"[INFO] Wrote histogram to {hist_path_o}")

        if passed_all_n.size:
            # 同理，让等于 ZN_N_SUSPECT 的点落在左侧 bin。
            eps = 1e-6
            passed_all_n_bins = passed_all_n.copy()
            passed_all_n_bins[np.isclose(passed_all_n_bins, ZN_N_SUSPECT)] -= eps

            passed_n_over_suspect = passed_all_n[passed_all_n > ZN_N_SUSPECT]
            xmin, xmax = float(np.min(passed_all_n_bins)), float(np.max(passed_all_n_bins))
            if xmin == xmax:
                xmin -= bin_width
                xmax += bin_width
            start_i = int(np.floor(xmin / bin_width))
            end_i = int(np.ceil(xmax / bin_width))
            bin_edges_n = np.arange(
                start_i * bin_width, (end_i + 1) * bin_width + 1e-9, bin_width
            )
            fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=150)
            ax.hist(
                passed_all_n_bins,
                bins=bin_edges_n,
                color="#F58518",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.6,
            )
            if passed_n_over_suspect.size:
                ax.hist(
                    passed_n_over_suspect,
                    bins=bin_edges_n,
                    color="red",
                    alpha=0.65,
                    edgecolor="black",
                    linewidth=0.6,
                )
            ax.axvline(ZN_N_SUSPECT, color="red", linestyle="--", linewidth=1.2)
            tick_start_i = int(np.floor(xmin / tick_step))
            tick_end_i = int(np.ceil(xmax / tick_step))
            ax.set_xticks([i * tick_step for i in range(tick_start_i, tick_end_i + 1)])
            ax.set_title("Passed conformers: max Zn-N (all), red=over suspect")
            ax.set_xlabel("Zn-N max distance (Angstrom)")
            ax.set_ylabel("count")
            fig.tight_layout()
            hist_path_n = args.gen_dir / "passed_hist_maxZnN.png"
            fig.savefig(hist_path_n)
            plt.close(fig)
            emit(f"[INFO] Wrote histogram to {hist_path_n}")

        emit(
            f"[INFO] Passed conformers (hist) totals: "
            f"total={passed_hist_total_conformers}, "
            f"Zn-O>{ZN_O_SUSPECT:.3f}={passed_hist_over_zn_o}, "
            f"Zn-N>{ZN_N_SUSPECT:.3f}={passed_hist_over_zn_n}, "
            f"any_over={passed_hist_over_any}, "
            f"both_over={passed_hist_over_both}"
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
        f"Thresholds: Zn-N < {args.zn_n_thresh:.3f} Å, "
        f"Zn-O < {args.zn_o_thresh:.3f} Å"
    )
    lines.append(
        f"Geometry/Topology criteria: ratio >= {args.ratio_criteria:.3f}, "
        f"min_dist >= {args.atom_d_criteria:.3f} Å, bond_ratio < {args.bond_criteria:.3f}"
    )
    lines.append("Topology reference: reconstructed from msmiles (TSV column `msmiles`).")
    lines.append("Definitions:")
    lines.append(
        "  - n_conf_stable: conformers passing Zn-N/Zn-O distance thresholds from metallogen.log."
    )
    lines.append(
        "  - n_conf_passed: subset of n_conf_stable that also passes geometry/topology checks, "
        "and whose corresponding final_relax_{k-1}.dftbplus.log does not indicate failure "
        '(requires "Geometry converged" and must not contain ERROR STOP).'
    )
    lines.append(
        "  - max_Zn-N / max_Zn-O: max Zn–N and Zn–O distances (Å) within the corresponding "
        '"Metal-ligand distances" block in metallogen.log for the passed lowest-E conformer.'
    )
    lines.append("  - set relation: passed_conformer_ids ⊆ stable_conformer_ids.")
    lines.append("")

    header_id = (
        "id\tformula_id\ttotal_conf\tn_conf_stable\tn_conf_passed\tlowest_E\t"
        "max_Zn-N_A\tmax_Zn-O_A\tpassed_conformer_ids"
    )
    lines.append(header_id)
    lines.append("-" * min(120, len(header_id) + 40))
    for oid in sorted(id_stats.keys()):
        stats = id_stats[oid]
        passed_ids = ",".join(str(k) for k in stats["passed_ids"])  # type: ignore[index]
        lowest_e = id_best_passed_e.get(oid)
        lowest_e_str = f"{lowest_e:.12f}" if lowest_e is not None else "none"
        mzn = id_best_max_zn_n.get(oid)
        mzo = id_best_max_zn_o.get(oid)
        mzn_str = f"{mzn:.6f}" if mzn is not None else "none"
        mzo_str = f"{mzo:.6f}" if mzo is not None else "none"
        fid = id_to_fid.get(oid)
        fid_str = str(fid) if fid is not None else "none"
        lines.append(
            f"{oid}\t{fid_str}\t{stats['total_conf']}\t{stats['n_conf_stable']}\t"
            f"{stats['n_conf_passed']}\t{lowest_e_str}\t{mzn_str}\t{mzo_str}\t{passed_ids}"
        )

    lines.append("")
    header_fid = (
        "formula_id\tformula\ttotal_conf\tn_conf_stable\tn_conf_passed\tlowest_E\t"
        "max_Zn-N_lowestE_A\tmax_Zn-O_lowestE_A\tpassed_lowE_list_by_id"
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
        f_mzn = fid_best_max_zn_n.get(fid)
        f_mzo = fid_best_max_zn_o.get(fid)
        f_mzn_str = f"{f_mzn:.6f}" if f_mzn is not None else "none"
        f_mzo_str = f"{f_mzo:.6f}" if f_mzo is not None else "none"
        lines.append(
            f"{fid}\t{formula}\t{agg['total_conf']}\t{agg['n_conf_stable']}\t"
            f"{agg['n_conf_passed']}\t{lowest_e_fid_str}\t{f_mzn_str}\t{f_mzo_str}\t{passed_list}"
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
    ana_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {ana_path}")


if __name__ == "__main__":
    main()

