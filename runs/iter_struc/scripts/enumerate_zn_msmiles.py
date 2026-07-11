#!/usr/bin/env python3
"""
Enumerate m-SMILES for Zn(II) complexes with MIm, MImH, and MeOH ligands (CN = 4, 5, 6).

Geometries: CN 4 → ``4_tetrahedral`` only; CN 5 → ``5_trigonal_bipyramidal`` and
``5_square_pyramidal``; CN 6 → ``6_octahedral`` only.

Distinct-structure counts use Burnside's lemma on the **full** point group of each
polyhedron (distance-preserving permutations of vertices, including reflections), so
enantiomeric pairs are identified with their mirror images.

Run on a compute node (example):
  srun -n 1 --cpus-per-task=2 -t 10:00 \\
    /path/to/metallogen/env/python enumerate_zn_msmiles.py

Requires MetalloGen installed in the active Python environment.

Validation (unless --skip-validate): MetalloGen must parse the m-SMILES, parsed charge
must match the expected model, and multiplicity must be 1. On any failure the script prints
to stderr and exits with code 1 without writing the TSV. Validation diagnostics are not
written to the TSV file.

Polyhedron vertices and atom-map site indices follow MetalloGen ``globalvars.known_geometries_vector_dict``
(site ``k`` → ``direction_vector[k-1]``). For each ``formula`` composition and geometry, one TSV row
per distinct isomer: ``id`` is 1-based over all output rows (sort order); ``formula_id``
is 1-based over distinct formulas (in output sort order); ``isomerid`` is 1-based within
that formula across all geometries; ``msmiles``
is the lexicographically canonical orbit representative (ligand blocks ordered by site).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# Ligand SMILES templates: {site} is replaced with the coordination index 1..CN at the
# donor heavy atom (Zn binds there in MetalloGen embedding).
#
# Coordination donors (chemistry):
#   MImH: the ring nitrogen that does *not* bear the imidazole NH — map on ``n``, not ``[nH]``.
#   MIm:  either ring N is a reasonable donor; here the map sits on the formally deprotonated
#         N (``[n-:{site}]``), consistent with RDKit/MIm(-) charge assignment.
#   MeOH: oxygen donor — ``[OH:{site}]`` attaches the map to O (not ``[O:{site}]`` alone).
#
# Charge model (same as MetalloGen: total charge = formal charge on metal + sum of ligand
# formal charges from RDKit):
#   Zn +2, MIm -1, MImH 0, MeOH 0  =>  complex charge = 2 - n_MIm
#
# SMILES match monomers/MIm.xyz, MImH.xyz, MeOH.xyz element counts (ORCA minima):
#   MIm:  Cc1ncc[n-:{site}]1
#   MImH: Cc1[n:{site}]cc[nH]1
# MeOH: C[OH:{site}] — keep hydroxyl H in SMILES (C[O:{site}] drops one H in RDKit).
LIGAND_TEMPLATES = {
    "MIm": "Cc1ncc[n-:{site}]1",
    "MImH": "Cc1[n:{site}]cc[nH]1",
    "MeOH": "C[OH:{site}]",
}

METAL = "[Zn+2]"
# Mononuclear complexes: one Zn center per m-SMILES row
N_ZN = 1

# Only these geometries are enumerated (must match MetalloGen ``globalvars`` keys).
GEOMETRIES_BY_CN: dict[int, tuple[str, ...]] = {
    4: ("4_tetrahedral",),
    5: ("5_trigonal_bipyramidal", "5_square_pyramidal"),
    6: ("6_octahedral",),
}

# Ligand kind for multiset colorings: 0=MIm, 1=MImH, 2=MeOH (lex order for canonical reps).
LIGAND_CODES: tuple[str, ...] = ("MIm", "MImH", "MeOH")


@lru_cache(maxsize=1)
def _metallogen_geometry_vertices() -> dict[str, np.ndarray]:
    """
    3D direction vectors exactly as in MetalloGen ``globalvars.known_geometries_vector_dict``.
    Atom map ``:k`` in m-SMILES refers to ``direction_vector[k-1]`` (see MetalloGen ``embed``).
    """
    from MetalloGen import globalvars as gv

    out: dict[str, np.ndarray] = {}
    for geoms in GEOMETRIES_BY_CN.values():
        for name in geoms:
            out[name] = np.asarray(gv.known_geometries_vector_dict[name], dtype=float)
    return out


def _distance_preserving_permutations(verts: np.ndarray) -> list[tuple[int, ...]]:
    """All permutations p of vertex indices with d(v_i,v_j)=d(v_{p[i]},v_{p[j]}) for all i,j."""
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    good: list[tuple[int, ...]] = []
    tol = 1e-5
    for p in itertools.permutations(range(n)):
        ok = True
        for i in range(n):
            for j in range(n):
                dij = float(np.linalg.norm(verts[i] - verts[j]))
                d = float(np.linalg.norm(verts[p[i]] - verts[p[j]]))
                if abs(dij - d) > tol:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(tuple(p))
    return list(set(good))


@lru_cache(maxsize=None)
def symmetry_group_permutations(geometry: str) -> tuple[tuple[int, ...], ...]:
    """Cached full point group as vertex permutations (rotations + reflections)."""
    verts = _metallogen_geometry_vertices()[geometry]
    return tuple(sorted(_distance_preserving_permutations(verts)))


def _permutation_cycle_lengths(p: tuple[int, ...]) -> list[int]:
    """Cycle lengths of permutation p with p[i] = image of vertex i (0-based)."""
    n = len(p)
    seen = [False] * n
    lengths: list[int] = []
    for i in range(n):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        lengths.append(length)
    return lengths


@lru_cache(maxsize=None)
def _count_colorings_fixed_by_cycles(
    cycle_lengths_key: tuple[int, ...], n_mim: int, n_mimh: int, n_meoh: int
) -> int:
    """
    Number of assignments of MIm/MImH/MeOH to disjoint cycles (each cycle monochromatic)
    that use exactly n_mim, n_mimh, n_meoh ligands.
    """
    cycles = tuple(sorted(cycle_lengths_key, reverse=True))

    @lru_cache(None)
    def dp(i: int, a: int, b: int, c: int) -> int:
        if i == len(cycles):
            return 1 if a == b == c == 0 else 0
        if a < 0 or b < 0 or c < 0:
            return 0
        L = cycles[i]
        s = 0
        if a >= L:
            s += dp(i + 1, a - L, b, c)
        if b >= L:
            s += dp(i + 1, a, b - L, c)
        if c >= L:
            s += dp(i + 1, a, b, c - L)
        return s

    return dp(0, n_mim, n_mimh, n_meoh)


def multiset_colorings(n_mim: int, n_mimh: int, n_meoh: int) -> set[tuple[int, ...]]:
    """All assignments of ligand codes 0/1/2 to sites with fixed counts."""
    n = n_mim + n_mimh + n_meoh
    base = (0,) * n_mim + (1,) * n_mimh + (2,) * n_meoh
    return set(itertools.permutations(base, n))


def orbit_canonical_coloring(
    coloring: tuple[int, ...], group: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Lexicographically smallest tuple in the orbit of ``coloring`` under ``group``."""
    best: tuple[int, ...] | None = None
    n = len(coloring)
    for p in group:
        t = tuple(coloring[p[i]] for i in range(n))
        if best is None or t < best:
            best = t
    assert best is not None
    return best


def distinct_structure_canonical_colorings(
    geometry: str, n_mim: int, n_mimh: int, n_meoh: int
) -> tuple[tuple[int, ...], ...]:
    """
    One canonical representative per orbit of multiset colorings under the polyhedron
    symmetry (same count as ``distinct_structure_count``).
    """
    G = symmetry_group_permutations(geometry)
    seen: set[tuple[int, ...]] = set()
    for c in multiset_colorings(n_mim, n_mimh, n_meoh):
        seen.add(orbit_canonical_coloring(c, G))
    return tuple(sorted(seen))


def distinct_structure_count(
    geometry: str, n_mim: int, n_mimh: int, n_meoh: int
) -> int:
    """
    Orbit count of multiset colorings (MIm / MImH / MeOH) on coordination sites under the
    full symmetry group of the polyhedron (Burnside). Mirror-related configurations are
    identified because reflections are included in the group.
    """
    G = symmetry_group_permutations(geometry)
    if not G:
        return 0
    total = 0
    for p in G:
        cl = tuple(sorted(_permutation_cycle_lengths(p)))
        total += _count_colorings_fixed_by_cycles(cl, n_mim, n_mimh, n_meoh)
    assert total % len(G) == 0, (geometry, total, len(G))
    return total // len(G)


def expected_complex_charge(n_mim: int, n_mimh: int, n_meoh: int) -> int:
    """Zn+2 + MIm(-1) + MImH(0) + MeOH(0) => 2 - n_mim."""
    _ = n_mimh, n_meoh
    return 2 - n_mim


def compositions_sum_n(n: int) -> list[tuple[int, int, int]]:
    """
    All non-negative integer triples (a, b, c) with a + b + c = n
    (counts of MIm, MImH, MeOH).
    """
    out = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            out.append((a, b, c))
    return out


def build_msmiles_from_coloring(coloring: tuple[int, ...], geometry: str) -> str:
    """
    Build m-SMILES: site ``k`` (atom map ``:k``) uses ``direction_vector[k-1]`` in MetalloGen.
    ``coloring[i]`` is 0=MIm, 1=MImH, 2=MeOH at site ``i+1``.
    Ligand blocks are ordered by site index 1..CN.
    """
    blocks = [METAL]
    for i, code in enumerate(coloring):
        tmpl = LIGAND_TEMPLATES[LIGAND_CODES[code]]
        blocks.append(tmpl.format(site=i + 1))
    blocks.append(geometry)
    return "|".join(blocks)


def stoichiometry_label(n_mim: int, n_mimh: int, n_meoh: int) -> str:
    return f"{N_ZN}Zn_{n_mim}MIm_{n_mimh}MImH_{n_meoh}MeOH"


def validate_msmiles(msmiles: str, expected_chg: int) -> int:
    """
    Parse m-SMILES with MetalloGen; verify charge equals expected_chg and multiplicity is 1.

    Returns the parsed total charge on success.

    On failure: raises whatever ``get_om_from_modified_smiles`` raises (parse error), or
    ``ValueError`` if charge or multiplicity does not match. No try/except here so parse
    failures are real exceptions; the caller collects them when iterating all rows.
    """
    from MetalloGen import om

    mc = om.get_om_from_modified_smiles(msmiles)
    parsed = mc.chg
    if parsed != expected_chg:
        raise ValueError(
            f"charge_mismatch: expected {expected_chg}, MetalloGen chg={parsed}"
        )
    if mc.multiplicity != 1:
        raise ValueError(
            f"multiplicity_mismatch: expected 1, MetalloGen multiplicity={mc.multiplicity}"
        )
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enumerate Zn MIm/MImH/MeOH m-SMILES combinatorics"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "zn_msmiles_enumeration.tsv",
        help="TSV output path",
    )
    ap.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip MetalloGen parsing and charge verification (faster)",
    )
    args = ap.parse_args()

    planned: list[tuple] = []
    for cn in (4, 5, 6):
        for geom in GEOMETRIES_BY_CN[cn]:
            for a, b, c in compositions_sum_n(cn):
                stoich = stoichiometry_label(a, b, c)
                si = distinct_structure_count(geom, a, b, c)
                reps = distinct_structure_canonical_colorings(geom, a, b, c)
                assert len(reps) == si, (
                    f"orbit count {len(reps)} != Burnside {si} for {geom} {stoich}"
                )
                exp = expected_complex_charge(a, b, c)
                for struct_idx, coloring in enumerate(reps, start=1):
                    msmiles = build_msmiles_from_coloring(coloring, geom)
                    planned.append(
                        (cn, geom, a, b, c, stoich, msmiles, exp, struct_idx)
                    )

    charges: list[int] = []
    if args.skip_validate:
        charges = [p[7] for p in planned]
    else:
        failures: list[tuple[str, str, str]] = []
        for cn, geom, a, b, c, stoich, msmiles, exp, _idx in planned:
            try:
                charges.append(validate_msmiles(msmiles, exp))
            except Exception as e:
                failures.append((stoich, geom, f"{type(e).__name__}: {e}"))
        if failures:
            print(
                f"ERROR: {len(failures)} row(s) failed MetalloGen validation "
                "(parse, charge, or multiplicity).",
                file=sys.stderr,
            )
            for stoich, geom, err in failures[:20]:
                print(f"  {stoich} | {geom} | {err}", file=sys.stderr)
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
            print("TSV was not written.", file=sys.stderr)
            sys.exit(1)

    # Sort output by (n_MIm+n_MImH, n_MIm, n_MeOH, geometry, per-geometry struct index)
    combined = list(zip(planned, charges))
    combined.sort(
        key=lambda pc: (
            pc[0][2] + pc[0][3],  # n_MIm+n_MImH
            pc[0][2],  # n_MIm
            pc[0][4],  # n_MeOH
            pc[0][1],  # geometry
            pc[0][8],  # struct index within (formula composition, geometry)
        )
    )

    rows: list[dict] = []
    formula_id = 0
    prev_stoich: str | None = None
    isomer_within_stoich = 0
    row_id = 0
    for (cn, geom, a, b, c, stoich, msmiles, _exp, _struct_idx), chg in combined:
        row_id += 1
        if stoich != prev_stoich:
            prev_stoich = stoich
            formula_id += 1
            isomer_within_stoich = 0
        isomer_within_stoich += 1
        total_imid = a + b
        rows.append(
            {
                "id": row_id,
                "formula_id": formula_id,
                "formula": stoich,
                "n_MIm+n_MImH": total_imid,
                "n_MIm": a,
                "n_MImH": b,
                "n_MeOH": c,
                "CN": cn,
                "charge": chg,
                "geometry": geom,
                "isomerid": isomer_within_stoich,
                "msmiles": msmiles,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "formula_id",
        "formula",
        "n_MIm+n_MImH",
        "n_MIm",
        "n_MImH",
        "n_MeOH",
        "CN",
        "charge",
        "geometry",
        "isomerid",
        "msmiles",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        # Unix LF only: csv's default dialect uses CRLF, which breaks shell pipelines
        # (last field keeps \r, so MetalloGen sees geometry names like "4_tetrahedral\r").
        w = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.skip_validate:
        print("Validation was skipped (--skip-validate).")
    else:
        print(
            "All entries passed: MetalloGen parse OK, charge OK, multiplicity=1."
        )


if __name__ == "__main__":
    main()
