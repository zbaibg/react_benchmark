#!/usr/bin/env python3
"""
Enumerate m-SMILES for Zn(II) complexes with NO3- and MeOH ligands
for CN = 4, 5, 6, merging mirror images as symmetry-equivalent.

Geometries:
  - CN 4:  4_tetrahedral
  - CN 5:  5_trigonal_bipyramidal, 5_square_pyramidal
  - CN 6:  6_octahedral

Ligand coordination modes:
  - MeOH:      monodentate O donor (occupies 1 site)
  - NO3_mono:  monodentate O donor (occupies 1 site)
  - NO3_bi:    bidentate, occupies 2 adjacent sites

For each CN and geometry, enumerate all non-negative integer triples
(n_NO3_bi, n_NO3_mono, n_MeOH) such that

    2 * n_NO3_bi + n_NO3_mono + n_MeOH = CN

and all symmetry-distinct placements of:
  - n_NO3_bi disjoint adjacent pairs,
  - n_NO3_mono monodentate NO3 on remaining sites,
  - n_MeOH on the rest.

Symmetry handling:
  - Uses the full distance-preserving permutation group of the geometry
    vertices, including reflections.
  - Therefore mirror-image complexes are merged.

Adjacency model for bidentate NO3:
  - 4_tetrahedral: all 6 pairs are adjacent
  - 5_trigonal_bipyramidal: all 10 pairs except the unique axial-axial pair
  - 5_square_pyramidal: all 10 pairs except the 2 basal-basal diagonals
  - 6_octahedral: all 15 pairs except the 3 trans pairs

These non-adjacent pairs are identified from distance classes of the
MetalloGen direction vectors, so the method is invariant to rotation,
reflection, vertex ordering, and axis choices.

Charge model:
  - Zn:  +2
  - NO3: -1  (mono or bi, same ligand charge)
  - MeOH: 0
  => total charge = 2 - (n_NO3_bi + n_NO3_mono)

Output TSV columns:
  id, formula_id, formula, n_NO3, n_NO3_mono, n_NO3_bi, n_MeOH,
  CN, charge, geometry, isomerid, msmiles

  formula / formula_id: formula is 1Zn_{n_NO3}NO3_{n_MeOH}MeOH (no mono/bi split).
  All rows sharing that string share one formula_id, regardless of n_NO3_mono /
  n_NO3_bi or CN.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


LIGAND_TEMPLATES = {
    "MeOH": "C[OH:{site}]",
    "NO3_mono": "[O-:{site}][N+](=O)[O-]",
    "NO3_bi": "[O-:{site1}][N+](=O)[O-:{site2}]",
}

METAL = "[Zn+2]"
N_ZN = 1

GEOMETRIES_BY_CN: dict[int, tuple[str, ...]] = {
    4: ("4_tetrahedral",),
    5: ("5_trigonal_bipyramidal", "5_square_pyramidal"),
    6: ("6_octahedral",),
}


@lru_cache(maxsize=1)
def _metallogen_geometry_vertices() -> dict[str, np.ndarray]:
    """
    Load MetalloGen direction vectors for the geometries we enumerate.
    """
    from MetalloGen import globalvars as gv

    out: dict[str, np.ndarray] = {}
    for geoms in GEOMETRIES_BY_CN.values():
        for name in geoms:
            out[name] = np.asarray(gv.known_geometries_vector_dict[name], dtype=float)
    return out


def _distance_preserving_permutations(verts: np.ndarray) -> list[tuple[int, ...]]:
    """
    All vertex permutations preserving the full pairwise distance matrix.
    Includes proper rotations and reflections.
    """
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    tol = 1e-5
    good: list[tuple[int, ...]] = []

    for p in itertools.permutations(range(n)):
        ok = True
        for i in range(n):
            for j in range(n):
                dij = float(np.linalg.norm(verts[i] - verts[j]))
                dimg = float(np.linalg.norm(verts[p[i]] - verts[p[j]]))
                if abs(dij - dimg) > tol:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(tuple(p))

    return sorted(set(good))


@lru_cache(maxsize=None)
def symmetry_group_permutations(geometry: str) -> tuple[tuple[int, ...], ...]:
    """
    Full point group as vertex permutations.
    """
    verts = _metallogen_geometry_vertices()[geometry]
    return tuple(_distance_preserving_permutations(verts))


@lru_cache(maxsize=None)
def _distance_classes(
    geometry: str,
) -> tuple[tuple[float, tuple[tuple[int, int], ...]], ...]:
    """
    Cluster pairwise distances into classes.

    Returns a tuple of:
        (representative_distance, ((i, j), ...))
    sorted by increasing representative distance.
    """
    verts = _metallogen_geometry_vertices()[geometry]
    n = len(verts)
    rel_tol = 1e-5

    pairs_with_d: list[tuple[float, tuple[int, int]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(verts[i] - verts[j]))
            pairs_with_d.append((d, (i, j)))
    pairs_with_d.sort(key=lambda x: x[0])

    classes: list[list[tuple[float, tuple[int, int]]]] = []
    for d, pair in pairs_with_d:
        if not classes:
            classes.append([(d, pair)])
            continue
        prev_d = classes[-1][0][0]
        if abs(d - prev_d) <= rel_tol * max(1.0, prev_d):
            classes[-1].append((d, pair))
        else:
            classes.append([(d, pair)])

    out: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    for cls in classes:
        rep = sum(d for d, _ in cls) / len(cls)
        pairs = tuple(sorted(pair for _, pair in cls))
        out.append((rep, pairs))
    return tuple(out)


@lru_cache(maxsize=None)
def _geometry_edges(geometry: str) -> tuple[tuple[int, int], ...]:
    """
    Return all adjacent site pairs for the given geometry.

    This implementation is purely distance-class based and uses the known
    ideal non-edge pattern of each polyhedron:

      tetrahedral:              no non-edges
      trigonal bipyramidal:     1 non-edge  (axial-axial)
      square pyramidal:         2 non-edges (basal diagonals)
      octahedral:               3 non-edges (trans pairs)
    """
    verts = _metallogen_geometry_vertices()[geometry]
    n = len(verts)
    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

    if geometry == "4_tetrahedral":
        edges = all_pairs

    else:
        dclasses = _distance_classes(geometry)
        if len(dclasses) < 2:
            raise ValueError(
                f"{geometry}: expected at least 2 distance classes, got {len(dclasses)}"
            )

        longest_pairs = set(dclasses[-1][1])

        if geometry == "5_trigonal_bipyramidal":
            if len(longest_pairs) != 1:
                raise ValueError(
                    f"{geometry}: expected 1 longest-distance pair (axial-axial), "
                    f"got {len(longest_pairs)}"
                )
            edges = all_pairs - longest_pairs

        elif geometry == "5_square_pyramidal":
            if len(longest_pairs) != 2:
                raise ValueError(
                    f"{geometry}: expected 2 longest-distance pairs (basal diagonals), "
                    f"got {len(longest_pairs)}"
                )
            edges = all_pairs - longest_pairs

        elif geometry == "6_octahedral":
            if len(longest_pairs) != 3:
                raise ValueError(
                    f"{geometry}: expected 3 longest-distance pairs (trans pairs), "
                    f"got {len(longest_pairs)}"
                )
            edges = all_pairs - longest_pairs

        else:
            raise ValueError(f"Unsupported geometry: {geometry!r}")

    edges_t = tuple(sorted(edges))

    # Sanity checks for ideal polyhedra.
    deg = [0] * n
    for i, j in edges_t:
        deg[i] += 1
        deg[j] += 1

    if geometry == "4_tetrahedral":
        if len(edges_t) != 6 or any(d != 3 for d in deg):
            raise ValueError(f"{geometry}: bad adjacency, n_edges={len(edges_t)}, deg={deg}")
    elif geometry == "5_trigonal_bipyramidal":
        if len(edges_t) != 9 or sorted(deg) != [3, 3, 4, 4, 4]:
            raise ValueError(f"{geometry}: bad adjacency, n_edges={len(edges_t)}, deg={deg}")
    elif geometry == "5_square_pyramidal":
        if len(edges_t) != 8 or sorted(deg) != [3, 3, 3, 3, 4]:
            raise ValueError(f"{geometry}: bad adjacency, n_edges={len(edges_t)}, deg={deg}")
    elif geometry == "6_octahedral":
        if len(edges_t) != 12 or any(d != 4 for d in deg):
            raise ValueError(f"{geometry}: bad adjacency, n_edges={len(edges_t)}, deg={deg}")

    return edges_t


def _edge_matchings(
    edges: tuple[tuple[int, int], ...], n_vertices: int, k: int
) -> list[tuple[tuple[int, int], ...]]:
    """
    All matchings of size k from the edge set.
    """
    results: list[tuple[tuple[int, int], ...]] = []

    def backtrack(
        start: int,
        remaining_k: int,
        used: list[bool],
        acc: list[tuple[int, int]],
    ) -> None:
        if remaining_k == 0:
            results.append(tuple(sorted(acc)))
            return

        for idx in range(start, len(edges)):
            i, j = edges[idx]
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            acc.append((i, j))
            backtrack(idx + 1, remaining_k - 1, used, acc)
            acc.pop()
            used[i] = used[j] = False

    if k == 0:
        return [()]

    used = [False] * n_vertices
    backtrack(0, k, used, [])
    return results


def _encode_configuration(
    n_vertices: int,
    pairs: tuple[tuple[int, int], ...],
    mono_no3_sites: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Encode a configuration as a hashable tuple.

    labels:
      0 = MeOH
      1 = monodentate NO3
      2 = vertex belonging to a bidentate NO3
    then -1 separator, then flattened sorted pair list.
    """
    labels = [0] * n_vertices

    for v in mono_no3_sites:
        labels[v] = 1

    pair_vertices: set[int] = set()
    for i, j in pairs:
        pair_vertices.add(i)
        pair_vertices.add(j)
    for v in pair_vertices:
        labels[v] = 2

    flat_pairs: list[int] = []
    for i, j in sorted(pairs):
        flat_pairs.extend([i, j])

    return tuple(labels + [-1] + flat_pairs)


def _decode_configuration(
    encoding: tuple[int, ...]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """
    Inverse of _encode_configuration.
    """
    sep_idx = encoding.index(-1)
    labels = encoding[:sep_idx]
    flat_pairs = encoding[sep_idx + 1 :]

    pairs: list[tuple[int, int]] = []
    for i in range(0, len(flat_pairs), 2):
        pairs.append((flat_pairs[i], flat_pairs[i + 1]))

    mono: list[int] = []
    meoh: list[int] = []
    for idx, lab in enumerate(labels):
        if lab == 1:
            mono.append(idx)
        elif lab == 0:
            meoh.append(idx)

    return tuple(sorted(pairs)), tuple(sorted(mono)), tuple(sorted(meoh))


def _canonical_configuration(
    geometry: str,
    n_vertices: int,
    pairs: tuple[tuple[int, int], ...],
    mono_no3_sites: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Canonical representative under the full symmetry group.
    """
    G = symmetry_group_permutations(geometry)
    mono_set = set(mono_no3_sites)
    pair_set = {tuple(sorted(p)) for p in pairs}

    best: tuple[int, ...] | None = None
    for p in G:
        img_pairs: set[tuple[int, int]] = set()
        for i, j in pair_set:
            ii, jj = p[i], p[j]
            if ii > jj:
                ii, jj = jj, ii
            img_pairs.add((ii, jj))

        img_mono = tuple(sorted(p[i] for i in mono_set))
        enc = _encode_configuration(
            n_vertices,
            tuple(sorted(img_pairs)),
            img_mono,
        )
        if best is None or enc < best:
            best = enc

    assert best is not None
    return best


def expected_complex_charge(n_no3_total: int) -> int:
    return 2 - n_no3_total


def compositions_for_cn(cn: int) -> list[tuple[int, int, int]]:
    """
    All triples (n_bi, n_mono, n_meoh) with
        2*n_bi + n_mono + n_meoh = cn
    """
    out: list[tuple[int, int, int]] = []
    for n_bi in range(cn // 2 + 1):
        remaining = cn - 2 * n_bi
        for n_mono in range(remaining + 1):
            n_meoh = remaining - n_mono
            out.append((n_bi, n_mono, n_meoh))
    return out


def build_msmiles_from_configuration(
    encoding: tuple[int, ...],
    geometry: str,
) -> tuple[str, int, int, int, int]:
    """
    Build m-SMILES from an encoded configuration.
    """
    pairs, mono_sites, meoh_sites = _decode_configuration(encoding)

    n_no3_bi = len(pairs)
    n_no3_mono = len(mono_sites)
    n_no3_total = n_no3_bi + n_no3_mono
    n_meoh = len(meoh_sites)

    ligands: list[tuple[str, tuple[int, ...]]] = []
    for i, j in pairs:
        ligands.append(("NO3_bi", (i, j)))
    for i in mono_sites:
        ligands.append(("NO3_mono", (i,)))
    for i in meoh_sites:
        ligands.append(("MeOH", (i,)))

    ligands.sort(key=lambda x: (min(x[1]), len(x[1]), x[0]))

    blocks = [METAL]
    for name, sites in ligands:
        if name == "NO3_bi":
            i, j = sites
            if i > j:
                i, j = j, i
            blocks.append(
                LIGAND_TEMPLATES["NO3_bi"].format(site1=i + 1, site2=j + 1)
            )
        elif name == "NO3_mono":
            (i,) = sites
            blocks.append(LIGAND_TEMPLATES["NO3_mono"].format(site=i + 1))
        elif name == "MeOH":
            (i,) = sites
            blocks.append(LIGAND_TEMPLATES["MeOH"].format(site=i + 1))
        else:
            raise ValueError(f"Unknown ligand name {name!r}")

    blocks.append(geometry)
    msmiles = "|".join(blocks)
    return msmiles, n_no3_total, n_no3_mono, n_no3_bi, n_meoh


def stoichiometry_label(
    n_no3_total: int,
    n_no3_mono: int,
    n_no3_bi: int,
    n_meoh: int,
) -> str:
    del n_no3_mono, n_no3_bi
    return f"{N_ZN}Zn_{n_no3_total}NO3_{n_meoh}MeOH"


def validate_msmiles(msmiles: str, expected_chg: int) -> int:
    """
    Parse with MetalloGen and verify charge and multiplicity.
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
        description="Enumerate Zn(NO3)x(MeOH)y m-SMILES for CN = 4, 5, 6"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("zn_no3_meoh_msmiles_enumeration.tsv"),
        help="TSV output path",
    )
    ap.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip MetalloGen validation",
    )
    args = ap.parse_args()

    planned: list[tuple] = []

    for cn in (4, 5, 6):
        for geom in GEOMETRIES_BY_CN[cn]:
            verts = _metallogen_geometry_vertices()[geom]
            n_vertices = len(verts)
            assert n_vertices == cn, f"{geom}: expected {cn} vertices, got {n_vertices}"

            edges = _geometry_edges(geom)

            for n_bi, n_mono, n_meoh in compositions_for_cn(cn):
                matchings = _edge_matchings(edges, n_vertices, n_bi)
                canonical_configs: set[tuple[int, ...]] = set()

                for pairs in matchings:
                    used_vertices = {v for pair in pairs for v in pair}
                    free_vertices = sorted(set(range(n_vertices)) - used_vertices)

                    mono_choices = [()] if n_mono == 0 else itertools.combinations(
                        free_vertices, n_mono
                    )

                    for mono_sites in mono_choices:
                        enc = _canonical_configuration(
                            geom,
                            n_vertices,
                            tuple(sorted(pairs)),
                            tuple(sorted(mono_sites)),
                        )
                        canonical_configs.add(enc)

                for struct_idx, enc in enumerate(sorted(canonical_configs), start=1):
                    msmiles, n_no3_total, n_no3_mono, n_no3_bi, n_meoh = (
                        build_msmiles_from_configuration(enc, geom)
                    )
                    assert 2 * n_no3_bi + n_no3_mono + n_meoh == cn

                    stoich = stoichiometry_label(
                        n_no3_total, n_no3_mono, n_no3_bi, n_meoh
                    )
                    exp = expected_complex_charge(n_no3_total)

                    planned.append(
                        (
                            cn,
                            geom,
                            n_no3_total,
                            n_no3_mono,
                            n_no3_bi,
                            n_meoh,
                            stoich,
                            msmiles,
                            exp,
                            struct_idx,
                        )
                    )

    charges: list[int] = []
    if args.skip_validate:
        charges = [p[8] for p in planned]
    else:
        failures: list[tuple[str, str, str]] = []
        for (
            _cn,
            geom,
            _n_no3_total,
            _n_no3_mono,
            _n_no3_bi,
            _n_meoh,
            stoich,
            msmiles,
            exp,
            _idx,
        ) in planned:
            try:
                charges.append(validate_msmiles(msmiles, exp))
            except Exception as e:
                failures.append((stoich, geom, f"{type(e).__name__}: {e}"))

        if failures:
            print(
                f"ERROR: {len(failures)} row(s) failed MetalloGen validation.",
                file=sys.stderr,
            )
            for stoich, geom, err in failures[:20]:
                print(f"  {stoich} | {geom} | {err}", file=sys.stderr)
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
            print("TSV was not written.", file=sys.stderr)
            sys.exit(1)

    combined = list(zip(planned, charges))
    combined.sort(
        key=lambda pc: (
            pc[0][2],  # n_NO3_total
            pc[0][3],  # n_NO3_mono
            pc[0][4],  # n_NO3_bi
            pc[0][5],  # n_MeOH
            pc[0][1],  # geometry
            pc[0][9],  # struct_idx
        )
    )

    # formula_id depends only on the stoichiometry string (n_NO3, n_MeOH), not on
    # how nitrate is split between mono- and bidentate. Use a global map so the
    # same formula always gets the same id even if sort does not keep all rows
    # contiguous.
    formula_sort_key: dict[str, tuple[int, int]] = {}
    for p in planned:
        stoich = p[6]
        formula_sort_key[stoich] = (p[2], p[5])  # n_NO3_total, n_MeOH
    unique_formulas = sorted(
        formula_sort_key.keys(),
        key=lambda s: formula_sort_key[s],
    )
    formula_id_by_stoich = {s: i + 1 for i, s in enumerate(unique_formulas)}

    rows: list[dict[str, object]] = []
    isomer_index_by_stoich: dict[str, int] = {}
    row_id = 0

    for (
        cn,
        geom,
        n_no3_total,
        n_no3_mono,
        n_no3_bi,
        n_meoh,
        stoich,
        msmiles,
        _exp,
        _struct_idx,
    ), chg in combined:
        row_id += 1
        isomer_index_by_stoich[stoich] = isomer_index_by_stoich.get(stoich, 0) + 1
        isomer_within_stoich = isomer_index_by_stoich[stoich]

        rows.append(
            {
                "id": row_id,
                "formula_id": formula_id_by_stoich[stoich],
                "formula": stoich,
                "n_NO3": n_no3_total,
                "n_NO3_mono": n_no3_mono,
                "n_NO3_bi": n_no3_bi,
                "n_MeOH": n_meoh,
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
        "n_NO3",
        "n_NO3_mono",
        "n_NO3_bi",
        "n_MeOH",
        "CN",
        "charge",
        "geometry",
        "isomerid",
        "msmiles",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.skip_validate:
        print("Validation was skipped (--skip-validate).")
    else:
        print("All entries passed: MetalloGen parse OK, charge OK, multiplicity=1.")


if __name__ == "__main__":
    main()