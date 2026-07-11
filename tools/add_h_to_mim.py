#!/usr/bin/env python3
"""
Add proton (H) to deprotonated MIm (methylimidazolate) in a Zn complex,
converting MIm back to neutral ImH form.

The H position is determined using:
  1. N-H bond length from ImH_monomer.xyz
  2. Bisector method: H placed along the outward bisector of the C-N-C angle
     at the N not coordinated to Zn

Usage:
    python add_h_to_mim.py <complex.xyz> [--monomer ImH_monomer.xyz] [-o output.xyz]
"""

import numpy as np
import argparse
import os


def read_xyz(filename):
    atoms = []
    with open(filename) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    comment = lines[1].strip()
    for i in range(2, 2 + natoms):
        parts = lines[i].split()
        atoms.append((parts[0], np.array([float(x) for x in parts[1:4]])))
    return atoms, comment


def write_xyz(filename, atoms, comment=""):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for elem, coord in atoms:
            f.write(
                f"  {elem:<2s}  {coord[0]:>20.14f}"
                f"  {coord[1]:>20.14f}"
                f"  {coord[2]:>20.14f}\n"
            )


def dist(a, b):
    return np.linalg.norm(a - b)


def get_nh_bond_length(monomer_atoms):
    for i, (ei, ci) in enumerate(monomer_atoms):
        if ei != "N":
            continue
        for j, (ej, cj) in enumerate(monomer_atoms):
            if ej == "H" and dist(ci, cj) < 1.15:
                return dist(ci, cj)
    raise ValueError("No N-H bond found in monomer")


def find_imidazolate_n_pairs(atoms):
    """
    Find pairs of N atoms in the same imidazole ring.
    Two ring-N atoms always share a common C neighbor (the C2 bridging carbon).
    """
    n_indices = [i for i, (e, _) in enumerate(atoms) if e == "N"]

    n_to_c = {}
    for ni in n_indices:
        n_to_c[ni] = {
            j
            for j, (ej, cj) in enumerate(atoms)
            if ej == "C" and dist(atoms[ni][1], cj) < 1.6
        }

    pairs = []
    used = set()
    for idx_a, na in enumerate(n_indices):
        for nb in n_indices[idx_a + 1 :]:
            if na in used or nb in used:
                continue
            if n_to_c[na] & n_to_c[nb]:
                pairs.append((na, nb))
                used.update([na, nb])
    return pairs


def find_fragment_atoms(atoms, seed_indices, exclude_elems=frozenset({"Zn"})):
    """BFS from seed atoms to find all bonded atoms in the same fragment."""
    fragment = set(seed_indices)
    queue = list(seed_indices)
    while queue:
        cur = queue.pop(0)
        e_cur = atoms[cur][0]
        for j, (ej, cj) in enumerate(atoms):
            if j in fragment or ej in exclude_elems:
                continue
            cutoff = 1.3 if ("H" in (e_cur, ej)) else 1.7
            if dist(atoms[cur][1], cj) < cutoff:
                fragment.add(j)
                queue.append(j)
    return fragment


def n_has_h(atoms, n_idx):
    coord = atoms[n_idx][1]
    return any(ej == "H" and dist(coord, cj) < 1.15 for j, (ej, cj) in enumerate(atoms))


def place_h_bisector(atoms, n_idx, nh_length):
    """Place H along the outward bisector of the C-N-C angle at the given N."""
    n_coord = atoms[n_idx][1]
    c_neighbors = [
        cj
        for (ej, cj) in atoms
        if ej == "C" and dist(n_coord, cj) < 1.6
    ]
    if len(c_neighbors) < 2:
        raise ValueError(f"N at index {n_idx} has {len(c_neighbors)} C neighbors (need >=2)")

    midpoint = (c_neighbors[0] + c_neighbors[1]) / 2.0
    direction = n_coord - midpoint
    direction /= np.linalg.norm(direction)
    return n_coord + nh_length * direction


def add_h_to_mim(complex_file, monomer_file, output_file=None):
    monomer_atoms, _ = read_xyz(monomer_file)
    complex_atoms, comment = read_xyz(complex_file)

    nh_length = get_nh_bond_length(monomer_atoms)
    print(f"N-H bond length from monomer: {nh_length:.4f} Angstrom")

    zn_indices = [i for i, (e, _) in enumerate(complex_atoms) if e == "Zn"]
    if not zn_indices:
        print("WARNING: No Zn found in complex")

    n_pairs = find_imidazolate_n_pairs(complex_atoms)
    print(f"Found {len(n_pairs)} imidazolate fragment(s)")

    if not n_pairs:
        print("No imidazolate fragments found. Nothing to do.")
        return

    insertions = []

    for n1, n2 in n_pairs:
        if zn_indices:
            d1 = min(dist(complex_atoms[n1][1], complex_atoms[zn][1]) for zn in zn_indices)
            d2 = min(dist(complex_atoms[n2][1], complex_atoms[zn][1]) for zn in zn_indices)
        else:
            d1, d2 = float("inf"), float("inf")

        if d1 < d2:
            coord_n, free_n = n1, n2
            d_coord, d_free = d1, d2
        else:
            coord_n, free_n = n2, n1
            d_coord, d_free = d2, d1

        if n_has_h(complex_atoms, free_n):
            print(f"  Fragment N({n1},{n2}): free N idx={free_n} already has H, skipping")
            continue

        h_coord = place_h_bisector(complex_atoms, free_n, nh_length)

        frag_indices = find_fragment_atoms(complex_atoms, [n1, n2])
        insert_after = max(frag_indices)

        insertions.append((insert_after, ("H", h_coord)))

        print(
            f"  Fragment N({n1},{n2}): Zn-bonded N={coord_n} (d_Zn={d_coord:.3f}), "
            f"free N={free_n} (d_Zn={d_free:.3f})"
        )
        print(f"    -> H inserted after atom idx {insert_after} (fragment atoms {min(frag_indices)}-{insert_after})")

    all_atoms = list(complex_atoms)
    for pos, h_atom in sorted(insertions, key=lambda x: x[0], reverse=True):
        all_atoms.insert(pos + 1, h_atom)

    if output_file is None:
        base, ext = os.path.splitext(complex_file)
        output_file = base + "_protonated" + ext

    write_xyz(output_file, all_atoms, comment + " (H added to MIm)")
    print(f"\nOutput: {output_file}")
    print(f"Atoms: {len(complex_atoms)} -> {len(all_atoms)} (+{len(insertions)} H)")


def main():
    parser = argparse.ArgumentParser(description="Add H to deprotonated MIm in Zn complexes")
    parser.add_argument("complex_xyz", help="Input complex xyz file")
    parser.add_argument(
        "--monomer",
        default=None,
        help="ImH monomer xyz file (default: ImH_monomer.xyz in same directory)",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output xyz file (default: <input>_protonated.xyz)"
    )
    args = parser.parse_args()

    if args.monomer is None:
        args.monomer = os.path.join(os.path.dirname(os.path.abspath(args.complex_xyz)), "ImH_monomer.xyz")

    if not os.path.exists(args.monomer):
        print(f"Error: Monomer file not found: {args.monomer}")
        return 1

    add_h_to_mim(args.complex_xyz, args.monomer, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
