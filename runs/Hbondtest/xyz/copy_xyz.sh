#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and software paths
_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do
  _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"
done
if [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; then
  echo "ERROR: could not locate repo root (software.yaml)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$_REPO_CANDIDATE/tools/repo_env.sh"

src_dir="$REPO_ROOT/structures/metaD"
out_dir="./xyz_files"
imz_atoms=9
wat_atoms=3

mkdir -p "$out_dir"

generated=0

while IFS= read -r -d '' xyz; do
    base="$(basename "$xyz" .xyz)"

    if [[ ! "$base" =~ ^IMZW([0-9]+)(_[0-9]+)?$ ]]; then
        printf 'Skip unexpected file name: %s\n' "$xyz" >&2
        continue
    fi

    expected_waters="${BASH_REMATCH[1]}"

    waters="$(
        awk \
            -v base="$base" \
            -v out_dir="$out_dir" \
            -v imz_atoms="$imz_atoms" \
            -v wat_atoms="$wat_atoms" \
            -v expected_waters="$expected_waters" '
                NR == 1 {
                    natoms = $1
                    next
                }
                NR == 2 {
                    next
                }
                NR > 2 {
                    atom[++natom_lines] = $0
                    next
                }
                END {
                    if (natom_lines != natoms) {
                        printf "Bad atom count in %s: header=%d, lines=%d\n", base, natoms, natom_lines > "/dev/stderr"
                        exit 1
                    }

                    if (natoms < imz_atoms || (natoms - imz_atoms) % wat_atoms != 0) {
                        printf "Cannot split %s: natoms=%d is incompatible with %d imidazole atoms and %d atoms/water\n", base, natoms, imz_atoms, wat_atoms > "/dev/stderr"
                        exit 1
                    }

                    waters = (natoms - imz_atoms) / wat_atoms
                    if (waters != expected_waters) {
                        printf "Water count mismatch in %s: file name says %d, coordinates contain %d\n", base, expected_waters, waters > "/dev/stderr"
                        exit 1
                    }

                    for (wat = 1; wat <= waters; wat++) {
                        out = sprintf("%s/%s_WAT%d.xyz", out_dir, base, wat)
                        print imz_atoms + wat_atoms > out
                        printf "%s_WAT%d\n", base, wat > out

                        for (i = 1; i <= imz_atoms; i++) {
                            print atom[i] > out
                        }

                        start = imz_atoms + (wat - 1) * wat_atoms + 1
                        for (i = start; i < start + wat_atoms; i++) {
                            print atom[i] > out
                        }

                        close(out)
                    }

                    print waters
                }
            ' "$xyz"
    )"

    generated=$((generated + waters))
done < <(find "$src_dir" -maxdepth 1 -type f -name 'IMZW*.xyz' -print0 | sort -z -V)

printf 'Wrote %d imidazole-water dimer xyz files to %s\n' "$generated" "$out_dir"
cp "$REPO_ROOT/runs/iter_struc/extra_test/qm_minimize/run41/ImH_monomer/min.xyz ./xyz_files/ImH_monomer.xyz
cp "$REPO_ROOT/runs/iter_struc/extra_test/qm_minimize/run41/Wat_monomer/min.xyz ./xyz_files/Wat_monomer.xyz