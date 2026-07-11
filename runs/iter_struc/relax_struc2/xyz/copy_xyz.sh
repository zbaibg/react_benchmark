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

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/xyz_files"
mkdir -p "$OUTDIR"

# run35_all/<dir>/min.xyz -> xyz_files: monomers -> <name>monomer.xyz; complexes (dir name 1Zn*) -> <dir>.xyz
MON_SRC=""$REPO_ROOT/runs/iter_struc/relax_struc/qm_minimize/run35_all"
MONOMERS=(MeOH H MIm MImH Zn)
is_monomer() {
	local n=$1 m
	for m in "${MONOMERS[@]}"; do
		[[ "$m" == "$n" ]] && return 0
	done
	return 1
}
n_copied=0
for dir in "$MON_SRC"/*/; do
	name=$(basename "$dir")
	src="${dir%/}/min.xyz"
	[[ -f "$src" ]] || continue
	if is_monomer "$name"; then
		cp "$src" "${OUTDIR}/${name}_monomer.xyz"
	elif [[ "$name" == 1Zn* ]]; then
		cp "$src" "${OUTDIR}/${name}.xyz"
	else
		continue
	fi
	n_copied=$((n_copied + 1))
done
cp manual_MeOH2.xyz "${OUTDIR}/MeOH2_monomer.xyz"
cp manual_MImH2.xyz "${OUTDIR}/MImH2_monomer.xyz"
echo "Copied ${n_copied} min.xyz from ${MON_SRC} -> ${OUTDIR} (monomers: *monomer.xyz; complexes: 1Zn_*.xyz)"