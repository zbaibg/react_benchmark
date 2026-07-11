#!/bin/bash
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${SCRIPT_DIR}/xyz_files"
cd "${SCRIPT_DIR}/xyz_files"

ADD_H="${SCRIPT_DIR}/add_h_to_mim.py"
SRC_4COORD=""$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP"
SRC_MONO=""$REPO_ROOT/structures/nbZIFFF-km/SingleNode_PBE_TZVP"

# Monomers
cp "$REPO_ROOT/structures/monomers/MImH.xyz" MImH_monomer.xyz
cp "${SRC_MONO}/0Zn_0MIm_1MeOH/cal.xyz" MeOH_monomer.xyz
cp "${SRC_MONO}/1Zn_0MIm_0MeOH/cal.xyz" Zn_monomer.xyz

# 4-coord complexes containing MIm: copy, protonate (MIm -> MImH), rename
for name in 1Zn_4MIm_0MeOH 1Zn_3MIm_1MeOH 1Zn_2MIm_2MeOH 1Zn_1MIm_3MeOH; do
    outname="${name//MIm/MImH}"
    cp "${SRC_4COORD}/${name}/cal.xyz" "${name}.xyz"
    python "${ADD_H}" "${name}.xyz" --monomer MImH_monomer.xyz -o "${outname}.xyz"
    rm "${name}.xyz"
done

for name in 1Zn_0MIm_4MeOH; do
    outname="${name//MIm/MImH}"
    cp "${SRC_4COORD}/${name}/cal.xyz" "${outname}.xyz"
done