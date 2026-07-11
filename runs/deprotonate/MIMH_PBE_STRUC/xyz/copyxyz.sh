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

# Only keep MImH and MIm
cp "$REPO_ROOT/structures/monomers/MImH.xyz" MImH_monomer.xyz
cp "$REPO_ROOT/structures/monomers/MIM.xyz" MIm_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Wat.xyz" Wat_monomer.xyz
cp ../Structure2D_COMPOUND_CID_123332_H3O+.xyz H3O_monomer.xyz
cp "$REPO_ROOT/runs/lig_exchange/MIMH_PBE_STRUC/qm_minimize/run16/1Zn_1MImH_3MeOH/min.xyz 1Zn_1MImH_3MeOH.xyz
cp "$REPO_ROOT/runs/lig_exchange/MIM_PBE_STRUC/qm_minimize/run16/1Zn_1MIm_3MeOH/min.xyz 1Zn_1MIm_3MeOH.xyz

# Additionally, manually write a xyz file for the H ion
cat > H_monomer.xyz <<EOF
1
H+
H     0.000000      0.000000      0.000000
EOF