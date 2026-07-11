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

mkdir -p xyz_files
cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_4MIm_0MeOH/cal.xyz xyz_files/1Zn_4MIm_0MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_3MIm_1MeOH/cal.xyz xyz_files/1Zn_3MIm_1MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_2MIm_2MeOH/cal.xyz xyz_files/1Zn_2MIm_2MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_1MIm_3MeOH/cal.xyz xyz_files/1Zn_1MIm_3MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_0MIm_4MeOH/cal.xyz xyz_files/1Zn_0MIm_4MeOH.xyz
cp "$REPO_ROOT/structures/monomers/MIM.xyz" xyz_files/MIm_monomer.xyz
cp "$REPO_ROOT/structures/monomers/MeOH.xyz" xyz_files/MeOH_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Zn_nbZIFFF.xyz" xyz_files/Zn_monomer.xyz