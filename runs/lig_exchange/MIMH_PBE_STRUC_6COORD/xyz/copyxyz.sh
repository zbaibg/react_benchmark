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
ADD_H="${SCRIPT_DIR}/add_h_to_mim.py"
rm -rf ${SCRIPT_DIR}/xyz_files
mkdir -p ${SCRIPT_DIR}/xyz_files
cd ${SCRIPT_DIR}/xyz_files
# 4-coord complexes
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_4MIm_0MeOH/cal.xyz 1Zn_4MIm_0MeOH.xyz  # pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_3MIm_1MeOH/cal.xyz 1Zn_3MIm_1MeOH.xyz  # pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_2MIm_2MeOH/cal.xyz 1Zn_2MIm_2MeOH.xyz # pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_1MIm_3MeOH/cal.xyz 1Zn_1MIm_3MeOH.xyz # pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_0MIm_4MeOH/cal.xyz 1Zn_0MIm_4MeOH.xyz # pass

# 5-coord complexes
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_5MIm_0MeOH/cal.xyz 1Zn_5MIm_0MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_4MIm_1MeOH/init_geo.xyz 1Zn_4MIm_1MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_3MIm_2MeOH/init_geo.xyz 1Zn_3MIm_2MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_2MIm_3MeOH/cal.xyz 1Zn_2MIm_3MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_1MIm_4MeOH/init_geo.xyz 1Zn_1MIm_4MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_0MIm_5MeOH/cal.xyz 1Zn_0MIm_5MeOH.xyz

# 6-coord complexes
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_6MIm_0MeOH/init_geo_run1.xyz 1Zn_6MIm_0MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_5MIm_1MeOH/init_geo_run1.xyz 1Zn_5MIm_1MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_4MIm_2MeOH/init_geo_run1.xyz 1Zn_4MIm_2MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_3MIm_3MeOH/init_geo_run1.xyz 1Zn_3MIm_3MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_2MIm_4MeOH/init_geo.xyz 1Zn_2MIm_4MeOH.xyz #Unstable before adding H
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_1MIm_5MeOH/cal.xyz 1Zn_1MIm_5MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_0MIm_6MeOH/cal.xyz 1Zn_0MIm_6MeOH.xyz

# Monomers
#cp "$REPO_ROOT/structures/monomers/MIM.xyz" MIm_monomer.xyz
cp "$REPO_ROOT/structures/monomers/MeOH.xyz" MeOH_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Zn_nbZIFFF.xyz" Zn_monomer.xyz
cp "$REPO_ROOT/structures/monomers/MImH.xyz" MImH_monomer.xyz

for name in 1Zn_*MIm_*MeOH.xyz; do
    outname="${name//MIm/MImH}"
    if [[ "$name" == *'_0MIm_'* ]]; then
        mv "$name" "$outname"
    else
        python "${ADD_H}" "${name}" --monomer MImH_monomer.xyz -o "${outname}"
        rm "${name}"
    fi
done
cd "${SCRIPT_DIR}"