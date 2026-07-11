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

# 4-coord complexes
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_4MIm_0MeOH/cal.xyz xyz_files/1Zn_4MIm_0MeOH.xyz #Stable but pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_3MIm_1MeOH/cal.xyz xyz_files/1Zn_3MIm_1MeOH.xyz #Stable but pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_2MIm_2MeOH/cal.xyz xyz_files/1Zn_2MIm_2MeOH.xyz #Stable but pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_1MIm_3MeOH/cal.xyz xyz_files/1Zn_1MIm_3MeOH.xyz #Stable but pass
#cp "$REPO_ROOT/structures/nbZIFFF-km/4coord_uff_PBE_TZVP/1Zn_0MIm_4MeOH/cal.xyz xyz_files/1Zn_0MIm_4MeOH.xyz #Stable but pass

# 5-coord complexes
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_5MIm_0MeOH/cal.xyz xyz_files/1Zn_5MIm_0MeOH.xyz  
#Stable by orca- with PBE-TZVP, but will be unstable after dl-find relaxation of qm_minimize/run16/ (PBE-D3(BJ) def2-TZVPPD), So I manually used the same file as min.xyz in qm_minimize/run16/1Zn_5MIm_0MeOH/

#cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_4MIm_1MeOH/cal.xyz xyz_files/1Zn_4MIm_1MeOH.xyz #Unstable
#cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_3MIm_2MeOH/cal.xyz xyz_files/1Zn_3MIm_2MeOH.xyz #Unstable
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_2MIm_3MeOH/cal.xyz xyz_files/1Zn_2MIm_3MeOH.xyz
#cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_1MIm_4MeOH/cal.xyz xyz_files/1Zn_1MIm_4MeOH.xyz #Unstable
cp "$REPO_ROOT/structures/nbZIFFF-km/5coord_uff_PBE_TZVP/1Zn_0MIm_5MeOH/cal.xyz xyz_files/1Zn_0MIm_5MeOH.xyz

# 6-coord complexes
#cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_6MIm_0MeOH/cal.xyz xyz_files/1Zn_6MIm_0MeOH.xyz #Unstable
#cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_5MIm_1MeOH/cal.xyz xyz_files/1Zn_5MIm_1MeOH.xyz #Unstable
#cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_4MIm_2MeOH/cal.xyz xyz_files/1Zn_4MIm_2MeOH.xyz #Unstable
#cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_3MIm_3MeOH/cal.xyz xyz_files/1Zn_3MIm_3MeOH.xyz #Unstable
#cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_2MIm_4MeOH/cal.xyz xyz_files/1Zn_2MIm_4MeOH.xyz #Unstable
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_1MIm_5MeOH/cal.xyz xyz_files/1Zn_1MIm_5MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/6coord_uff_PBE_TZVP/1Zn_0MIm_6MeOH/cal.xyz xyz_files/1Zn_0MIm_6MeOH.xyz

# Monomers
cp "$REPO_ROOT/structures/monomers/MIM.xyz" xyz_files/MIm_monomer.xyz
cp "$REPO_ROOT/structures/monomers/MeOH.xyz" xyz_files/MeOH_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Zn_nbZIFFF.xyz" xyz_files/Zn_monomer.xyz