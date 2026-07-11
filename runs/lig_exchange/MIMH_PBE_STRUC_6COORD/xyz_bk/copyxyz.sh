#!/bin/bash
_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"; done
source "$_REPO_CANDIDATE/tools/repo_env.sh"
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADD_H="${SCRIPT_DIR}/add_h_to_mim.py"
rm -rf ${SCRIPT_DIR}/xyz_files
mkdir -p ${SCRIPT_DIR}/xyz_files
cd ${SCRIPT_DIR}/xyz_files
# 4-coord complexes
#1Zn_4MIm_0MeOH.xyz  # pass
#1Zn_3MIm_1MeOH.xyz  # pass
#1Zn_2MIm_2MeOH.xyz # pass
#1Zn_1MIm_3MeOH.xyz # pass
#1Zn_0MIm_4MeOH.xyz # pass
source_dir='$REPO_ROOT/runs/lig_exchange/MIMH_PBE_STRUC_6COORD/qm_minimize/run16'
# 5-coord complexes
cp ${source_dir}/1Zn_5MImH_0MeOH/min.xyz 1Zn_5MImH_0MeOH.xyz
#cp ${source_dir}/1Zn_4MImH_1MeOH/min.xyz 1Zn_4MImH_1MeOH.xyz #Unstable after relaxation
cp ${source_dir}/1Zn_3MImH_2MeOH/min.xyz 1Zn_3MImH_2MeOH.xyz
cp ${source_dir}/1Zn_2MImH_3MeOH/min.xyz 1Zn_2MImH_3MeOH.xyz
#cp ${source_dir}/1Zn_1MImH_4MeOH/min.xyz 1Zn_1MImH_4MeOH.xyz #Unstable after relaxation
cp ${source_dir}/1Zn_0MImH_5MeOH/min.xyz 1Zn_0MImH_5MeOH.xyz

# 6-coord complexes
#cp ${source_dir}/1Zn_6MImH_0MeOH/min.xyz 1Zn_6MImH_0MeOH.xyz #Unstable after relaxation
#cp ${source_dir}/1Zn_5MImH_1MeOH/min.xyz 1Zn_5MImH_1MeOH.xyz #Unstable after relaxation
#cp ${source_dir}/1Zn_4MImH_2MeOH/min.xyz 1Zn_4MImH_2MeOH.xyz #Unstable after relaxation
#cp ${source_dir}/1Zn_3MImH_3MeOH/min.xyz 1Zn_3MImH_3MeOH.xyz #Unstable after relaxation
#cp ${source_dir}/1Zn_2MImH_4MeOH/min.xyz 1Zn_2MImH_4MeOH.xyz #Unstable after relaxation
cp ${source_dir}/1Zn_1MImH_5MeOH/min.xyz 1Zn_1MImH_5MeOH.xyz
cp ${source_dir}/1Zn_0MImH_6MeOH/min.xyz 1Zn_0MImH_6MeOH.xyz

# Monomers
cp ${source_dir}/MeOH_monomer/min.xyz MeOH_monomer.xyz
cp ${source_dir}/Zn_monomer/min.xyz Zn_monomer.xyz
cp ${source_dir}/MImH_monomer/min.xyz MImH_monomer.xyz
