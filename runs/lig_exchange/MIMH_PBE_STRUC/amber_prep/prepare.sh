#!/bin/bash
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

######################## Settings ########################
# Settings for creating IMH topology
FF_IMH=gaff2
CHARGE_METHOD_IMH=abcg2
IMH_xyz="$REPO_ROOT/structures/monomers/ImH.xyz"
charge_IMH=0
#settings regarding creating NO3
FF_NO3=gaff2
CHARGE_METHOD_NO3=abcg2
NO3_xyz="$REPO_ROOT/structures/monomers/NO3.xyz"
charge_NO3=-1
#settings for creating MIM
FF_MIM=gaff2
CHARGE_METHOD_MIM=abcg2
MIM_xyz="$REPO_ROOT/structures/monomers/MIM.xyz"
charge_MIM=-1
#settings for creating MIM
FF_MIH=gaff2
CHARGE_METHOD_MIH=abcg2
MIH_xyz="$REPO_ROOT/structures/monomers/MImH.xyz"
charge_MIH=0
# Helper script paths
zif_meoh_assign_name_py_path="$REPO_ROOT/tools/zif_meoh_assign_name.py"
fix_charges_py_path="$REPO_ROOT/tools/form_charges.py"

####################### End of Settings #####################

function init_amber() {
    source "$CONDAINIT"
    conda activate amber
}

function create_IMH() {
    # IMH anion
    echo "Start to create IMH anion"
    if [ -f IMH.xyz ]; then
        rm IMH.xyz
    fi
    ln -s $IMH_xyz ./IMH.xyz
    if [ ! -f IMH.xyz ]; then echo "ERROR: IMH.xyz not found." && exit 1; fi
    
    python -c "
import MDAnalysis as mda
u = mda.Universe('IMH.xyz')
u.atoms.write('IMH.pdb')
"
    
    python ${zif_meoh_assign_name_py_path} IMH.pdb IMH_fixed.pdb
    pdb4amber -i IMH_fixed.pdb -o IMH_clean.pdb
    
    antechamber -i IMH_clean.pdb -fi pdb \
                -o IMH.mol2 -fo mol2 \
                -c ${CHARGE_METHOD_IMH} -s 2 -nc $charge_IMH -at $FF_IMH -pf y
    
    # Fix charges to be exactly integer
    python ${fix_charges_py_path} IMH.mol2 $charge_IMH
    
    parmchk2 -i IMH.mol2 -f mol2 -o IMH.frcmod -s $FF_IMH
    
    echo "Finished creating IMH mol2 and frcmod"
}
function create_MIH() {
    # MIH anion
    echo "Start to create MIH anion"
    if [ -f MIH.xyz ]; then
        rm MIH.xyz
    fi
    ln -s $MIH_xyz ./MIH.xyz
    if [ ! -f MIH.xyz ]; then echo "ERROR: MIH.xyz not found." && exit 1; fi
    
    python -c "
import MDAnalysis as mda
u = mda.Universe('MIH.xyz')
u.atoms.write('MIH.pdb')
"
    
    python ${zif_meoh_assign_name_py_path} MIH.pdb MIH_fixed.pdb
    pdb4amber -i MIH_fixed.pdb -o MIH_clean.pdb
    
    antechamber -i MIH_clean.pdb -fi pdb \
                -o MIH.mol2 -fo mol2 \
                -c ${CHARGE_METHOD_MIH} -s 2 -nc $charge_MIH -at $FF_MIH -pf y
    
    # Fix charges to be exactly integer
    python ${fix_charges_py_path} MIH.mol2 $charge_MIH
    
    parmchk2 -i MIH.mol2 -f mol2 -o MIH.frcmod -s $FF_MIH
    
    echo "Finished creating MIH mol2 and frcmod"
}
function create_MIM() {
    # MIM anion
    echo "Start to create MIM anion"
    if [ -f MIM.xyz ]; then
        rm MIM.xyz
    fi
    ln -s $MIM_xyz ./MIM.xyz
    if [ ! -f MIM.xyz ]; then echo "ERROR: MIM.xyz not found." && exit 1; fi
    
    python -c "
import MDAnalysis as mda
u = mda.Universe('MIM.xyz')
u.atoms.write('MIM.pdb')
"
    
    python ${zif_meoh_assign_name_py_path} MIM.pdb MIM_fixed.pdb
    pdb4amber -i MIM_fixed.pdb -o MIM_clean.pdb
    
    antechamber -i MIM_clean.pdb -fi pdb \
                -o MIM.mol2 -fo mol2 \
                -c ${CHARGE_METHOD_MIM} -s 2 -nc $charge_MIM -at $FF_MIM -pf y
    
    # Fix charges to be exactly integer
    python ${fix_charges_py_path} MIM.mol2 $charge_MIM
    
    parmchk2 -i MIM.mol2 -f mol2 -o MIM.frcmod -s $FF_MIM
    
    echo "Finished creating MIM mol2 and frcmod"
}
function create_NO3(){
    echo "Start to create NO3"
    if [ -f NO3.xyz ]; then
        rm NO3.xyz
    fi
    ln -s $NO3_xyz ./NO3.xyz
    if [ ! -f NO3.xyz ]; then echo "ERROR: NO3.xyz not found." && exit 1; fi
    python -c "
import MDAnalysis as mda
u = mda.Universe('NO3.xyz')
if not hasattr(u.atoms, 'resnames'):
    u.add_TopologyAttr('resnames')
new_res=u.add_Residue(resname='NO3',resid=1,resnum=1,icode='')
u.atoms.residues=new_res
O_index = 1
N_index = 1
for i,atom in enumerate(u.atoms):
    if atom.element == 'O':
        atom.name = 'O' + str(O_index)
        O_index += 1
    elif atom.element == 'N':
        atom.name = 'N' + str(N_index)
        N_index += 1
u.atoms.write('NO3.pdb')
"
    python ${zif_meoh_assign_name_py_path} NO3.pdb NO3_fixed.pdb
    pdb4amber -i NO3_fixed.pdb -o NO3_clean.pdb
    antechamber -i NO3_clean.pdb -fi pdb \
        -o NO3.mol2 -fo mol2 \
        -c $CHARGE_METHOD_NO3 -s 2 -nc $charge_NO3 -at $FF_NO3 -pf y
    # Fix charges to be exactly integer
    python ${fix_charges_py_path} NO3.mol2 $charge_NO3
    
    parmchk2 -i NO3.mol2 -f mol2 -o NO3.frcmod -s $FF_NO3
    echo "Finished creating NO3 mol2 and frcmod"

}

# Main execution
init_amber
create_IMH
create_MIH
create_MIM
create_NO3
echo "All done! Generated IMH.mol2, IMH.frcmod, MIM.mol2, MIM.frcmod, NO3.mol2 and NO3.frcmod"
