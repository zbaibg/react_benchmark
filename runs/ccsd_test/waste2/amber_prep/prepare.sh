#!/bin/bash
set -euo pipefail

######################## Settings ########################
# Per-molecule: xyz path, charge, force field, charge method
# Format: MOL_xyz[name], MOL_charge[name], MOL_ff[name], MOL_cm[name]
declare -A MOL_xyz MOL_charge MOL_ff MOL_cm

MOL_xyz[IMH]=/home/zbai29/data/qmmm_test/coord_ImH/SingleNode_PBE_TZVP/0Zn_1ImH_0Wat/cal.xyz
MOL_charge[IMH]=0
MOL_ff[IMH]=gaff2
MOL_cm[IMH]=abcg2

MOL_xyz[NO3]=/home/zbai29/JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/0Zn_0MIm_0MeOH_1N/cal.xyz
MOL_charge[NO3]=-1
MOL_ff[NO3]=gaff2
MOL_cm[NO3]=abcg2

MOL_xyz[MIM]=/home/zbai29/JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/0Zn_1MIm_0MeOH/cal.xyz
MOL_charge[MIM]=-1
MOL_ff[MIM]=gaff2
MOL_cm[MIM]=abcg2

MOL_xyz[MIH]=/home/zbai29/data/qmmm_test/coord_MImH/SingleNode_PBE_TZVP/MImH/cal.xyz
MOL_charge[MIH]=0
MOL_ff[MIH]=gaff2
MOL_cm[MIH]=abcg2

# Molecule names: auto-inferred from MOL_xyz keys
MOL_NAMES=("${!MOL_xyz[@]}")

# Helper script paths
zif_meoh_assign_name_py_path="/home/zbai29/JR/data/qmmm_test/zif_meoh_assign_name.py"
fix_charges_py_path="/home/zbai29/data/qmmm_test/metaD_ZnImH_SPCE/EndPointCorrection/IMH_WAT_cluster_bulk/fix_charges.py"
xyz_to_radical_py_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/xyz_to_radical_lib.py"
RADICAL_XYZ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../xyz/xyz_files" && pwd)"

####################### End of Settings #####################

function init_amber() {
    source ~/condainit.sh
    conda activate amber
}

# -----------------------------------------------------------------------------
# Create unsupported_mol/ion .lib from XYZ via xyz_to_radical_lib.py
# Geometry and bonds from XYZ; force constants placeholder (QM used later).
# Usage: create_unsupported_mol_from_xyz RESNAME XYZ_PATH CHARGE
# -----------------------------------------------------------------------------
function create_unsupported_mol_from_xyz() {
    local resname="$1"
    local xyz_path="$2"
    local charge="${3:-0}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    if [ ! -f "$xyz_path" ]; then
        echo "ERROR: XYZ not found: $xyz_path" && exit 1
    fi
    python "${xyz_to_radical_py_path}" "$xyz_path" -r "$resname" -q "$charge" -o "."
    if [ -f "${resname}.lib" ]; then
        echo "Finished creating ${resname}.lib"
    else
        echo "ERROR: ${resname}.lib was not created." && exit 1
    fi
}
# Generic function: create_mol NAME
# NAME must exist in MOL_xyz, MOL_charge, MOL_ff, MOL_cm associative arrays
# Residue/atom names (including NO3) are assigned by zif_meoh_assign_name.py
function create_mol() {
    local name=$1
    local xyz="${MOL_xyz[$name]}"
    local charge="${MOL_charge[$name]}"
    local ff="${MOL_ff[$name]}"
    local charge_method="${MOL_cm[$name]}"

    echo "Start to create ${name}"
    [ -f "${name}.xyz" ] && rm "${name}.xyz"
    ln -s "$xyz" "./${name}.xyz"
    if [ ! -f "${name}.xyz" ]; then echo "ERROR: ${name}.xyz not found." && exit 1; fi

    python -c "
import MDAnalysis as mda
u = mda.Universe('${name}.xyz')
u.atoms.write('${name}.pdb')
"

    python "${zif_meoh_assign_name_py_path}" "${name}.pdb" "${name}_fixed.pdb"
    pdb4amber -i "${name}_fixed.pdb" -o "${name}_clean.pdb"

    antechamber -i "${name}_clean.pdb" -fi pdb \
                -o "${name}.mol2" -fo mol2 \
                -c "${charge_method}" -s 2 -nc "${charge}" -at "${ff}" -pf y

    python "${fix_charges_py_path}" "${name}.mol2" "${charge}"
    parmchk2 -i "${name}.mol2" -f mol2 -o "${name}.frcmod" -s "${ff}"

    echo "Finished creating ${name} mol2 and frcmod"
}

# Main execution
init_amber
for name in "${MOL_NAMES[@]}"; do
    create_mol "$name"
done
create_unsupported_mol_from_xyz H   "${RADICAL_XYZ_DIR}/H_monomer.xyz"   1
create_unsupported_mol_from_xyz H3O "${RADICAL_XYZ_DIR}/H3O_monomer.xyz" 1
echo "All done! Generated mol2, frcmod, H.lib and H3O.lib for: ${MOL_NAMES[*]}"
