#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare.log
#SBATCH --error=prepare.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
set -eo pipefail

######################## Settings ########################
#### Settings regarding job submission
node_partition="batch"
core_number_to_apply=1
mem_per_cpu=6000M

##### Clean and initialization
clean_before_prepare=false
SKIP_PRMTOP_HARMONICRST_NDX_GENERATION=false

##### Settings regarding creating MOL topology
MOL_xyz=/home/zbai29/data/qmmm_test/react_benchmark/Hbondtest/xyz/xyz_files/IMZW6_1_WAT3.xyz
RST_SOURCE_DIR=
PRMTOP_SOURCE_DIR=
charge_MOL=0
MOL_mask=":IMH"

###### Sander settings
cut=99999
qmcut=$cut
qm_ewald=1
kmaxq=8
ksqmaxq=100
nfft=auto
process_ls=("min")
min_maxcyc=1
use_dlfind_for_min=true

##### Scripts (expand_nh_oh_radius / delete_wrong_bonds: filled from run_configs.yaml by generate_workflow)
zif_meoh_assign_name_py_path="/home/zbai29/data/qmmm_test/zif_meoh_assign_name.py"
expand_nh_oh_bond_radius=false
delete_wrong_bonds=true

declare -A open_qm
open_qm["min"]=true

####################### End of Settings #####################

function create_MOL() {
    echo "Start to create MOL"
    if [ -f MOL.xyz ]; then
        rm MOL.xyz
    fi
    ln -s "$MOL_xyz" ./MOL.xyz
    if [ ! -f MOL.xyz ]; then echo "ERROR: MOL.xyz not found." && exit 1; fi
    
    python -c "
import MDAnalysis as mda
u = mda.Universe('MOL.xyz')
u.atoms.write('MOL.pdb')
"
    zif_flags=()
    if [ "$expand_nh_oh_bond_radius" = true ]; then
        zif_flags+=(--expand-nh-oh-bond-radius)
    fi
    if [ "$delete_wrong_bonds" = true ]; then
        zif_flags+=(--delete-wrong-bonds)
    fi
    python ${zif_meoh_assign_name_py_path} MOL.pdb MOL_fixed.pdb "${zif_flags[@]}"
    pdb4amber -i MOL_fixed.pdb -o MOL_clean.pdb

    # Path to amber_prep (relative to job dir, e.g. run0/H_monomer -> ../../../amber_prep)
    AMPREP="../../../amber_prep"
    cat >tleap.in <<EOF
source leaprc.protein.ff14SB #For amber's native meoh
source leaprc.water.spce
source leaprc.gaff2
# H+, H3O+ from xyz_to_radical_lib.py (load frcmod before lib)
loadamberparams ${AMPREP}/frcmod.H
loadoff ${AMPREP}/H.lib
#loadamberparams ${AMPREP}/frcmod.H3O
#loadoff ${AMPREP}/H3O.lib
#loadamberparams ${AMPREP}/frcmod.MO+
#loadoff ${AMPREP}/MO+.lib
#loadamberparams ${AMPREP}/frcmod.MI+
#loadoff ${AMPREP}/MI+.lib
IMH=loadmol2 ${AMPREP}/IMH.mol2
loadamberparams ${AMPREP}/IMH.frcmod
IM-=loadmol2 ${AMPREP}/IM-.mol2
loadamberparams ${AMPREP}/IM-.frcmod
MIH=loadmol2 ${AMPREP}/MIH.mol2
loadamberparams ${AMPREP}/MIH.frcmod
NO3=loadmol2 ${AMPREP}/NO3.mol2
loadamberparams ${AMPREP}/NO3.frcmod
MIM=loadmol2 ${AMPREP}/MIM.mol2
loadamberparams ${AMPREP}/MIM.frcmod
loadAmberParams frcmod.meoh #get methanol parameters
loadAmberPrep meoh.in #load meoh geometry and charges (MOH)

box_gas=loadpdb MOL_clean.pdb
saveamberparm box_gas box_orig.prmtop box_orig.inpcrd
savemol2 box_gas box_orig.mol2 0
quit
EOF

    tleap -f tleap.in
    if [ ! -f box_orig.prmtop ]; then echo "ERROR:tleap failed in $PWD" && exit 1; fi
    
    # Prepare for stripping/usage
    cp box_orig.inpcrd min_orig.rst
}

function get_stripped_topology() {
    local CURRENT_JOB_NAME=$(basename "$PWD")

    if [[ "$CURRENT_JOB_NAME" =~ _ghost$ ]]; then
         # Ghost structures: keep full system for BSSE counterpoise
         echo "Generating ghost structure (full system kept for BSSE)"
         cp box_orig.prmtop box.prmtop
         cp min_orig.rst init.rst

    elif [[ "$CURRENT_JOB_NAME" =~ _(monomer|dimer|trimer)_([0-9_]+)$ ]]; then
         # Generic handler for monomer/dimer/trimer: extract all molecule IDs,
         # convert to 1-based residue IDs, build cpptraj strip mask.
         local frag_type="${BASH_REMATCH[1]}"
         local id_string="${BASH_REMATCH[2]}"
         local ids=()
         IFS='_' read -ra ids <<< "$id_string"

         local mask_parts=()
         for id in "${ids[@]}"; do
             mask_parts+=(":$((id + 1))")
         done
         local mask
         mask=$(IFS='|'; echo "${mask_parts[*]}")

         echo "Stripping to ${frag_type} (residues: ${mask})"
         cat > strip.in <<EOF
parm box_orig.prmtop
trajin min_orig.rst
strip !(${mask}) parmout box.prmtop
trajout init.rst restart
run
EOF
         cpptraj -i strip.in > strip.log

    
    elif [[ "$CURRENT_JOB_NAME" == "24wat" ]]; then
         echo "Generating partial structure for 24wat (Keep :2|:3|:4|:5|:6|:7|:8|:9|:10|:11|:12|:13|:14|:15|:16|:17|:18|:19|:20|:21|:22|:23|:24|:25)"
         cat > strip.in <<EOF
parm box_orig.prmtop
trajin min_orig.rst
strip !(:2|:3|:4|:5|:6|:7|:8|:9|:10|:11|:12|:13|:14|:15|:16|:17|:18|:19|:20|:21|:22|:23|:24|:25) parmout box.prmtop
trajout init.rst restart
run
EOF
         cpptraj -i strip.in > strip.log


    else
         echo "Using full system (no stripping detected from job name)"
         cp box_orig.prmtop box.prmtop
         cp min_orig.rst init.rst
    fi

    if [ ! -f box.prmtop ] || [ ! -f init.rst ]; then
         echo "ERROR: Stripping/Copying failed for $CURRENT_JOB_NAME"
         exit 1
    fi
}

function init_amber() {
    source ~/condainit.sh
    conda activate amber
}

function create_input_files(){
    qmmm_cntrl_string=\
"  ifqnt=1,           ! QM/MM calculation
"

    qmmm_non_cntrl_string_non_adaptive_qmmm=\
"&qmmm
  qm_theory='DFTBPLUS',
  qmcut=${qmcut}, 
  !writepdb=1,
  qmmm_int=1, 
  qm_ewald=${qm_ewald}, 
  !verbosity=3,
  kmaxqx=${kmaxq},
  kmaxqy=${kmaxq},
  kmaxqz=${kmaxq},
  ksqmaxq=${ksqmaxq},
  qmmask='${MOL_mask}', qmcharge=${charge_MOL},
/
&dftbplus
  qm_level = 'DFTB3-D3', 
  tfermi = 0, 
  scftol = 1.d-7, 
  maxiter = 250, 
  hcorrection = 0, 
  silent = F, 
  debug = T, 
  mixer = \"BROYDEN\" 
  mdftb= .true.,
  mdftb_scale= .true.,
  skroot= '/home/zbai29/data/qmmm_test/react_benchmark/refit/_repopt/run59/optimized_skf',
/
&xtb
  qm_level='DFTB3-D3',
  maxiter=250,
  mmhardness=0,
/
&orc
  use_template=1,
/
" 
    if [ "$nfft" != "auto" ]; then
        ewald_string=\
"&ewald
  !verbose=3,
  nfft1=${nfft}, nfft2=${nfft}, nfft3=${nfft},
/
"
    else
        ewald_string=\
"&ewald
  !verbose=3,
/
"
    fi
    
    declare -A cntrl_string
    if [ "${use_dlfind_for_min}" = true ]; then
        cntrl_string["min"]=\
"  imin=1,        ! Perform energy minimization
  ntmin=5, ! use DL-Find module for minimization.
  ntc=1, ! do not use SHAKE. Suggested by DL-Find setting
  ntf=1, ! calculate all forces. Suggested by DL-Find setting
  !ncyc=10,      ! Use steepest descent for the first 10 steps, then conjugate gradient
  cut=${cut},       ! (Default: 8.0) Non-bonded interaction cutoff distance (in Å).
  ntb=0,         ! Periodic boundary conditions, constant volume
"
    else
        cntrl_string["min"]=\
"  imin=1,        ! Perform energy minimization
  maxcyc=${min_maxcyc},   ! Maximum number of minimization steps
  !ncyc=10,      ! Use steepest descent for the first 10 steps, then conjugate gradient
  cut=${cut},       ! (Default: 8.0) Non-bonded interaction cutoff distance (in Å).
  ntb=0,         ! Periodic boundary conditions, constant volume
"
    fi
    
    dlfind_string=\
"&dlfind
 ! use the default method (LBFGS minimization)
 maxcycle=${min_maxcyc},
 tol=4.5d-4 ! maximum gradient component in a.u. (default: 4.5d-4)
 tole=5.0d-6 ! maximum predicted energy change in a.u. (default: 1.0d-6)
 !optalg='NR'
 !hessupd='BFGS'
 crdrep='DLC'
/
"

    for name in "${process_ls[@]}"; do
        local_qmmm_non_cntrl_string=$qmmm_non_cntrl_string_non_adaptive_qmmm
        cat >"${name}.in" <<EOF
####
&cntrl
${cntrl_string[$name]}
$(if [ "${open_qm[$name]}" = true ]; then echo "$qmmm_cntrl_string"; fi)
/
$ewald_string
$(if [ "${open_qm[$name]}" = true ]; then echo "$local_qmmm_non_cntrl_string"; fi)
$(if [ "${name}" = "min" ] && [ "${use_dlfind_for_min}" = true ]; then echo "$dlfind_string"; fi)
EOF
        sed -i '/^[[:space:]]*$/d' "${name}.in"
    done
}

function create_sbatch() {
  local jobname=$1
  local run_commands=$2
    cat <<EOF >"sbatch_${jobname}.sh"
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --output=${jobname}_sbatch.log
#SBATCH --error=${jobname}_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=$node_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${core_number_to_apply}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --exclude=compute-0-[0-36]

source ~/condainit.sh

if [ -n "\$SLURM_ARRAY_JOB_ID" ]; then
    export TMPDIR=/scratch/\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}
else
    export TMPDIR=/scratch/\${SLURM_JOB_ID}
fi
mkdir -p "\$TMPDIR"
#BIND_OPTS="--bind /home/\$USER --bind \$TMPDIR"
BIND_OPTS="--bind /home/\$USER"
[ -d /scratch ] && BIND_OPTS="\$BIND_OPTS --bind /scratch"

apptainer exec -e \\
  \$BIND_OPTS \\
  /home/zbai29/JR/soft/apptainer/amber_mdftb_fix.sif bash -c "\\
    echo \\"HOSTNAME: \\\$(hostname)\\" &&
    export SLURM_ARRAY_JOB_ID=\${SLURM_ARRAY_JOB_ID} &&
    export SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID} &&
    export SLURM_JOB_ID=\${SLURM_JOB_ID} &&
    export USER=\${USER} &&
    source /etc/bash.bashrc && 
    ulimit -s unlimited &&
    export PATH=\\"\\\$PWD:\\\$PATH:/home/zbai29/JR/soft/orca_6_1_1_linux_x86-64_shared_openmpi418_frag_move/\\" &&
    $run_commands
    "
rm -rf "\$TMPDIR"
EOF
}

function main() {
    if [ "${clean_before_prepare}" = true ]; then
        rm -rf */
        find ! -name prepare.sh ! -name notes.yaml ! -name prepare.log -delete
    fi

    init_amber

    if [ "${SKIP_PRMTOP_HARMONICRST_NDX_GENERATION}" != true ]; then
        create_MOL
    else
        if [ -d "$RST_SOURCE_DIR" ]; then
            echo "Copying box.prmtop and min.rst from $RST_SOURCE_DIR"

            if [ -f "$RST_SOURCE_DIR/box.prmtop" ]; then
                 cp "$RST_SOURCE_DIR/box.prmtop" ./box_orig.prmtop
            elif [ -f "$PRMTOP_SOURCE_DIR/box.prmtop" ]; then
                 cp "$PRMTOP_SOURCE_DIR/box.prmtop" ./box_orig.prmtop
            else
                 echo "ERROR: box.prmtop not found in $RST_SOURCE_DIR or $PRMTOP_SOURCE_DIR"
                 exit 1
            fi

            if [ -f "$RST_SOURCE_DIR/min.rst" ]; then
                cp "$RST_SOURCE_DIR/min.rst" ./min_orig.rst
            else
                echo "Warning: min.rst not found in $RST_SOURCE_DIR. Trying to find any .rst"
                exit 1
            fi
        else
            echo "ERROR: RST_SOURCE_DIR $RST_SOURCE_DIR not found"
            exit 1
        fi
    fi

    get_stripped_topology

    local prmtop_file="box.prmtop"
    create_input_files

    local input_coord ref_coord
    if [ -f "init.rst" ]; then
        input_coord="init.rst"
        ref_coord="init.rst"
    else
        input_coord="box.inpcrd"
        ref_coord="box.inpcrd"
    fi

    local run_command_strings="sander -O -i min.in -o min.out -p ${prmtop_file} -c ${input_coord} -r min.rst -ref ${ref_coord} -x min.nc -inf min.info > min.log 2> min.err &&
 cpptraj -p ${prmtop_file} -y min.rst -x min.xyz > xyz_conv.log 2>&1 &&
  rm -f orc_job.gbw orc_job.densities
  "

    create_sbatch "prepare" "$run_command_strings"
    echo "all done"
}

main
