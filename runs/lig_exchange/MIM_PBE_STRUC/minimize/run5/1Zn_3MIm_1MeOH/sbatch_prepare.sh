#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

source "$CONDAINIT"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    export TMPDIR="$SCRATCH_ROOT"/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export TMPDIR="$SCRATCH_ROOT"/${SLURM_JOB_ID}
fi
mkdir -p "$TMPDIR"
BIND_OPTS="--bind /home/$USER --bind $TMPDIR"
[ -d "$SCRATCH_ROOT"/"$USER" ] && BIND_OPTS="$BIND_OPTS --bind "$SCRATCH_ROOT"/$USER"

apptainer exec -e \
  $BIND_OPTS \
  /home/zbai29/JR/soft/apptainer/amber_mdftb.sif bash -c "\
    echo \"HOSTNAME: \$(hostname)\" &&
    export SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} &&
    export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} &&
    export SLURM_JOB_ID=${SLURM_JOB_ID} &&
    export USER=${USER} &&
    source /etc/bash.bashrc && 
    source "$AMBER_SH_LEGACY" &&
    ulimit -s unlimited &&
    export PATH=\"\$PWD:\$PATH:/home/zbai29/JR/soft/orca_6_0_1_linux_x86-64_shared_openmpi416/\" &&
    sander -O -i min.in -o min.out -p box.prmtop -c init.rst -r min.rst -ref init.rst -x min.nc -inf min.info > min.log 2> min.err &&
 cpptraj -p box.prmtop -y min.rst -x min.xyz > xyz_conv.log 2>&1 &&
  rm -f orc_job.gbw orc_job.densities
  
    "
rm -rf "$TMPDIR"
