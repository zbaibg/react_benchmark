#!/bin/bash
#SBATCH --job-name=scc_ImH__t3000_broyde
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=0-01:00:00
#SBATCH --partition=pre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

source "/home/zbai29/condainit.sh"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    export TMPDIR="/scratch"/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export TMPDIR="/scratch"/${SLURM_JOB_ID}
fi
mkdir -p "$TMPDIR"
#BIND_OPTS="--bind /home/$USER --bind $TMPDIR"
BIND_OPTS="--bind /home/$USER"
[ -d /scratch ] && BIND_OPTS="$BIND_OPTS --bind /scratch"

apptainer exec -e \
  $BIND_OPTS \
  "/home/zbai29/JR/soft/apptainer/noamber.sif" bash -c "\
    echo \"HOSTNAME: \$(hostname)\" &&
    export SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} &&
    export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} &&
    export SLURM_JOB_ID=${SLURM_JOB_ID} &&
    export USER=${USER} &&
    source /etc/bash.bashrc && 
    ulimit -s unlimited &&
    export PATH=\"\$PWD:\$PATH:"/home/zbai29/JR/soft/orca_6_1_1_linux_x86-64_shared_openmpi418_frag_move"/\" &&
    source "/home/zbai29/soft/ambertools25_dev/d07d0b1_noamber/amber.sh" &&
    sander -O -i min.in -o min.out -p box.prmtop -c init.rst -r min.rst -ref init.rst -x min.nc -inf min.info > min.log 2> min.err &&
 cpptraj -p box.prmtop -y min.rst -x min.xyz > xyz_conv.log 2>&1 &&
  rm -f orc_job.gbw orc_job.densities
  
    "
rm -rf "$TMPDIR"
