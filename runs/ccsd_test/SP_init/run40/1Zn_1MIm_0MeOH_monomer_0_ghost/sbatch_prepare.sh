#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=41G
#SBATCH --exclude=compute-0-[0-40,44]
#SBATCH --exclusive
source ~/condainit.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PWD:$PATH:/home/zbai29/JR/soft/orca_6_1_1_linux_x86-64_shared_openmpi418_frag_move/"
orca orc_job.inp > orc_job.dat 2>&1
rm -f orc_job.gbw orc_job.densities
cp orc_job.xyz min.xyz
