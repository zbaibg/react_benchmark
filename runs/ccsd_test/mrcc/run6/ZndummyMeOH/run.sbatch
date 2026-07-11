#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=sbatch.log
#SBATCH --error=sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8448M
#SBATCH --exclude=compute-0-[0-40,44]

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export PATH="/home/zbai29/soft/mrcc_2022_03_18:$PATH"
TEMPLATE="${SLURM_JOB_ID}_XXXXXX"
SCRATCH_DIR=$(mktemp -d "/scratch/${TEMPLATE}" 2>&1)
cp GENBAS MINP "${SCRATCH_DIR}/"

export I_MPI_SPAWN=on

(
cd "${SCRATCH_DIR}"
dmrcc > $SLURM_SUBMIT_DIR/out 2>&1
#cp -rf ./* "$SLURM_SUBMIT_DIR/"
)
rm -rf "${SCRATCH_DIR}"