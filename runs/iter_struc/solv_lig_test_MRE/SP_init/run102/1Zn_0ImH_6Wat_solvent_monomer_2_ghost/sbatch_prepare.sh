#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=pre
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
set -euo pipefail

export inputfilename='run.input'
export LANG=en_US
export PATH="/home/zbai29/soft/molpro_2025_4_1/bin:/sbin:/usr/sbin:/bin:/usr/bin:${PATH}"
export LD_LIBRARY_PATH="/home/zbai29/soft/molpro_2025_4_1/lib:$LD_LIBRARY_PATH"
TEMPLATE="${SLURM_JOB_ID}_XXXXXX"
mkdir -p "/local/zbai29"
SCRATCH_DIR=$(mktemp -d "/local/zbai29/${TEMPLATE}")
rm -rf /tmp/molpro*
cp $inputfilename "${SCRATCH_DIR}/"
(
cd "${SCRATCH_DIR}"
unshare --net --user --map-root-user bash -lc "
ip link set lo up
for var in \$(compgen -v | grep SLURM); do unset \$var; done
export I_MPI_HYDRA_BOOTSTRAP=fork
export HYDRA_BOOTSTRAP=fork
export I_MPI_FABRICS=shm; molpro -d ${SCRATCH_DIR} -n 32 --ga-impl disk -m 183m --stdout $inputfilename > $SLURM_SUBMIT_DIR/molpro.log 2>&1
"
cp -rf ./* "$SLURM_SUBMIT_DIR/"
)
rm -rf "${SCRATCH_DIR}"
