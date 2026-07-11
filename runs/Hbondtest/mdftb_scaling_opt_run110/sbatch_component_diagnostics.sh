#!/bin/bash
#SBATCH --job-name=mdftb_comp_diag
#SBATCH --output=mdftb_scaling_opt_run110/slurm-component-%j.out
#SBATCH --error=mdftb_scaling_opt_run110/slurm-component-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000M
#SBATCH --exclude=compute-0-[0-40,44]

set -euo pipefail

_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do
  _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"
done
source "$_REPO_CANDIDATE/tools/repo_env.sh"


cd "$SLURM_SUBMIT_DIR"

source "$CONDAINIT"
conda activate mybase

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/mdftb_scaling_opt_run110/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "DFTB+: $(command -v dftb+)"
echo "Python: $(command -v python)"

python mdftb_scaling_opt_run110/component_diagnostics.py \
    --run-slope \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --threads 1
