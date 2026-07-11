#!/bin/bash
#SBATCH --job-name=mdftb_de_sm
#SBATCH --output=mdftb_scaling_opt_run110/slurm-de-slope-mean-%j.out
#SBATCH --error=mdftb_scaling_opt_run110/slurm-de-slope-mean-%j.err
#SBATCH --time=12:00:00
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
export MDFTB_OPT_RESULTS_DIR="$PWD/mdftb_scaling_opt_run110/results_de_slope_mean"
mkdir -p "$MPLCONFIGDIR" "$MDFTB_OPT_RESULTS_DIR"

if [ -d /scratch ]; then
    export MDFTB_OPT_TMP_DIR="/scratch/${SLURM_JOB_ID}/mdftb_de_slope_mean_tmp"
else
    export MDFTB_OPT_TMP_DIR="$PWD/mdftb_scaling_opt_run110/tmp/${SLURM_JOB_ID}_de_slope_mean"
fi
mkdir -p "$MDFTB_OPT_TMP_DIR"
cleanup() {
    rm -rf "$MDFTB_OPT_TMP_DIR"
}
trap cleanup EXIT

WORKERS="${SLURM_CPUS_PER_TASK:-16}"
DE_MAXITER="${DE_MAXITER:-25}"
DE_POPSIZE="${DE_POPSIZE:-6}"
SLOPE_SCALE="${SLOPE_SCALE:-0.05}"
MEAN_SCALE="${MEAN_SCALE:-0.25}"

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "MODE: de"
echo "OBJECTIVE: slope_mean"
echo "DE_MAXITER: ${DE_MAXITER}"
echo "DE_POPSIZE: ${DE_POPSIZE}"
echo "SLOPE_SCALE: ${SLOPE_SCALE}"
echo "MEAN_SCALE: ${MEAN_SCALE}"
echo "WORKERS: ${WORKERS}"
echo "MDFTB_OPT_RESULTS_DIR: ${MDFTB_OPT_RESULTS_DIR}"
echo "MDFTB_OPT_TMP_DIR: ${MDFTB_OPT_TMP_DIR}"
echo "DFTB+: $(command -v dftb+)"
echo "Python: $(command -v python)"

python mdftb_scaling_opt_run110/optimize_mdftb_scalings.py \
    --mode de \
    --objective slope_mean \
    --slope-scale "$SLOPE_SCALE" \
    --mean-scale "$MEAN_SCALE" \
    --de-maxiter "$DE_MAXITER" \
    --de-popsize "$DE_POPSIZE" \
    --workers "$WORKERS" \
    --threads 1
