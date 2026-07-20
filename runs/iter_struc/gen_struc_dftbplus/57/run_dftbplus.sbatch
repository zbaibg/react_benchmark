#!/bin/bash
#SBATCH --job-name=mg_dftb_57
#SBATCH --output=run_metallogen.log
#SBATCH --error=run_metallogen.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/condainit.sh"
conda activate metallogen
# DFTB+ binary must be on PATH, or set e.g. export DFTBPLUS_COMMAND="/path/to/dftb+"

DFTB_TEMPLATE="/home/zbai29/data/qmmm_test/react_benchmark/deprotonate/MIMH_PBE_STRUC/minimize/run6/1Zn_1MImH_3MeOH/dftb_pin.hsd"
ID="57"
MSMILES='[Zn+2]|Cc1ncc[n-:1]1|Cc1[n:2]cc[nH]1|C[OH:3]|C[OH:4]|Cc1[n:5]cc[nH]1|5_square_pyramidal'
WORKDIR="${SLURM_SUBMIT_DIR}/work_dir"
SAVEDIR="${SLURM_SUBMIT_DIR}/save_dir"
mkdir -p "${WORKDIR}" "${SAVEDIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START id=${ID} ${MSMILES}"
{
  echo "=== id=${ID} metallogen msmiles: ${MSMILES}"
  export PYTHONUNBUFFERED=1
  metallogen \
    -s "${MSMILES}" \
    -wd "${WORKDIR}" \
    -sd "${SAVEDIR}" \
    -r 1 \
    -nc 20 \
    --skip_failed_clean \
    --no_qc_clean_lig_broken_bond \
    --always_qc \
    -c dftbplus \
    --dftbplus_template "${DFTB_TEMPLATE}"
} > metallogen.log 2>&1
rc=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END id=${ID} exit=${rc}"
exit "${rc}"
