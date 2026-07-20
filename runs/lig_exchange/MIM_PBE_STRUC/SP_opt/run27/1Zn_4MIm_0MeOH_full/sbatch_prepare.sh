#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare.log
#SBATCH --error=prepare.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
set -euo pipefail

# gxtb mode: no Amber; create input.xyz on the fly if needed
source "$CONDAINIT"
export GXTBHOME="/home/zbai29/soft/g-xtb/parameters"
export PATH="/home/zbai29/soft/g-xtb/binary:$PATH"

echo "-2" > .CHRG

if [[ -f "KEEP_MOLS" ]]; then
  python "/home/zbai29/data/qmmm_test/python_scripts/gxtb_trim_xyz.py" --in source.xyz --out input.xyz --keep-file KEEP_MOLS
else
  cp -f source.xyz input.xyz
fi

# Single-point energy with gxtb on (already optimized) input.xyz
gxtb -c input.xyz -molden > sp_opt.out 2>&1
echo "sp_opt.out written."
