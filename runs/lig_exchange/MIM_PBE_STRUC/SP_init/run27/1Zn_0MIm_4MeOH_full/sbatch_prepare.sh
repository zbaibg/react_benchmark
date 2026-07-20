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

# gxtb mode: no Amber; input.xyz already in place
source ~/condainit.sh
export GXTBHOME="/home/zbai29/soft/g-xtb/parameters"
export PATH="/home/zbai29/soft/g-xtb/binary:$PATH"

echo "2" > .CHRG

# Single-point energy with gxtb
gxtb -c input.xyz -molden > sp_init.out 2>&1
echo "sp_init.out written."
