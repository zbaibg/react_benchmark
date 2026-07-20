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
source "$CONDAINIT"
export GXTBHOME="/home/zbai29/soft/g-xtb/parameters"
export PATH="/home/zbai29/soft/g-xtb/binary:$PATH"

echo "-1" > .CHRG

# Geometry optimization with xtb using gxtb as driver (numerical gradient)
xtb input.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt
cp xtbopt.xyz min.xyz
echo "min.xyz written."
