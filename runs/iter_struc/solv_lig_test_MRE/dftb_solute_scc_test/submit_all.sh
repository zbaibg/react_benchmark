#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "submit Im-_MeOH__t1000_broyden"; (cd "$ROOT/Im-_MeOH__t1000_broyden" && sbatch sbatch_prepare.sh)
echo "submit Im-_MeOH__t300_broyden"; (cd "$ROOT/Im-_MeOH__t300_broyden" && sbatch sbatch_prepare.sh)
echo "submit Im-_MeOH__t300_broyden_max1000"; (cd "$ROOT/Im-_MeOH__t300_broyden_max1000" && sbatch sbatch_prepare.sh)
echo "submit Im-_MeOH__t300_diis"; (cd "$ROOT/Im-_MeOH__t300_diis" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t0_broyden"; (cd "$ROOT/ImH_Wat__t0_broyden" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t0_broyden_max1000"; (cd "$ROOT/ImH_Wat__t0_broyden_max1000" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t0_diis"; (cd "$ROOT/ImH_Wat__t0_diis" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t1000_broyden"; (cd "$ROOT/ImH_Wat__t1000_broyden" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t3000_broyden"; (cd "$ROOT/ImH_Wat__t3000_broyden" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t300_broyden"; (cd "$ROOT/ImH_Wat__t300_broyden" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t300_broyden_max1000"; (cd "$ROOT/ImH_Wat__t300_broyden_max1000" && sbatch sbatch_prepare.sh)
echo "submit ImH_Wat__t300_diis"; (cd "$ROOT/ImH_Wat__t300_diis" && sbatch sbatch_prepare.sh)
