#!/usr/bin/env bash
# Usage: ./modify_meoh_core.sh /path/to/run42
# Edits:
# - */{1Zn_0MIm_1MeOH_monomer_1,1Zn_0MIm_1MeOH_monomer_1_ghost,MeOH_monomer}/orc_job.inp : nprocs 32 -> nprocs 16
# - */{1Zn_0MIm_1MeOH_monomer_1,1Zn_0MIm_1MeOH_monomer_1_ghost,MeOH_monomer}/sbatch_prepare.sh : #SBATCH --ntasks=32 -> #SBATCH --ntasks=16

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <base_dir>" >&2
  exit 2
fi

base_dir="$1"
if [[ ! -d "$base_dir" ]]; then
  echo "Error: not a directory: $base_dir" >&2
  exit 2
fi

targets=(
  "1Zn_0MIm_1MeOH_monomer_1"
  "1Zn_0MIm_1MeOH_monomer_1_ghost"
  "MeOH_monomer"
)

for d in "${targets[@]}"; do
  inp="$base_dir/$d/orc_job.inp"
  sb="$base_dir/$d/sbatch_prepare.sh"

  if [[ -f "$inp" ]]; then
    before_nprocs=$(grep -E '^[[:space:]]*nprocs[[:space:]]+' "$inp" || true)
    # replace whole line containing nprocs 32 (allow spaces)
    sed -i -E 's/^([[:space:]]*nprocs[[:space:]]+)32([[:space:]]*)$/\116\2/' "$inp"
    after_nprocs=$(grep -E '^[[:space:]]*nprocs[[:space:]]+' "$inp" || true)
    printf '[LOG] %s\n  before: %s\n  after:  %s\n' "$inp" "$before_nprocs" "$after_nprocs"
  else
    echo "Warn: missing file: $inp" >&2
  fi

  if [[ -f "$sb" ]]; then
    before_ntasks=$(grep -E '^[[:space:]]*#SBATCH[[:space:]]+--ntasks=' "$sb" || true)
    # replace whole line containing #SBATCH --ntasks=32 (allow spaces)
    sed -i -E 's/^([[:space:]]*#SBATCH[[:space:]]+--ntasks=)32([[:space:]]*)$/\116\2/' "$sb"
    after_ntasks=$(grep -E '^[[:space:]]*#SBATCH[[:space:]]+--ntasks=' "$sb" || true)
    printf '[LOG] %s\n  before: %s\n  after:  %s\n' "$sb" "$before_ntasks" "$after_ntasks"
  else
    echo "Warn: missing file: $sb" >&2
  fi
done