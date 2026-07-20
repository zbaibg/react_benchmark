#!/usr/bin/env bash
# Usage: ./modify_meoh_orc_resources.sh /path/to/run46
# Edits only:
# - <base_dir>/1Zn_1MIm*_0MeOH_full/orc_job.inp
# - <base_dir>/1Zn_1MIm*_0MeOH_monomer*_ghost/orc_job.inp
# Replacements:
# - %maxcore 6000 -> %maxcore 9000
# - nprocs 32 -> nprocs 24

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

shopt -s nullglob

targets=(
  "$base_dir"/1Zn_1MIm*_0MeOH_full/orc_job.inp
  "$base_dir"/1Zn_1MIm*_0MeOH_monomer*_ghost/orc_job.inp
)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "Warn: no matching orc_job.inp files found under $base_dir" >&2
  exit 0
fi

for inp in "${targets[@]}"; do
  if [[ ! -f "$inp" ]]; then
    continue
  fi

  before_maxcore=$(grep -E '^[[:space:]]*%maxcore[[:space:]]+' "$inp" || true)
  before_nprocs=$(grep -E '^[[:space:]]*nprocs[[:space:]]+' "$inp" || true)

  sed -i -E 's/^([[:space:]]*%maxcore[[:space:]]+)6000([[:space:]]*)$/\19000\2/' "$inp"
  sed -i -E 's/^([[:space:]]*nprocs[[:space:]]+)32([[:space:]]*)$/\124\2/' "$inp"

  after_maxcore=$(grep -E '^[[:space:]]*%maxcore[[:space:]]+' "$inp" || true)
  after_nprocs=$(grep -E '^[[:space:]]*nprocs[[:space:]]+' "$inp" || true)

  printf '[LOG] %s\n  maxcore: %s -> %s\n  nprocs:  %s -> %s\n' \
    "$inp" "${before_maxcore:-<missing>}" "${after_maxcore:-<missing>}" \
    "${before_nprocs:-<missing>}" "${after_nprocs:-<missing>}"
done
