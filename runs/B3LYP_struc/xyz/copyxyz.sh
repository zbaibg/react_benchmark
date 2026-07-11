#!/usr/bin/env bash
# Resolve repo root and software paths
_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do
  _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"
done
if [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; then
  echo "ERROR: could not locate repo root (software.yaml)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$_REPO_CANDIDATE/tools/repo_env.sh"

# Copy lig_exchange/*/qm_minimize/run16/<STRUCTNAME>/min.xyz and
# deprotonate/*/qm_minimize/run16/<STRUCTNAME>/min.xyz into xyz_files/<STRUCTNAME>.xyz
# (react_benchmark is the parent of B3LYP_struc).
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REACT_BENCH=$(cd "$SCRIPT_DIR/../.." && pwd)
OUT="$SCRIPT_DIR/xyz_files"

mkdir -p "$OUT"

cp "$REPO_ROOT/structures/nbZIFFF-km/1coord_PBE_TZVP/1Zn_0MIm_1MeOH/cal.xyz $OUT/1Zn_0MIm_1MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/1coord_PBE_TZVP/1Zn_1MIm_0MeOH/cal.xyz $OUT/1Zn_1MIm_0MeOH.xyz
cp "$REPO_ROOT/structures/nbZIFFF-km/1coord_PBE_TZVP/1Zn_1MImH_0MeOH/cal.xyz $OUT/1Zn_1MImH_0MeOH.xyz


declare -A seen=()

register_and_copy() {
  local f="$1"
  local struct="$2"
  if [[ -n "${seen[$struct]+x}" ]]; then
    echo "WARNING: duplicate STRUCTNAME '${struct}'; keeping first (${seen[$struct]}), skipping: $f" >&2
    return 0
  fi
  seen[$struct]=$f
  cp -- "$f" "$OUT/${struct}.xyz"
  echo "  $f -> $OUT/${struct}.xyz"
}

echo "REACT_BENCH=$REACT_BENCH"
echo "Output: $OUT"
echo

for sub in lig_exchange deprotonate; do
  root="$REACT_BENCH/$sub"
  if [[ ! -d "$root" ]]; then
    echo "SKIP: missing $root" >&2
    continue
  fi
  echo "Collecting $root/*/qm_minimize/run16/*/min.xyz ..."
  for f in "$root"/*/qm_minimize/run16/*/min.xyz; do
    [[ -f "$f" ]] || continue
    struct=$(basename "$(dirname "$f")")
    register_and_copy "$f" "$struct"
  done
done

echo
n=$(find "$OUT" -type f -name '*.xyz' | wc -l)
echo "Done. $n .xyz file(s) in $OUT"
