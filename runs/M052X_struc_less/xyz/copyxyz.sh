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

# Copy B3LYP_struc/qm_minimize/run23/<STRUCTNAME>/min.xyz into xyz_files/<STRUCTNAME>.xyz
# (react_benchmark is the parent of B3LYP_struc and M052X_struc).
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REACT_BENCH=$(cd "$SCRIPT_DIR/../.." && pwd)
OUT="$SCRIPT_DIR/xyz_files"

mkdir -p "$OUT"


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

root="$REACT_BENCH/B3LYP_struc/qm_minimize/run23"
if [[ ! -d "$root" ]]; then
  echo "ERROR: missing $root" >&2
  exit 1
fi
echo "Collecting $root/*/min.xyz ..."
for f in "$root"/*/min.xyz; do
  [[ -f "$f" ]] || continue
  struct=$(basename "$(dirname "$f")")
  register_and_copy "$f" "$struct"
done

echo
n=$(find "$OUT" -type f -name '*.xyz' | wc -l)
echo "Done. $n .xyz file(s) in $OUT"
