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

mkdir -p xyz_files
for name in 1Zn_0MIm_1MeOH 1Zn_1MIm_0MeOH 1Zn_1MImH_0MeOH
do
cp "$REPO_ROOT/runs/M052X_struc/qm_minimize/run30/$name/min.xyz xyz_files/$name.xyz
done
