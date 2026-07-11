#!/bin/bash
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
cp ../initial_opt/1Zn_1ImH_6Wat_1Hbond.xyz xyz_files/
cp ../initial_opt/1Zn_1ImH_6Wat_2Hbond.xyz xyz_files/
python generate_imh_scan_structures.py
python remove_zn_from_xyz_files.py