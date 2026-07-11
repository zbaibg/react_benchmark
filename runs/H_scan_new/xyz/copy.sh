#!/usr/bin/env bash
set -eo pipefail

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

# Copy original structures to xyz_files directory in the current path
for name in MImH2_monomer MImH_monomer MIm_monomer MeOH2_monomer MeOH_monomer; do
    cp "$REPO_ROOT/runs/iter_struc/relax_struc2/qm_minimize/run41/$name/min.xyz xyz_files/${name}.xyz
done

# Make sure the Python structure grid script is executable and generate H-scan grid structures with srun
chmod +x ./generate_h_scan_structures.py
srun ./generate_h_scan_structures.py

# Remove the original non-scanned xyz files, keep only the *_delta.xyz grid structures
for name in MImH2_monomer MImH_monomer MIm_monomer MeOH2_monomer MeOH_monomer; do
    rm -f "xyz_files/${name}.xyz"
done
