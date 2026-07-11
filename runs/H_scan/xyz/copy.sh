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
for name in H3O_monomer H_monomer MImH_monomer MIm_monomer Wat_monomer; do
    cp "$REPO_ROOT/runs/deprotonate/MIMH_PBE_STRUC/qm_minimize/run16/$name/min.xyz xyz_files/${name}.xyz
done
cp "$REPO_ROOT/runs/lig_exchange/MIMH_PBE_STRUC/qm_minimize/run16/MeOH_monomer/min.xyz xyz_files/MeOH_monomer.xyz

# Make sure the Python structure grid script is executable and generate H-scan grid structures with srun
chmod +x ./generate_h_scan_structures.py
srun ./generate_h_scan_structures.py

# Remove the original non-scanned xyz files, keep only the *_delta.xyz grid structures
for name in H3O_monomer H_monomer MImH_monomer MIm_monomer Wat_monomer MeOH_monomer; do
    rm -f "xyz_files/${name}.xyz"
done
