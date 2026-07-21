#!/usr/bin/env bash
set -euo pipefail

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_DIR="${SCRIPT_DIR}/xyz_files"
LOG_DIR="${SCRIPT_DIR}/transform_logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
rm -f "${OUT_DIR}"/*.xyz

run_transform() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "${SCRIPT_DIR}/transform_xyz.py" --out-dir "${OUT_DIR}" --log-dir "${LOG_DIR}"
        return 0
    fi

    if [[ -f "${HOME}/condainit.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/condainit.sh"
    fi
    conda activate amber
    python3 "${SCRIPT_DIR}/transform_xyz.py" --out-dir "${OUT_DIR}" --log-dir "${LOG_DIR}"
}

run_transform
echo "Wrote XYZ files to ${OUT_DIR}"
echo "Includes solvated complexes plus solvent-free Zn+ligand (MIm/MImH/Im-/ImH)."
