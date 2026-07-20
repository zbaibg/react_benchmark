#!/usr/bin/env bash
# For each subdirectory of run35: if run35_tol3d-4_tole5d-6/<same>/min.rst exists,
# replace run35/<same>/init.rst with a symlink to that min.rst (name: init.rst), remove MOL.xyz.
# If that tol folder also has min.xyz, copy min.out, old.orc_job.dat, min.xyz, path.xyz (each if present there).
#
# Usage:
#   ./link_files_to_restart.sh              # script lives in qm_minimize, or set QM_MINIMIZE_ROOT
#   QM_MINIMIZE_ROOT=/path/to/qm_minimize ./link_files_to_restart.sh
#   ./link_files_to_restart.sh --dry-run    # print actions only

set -uo pipefail

RUN35_NAME="run35"
TOL_NAME="run35_tol3d-4_tole5d-6"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
  esac
done

ROOT="${QM_MINIMIZE_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")" && pwd)"
fi

RUN35="${ROOT}/${RUN35_NAME}"
TOL="${ROOT}/${TOL_NAME}"

if [[ ! -d "$RUN35" ]]; then
  echo "error: run35 directory not found: $RUN35" >&2
  exit 1
fi
if [[ ! -d "$TOL" ]]; then
  echo "error: tol directory not found: $TOL" >&2
  exit 1
fi

processed=0   # tol 有 min.rst，已执行链接/复制
skipped=0     # tol 无 min.rst，未执行文件操作
need_restart=() # tol 未同时有 min.rst 与 min.xyz（含无 min.rst、仅有 min.rst）
no_restart=()   # tol 同时有 min.rst 与 min.xyz

while IFS= read -r -d '' d; do
  name="$(basename "$d")"
  tol_sub="${TOL}/${name}"
  min_src="${tol_sub}/min.rst"
  min_xyz_src="${tol_sub}/min.xyz"

  # 汇总：仅当 tol 同时有 min.rst 与 min.xyz 才算「不需要重启」
  if [[ -f "$min_src" && -f "$min_xyz_src" ]]; then
    no_restart+=("$name")
  else
    need_restart+=("$name")
  fi

  if [[ ! -f "$min_src" ]]; then
    ((skipped++)) || true
    continue
  fi

  init_dst="${d}/init.rst"
  mol_dst="${d}/MOL.xyz"
  tol_sub="${TOL}/${name}"
  # From run35/<name>/: ../.. is qm_minimize; link min.rst as init.rst
  rel_tol="../../${TOL_NAME}/${name}"
  min_link_target="${rel_tol}/min.rst"
  extra_files=(min.out old.orc_job.dat min.xyz path.xyz)

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $name:"
    echo "  rm -f -- $init_dst"
    echo "  ln -sfn -- $min_link_target $init_dst"
    echo "  rm -f -- $mol_dst"
    if [[ -f "$min_xyz_src" ]]; then
      for f in "${extra_files[@]}"; do
        src="${tol_sub}/${f}"
        if [[ -e "$src" ]]; then
          echo "  rm -f -- ${d}/${f}"
          echo "  cp -f -- $src ${d}/${f}"
        fi
      done
    fi
  else
    rm -f -- "$init_dst"
    ln -sfn -- "$min_link_target" "$init_dst"
    rm -f -- "$mol_dst"
    if [[ -f "$min_xyz_src" ]]; then
      for f in "${extra_files[@]}"; do
        src="${tol_sub}/${f}"
        if [[ -e "$src" ]]; then
          rm -f -- "${d}/${f}"
          cp -f -- "$src" "${d}/${f}"
        fi
      done
    fi
    echo "updated: $name"
  fi
  ((processed++)) || true
done < <(find "$RUN35" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

total_dirs=$((processed + skipped))
echo "done: run35_subdirs=${total_dirs} linked_copied=${processed} skipped_no_minrst=${skipped} root=$ROOT"
echo ""
echo "需要重新启动（对应 ${TOL_NAME} 未同时具备 min.rst 与 min.xyz）:"
if ((${#need_restart[@]})); then
  printf '  %s\n' "${need_restart[@]}"
else
  echo "  (无)"
fi
echo ""
echo "不需要重新启动（对应 ${TOL_NAME} 同时有 min.rst 与 min.xyz）:"
if ((${#no_restart[@]})); then
  printf '  %s\n' "${no_restart[@]}"
else
  echo "  (无)"
fi
