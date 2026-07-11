#!/usr/bin/env bash
# Source from bash or zsh scripts:
#   source "$REPO_ROOT/tools/repo_env.sh"
#
# Exports REPO_ROOT and software paths from software.yaml.

# Resolve this file's directory in bash and zsh.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _REPO_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _REPO_ENV_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  _REPO_ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
export REPO_ROOT="$(cd "$_REPO_ENV_DIR/.." && pwd)"
eval "$("$_REPO_ENV_DIR/export_software_env.py")"
