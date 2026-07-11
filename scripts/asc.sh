#!/usr/bin/env bash
# Simple helper to run alpha-screener commands in the project virtualenv.
# Usage:
#   scripts/asc.sh               # run `asc --help`
#   scripts/asc.sh --help         # show asc help
#   scripts/asc.sh sync --top 5   # run any asc subcommand

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
VENV_BIN="${PROJECT_ROOT}/.venv/bin"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "未找到虚拟环境激活脚本：${VENV_ACTIVATE}"
  echo "请先在仓库根目录创建 .venv，或确认路径是否正确。"
  exit 1
fi

. "${VENV_ACTIVATE}"

# Make project import path explicit when executed outside repo root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ "$#" -eq 0 ]; then
  exec "${VENV_BIN}/asc" --help
else
  exec "${VENV_BIN}/asc" "$@"
fi
