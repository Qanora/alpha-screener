#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_BIN="${PROJECT_ROOT}/.venv/bin"
PYTHON_BIN="${VENV_BIN}/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 .venv 环境: ${PYTHON_BIN}" >&2
  echo "请先执行：uv sync --extra dev" >&2
  exit 1
fi

run_with_runner() {
  local tool=$1
  shift
  if [ "${ASC_USE_UV:-0}" = "1" ] && command -v uv >/dev/null 2>&1; then
    if uv run "$tool" "$@"; then
      return 0
    fi
  fi

  local tool_bin="${VENV_BIN}/${tool}"
  if [ -x "$tool_bin" ]; then
    "$tool_bin" "$@"
    return $?
  fi

  "$PYTHON_BIN" -m "$tool" "$@"
}

echo "=== Codex Inspection ==="

echo "Environment:"
if [ -f ".env" ]; then
  echo "Found .env in project root"
else
  echo "No .env file"
fi

echo "=== Inspection Step 1: lint ==="
run_with_runner ruff check alphascreener tests

echo "=== Inspection Step 2: CLI smoke tests ==="
run_with_runner pytest tests/test_cli.py tests/test_evaluation.py -q

echo "=== Inspection Step 3: CLI help smoke ==="
run_with_runner asc --help

echo "=== Inspection complete ==="
