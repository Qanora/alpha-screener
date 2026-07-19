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

echo "=== Codex Plan Pipeline Check ==="
echo "Project: ${PROJECT_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import sys
import platform

print(f"Python: {platform.python_version()}")
print(f"Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
PY

echo "=== Step 1: Lint (ruff) ==="
run_with_runner ruff check alphascreener tests

echo "=== Step 2: Unit tests (full) ==="
run_with_runner pytest

echo "=== Step 3: Key smoke test ==="
run_with_runner asc --help

echo "=== Plan pipeline check done ==="
