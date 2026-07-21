#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装 uv。" >&2
  exit 1
fi

echo "=== Alpha Screener Validation ==="

echo "=== Step 1: Lint ==="
uv run ruff check alphascreener tests

echo "=== Step 2: Tests ==="
uv run pytest

echo "=== Step 3: CLI help ==="
uv run asc --help

echo "=== Validation complete ==="
