"""Shared path helpers for the universe module (Issue #88)."""

from __future__ import annotations

from pathlib import Path

from alphascreener.data.paths import get_data_home


def _universe_dir() -> Path:
    """Return the universe data directory (~/.alphascreener/universe/)."""
    return get_data_home() / "universe"
