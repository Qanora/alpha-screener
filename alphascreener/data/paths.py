"""Paths for the local OHLCV store."""

from pathlib import Path


def get_data_home() -> Path:
    """Return the base data directory: ~/.alphascreener."""
    return Path.home() / ".alphascreener"


def get_ohlcv_dir() -> Path:
    """Return the local OHLCV partition directory."""
    return get_data_home() / "data" / "ohlcv"
