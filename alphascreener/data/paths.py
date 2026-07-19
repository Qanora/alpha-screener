"""Paths for the local OHLCV store."""

from datetime import date
from pathlib import Path

DATA_CATEGORIES: frozenset[str] = frozenset({"ohlcv"})


def get_data_home() -> Path:
    """Return the base data directory: ~/.alphascreener."""
    return Path.home() / ".alphascreener"


def get_data_dir(category: str) -> Path:
    """Return the data directory for a given category.

    Args:
        category: ``ohlcv``.

    Returns:
        Path like ~/.alphascreener/data/ohlcv/

    Raises:
        ValueError: If category is not in DATA_CATEGORIES.
    """
    if category not in DATA_CATEGORIES:
        raise ValueError(
            f"Unknown data category: {category!r}. Must be one of {sorted(DATA_CATEGORIES)}"
        )
    return get_data_home() / "data" / category


def _resolve_date(dt: date | str) -> date:
    """Convert a str or date to a date object."""
    if isinstance(dt, str):
        return date.fromisoformat(dt)
    return dt


def get_partition_path(category: str, dt: date | str) -> Path:
    """Return the partition directory path for a category and date.

    OHLCV partitions use ``dt=YYYY-MM-DD/``.

    Args:
        category: One of DATA_CATEGORIES.
        dt: A date or ISO-format date string.

    Returns:
        Full partition directory path.
    """
    d = _resolve_date(dt)
    base = get_data_dir(category)
    return base / f"dt={d.isoformat()}"
