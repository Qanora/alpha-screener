"""Paths for the local OHLCV store."""

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
