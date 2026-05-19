"""Path utilities for the Data I/O layer (Issue #86).

Resolves ~/.alphascreener/data/{category}/ paths and partition directories.
Reference: PRD 7.6.1 Storage Tier.
"""

from datetime import date
from pathlib import Path

# Categories matching PRD 7.6.1 table
DATA_CATEGORIES: frozenset[str] = frozenset({"ohlcv", "factors", "signals", "backtest"})

# Categories that use daily partitions (dt=YYYY-MM-DD)
_DAILY_CATEGORIES: frozenset[str] = frozenset({"ohlcv", "factors", "signals"})

# Categories that use monthly partitions (dt=YYYY-MM)
_MONTHLY_CATEGORIES: frozenset[str] = frozenset({"backtest"})


def get_data_home() -> Path:
    """Return the base data directory: ~/.alphascreener."""
    return Path.home() / ".alphascreener"


def get_data_dir(category: str) -> Path:
    """Return the data directory for a given category.

    Args:
        category: One of DATA_CATEGORIES (ohlcv, factors, signals, backtest).

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

    Daily categories:  dt=YYYY-MM-DD/
    Monthly categories: dt=YYYY-MM/

    Args:
        category: One of DATA_CATEGORIES.
        dt: A date or ISO-format date string.

    Returns:
        Full partition directory path.
    """
    d = _resolve_date(dt)
    base = get_data_dir(category)
    if category in _DAILY_CATEGORIES:
        return base / f"dt={d.isoformat()}"
    else:
        return base / f"dt={d.strftime('%Y-%m')}"


def get_archive_dir(category: str | None = None) -> Path:
    """Return the archive directory.

    Args:
        category: Optional category subdirectory within archive.

    Returns:
        Path like ~/.alphascreener/archive/ or ~/.alphascreener/archive/ohlcv/
    """
    base = get_data_home() / "archive"
    if category:
        return base / category
    return base
