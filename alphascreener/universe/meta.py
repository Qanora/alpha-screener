"""Universe metadata cache: sector/industry/market_cap parquet (Issue #88).

Reference: PRD 7.7.1 - universe_meta.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alphascreener.universe import _paths

META_FILENAME: str = "universe_meta.parquet"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "sector", "industry", "market_cap", "index_source", "refreshed_at"}
)


def _meta_path() -> Path:
    """Return the full path to universe_meta.parquet."""
    return _paths._universe_dir() / META_FILENAME


def write_meta_cache(df: pl.DataFrame) -> None:
    """Write sector/industry/market_cap metadata to universe_meta.parquet.

    Args:
        df: DataFrame with columns: ticker, sector, industry, market_cap,
            index_source, refreshed_at.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Meta DataFrame missing required columns: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_COLUMNS)}"
        )

    _paths._universe_dir().mkdir(parents=True, exist_ok=True)
    df.write_parquet(_meta_path())


def read_meta_cache() -> pl.LazyFrame:
    """Read universe metadata from universe_meta.parquet.

    Returns a LazyFrame for predicate pushdown (e.g. filter by sector).

    Returns:
        LazyFrame with columns: ticker, sector, industry, market_cap,
        index_source, refreshed_at.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    path = _meta_path()
    if not path.exists():
        raise FileNotFoundError(f"Universe meta cache not found: {path}")
    return pl.scan_parquet(path)
