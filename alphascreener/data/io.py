"""Parquet read/write with Hive-style partitioning and zstd archiving (Issue #86).

Supports daily partitions (ohlcv/factors/signals: dt=YYYY-MM-DD) and
monthly partitions (backtest: dt=YYYY-MM). All reads return polars LazyFrame
for streaming with predicate pushdown.

Reference: PRD 7.6.1 Storage Tier / 7.6.3 Data Retention.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl

from alphascreener.data.paths import (
    _DAILY_CATEGORIES,
    DATA_CATEGORIES,
    _resolve_date,
    get_archive_dir,
    get_data_dir,
)

# -- partition key generation ------------------------------------------------


def _partition_key(dt: date, category: str) -> str:
    """Return the Hive-style partition value for a date and category."""
    if category in _DAILY_CATEGORIES:
        return dt.isoformat()
    return dt.strftime("%Y-%m")


def _write_partition(df: pl.DataFrame, partition_dir: Path) -> None:
    """Write a DataFrame slice to a single partition directory.

    Clears existing parquet files in the directory first so that every write
    is a full replacement (overwrite semantics).  Callers are responsible for
    grouping all rows that belong to the same partition into a single call.
    """
    partition_dir.mkdir(parents=True, exist_ok=True)
    # Delete old parquet files to prevent data accumulation across sync runs
    for old_file in partition_dir.glob("*.parquet"):
        old_file.unlink()
    df.write_parquet(partition_dir / "data.parquet")


# -- public API ---------------------------------------------------------------


def write_parquet(df: pl.DataFrame, category: str) -> None:
    """Write a DataFrame to Hive-partitioned Parquet files.

    The DataFrame **must** contain a ``dt`` column (date type). Rows are
    grouped by partition key and written into the appropriate partition
    directory.  Existing files in the target partition are removed before
    writing (overwrite semantics).

    Daily categories (ohlcv/factors/signals) produce dt=YYYY-MM-DD/.
    Monthly category (backtest) produces dt=YYYY-MM/.

    Args:
        df: DataFrame with a ``dt`` column of type ``pl.Date``.
        category: One of ``DATA_CATEGORIES``.

    Raises:
        ValueError: If ``category`` is unknown or ``df`` lacks a ``dt`` column.
    """
    if category not in DATA_CATEGORIES:
        raise ValueError(
            f"Unknown data category: {category!r}. Must be one of {sorted(DATA_CATEGORIES)}"
        )

    if "dt" not in df.columns:
        raise ValueError("DataFrame must contain a 'dt' column")

    # Validate and coerce dt column type
    dt_dtype = df.schema["dt"]
    if dt_dtype == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    elif dt_dtype != pl.Date:
        raise ValueError(f"DataFrame 'dt' column must be pl.Date or pl.Datetime, got {dt_dtype!r}")

    # Group rows by partition key so that monthly partitions (backtest)
    # have all their dates gathered before the directory is cleared.
    df = df.with_columns(
        pl.col("dt")
        .map_elements(lambda d: _partition_key(d, category), return_dtype=pl.String)
        .alias("_part_key")
    )

    for (part_key,) in df.select("_part_key").unique().sort("_part_key").iter_rows():
        part_dir = get_data_dir(category) / f"dt={part_key}"
        part_df = df.filter(pl.col("_part_key") == part_key).drop("_part_key")
        _write_partition(part_df, part_dir)


def scan_parquet(category: str, *, date_filter: date | str | None = None) -> pl.LazyFrame:
    """Return a LazyFrame scanning all Parquet files in a category.

    Uses ``pl.scan_parquet`` on the glob pattern so that predicate pushdown
    and projection pushdown are preserved.

    Args:
        category: One of DATA_CATEGORIES.
        date_filter: Optional single date to restrict scanning to one partition.

    Returns:
        polars LazyFrame ready for further filtering/aggregation.
    """
    if category not in DATA_CATEGORIES:
        raise ValueError(
            f"Unknown data category: {category!r}. Must be one of {sorted(DATA_CATEGORIES)}"
        )

    if date_filter is not None:
        d = _resolve_date(date_filter)
        part_key = _partition_key(d, category)
        glob_pattern = str(get_data_dir(category) / f"dt={part_key}" / "*.parquet")
    else:
        glob_pattern = str(get_data_dir(category) / "**" / "*.parquet")

    import glob as _glob
    matches = _glob.glob(glob_pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No Parquet files found matching: {glob_pattern}")
    try:
        return pl.scan_parquet(glob_pattern)
    except Exception:
        raise FileNotFoundError(f"No Parquet files found matching: {glob_pattern}")


def read_parquet(category: str) -> pl.LazyFrame:
    """Read all partitions for a category and return a LazyFrame.

    Convenience wrapper around ``scan_parquet`` with no date filter.

    Args:
        category: One of DATA_CATEGORIES.

    Returns:
        polars LazyFrame.
    """
    return scan_parquet(category, date_filter=None)


def archive_old_data(category: str, *, before_date: date | str) -> None:
    """Move old partition directories to the archive as zstd Parquet.

    Any partition whose date is strictly before ``before_date`` is:
    1. Read into memory as a DataFrame.
    2. Written as a single zstd-compressed Parquet file under
       ``~/.alphascreener/archive/{category}/``.
    3. Removed from the active data directory.

    Args:
        category: One of DATA_CATEGORIES.
        before_date: Cutoff date. Partitions with dates < before_date
                     are archived.
    """
    if category not in DATA_CATEGORIES:
        raise ValueError(
            f"Unknown data category: {category!r}. Must be one of {sorted(DATA_CATEGORIES)}"
        )

    cutoff = _resolve_date(before_date)
    data_dir = get_data_dir(category)
    archive_dir = get_archive_dir(category)
    archive_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        return  # idempotent no-op: nothing to archive

    for part_dir in sorted(data_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        # Parse partition name: dt=YYYY-MM-DD or dt=YYYY-MM
        part_name = part_dir.name
        if not part_name.startswith("dt="):
            continue
        date_str = part_name[3:]  # strip "dt="

        # Determine if this partition is before the cutoff
        if category in _DAILY_CATEGORIES:
            try:
                part_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            if part_date >= cutoff:
                continue
        else:
            # Monthly partition: dt=YYYY-MM. Compare first day of month.
            try:
                part_date = date.fromisoformat(date_str + "-01")
            except ValueError:
                continue
            # A monthly partition is archived if its first day is before the
            # first day of the cutoff month.
            cutoff_month = date(cutoff.year, cutoff.month, 1)
            if part_date >= cutoff_month:
                continue

        # Read all Parquet files in this partition
        parquet_files = list(part_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        dfs = [pl.read_parquet(str(f)) for f in parquet_files]
        merged = pl.concat(dfs)

        # Write to archive as zstd Parquet with unique timestamp suffix
        import time

        ts = int(time.time() * 1_000_000)
        archive_file = archive_dir / f"{part_name}_{ts}.parquet"
        merged.write_parquet(archive_file, compression="zstd")

        # Remove the original partition directory
        shutil.rmtree(part_dir)
