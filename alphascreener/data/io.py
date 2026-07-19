"""Parquet storage for the local OHLCV history."""

from __future__ import annotations

import glob
from datetime import date

import polars as pl

from alphascreener.data.paths import DATA_CATEGORIES, get_data_dir


def write_parquet(df: pl.DataFrame, category: str = "ohlcv") -> None:
    """Replace each affected daily OHLCV partition with its supplied rows."""
    if category not in DATA_CATEGORIES:
        raise ValueError(f"Unknown data category: {category!r}")
    if "dt" not in df.columns:
        raise ValueError("DataFrame must contain a 'dt' column")
    if df.schema["dt"] == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    if df.schema["dt"] != pl.Date:
        raise ValueError("DataFrame 'dt' column must be a date")
    for value in df["dt"].unique().to_list():
        partition = get_data_dir(category) / f"dt={value.isoformat()}"
        partition.mkdir(parents=True, exist_ok=True)
        for existing in partition.glob("*.parquet"):
            existing.unlink()
        df.filter(pl.col("dt") == value).write_parquet(partition / "data.parquet")


def scan_parquet(category: str = "ohlcv", *, date_filter: date | None = None) -> pl.LazyFrame:
    """Scan stored OHLCV partitions."""
    if category not in DATA_CATEGORIES:
        raise ValueError(f"Unknown data category: {category!r}")
    base = get_data_dir(category)
    pattern = base / (f"dt={date_filter.isoformat()}/*.parquet" if date_filter else "**/*.parquet")
    if not glob.glob(str(pattern), recursive=True):
        raise FileNotFoundError(f"No Parquet files found matching: {pattern}")
    return pl.scan_parquet(str(pattern))


def read_parquet(category: str = "ohlcv") -> pl.LazyFrame:
    """Compatibility wrapper for scanning the OHLCV store."""
    return scan_parquet(category)
