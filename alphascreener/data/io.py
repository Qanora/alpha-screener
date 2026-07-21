"""Parquet storage for the local OHLCV history."""

from __future__ import annotations

import glob
from datetime import date

import polars as pl

from alphascreener.data.paths import get_ohlcv_dir


def write_ohlcv(df: pl.DataFrame) -> None:
    """Replace each affected daily OHLCV partition with its supplied rows."""
    if "dt" not in df.columns:
        raise ValueError("DataFrame must contain a 'dt' column")
    if df.schema["dt"] == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    if df.schema["dt"] != pl.Date:
        raise ValueError("DataFrame 'dt' column must be a date")
    for value in df["dt"].unique().to_list():
        partition = get_ohlcv_dir() / f"dt={value.isoformat()}"
        partition.mkdir(parents=True, exist_ok=True)
        for existing in partition.glob("*.parquet"):
            existing.unlink()
        df.filter(pl.col("dt") == value).write_parquet(partition / "data.parquet")


def scan_ohlcv(*, date_filter: date | None = None) -> pl.LazyFrame:
    """Scan stored OHLCV partitions."""
    base = get_ohlcv_dir()
    pattern = base / (f"dt={date_filter.isoformat()}/*.parquet" if date_filter else "**/*.parquet")
    if not glob.glob(str(pattern), recursive=True):
        raise FileNotFoundError(f"No Parquet files found matching: {pattern}")
    return pl.scan_parquet(str(pattern))
