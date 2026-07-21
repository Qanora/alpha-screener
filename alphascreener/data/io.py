"""Parquet storage for the local OHLCV history."""

from __future__ import annotations

import glob
from datetime import date

import polars as pl

from alphascreener.data.paths import get_ohlcv_dir


def write_ohlcv(df: pl.DataFrame) -> None:
    """Merge OHLCV rows by ticker/date and atomically replace affected partitions."""
    required = {"ticker", "dt"}
    if missing := required - set(df.columns):
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    if df.schema["dt"] == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    if df.schema["dt"] != pl.Date:
        raise ValueError("DataFrame 'dt' column must be a date")
    for value in df["dt"].unique().to_list():
        partition = get_ohlcv_dir() / f"dt={value.isoformat()}"
        partition.mkdir(parents=True, exist_ok=True)
        existing_paths = sorted(partition.glob("*.parquet"))
        frames = [pl.read_parquet(path) for path in existing_paths]
        frames.append(df.filter(pl.col("dt") == value))
        merged = (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset=["ticker", "dt"], keep="last")
            .sort(["ticker", "dt"])
        )
        output = partition / "data.parquet"
        temporary = partition / "data.parquet.tmp"
        merged.write_parquet(temporary)
        temporary.replace(output)
        for stale in existing_paths:
            if stale != output:
                stale.unlink()


def scan_ohlcv(*, date_filter: date | None = None) -> pl.LazyFrame:
    """Scan stored OHLCV partitions."""
    base = get_ohlcv_dir()
    pattern = base / (f"dt={date_filter.isoformat()}/*.parquet" if date_filter else "**/*.parquet")
    if not glob.glob(str(pattern), recursive=True):
        raise FileNotFoundError(f"No Parquet files found matching: {pattern}")
    return pl.scan_parquet(str(pattern))
