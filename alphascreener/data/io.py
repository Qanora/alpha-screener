"""Parquet storage for the local OHLCV history."""

from __future__ import annotations

import glob
from datetime import date

import polars as pl

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_ohlcv_dir

_REQUIRED_COLUMNS = {"ticker", "dt", "open", "high", "low", "close", "volume"}
_PRICE_COLUMNS = ("open", "high", "low", "close")


def write_ohlcv(df: pl.DataFrame) -> None:
    """Merge OHLCV rows by ticker/date and atomically replace affected partitions."""
    if missing := _REQUIRED_COLUMNS - set(df.columns):
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    if df.schema["ticker"] != pl.String:
        raise ValueError("DataFrame 'ticker' column must be a string")
    if df.schema["dt"] == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    if df.schema["dt"] != pl.Date:
        raise ValueError("DataFrame 'dt' column must be a date")
    numeric_columns = (*_PRICE_COLUMNS, "volume")
    if any(not df.schema[column].is_numeric() for column in numeric_columns):
        raise ValueError("OHLCV price and volume columns must be numeric")
    invalid = (
        pl.col("ticker").is_null()
        | (pl.col("ticker").str.strip_chars() == "")
        | pl.col("dt").is_null()
        | pl.any_horizontal([
            pl.col(column).is_null()
            | ~pl.col(column).cast(pl.Float64).is_finite()
            | (pl.col(column) <= 0)
            for column in _PRICE_COLUMNS
        ])
        | pl.col("volume").is_null()
        | ~pl.col("volume").cast(pl.Float64).is_finite()
        | (pl.col("volume") < 0)
    )
    if not df.filter(invalid).is_empty():
        raise ValueError("OHLCV rows must contain finite positive prices and non-negative volume")

    base = get_ohlcv_dir()
    with exclusive_file_lock(base / ".write.lock"):
        for value in df["dt"].unique().to_list():
            partition = base / f"dt={value.isoformat()}"
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
