"""Field-level OHLCV comparison between primary and fallback data sources.

Issue #91: Stooq fallback adapter + cross-validation.
Reference: PRD 7.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFF_THRESHOLD_PCT: float = 0.5  # 0.5% relative difference triggers a diff record
OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OHLCVFieldDiffs:
    """Container for field-level difference records between two OHLCV sources.

    Attributes:
        records: List of dicts with keys: ticker, dt, field, primary_value,
            fallback_value, fallback_source, diff_pct.
        diff_tickers: Set of ticker symbols that have at least one field diff > threshold.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    diff_tickers: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compute_diff_pct(primary_value: float, fallback_value: float) -> float:
    """Compute the absolute relative difference between two values.

    Uses the primary (yfinance) value as the denominator. Returns 0.0 when
    the primary value is zero to avoid division by zero.

    Args:
        primary_value: Value from primary source (yfinance).
        fallback_value: Value from fallback source (Stooq).

    Returns:
        Absolute relative difference as a percentage (e.g. 0.5 means 0.5%).
    """
    if primary_value == 0.0:
        if fallback_value == 0.0:
            return 0.0
        return float("inf")  # primary is 0 but fallback is non-zero
    denom = abs(primary_value)
    return abs((primary_value - fallback_value) / denom) * 100.0


def _ensure_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a DataFrame has the expected OHLCV schema with correct types.

    Handles both date and datetime types in the ``dt`` column, coercing to date.
    """
    required = {"ticker", "dt", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Coerce types for safety
    df = df.with_columns(
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )

    dt_dtype = df.schema["dt"]
    if dt_dtype == pl.Datetime:
        df = df.with_columns(pl.col("dt").dt.date())
    elif dt_dtype != pl.Date:
        df = df.with_columns(pl.col("dt").cast(pl.Date))

    return df


def compare_ohlcv_dataframes(
    primary_df: pl.DataFrame,
    fallback_df: pl.DataFrame,
    fallback_source: str = "stooq",
    threshold_pct: float = DIFF_THRESHOLD_PCT,
) -> OHLCVFieldDiffs:
    """Compare OHLCV data between primary and fallback sources.

    Joins the two DataFrames on (ticker, dt) and computes the relative difference
    for Open, High, Low, and Close. Any field with an absolute relative
    difference > *threshold_pct* is recorded as a diff.

    Args:
        primary_df: OHLCV DataFrame from primary source (yfinance).
            Columns: ticker, dt, open, high, low, close, volume.
        fallback_df: OHLCV DataFrame from fallback source (Stooq).
            Columns: ticker, dt, open, high, low, close, volume.
        fallback_source: Identifier for the fallback source (stooq/alpaca/polygon).
        threshold_pct: Relative difference threshold in percent (default 0.5%).

    Returns:
        :class:`OHLCVFieldDiffs` with diff records and affected ticker set.
    """
    if primary_df.height == 0 or fallback_df.height == 0:
        return OHLCVFieldDiffs()

    primary = _ensure_schema(primary_df)
    fallback = _ensure_schema(fallback_df)

    # Inner join on (ticker, dt)
    joined = primary.join(
        fallback,
        on=["ticker", "dt"],
        how="inner",
        suffix="_fallback",
    )

    if joined.height == 0:
        return OHLCVFieldDiffs()

    diff_tickers: set[str] = set()
    records: list[dict[str, Any]] = []

    for row in joined.iter_rows(named=True):
        ticker: str = row["ticker"]
        dt_val: date = row["dt"]

        for field_name in OHLCV_FIELDS:
            primary_val: float = float(row[field_name])
            fallback_val: float = float(row.get(f"{field_name}_fallback", 0.0))

            # Skip comparison when both values are zero (no data)
            if primary_val == 0.0 and fallback_val == 0.0:
                continue

            # Skip NaN values
            if math.isnan(primary_val) or math.isnan(fallback_val):
                continue

            diff_pct = compute_diff_pct(primary_val, fallback_val)

            if diff_pct > threshold_pct:
                diff_tickers.add(ticker)
                records.append(
                    {
                        "ticker": ticker,
                        "dt": dt_val,
                        "field": field_name,
                        "primary_value": primary_val,
                        "fallback_value": fallback_val,
                        "fallback_source": fallback_source,
                        "diff_pct": diff_pct,
                    }
                )

    return OHLCVFieldDiffs(records=records, diff_tickers=diff_tickers)
