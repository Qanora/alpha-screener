"""Pre-filter conditions for universe screening (Issue #88).

Reference: PRD 3.3 - 预过滤条件:
  - 近20日平均成交额 > 20M
  - 总市值 > 300M
  - 股价 > 5
  - 上市 >= 12个月 (>= 252 trading days)
  - 非停牌非退市 (status == "Active")
"""

from __future__ import annotations

import polars as pl

# Default thresholds
DEFAULT_MIN_DOLLAR_VOLUME: float = 20_000_000.0  # $20M
DEFAULT_MIN_MARKET_CAP: float = 300_000_000.0  # $300M
DEFAULT_MIN_PRICE: float = 5.0  # $5
DEFAULT_MIN_DAYS_LISTED: int = 252  # ~12 trading months


def _filter_dollar_volume(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """avg_dollar_volume_20d > threshold."""
    return df.filter(pl.col("avg_dollar_volume_20d") >= threshold)


def _filter_market_cap(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """market_cap > threshold."""
    return df.filter(pl.col("market_cap") >= threshold)


def _filter_price(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """last_price > threshold."""
    return df.filter(pl.col("last_price") >= threshold)


def _filter_days_listed(df: pl.DataFrame, min_days: int) -> pl.DataFrame:
    """days_listed >= min_days."""
    return df.filter(pl.col("days_listed") >= min_days)


def _filter_status(df: pl.DataFrame) -> pl.DataFrame:
    """status == 'Active' (not halted, not delisted)."""
    return df.filter(pl.col("status").str.to_lowercase() == "active")


def pre_filter(
    df: pl.DataFrame,
    *,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    min_price: float = DEFAULT_MIN_PRICE,
    min_days_listed: int = DEFAULT_MIN_DAYS_LISTED,
) -> pl.DataFrame:
    """Apply all pre-filter conditions in a single pass.

    Returns only tickers that pass ALL conditions:
      - avg_dollar_volume_20d >= min_dollar_volume
      - market_cap >= min_market_cap
      - last_price >= min_price
      - days_listed >= min_days_listed
      - status is "Active" (case-insensitive)

    Args:
        df: DataFrame with columns: ticker, avg_dollar_volume_20d, market_cap,
            last_price, days_listed, status.
        min_dollar_volume: Minimum 20-day average dollar volume.
        min_market_cap: Minimum market capitalization.
        min_price: Minimum last traded price.
        min_days_listed: Minimum number of trading days since IPO.

    Returns:
        Filtered DataFrame preserving all input columns.
    """
    if df.height == 0:
        return df

    result = (
        df.pipe(_filter_dollar_volume, min_dollar_volume)
        .pipe(_filter_market_cap, min_market_cap)
        .pipe(_filter_price, min_price)
        .pipe(_filter_days_listed, min_days_listed)
        .pipe(_filter_status)
    )
    return result
