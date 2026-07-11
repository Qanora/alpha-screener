"""Low-degree features computed from the 60-session price and volume window."""

from __future__ import annotations

import polars as pl


def compute_60d_features(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """Add momentum, breakout, volatility, volume, and market-relative features.

    All rolling windows are 60 sessions or shorter and are calculated per
    ticker.  The caller selects the decision-date row only after this function
    returns, so no value uses future observations.
    """
    required = {"ticker", "dt", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    data = ohlcv.sort(["ticker", "dt"])
    ret_5 = pl.col("close") / pl.col("close").shift(5).over("ticker") - 1.0
    ret_20 = pl.col("close") / pl.col("close").shift(20).over("ticker") - 1.0
    high_60 = pl.col("close").rolling_max(60).over("ticker")
    daily_return = pl.col("close").pct_change().over("ticker")
    volume_mean = pl.col("volume").rolling_mean(20).over("ticker")
    volume_std = pl.col("volume").rolling_std(20).over("ticker")

    features = data.with_columns(
        ret_5.alias("return_5d"),
        ret_20.alias("return_20d"),
        (pl.col("close") / high_60 - 1.0).alias("distance_to_60d_high"),
        (
            daily_return.rolling_std(5).over("ticker")
            / daily_return.rolling_std(20).over("ticker")
        ).alias("volatility_ratio_5_20"),
        pl.when(volume_std > 0)
        .then((pl.col("volume") - volume_mean) / volume_std)
        .otherwise(0.0)
        .alias("volume_zscore_20"),
    )
    market_returns = features.filter(pl.col("ticker") == "SPY").select(
        "dt", pl.col("return_20d").alias("market_return_20d")
    )
    return features.join(market_returns, on="dt", how="left").with_columns(
        (pl.col("return_20d") - pl.col("market_return_20d")).alias("relative_strength_20d")
    )
