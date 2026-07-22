"""Point-in-time candidate ranking."""

from __future__ import annotations

from datetime import date

import polars as pl

from alphascreener.features import compute_60d_features
from alphascreener.prediction_contract import INPUT_LOOKBACK_SESSIONS
from alphascreener.universe import build_universe_snapshot


def rank_candidates(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, date]:
    """Rank the complete eligible universe at the latest date in ``ohlcv``."""
    cutoff = ohlcv["dt"].max()
    market_dates = (
        ohlcv.filter((pl.col("ticker") == "SPY") & (pl.col("dt") <= cutoff))["dt"]
        .unique()
        .sort()
        .tail(INPUT_LOOKBACK_SESSIONS)
        .to_list()
    )
    window = ohlcv.filter(pl.col("dt").is_in(market_dates))
    snapshot = build_universe_snapshot(window, cutoff_date=cutoff)
    benchmark = snapshot.filter(pl.col("ticker") == "SPY")
    if benchmark.height != 1 or not benchmark.item(0, "eligible"):
        reason = (
            "missing"
            if benchmark.is_empty()
            else str(benchmark.item(0, "exclusion_reason"))
        )
        raise ValueError(f"SPY benchmark unavailable on {cutoff}: {reason}")
    eligible = snapshot.filter(pl.col("eligible") & (pl.col("ticker") != "SPY"))[
        "ticker"
    ].to_list()
    if not eligible:
        return pl.DataFrame(schema={"ticker": pl.String, "score": pl.Float64}), cutoff
    feature_tickers = [*eligible, "SPY"]
    feature_window = window.filter(pl.col("ticker").is_in(feature_tickers))
    features = compute_60d_features(feature_window).filter(
        (pl.col("dt") == cutoff) & (pl.col("ticker") != "SPY")
    )
    signals = [
        "return_5d",
        "return_20d",
        "distance_to_60d_high",
        "volume_zscore_20",
        "relative_strength_20d",
    ]
    ranked = features.with_columns([
        pl.col(signal).fill_null(0.0).rank("average").alias(f"_rank_{signal}")
        for signal in signals
    ]).with_columns(
        pl.mean_horizontal([pl.col(f"_rank_{signal}") for signal in signals]).alias("score")
    )
    ranking = (
        ranked.select("ticker", "score")
        .sort("score", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.Int64))
    )
    return ranking, cutoff
