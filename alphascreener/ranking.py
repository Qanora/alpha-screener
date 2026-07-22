"""Point-in-time candidate ranking."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from alphascreener.features import compute_60d_features
from alphascreener.market_calendar import infer_market_dates
from alphascreener.prediction_contract import (
    INPUT_LOOKBACK_SESSIONS,
    MAX_CANDIDATES,
    MIN_AVERAGE_DOLLAR_VOLUME,
    MIN_CANDIDATE_CLOSE,
)

_RANK_SIGNALS = (
    "return_5d",
    "return_20d",
    "distance_to_60d_high",
    "volume_zscore_20",
    "relative_strength_20d",
)


def rank_candidates(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, date]:
    """Rank the complete eligible universe at the latest date in ``ohlcv``."""
    if ohlcv.is_empty():
        raise ValueError("OHLCV data is empty")
    all_market_dates = infer_market_dates(ohlcv)
    cutoff = all_market_dates[-1]
    if ohlcv.filter(pl.col("ticker") == "SPY").is_empty():
        raise ValueError(f"SPY benchmark unavailable on {cutoff}: missing")
    market_dates = all_market_dates[-INPUT_LOOKBACK_SESSIONS:]
    if len(market_dates) < INPUT_LOOKBACK_SESSIONS:
        raise ValueError(
            f"SPY benchmark unavailable on {cutoff}: insufficient_history"
        )
    window = ohlcv.filter(pl.col("dt").is_in(market_dates))
    rankings = rank_candidate_dates(window, [cutoff])
    ranking = (
        rankings.filter(pl.col("decision_date") == cutoff)
        .select("ticker", "score", "rank")
        .sort("rank")
    )
    if ranking.is_empty():
        features = compute_60d_features(window)
        benchmark = features.filter(
            (pl.col("ticker") == "SPY") & (pl.col("dt") == cutoff)
        )
        reason = _benchmark_exclusion_reason(benchmark)
        if reason is not None:
            raise ValueError(f"SPY benchmark unavailable on {cutoff}: {reason}")
    return ranking, cutoff


def rank_candidate_dates(
    ohlcv: pl.DataFrame,
    decision_dates: Sequence[date],
) -> pl.DataFrame:
    """Rank many decision dates from one backward-looking feature panel."""
    if not decision_dates:
        return _empty_rankings()
    features = compute_60d_features(ohlcv).filter(
        pl.col("dt").is_in(list(decision_dates))
    )
    eligible = features.filter(
        pl.col("history_complete_60d")
        & (pl.col("raw_close") >= MIN_CANDIDATE_CLOSE)
        & (pl.col("average_dollar_volume_20d") >= MIN_AVERAGE_DOLLAR_VOLUME)
    )
    benchmark_dates = eligible.filter(pl.col("ticker") == "SPY")["dt"].unique()
    candidates = eligible.filter(
        (pl.col("ticker") != "SPY") & pl.col("dt").is_in(benchmark_dates.implode())
    ).sort(
        ["dt", "average_dollar_volume_20d", "ticker"],
        descending=[False, True, False],
    ).with_columns(
        pl.col("ticker").cum_count().over("dt").alias("_liquidity_rank")
    ).filter(
        pl.col("_liquidity_rank") <= MAX_CANDIDATES
    )
    if candidates.is_empty():
        return _empty_rankings()
    ranked = candidates.with_columns([
        pl.col(signal)
        .fill_null(0.0)
        .rank("average")
        .over("dt")
        .alias(f"_rank_{signal}")
        for signal in _RANK_SIGNALS
    ]).with_columns(
        pl.mean_horizontal([
            pl.col(f"_rank_{signal}") for signal in _RANK_SIGNALS
        ]).alias("score")
    )
    return (
        ranked.select(
            "ticker",
            pl.col("dt").alias("decision_date"),
            "score",
        )
        .sort(
            ["decision_date", "score", "ticker"],
            descending=[False, True, False],
        )
        .with_columns(
            pl.col("ticker").cum_count().over("decision_date").cast(pl.Int64).alias("rank")
        )
        .select("ticker", "decision_date", "score", "rank")
    )


def _benchmark_exclusion_reason(benchmark: pl.DataFrame) -> str | None:
    if benchmark.is_empty():
        return "missing"
    row = benchmark.row(0, named=True)
    if not row["history_complete_60d"]:
        return "insufficient_history"
    if row["raw_close"] < MIN_CANDIDATE_CLOSE:
        return "low_price"
    if row["average_dollar_volume_20d"] < MIN_AVERAGE_DOLLAR_VOLUME:
        return "low_dollar_volume"
    return None


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ticker": pl.String,
        "decision_date": pl.Date,
        "score": pl.Float64,
        "rank": pl.Int64,
    })
