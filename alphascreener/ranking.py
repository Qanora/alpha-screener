"""Point-in-time candidate ranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import polars as pl

from alphascreener.features import compute_60d_features
from alphascreener.market_calendar import infer_market_dates
from alphascreener.prediction_contract import (
    INPUT_LOOKBACK_SESSIONS,
    MAX_CANDIDATES,
    MIN_AVERAGE_DOLLAR_VOLUME,
    MIN_CANDIDATE_CLOSE,
    MIN_MEDIAN_DOLLAR_VOLUME_PRIOR_20D,
    MIN_VALID_PRICE_VOLUME_SESSIONS_PRIOR_20D,
    RISK_RERANK_CANDIDATES,
)

RANK_V6_SIGNALS = (
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
        raise ValueError(f"SPY benchmark unavailable on {cutoff}: insufficient_history")
    window = ohlcv.filter(pl.col("dt").is_in(market_dates))
    rankings = rank_candidate_dates(window, [cutoff])
    ranking = (
        rankings.filter(pl.col("decision_date") == cutoff)
        .select("ticker", "score", "rank")
        .sort("rank")
    )
    if ranking.is_empty():
        features = compute_60d_features(window)
        benchmark = features.filter((pl.col("ticker") == "SPY") & (pl.col("dt") == cutoff))
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
    features = compute_60d_features(ohlcv)
    candidates = select_eligible_candidate_features(features, decision_dates)
    return score_rank_v6(candidates)


def apply_definitive_transaction_filter(
    ranking: pl.DataFrame,
    decision_date: date,
    *,
    status_provider: Callable[[Sequence[str], date], Mapping[str, object]] | None = None,
    inspect_count: int = RISK_RERANK_CANDIDATES,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Remove point-in-time definitive transactions from the preliminary leaders.

    The check expands when an exclusion promotes a previously unchecked stock,
    so every member of the final preliminary Top-N has a complete SEC status.
    The remaining full ranking is kept in original score order and receives
    contiguous ranks.
    """
    if inspect_count <= 0:
        raise ValueError("inspect_count must be positive")
    if ranking.is_empty():
        return ranking, {}
    required = {"ticker", "score", "rank"}
    if missing := required - set(ranking.columns):
        raise ValueError(f"ranking missing columns: {sorted(missing)}")
    if status_provider is None:
        from alphascreener.corporate_actions import corporate_action_statuses

        status_provider = corporate_action_statuses

    ordered = ranking.sort("rank")
    checked: dict[str, object] = {}
    excluded: set[str] = set()
    while True:
        leaders = [
            str(ticker)
            for ticker in ordered.filter(~pl.col("ticker").is_in(sorted(excluded)))
            .head(inspect_count)["ticker"]
            .to_list()
        ]
        unchecked = [ticker for ticker in leaders if ticker not in checked]
        if not unchecked:
            break
        batch = dict(status_provider(unchecked, decision_date))
        if set(batch) != set(unchecked):
            missing = sorted(set(unchecked) - set(batch))
            extra = sorted(set(batch) - set(unchecked))
            raise ValueError(
                f"corporate-action status coverage mismatch: missing={missing}, extra={extra}"
            )
        for ticker, status in batch.items():
            if not hasattr(status, "exclude_from_ranking"):
                raise ValueError(f"corporate-action status for {ticker} has no exclusion decision")
            checked[ticker] = status
            if bool(getattr(status, "exclude_from_ranking")):
                excluded.add(ticker)

    filtered = ordered.filter(~pl.col("ticker").is_in(sorted(excluded))).drop("rank")
    filtered = filtered.with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int64))
    return filtered.select(ranking.columns), checked


def select_eligible_candidate_features(
    features: pl.DataFrame,
    decision_dates: Sequence[date],
) -> pl.DataFrame:
    """Return the exact daily prefiltered universe shared by every ranker."""
    required = {
        "ticker",
        "dt",
        "raw_close",
        "history_complete_60d",
        "average_dollar_volume_20d",
        "median_dollar_volume_prior_20d",
        "valid_price_volume_sessions_prior_20d",
        *RANK_V6_SIGNALS,
    }
    if missing := required - set(features.columns):
        raise ValueError(f"feature data missing columns: {sorted(missing)}")
    features = features.filter(pl.col("dt").is_in(list(decision_dates)))
    eligible = features.filter(
        pl.col("history_complete_60d")
        & (pl.col("raw_close") >= MIN_CANDIDATE_CLOSE)
        & (pl.col("average_dollar_volume_20d") >= MIN_AVERAGE_DOLLAR_VOLUME)
        & (pl.col("median_dollar_volume_prior_20d") >= MIN_MEDIAN_DOLLAR_VOLUME_PRIOR_20D)
        & (
            pl.col("valid_price_volume_sessions_prior_20d")
            >= MIN_VALID_PRICE_VOLUME_SESSIONS_PRIOR_20D
        )
    )
    benchmark_dates = eligible.filter(pl.col("ticker") == "SPY")["dt"].unique()
    candidates = (
        eligible.filter((pl.col("ticker") != "SPY") & pl.col("dt").is_in(benchmark_dates.implode()))
        .sort(
            ["dt", "average_dollar_volume_20d", "ticker"],
            descending=[False, True, False],
        )
        .with_columns(pl.col("ticker").cum_count().over("dt").alias("_liquidity_rank"))
        .filter(pl.col("_liquidity_rank") <= MAX_CANDIDATES)
        .drop("_liquidity_rank")
    )
    return candidates


def score_rank_v6(candidates: pl.DataFrame) -> pl.DataFrame:
    """Apply the frozen rank-v6 score to an already eligible feature panel."""
    if candidates.is_empty():
        return _empty_rankings()
    required = {"ticker", "dt", *RANK_V6_SIGNALS}
    if missing := required - set(candidates.columns):
        raise ValueError(f"candidate features missing columns: {sorted(missing)}")
    ranked = candidates.with_columns(
        [
            pl.col(signal).fill_null(0.0).rank("average").over("dt").alias(f"_rank_{signal}")
            for signal in RANK_V6_SIGNALS
        ]
    ).with_columns(
        pl.mean_horizontal([pl.col(f"_rank_{signal}") for signal in RANK_V6_SIGNALS]).alias("score")
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
    valid_sessions = row["valid_price_volume_sessions_prior_20d"]
    if valid_sessions is None or valid_sessions < MIN_VALID_PRICE_VOLUME_SESSIONS_PRIOR_20D:
        return "insufficient_liquidity_history"
    median_dollar_volume = row["median_dollar_volume_prior_20d"]
    if median_dollar_volume is None or median_dollar_volume < MIN_MEDIAN_DOLLAR_VOLUME_PRIOR_20D:
        return "low_median_dollar_volume"
    return None


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "decision_date": pl.Date,
            "score": pl.Float64,
            "rank": pl.Int64,
        }
    )
