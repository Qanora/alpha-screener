"""Recent current-universe walk-forward evidence for each screen."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_data_home
from alphascreener.evaluation import (
    compute_forward_labels,
    evaluate_daily_rankings,
    mature_predictions,
)
from alphascreener.prediction_contract import (
    DEFAULT_TOP_K,
    FORECAST_HORIZON_SESSIONS,
    INPUT_LOOKBACK_SESSIONS,
    REQUIRED_BACKTEST_DAYS,
    STRATEGY_VERSION,
)
from alphascreener.ranking import rank_candidates

BACKTEST_UNIVERSE_SOURCE = "current-directory"


def run_recent_backtest(
    ohlcv: pl.DataFrame,
    *,
    required_days: int = REQUIRED_BACKTEST_DAYS,
) -> pl.DataFrame:
    """Walk the current strategy over the latest consecutive matured market dates."""
    if required_days <= 0:
        raise ValueError("required_days must be positive")
    market_dates = (
        ohlcv.filter(pl.col("ticker") == "SPY")["dt"].unique().sort().to_list()
    )
    last_decision_index = len(market_dates) - FORECAST_HORIZON_SESSIONS - 1
    first_decision_index = last_decision_index - required_days + 1
    if first_decision_index < INPUT_LOOKBACK_SESSIONS - 1:
        raise ValueError(
            f"need at least {required_days} matured backtest dates with "
            f"{INPUT_LOOKBACK_SESSIONS} prior sessions"
        )

    labels = compute_forward_labels(ohlcv)
    records: list[dict[str, str | date | float | int | bool]] = []
    for decision_index in range(first_decision_index, last_decision_index + 1):
        decision_date = market_dates[decision_index]
        result_date = market_dates[decision_index + FORECAST_HORIZON_SESSIONS]
        history_dates = market_dates[
            decision_index - INPUT_LOOKBACK_SESSIONS + 1 : decision_index + 1
        ]
        history = ohlcv.filter(pl.col("dt").is_in(history_dates))
        ranking, ranked_date = rank_candidates(history)
        if ranked_date != decision_date or ranking.height < DEFAULT_TOP_K:
            raise ValueError(f"backtest universe incomplete on {decision_date}")
        predictions = ranking.with_columns(
            pl.lit(decision_date).cast(pl.Date).alias("decision_date"),
            pl.lit(STRATEGY_VERSION).alias("strategy_version"),
            pl.lit(ranking.height).cast(pl.Int64).alias("universe_size"),
        )
        daily = evaluate_daily_rankings(mature_predictions(predictions, labels))
        if daily.height != 1:
            raise ValueError(f"backtest outcomes incomplete on {decision_date}")
        metric = daily.row(0, named=True)
        records.append({
            **{key: value for key, value in metric.items() if key != "precision_at_k"},
            "precision_at_10": metric["precision_at_k"],
            "result_date": result_date,
            "universe_source": BACKTEST_UNIVERSE_SOURCE,
        })
    return pl.DataFrame(records, schema=_backtest_schema()).sort("decision_date")


def write_backtest_records(records: pl.DataFrame) -> Path:
    """Atomically replace the current strategy's reproducible backtest evidence."""
    required = set(_backtest_schema())
    if missing := required - set(records.columns):
        raise ValueError(f"backtest records missing columns: {sorted(missing)}")
    strategies = records["strategy_version"].unique().to_list()
    if records.height < REQUIRED_BACKTEST_DAYS or len(strategies) != 1:
        raise ValueError("backtest commit requires one strategy and at least three dates")

    root = get_data_home() / "backtests"
    path = root / f"strategy={strategies[0]}"
    output = path / "recent.parquet"
    temporary = path / "recent.parquet.tmp"
    with exclusive_file_lock(root / ".write.lock"):
        path.mkdir(parents=True, exist_ok=True)
        try:
            records.select(_backtest_schema().keys()).write_parquet(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def _backtest_schema() -> dict[str, pl.DataType]:
    return {
        "strategy_version": pl.String,
        "decision_date": pl.Date,
        "universe_size": pl.Int64,
        "outcome_coverage": pl.Float64,
        "precision_at_10": pl.Float64,
        "base_explosion_rate": pl.Float64,
        "passed": pl.Boolean,
        "result_date": pl.Date,
        "universe_source": pl.String,
    }
