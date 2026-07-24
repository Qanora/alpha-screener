"""On-demand current-universe walk-forward diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import polars as pl

from alphascreener.corporate_actions import CorporateActionDataError
from alphascreener.evaluation import (
    MIN_OUTCOME_COVERAGE,
    compute_forward_labels,
    evaluate_daily_rankings,
    mature_predictions,
)
from alphascreener.market_calendar import infer_market_dates, require_spy
from alphascreener.prediction_contract import (
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_TOP_K,
    FORECAST_HORIZON_SESSIONS,
    INPUT_LOOKBACK_SESSIONS,
    MAX_BACKTEST_DAYS,
    RISK_RERANK_CANDIDATES,
    STRATEGY_VERSION,
)
from alphascreener.ranking import (
    apply_definitive_transaction_filter,
    rank_candidate_dates,
)

BACKTEST_UNIVERSE_SOURCE = "current-directory"


def run_backtest(
    ohlcv: pl.DataFrame,
    *,
    days: int = DEFAULT_BACKTEST_DAYS,
    corporate_action_status_provider: (
        Callable[[Sequence[str], date], Mapping[str, object]] | None
    ) = None,
) -> pl.DataFrame:
    """Recompute the latest matured decision dates without persisting results.

    Once the requested dates can be identified from the market calendar, a
    problem on one date is represented by an INVALID row instead of aborting
    the remaining walk-forward simulation.
    """
    if not 1 <= days <= MAX_BACKTEST_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_BACKTEST_DAYS}")
    required = {"ticker", "dt", "close", "volume"}
    if missing := required - set(ohlcv.columns):
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")

    data = (
        ohlcv.with_columns(pl.col("dt").cast(pl.Date))
        .unique(subset=["ticker", "dt"], keep="last")
        .sort(["ticker", "dt"])
    )
    require_spy(data)
    market_dates = infer_market_dates(data)
    matured_date_count = len(market_dates) - FORECAST_HORIZON_SESSIONS
    if matured_date_count < days:
        raise ValueError(
            f"SPY market calendar has {max(0, matured_date_count)} matured dates; {days} requested"
        )

    first_decision_index = matured_date_count - days
    decision_dates = market_dates[first_decision_index:matured_date_count]
    first_required_index = max(
        0,
        first_decision_index - INPUT_LOOKBACK_SESSIONS + 1,
    )
    required_dates = market_dates[first_required_index:]
    backtest_data = data.filter(pl.col("dt").is_in(required_dates))
    labels = compute_forward_labels(backtest_data).filter(pl.col("dt").is_in(decision_dates))
    rankings = rank_candidate_dates(backtest_data, decision_dates)
    _prefetch_corporate_actions(
        rankings,
        decision_dates,
        corporate_action_status_provider,
    )
    records: list[dict[str, object]] = []
    # SEC search caches cover an as-of date and can safely answer earlier
    # decisions after filtering by each filing's acceptance timestamp.  Work
    # newest-to-oldest so a 45-day replay fetches each issuer snapshot once.
    for decision_index in reversed(range(first_decision_index, matured_date_count)):
        decision_date = market_dates[decision_index]
        result_date = market_dates[decision_index + FORECAST_HORIZON_SESSIONS]
        if decision_index < INPUT_LOOKBACK_SESSIONS - 1:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    "insufficient_history",
                )
            )
            continue

        spy_problem = _spy_history_problem(
            backtest_data,
            market_dates,
            decision_index,
        )
        if spy_problem is not None:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    spy_problem,
                )
            )
            continue

        ranking = (
            rankings.filter(pl.col("decision_date") == decision_date)
            .select("ticker", "score", "rank")
            .sort("rank")
        )
        if corporate_action_status_provider is not None:
            try:
                ranking, _ = apply_definitive_transaction_filter(
                    ranking,
                    decision_date,
                    status_provider=corporate_action_status_provider,
                )
            except (CorporateActionDataError, OSError, ValueError):
                records.append(
                    _invalid_record(
                        decision_date,
                        result_date,
                        "corporate_action_status_unavailable",
                        universe_size=ranking.height,
                    )
                )
                continue
        if ranking.height < DEFAULT_TOP_K:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    f"eligible_universe_below_top_{DEFAULT_TOP_K}",
                    universe_size=ranking.height,
                )
            )
            continue

        predictions = ranking.with_columns(
            pl.lit(decision_date).cast(pl.Date).alias("decision_date"),
            pl.lit(STRATEGY_VERSION).alias("strategy_version"),
            pl.lit(ranking.height).cast(pl.Int64).alias("universe_size"),
        )
        matured = mature_predictions(predictions, labels)
        outcome_coverage = matured["forward_return"].is_not_null().sum() / ranking.height
        if outcome_coverage < MIN_OUTCOME_COVERAGE:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    "outcome_coverage_below_90pct",
                    universe_size=ranking.height,
                    outcome_coverage=outcome_coverage,
                )
            )
            continue
        top_ranks = set(
            matured.filter(
                (pl.col("rank") <= DEFAULT_TOP_K) & pl.col("forward_return").is_not_null()
            )["rank"].to_list()
        )
        if top_ranks != set(range(1, DEFAULT_TOP_K + 1)):
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    f"top_{DEFAULT_TOP_K}_outcomes_incomplete",
                    universe_size=ranking.height,
                    outcome_coverage=outcome_coverage,
                )
            )
            continue
        if outcome_coverage < 1.0:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    "complete_universe_outcomes_required",
                    universe_size=ranking.height,
                    outcome_coverage=outcome_coverage,
                )
            )
            continue

        daily = evaluate_daily_rankings(matured)
        if daily.height != 1:
            records.append(
                _invalid_record(
                    decision_date,
                    result_date,
                    "evaluation_failed",
                    universe_size=ranking.height,
                    outcome_coverage=outcome_coverage,
                )
            )
            continue
        metric = daily.row(0, named=True)
        precision = float(metric["precision_at_k"])
        records.append(
            {
                "strategy_version": str(metric["strategy_version"]),
                "decision_date": decision_date,
                "result_date": result_date,
                "status": "VALID",
                "invalid_reason": None,
                "universe_size": int(metric["universe_size"]),
                "outcome_coverage": float(metric["outcome_coverage"]),
                "hits_at_10": round(precision * DEFAULT_TOP_K),
                "precision_at_10": precision,
                "base_explosion_rate": float(metric["base_explosion_rate"]),
                "downside_at_10": float(metric["downside_at_k"]),
                "catastrophic_loss_at_10": float(metric["catastrophic_loss_at_k"]),
                "adverse_path_at_10": (
                    None
                    if metric["adverse_path_at_k"] is None
                    else float(metric["adverse_path_at_k"])
                ),
                "basket_return_14": float(metric["basket_return_14"]),
                "passed": bool(metric["passed"]),
                "universe_source": BACKTEST_UNIVERSE_SOURCE,
            }
        )
    return pl.DataFrame(records, schema=_backtest_schema()).sort("decision_date")


def _prefetch_corporate_actions(
    rankings: pl.DataFrame,
    decision_dates: Sequence[date],
    status_provider: object | None,
) -> None:
    """Warm production SEC caches for the union of preliminary daily leaders."""
    if status_provider is None or not decision_dates:
        return
    prefetch = getattr(status_provider, "prefetch", None)
    if not callable(prefetch):
        return
    leaders = (
        rankings.filter(pl.col("rank") <= RISK_RERANK_CANDIDATES)
        .select("ticker")
        .unique()
        .sort("ticker")["ticker"]
        .to_list()
    )
    if not leaders:
        return
    try:
        prefetch(leaders, decision_dates[-1])
    except (CorporateActionDataError, OSError, ValueError):
        # Prefetch is only a latency optimization.  Each date still performs
        # its own fail-closed status call and records a scoped INVALID result
        # if the required SEC state cannot be established.
        return


def _spy_history_problem(
    data: pl.DataFrame,
    market_dates: list[date],
    decision_index: int,
) -> str | None:
    """Identify a missing benchmark date without collapsing it out of time."""
    decision_date = market_dates[decision_index]
    spy_dates = set(data.filter(pl.col("ticker") == "SPY")["dt"].cast(pl.Date).to_list())
    if decision_date not in spy_dates:
        return "spy_missing_on_decision_date"
    expected = set(market_dates[decision_index - INPUT_LOOKBACK_SESSIONS + 1 : decision_index + 1])
    if not expected.issubset(spy_dates):
        return "spy_history_incomplete"
    return None


def _invalid_record(
    decision_date: date,
    result_date: date,
    reason: str,
    *,
    universe_size: int | None = None,
    outcome_coverage: float | None = None,
) -> dict[str, object]:
    return {
        "strategy_version": STRATEGY_VERSION,
        "decision_date": decision_date,
        "result_date": result_date,
        "status": "INVALID",
        "invalid_reason": reason,
        "universe_size": universe_size,
        "outcome_coverage": outcome_coverage,
        "hits_at_10": None,
        "precision_at_10": None,
        "base_explosion_rate": None,
        "downside_at_10": None,
        "catastrophic_loss_at_10": None,
        "adverse_path_at_10": None,
        "basket_return_14": None,
        "passed": None,
        "universe_source": BACKTEST_UNIVERSE_SOURCE,
    }


def _backtest_schema() -> dict[str, pl.DataType]:
    return {
        "strategy_version": pl.String,
        "decision_date": pl.Date,
        "result_date": pl.Date,
        "status": pl.String,
        "invalid_reason": pl.String,
        "universe_size": pl.Int64,
        "outcome_coverage": pl.Float64,
        "hits_at_10": pl.Int64,
        "precision_at_10": pl.Float64,
        "base_explosion_rate": pl.Float64,
        "downside_at_10": pl.Float64,
        "catastrophic_loss_at_10": pl.Float64,
        "adverse_path_at_10": pl.Float64,
        "basket_return_14": pl.Float64,
        "passed": pl.Boolean,
        "universe_source": pl.String,
    }
