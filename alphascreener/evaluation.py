"""Future-14-session labels, immutable rankings, and strategy-aware metrics."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_data_home
from alphascreener.market_calendar import infer_market_dates, require_spy
from alphascreener.prediction_contract import DEFAULT_TOP_K, ExplosionLabelSpec

MIN_OUTCOME_COVERAGE = 0.90
DAILY_PRECISION_TARGET = 0.10
REQUIRED_CONSECUTIVE_PASSES = 5


def compute_forward_labels(
    ohlcv: pl.DataFrame,
    *,
    spec: ExplosionLabelSpec = ExplosionLabelSpec(),
) -> pl.DataFrame:
    """Compute returns on the exact 14th later NYSE session.

    A per-ticker shift is not equivalent to a market-session horizon when a
    security is suspended or has a missing observation.  A stable exchange
    calendar defines the horizon, and future prices are joined on that exact
    result date.
    """
    required = {"ticker", "dt", "close"}
    if missing := required - set(ohlcv.columns):
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    require_spy(ohlcv)
    market_dates = infer_market_dates(ohlcv)
    if len(market_dates) <= spec.horizon_sessions:
        raise ValueError(
            f"SPY market calendar needs more than {spec.horizon_sessions} sessions"
        )
    calendar = pl.DataFrame({
        "dt": market_dates[:-spec.horizon_sessions],
        "result_date": market_dates[spec.horizon_sessions:],
    })
    prices = (
        ohlcv.select("ticker", pl.col("dt").cast(pl.Date), "close")
        .unique(subset=["ticker", "dt"], keep="last")
    )
    future_prices = prices.select(
        "ticker",
        pl.col("dt").alias("result_date"),
        pl.col("close").alias("future_close"),
    )
    labels = prices.join(calendar, on="dt", how="inner").join(
        future_prices,
        on=["ticker", "result_date"],
        how="inner",
    )
    return (
        labels.with_columns(
            ((pl.col("future_close") / pl.col("close")) - 1.0).alias("forward_return")
        )
        .select(_label_schema().keys())
        .sort(["dt", "ticker"])
    )


def write_prediction_ledger(predictions: pl.DataFrame) -> Path:
    """Persist one complete decision-date ranking before outcomes are observable."""
    required = {
        "ticker",
        "decision_date",
        "score",
        "rank",
        "strategy_version",
        "universe_size",
    }
    if missing := required - set(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    dates = predictions["decision_date"].cast(pl.Date).unique().to_list()
    strategies = predictions["strategy_version"].unique().to_list()
    sizes = predictions["universe_size"].unique().to_list()
    if len(dates) != 1 or len(strategies) != 1 or len(sizes) != 1:
        raise ValueError("ledger write requires one date, strategy, and universe size")
    strategy = str(strategies[0])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", strategy):
        raise ValueError("strategy_version contains unsafe path characters")
    if sizes[0] != predictions.height:
        raise ValueError("universe_size must equal the complete ranking size")
    expected_ranks = list(range(1, predictions.height + 1))
    if sorted(predictions["rank"].cast(pl.Int64).to_list()) != expected_ranks:
        raise ValueError("rank must contain every ordinal from 1 through universe_size")

    prediction_root = get_data_home() / "predictions"
    path = (
        prediction_root
        / f"dt={dates[0].isoformat()}"
        / f"strategy={strategy}"
    )
    with exclusive_file_lock(prediction_root / ".write.lock"):
        path.mkdir(parents=True, exist_ok=True)
        output = path / "ranking.parquet"
        if output.exists():
            raise FileExistsError(
                f"prediction ledger already exists for {dates[0].isoformat()} and {strategy}"
            )
        serialized = predictions.with_columns(
            pl.col("decision_date").cast(pl.Date),
            pl.col("rank").cast(pl.Int64),
            pl.col("universe_size").cast(pl.Int64),
        )
        temporary = path / "ranking.parquet.tmp"
        try:
            serialized.write_parquet(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def read_prediction_ledger() -> pl.DataFrame:
    """Read complete rankings written under the current ledger contract."""
    paths = sorted(
        (get_data_home() / "predictions").glob("dt=*/strategy=*/ranking.parquet")
    )
    if not paths:
        return pl.DataFrame(schema=_prediction_schema())
    frames: list[pl.DataFrame] = []
    required = set(_prediction_schema())
    for path in paths:
        frame = pl.read_parquet(path)
        if missing := required - set(frame.columns):
            raise ValueError(f"prediction ledger {path} missing columns: {sorted(missing)}")
        frames.append(
            frame.select(_prediction_schema().keys()).with_columns(
                pl.col("decision_date").cast(pl.Date),
                pl.col("rank").cast(pl.Int64),
                pl.col("universe_size").cast(pl.Int64),
            )
        )
    return pl.concat(frames)


def mature_predictions(
    predictions: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    spec: ExplosionLabelSpec = ExplosionLabelSpec(),
) -> pl.DataFrame:
    """Join outcomes and classify explosions within each ranked eligible universe."""
    required_predictions = set(_prediction_schema())
    required_labels = {"ticker", "dt", "result_date", "forward_return"}
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    joined = predictions.join(
        labels.select(["ticker", "dt", "result_date", "forward_return"]),
        left_on=["ticker", "decision_date"],
        right_on=["ticker", "dt"],
        how="inner",
    )
    matured: list[pl.DataFrame] = []
    for _, group in joined.group_by(
        ["strategy_version", "decision_date"], maintain_order=True
    ):
        threshold = spec.threshold(group["forward_return"].to_list())
        matured.append(
            group.with_columns(
                pl.lit(threshold).alias("hit_threshold"),
                (pl.col("forward_return") >= threshold).alias("is_explosion"),
            )
        )
    if not matured:
        return joined.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("hit_threshold"),
            pl.lit(None, dtype=pl.Boolean).alias("is_explosion"),
        )
    return pl.concat(matured)


def evaluate_daily_rankings(
    matured: pl.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pl.DataFrame:
    """Return valid per-date Precision@K measurements without rank substitution."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "decision_date",
        "score",
        "rank",
        "strategy_version",
        "universe_size",
        "is_explosion",
        "forward_return",
    }
    if missing := required - set(matured.columns):
        raise ValueError(f"matured predictions missing columns: {sorted(missing)}")
    if matured.is_empty():
        return pl.DataFrame(schema=_daily_metric_schema())

    results: list[dict[str, str | date | float | int | bool]] = []
    for (strategy, decision_date), group in matured.group_by(
        ["strategy_version", "decision_date"], maintain_order=True
    ):
        sizes = group["universe_size"].drop_nulls().unique().to_list()
        if len(sizes) != 1 or sizes[0] < top_k:
            continue
        universe_size = int(sizes[0])
        outcome_coverage = group.height / universe_size
        if outcome_coverage < MIN_OUTCOME_COVERAGE:
            continue
        expected_top_ranks = set(range(1, top_k + 1))
        ordered = group.filter(pl.col("rank") <= top_k).sort("rank")
        if set(ordered["rank"].to_list()) != expected_top_ranks:
            continue
        base_rate = float(group["is_explosion"].mean())
        precision = float(ordered["is_explosion"].mean())
        results.append({
            "strategy_version": str(strategy),
            "decision_date": decision_date,
            "universe_size": universe_size,
            "outcome_coverage": outcome_coverage,
            "precision_at_k": precision,
            "base_explosion_rate": base_rate,
            "passed": precision >= DAILY_PRECISION_TARGET and precision > base_rate,
        })
    return pl.DataFrame(results, schema=_daily_metric_schema()).sort(
        ["strategy_version", "decision_date"]
    )


def longest_consecutive_passes(
    daily: pl.DataFrame,
    market_dates: list[date],
    *,
    strategy_version: str,
) -> int:
    """Return the longest run of passing metrics on consecutive market dates."""
    positions = {value: index for index, value in enumerate(market_dates)}
    rows = (
        daily.filter(pl.col("strategy_version") == strategy_version)
        .sort("decision_date")
        .select("decision_date", "passed")
        .iter_rows()
    )
    longest = 0
    current = 0
    previous_position: int | None = None
    for decision_date, passed in rows:
        position = positions.get(decision_date)
        if not passed or position is None:
            current = 0
        elif previous_position is not None and position == previous_position + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous_position = position
    return longest


def _prediction_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "decision_date": pl.Date,
        "score": pl.Float64,
        "rank": pl.Int64,
        "strategy_version": pl.String,
        "universe_size": pl.Int64,
    }


def _label_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "dt": pl.Date,
        "result_date": pl.Date,
        "close": pl.Float64,
        "future_close": pl.Float64,
        "forward_return": pl.Float64,
    }


def _daily_metric_schema() -> dict[str, pl.DataType]:
    return {
        "strategy_version": pl.String,
        "decision_date": pl.Date,
        "universe_size": pl.Int64,
        "outcome_coverage": pl.Float64,
        "precision_at_k": pl.Float64,
        "base_explosion_rate": pl.Float64,
        "passed": pl.Boolean,
    }
