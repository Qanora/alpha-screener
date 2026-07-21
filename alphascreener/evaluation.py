"""Future-14-session labels, immutable rankings, and strategy-aware metrics."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import polars as pl

from alphascreener.data.paths import get_data_home
from alphascreener.prediction_contract import DEFAULT_TOP_K, ExplosionLabelSpec

LEGACY_STRATEGY_VERSION = "legacy-unversioned"
MIN_OUTCOME_COVERAGE = 0.90
MIN_CONFIDENCE_INTERVAL_DAYS = 20


def compute_forward_labels(
    ohlcv: pl.DataFrame,
    *,
    spec: ExplosionLabelSpec = ExplosionLabelSpec(),
) -> pl.DataFrame:
    """Compute each ticker's return over the next 14 observed sessions."""
    required = {"ticker", "dt", "close"}
    if missing := required - set(ohlcv.columns):
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    data = ohlcv.sort(["ticker", "dt"]).with_columns(
        pl.col("close").shift(-spec.horizon_sessions).over("ticker").alias("future_close")
    )
    data = data.with_columns(
        ((pl.col("future_close") / pl.col("close")) - 1.0).alias("forward_return")
    ).drop_nulls("forward_return")

    return data.select(_label_schema().keys()).sort(["dt", "ticker"])


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

    path = (
        get_data_home()
        / "predictions"
        / f"dt={dates[0].isoformat()}"
        / f"strategy={strategy}"
    )
    path.mkdir(parents=True, exist_ok=True)
    output = path / "ranking.parquet"
    if output.exists():
        raise FileExistsError(
            f"prediction ledger already exists for {dates[0].isoformat()} and {strategy}"
        )
    predictions.with_columns(
        pl.col("decision_date").cast(pl.Date),
        pl.col("rank").cast(pl.Int64),
        pl.col("universe_size").cast(pl.Int64),
    ).write_parquet(output)
    return output


def read_prediction_ledger() -> pl.DataFrame:
    """Read rankings and normalize legacy three-column ledger files."""
    paths = sorted((get_data_home() / "predictions").rglob("ranking.parquet"))
    if not paths:
        return pl.DataFrame(schema=_prediction_schema())
    return pl.concat([_normalize_ledger(pl.read_parquet(path)) for path in paths])


def _normalize_ledger(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(pl.col("decision_date").cast(pl.Date))
    if "strategy_version" not in frame.columns:
        frame = frame.with_columns(pl.lit(LEGACY_STRATEGY_VERSION).alias("strategy_version"))
    if "rank" not in frame.columns:
        frame = frame.with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over(["decision_date", "strategy_version"])
            .cast(pl.Int64)
            .alias("rank")
        )
    if "universe_size" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("universe_size"))
    return frame.select(_prediction_schema().keys())


def mature_predictions(
    predictions: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    spec: ExplosionLabelSpec = ExplosionLabelSpec(),
) -> pl.DataFrame:
    """Join outcomes and classify explosions within each ranked eligible universe."""
    required_predictions = set(_prediction_schema())
    required_labels = {"ticker", "dt", "forward_return"}
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    joined = predictions.join(
        labels.select(["ticker", "dt", "forward_return"]),
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


def evaluate_rankings(
    matured: pl.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, str | float | int | None]]:
    """Evaluate each strategy independently against its complete eligible universe."""
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
        return []

    results: list[dict[str, str | float | int | None]] = []
    for (strategy,), strategy_rows in matured.group_by("strategy_version", maintain_order=True):
        daily: list[tuple[float, float | None, float, float]] = []
        skipped_days = 0
        for _, group in strategy_rows.group_by("decision_date", maintain_order=True):
            sizes = group["universe_size"].drop_nulls().unique().to_list()
            if len(sizes) != 1 or sizes[0] <= 0:
                skipped_days += 1
                continue
            outcome_coverage = group.height / int(sizes[0])
            if outcome_coverage < MIN_OUTCOME_COVERAGE:
                skipped_days += 1
                continue
            ordered = group.sort("rank").head(top_k)
            base_rate = float(group["is_explosion"].mean())
            precision = float(ordered["is_explosion"].mean())
            lift = precision / base_rate if base_rate > 0 else None
            daily.append(
                (precision, lift, float(ordered["forward_return"].mean()), outcome_coverage)
            )

        precisions = np.array([item[0] for item in daily], dtype=np.float64)
        ci_lower, ci_upper = _block_bootstrap_ci(precisions)
        lifts = [item[1] for item in daily if item[1] is not None]
        results.append({
            "strategy_version": str(strategy),
            "days": len(daily),
            "skipped_days": skipped_days,
            "precision_at_k": float(precisions.mean()) if len(precisions) else None,
            "lift_at_k": float(np.mean(lifts)) if lifts else None,
            "mean_forward_return": (
                float(np.mean([item[2] for item in daily])) if daily else None
            ),
            "mean_outcome_coverage": (
                float(np.mean([item[3] for item in daily])) if daily else None
            ),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
    return results


def _block_bootstrap_ci(data: np.ndarray) -> tuple[float | None, float | None]:
    """Return a deterministic 95% interval after enough daily observations exist."""
    if len(data) < MIN_CONFIDENCE_INTERVAL_DAYS:
        return None, None
    rng = np.random.default_rng(0)
    block_size = min(5, len(data))
    n_blocks = math.ceil(len(data) / block_size)
    samples = np.empty(1_000, dtype=np.float64)
    for index in range(len(samples)):
        blocks = [
            data[start : start + block_size]
            for start in rng.integers(0, len(data) - block_size + 1, size=n_blocks)
        ]
        samples[index] = np.mean(np.concatenate(blocks)[: len(data)])
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


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
        "close": pl.Float64,
        "future_close": pl.Float64,
        "forward_return": pl.Float64,
    }
