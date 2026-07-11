"""Future-14-session labels, prediction ledger, and out-of-sample metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from alphascreener.acceptance import block_bootstrap_ci
from alphascreener.data.paths import get_data_home
from alphascreener.prediction_contract import DEFAULT_TOP_K, ExplosionLabelSpec


def compute_forward_labels(
    ohlcv: pl.DataFrame,
    *,
    spec: ExplosionLabelSpec = ExplosionLabelSpec(),
) -> pl.DataFrame:
    """Label each decision date using only its later 14-session close."""
    required = {"ticker", "dt", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    data = ohlcv.sort(["ticker", "dt"]).with_columns(
        pl.col("close").shift(-spec.horizon_sessions).over("ticker").alias("future_close")
    )
    data = data.with_columns(
        ((pl.col("future_close") / pl.col("close")) - 1.0).alias("forward_return")
    ).drop_nulls("forward_return")

    labeled: list[pl.DataFrame] = []
    for (decision_date,), group in data.group_by("dt", maintain_order=True):
        returns = group["forward_return"].to_list()
        threshold = spec.threshold(returns)
        labeled.append(
            group.with_columns(
                pl.lit(threshold).alias("hit_threshold"),
                (pl.col("forward_return") >= threshold).alias("is_explosion"),
            )
        )
    if not labeled:
        return pl.DataFrame(schema=_label_schema())
    return pl.concat(labeled).select(_label_schema().keys()).sort(["dt", "ticker"])


def write_prediction_ledger(predictions: pl.DataFrame) -> Path:
    """Persist one decision-date ranking before its outcome is observable."""
    required = {"ticker", "decision_date", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    dates = predictions["decision_date"].cast(pl.Date).unique().to_list()
    if len(dates) != 1:
        raise ValueError("ledger write requires exactly one decision_date")
    path = get_data_home() / "predictions" / f"dt={dates[0].isoformat()}"
    path.mkdir(parents=True, exist_ok=True)
    output = path / "ranking.parquet"
    predictions.with_columns(pl.col("decision_date").cast(pl.Date)).write_parquet(output)
    return output


def mature_predictions(predictions: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Join a previously stored ranking to labels only after they mature."""
    required_predictions = {"ticker", "decision_date", "score"}
    required_labels = {"ticker", "dt", "forward_return", "is_explosion"}
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    return predictions.join(
        labels.select(["ticker", "dt", "forward_return", "is_explosion"]),
        left_on=["ticker", "decision_date"],
        right_on=["ticker", "dt"],
        how="inner",
    )


def evaluate_rankings(
    matured: pl.DataFrame, *, top_k: int = DEFAULT_TOP_K
) -> dict[str, float | int | None]:
    """Aggregate daily Top-K precision, lift, return, and bootstrap CI."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {"decision_date", "score", "is_explosion", "forward_return"}
    if missing := required - set(matured.columns):
        raise ValueError(f"matured predictions missing columns: {sorted(missing)}")

    daily: list[tuple[float, float, float]] = []
    for _, group in matured.group_by("decision_date", maintain_order=True):
        ordered = group.sort("score", descending=True).head(top_k)
        base_rate = float(group["is_explosion"].mean())
        precision = float(ordered["is_explosion"].mean())
        lift = precision / base_rate if base_rate > 0 else None
        daily.append((precision, lift or 0.0, float(ordered["forward_return"].mean())))
    if not daily:
        return {"days": 0, "precision_at_k": None, "lift_at_k": None, "mean_forward_return": None,
                "ci_lower": None, "ci_upper": None}
    precisions = np.array([item[0] for item in daily])
    ci_lower, ci_upper = block_bootstrap_ci(precisions, np.mean)
    return {
        "days": len(daily),
        "precision_at_k": float(precisions.mean()),
        "lift_at_k": float(np.mean([item[1] for item in daily])),
        "mean_forward_return": float(np.mean([item[2] for item in daily])),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _label_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "dt": pl.Date,
        "close": pl.Float64,
        "future_close": pl.Float64,
        "forward_return": pl.Float64,
        "hit_threshold": pl.Float64,
        "is_explosion": pl.Boolean,
    }
