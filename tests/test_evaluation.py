"""Tests for future labels, versioned ledgers, and strategy-aware evaluation."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from alphascreener.evaluation import (
    LEGACY_STRATEGY_VERSION,
    _block_bootstrap_ci,
    compute_forward_labels,
    evaluate_rankings,
    mature_predictions,
    read_prediction_ledger,
    write_prediction_ledger,
)
from alphascreener.prediction_contract import ExplosionLabelSpec


def _ohlcv() -> pl.DataFrame:
    rows = []
    for ticker, multiplier in [("WIN", 1.02), ("LOSE", 1.001)]:
        for index in range(29):
            rows.append({
                "ticker": ticker,
                "dt": date(2025, 1, 1) + timedelta(days=index),
                "close": 100.0 * multiplier**index,
            })
    return pl.DataFrame(rows)


def _predictions(strategy: str = "rank-v1") -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": ["WIN", "LOSE"],
        "decision_date": [date(2025, 1, 1)] * 2,
        "score": [0.9, 0.1],
        "rank": [1, 2],
        "strategy_version": [strategy] * 2,
        "universe_size": [2, 2],
    })


def test_forward_labels_use_exactly_14_later_sessions() -> None:
    labels = compute_forward_labels(_ohlcv(), spec=ExplosionLabelSpec(absolute_return=0.01))

    assert labels.filter(pl.col("ticker") == "WIN").height == 15
    first = labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == date(2025, 1, 1)))
    assert first.item(0, "future_close") == 100.0 * 1.02**14


def test_metrics_use_the_complete_universe_and_keep_strategies_separate() -> None:
    labels = compute_forward_labels(_ohlcv(), spec=ExplosionLabelSpec(absolute_return=0.01))
    inverse = _predictions("inverse").with_columns(
        pl.when(pl.col("ticker") == "LOSE")
        .then(1)
        .otherwise(2)
        .cast(pl.Int64)
        .alias("rank")
    )
    matured = mature_predictions(pl.concat([_predictions(), inverse]), labels)

    by_strategy = {
        metrics["strategy_version"]: metrics
        for metrics in evaluate_rankings(matured, top_k=1)
    }

    assert by_strategy["rank-v1"]["precision_at_k"] == 1.0
    assert by_strategy["rank-v1"]["base_explosion_rate"] == 0.5
    assert by_strategy["rank-v1"]["lift_at_k"] == 2.0
    assert by_strategy["inverse"]["precision_at_k"] == 0.0
    assert by_strategy["inverse"]["lift_at_k"] == 0.0


def test_low_outcome_coverage_is_skipped() -> None:
    labels = compute_forward_labels(_ohlcv(), spec=ExplosionLabelSpec(absolute_return=0.01))
    incomplete = _predictions().with_columns(pl.lit(3).alias("universe_size"))

    metrics = evaluate_rankings(mature_predictions(incomplete, labels), top_k=1)[0]

    assert metrics["days"] == 0
    assert metrics["skipped_days"] == 1


def test_prediction_ledgers_are_immutable_per_date_and_strategy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)
    predictions = _predictions()

    path = write_prediction_ledger(predictions)

    assert path == (
        tmp_path
        / "predictions"
        / "dt=2025-01-01"
        / "strategy=rank-v1"
        / "ranking.parquet"
    )
    assert pl.read_parquet(path).equals(predictions)
    assert read_prediction_ledger().equals(predictions)

    try:
        write_prediction_ledger(predictions)
    except FileExistsError:
        pass
    else:
        raise AssertionError("a date/strategy ledger must be immutable")

    second_path = write_prediction_ledger(_predictions("rank-v2"))
    assert second_path.exists()
    assert read_prediction_ledger().height == 4


def test_legacy_ledger_is_readable_but_has_no_claimed_universe_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)
    path = tmp_path / "predictions" / "dt=2025-01-01" / "ranking.parquet"
    path.parent.mkdir(parents=True)
    legacy = _predictions().select("ticker", "decision_date", "score")
    legacy.write_parquet(path)

    normalized = read_prediction_ledger()

    assert normalized["strategy_version"].unique().to_list() == [LEGACY_STRATEGY_VERSION]
    assert normalized["universe_size"].null_count() == normalized.height


def test_bootstrap_interval_is_deterministic_after_twenty_dates() -> None:
    values = np.linspace(0.0, 1.0, 20)

    assert _block_bootstrap_ci(values) == _block_bootstrap_ci(values)
