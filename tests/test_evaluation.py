"""Tests for future-14-session label and ranking evaluation workflows."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from alphascreener.evaluation import (
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
            rows.append({"ticker": ticker, "dt": date(2025, 1, 1) + timedelta(days=index),
                         "close": 100.0 * multiplier**index})
    return pl.DataFrame(rows)


def test_forward_labels_use_exactly_14_later_sessions() -> None:
    labels = compute_forward_labels(_ohlcv(), spec=ExplosionLabelSpec(absolute_return=0.01))

    assert labels.filter(pl.col("ticker") == "WIN").height == 15
    first = labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == date(2025, 1, 1)))
    assert first.item(0, "future_close") == 100.0 * 1.02**14


def test_matured_ranking_and_metrics_are_decision_date_scoped() -> None:
    labels = compute_forward_labels(_ohlcv(), spec=ExplosionLabelSpec(absolute_return=0.01))
    predictions = pl.DataFrame({"ticker": ["WIN", "LOSE"], "decision_date": [date(2025, 1, 1)] * 2,
                                "score": [0.9, 0.1]})

    matured = mature_predictions(predictions, labels)
    metrics = evaluate_rankings(matured, top_k=1)

    assert matured.height == 2
    assert metrics["days"] == 1
    assert metrics["precision_at_k"] == 1.0
    assert metrics["lift_at_k"] == 2.0


def test_prediction_ledger_is_written_before_outcome(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)
    predictions = pl.DataFrame(
        {"ticker": ["WIN"], "decision_date": [date(2025, 1, 1)], "score": [0.9]}
    )

    path = write_prediction_ledger(predictions)

    assert path.exists()
    assert pl.read_parquet(path).equals(predictions)
    assert read_prediction_ledger().equals(predictions)

    try:
        write_prediction_ledger(predictions)
    except FileExistsError:
        pass
    else:
        raise AssertionError("a decision-date ledger must be immutable")
