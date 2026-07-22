"""Tests for future labels, versioned ledgers, and strategy-aware evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from alphascreener.evaluation import (
    compute_forward_labels,
    evaluate_daily_rankings,
    longest_consecutive_passes,
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


def _matured_rows(*, missing_ranks: set[int] = set()) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [f"T{rank}" for rank in range(1, 21) if rank not in missing_ranks],
        "decision_date": [date(2025, 1, 1)] * (20 - len(missing_ranks)),
        "score": [float(21 - rank) for rank in range(1, 21) if rank not in missing_ranks],
        "rank": [rank for rank in range(1, 21) if rank not in missing_ranks],
        "strategy_version": ["rank-v1"] * (20 - len(missing_ranks)),
        "universe_size": [20] * (20 - len(missing_ranks)),
        "is_explosion": [rank in {1, 2} for rank in range(1, 21) if rank not in missing_ranks],
        "forward_return": [
            0.20 if rank in {1, 2} else 0.0
            for rank in range(1, 21)
            if rank not in missing_ranks
        ],
    })


def test_daily_metrics_use_the_complete_universe() -> None:
    metrics = evaluate_daily_rankings(_matured_rows()).row(0, named=True)

    assert metrics["precision_at_k"] == 0.2
    assert metrics["base_explosion_rate"] == 0.1
    assert metrics["outcome_coverage"] == 1.0
    assert metrics["passed"] is True


def test_low_outcome_coverage_is_skipped() -> None:
    assert evaluate_daily_rankings(_matured_rows(missing_ranks={18, 19, 20})).is_empty()


def test_missing_top_rank_is_not_replaced_by_a_lower_rank() -> None:
    assert evaluate_daily_rankings(_matured_rows(missing_ranks={1})).is_empty()


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


def test_failed_ledger_replace_does_not_create_an_immutable_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)

    def fail_replace(self, target):
        raise OSError("replace failed")

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            write_prediction_ledger(_predictions())

    output = (
        tmp_path
        / "predictions"
        / "dt=2025-01-01"
        / "strategy=rank-v1"
        / "ranking.parquet"
    )
    assert not output.exists()
    assert not output.with_suffix(".parquet.tmp").exists()
    assert write_prediction_ledger(_predictions()) == output


def test_concurrent_ledger_writes_keep_exactly_one_immutable_result(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)

    def attempt_write():
        try:
            return write_prediction_ledger(_predictions())
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt_write(), range(2)))

    assert sum(result is not None for result in results) == 1
    assert read_prediction_ledger().equals(_predictions())


def test_legacy_ledger_outside_current_contract_is_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)
    path = tmp_path / "predictions" / "dt=2025-01-01" / "ranking.parquet"
    path.parent.mkdir(parents=True)
    legacy = _predictions().select("ticker", "decision_date", "score")
    legacy.write_parquet(path)

    assert read_prediction_ledger().is_empty()


def test_longest_consecutive_passes_requires_adjacent_market_dates() -> None:
    market_dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(7)]
    daily = pl.DataFrame({
        "strategy_version": ["rank-v2"] * 6,
        "decision_date": [*market_dates[:3], *market_dates[4:7]],
        "universe_size": [20] * 6,
        "outcome_coverage": [1.0] * 6,
        "precision_at_k": [0.1] * 6,
        "base_explosion_rate": [0.05] * 6,
        "passed": [True] * 6,
    })

    assert longest_consecutive_passes(
        daily, market_dates, strategy_version="rank-v2"
    ) == 3
