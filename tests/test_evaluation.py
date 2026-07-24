"""Tests for future labels, versioned ledgers, and strategy-aware evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alphascreener.evaluation import (
    compute_forward_labels,
    evaluate_daily_rankings,
    expected_shortfall,
    longest_consecutive_passes,
    mature_predictions,
    read_prediction_ledger,
    write_prediction_ledger,
)
from alphascreener.market_calendar import infer_market_dates, market_dates_between
from alphascreener.prediction_contract import ExplosionLabelSpec


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2025, 1, 2), date(2025, 12, 31))
    assert len(dates) >= count
    return dates[:count]


def _ohlcv() -> pl.DataFrame:
    rows = []
    market_dates = _market_dates(29)
    for ticker, multiplier in [("SPY", 1.001), ("WIN", 1.02), ("LOSE", 1.001)]:
        for index, market_date in enumerate(market_dates):
            rows.append(
                {
                    "ticker": ticker,
                    "dt": market_date,
                    "close": 100.0 * multiplier**index,
                }
            )
    return pl.DataFrame(rows)


def _predictions(strategy: str = "rank-v1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["WIN", "LOSE"],
            "decision_date": [date(2025, 1, 1)] * 2,
            "score": [0.9, 0.1],
            "rank": [1, 2],
            "strategy_version": [strategy] * 2,
            "universe_size": [2, 2],
        }
    )


def test_forward_labels_use_exactly_14_later_sessions() -> None:
    data = _ohlcv()
    market_dates = infer_market_dates(data)
    labels = compute_forward_labels(data, spec=ExplosionLabelSpec(absolute_return=0.01))

    assert labels.filter(pl.col("ticker") == "WIN").height == 15
    first = labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == market_dates[0]))
    assert first.item(0, "future_close") == 100.0 * 1.02**14
    assert first.item(0, "future_min_close") == 100.0 * 1.02
    assert first.item(0, "max_adverse_return") == pytest.approx(0.02)
    assert first.item(0, "result_date") == market_dates[14]


def test_forward_labels_follow_the_spy_calendar_when_a_ticker_misses_a_session() -> None:
    market_dates = _market_dates(16)
    rows = [{"ticker": "SPY", "dt": market_date, "close": 100.0} for market_date in market_dates]
    rows.extend(
        {
            "ticker": "WIN",
            "dt": market_date,
            "close": 100.0 + index,
        }
        for index, market_date in enumerate(market_dates)
        if index != 5
    )

    labels = compute_forward_labels(pl.DataFrame(rows))

    first = labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == market_dates[0]))
    assert first.height == 1
    assert first.item(0, "result_date") == market_dates[14]
    assert first.item(0, "future_close") == 114.0
    assert first.item(0, "future_min_close") is None
    assert first.item(0, "max_adverse_return") is None


def test_forward_labels_do_not_substitute_a_later_ticker_session() -> None:
    market_dates = _market_dates(16)
    rows = [{"ticker": "SPY", "dt": market_date, "close": 100.0} for market_date in market_dates]
    rows.extend(
        {
            "ticker": "WIN",
            "dt": market_date,
            "close": 100.0 + index,
        }
        for index, market_date in enumerate(market_dates)
        if index != 14
    )

    labels = compute_forward_labels(pl.DataFrame(rows))

    assert labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == market_dates[0])).is_empty()


def test_forward_labels_do_not_shift_when_spy_misses_a_market_session() -> None:
    market_dates = _market_dates(16)
    rows = []
    for ticker in ("SPY", "WIN", "AUX"):
        rows.extend(
            {
                "ticker": ticker,
                "dt": market_date,
                "close": 100.0 + index,
            }
            for index, market_date in enumerate(market_dates)
            if ticker != "SPY" or index != 5
        )

    labels = compute_forward_labels(pl.DataFrame(rows))

    first = labels.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == market_dates[0]))
    assert first.item(0, "result_date") == market_dates[14]
    assert first.item(0, "future_close") == 114.0


def test_forward_labels_require_a_market_calendar() -> None:
    with pytest.raises(ValueError, match="SPY market calendar unavailable"):
        compute_forward_labels(_ohlcv().filter(pl.col("ticker") != "SPY"))


def _matured_rows(*, missing_ranks: set[int] = set()) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [f"T{rank}" for rank in range(1, 21) if rank not in missing_ranks],
            "decision_date": [date(2025, 1, 1)] * (20 - len(missing_ranks)),
            "score": [float(21 - rank) for rank in range(1, 21) if rank not in missing_ranks],
            "rank": [rank for rank in range(1, 21) if rank not in missing_ranks],
            "strategy_version": ["rank-v1"] * (20 - len(missing_ranks)),
            "universe_size": [20] * (20 - len(missing_ranks)),
            "is_explosion": [rank in {1, 2} for rank in range(1, 21) if rank not in missing_ranks],
            "is_severe_downside": [
                rank in {3, 4} for rank in range(1, 21) if rank not in missing_ranks
            ],
            "is_catastrophic_loss": [
                rank == 4 for rank in range(1, 21) if rank not in missing_ranks
            ],
            "has_adverse_path": [
                rank in {3, 4} for rank in range(1, 21) if rank not in missing_ranks
            ],
            "forward_return": [
                0.20 if rank in {1, 2} else (-0.25 if rank == 4 else (-0.12 if rank == 3 else 0.0))
                for rank in range(1, 21)
                if rank not in missing_ranks
            ],
        }
    )


def test_daily_metrics_use_the_complete_universe() -> None:
    metrics = evaluate_daily_rankings(_matured_rows()).row(0, named=True)

    assert metrics["precision_at_k"] == 0.2
    assert metrics["base_explosion_rate"] == 0.1
    assert metrics["downside_at_k"] == 0.2
    assert metrics["catastrophic_loss_at_k"] == 0.1
    assert metrics["adverse_path_at_k"] == 0.2
    assert metrics["basket_return_14"] == pytest.approx(0.003)
    assert metrics["outcome_coverage"] == 1.0
    assert metrics["passed"] is True


def test_low_outcome_coverage_is_skipped() -> None:
    assert evaluate_daily_rankings(_matured_rows(missing_ranks={18, 19, 20})).is_empty()


def test_missing_top_rank_is_not_replaced_by_a_lower_rank() -> None:
    assert evaluate_daily_rankings(_matured_rows(missing_ranks={1})).is_empty()


def test_missing_non_top_outcome_cannot_redefine_the_full_pool_threshold() -> None:
    predictions = pl.DataFrame(
        {
            "ticker": [f"T{rank}" for rank in range(1, 21)],
            "decision_date": [date(2025, 1, 2)] * 20,
            "score": [float(21 - rank) for rank in range(1, 21)],
            "rank": list(range(1, 21)),
            "strategy_version": ["rank-v1"] * 20,
            "universe_size": [20] * 20,
        }
    )
    labels = pl.DataFrame(
        {
            "ticker": [f"T{rank}" for rank in range(1, 20)],
            "dt": [date(2025, 1, 2)] * 19,
            "result_date": [date(2025, 1, 23)] * 19,
            "forward_return": [0.20 if rank == 1 else 0.0 for rank in range(1, 20)],
            "max_adverse_return": [0.0] * 19,
        }
    )

    matured = mature_predictions(predictions, labels)

    assert matured.height == 20
    assert matured["hit_threshold"].null_count() == 20
    assert matured["is_explosion"].null_count() == 20
    assert matured["is_severe_downside"].null_count() == 20
    assert evaluate_daily_rankings(matured).is_empty()


def test_risk_labels_include_the_exact_frozen_thresholds() -> None:
    predictions = pl.DataFrame(
        {
            "ticker": ["SEVERE", "ABOVE_SEVERE", "CATASTROPHIC", "ABOVE_PATH"],
            "decision_date": [date(2025, 1, 2)] * 4,
            "score": [4.0, 3.0, 2.0, 1.0],
            "rank": [1, 2, 3, 4],
            "strategy_version": ["rank-v1"] * 4,
            "universe_size": [4] * 4,
        }
    )
    labels = pl.DataFrame(
        {
            "ticker": predictions["ticker"],
            "dt": [date(2025, 1, 2)] * 4,
            "result_date": [date(2025, 1, 23)] * 4,
            "forward_return": [-0.10, -0.099999, -0.20, 0.0],
            "max_adverse_return": [-0.15, 0.0, -0.149999, -0.149999],
        }
    )

    matured = mature_predictions(predictions, labels).sort("rank")

    assert matured["is_severe_downside"].to_list() == [
        True,
        False,
        True,
        False,
    ]
    assert matured["is_catastrophic_loss"].to_list() == [
        False,
        False,
        True,
        False,
    ]
    assert matured["has_adverse_path"].to_list() == [
        True,
        False,
        False,
        False,
    ]


def test_expected_shortfall_uses_the_lower_tail_and_at_least_one_row() -> None:
    assert expected_shortfall([-0.30, -0.20, -0.10, 0.10, 0.20], quantile=0.4) == -0.25
    assert expected_shortfall([0.05], quantile=0.1) == 0.05


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (9, -9.0),
        (10, -10.0),
        (11, -10.5),
    ],
)
def test_expected_shortfall_uses_ceil_for_the_ten_percent_tail(
    count,
    expected,
) -> None:
    returns = [-float(value) for value in range(1, count + 1)]

    assert expected_shortfall(returns, quantile=0.1) == expected


def test_prediction_ledgers_are_immutable_per_date_and_strategy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.evaluation.get_data_home", lambda: tmp_path)
    predictions = _predictions()

    path = write_prediction_ledger(predictions)

    assert path == (
        tmp_path / "predictions" / "dt=2025-01-01" / "strategy=rank-v1" / "ranking.parquet"
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

    output = tmp_path / "predictions" / "dt=2025-01-01" / "strategy=rank-v1" / "ranking.parquet"
    assert not output.exists()
    assert not output.with_suffix(".parquet.tmp").exists()
    assert write_prediction_ledger(_predictions()) == output


def test_concurrent_ledger_writes_keep_exactly_one_immutable_result(tmp_path, monkeypatch) -> None:
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
    market_dates = _market_dates(7)
    daily = pl.DataFrame(
        {
            "strategy_version": ["rank-v2"] * 6,
            "decision_date": [*market_dates[:3], *market_dates[4:7]],
            "universe_size": [20] * 6,
            "outcome_coverage": [1.0] * 6,
            "precision_at_k": [0.1] * 6,
            "base_explosion_rate": [0.05] * 6,
            "downside_at_k": [0.0] * 6,
            "catastrophic_loss_at_k": [0.0] * 6,
            "adverse_path_at_k": [0.0] * 6,
            "basket_return_14": [0.1] * 6,
            "passed": [True] * 6,
        }
    )

    assert longest_consecutive_passes(daily, market_dates, strategy_version="rank-v2") == 3
