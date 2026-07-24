"""Tests for on-demand current-universe walk-forward diagnostics."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

import alphascreener.backtest as backtest_module
from alphascreener.backtest import run_backtest
from alphascreener.market_calendar import infer_market_dates, market_dates_between
from alphascreener.ranking import rank_candidate_dates


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2024, 1, 2), date(2025, 12, 31))
    assert len(dates) >= count
    return dates[:count]


def _market_data(sessions: int = 125) -> pl.DataFrame:
    rows = []
    market_dates = _market_dates(sessions)
    tickers = [("SPY", 1.001), ("WIN", 1.02)] + [(f"T{index}", 1.001) for index in range(11)]
    for ticker, growth in tickers:
        for index, market_date in enumerate(market_dates):
            rows.append(
                {
                    "ticker": ticker,
                    "dt": market_date,
                    "close": 100.0 * growth**index,
                    "volume": 2_000_000,
                }
            )
    return pl.DataFrame(rows)


def test_default_backtest_uses_30_matured_dates() -> None:
    data = _market_data()

    records = run_backtest(data)

    market_dates = infer_market_dates(data)
    assert records.height == 30
    assert records["decision_date"].to_list() == market_dates[-44:-14]
    assert records["result_date"].to_list() == market_dates[-30:]
    assert records["status"].unique().to_list() == ["VALID"]
    assert records["universe_source"].unique().to_list() == ["current-directory"]


def test_backtest_accepts_one_and_45_days() -> None:
    data = _market_data()

    assert run_backtest(data, days=1).height == 1
    assert run_backtest(data, days=45).height == 45


@pytest.mark.parametrize(("days", "expected_sessions"), [(1, 74), (45, 118)])
def test_backtest_only_builds_the_required_history_and_outcome_window(
    monkeypatch,
    days,
    expected_sessions,
) -> None:
    data = _market_data()
    rank_sessions: list[int] = []
    label_sessions: list[int] = []
    original_labels = backtest_module.compute_forward_labels

    def capture_window(history, decision_dates):
        rank_sessions.append(history["dt"].n_unique())
        return rank_candidate_dates(history, decision_dates)

    def capture_labels(history):
        label_sessions.append(history["dt"].n_unique())
        return original_labels(history)

    monkeypatch.setattr(
        "alphascreener.backtest.rank_candidate_dates",
        capture_window,
    )
    monkeypatch.setattr(
        "alphascreener.backtest.compute_forward_labels",
        capture_labels,
    )

    run_backtest(data, days=days)

    assert rank_sessions == [expected_sessions]
    assert label_sessions == [expected_sessions]


def test_45_day_history_boundary_is_exact() -> None:
    exact = run_backtest(_market_data(sessions=118), days=45)
    one_short = run_backtest(_market_data(sessions=117), days=45)

    assert exact.filter(pl.col("status") == "VALID").height == 45
    assert one_short.filter(pl.col("status") == "INVALID").height == 1
    assert one_short.row(0, named=True)["invalid_reason"] == "insufficient_history"
    assert one_short.filter(pl.col("status") == "VALID").height == 44


@pytest.mark.parametrize("days", [0, 46])
def test_backtest_rejects_days_outside_public_range(days: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 45"):
        run_backtest(_market_data(), days=days)


def test_one_invalid_date_does_not_abort_later_dates(monkeypatch) -> None:
    data = _market_data()
    market_dates = infer_market_dates(data)
    broken_date = market_dates[-16]

    def sometimes_incomplete(history, decision_dates):
        rankings = rank_candidate_dates(history, decision_dates)
        return pl.concat(
            [
                rankings.filter(pl.col("decision_date") != broken_date),
                rankings.filter(pl.col("decision_date") == broken_date).head(9),
            ]
        )

    monkeypatch.setattr(
        "alphascreener.backtest.rank_candidate_dates",
        sometimes_incomplete,
    )

    records = run_backtest(data, days=3)

    assert records.height == 3
    broken = records.filter(pl.col("decision_date") == broken_date)
    assert broken.item(0, "status") == "INVALID"
    assert broken.item(0, "invalid_reason") == "eligible_universe_below_top_10"
    assert records.filter(pl.col("status") == "VALID").height == 2


def test_low_outcome_coverage_is_an_invalid_row() -> None:
    data = _market_data()
    market_dates = infer_market_dates(data)
    result_date = market_dates[-1]
    data = data.filter(~(pl.col("ticker").is_in(["T0", "T1"]) & (pl.col("dt") == result_date)))

    row = run_backtest(data, days=1).row(0, named=True)

    assert row["status"] == "INVALID"
    assert row["invalid_reason"] == "outcome_coverage_below_90pct"
    assert row["passed"] is None


def test_missing_top_10_outcome_is_not_replaced() -> None:
    data = _market_data()
    market_dates = infer_market_dates(data)
    result_date = market_dates[-1]
    data = data.filter(~((pl.col("ticker") == "WIN") & (pl.col("dt") == result_date)))

    row = run_backtest(data, days=1).row(0, named=True)

    assert row["status"] == "INVALID"
    assert row["invalid_reason"] == "top_10_outcomes_incomplete"
    assert row["passed"] is None


def test_missing_spy_decision_date_is_reported_without_shifting_the_calendar() -> None:
    data = _market_data()
    market_dates = infer_market_dates(data)
    decision_date = market_dates[-15]
    data = data.filter(~((pl.col("ticker") == "SPY") & (pl.col("dt") == decision_date)))

    row = run_backtest(data, days=1).row(0, named=True)

    assert row["decision_date"] == decision_date
    assert row["result_date"] == market_dates[-1]
    assert row["status"] == "INVALID"
    assert row["invalid_reason"] == "spy_missing_on_decision_date"


def test_backtest_applies_the_same_point_in_time_transaction_filter() -> None:
    data = _market_data()

    def statuses(tickers, decision_date):
        return {ticker: SimpleNamespace(exclude_from_ranking=ticker == "WIN") for ticker in tickers}

    row = run_backtest(
        data,
        days=1,
        corporate_action_status_provider=statuses,
    ).row(0, named=True)

    assert row["status"] == "VALID"
    assert row["universe_size"] == 11


def test_one_sec_failure_invalidates_only_that_backtest_date() -> None:
    data = _market_data()
    market_dates = infer_market_dates(data)
    broken_date = market_dates[-16]

    def statuses(tickers, decision_date):
        if decision_date == broken_date:
            raise OSError("SEC unavailable")
        return {ticker: SimpleNamespace(exclude_from_ranking=False) for ticker in tickers}

    records = run_backtest(
        data,
        days=3,
        corporate_action_status_provider=statuses,
    )

    broken = records.filter(pl.col("decision_date") == broken_date)
    assert broken.item(0, "status") == "INVALID"
    assert broken.item(0, "invalid_reason") == "corporate_action_status_unavailable"
    assert records.filter(pl.col("status") == "VALID").height == 2


def test_45_day_sec_provider_receives_one_leader_batch_per_date() -> None:
    data = _market_data(sessions=118)
    calls: list[tuple[date, tuple[str, ...]]] = []

    def statuses(tickers, decision_date):
        calls.append((decision_date, tuple(tickers)))
        return {ticker: SimpleNamespace(exclude_from_ranking=False) for ticker in tickers}

    records = run_backtest(
        data,
        days=45,
        corporate_action_status_provider=statuses,
    )

    assert records.height == 45
    assert len(calls) == 45
    assert [decision_date for decision_date, _ in calls] == sorted(
        records["decision_date"].to_list(),
        reverse=True,
    )
    assert all(len(tickers) == 12 for _, tickers in calls)


def test_45_day_sec_provider_prefetches_the_leader_union_once() -> None:
    data = _market_data(sessions=118)

    class Provider:
        def __init__(self):
            self.prefetch_calls: list[tuple[date, tuple[str, ...]]] = []
            self.daily_calls = 0

        def prefetch(self, tickers, decision_date):
            self.prefetch_calls.append((decision_date, tuple(tickers)))

        def __call__(self, tickers, decision_date):
            self.daily_calls += 1
            return {ticker: SimpleNamespace(exclude_from_ranking=False) for ticker in tickers}

    provider = Provider()
    records = run_backtest(
        data,
        days=45,
        corporate_action_status_provider=provider,
    )

    assert records.height == 45
    assert provider.daily_calls == 45
    assert provider.prefetch_calls == [
        (
            records["decision_date"].max(),
            tuple(
                sorted(ticker for ticker in data["ticker"].unique().to_list() if ticker != "SPY")
            ),
        )
    ]
