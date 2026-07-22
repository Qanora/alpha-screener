"""Tests for the automatic current-universe walk-forward backtest."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from alphascreener.backtest import run_recent_backtest, write_backtest_records
from alphascreener.ranking import rank_candidates


def _market_data(sessions: int = 80) -> pl.DataFrame:
    rows = []
    tickers = [("SPY", 1.001), ("WIN", 1.02)] + [
        (f"T{index}", 1.001) for index in range(11)
    ]
    for ticker, growth in tickers:
        for index in range(sessions):
            rows.append({
                "ticker": ticker,
                "dt": date(2025, 1, 1) + timedelta(days=index),
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })
    return pl.DataFrame(rows)


def test_recent_backtest_uses_three_consecutive_matured_dates(monkeypatch) -> None:
    data = _market_data()
    seen_cutoffs = []
    seen_session_counts = []

    def observe_history(history):
        seen_cutoffs.append(history["dt"].max())
        seen_session_counts.append(
            history.filter(pl.col("ticker") == "SPY")["dt"].n_unique()
        )
        return rank_candidates(history)

    monkeypatch.setattr("alphascreener.backtest.rank_candidates", observe_history)

    records = run_recent_backtest(data)

    market_dates = data.filter(pl.col("ticker") == "SPY")["dt"].unique().sort().to_list()
    assert records["decision_date"].to_list() == market_dates[-17:-14]
    assert records["result_date"].to_list() == market_dates[-3:]
    assert seen_cutoffs == market_dates[-17:-14]
    assert seen_session_counts == [60, 60, 60]
    assert records["universe_source"].unique().to_list() == ["current-directory"]


def test_recent_backtest_requires_three_matured_dates() -> None:
    with pytest.raises(ValueError, match="at least 3 matured backtest dates"):
        run_recent_backtest(_market_data(75))


def test_backtest_records_are_committed_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.backtest.get_data_home", lambda: tmp_path)
    records = run_recent_backtest(_market_data())

    output = write_backtest_records(records)

    assert output == tmp_path / "backtests" / "strategy=rank-v4" / "recent.parquet"
    assert pl.read_parquet(output).equals(records)
    assert not output.with_suffix(".parquet.tmp").exists()
