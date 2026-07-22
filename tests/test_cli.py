from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from click.testing import CliRunner

from alphascreener.cli import cli
from alphascreener.data.sync import SyncResult
from alphascreener.ranking import rank_candidates


def _backtest_records() -> pl.DataFrame:
    start = date(2025, 1, 1)
    return pl.DataFrame({
        "strategy_version": ["rank-v4"] * 3,
        "decision_date": [start + timedelta(days=index) for index in range(3)],
        "universe_size": [100] * 3,
        "outcome_coverage": [1.0] * 3,
        "precision_at_10": [0.1] * 3,
        "base_explosion_rate": [0.05] * 3,
        "passed": [True] * 3,
        "result_date": [start + timedelta(days=14 + index) for index in range(3)],
        "universe_source": ["current-directory"] * 3,
    })


def _stub_backtest(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphascreener.backtest.run_recent_backtest", lambda data: _backtest_records()
    )
    monkeypatch.setattr(
        "alphascreener.backtest.write_backtest_records", lambda records: None
    )


def _stub_complete_sync(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda: SyncResult(0, 100, 100, ()),
    )


def test_help_exposes_only_prediction_evaluation() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "backtest" not in result.output
    assert "optimize" not in result.output
    assert "sync" not in result.output


def test_rank_candidates_uses_the_60_session_window() -> None:
    rows = []
    for ticker, growth in [("SPY", 1.02), ("WIN", 1.01)]:
        for index in range(60):
            rows.append({
                "ticker": ticker,
                "dt": date(2025, 1, 1) + timedelta(days=index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })

    ranked, cutoff = rank_candidates(pl.DataFrame(rows))

    assert cutoff == date(2025, 3, 1)
    assert ranked.item(0, "ticker") == "WIN"


def test_rank_candidates_requires_a_current_spy_benchmark() -> None:
    rows = []
    for index in range(60):
        rows.append({
            "ticker": "WIN",
            "dt": date(2025, 1, 1) + timedelta(days=index),
            "close": 100.0 * 1.01**index,
            "volume": 2_000_000,
        })

    with pytest.raises(ValueError, match="SPY benchmark unavailable"):
        rank_candidates(pl.DataFrame(rows))


def test_top_option_limits_display_but_ledger_receives_full_ranking(monkeypatch) -> None:
    _stub_backtest(monkeypatch)
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda: SyncResult(
            0,
            3,
            3,
            (),
            ("SPY", "WIN", "SECOND"),
        ),
    )
    rows = []
    start = date.today() - timedelta(days=90)
    for ticker, growth in [
        ("SPY", 1.001),
        ("WIN", 1.01),
        ("SECOND", 1.005),
        ("STALE", 1.02),
    ]:
        for index in range(91):
            rows.append({
                "ticker": ticker,
                "dt": start + timedelta(days=index),
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })
    data = pl.DataFrame(rows)
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli, ["--top", "1"])

    assert result.exit_code == 0
    assert ledger_writes[0].height == 2
    assert ledger_writes[0]["universe_size"].unique().to_list() == [2]
    assert "STALE" not in ledger_writes[0]["ticker"].to_list()
    assert "Recent current-universe walk-forward backtest" in result.output


def test_incomplete_sync_does_not_record_a_ranking(monkeypatch) -> None:
    ledger_writes = []

    def no_local_data():
        raise FileNotFoundError

    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", no_local_data)
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda: SyncResult(10, 100, 50, ("FAILED",)),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert ledger_writes == []


def test_same_day_rerun_displays_the_immutable_recorded_ranking(monkeypatch) -> None:
    _stub_backtest(monkeypatch)
    _stub_complete_sync(monkeypatch)
    start = date.today() - timedelta(days=90)
    rows = []
    for ticker, growth in [("SPY", 1.001), ("NEW", 1.01)]:
        for index in range(91):
            rows.append({
                "ticker": ticker,
                "dt": start + timedelta(days=index),
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })
    data = pl.DataFrame(rows)
    recorded = pl.DataFrame({
        "ticker": ["SAVED"],
        "decision_date": [date.today()],
        "score": [42.0],
        "rank": [1],
        "strategy_version": ["rank-v4"],
        "universe_size": [1],
    })
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.read_prediction_ledger", lambda: recorded
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "SAVED" in result.output
    assert "NEW" not in result.output


def test_cli_does_not_record_a_ranking_without_spy(monkeypatch) -> None:
    _stub_backtest(monkeypatch)
    _stub_complete_sync(monkeypatch)
    start = date.today() - timedelta(days=90)
    data = pl.DataFrame([
        {
            "ticker": "WIN",
            "dt": start + timedelta(days=index),
            "close": 100.0 * 1.01**index,
            "volume": 2_000_000,
        }
        for index in range(91)
    ])
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "SPY benchmark unavailable" in result.output
    assert ledger_writes == []


def test_cli_does_not_record_a_ranking_without_three_backtest_dates(monkeypatch) -> None:
    _stub_complete_sync(monkeypatch)
    rows = []
    start = date.today() - timedelta(days=90)
    for ticker, growth in [("SPY", 1.001), ("WIN", 1.01)]:
        for index in range(91):
            rows.append({
                "ticker": ticker,
                "dt": start + timedelta(days=index),
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })
    data = pl.DataFrame(rows)
    ledger_writes = []

    def fail_backtest(data):
        raise ValueError("need three dates")

    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.backtest.run_recent_backtest",
        fail_backtest,
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "need three dates" in result.output
    assert ledger_writes == []
