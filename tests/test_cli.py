from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from click.testing import CliRunner

from alphascreener.cli import _rank_candidates, cli
from alphascreener.data.sync import SyncResult


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

    ranked, cutoff = _rank_candidates(pl.DataFrame(rows))

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
        _rank_candidates(pl.DataFrame(rows))


def test_top_option_limits_display_but_ledger_receives_full_ranking(monkeypatch) -> None:
    rows = []
    start = date.today() - timedelta(days=90)
    for ticker, growth in [("SPY", 1.001), ("WIN", 1.01), ("SECOND", 1.005)]:
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
    monkeypatch.setattr("alphascreener.data.sync.last_sync_date", lambda: date.today())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli, ["--top", "1"])

    assert result.exit_code == 0
    assert ledger_writes[0].height == 2
    assert ledger_writes[0]["universe_size"].unique().to_list() == [2]


def test_incomplete_sync_does_not_record_a_ranking(monkeypatch) -> None:
    ledger_writes = []

    def no_local_data():
        raise FileNotFoundError

    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", no_local_data)
    monkeypatch.setattr("alphascreener.data.sync.last_sync_date", lambda: None)
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


def test_cli_does_not_record_a_ranking_without_spy(monkeypatch) -> None:
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
    monkeypatch.setattr("alphascreener.data.sync.last_sync_date", lambda: date.today())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "SPY benchmark unavailable" in result.output
    assert ledger_writes == []
