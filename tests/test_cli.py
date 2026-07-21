from __future__ import annotations

from datetime import date, timedelta

import polars as pl
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

    ranked, cutoff = _rank_candidates(pl.DataFrame(rows), top=1)

    assert cutoff == date(2025, 3, 1)
    assert ranked.item(0, "ticker") == "WIN"


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
