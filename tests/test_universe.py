"""Tests for point-in-time tradable universe construction."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from alphascreener.universe import UniverseRules, build_universe_snapshot


def _ohlcv(ticker: str, sessions: int, *, close: float = 20.0, volume: int = 500_000) -> list[dict]:
    start = date(2025, 1, 1)
    return [
        {"ticker": ticker, "dt": start + timedelta(days=index), "close": close, "volume": volume}
        for index in range(sessions)
    ]


def test_snapshot_records_eligible_ticker_and_each_exclusion_reason() -> None:
    rows = _ohlcv("GOOD", 60)
    rows += _ohlcv("SHORT", 59)
    rows += _ohlcv("CHEAP", 60, close=4.0)
    rows += _ohlcv("ILLIQUID", 60, volume=1_000)
    stale = [
        {**row, "dt": row["dt"] - timedelta(days=1)}
        for row in _ohlcv("STALE", 60)
    ]
    rows += stale

    snapshot = build_universe_snapshot(pl.DataFrame(rows), cutoff_date=date(2025, 3, 1))
    by_ticker = {row["ticker"]: row for row in snapshot.to_dicts()}

    assert by_ticker["GOOD"]["eligible"] is True
    assert by_ticker["SHORT"]["exclusion_reason"] == "insufficient_history"
    assert by_ticker["CHEAP"]["exclusion_reason"] == "low_price"
    assert by_ticker["ILLIQUID"]["exclusion_reason"] == "low_dollar_volume"
    assert by_ticker["STALE"]["exclusion_reason"] == "stale_data"


def test_snapshot_uses_only_data_at_or_before_cutoff() -> None:
    rows = _ohlcv("GOOD", 61)
    snapshot = build_universe_snapshot(pl.DataFrame(rows), cutoff_date=date(2025, 3, 1))

    assert snapshot.item(0, "history_sessions") == 60
    assert snapshot.item(0, "cutoff_date") == date(2025, 3, 1)


def test_snapshot_rejects_missing_ohlcv_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_universe_snapshot(pl.DataFrame({"ticker": ["AAPL"]}))


def test_snapshot_excludes_non_finite_observations() -> None:
    rows = _ohlcv("BAD", 60)
    rows[-1]["close"] = float("nan")

    snapshot = build_universe_snapshot(pl.DataFrame(rows), cutoff_date=date(2025, 3, 1))

    assert snapshot.item(0, "eligible") is False
    assert snapshot.item(0, "exclusion_reason") == "invalid_data"


def test_rules_can_be_made_stricter_without_changing_snapshot_contract() -> None:
    snapshot = build_universe_snapshot(
        pl.DataFrame(_ohlcv("AAPL", 60)),
        cutoff_date=date(2025, 3, 1),
        rules=UniverseRules(min_average_dollar_volume=20_000_000.0),
    )

    assert snapshot.item(0, "eligible") is False
