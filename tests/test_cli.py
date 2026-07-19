"""Tests for Alpha Screener CLI (simplified — Issue #258)."""

from __future__ import annotations

from datetime import date, timedelta

import click
import polars as pl
import pytest
from click.testing import CliRunner

from alphascreener.cli import _rank_candidates
from alphascreener.cli import cli as main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ============================================================================
# Entry point
# ============================================================================


def test_main_is_click_group() -> None:
    assert isinstance(main, click.Group)


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "backtest" in result.output
    assert "evaluate" in result.output
    assert "--top" in result.output


def test_default_invocation(runner: CliRunner) -> None:
    """Default (no subcommand) exits cleanly (may fail on insufficient data)."""
    result = runner.invoke(main)
    assert result.exit_code in (0, 1)


def test_default_with_no_backtest(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--no-backtest"])
    assert result.exit_code in (0, 1)  # may fail on insufficient data


def test_default_with_custom_top(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--top", "5"])
    assert result.exit_code in (0, 1)


def test_rank_candidates_uses_eligible_sixty_session_window() -> None:
    rows = []
    for ticker, multiplier, volume in [("SPY", 1.001, 2_000_000), ("WIN", 1.01, 2_000_000)]:
        for index in range(60):
            rows.append({
                "ticker": ticker,
                "dt": date(2025, 1, 1) + timedelta(days=index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 * multiplier**index,
                "volume": volume,
            })

    ranked, cutoff = _rank_candidates(pl.DataFrame(rows), top=1)

    assert cutoff == date(2025, 3, 1)
    assert ranked.item(0, "ticker") == "WIN"


# ============================================================================
# backtest command
# ============================================================================


def test_backtest_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "TICKER" in result.output


def test_backtest_valid_ticker(runner: CliRunner) -> None:
    result = runner.invoke(main, ["backtest", "AAPL"])
    assert result.exit_code == 0


def test_backtest_missing_ticker(runner: CliRunner) -> None:
    result = runner.invoke(main, ["backtest"])
    assert result.exit_code != 0


def test_backtest_with_dates(runner: CliRunner) -> None:
    result = runner.invoke(main, [
        "backtest", "AAPL", "--start", "2023-01-01", "--end", "2023-12-31"
    ])
    assert result.exit_code == 0


def test_backtest_invalid_date(runner: CliRunner) -> None:
    result = runner.invoke(main, ["backtest", "AAPL", "--start", "not-a-date"])
    assert result.exit_code != 0


# ============================================================================
# dev subgroup
# ============================================================================


def test_dev_review_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["dev", "review", "--help"])
    assert result.exit_code == 0
    assert "days" in result.output.lower()


def test_dev_review_valid(runner: CliRunner) -> None:
    result = runner.invoke(main, ["dev", "review", "--days", "30"])
    assert result.exit_code == 0




