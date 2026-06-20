"""Tests for Alpha Screener CLI (simplified — Issue #258)."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

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
    assert "--top" in result.output
    assert "--no-backtest" in result.output


def test_default_invocation(runner: CliRunner) -> None:
    """Default (no subcommand) should not crash."""
    result = runner.invoke(main)
    assert result.exit_code == 0


def test_default_with_no_backtest(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--no-backtest"])
    assert result.exit_code == 0


def test_default_with_custom_top(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--top", "5"])
    assert result.exit_code == 0


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


def test_dev_validate_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["dev", "validate", "--help"])
    assert result.exit_code == 0


def test_dev_validate_valid(runner: CliRunner) -> None:
    result = runner.invoke(main, ["dev", "validate", "--version", "v2.0.0"])
    assert result.exit_code == 0


def test_dev_validate_missing_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["dev", "validate"])
    assert result.exit_code != 0
