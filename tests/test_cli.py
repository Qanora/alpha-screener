"""Tests for Click-based CLI (Issue #106).

Covers:
  - CLI group and entry point
  - screen command with --market and --top options
  - backtest command with --start and --end options
  - evolve command group with review-last and propose-factor subcommands
  - walk-forward command with --version option
  - Parameter validation and help output
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from alphascreener.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click CLI runner for testing."""
    return CliRunner()


# ============================================================================
# Entry point / group
# ============================================================================


def test_main_is_click_group() -> None:
    """The main entry point should be a click.Group."""
    assert isinstance(main, click.Group)


def test_cli_help(runner: CliRunner) -> None:
    """CLI should display help text."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "screen" in result.output
    assert "backtest" in result.output
    assert "evolve" in result.output
    assert "walk-forward" in result.output


# ============================================================================
# screen command
# ============================================================================


def test_screen_help(runner: CliRunner) -> None:
    """screen command should display its own help text."""
    result = runner.invoke(main, ["screen", "--help"])
    assert result.exit_code == 0
    assert "--market" in result.output
    assert "--top" in result.output


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--market", "US"],
        ["--market", "US", "--top", "20"],
        ["--market", "US", "--top", "10"],
    ],
)
def test_screen_valid_args(runner: CliRunner, args: list[str]) -> None:
    """screen command should accept valid arguments."""
    result = runner.invoke(main, ["screen", *args])
    # Should succeed even if no data is available (graceful handling)
    assert result.exit_code == 0


def test_screen_rejects_invalid_market(runner: CliRunner) -> None:
    """screen command should reject invalid --market values."""
    result = runner.invoke(main, ["screen", "--market", "INVALID"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "Error" in result.output


def test_screen_rejects_invalid_top(runner: CliRunner) -> None:
    """screen command should reject out-of-range --top values."""
    result = runner.invoke(main, ["screen", "--top", "0"])
    assert result.exit_code != 0


def test_screen_rejects_negative_top(runner: CliRunner) -> None:
    """screen command should reject negative --top values."""
    result = runner.invoke(main, ["screen", "--top", "-5"])
    assert result.exit_code != 0


# ============================================================================
# backtest command
# ============================================================================


def test_backtest_help(runner: CliRunner) -> None:
    """backtest command should display its own help text."""
    result = runner.invoke(main, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.output
    assert "--end" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--start", "2023-01-01"],
        ["--start", "2023-01-01", "--end", "2023-12-31"],
    ],
)
def test_backtest_valid_args(runner: CliRunner, args: list[str]) -> None:
    """backtest command should accept valid arguments."""
    result = runner.invoke(main, ["backtest", *args])
    assert result.exit_code == 0


def test_backtest_missing_start(runner: CliRunner) -> None:
    """backtest command should require --start."""
    result = runner.invoke(main, ["backtest"])
    assert result.exit_code != 0


def test_backtest_start_after_end(runner: CliRunner) -> None:
    """backtest command should reject --start after --end."""
    result = runner.invoke(main, ["backtest", "--start", "2023-12-31", "--end", "2023-01-01"])
    assert result.exit_code != 0


def test_backtest_invalid_date_format(runner: CliRunner) -> None:
    """backtest command should reject malformed dates."""
    result = runner.invoke(main, ["backtest", "--start", "not-a-date"])
    assert result.exit_code != 0


# ============================================================================
# evolve command group
# ============================================================================


def test_evolve_is_click_group() -> None:
    """evolve should be a click.Group with subcommands."""
    from alphascreener.cli import evolve

    assert isinstance(evolve, click.Group)


def test_evolve_help(runner: CliRunner) -> None:
    """evolve command should display help with subcommands."""
    result = runner.invoke(main, ["evolve", "--help"])
    assert result.exit_code == 0
    assert "review-last" in result.output
    assert "propose-factor" in result.output


# -- review-last subcommand ---------------------------------------------------


def test_evolve_review_last_help(runner: CliRunner) -> None:
    """evolve review-last command should display help."""
    result = runner.invoke(main, ["evolve", "review-last", "--help"])
    assert result.exit_code == 0
    assert "days" in result.output.lower() or "--review-last" in result.output


def test_evolve_review_last_valid(runner: CliRunner) -> None:
    """evolve review-last should accept valid --days."""
    result = runner.invoke(main, ["evolve", "review-last", "--days", "30"])
    assert result.exit_code == 0


def test_evolve_review_last_default(runner: CliRunner) -> None:
    """evolve review-last should work with default --days."""
    result = runner.invoke(main, ["evolve", "review-last"])
    assert result.exit_code == 0


def test_evolve_review_last_rejects_invalid_days(runner: CliRunner) -> None:
    """evolve review-last should reject non-positive --days."""
    result = runner.invoke(main, ["evolve", "review-last", "--days", "0"])
    assert result.exit_code != 0


# -- propose-factor subcommand -------------------------------------------------


def test_evolve_propose_factor_help(runner: CliRunner) -> None:
    """evolve propose-factor command should display help."""
    result = runner.invoke(main, ["evolve", "propose-factor", "--help"])
    assert result.exit_code == 0
    assert "formula" in result.output.lower()


def test_evolve_propose_factor_valid(runner: CliRunner) -> None:
    """evolve propose-factor should accept a formula string."""
    result = runner.invoke(main, ["evolve", "propose-factor", "--formula", "MOM_5D > 0.02"])
    assert result.exit_code == 0


def test_evolve_propose_factor_missing_formula(runner: CliRunner) -> None:
    """evolve propose-factor should require --formula."""
    result = runner.invoke(main, ["evolve", "propose-factor"])
    assert result.exit_code != 0


def test_evolve_propose_factor_empty_formula(runner: CliRunner) -> None:
    """evolve propose-factor should reject empty formula."""
    result = runner.invoke(main, ["evolve", "propose-factor", "--formula", ""])
    assert result.exit_code != 0


# ============================================================================
# walk-forward command
# ============================================================================


def test_walk_forward_help(runner: CliRunner) -> None:
    """walk-forward command should display help."""
    result = runner.invoke(main, ["walk-forward", "--help"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_walk_forward_valid(runner: CliRunner) -> None:
    """walk-forward command should accept a --version argument."""
    result = runner.invoke(main, ["walk-forward", "--version", "v2.0.0"])
    assert result.exit_code == 0


def test_walk_forward_missing_version(runner: CliRunner) -> None:
    """walk-forward command should require --version."""
    result = runner.invoke(main, ["walk-forward"])
    assert result.exit_code != 0


def test_walk_forward_empty_version(runner: CliRunner) -> None:
    """walk-forward command should reject empty --version."""
    result = runner.invoke(main, ["walk-forward", "--version", ""])
    assert result.exit_code != 0
