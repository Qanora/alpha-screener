"""Click-based CLI for Alpha Screener (Issue #106).

Commands:
  screen      Full-market coarse scan
  backtest    Historical backtesting
  evolve      Factor evolution review / proposal
  walk-forward Factor version upgrade validation

Entry point: ``alphascreener`` (registered in pyproject.toml).
Reference: PRD 8.5.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import click

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_MARKET_CHOICES = click.Choice(["US"], case_sensitive=False)
_TOP_RANGE = click.IntRange(min=1, max=100)


def _validate_date_range(
    start: str | None, end: str | None, ctx: click.Context | None = None
) -> None:
    """Validate that start <= end when both are provided."""
    if start and end:
        try:
            s = date.fromisoformat(start)
            e = date.fromisoformat(end)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--start/--end") from exc
        if s > e:
            raise click.BadParameter(
                f"start ({start}) must be before end ({end})", param_hint="--start"
            )


def _echo_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned table to stdout."""
    if not rows:
        click.echo("  (no data)")
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    click.echo(fmt.format(*headers))
    click.echo("  " + "  ".join("-" * w for w in col_widths))
    for row in rows:
        click.echo(fmt.format(*[str(c) for c in row]))


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------


@click.command()
@click.option("--market", type=_MARKET_CHOICES, default="US", show_default=True,
              help="Target market (currently only US supported).")
@click.option("--top", type=_TOP_RANGE, default=20, show_default=True,
              help="Number of top candidates to output after dedup.")
def screen(market: str, top: int) -> None:
    """Run a full-market coarse screening scan.

    Loads the most recent OHLCV data, computes technical factors, applies
    Phase 1 hard filters and Phase 2 weighted scoring + industry dedup,
    then prints the top candidates.

    Example:

        alphascreener screen --market US --top 20
    """
    click.echo("\nAlpha Screener — Full-Market Scan")
    click.echo(f"  Market : {market.upper()}")
    click.echo(f"  Top N  : {top}")
    click.echo()

    try:
        import polars as pl

        from alphascreener.data.io import scan_parquet
    except ImportError:
        click.echo("Error: Data I/O layer not available.", err=True)
        sys.exit(1)

    # Find the most recent OHLCV date
    try:
        lf = scan_parquet("ohlcv")
        ohlcv = lf.collect()
    except Exception:
        click.echo(
            "No OHLCV data found or data format incompatible. Run data sync first.",
            err=True,
        )
        return

    if ohlcv.height == 0:
        click.echo("No OHLCV data found. Run data sync first.", err=True)
        return

    latest_date = ohlcv["dt"].max()
    click.echo(f"  Latest data: {latest_date}")

    # Load universe metadata for sector/industry (for Phase 2 dedup)
    meta = None
    try:
        from alphascreener.universe.meta import read_meta_cache

        meta = read_meta_cache().collect()
    except Exception:
        pass

    # Filter to latest date, dedup, compute factors, run screening
    df = ohlcv.filter(pl.col("dt") == latest_date)
    # Defensive dedup: keep last occurrence of duplicate (ticker, dt) rows,
    # then sort so downstream factor computation sees correct time-series order.
    df = df.unique(subset=["ticker", "dt"], keep="last", maintain_order=True).sort(["ticker", "dt"])
    n_tickers = df["ticker"].n_unique()

    try:
        from alphascreener.factors.engine import compute_factors
        from alphascreener.screening.phase1 import hard_filter
        from alphascreener.screening.phase2 import phase2_pipeline

        # Compute factors
        click.echo(f"  Computing factors for {n_tickers} tickers ...")
        factors = compute_factors(df, dt=latest_date)

        # Join sector/industry from universe meta if available
        if meta is not None and meta.height > 0:
            meta_subset = meta.select(["ticker", "sector", "industry"])
            factors = factors.join(meta_subset, on="ticker", how="left")

        # Phase 1 hard filter
        filtered = hard_filter(factors)
        passed = filtered.filter(pl.col("pass_phase1"))
        click.echo(f"  Phase 1 pass: {passed.height} / {filtered.height}")

        if passed.height == 0:
            click.echo("  No tickers passed Phase 1 hard filters.")
            return

        # Phase 2 weighted scoring + dedup
        result = phase2_pipeline(passed, n_final=top)
        click.echo(f"  Phase 2 output: {result.height} candidates\n")

        # Print results
        headers = ["#", "Ticker", "Breakout Score"]
        rows = []
        for i, row in enumerate(result.select(["ticker", "breakout_score"]).iter_rows(named=True)):
            rows.append([str(i + 1), str(row["ticker"]), f"{row['breakout_score']:.4f}"])

        _echo_table(headers, rows)
        click.echo()

    except Exception as e:
        click.echo(f"Error during screening: {e}", err=True)
        return


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@click.command()
@click.option("--start", required=True, metavar="YYYY-MM-DD",
              help="Backtest start date (inclusive).")
@click.option("--end", default=None, metavar="YYYY-MM-DD",
              help="Backtest end date (inclusive). Defaults to today.")
def backtest(start: str, end: str | None) -> None:
    """Run a historical backtest using the SevenDayBreakoutStrategy.

    Loads OHLCV data and signal data for the given date range and runs
    a backtrader backtest with SPY benchmark comparison.

    Example:

        alphascreener backtest --start 2023-01-01 --end 2023-12-31
    """
    _validate_date_range(start, end)

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end) if end else date.today()
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--start/--end") from exc

    click.echo("\nAlpha Screener — Backtest")
    click.echo(f"  Start : {s.isoformat()}")
    click.echo(f"  End   : {e.isoformat()}")
    click.echo()

    try:
        from alphascreener.backtrader import _load_ohlcv_data, _load_signals_data, run_backtest
    except ImportError:
        click.echo("Error: Backtrader module not available.", err=True)
        return

    try:
        ticker_dfs = _load_ohlcv_data(start_date=s, end_date=e)
    except Exception as exc:
        click.echo(f"No OHLCV data available: {exc}", err=True)
        return

    if not ticker_dfs:
        click.echo("No OHLCV data available for the given period.", err=True)
        return

    signals = _load_signals_data(start_date=s, end_date=e)
    spy_data = ticker_dfs.get("SPY")

    click.echo(f"  Tickers loaded : {len(ticker_dfs)}")
    click.echo("  Running backtest ...\n")

    try:
        result = run_backtest(ticker_dfs, signals=signals, spy_data=spy_data)
    except Exception as exc:
        click.echo(f"Error during backtest: {exc}", err=True)
        return

    metrics = result["metrics"]
    click.echo("  Results:")
    click.echo(f"    Total Return      : {metrics['total_return']:.2%}")
    click.echo(f"    Annualized Return : {metrics['annualized_return']:.2%}")
    click.echo(f"    Sharpe Ratio      : {metrics['sharpe_ratio']:.2f}")
    click.echo(f"    Max Drawdown      : {metrics['max_drawdown']:.2%}")
    click.echo(f"    Win Rate          : {metrics['win_rate']:.2%}")
    click.echo(f"    Volatility        : {metrics['volatility']:.2%}")
    click.echo(f"    Trades            : {result['n_trades']}")
    click.echo(f"    Final Value       : ${result['final_value']:,.0f}")

    if "benchmark_total_return" in metrics:
        click.echo(f"    Benchmark Return  : {metrics['benchmark_total_return']:.2%}")
    if "excess_return" in metrics:
        click.echo(f"    Excess Return     : {metrics['excess_return']:.2%}")
    if "information_ratio" in metrics:
        click.echo(f"    Information Ratio : {metrics['information_ratio']:.2f}")
    click.echo()


# ---------------------------------------------------------------------------
# evolve (group)
# ---------------------------------------------------------------------------


@click.group()
def evolve() -> None:
    """Factor evolution: review historical metrics or propose new factors.

    Subcommands:

        review-last     Review recent alpha acceptance metrics
        propose-factor  Propose a new factor formula
    """
    pass


# -- review-last --------------------------------------------------------------


@evolve.command(name="review-last")
@click.option("--days", type=click.IntRange(min=1), default=30, show_default=True,
              help="Number of days to look back for acceptance metrics.")
def evolve_review_last(days: int) -> None:
    """Review the last N days of alpha acceptance metrics.

    Fetches alpha_acceptance_daily records from the database and displays
    the key metrics (base_rate, precision, lift, IC) over the selected window.

    Example:

        alphascreener evolve review-last --days 30
    """
    from datetime import date as date_type

    click.echo(f"\nAlpha Screener — Evolution Review (last {days} days)")
    click.echo()

    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from alphascreener.config import Settings
        from alphascreener.db.engine import create_db_engine
        from alphascreener.db.models import AlphaAcceptanceDaily
    except ImportError as exc:
        click.echo(f"Error: Required module not available ({exc})", err=True)
        sys.exit(1)

    try:
        settings = Settings()
        engine = create_db_engine(settings.get_db_url())
    except Exception as exc:
        click.echo(f"Error: Failed to initialize database: {exc}", err=True)
        return

    try:
        with Session(engine) as session:
            cutoff = date_type.today() - timedelta(days=days)
            stmt = (
                select(AlphaAcceptanceDaily)
                .where(AlphaAcceptanceDaily.metric_date >= cutoff)
                .order_by(AlphaAcceptanceDaily.metric_date.desc())
            )
            records = session.execute(stmt).scalars().all()
    except Exception as exc:
        from sqlalchemy.exc import OperationalError

        err_msg = str(exc)
        if isinstance(exc, OperationalError) and ("no such table" in err_msg or "no such column" in err_msg):
            click.echo(
                "Error: Database schema not ready. Run:\n"
                "  alembic upgrade head\n"
                "to create/migrate the database tables, then retry.",
                err=True,
            )
            return
        click.echo(f"No acceptance metrics available: {exc}", err=True)
        return
    finally:
        engine.dispose()

    if not records:
        click.echo(f"  No acceptance metrics found in the last {days} days.")
        click.echo()
        return

    click.echo(f"  Found {len(records)} record(s) in the last {days} days\n")

    headers = [
        "Date",
        "Base Rate",
        "P@20 Pure",
        "P@20 LLM",
        "IC Pure",
        "IC LLM",
        "Lift Pure",
        "N",
    ]
    rows = []
    for r in records:
        rows.append([
            r.metric_date.isoformat(),
            f"{r.base_rate:.3f}" if r.base_rate is not None else "-",
            f"{r.precision_at_20_pure:.3f}" if r.precision_at_20_pure is not None else "-",
            f"{r.precision_at_20_llm:.3f}" if r.precision_at_20_llm is not None else "-",
            f"{r.ic_pure:.3f}" if r.ic_pure is not None else "-",
            f"{r.ic_llm:.3f}" if r.ic_llm is not None else "-",
            f"{r.lift_at_20_pure:.2f}" if r.lift_at_20_pure is not None else "-",
            str(r.sample_size) if r.sample_size is not None else "-",
        ])
    _echo_table(headers, rows)
    click.echo()


# -- propose-factor ------------------------------------------------------------


@evolve.command(name="propose-factor")
@click.option("--formula", required=True, metavar="FORMULA",
              help="Factor formula expression (e.g. 'MOM_5D > 0.02').")
def evolve_propose_factor(formula: str) -> None:
    """Propose a new factor formula for evaluation.

    Validates the formula syntax and records it for backtesting.
    The formula is parsed against the known factor namespace.

    Example:

        alphascreener evolve propose-factor --formula "MOM_5D > 0.02"
    """
    if not formula.strip():
        raise click.BadParameter("Formula must not be empty.", param_hint="--formula")

    from alphascreener.factors.formulas import FACTOR_NAMES

    click.echo("\nAlpha Screener — Factor Proposal")
    click.echo(f"  Formula : {formula}")
    click.echo()

    # Validate that tokens in the formula reference known factor names
    # Extract identifiers from the formula (naive tokenization)
    import re

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    known = set(FACTOR_NAMES)
    unknown = tokens - known - {"and", "or", "not", "True", "False", "None", "is", "in"}

    if unknown:
        click.echo(f"  Warning: Unrecognized tokens: {', '.join(sorted(unknown))}")
        click.echo(f"  Known factors: {', '.join(sorted(known))}")
    else:
        click.echo("  All tokens recognized in the factor namespace.")

    click.echo("  Proposal recorded (stub — full evaluation pipeline pending).")
    click.echo()


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


@click.command(name="walk-forward")
@click.option("--version", required=True, metavar="VERSION",
              help="New factor version identifier (e.g. 'v2.0.0').")
def walk_forward(version: str) -> None:
    """Validate a factor version upgrade via walk-forward cross-validation.

    Compares the new factor version's predictions against the current
    production version on out-of-sample data, reporting stability and
    predictive performance metrics.

    Example:

        alphascreener walk-forward --version v2.0.0
    """
    if not version.strip():
        raise click.BadParameter("Version must not be empty.", param_hint="--version")

    click.echo("\nAlpha Screener — Walk-Forward Validation")
    click.echo(f"  Version : {version}")
    click.echo()

    try:
        from alphascreener.cross_validation.health_monitor import YFinanceHealthMonitor
    except ImportError:
        click.echo("Error: Cross-validation module not available.", err=True)
        sys.exit(1)

    # Walk-forward checks data source health via cross-validation
    monitor = YFinanceHealthMonitor()
    healthy = not monitor.fallback_activated
    click.echo(f"  Data source health : {'OK' if healthy else 'DEGRADED'}")

    # Stub: load factor data for the proposed version and compare
    click.echo("  Validation result  : pending (full pipeline not yet implemented)")
    click.echo("  Recommendation     : hold for manual review")
    click.echo()


# ---------------------------------------------------------------------------
# case-library (group)
# ---------------------------------------------------------------------------


@click.group()
def case_library() -> None:
    """Manage the breakout case library for similarity search.

    Subcommands:

        init     Build or rebuild the case library from historical data
        status   Show current case library status

    The case library is used by the Breakout Analyst to find similar
    historical breakout patterns when evaluating new candidates.
    """
    pass


@case_library.command(name="init")
@click.option(
    "--score-pct",
    type=click.FloatRange(0.0, 1.0),
    default=0.75,
    show_default=True,
    help="Percentile threshold for breakout_score (0.75 = top 25%).",
)
@click.option(
    "--min-return",
    type=click.FloatRange(0.0, 1.0),
    default=0.10,
    show_default=True,
    help="Minimum T+7 forward return as a decimal (0.10 = 10%).",
)
def case_library_init(score_pct: float, min_return: float) -> None:
    """Build or rebuild the breakout case library from historical data.

    Scans all available factor data in the Parquet store, computes T+7
    forward returns from OHLCV data, identifies positive breakout cases
    (high breakout score + strong forward return), and writes them to
    ~/.alphascreener/data/case_library/cases.parquet.

    Example:

        alphascreener case-library init

        alphascreener case-library init --score-pct 0.80 --min-return 0.15
    """
    click.echo("\nAlpha Screener — Case Library Init")
    click.echo(f"  Score percentile : {score_pct}")
    click.echo(f"  Min return       : {min_return}")
    click.echo()

    try:
        from alphascreener.tradingagents.case_library import rebuild_case_library

        n = rebuild_case_library(
            breakout_score_pct=score_pct,
            min_return=min_return,
        )
    except Exception as exc:
        raise click.ClickException(f"Error building case library: {exc}") from exc

    if n > 0:
        click.echo(f"  Case library built with {n} positive cases.")
    else:
        click.echo(
            "  No positive cases found. This may be expected if:\n"
            "    - No historical factor/OHLCV data exists yet\n"
            "    - Threshold is too strict\n"
            "    - Forward returns are not yet available (need OHLCV data T+7 later)"
        )
    click.echo()


@case_library.command(name="status")
def case_library_status_cmd() -> None:
    """Show the current breakout case library status.

    Displays the number of cases, unique tickers, and date range
    of the cases.parquet file.

    Example:

        alphascreener case-library status
    """
    click.echo("\nAlpha Screener — Case Library Status")
    click.echo()

    try:
        from alphascreener.tradingagents.case_library import case_library_status

        info = case_library_status()
    except Exception as exc:
        raise click.ClickException(f"Error reading case library: {exc}") from exc

    click.echo(f"  Path       : {info['path']}")
    click.echo(f"  Exists     : {info['exists']}")
    click.echo(f"  Cases      : {info['n_cases']}")
    click.echo(f"  Tickers    : {info['n_unique_tickers']}")
    if info["date_range"] is not None:
        click.echo(f"  Date range : {info['date_range'][0]} ~ {info['date_range'][1]}")
    click.echo()


# ---------------------------------------------------------------------------
# Main CLI group and entry point
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(message="Alpha Screener v0.1.0", package_name="alpha-screener")
def main() -> None:
    """Alpha Screener — AI-native quantitative strategy validation platform.

    A CLI toolkit for US equity factor screening, backtesting, factor
    evolution, and walk-forward validation.

    Run 'alphascreener COMMAND --help' for detailed usage of each command.
    """
    pass


main.add_command(screen)
main.add_command(backtest)
main.add_command(evolve)
main.add_command(walk_forward)
main.add_command(case_library)


if __name__ == "__main__":
    main()
