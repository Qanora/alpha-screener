"""Click-based CLI for Alpha Screener — rich terminal output.

Commands:
  screen         Full-market coarse scan
  backtest       Historical backtesting
  evolve         Factor evolution review / proposal
  walk-forward   Factor version upgrade validation
  case-library   Breakout case library management

Entry point: ``alphascreener`` (registered in pyproject.toml).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import click

from alphascreener.display import (
    Color,
    empty_state,
    kv,
    kv_table,
    note,
    panel,
    result_table,
    rule,
    success_banner,
    warn_card,
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_MARKET_CHOICES = click.Choice(["US"], case_sensitive=False)
_TOP_RANGE = click.IntRange(min=1, max=100)


def _validate_date_range(
    start: str | None, end: str | None, ctx: click.Context | None = None
) -> None:
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


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--market",
    type=_MARKET_CHOICES,
    default="US",
    show_default=True,
    help="Target market (currently only US supported).",
)
@click.option(
    "--top",
    type=_TOP_RANGE,
    default=20,
    show_default=True,
    help="Number of top candidates to output after dedup.",
)
def screen(market: str, top: int) -> None:
    """Run a full-market coarse screening scan.

    Example:

        alphascreener screen --market US --top 20
    """
    rule("Alpha Screener — Full-Market Scan")

    kv_table([
        ("Market", market.upper()),
        ("Top N", str(top)),
    ])

    try:
        import polars as pl

        from alphascreener.data.io import scan_parquet
    except ImportError:
        warn_card("Data I/O layer not available")
        sys.exit(1)

    try:
        lf = scan_parquet("ohlcv")
        ohlcv = lf.collect()
    except Exception:
        warn_card("No OHLCV data found or format incompatible. Run data sync first.")
        return

    if ohlcv.height == 0:
        warn_card("No OHLCV data found. Run data sync first.")
        return

    latest_date = ohlcv["dt"].max()
    click.echo(f"  {note('Latest data:')} {latest_date}")

    meta = None
    try:
        from alphascreener.universe.meta import read_meta_cache
        meta = read_meta_cache().collect()
    except Exception:
        pass

    # Factor computation needs full time series per ticker (rolling windows),
    # so pass all OHLCV data — not just the latest date.
    df = ohlcv.unique(subset=["ticker", "dt"], keep="last", maintain_order=True).sort(["ticker", "dt"])
    n_tickers = df["ticker"].n_unique()

    try:
        from alphascreener.factors.engine import compute_factors
        from alphascreener.screening.phase1 import hard_filter_with_fallback
        from alphascreener.screening.phase2 import phase2_pipeline

        click.echo(f"  {note('Computing factors for')} {n_tickers} {note('tickers ...')}")
        factors = compute_factors(df, dt=latest_date)

        if meta is not None and meta.height > 0:
            meta_subset = meta.select(["ticker", "sector", "industry"])
            factors = factors.join(meta_subset, on="ticker", how="left")

        filtered, relaxed_used = hard_filter_with_fallback(factors)
        passed = filtered.filter(pl.col("pass_phase1"))
        relax_note = " (relaxed thresholds)" if relaxed_used else ""
        click.echo(f"  {note('Phase 1 pass:')} {passed.height} / {filtered.height}{relax_note}")

        if passed.height == 0:
            warn_card("No tickers passed Phase 1 hard filters (even after relaxation).")
            return

        result = phase2_pipeline(passed, n_final=top)
        click.echo(f"  {note('Phase 2 output:')} {result.height} candidates\n")

        headers = ["#", "Ticker", "Breakout Score"]
        rows = []
        for i, row in enumerate(result.select(["ticker", "breakout_score"]).iter_rows(named=True)):
            rows.append([str(i + 1), str(row["ticker"]), f"{row['breakout_score']:.4f}"])

        result_table(headers, rows)
        click.echo()

    except Exception as e:
        warn_card(f"Error during screening: {e}")
        return


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--start", required=True, metavar="YYYY-MM-DD", help="Backtest start date (inclusive)."
)
@click.option(
    "--end",
    default=None,
    metavar="YYYY-MM-DD",
    help="Backtest end date (inclusive). Defaults to today.",
)
def backtest(start: str, end: str | None) -> None:
    """Run a historical backtest using the SevenDayBreakoutStrategy.

    Example:

        alphascreener backtest --start 2023-01-01 --end 2023-12-31
    """
    _validate_date_range(start, end)

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end) if end else date.today()
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--start/--end") from exc

    rule("Alpha Screener — Backtest")
    kv_table([
        ("Start", s.isoformat()),
        ("End", e.isoformat()),
    ])

    try:
        from alphascreener.backtrader import _load_ohlcv_data, _load_signals_data, run_backtest
    except ImportError:
        warn_card("Backtrader module not available.")
        return

    try:
        ticker_dfs = _load_ohlcv_data(start_date=s, end_date=e)
    except Exception as exc:
        warn_card(f"No OHLCV data available: {exc}")
        return

    if not ticker_dfs:
        warn_card("No OHLCV data available for the given period.")
        return

    signals = _load_signals_data(start_date=s, end_date=e)
    spy_data = ticker_dfs.get("SPY")

    click.echo(f"  {note('Tickers loaded:')} {len(ticker_dfs)}")
    click.echo(f"  {note('Running backtest ...')}\n")

    try:
        result = run_backtest(ticker_dfs, signals=signals, spy_data=spy_data)
    except Exception as exc:
        warn_card(f"Error during backtest: {exc}")
        return

    metrics = result["metrics"]

    panel("Backtest Results", [
        kv("Total Return", f"{metrics['total_return']:.2%}"),
        kv("Annualized Return", f"{metrics['annualized_return']:.2%}"),
        kv("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
        kv("Max Drawdown", f"{metrics['max_drawdown']:.2%}"),
        kv("Win Rate", f"{metrics['win_rate']:.2%}"),
        kv("Volatility", f"{metrics['volatility']:.2%}"),
        kv("Trades", str(result["n_trades"])),
        kv("Final Value", f"${result['final_value']:,.0f}"),
    ])

    extras = []
    if "benchmark_total_return" in metrics:
        extras.append(kv("Benchmark Return", f"{metrics['benchmark_total_return']:.2%}"))
    if "excess_return" in metrics:
        extras.append(kv("Excess Return", f"{metrics['excess_return']:.2%}"))
    if "information_ratio" in metrics:
        extras.append(kv("Information Ratio", f"{metrics['information_ratio']:.2f}"))
    if extras:
        panel("Benchmarks", extras, border=Color.border_dim)
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


@evolve.command(name="review-last")
@click.option(
    "--days",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Number of days to look back for acceptance metrics.",
)
def evolve_review_last(days: int) -> None:
    """Review the last N days of alpha acceptance metrics.

    Example:

        alphascreener evolve review-last --days 30
    """
    from datetime import date as date_type

    rule(f"Alpha Screener — Evolution Review (last {days} days)")

    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from alphascreener.config import Settings
        from alphascreener.db.engine import create_db_engine
        from alphascreener.db.models import AlphaAcceptanceDaily
    except ImportError as exc:
        warn_card(f"Required module not available ({exc})")
        sys.exit(1)

    try:
        settings = Settings()
        engine = create_db_engine(settings.get_db_url())
    except Exception as exc:
        warn_card(f"Failed to initialize database: {exc}")
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

        if isinstance(exc, OperationalError) and (
            "no such table" in str(exc) or "no such column" in str(exc)
        ):
            warn_card(
                "Database schema not ready. Run:\n"
                "  alembic upgrade head\n"
                "to create/migrate the database tables, then retry."
            )
            return
        warn_card(f"No acceptance metrics available: {exc}")
        return
    finally:
        engine.dispose()

    if not records:
        empty_state(f"No acceptance metrics found in the last {days} days.")
        return

    click.echo(f"  {note('Found')} {len(records)} {note('record(s) in the last')} {days} {note('days')}\n")

    headers = [
        "Date", "Base Rate", "P@20 Pure", "P@20 LLM",
        "IC Pure", "IC LLM", "Lift Pure", "N",
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
    result_table(headers, rows)
    click.echo()


@evolve.command(name="propose-factor")
@click.option(
    "--formula",
    required=True,
    metavar="FORMULA",
    help="Factor formula expression (e.g. 'MOM_5D > 0.02').",
)
def evolve_propose_factor(formula: str) -> None:
    """Propose a new factor formula for evaluation.

    Example:

        alphascreener evolve propose-factor --formula "MOM_5D > 0.02"
    """
    if not formula.strip():
        raise click.BadParameter("Formula must not be empty.", param_hint="--formula")

    from alphascreener.factors.formulas import FACTOR_NAMES

    rule("Alpha Screener — Factor Proposal")
    kv_table([("Formula", formula)])

    import re
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    known = set(FACTOR_NAMES)
    unknown = tokens - known - {"and", "or", "not", "True", "False", "None", "is", "in"}

    if unknown:
        warn_card(f"Unrecognized tokens: {', '.join(sorted(unknown))}")
        click.echo(f"  {note('Known factors:')} {', '.join(sorted(known))}")
    else:
        success_banner("All tokens recognized in the factor namespace.")

    click.echo(f"  {note('Proposal recorded (stub — full evaluation pipeline pending).')}\n")


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


@click.command(name="walk-forward")
@click.option(
    "--version",
    required=True,
    metavar="VERSION",
    help="New factor version identifier (e.g. 'v2.0.0').",
)
def walk_forward(version: str) -> None:
    """Validate a factor version upgrade via walk-forward cross-validation.

    Example:

        alphascreener walk-forward --version v2.0.0
    """
    if not version.strip():
        raise click.BadParameter("Version must not be empty.", param_hint="--version")

    rule("Alpha Screener — Walk-Forward Validation")
    kv_table([("Version", version)])

    try:
        from alphascreener.cross_validation.health_monitor import YFinanceHealthMonitor
    except ImportError:
        warn_card("Cross-validation module not available.")
        sys.exit(1)

    monitor = YFinanceHealthMonitor()
    healthy = not monitor.fallback_activated
    click.echo(f"  {note('Data source health:')} {'OK' if healthy else 'DEGRADED'}")

    click.echo(f"  {note('Validation result:')} pending (full pipeline not yet implemented)")
    click.echo(f"  {note('Recommendation:')} hold for manual review\n")


# ---------------------------------------------------------------------------
# case-library (group)
# ---------------------------------------------------------------------------


@click.group()
def case_library() -> None:
    """Manage the breakout case library for similarity search.

    Subcommands:

        init     Build or rebuild the case library from historical data
        status   Show current case library status
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

    Example:

        alphascreener case-library init
        alphascreener case-library init --score-pct 0.80 --min-return 0.15
    """
    rule("Alpha Screener — Case Library Init")
    kv_table([
        ("Score percentile", str(score_pct)),
        ("Min return", str(min_return)),
    ])

    try:
        from alphascreener.tradingagents.case_library import rebuild_case_library

        n = rebuild_case_library(
            breakout_score_pct=score_pct,
            min_return=min_return,
        )
    except Exception as exc:
        raise click.ClickException(f"Error building case library: {exc}") from exc

    if n > 0:
        success_banner(f"Case library built with {n} positive cases.")
    else:
        warn_card(
            "No positive cases found. This may be expected if:\n"
            "  - No historical factor/OHLCV data exists yet\n"
            "  - Threshold is too strict\n"
            "  - Forward returns are not yet available"
        )
    click.echo()


@case_library.command(name="status")
def case_library_status_cmd() -> None:
    """Show the current breakout case library status.

    Example:

        alphascreener case-library status
    """
    rule("Alpha Screener — Case Library Status")

    try:
        from alphascreener.tradingagents.case_library import case_library_status

        info = case_library_status()
    except Exception as exc:
        raise click.ClickException(f"Error reading case library: {exc}") from exc

    pairs = [
        ("Path", info["path"]),
        ("Exists", str(info["exists"])),
        ("Cases", str(info["n_cases"])),
        ("Tickers", str(info["n_unique_tickers"])),
    ]
    if info["date_range"] is not None:
        pairs.append(("Date range", f"{info['date_range'][0]} ~ {info['date_range'][1]}"))

    kv_table(pairs)
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
    # Suppress log noise on stderr — user-facing output is via rich.
    # WARNING/INFO/DEBUG go to file only; ERROR still visible on stderr.
    import logging
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    # Ensure DB schema exists before any subcommand writes data (Issue #214).
    try:
        from alphascreener.config import Settings
        from alphascreener.db.engine import create_db_engine
        from alphascreener.db.ensure_schema import _ensure_schema

        settings = Settings()
        engine = create_db_engine(settings.get_db_url())
        try:
            _ensure_schema(engine)
        finally:
            engine.dispose()
    except Exception:
        pass


main.add_command(screen)
main.add_command(backtest)
main.add_command(evolve)
main.add_command(walk_forward)
main.add_command(case_library)


if __name__ == "__main__":
    main()
