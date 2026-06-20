"""Alpha Screener CLI — US equity breakout screening + backtest.

Usage:
    alphascreener                    Run full scan + auto-backtest (default)
    alphascreener --top 20           Show top 20 candidates
    alphascreener --no-backtest      Skip backtest, show screening only
    alphascreener backtest TICKER    Backtest a specific ticker
    alphascreener dev ...            Advanced tools (evolution, case-library)
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

import click

from alphascreener.display import Color, kv_table, note, panel, result_table, rule, warn_card


def _suppress_log_noise() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


# ═══════════════════════════════════════════════════════════════════════════════
# main — default command: screen + auto-backtest
# ═══════════════════════════════════════════════════════════════════════════════


def _run_screen(top: int, no_backtest: bool, market: str) -> None:
    _suppress_log_noise()

    # Auto-sync if data is stale (>1 day)
    from alphascreener.data.sync import last_sync_date, sync_ohlcv
    last = last_sync_date()
    if last is None or (date.today() - last).days > 1:
        click.echo(f"  {note('Data stale, auto-updating ...')}")
        try:
            sync_ohlcv(progress_callback=None)
        except Exception:
            pass  # proceed with existing data if sync fails

    try:
        import polars as pl
        from alphascreener.data.io import scan_parquet
    except ImportError:
        warn_card("Data I/O layer not available.")
        return

    # ── Load OHLCV ────────────────────────────────────────────────────────
    try:
        ohlcv = scan_parquet("ohlcv").collect()
    except Exception:
        warn_card("No OHLCV data. Run data sync first.")
        return

    if ohlcv.height == 0:
        warn_card("No OHLCV data found.")
        return

    latest_date = ohlcv["dt"].max()
    df = ohlcv.unique(subset=["ticker", "dt"], keep="last", maintain_order=True).sort(["ticker", "dt"])

    # ── Factors ───────────────────────────────────────────────────────────
    try:
        from alphascreener.factors.engine import compute_factors
        from alphascreener.screening.phase1 import hard_filter_with_fallback
        from alphascreener.screening.phase2 import phase2_pipeline
    except ImportError as exc:
        warn_card(f"Required module not available: {exc}")
        return

    factors = compute_factors(df, dt=latest_date)

    # ── Screening ─────────────────────────────────────────────────────────
    filtered, relaxed_used = hard_filter_with_fallback(factors)
    passed = filtered.filter(pl.col("pass_phase1"))
    relax_note = " (relaxed)" if relaxed_used else ""

    if passed.height == 0:
        rule("Alpha Screener")
        warn_card("No tickers passed screening. Try again with more data.")
        return

    result = phase2_pipeline(passed, n_final=top)

    rule("Alpha Screener")
    click.echo(f"  {note('Date:')} {latest_date}  |  "
               f"{note('Passed:')} {result.height}/{filtered.height}{relax_note}  |  "
               f"{note('Data:')} {df['ticker'].n_unique()} tickers\n")

    # ── Results table ─────────────────────────────────────────────────────
    headers = ["#", "Ticker", "Score"]
    rows = []
    tickers = []
    for i, row in enumerate(result.select(["ticker", "breakout_score"]).iter_rows(named=True)):
        rows.append([str(i + 1), str(row["ticker"]), f"{row['breakout_score']:.4f}"])
        tickers.append(str(row["ticker"]))

    result_table(headers, rows)

    # ── Auto-backtest ─────────────────────────────────────────────────────
    if no_backtest:
        click.echo(f"\n  {note('Run')} alphascreener backtest TICKER {note('for detailed backtest.')}\n")
        return

    click.echo(f"\n  {note('Running backtest on top candidates ...')}\n")

    try:
        from alphascreener.backtrader import _load_ohlcv_data, _load_signals_data, run_backtest
    except ImportError:
        warn_card("Backtrader module not available.")
        return

    # Use a reasonable backtest window
    s = date.today() - timedelta(days=365 * 2)
    e = date.today()

    try:
        ticker_dfs = _load_ohlcv_data(start_date=s, end_date=e)
    except Exception:
        warn_card("Not enough OHLCV history for backtest.")
        return

    signals = _load_signals_data(start_date=s, end_date=e)

    bt_headers = ["Ticker", "Return", "Ann.Ret", "Sharpe", "MaxDD", "Win%"]
    bt_rows = []

    for ticker in tickers[:5]:  # backtest top 5
        df_t = ticker_dfs.get(ticker)
        if df_t is None or df_t.height == 0:
            continue
        try:
            bt = run_backtest({ticker: df_t}, signals=signals)
            m = bt["metrics"]
            bt_rows.append([
                ticker,
                f"{m['total_return']:.1%}",
                f"{m['annualized_return']:.1%}",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['max_drawdown']:.1%}",
                f"{m['win_rate']:.1%}",
            ])
        except Exception:
            continue

    if bt_rows:
        panel(f"Backtest ({s.isoformat()} → {e.isoformat()})", [])
        result_table(bt_headers, bt_rows)

    # Benchmark
    spy = ticker_dfs.get("SPY")
    if spy is not None and not spy.height == 0:
        try:
            bench = run_backtest({"SPY": spy}, signals=signals)
            click.echo(f"  {note('SPY benchmark:')} "
                       f"return {bench['metrics']['total_return']:.1%}  "
                       f"sharpe {bench['metrics']['sharpe_ratio']:.2f}")
        except Exception:
            pass

    click.echo()


# ═══════════════════════════════════════════════════════════════════════════════
# backtest — explicit single-ticker backtest
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.argument("ticker")
@click.option("--start", default=None, metavar="YYYY-MM-DD", help="Start date (default: 2yr ago).")
@click.option("--end", default=None, metavar="YYYY-MM-DD", help="End date (default: today).")
def backtest(ticker: str, start: str | None, end: str | None) -> None:
    """Backtest a specific ticker.

    Example:

        alphascreener backtest AAPL

        alphascreener backtest AAPL --start 2023-01-01 --end 2024-12-31
    """
    _suppress_log_noise()

    try:
        s = date.fromisoformat(start) if start else date.today() - timedelta(days=365 * 2)
        e = date.fromisoformat(end) if end else date.today()
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    rule(f"Alpha Screener — Backtest {ticker.upper()}")
    click.echo()

    try:
        from alphascreener.backtrader import _load_ohlcv_data, _load_signals_data, run_backtest
    except ImportError:
        warn_card("Backtrader module not available.")
        return

    try:
        ticker_dfs = _load_ohlcv_data(start_date=s, end_date=e)
    except Exception as exc:
        warn_card(f"No OHLCV data: {exc}")
        return

    df_t = ticker_dfs.get(ticker.upper())
    if df_t is None or df_t.height == 0:
        warn_card(f"No OHLCV data for {ticker.upper()}.")
        return

    signals = _load_signals_data(start_date=s, end_date=e)

    try:
        bt = run_backtest({ticker.upper(): df_t}, signals=signals)
    except Exception as exc:
        warn_card(f"Backtest failed: {exc}")
        return

    m = bt["metrics"]
    panel(f"Backtest — {ticker.upper()}  ({s.isoformat()} → {e.isoformat()})", [
        f"{note('Total Return:')}      {m['total_return']:.2%}",
        f"{note('Annualized Return:')} {m['annualized_return']:.2%}",
        f"{note('Sharpe Ratio:')}      {m['sharpe_ratio']:.2f}",
        f"{note('Max Drawdown:')}      {m['max_drawdown']:.2%}",
        f"{note('Win Rate:')}          {m['win_rate']:.2%}",
        f"{note('Volatility:')}        {m['volatility']:.2%}",
        f"{note('Trades:')}            {bt['n_trades']}",
    ])

    # Benchmark
    spy = ticker_dfs.get("SPY")
    if spy is not None and not spy.height == 0:
        try:
            bench = run_backtest({"SPY": spy}, signals=signals)
            bm = bench["metrics"]
            click.echo(f"  {note('SPY:')} return {bm['total_return']:.1%}  "
                       f"sharpe {bm['sharpe_ratio']:.2f}  "
                       f"excess {m.get('excess_return', 0):.1%}")
        except Exception:
            pass

    click.echo()


# ═══════════════════════════════════════════════════════════════════════════════
# dev — advanced tools (hidden from top-level help)
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--full", is_flag=True, help="Full re-download.")
def sync(full: bool) -> None:
    """Update OHLCV data from Yahoo Finance."""
    _suppress_log_noise()
    rule("Alpha Screener — Data Sync")

    from alphascreener.data.sync import _default_universe, last_sync_date, sync_ohlcv

    tickers = _default_universe()
    last = last_sync_date()
    click.echo(f"  {note('Tickers:')} {len(tickers)}  |  {note('Last sync:')} {last or 'never'}\n")

    def progress(total, batch, batches):
        pct = min(100, int(batch / max(batches, 1) * 100))
        click.echo(f"\r  {note(f'[{pct}%]')} batch {batch}/{batches}", nl=False)

    try:
        n = sync_ohlcv(tickers, progress_callback=progress)
        click.echo(f"\n  {note('New rows:')} {n}")
    except Exception as exc:
        warn_card(f"Sync failed: {exc}")
        return

    click.echo()


@click.group(hidden=True)
def dev() -> None:
    """Advanced tools."""
    pass


@dev.command()
@click.option("--days", default=30, show_default=True)
def review(days: int) -> None:
    """Review alpha acceptance metrics."""
    _suppress_log_noise()
    rule(f"Evolution Review (last {days} days)")

    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from alphascreener.config import Settings
        from alphascreener.db.engine import create_db_engine
        from alphascreener.db.models import AlphaAcceptanceDaily
    except ImportError as exc:
        warn_card(f"Required module not available: {exc}")
        return

    settings = Settings()
    engine = create_db_engine(settings.get_db_url())
    try:
        with Session(engine) as session:
            cutoff = date.today() - timedelta(days=days)
            stmt = (
                select(AlphaAcceptanceDaily)
                .where(AlphaAcceptanceDaily.metric_date >= cutoff)
                .order_by(AlphaAcceptanceDaily.metric_date.desc())
            )
            records = session.execute(stmt).scalars().all()
    except Exception as exc:
        warn_card(f"No data: {exc}")
        return
    finally:
        engine.dispose()

    if not records:
        click.echo(f"  {note('No metrics in last')} {days} {note('days.')}\n")
        return

    headers = ["Date", "Base Rate", "P@20 Pure", "P@20 LLM", "IC Pure", "IC LLM", "N"]
    rows = []
    for r in records:
        rows.append([
            r.metric_date.isoformat(),
            f"{r.base_rate:.3f}" if r.base_rate is not None else "-",
            f"{r.precision_at_20_pure:.3f}" if r.precision_at_20_pure is not None else "-",
            f"{r.precision_at_20_llm:.3f}" if r.precision_at_20_llm is not None else "-",
            f"{r.ic_pure:.3f}" if r.ic_pure is not None else "-",
            f"{r.ic_llm:.3f}" if r.ic_llm is not None else "-",
            str(r.sample_size) if r.sample_size is not None else "-",
        ])
    result_table(headers, rows)
    click.echo()


# CLI assembly
# ═══════════════════════════════════════════════════════════════════════════════


@click.group(invoke_without_command=True)
@click.option("--top", default=10, show_default=True, help="Number of candidates to show.")
@click.option(
    "--no-backtest",
    is_flag=True,
    help="Skip automatic backtest, show screening results only.",
)
@click.option("--market", default="US", hidden=True)
@click.version_option(message="Alpha Screener v0.1.0", package_name="alpha-screener")
@click.pass_context
def cli(ctx: click.Context, top: int, no_backtest: bool, market: str) -> None:
    """Alpha Screener — US equity breakout screening + backtest.

    Default (no subcommand): scan the full US market, show top breakout
    candidates, then backtest each one.

    \b
    Examples:
      alphascreener                  # default: top 10 + backtest
      alphascreener --top 5           # top 5 + backtest
      alphascreener --no-backtest     # screening only
      alphascreener backtest AAPL     # backtest a specific ticker
    """
    if ctx.invoked_subcommand is not None:
        return
    _run_screen(top=top, no_backtest=no_backtest, market=market)


cli.add_command(backtest)
cli.add_command(sync)
cli.add_command(dev)

if __name__ == "__main__":
    cli()
