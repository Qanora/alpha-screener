"""Alpha Screener CLI — US equity 14-session breakout prediction.

Usage:
    asc                    Rank candidates and record an immutable prediction ledger
    asc --top 20           Show top 20 candidates
    asc evaluate           Evaluate rankings whose 14-session outcomes matured
    asc backtest TICKER    Backtest a specific ticker
    asc dev ...            Advanced tools (evolution, case-library)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import click
import polars as pl

from alphascreener.data.paths import get_data_home
from alphascreener.display import panel, result_table, rule, warn_card


def _n(t): return t


def _suppress_log_noise() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


# ═══════════════════════════════════════════════════════════════════════════════
# main — default command: screen + auto-backtest
# ═══════════════════════════════════════════════════════════════════════════════


def _rank_candidates(ohlcv: pl.DataFrame, *, top: int) -> tuple[pl.DataFrame, date]:
    """Rank eligible tickers from their latest 60 trading sessions only."""
    from alphascreener.features import compute_60d_features
    from alphascreener.universe import build_universe_snapshot

    cutoff = ohlcv["dt"].max()
    snapshot = build_universe_snapshot(ohlcv, cutoff_date=cutoff)
    eligible = snapshot.filter(pl.col("eligible"))["ticker"].to_list()
    if not eligible:
        return pl.DataFrame(schema={"ticker": pl.String, "score": pl.Float64}), cutoff
    window = (
        ohlcv.filter(pl.col("ticker").is_in(eligible)).sort(["ticker", "dt"])
        .group_by("ticker", maintain_order=True).tail(60)
    )
    features = compute_60d_features(window).filter(pl.col("dt") == cutoff)
    signals = [
        "return_5d", "return_20d", "distance_to_60d_high",
        "volume_zscore_20", "relative_strength_20d",
    ]
    ranked = features.with_columns([
        pl.col(signal).fill_null(0.0).rank("average").alias(f"_rank_{signal}") for signal in signals
    ]).with_columns(
        pl.mean_horizontal([pl.col(f"_rank_{signal}") for signal in signals]).alias("score")
    )
    return ranked.select("ticker", "score").sort("score", descending=True).head(top), cutoff


def _run_screen(top: int, no_backtest: bool, market: str) -> None:
    _suppress_log_noise()

    try:
        import polars as pl

        from alphascreener.data.io import scan_parquet
    except ImportError:
        warn_card("Data I/O layer not available.")
        return

    ohlcv = None
    try:
        ohlcv = scan_parquet("ohlcv").collect()
    except Exception:
        pass

    # Auto-sync if stale or insufficient for the 60-session contract.
    from alphascreener.data.sync import last_sync_date, sync_ohlcv
    last = last_sync_date()
    needs_sync = last is None or (date.today() - last).days > 1
    if not needs_sync and ohlcv is not None and ohlcv.height > 0:
        data_span = (last - ohlcv["dt"].min()).days if last else 0
        if data_span < 90:
            needs_sync = True
    if needs_sync:
        click.echo(f"  {_n('Updating data ...')}")
        try:
            n = sync_ohlcv(progress_callback=None)
            ohlcv = scan_parquet("ohlcv").collect()
            if n == 0:
                warn_card("Sync returned 0 new rows. Results may be based on stale data.")
        except Exception:
            warn_card("Sync failed. Results based on existing data — may be unreliable.")

    if ohlcv is None or ohlcv.height == 0:
        warn_card("No OHLCV data. Run asc sync first.")
        return

    latest_date = ohlcv["dt"].max()
    n_tickers = ohlcv["ticker"].n_unique()
    data_start = ohlcv["dt"].min()
    data_days = (latest_date - data_start).days

    click.echo(
        f"  {_n('Data:')} {n_tickers} tickers, {data_start} -> "
        f"{latest_date} ({data_days}d)"
    )

    df = ohlcv.unique(
        subset=["ticker", "dt"], keep="last", maintain_order=True
    ).sort(["ticker", "dt"])

    result, latest_date = _rank_candidates(df, top=top)
    if result.is_empty():
        warn_card("No tickers meet the 60-session tradable-universe requirements.")
        return
    from alphascreener.evaluation import write_prediction_ledger
    predictions = result.with_columns(pl.lit(latest_date).cast(pl.Date).alias("decision_date"))
    try:
        write_prediction_ledger(predictions.select("ticker", "decision_date", "score"))
    except FileExistsError:
        click.echo("  Ledger: ranking already recorded for this date")

    rule("Alpha Screener")
    click.echo(f"  {_n('Date:')} {latest_date}  |  "
               f"{_n('Candidates:')} {result.height}  |  "
               f"{_n('Data:')} {df['ticker'].n_unique()} tickers\n")

    headers = ["#", "Ticker", "Score"]
    rows = []
    for i, row in enumerate(result.iter_rows(named=True)):
        rows.append([str(i + 1), str(row["ticker"]), f"{row['score']:.4f}"])
    result_table(headers, rows)
    click.echo("\n  Ledger: recorded before the 14-session outcome is available.\n")


@click.command()
def evaluate() -> None:
    """Evaluate ledger rankings after their 14-session outcomes mature."""
    from alphascreener.data.io import scan_parquet
    from alphascreener.evaluation import (
        compute_forward_labels,
        evaluate_rankings,
        mature_predictions,
        read_prediction_ledger,
    )

    try:
        predictions = read_prediction_ledger()
        labels = compute_forward_labels(scan_parquet("ohlcv").collect())
    except (FileNotFoundError, ValueError) as exc:
        warn_card(f"Cannot evaluate predictions: {exc}")
        return
    matured = mature_predictions(predictions, labels)
    metrics = evaluate_rankings(matured)
    rule("Alpha Screener — Matured Predictions")
    if not metrics["days"]:
        click.echo("  No prediction dates have matured yet.\n")
        return
    panel("14-session prediction quality", [
        f"Decision dates: {metrics['days']}",
        f"Precision@10: {metrics['precision_at_k']:.3f}",
        f"Lift@10: {metrics['lift_at_k']:.2f}",
        f"Mean forward return: {metrics['mean_forward_return']:.2%}",
        f"Bootstrap CI: [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]",
    ])


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

        asc backtest AAPL

        asc backtest AAPL --start 2023-01-01 --end 2024-12-31
    """
    _suppress_log_noise()

    try:
        s = date.fromisoformat(start) if start else date.today() - timedelta(days=183)  # 6 months
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
        f"{_n('Total Return:')}      {m['total_return']:.2%}",
        f"{_n('Annualized Return:')} {m['annualized_return']:.2%}",
        f"{_n('Sharpe Ratio:')}      {m['sharpe_ratio']:.2f}",
        f"{_n('Max Drawdown:')}      {m['max_drawdown']:.2%}",
        f"{_n('Win Rate:')}          {m['win_rate']:.2%}",
        f"{_n('Volatility:')}        {m['volatility']:.2%}",
        f"{_n('Trades:')}            {bt['n_trades']}",
    ])

    # Benchmark
    spy = ticker_dfs.get("SPY")
    if spy is not None and not spy.height == 0:
        try:
            bench = run_backtest({"SPY": spy}, signals=signals)
            bm = bench["metrics"]
            click.echo(f"  {_n('SPY:')} return {bm['total_return']:.1%}  "
                       f"sharpe {bm['sharpe_ratio']:.2f}  "
                       f"excess {m.get('excess_return', 0):.1%}")
        except Exception:
            pass

    click.echo()


# ═══════════════════════════════════════════════════════════════════════════════
# dev — advanced tools (hidden from top-level help)
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--rounds", default=50, show_default=True, help="Max optimization windows.")
@click.option("--train", default=2, show_default=True, help="Training window (years).")
@click.option(
    "--regime-filter/--no-regime-filter",
    default=True,
    show_default=True,
    help="Only activate strategy in bull regime (>60%% up-days in 63-day lookback; Singha 2025)."
)
def optimize(rounds: int, train: int, regime_filter: bool) -> None:
    """Optimize factor weights via walk-forward backtesting.

    Iteratively perturbs weights and measures out-of-sample performance
    on rolling train/test windows. Outputs convergence report with
    Precision@20, Lift@20, Sharpe, and Max Drawdown per window.

    Example:

        asc optimize

        asc optimize --rounds 100 --train 3

        asc optimize --regime-filter
    """
    _suppress_log_noise()
    rule("Alpha Screener — Weight Optimization")

    from alphascreener.data.io import scan_parquet
    from alphascreener.optimize import optimize_weights
    from alphascreener.screening.phase2 import MVP_WEIGHTS

    click.echo(f"  {_n('Loading OHLCV data ...')}")
    try:
        ohlcv = scan_parquet("ohlcv").collect()
    except Exception:
        warn_card("No OHLCV data. Run asc sync first.")
        return

    if ohlcv.height == 0:
        warn_card("No OHLCV data found.")
        return

    # ── Load universe metadata (Issue #325) ──
    universe_meta = None
    universe_path = get_data_home() / "universe_meta.parquet"
    umeta_label = f"  {_n('Universe meta:')}"
    if universe_path.exists():
        try:
            universe_meta = pl.read_parquet(universe_path)
            click.echo(f"{umeta_label} {universe_meta.height} tickers with sector/industry")
        except Exception:
            click.echo(f"{umeta_label} failed to load, proceeding without industry dedup")
    else:
        click.echo(f"{umeta_label} not found, proceeding without industry dedup")

    # Dedup and sort
    ohlcv = (
        ohlcv
        .unique(subset=["ticker", "dt"], keep="last", maintain_order=True)
        .sort(["ticker", "dt"])
    )
    data_start = ohlcv["dt"].min()
    data_end = ohlcv["dt"].max()
    n_tickers = ohlcv["ticker"].n_unique()

    click.echo(f"  {_n('Data:')} {n_tickers} tickers, {data_start} → {data_end}")
    click.echo(f"  {_n('Initial weights:')} {len(MVP_WEIGHTS)} factors, train={train}y\n")

    report = optimize_weights(
        ohlcv,
        MVP_WEIGHTS,
        train_years=train,
        test_months=6,
        step_months=6,
        max_windows=rounds,
        universe_meta=universe_meta,
        regime_filter=regime_filter,
    )

    # ── Output ──
    click.echo(
        f"  {_n('Windows evaluated:')} {report.iterations}  |  "
        f"{_n('Converged:')} {report.converged}\n"
    )

    # Weight changes
    if report.weight_changes:
        click.echo(f"  {_n('Factor Weight Evolution:')}")
        sorted_changes = sorted(
            report.weight_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for factor, delta in sorted_changes:
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "—"
            click.echo(
                f"    {factor:20s}  "
                f"{report.initial_weights.get(factor,0):.3f} → "
                f"{report.final_weights.get(factor,0):.3f}  "
                f"{delta:+.3f} {direction}"
            )
        click.echo()

    # Window results
    if report.windows:
        headers = ["Window", "Train→Test", "P@20", "Lift", "Sharpe", "MaxDD"]
        rows = []
        for i, w in enumerate(report.windows):
            rows.append([
                str(i + 1),
                f"{w.test_start}→{w.test_end}",
                f"{w.precision_at_20:.3f}",
                f"{w.lift_at_20:.2f}",
                f"{w.sharpe:.2f}",
                f"{w.max_drawdown:.1%}",
            ])
        result_table(headers, rows)
        click.echo()

    if report.final_weights:
        click.echo(f"  {_n('Weights:')} computed for this report only; source was not modified.\n")


@click.command()
@click.option("--full", is_flag=True, help="Full re-download.")
def sync(full: bool) -> None:
    """Update OHLCV data from Yahoo Finance."""
    _suppress_log_noise()
    rule("Alpha Screener — Data Sync")

    from alphascreener.data.sync import _default_universe, last_sync_date, sync_ohlcv

    tickers = _default_universe()
    last = last_sync_date()
    click.echo(f"  {_n('Tickers:')} {len(tickers)}  |  {_n('Last sync:')} {last or 'never'}\n")

    def progress(total, batch, batches):
        pct = min(100, int(batch / max(batches, 1) * 100))
        click.echo(f"\r  {_n(f'[{pct}%]')} batch {batch}/{batches}", nl=False)

    try:
        n = sync_ohlcv(tickers, progress_callback=progress)
        click.echo(f"\n  {_n('New rows:')} {n}")
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
        click.echo(f"  {_n('No metrics in last')} {days} {_n('days.')}\n")
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
    hidden=True,
    help="Deprecated; the default command no longer runs a backtest.",
)
@click.option("--market", default="US", hidden=True)
@click.version_option(message="Alpha Screener v0.1.0", package_name="alpha-screener")
@click.pass_context
def cli(ctx: click.Context, top: int, no_backtest: bool, market: str) -> None:
    """Alpha Screener — US equity 14-session breakout prediction.

    Default (no subcommand): scan the tradable US universe, show top breakout
    candidates, and record the ranking before outcomes are available.

    \b
    Examples:
      asc                  # default: top 10 + prediction ledger
      asc --top 5          # top 5 + prediction ledger
      asc evaluate         # evaluate matured prediction dates
      asc backtest AAPL     # backtest a specific ticker
    """
    if ctx.invoked_subcommand is not None:
        return
    _run_screen(top=top, no_backtest=no_backtest, market=market)


cli.add_command(backtest)
cli.add_command(evaluate)
cli.add_command(optimize)
cli.add_command(sync)
cli.add_command(dev)

if __name__ == "__main__":
    cli()
