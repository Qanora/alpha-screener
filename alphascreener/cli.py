"""Alpha Screener's 60-session, 14-session prediction CLI."""

from __future__ import annotations

import click
import polars as pl

from alphascreener.display import panel, result_table, rule, warn_card
from alphascreener.ranking import rank_candidates


def _run_screen(top: int) -> None:
    from alphascreener.backtest import run_recent_backtest, write_backtest_records
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.data.sync import MIN_SYNC_COVERAGE, sync_ohlcv
    from alphascreener.evaluation import read_prediction_ledger, write_prediction_ledger

    click.echo("  Updating data ...")
    try:
        sync_status = sync_ohlcv()
        if sync_status.coverage < MIN_SYNC_COVERAGE:
            warn_card(
                "Sync incomplete; no ranking was recorded.",
                f"Decision-ready coverage: {sync_status.coverage:.1%}",
            )
            return
        ohlcv = scan_ohlcv().collect()
        if sync_status.ready_tickers:
            ohlcv = ohlcv.filter(pl.col("ticker").is_in(sync_status.ready_tickers))
    except Exception as exc:
        warn_card(f"Sync failed: {exc}")
        return
    if not ohlcv.height:
        warn_card("No OHLCV data available.")
        return

    data = ohlcv.unique(subset=["ticker", "dt"], keep="last").sort(["ticker", "dt"])
    try:
        backtest = run_recent_backtest(data)
        write_backtest_records(backtest)
        ranking, decision_date = rank_candidates(data)
    except (OSError, ValueError) as exc:
        warn_card(f"Cannot rank candidates: {exc}")
        return
    if ranking.is_empty():
        warn_card("No tickers meet the 60-session tradable-universe requirements.")
        return
    from alphascreener.prediction_contract import STRATEGY_VERSION

    predictions = ranking.with_columns(
        pl.lit(decision_date).cast(pl.Date).alias("decision_date"),
        pl.lit(STRATEGY_VERSION).alias("strategy_version"),
        pl.lit(ranking.height).cast(pl.Int64).alias("universe_size"),
    )
    try:
        write_prediction_ledger(predictions)
    except FileExistsError:
        click.echo("  Ledger: ranking already recorded for this date and strategy")
        try:
            recorded = read_prediction_ledger().filter(
                (pl.col("decision_date") == decision_date)
                & (pl.col("strategy_version") == STRATEGY_VERSION)
            )
            if recorded.is_empty():
                raise ValueError("recorded ranking could not be read")
            ranking = recorded.select("ticker", "score", "rank").sort("rank")
        except (OSError, ValueError) as exc:
            warn_card(f"Cannot read immutable ranking: {exc}")
            return

    displayed = ranking.head(top)
    rule("Alpha Screener")
    click.echo("  Recent current-universe walk-forward backtest:\n")
    result_table(
        ["Decision", "Result", "Precision@10", "Base", "Coverage", "Pass"],
        [
            [
                str(row["decision_date"]),
                str(row["result_date"]),
                f"{row['precision_at_10']:.0%}",
                f"{row['base_explosion_rate']:.1%}",
                f"{row['outcome_coverage']:.1%}",
                "yes" if row["passed"] else "no",
            ]
            for row in backtest.iter_rows(named=True)
        ],
    )
    click.echo()
    click.echo(
        f"  Date: {decision_date}  |  Displayed: {displayed.height}"
        f"  |  Eligible universe: {ranking.height}\n"
    )
    rows = [
        [str(index), ticker, f"{score:.4f}"]
        for index, (ticker, score) in enumerate(
            displayed.select("ticker", "score").iter_rows(), 1
        )
    ]
    result_table(["#", "Ticker", "Score"], rows)
    click.echo("\n  Ledger: recorded before the 14-session outcome is available.\n")


@click.command()
def evaluate() -> None:
    """Evaluate rankings after their 14-session outcomes mature."""
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.evaluation import (
        compute_forward_labels,
        evaluate_daily_rankings,
        longest_consecutive_passes,
        mature_predictions,
        read_prediction_ledger,
    )
    from alphascreener.prediction_contract import STRATEGY_VERSION

    try:
        ohlcv = scan_ohlcv().collect()
        matured = mature_predictions(read_prediction_ledger(), compute_forward_labels(ohlcv))
    except (FileNotFoundError, ValueError) as exc:
        warn_card(f"Cannot evaluate predictions: {exc}")
        return
    daily = evaluate_daily_rankings(matured)
    rule("Alpha Screener — Matured Predictions")
    if daily.is_empty():
        click.echo("  No prediction dates have matured yet.\n")
        return
    current = daily.filter(pl.col("strategy_version") == STRATEGY_VERSION).sort("decision_date")
    result_table(
        ["Decision", "Precision@10", "Base", "Coverage", "Pass"],
        [
            [
                str(row["decision_date"]),
                f"{row['precision_at_k']:.0%}",
                f"{row['base_explosion_rate']:.1%}",
                f"{row['outcome_coverage']:.1%}",
                "yes" if row["passed"] else "no",
            ]
            for row in current.tail(5).iter_rows(named=True)
        ],
    )
    market_dates = ohlcv.filter(pl.col("ticker") == "SPY")["dt"].unique().sort().to_list()
    streak = longest_consecutive_passes(
        daily, market_dates, strategy_version=STRATEGY_VERSION
    )
    panel(
        f"14-session target — {STRATEGY_VERSION}",
        [
            f"Best consecutive passing days: {streak}/5",
            "Target reached: yes" if streak >= 5 else "Target reached: no",
        ],
    )


@click.group(invoke_without_command=True)
@click.option(
    "--top",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of candidates to show.",
)
@click.version_option(message="Alpha Screener %(version)s", package_name="alpha-screener")
@click.pass_context
def cli(ctx: click.Context, top: int) -> None:
    """Rank US equities and record predictions before outcomes are known."""
    if ctx.invoked_subcommand is None:
        _run_screen(top)


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
