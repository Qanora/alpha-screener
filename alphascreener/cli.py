"""Alpha Screener's 60-session, 14-session prediction CLI."""

from __future__ import annotations

from datetime import date

import click
import polars as pl

from alphascreener.display import panel, result_table, rule, warn_card


def _rank_candidates(ohlcv: pl.DataFrame, *, top: int) -> tuple[pl.DataFrame, date]:
    """Rank tradable tickers from their latest 60 trading sessions only."""
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


def _run_screen(top: int) -> None:
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.data.sync import last_sync_date, sync_ohlcv
    from alphascreener.evaluation import write_prediction_ledger

    try:
        ohlcv = scan_ohlcv().collect()
    except FileNotFoundError:
        ohlcv = None
    last = last_sync_date()
    needs_sync = last is None
    if not needs_sync and ohlcv is not None and ohlcv.height:
        needs_sync = (date.today() - last).days > 1 or (last - ohlcv["dt"].min()).days < 90
    if needs_sync:
        click.echo("  Updating data ...")
        try:
            sync_ohlcv()
            ohlcv = scan_ohlcv().collect()
        except Exception as exc:
            warn_card(f"Sync failed: {exc}")
            return
    if ohlcv is None or not ohlcv.height:
        warn_card("No OHLCV data available.")
        return

    data = ohlcv.unique(subset=["ticker", "dt"], keep="last").sort(["ticker", "dt"])
    result, decision_date = _rank_candidates(data, top=top)
    if result.is_empty():
        warn_card("No tickers meet the 60-session tradable-universe requirements.")
        return
    predictions = result.with_columns(pl.lit(decision_date).cast(pl.Date).alias("decision_date"))
    try:
        write_prediction_ledger(predictions.select("ticker", "decision_date", "score"))
    except FileExistsError:
        click.echo("  Ledger: ranking already recorded for this date")

    rule("Alpha Screener")
    click.echo(f"  Date: {decision_date}  |  Candidates: {result.height}\n")
    rows = [
        [str(index), ticker, f"{score:.4f}"]
        for index, (ticker, score) in enumerate(result.iter_rows(), 1)
    ]
    result_table(["#", "Ticker", "Score"], rows)
    click.echo("\n  Ledger: recorded before the 14-session outcome is available.\n")


@click.command()
def evaluate() -> None:
    """Evaluate rankings after their 14-session outcomes mature."""
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.evaluation import (
        compute_forward_labels,
        evaluate_rankings,
        mature_predictions,
        read_prediction_ledger,
    )

    try:
        matured = mature_predictions(
            read_prediction_ledger(), compute_forward_labels(scan_ohlcv().collect())
        )
    except (FileNotFoundError, ValueError) as exc:
        warn_card(f"Cannot evaluate predictions: {exc}")
        return
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


@click.group(invoke_without_command=True)
@click.option("--top", default=10, show_default=True, help="Number of candidates to show.")
@click.version_option(message="Alpha Screener %(version)s", package_name="alpha-screener")
@click.pass_context
def cli(ctx: click.Context, top: int) -> None:
    """Rank US equities and record predictions before outcomes are known."""
    if ctx.invoked_subcommand is None:
        _run_screen(top)


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
