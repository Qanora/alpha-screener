"""Alpha Screener's 60-session, 14-session prediction CLI."""

from __future__ import annotations

from datetime import date

import click
import polars as pl

from alphascreener.display import panel, result_table, rule, warn_card


def _rank_candidates(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, date]:
    """Rank the complete eligible universe from its latest 60 sessions."""
    from alphascreener.features import compute_60d_features
    from alphascreener.universe import build_universe_snapshot

    cutoff = ohlcv["dt"].max()
    snapshot = build_universe_snapshot(ohlcv, cutoff_date=cutoff)
    eligible = snapshot.filter(pl.col("eligible") & (pl.col("ticker") != "SPY"))["ticker"].to_list()
    if not eligible:
        return pl.DataFrame(schema={"ticker": pl.String, "score": pl.Float64}), cutoff
    feature_tickers = [*eligible, "SPY"]
    window = (
        ohlcv.filter(pl.col("ticker").is_in(feature_tickers)).sort(["ticker", "dt"])
        .group_by("ticker", maintain_order=True).tail(60)
    )
    features = compute_60d_features(window).filter(
        (pl.col("dt") == cutoff) & (pl.col("ticker") != "SPY")
    )
    signals = [
        "return_5d", "return_20d", "distance_to_60d_high",
        "volume_zscore_20", "relative_strength_20d",
    ]
    ranked = features.with_columns([
        pl.col(signal).fill_null(0.0).rank("average").alias(f"_rank_{signal}") for signal in signals
    ]).with_columns(
        pl.mean_horizontal([pl.col(f"_rank_{signal}") for signal in signals]).alias("score")
    )
    ranking = (
        ranked.select("ticker", "score")
        .sort("score", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.Int64))
    )
    return ranking, cutoff


def _run_screen(top: int) -> None:
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.data.sync import MIN_SYNC_COVERAGE, last_sync_date, sync_ohlcv
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
            sync_status = sync_ohlcv()
            if sync_status.coverage < MIN_SYNC_COVERAGE:
                warn_card(
                    "Sync incomplete; no ranking was recorded.",
                    f"Ticker coverage: {sync_status.coverage:.1%}",
                )
                return
            ohlcv = scan_ohlcv().collect()
        except Exception as exc:
            warn_card(f"Sync failed: {exc}")
            return
    if ohlcv is None or not ohlcv.height:
        warn_card("No OHLCV data available.")
        return

    data = ohlcv.unique(subset=["ticker", "dt"], keep="last").sort(["ticker", "dt"])
    ranking, decision_date = _rank_candidates(data)
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

    displayed = ranking.head(top)
    rule("Alpha Screener")
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
    strategy_metrics = evaluate_rankings(matured)
    rule("Alpha Screener — Matured Predictions")
    valid_metrics = [metrics for metrics in strategy_metrics if metrics["days"]]
    if not valid_metrics:
        click.echo("  No prediction dates have matured yet.\n")
        return
    for metrics in valid_metrics:
        interval = (
            f"[{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]"
            if metrics["ci_lower"] is not None
            else "pending at least 20 matured dates"
        )
        lift = (
            f"{metrics['lift_at_k']:.2f}"
            if metrics["lift_at_k"] is not None
            else "n/a"
        )
        panel(f"14-session quality — {metrics['strategy_version']}", [
            f"Decision dates: {metrics['days']} (skipped: {metrics['skipped_days']})",
            f"Precision@10: {metrics['precision_at_k']:.3f}",
            f"Base explosion rate: {metrics['base_explosion_rate']:.3f}",
            f"Lift@10: {lift}",
            f"Mean forward return: {metrics['mean_forward_return']:.2%}",
            f"Mean outcome coverage: {metrics['mean_outcome_coverage']:.1%}",
            f"Bootstrap CI: {interval}",
        ])


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
