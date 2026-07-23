"""Alpha Screener's 60-session, 14-session prediction CLI."""

from __future__ import annotations

from datetime import date

import click
import polars as pl

from alphascreener.display import panel, result_table, rule, warn_card
from alphascreener.prediction_contract import (
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_TOP_K,
    MAX_BACKTEST_DAYS,
)
from alphascreener.ranking import rank_candidates


def _ledger_outcome_requirements() -> tuple[tuple[str, date, date], ...]:
    """Return ledger symbols and exact result dates still missing locally."""
    from alphascreener.evaluation import read_prediction_ledger
    from alphascreener.market_calendar import future_market_date
    from alphascreener.prediction_contract import (
        FORECAST_HORIZON_SESSIONS,
        STRATEGY_VERSION,
    )

    try:
        ledger = read_prediction_ledger().filter(
            pl.col("strategy_version") == STRATEGY_VERSION
        )
    except (OSError, ValueError):
        return ()
    if ledger.is_empty():
        return ()
    result_dates: dict[date, date] = {}
    for decision_date in ledger["decision_date"].cast(pl.Date).unique().to_list():
        try:
            result_date = future_market_date(
                decision_date,
                FORECAST_HORIZON_SESSIONS,
            )
        except ValueError:
            continue
        result_dates[decision_date] = result_date
    try:
        from alphascreener.data.io import scan_ohlcv

        observations = set(
            scan_ohlcv()
            .select("ticker", "dt")
            .unique()
            .collect()
            .iter_rows()
        )
    except FileNotFoundError:
        observations = set()
    missing: set[tuple[str, date, date]] = set()
    for ticker, decision_date in ledger.select("ticker", "decision_date").iter_rows():
        result_date = result_dates.get(decision_date)
        if result_date is not None and (ticker, result_date) not in observations:
            missing.add((ticker, decision_date, result_date))
    return tuple(sorted(missing))


def _load_synced_ohlcv() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    """Return full, current-directory, and current-ready OHLCV panels."""
    from alphascreener.data.io import scan_ohlcv
    from alphascreener.data.sync import MIN_SYNC_COVERAGE, sync_ohlcv

    click.echo("  Updating data ...")
    try:
        sync_status = sync_ohlcv(
            outcome_requirements=_ledger_outcome_requirements()
        )
        if not sync_status.is_fresh:
            warn_card(
                "Market data is stale; analysis was not produced.",
                f"Latest SPY date: {sync_status.as_of_date or 'unavailable'}",
            )
            return None
        if sync_status.coverage < MIN_SYNC_COVERAGE:
            warn_card(
                "Sync incomplete; analysis was not produced.",
                f"Decision-ready coverage: {sync_status.coverage:.1%}",
            )
            return None
        ohlcv = scan_ohlcv().collect()
        if sync_status.as_of_date is not None:
            ohlcv = ohlcv.filter(pl.col("dt") <= sync_status.as_of_date)
    except Exception as exc:
        warn_card(f"Sync failed: {exc}")
        return None
    if ohlcv.is_empty():
        warn_card("No OHLCV data available.")
        return None
    full = ohlcv.unique(subset=["ticker", "dt"], keep="last").sort(["ticker", "dt"])
    requested = set(sync_status.requested_symbols)
    if not requested:
        requested = set(sync_status.ready_tickers) | set(sync_status.failed_tickers)
    if not requested:
        requested = set(full["ticker"].unique().to_list())
    ready = set(sync_status.ready_tickers) or requested
    current_directory = full.filter(pl.col("ticker").is_in(sorted(requested)))
    current_ready = full.filter(pl.col("ticker").is_in(sorted(ready)))
    return full, current_directory, current_ready


def _record_or_restore_ranking(
    ranking: pl.DataFrame,
    decision_date: object,
) -> tuple[pl.DataFrame, str]:
    """Record a new ranking, or restore the immutable same-day ranking."""
    from alphascreener.evaluation import read_prediction_ledger, write_prediction_ledger
    from alphascreener.prediction_contract import STRATEGY_VERSION

    predictions = ranking.with_columns(
        pl.lit(decision_date).cast(pl.Date).alias("decision_date"),
        pl.lit(STRATEGY_VERSION).alias("strategy_version"),
        pl.lit(ranking.height).cast(pl.Int64).alias("universe_size"),
    )
    try:
        write_prediction_ledger(predictions)
    except FileExistsError:
        try:
            recorded = read_prediction_ledger().filter(
                (pl.col("decision_date") == decision_date)
                & (pl.col("strategy_version") == STRATEGY_VERSION)
            )
            if recorded.is_empty():
                raise ValueError("recorded ranking could not be read")
            restored = recorded.select("ticker", "score", "rank").sort("rank")
            return restored, "using the immutable ranking already recorded for this date"
        except (OSError, ValueError) as exc:
            warn_card(
                "Immutable ranking exists but could not be read.",
                f"Displaying the current unrecorded calculation: {exc}",
            )
            return ranking, "NOT RECORDED in this run"
    except (OSError, ValueError) as exc:
        warn_card(
            "Ranking was not recorded.",
            f"Candidates are still shown, but are not prospective evidence: {exc}",
        )
        return ranking, "NOT RECORDED"
    return ranking, "recorded before the 14-session outcome is available"


def _render_backtest_summary(records: pl.DataFrame, *, requested_days: int) -> None:
    """Render an aggregate diagnostic without printing every historical date."""
    valid = records.filter(pl.col("status") == "VALID")
    valid_days = valid.height
    invalid_days = requested_days - valid_days
    if valid_days:
        hits = int(valid["hits_at_10"].sum())
        precision = hits / (valid_days * 10)
        base_rate = float(valid["base_explosion_rate"].mean())
        passing_days = int(valid["passed"].sum())
        precision_text = f"{precision:.1%} ({hits}/{valid_days * 10})"
        base_text = f"{base_rate:.1%}"
    else:
        passing_days = 0
        precision_text = "n/a"
        base_text = "n/a"
    overall_status = "COMPLETE" if invalid_days == 0 else "INCONCLUSIVE"
    invalid_details = [
        f"INVALID {row['decision_date']}: {row['invalid_reason']}"
        for row in records.filter(pl.col("status") == "INVALID").iter_rows(named=True)
    ]
    panel(
        f"{requested_days}-day current-universe backtest",
        [
            "Evidence type: CURRENT_UNIVERSE_DIAGNOSTIC",
            f"Overall status: {overall_status}",
            f"Valid days: {valid_days}/{requested_days}  |  Invalid: {invalid_days}",
            f"Valid-date pooled Precision@10: {precision_text}  |  Mean base: {base_text}",
            f"Passing days: {passing_days}/{valid_days}",
            *invalid_details,
        ],
    )


def _render_backtest_details(records: pl.DataFrame) -> None:
    """Render every requested walk-forward date, including invalid dates."""

    def percent(value: object, *, digits: int = 1) -> str:
        return "—" if value is None else f"{float(value):.{digits}%}"

    def integer(value: object) -> str:
        return "—" if value is None else str(value)

    click.echo("  Detailed walk-forward dates:")
    for row in records.iter_rows(named=True):
        passed = row["passed"]
        pass_text = "—" if passed is None else ("yes" if passed else "no")
        click.echo(
            f"  strategy={row['strategy_version']}"
            f" | {row['decision_date']} -> {row['result_date']}"
            f" | {row['status']}"
            f" | universe={integer(row['universe_size'])}"
            f" | coverage={percent(row['outcome_coverage'])}"
            f" | hits@10={integer(row['hits_at_10'])}"
            f" | P@10={percent(row['precision_at_10'], digits=0)}"
            f" | base={percent(row['base_explosion_rate'])}"
            f" | pass={pass_text}"
            f" | invalid_reason={row['invalid_reason'] or '—'}"
        )


def _render_matured_evidence(ohlcv: pl.DataFrame) -> None:
    """Evaluate current-strategy ledger entries whose outcomes are available."""
    from alphascreener.evaluation import (
        MIN_OUTCOME_COVERAGE,
        compute_forward_labels,
        evaluate_daily_rankings,
        longest_consecutive_passes,
        mature_predictions,
        read_prediction_ledger,
    )
    from alphascreener.market_calendar import future_market_date, infer_market_dates
    from alphascreener.prediction_contract import (
        FORECAST_HORIZON_SESSIONS,
        STRATEGY_VERSION,
    )

    ledger = read_prediction_ledger()
    current_ledger = ledger.filter(pl.col("strategy_version") == STRATEGY_VERSION)
    if current_ledger.is_empty():
        panel(
            "Prospective evidence",
            "No saved predictions yet; this does not block today's ranking.",
        )
        return

    labels = compute_forward_labels(ohlcv)
    matured = mature_predictions(current_ledger, labels)
    daily = evaluate_daily_rankings(matured).sort("decision_date")
    market_dates = infer_market_dates(ohlcv)
    latest_market_date = market_dates[-1]
    evidence_rows: list[dict[str, object]] = []
    for decision_date, group in current_ledger.group_by(
        "decision_date",
        maintain_order=True,
    ):
        decision_date = decision_date[0]
        result_date = future_market_date(
            decision_date,
            FORECAST_HORIZON_SESSIONS,
        )
        if result_date > latest_market_date:
            continue
        outcome_group = matured.filter(pl.col("decision_date") == decision_date)
        sizes = group["universe_size"].unique().to_list()
        universe_size = int(sizes[0]) if len(sizes) == 1 else 0
        outcome_count = outcome_group["forward_return"].is_not_null().sum()
        coverage = outcome_count / universe_size if universe_size else 0.0
        top_ranks = set(
            outcome_group.filter(
                (pl.col("rank") <= DEFAULT_TOP_K)
                & pl.col("forward_return").is_not_null()
            )["rank"].to_list()
        )
        metric = daily.filter(pl.col("decision_date") == decision_date)
        if universe_size < DEFAULT_TOP_K:
            reason = f"eligible_universe_below_top_{DEFAULT_TOP_K}"
        elif coverage < MIN_OUTCOME_COVERAGE:
            reason = "outcome_coverage_below_90pct"
        elif top_ranks != set(range(1, DEFAULT_TOP_K + 1)):
            reason = f"top_{DEFAULT_TOP_K}_outcomes_incomplete"
        elif coverage < 1.0:
            reason = "complete_universe_outcomes_required"
        elif metric.height != 1:
            reason = "evaluation_failed"
        else:
            row = metric.row(0, named=True)
            evidence_rows.append({
                "decision_date": decision_date,
                "result_date": result_date,
                "status": "VALID",
                "precision": float(row["precision_at_k"]),
                "base_rate": float(row["base_explosion_rate"]),
                "coverage": float(row["outcome_coverage"]),
                "passed": bool(row["passed"]),
                "reason": None,
            })
            continue
        evidence_rows.append({
            "decision_date": decision_date,
            "result_date": result_date,
            "status": "INVALID",
            "precision": None,
            "base_rate": None,
            "coverage": coverage,
            "passed": None,
            "reason": reason,
        })

    if not evidence_rows:
        panel(
            f"Prospective evidence — {STRATEGY_VERSION}",
            "No prediction dates have matured yet; today's ranking is unaffected.",
        )
        return

    rule("Matured prospective predictions")
    result_table(
        ["Decision", "Result", "Status", "P@10", "Base", "Coverage", "Pass / reason"],
        [
            [
                str(row["decision_date"]),
                str(row["result_date"]),
                str(row["status"]),
                "—" if row["precision"] is None else f"{row['precision']:.0%}",
                "—" if row["base_rate"] is None else f"{row['base_rate']:.1%}",
                f"{row['coverage']:.1%}",
                row["reason"]
                if row["reason"] is not None
                else ("yes" if row["passed"] else "no"),
            ]
            for row in evidence_rows[-5:]
        ],
    )
    for row in evidence_rows[-5:]:
        if row["status"] == "INVALID":
            click.echo(
                f"  INVALID {row['decision_date']}: {row['reason']}"
            )
    streak = longest_consecutive_passes(
        daily,
        market_dates,
        strategy_version=STRATEGY_VERSION,
    )
    panel(
        f"14-session target — {STRATEGY_VERSION}",
        [
            f"Best consecutive passing days: {streak}/5",
            "Target reached: yes" if streak >= 5 else "Target reached: no",
        ],
    )


def _run_screen(top: int) -> None:
    from alphascreener.backtest import run_backtest

    synced = _load_synced_ohlcv()
    if synced is None:
        return
    full_data, current_directory_data, current_ready_data = synced

    try:
        ranking, decision_date = rank_candidates(current_ready_data)
    except (OSError, ValueError) as exc:
        warn_card(f"Cannot rank candidates: {exc}")
        return
    if ranking.is_empty():
        warn_card("No tickers meet the 60-session tradable-universe requirements.")
        return
    if ranking.height < DEFAULT_TOP_K:
        warn_card(
            "Eligible universe is too small; no ranking was recorded.",
            f"Need at least {DEFAULT_TOP_K} candidates, found {ranking.height}.",
        )
        return

    ranking, ledger_note = _record_or_restore_ranking(ranking, decision_date)

    try:
        backtest_records = run_backtest(
            current_directory_data,
            days=DEFAULT_BACKTEST_DAYS,
        )
    except Exception as exc:
        backtest_records = None
        warn_card(
            "Historical backtest is unavailable.",
            f"Today's ranking is unaffected: {exc}",
        )

    displayed = ranking.head(top)
    rule("Alpha Screener")
    click.echo(
        f"  Date: {decision_date}  |  Displayed: {displayed.height}"
        f"  |  Eligible universe: {ranking.height}\n"
    )
    result_table(
        ["#", "Ticker", "Score"],
        [
            [str(rank), str(ticker), f"{score:.4f}"]
            for ticker, score, rank in displayed.select(
                "ticker", "score", "rank"
            ).iter_rows()
        ],
    )
    click.echo(f"\n  Ledger: {ledger_note}.\n")

    if backtest_records is not None:
        _render_backtest_summary(
            backtest_records,
            requested_days=DEFAULT_BACKTEST_DAYS,
        )

    try:
        _render_matured_evidence(full_data)
    except Exception as exc:
        warn_card(
            "Prospective evidence is unavailable.",
            f"Today's ranking is unaffected: {exc}",
        )


@click.command(name="backtest")
@click.option(
    "--days",
    default=DEFAULT_BACKTEST_DAYS,
    type=click.IntRange(min=1, max=MAX_BACKTEST_DAYS),
    show_default=True,
    help="Number of matured decision dates to recompute (1-45).",
)
def backtest_command(days: int) -> None:
    """Show detailed current-universe walk-forward diagnostics."""
    from alphascreener.backtest import run_backtest

    synced = _load_synced_ohlcv()
    if synced is None:
        return
    _, current_directory_data, _ = synced
    try:
        records = run_backtest(current_directory_data, days=days)
    except Exception as exc:
        warn_card(f"Cannot run backtest: {exc}")
        return

    rule(f"Alpha Screener — {days}-day backtest")
    click.echo("  Evidence type: CURRENT_UNIVERSE_DIAGNOSTIC\n")
    _render_backtest_details(records)
    click.echo()
    _render_backtest_summary(records, requested_days=days)


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
    """Rank US equities and retain prospective evidence."""
    if ctx.invoked_subcommand is None:
        _run_screen(top)


cli.add_command(backtest_command)


if __name__ == "__main__":
    cli()
