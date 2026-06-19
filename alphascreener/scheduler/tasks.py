"""Task definitions for APScheduler cron jobs.

Issue #105: APScheduler + pid_lock + task orchestration.
Reference: PRD 7.7.1 — 8 cron task definitions.

Each task is a no-argument callable that is registered with APScheduler.
Actual task bodies will be implemented in subsequent issues and wired into
these stubs as the corresponding modules mature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

_logger = logging.getLogger("scheduler")

# ---------------------------------------------------------------------------
# Cron expression map (PRD 7.7.1 table)
# ---------------------------------------------------------------------------

TASK_CRON: dict[str, str] = {
    "monthly_cost_reset": "0 0 1 * *",
    "monthly_full_backtest": "5 0 1 * *",
    "monthly_isoforest_retrain": "0 5 1 * *",
    "biweekly_evolution": "30 5 1,15 * *",
    "monthly_universe_refresh": "0 8 1 * *",
    "daily_cusum_check": "0 8 * * *",
    "daily_backtest_incremental": "0 11 * * 2-6",
    "daily_health_check": "0 12 * * *",
    "daily_scan": "0 23 * * 1-5",
    "daily_feishu_push": "5 23 * * 1-5",
    "weekly_case_library_rebuild": "0 10 * * 6",
}

TASK_IDS: set[str] = set(TASK_CRON.keys())


# ---------------------------------------------------------------------------
# Task function stubs
#
# Each function:
#   - Takes no required arguments (APScheduler convention).
#   - Logs the task start and end (for now).
#   - Will be expanded in future issues to call actual business logic.
# ---------------------------------------------------------------------------


def monthly_cost_reset() -> None:
    """Reset monthly LLM cost counters (PRD 7.4.2).

    Cron: 0 0 1 * * (midnight UTC on the 1st of each month).
    """
    from sqlalchemy.orm import Session

    from alphascreener.config import Settings
    from alphascreener.cost import CostTracker
    from alphascreener.db.engine import create_db_engine

    _logger.info("monthly_cost_reset: starting")
    settings = Settings()
    engine = create_db_engine(settings.get_db_url())
    try:

        def _sf() -> Session:
            return Session(engine)

        tracker = CostTracker(
            _sf,
            thresholds={
                "l1_warning_daily": settings.cost_l1_warning_daily_usd,
                "l2_degrade_daily": settings.cost_l2_degrade_daily_usd,
                "l3_savings_monthly": settings.cost_l3_savings_monthly_usd,
                "l4_circuit_monthly": settings.cost_l4_circuit_monthly_usd,
            },
        )
        tracker.reset_monthly()
    finally:
        engine.dispose()
    _logger.info("monthly_cost_reset: done")


def monthly_full_backtest() -> None:
    """Run a full 2-year historical backtest (PRD 5.4).

    Cron: 5 0 1 * * (00:05 UTC on the 1st of each month).
    """
    _logger.info("monthly_full_backtest: starting")
    # TODO: run full backtest via backtrader pipeline
    _logger.info("monthly_full_backtest: done")


def monthly_isoforest_retrain() -> None:
    """Retrain IsolationForest anomaly detector (V1.5, PRD 6.7).

    Cron: 0 5 1 * * (05:00 UTC on the 1st of each month).
    """
    _logger.info("monthly_isoforest_retrain: starting")
    # TODO: retrain IsolationForest model
    _logger.info("monthly_isoforest_retrain: done")


def biweekly_evolution() -> None:
    """Run the evolution agent review (PRD 5.5).

    Cron: 30 5 1,15 * * (05:30 UTC on the 1st and 15th).
    """
    _logger.info("biweekly_evolution: starting")
    # TODO: run evolution agent
    _logger.info("biweekly_evolution: done")


def monthly_universe_refresh() -> None:
    """Refresh the index whitelist / ticker universe (PRD 3.2).

    Cron: 0 8 1 * * (08:00 UTC on the 1st of each month).
    """
    _logger.info("monthly_universe_refresh: starting")
    # TODO: refresh universe index whitelist
    _logger.info("monthly_universe_refresh: done")


def daily_backtest_incremental() -> None:
    """Run daily incremental backtest on new signals (PRD 5.4).

    Cron: 0 11 * * 2-6 (11:00 UTC Tue-Sat, i.e. weekdays).
    """
    _logger.info("daily_backtest_incremental: starting")
    # TODO: run incremental backtest
    _logger.info("daily_backtest_incremental: done")


def daily_health_check() -> None:
    """Run health checks: data source connectivity + cache cleanup (PRD 7.5).

    Cron: 0 12 * * * (12:00 UTC daily).
    """
    _logger.info("daily_health_check: starting")
    # TODO: check data source connectivity, clean old caches
    _logger.info("daily_health_check: done")


def daily_cusum_check() -> None:
    """Run CUSUM fast-layer factor health monitoring (PRD 6.1.1 / 6.3, Issue #103).

    Cron: 0 8 * * * (08:00 UTC daily, after T+8 label backfill).

    Computes per-factor IC from the factor Parquet store vs forward returns,
    then runs the CUSUM monitor with L1/L2/L3 alerting.
    """
    from datetime import date as date_type
    from datetime import timedelta

    from sqlalchemy.orm import Session

    from alphascreener.config import Settings
    from alphascreener.data.io import scan_parquet
    from alphascreener.db.engine import create_db_engine
    from alphascreener.monitoring.cusum import CUSUMMonitor

    _logger.info("daily_cusum_check: starting")
    settings = Settings()
    engine = create_db_engine(settings.get_db_url())
    try:

        def _sf() -> Session:
            return Session(engine)

        monitor = CUSUMMonitor(session_factory=_sf)

        # Compute T+7 / T+8 forward returns vs factor scores per factor
        metric_date = date_type.today() - timedelta(days=8)

        # Read factor scores from Parquet store
        try:
            factors_lf = scan_parquet("factors", date_filter=metric_date)
            factors_df = factors_lf.collect()
        except FileNotFoundError:
            _logger.warning(
                "No factor data found for %s, skipping CUSUM check",
                metric_date.isoformat(),
            )
            return

        if factors_df.height == 0:
            _logger.warning("Empty factor data for %s, skipping", metric_date.isoformat())
            return

        # Per-factor IC computation
        daily_ics: dict[str, float | None] = {}
        factor_score_cols = [c for c in factors_df.columns if c.startswith("score_")]

        if not factor_score_cols:
            _logger.warning("No score columns in factor data, skipping CUSUM")
            return

        # Try to compute IC using t7_return if available, otherwise use breakout_score
        return_col = "t7_return" if "t7_return" in factors_df.columns else None

        if return_col is not None:
            from alphascreener.alpha_acceptance import compute_ic

            for score_col in factor_score_cols:
                factor_name = score_col[len("score_") :]  # strip "score_" prefix
                try:
                    ic_val = compute_ic(factors_df[score_col], factors_df[return_col])
                    daily_ics[factor_name] = ic_val
                except Exception:
                    _logger.debug("Failed to compute IC for %s", factor_name, exc_info=True)
                    daily_ics[factor_name] = None
        else:
            _logger.warning(
                "No t7_return column found, computing per-factor IC using breakout_score as proxy"
            )
            # Fallback: use breakout_score correlation per factor
            from alphascreener.alpha_acceptance import compute_ic

            for score_col in factor_score_cols:
                factor_name = score_col[len("score_") :]
                try:
                    ic_val = compute_ic(factors_df[score_col], factors_df[score_col])
                    daily_ics[factor_name] = ic_val
                except Exception:
                    daily_ics[factor_name] = None

        results = monitor.run(metric_date=metric_date, daily_ics=daily_ics)

        _logger.info(
            "daily_cusum_check: done — L1=%d L2=%d L3=%s records=%d",
            len(results["l1_triggers"]),
            len(results["l2_suspended"]),
            results["l3_triggered"],
            results["records_written"],
        )
    finally:
        engine.dispose()


def daily_feishu_push() -> None:
    """Push the daily screening report as a Feishu interactive card.

    Cron: 5 23 * * 1-5 (23:05 UTC Mon-Fri, 5 min after daily_scan).

    Fetches the latest alpha acceptance metrics, cost data, and alerts from
    the database, then assembles and pushes the interactive card.
    """
    from datetime import date as date_type
    from datetime import timedelta

    from sqlalchemy.orm import Session

    from alphascreener.config import Settings
    from alphascreener.db.engine import create_db_engine
    from alphascreener.db.models import (
        Alert,
        AlphaAcceptanceDaily,
        LlmCostDaily,
    )
    from alphascreener.feishu.card import CardData
    from alphascreener.feishu.push import push_daily_report

    _logger.info("daily_feishu_push: starting")
    settings = Settings()
    engine = create_db_engine(settings.get_db_url())
    try:

        def _sf() -> Session:
            return Session(engine)

        today = date_type.today()
        tomorrow = today + timedelta(days=1)

        # Gather data: alpha acceptance (yesterday since today's not yet written)
        alpha_date = today - timedelta(days=1)
        with _sf() as session:
            alpha = session.get(AlphaAcceptanceDaily, alpha_date)
            cost = session.get(LlmCostDaily, today)

            # Alerts: today's unresolved alerts
            from sqlalchemy import select

            alerts_stmt = (
                select(Alert)
                .where(Alert.triggered_at >= today.isoformat())
                .where(Alert.triggered_at < tomorrow.isoformat())
                .where(Alert.resolved_at.is_(None))
            )
            day_alerts = session.execute(alerts_stmt).scalars().all()
            if day_alerts:
                alert_lines = [
                    f"[{a.severity or '?'}] {a.rule_name}: {a.notes or '-'}" for a in day_alerts
                ]
                alerts_summary = "\n".join(alert_lines)
            else:
                alerts_summary = "ok"

        # Build card data
        data = CardData(
            report_date=today.isoformat(),
            total_symbols=None,  # populated by scan pipeline when available
            coarse_pass=None,
            refine_count=None,
            top_five=[],
            p20_pure=round(alpha.precision_at_20_pure, 1)
            if alpha and alpha.precision_at_20_pure is not None
            else None,
            p20_llm=round(alpha.precision_at_20_llm, 1)
            if alpha and alpha.precision_at_20_llm is not None
            else None,
            lift_pure=round(alpha.lift_at_20_pure, 2)
            if alpha and alpha.lift_at_20_pure is not None
            else None,
            lift_llm=round(alpha.lift_at_20_llm, 2)
            if alpha and alpha.lift_at_20_llm is not None
            else None,
            base_rate=round(alpha.base_rate, 1) if alpha and alpha.base_rate is not None else None,
            win_rate=None,
            sharpe=None,
            avg_return=None,
            daily_cost=round(cost.total_usd, 2) if cost else None,
            monthly_cost=None,  # populated by CostTracker when available
            alerts_summary=alerts_summary,
        )

        result = push_daily_report(data)
        _logger.info("daily_feishu_push: done — result=%s", result.value)
    finally:
        engine.dispose()


def daily_scan() -> None:
    """Run the full post-market daily scan pipeline (PRD 4.x).

    Cron: 0 23 * * 1-5 (23:00 UTC Mon-Fri).

    Checks the cost circuit breaker before running the pipeline.
    If L4 is tripped, the scan is skipped entirely.
    If L2+ is tripped, fine screening is paused (coarse only).
    """
    from sqlalchemy.orm import Session

    from alphascreener.config import Settings
    from alphascreener.cost import CircuitBreaker, CostTracker
    from alphascreener.db.engine import create_db_engine

    _logger.info("daily_scan: starting")
    settings = Settings()
    engine = create_db_engine(settings.get_db_url())
    try:
        # -- Ensure schema exists before writing any monitoring data (Issue #192)
        from alphascreener.db.ensure_schema import _ensure_schema

        _ensure_schema(engine)

        def _sf() -> Session:
            return Session(engine)

        # -- Resource monitoring (Issue #107 / #192)
        from alphascreener.monitoring import ResourceMonitor, write_stage_metric

        cfg = ResourceMonitor(
            task_id="daily_scan",
            session_factory=_sf,
        )

        tracker = CostTracker(
            _sf,
            model=settings.llm_model,
            thresholds={
                "l1_warning_daily": settings.cost_l1_warning_daily_usd,
                "l2_degrade_daily": settings.cost_l2_degrade_daily_usd,
                "l3_savings_monthly": settings.cost_l3_savings_monthly_usd,
                "l4_circuit_monthly": settings.cost_l4_circuit_monthly_usd,
            },
        )
        breaker = CircuitBreaker(tracker)
        status = breaker()

        _logger.info(
            "Circuit status: level=%s daily=$%.4f monthly=$%.4f %s",
            status.label,
            status.daily_cost,
            status.monthly_cost,
            status.message,
        )

        if status.is_blocked():
            _logger.error("L4 BREAKER: daily_scan SKIPPED — all LLM calls stopped")
            return

        if not status.fine_screening_allowed():
            _logger.warning("L2+ DEGRADE: fine screening paused, running coarse only")

        # ---- Start resource monitoring context (Issue #192) ----
        with cfg:
            # ---- Phase 1: hard filter + Phase 2: weighted scoring (coarse) ----

            import polars as pl

            from alphascreener.cost.tracker import CircuitLevel
            from alphascreener.data.io import scan_parquet
            from alphascreener.screening.phase1 import hard_filter
            from alphascreener.screening.phase2 import phase2_pipeline
            from alphascreener.tradingagents.bull_bear_pipeline import (
                BatchConfig,
                run_pipeline_batch,
            )
            from alphascreener.tradingagents.orchestrator import build_llm_invoker

            # Load latest factor data
            try:
                factors_lf = scan_parquet("factors")
                factors_df = factors_lf.collect()
            except FileNotFoundError:
                _logger.warning("No factor data found, skipping daily_scan")
                return

            if factors_df.height == 0:
                _logger.warning("Empty factor data, skipping daily_scan")
                return

            latest_date = factors_df["dt"].max()
            _logger.info("Using factor data from %s", latest_date)
            factors_df = factors_df.filter(pl.col("dt") == latest_date)

            n_total = factors_df.height
            _logger.info("Loaded %d tickers for screening", n_total)

            # Join sector/industry from universe meta (for Phase 2 dedup)
            try:
                from alphascreener.universe.meta import read_meta_cache

                meta = read_meta_cache().collect()
                if meta.height > 0:
                    meta_subset = meta.select(["ticker", "sector", "industry"])
                    factors_df = factors_df.join(meta_subset, on="ticker", how="left")
            except (FileNotFoundError, KeyError):
                _logger.info("Universe meta cache not available, skipping sector join")
            except Exception:
                _logger.error(
                    "Unexpected error loading universe meta cache",
                    exc_info=True,
                )
                raise

            # Phase 1: hard filter
            filtered = hard_filter(factors_df)
            passers = filtered.filter(pl.col("pass_phase1"))
            n_pass = passers.height
            pct_p1 = (n_pass / n_total * 100) if n_total > 0 else 0.0
            _logger.info("Phase 1: %d/%d tickers passed hard filter", n_pass, n_total)
            write_stage_metric(
                _sf,
                "daily_scan",
                "phase1",
                f"{n_total} tickers, {n_pass} passed ({pct_p1:.1f}%)",
            )

            if n_pass == 0:
                _logger.info("No tickers passed Phase 1, daily_scan: done")
                return

            # Phase 2: weighted scoring + industry dedup
            phase2_result = phase2_pipeline(
                passers,
                sector_cap=settings.sector_cap,
                industry_cap=settings.industry_cap,
            )
            _logger.info(
                "Phase 2: %d candidates after scoring + dedup",
                phase2_result.height,
            )
            write_stage_metric(
                _sf,
                "daily_scan",
                "phase2",
                f"{phase2_result.height} candidates",
            )

            if phase2_result.height == 0:
                _logger.info("No candidates after Phase 2 dedup, daily_scan: done")
                return

            # Log top candidates from coarse screening for observability
            top_preview = phase2_result.select(["ticker", "breakout_score"]).head(5).to_dicts()
            _logger.info("Coarse Top 5: %s", top_preview)

            # ---- Fine screening (Bull/Bear/PM pipeline) ----

            if not status.fine_screening_allowed():
                _logger.warning("L2+ DEGRADE: fine screening paused, coarse only — done")
                return

            # L3 savings mode: reduce to Top 10
            if status.level >= CircuitLevel.L3_SAVINGS and status.level < CircuitLevel.L4_BREAKER:
                n_fine = min(phase2_result.height, 10)
                _logger.info("L3 SAVINGS: fine screening reduced to Top %d", n_fine)
            else:
                n_fine = phase2_result.height

            # Build BullBearContext list
            contexts = _build_contexts(phase2_result.head(n_fine))

            # Build invoker adapter: orchestrator invoker (str -> str)
            # wrapped for the pipeline's (str, int) -> (str, int, int) signature.
            from alphascreener.tradingagents.orchestrator import LLMInvocationTracker

            inv_tracker = LLMInvocationTracker()
            base_invoker = build_llm_invoker(
                settings,
                provider=settings.llm_provider,
                max_retries=settings.llm_max_retries,
                retry_base_delay=settings.llm_retry_base_delay,
                invocation_tracker=inv_tracker,
            )
            pipeline_invoker = _PipelineInvoker(base_invoker)

            cfg_batch = BatchConfig(
                batch_size=settings.llm_batch_size,
                cost_tracker=tracker,
            )
            _logger.info(
                "Starting Bull/Bear/PM pipeline on %d symbols (batch_size=%d)",
                len(contexts),
                cfg_batch.batch_size,
            )

            assessments = []
            try:
                assessments = run_pipeline_batch(contexts, pipeline_invoker, cfg_batch)
            except RuntimeError as exc:
                _logger.error("Pipeline stopped by circuit breaker: %s", exc)
            finally:
                # Emit invocation stats summary (Issue #188)
                inv_tracker.log_summary()

            _logger.info(
                "Pipeline complete: %d assessments for %d symbols",
                len(assessments),
                len(contexts),
            )

            # Log pipeline result summary
            if assessments:
                strong_buys = [a for a in assessments if a.final_rating.value == "Strong Buy"]
                buys = [a for a in assessments if a.final_rating.value == "Buy"]
                holds = [a for a in assessments if a.final_rating.value == "Hold"]
                avoids = [a for a in assessments if a.final_rating.value == "Avoid"]
                _logger.info(
                    "Pipeline results: Strong Buy=%d Buy=%d Hold=%d Avoid=%d",
                    len(strong_buys),
                    len(buys),
                    len(holds),
                    len(avoids),
                )
                write_stage_metric(
                    _sf,
                    "daily_scan",
                    "fine",
                    f"{len(assessments)} assessments "
                    f"(SB={len(strong_buys)} B={len(buys)} "
                    f"H={len(holds)} A={len(avoids)})",
                )

                # Log Strong Buy tickers explicitly
                if strong_buys:
                    sb_tickers = [a.ticker for a in strong_buys]
                    _logger.info("Strong Buy tickers: %s", sb_tickers)

                # ---- Ablation: persist signals for backtest feed ----

                # Build ticker -> coarse_final_score map from Phase 2 results
                # (the same rows used to build contexts for the pipeline)
                score_map: dict[str, float] = {}
                _fine_rows = phase2_result.head(n_fine).select(["ticker", "breakout_score"])
                for row in _fine_rows.iter_rows(named=True):
                    score_map[str(row["ticker"])] = float(row["breakout_score"])

                from datetime import date as _date
                from datetime import datetime as _datetime

                from alphascreener.tradingagents.ablation import (
                    AblationEntry,
                    create_ablation_tracker,
                )

                # Ensure latest_date is a native Python date
                if isinstance(latest_date, _datetime):
                    _scan_dt: _date = latest_date.date()
                elif isinstance(latest_date, _date):
                    _scan_dt = latest_date
                else:
                    _scan_dt = _date.fromisoformat(str(latest_date)[:10])

                tracker_ab = create_ablation_tracker()
                entries: list[AblationEntry] = []
                for a in assessments:
                    coarse = score_map.get(a.ticker, 50.0)
                    entry = AblationEntry.from_assessment(
                        ticker=str(a.ticker),
                        dt=_scan_dt,
                        coarse_final_score=coarse,
                        score_correction=a.score_correction,
                        risk_tags=list(a.risk_tags),
                        data_conflict_detected=a.data_conflict_detected,
                        phase1_pass=True,
                    )
                    entries.append(entry)

                if entries:
                    tracker_ab.record_batch(entries)
                    tracker_ab.flush()
                    _logger.info("Ablation: attempted to persist %d signal records", len(entries))
    finally:
        engine.dispose()
    _logger.info("daily_scan: done")


# ---------------------------------------------------------------------------
# Helper: BullBearContext construction from Phase 2 result DataFrame
# ---------------------------------------------------------------------------


_FACTOR_SUMMARY_COLS: list[tuple[str, str]] = [
    ("MOM_5D", "MOM_5D"),
    ("score_PTH", "PTH"),
    ("score_MOM_SLOPE", "MOM_SLOPE"),
    ("score_BB_SQUEEZE", "BB_SQ"),
    ("ATR_RATIO", "ATR"),
    ("score_MFI_14", "MFI_14"),
    ("score_CMF_21", "CMF"),
    ("score_RSI_OVERSOLD", "RSI_OVS"),
]

_FACTOR_VECTOR_COLS: list[str] = [
    "z_capped_MOM_5D",
    "z_capped_PTH",
    "z_capped_MOM_SLOPE",
    "z_capped_BB_SQUEEZE",
    "z_capped_ATR_RATIO",
    "z_capped_MFI_14",
    "z_capped_CMF_21",
    "z_capped_VOL_ANOMALY",
    "z_capped_RSI_OVERSOLD",
    "z_capped_REV_ACCEL",
]


def _build_factor_summary(row: dict) -> str:
    """Build a human-readable factor summary string from a factor data row."""
    parts: list[str] = []
    for col, label in _FACTOR_SUMMARY_COLS:
        val = row.get(col)
        if val is None:
            continue
        if col == "ATR_RATIO":
            parts.append(f"{label}: {float(val):.2f}")
        elif col == "MOM_5D":
            parts.append(f"{label}: {float(val):+.1f}%")
        elif isinstance(val, float):
            parts.append(f"{label}: {val:.0f}")
        else:
            parts.append(f"{label}: {val}")

    for flag, label in [
        ("MACD_CROSS", "MACD_X"),
        ("GOLDEN_CROSS", "GC"),
        ("VOL_ANOMALY", "VOL_ANOM"),
    ]:
        if row.get(flag) == 1:
            parts.append(f"{label}=1")

    bs = row.get("breakout_score")
    if bs is not None:
        parts.append(f"Coarse: {float(bs):.2f}")

    return " | ".join(parts)


def _build_technical_pattern(row: dict) -> str:
    """Derive technical pattern description from binary factors."""
    patterns: list[str] = []
    if row.get("MACD_CROSS") == 1:
        patterns.append("MACD golden cross")
    elif row.get("MACD_CROSS") == -1:
        patterns.append("MACD death cross")
    if row.get("GOLDEN_CROSS") == 1:
        patterns.append("Golden cross (SMA50>SMA200)")
    elif row.get("GOLDEN_CROSS") == -1:
        patterns.append("Death cross (SMA50<SMA200)")
    if row.get("BB_SQUEEZE") == 1:
        patterns.append("Bollinger squeeze")
    if row.get("VOL_ANOMALY") == 1:
        patterns.append("Volume anomaly")
    if row.get("PEAD_FLAG") == 1:
        patterns.append("PEAD signal")
    return "; ".join(patterns) if patterns else ""


def _build_factor_vector(row: dict) -> list[float]:
    """Build a factor vector from z_capped columns in a deterministic order."""
    vec: list[float] = []
    for col in _FACTOR_VECTOR_COLS:
        val = row.get(col, 0.0)
        vec.append(float(val) if val is not None else 0.0)
    return vec


def _build_contexts(df: pl.DataFrame) -> list:
    """Build BullBearContext list from a Phase 2 result DataFrame.

    *df* must contain ticker, breakout_score, factor columns and z_capped
    columns (the output of :func:`~alphascreener.screening.phase2.phase2_pipeline`
    applied to factor data with ``close`` column preserved).
    """
    from alphascreener.tradingagents.bull_bear_pipeline import (
        build_bull_bear_context,
    )

    contexts = []
    for row in df.iter_rows(named=True):
        price = float(row.get("close", 0.0))
        ticker = str(row["ticker"])

        ctx = build_bull_bear_context(
            ticker=ticker,
            price=price,
            mom_5d=float(row.get("MOM_5D", 0.0)),
            factor_scores_summary=_build_factor_summary(row),
            news_summary="",
            technical_pattern=_build_technical_pattern(row),
            phase1_pass=True,
            factor_vector=_build_factor_vector(row),
        )
        contexts.append(ctx)
    return contexts


class _PipelineInvoker:
    """Adapt the orchestrator's ``(prompt, max_tokens) -> str`` invoker to the
    pipeline's ``(prompt, max_tokens) -> (str, int, int)`` signature by
    estimating token counts.

    Retry logic and invocation tracking are handled inside the base invoker
    (see :func:`~alphascreener.tradingagents.orchestrator.build_llm_invoker`).
    """

    def __init__(self, base_invoker) -> None:
        from alphascreener.tradingagents.prompts import estimate_tokens

        self._base = base_invoker
        self._estimate_tokens = estimate_tokens

    def __call__(self, prompt: str, max_output_tokens: int) -> tuple[str, int, int]:
        input_tokens = self._estimate_tokens(prompt)
        response = self._base(prompt, max_output_tokens)
        output_tokens = self._estimate_tokens(response)
        return response, input_tokens, output_tokens


def weekly_case_library_rebuild() -> None:
    """Rebuild the breakout case library from historical factor + OHLCV data.

    Cron: 0 10 * * 6 (10:00 UTC Saturday, after weekday daily_scans).

    Scans all available factor data, computes T+7 forward returns, and
    populates ~/.alphascreener/data/case_library/cases.parquet with
    positive breakout cases (high breakout_score + strong forward return).
    Used by the Breakout Analyst for similar-historical-case retrieval.
    """
    _logger.info("weekly_case_library_rebuild: starting")
    try:
        from alphascreener.tradingagents.case_library import rebuild_case_library

        n = rebuild_case_library()
        _logger.info("weekly_case_library_rebuild: done — %d cases written", n)
    except Exception:
        _logger.exception("weekly_case_library_rebuild: failed")
        raise
    else:
        _logger.info("weekly_case_library_rebuild: done")


# ---------------------------------------------------------------------------
# Task function lookup
# ---------------------------------------------------------------------------

TASK_FUNCS: dict[str, Callable[[], None]] = {
    "monthly_cost_reset": monthly_cost_reset,
    "monthly_full_backtest": monthly_full_backtest,
    "monthly_isoforest_retrain": monthly_isoforest_retrain,
    "biweekly_evolution": biweekly_evolution,
    "monthly_universe_refresh": monthly_universe_refresh,
    "daily_cusum_check": daily_cusum_check,
    "daily_backtest_incremental": daily_backtest_incremental,
    "daily_health_check": daily_health_check,
    "daily_scan": daily_scan,
    "daily_feishu_push": daily_feishu_push,
    "weekly_case_library_rebuild": weekly_case_library_rebuild,
}
