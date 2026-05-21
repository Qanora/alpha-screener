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
    "daily_backtest_incremental": "0 11 * * 2-6",
    "daily_health_check": "0 12 * * *",
    "daily_scan": "0 23 * * 1-5",
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

        def _sf() -> Session:
            return Session(engine)

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

        # TODO: Phase 1 hard filter + Phase 2 weighted scoring (coarse)
        # TODO: If fine screening allowed, run Bull/Bear/PM pipeline
        #       with cost_tracker=tracker passed to BatchConfig / run_pipeline_batch
    finally:
        engine.dispose()
    _logger.info("daily_scan: done")


# ---------------------------------------------------------------------------
# Task function lookup
# ---------------------------------------------------------------------------

TASK_FUNCS: dict[str, Callable[[], None]] = {
    "monthly_cost_reset": monthly_cost_reset,
    "monthly_full_backtest": monthly_full_backtest,
    "monthly_isoforest_retrain": monthly_isoforest_retrain,
    "biweekly_evolution": biweekly_evolution,
    "monthly_universe_refresh": monthly_universe_refresh,
    "daily_backtest_incremental": daily_backtest_incremental,
    "daily_health_check": daily_health_check,
    "daily_scan": daily_scan,
}
