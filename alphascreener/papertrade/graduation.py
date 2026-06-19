"""Tier 1 & Tier 2 graduation condition checks for paper trading system.

Issue #102: Paper Trading tracker.
Reference: PRD 7.6.2.1 (simplified for CLI-only mode).

Provides:
  - EngineeringGraduationResult: dataclass for Tier 1 engineering checks.
  - check_engineering_graduation: evaluates 3 Tier 1 conditions.
  - StrategyGraduationResult: dataclass for Tier 2 strategy checks (reserved).
  - check_strategy_graduation: reserved interface, returns NOT_IMPLEMENTED.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import Engine

from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ============================================================================
# Tier 1 thresholds (PRD 7.6.2.1, CLI-adapted)
# ============================================================================

_MIN_DAYS_IN_OPERATION: int = 60
_MAX_L3_L4_EVENTS_30D: int = 0
_NAN_RATE_MAX: float = 0.05  # strict < 5%


# ============================================================================
# EngineeringGraduationResult
# ============================================================================


@dataclass
class EngineeringGraduationResult:
    """Result of Tier 1 engineering graduation condition check.

    Attributes:
        passed: True if all conditions are met.
        days_in_operation: Number of days the system has been running.
        l3_l4_event_count: Count of circuit breaker events in last 30 days.
        nan_rate: Fraction of NaN values in metrics (0.0-1.0).
        failed_checks: List of check names that did NOT pass.
    """

    passed: bool
    days_in_operation: int
    l3_l4_event_count: int
    nan_rate: float
    failed_checks: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable one-line summary of graduation check result."""
        if self.passed:
            return (
                f"Tier 1 Engineering Graduation: PASS "
                f"(days={self.days_in_operation}, l3l4={self.l3_l4_event_count}, "
                f"nan={self.nan_rate:.1%})"
            )
        return (
            f"Tier 1 Engineering Graduation: FAIL "
            f"(days={self.days_in_operation}, l3l4={self.l3_l4_event_count}, "
            f"nan={self.nan_rate:.1%}) "
            f"failed={self.failed_checks}"
        )


# ============================================================================
# check_engineering_graduation
# ============================================================================


def check_engineering_graduation(
    *,
    session_factory: Callable[[], object],
    db_engine: Engine,
    days_in_operation: int,
    l3_l4_events_last_30d: int,
    nan_rate: float,
) -> EngineeringGraduationResult:
    """Evaluate engineering graduation conditions (CLI-adapted).

    Conditions (PRD 7.6.2.1, simplified for CLI-only mode):

    1. **>= 60 days** of continuous operation.
    2. **No circuit breaker** events in the last 30 days.
    3. **NaN rate < 5%** across all metrics.

    Args:
        session_factory: Zero-arg callable returning a SQLAlchemy Session.
        db_engine: SQLAlchemy Engine. Reserved for future use.
        days_in_operation: Number of days since first production run.
        l3_l4_events_last_30d: Count of circuit breaker trips in last 30d.
        nan_rate: Fraction of NaN values in core metrics (0.0-1.0).

    Returns:
        :class:`EngineeringGraduationResult` with pass/fail status and details.
    """
    failed: list[str] = []

    if days_in_operation < _MIN_DAYS_IN_OPERATION:
        failed.append("days_in_operation")

    if l3_l4_events_last_30d > _MAX_L3_L4_EVENTS_30D:
        failed.append("l3_l4_events")

    if nan_rate >= _NAN_RATE_MAX:
        failed.append("nan_rate")

    passed = len(failed) == 0

    result = EngineeringGraduationResult(
        passed=passed,
        days_in_operation=days_in_operation,
        l3_l4_event_count=l3_l4_events_last_30d,
        nan_rate=nan_rate,
        failed_checks=failed,
    )

    _logger.info("Engineering graduation: %s", result.summary)
    return result


# ============================================================================
# StrategyGraduationResult
# ============================================================================


@dataclass
class StrategyGraduationResult:
    """Result of Tier 2 strategy graduation condition check (reserved).

    Tier 2 conditions (PRD 7.6.2.1), not yet implemented:

    1. >= 2 years walk-forward backtest.
    2. >= 6 months live shadow trading.
    3. Lift@20 > 1.10 (LLM track).
    4. LLM Delta-Lift >= 0.05 (LLM track lift - pure track lift).
    5. IC decay < 50% (rolling IC not decayed by more than half from peak).

    Attributes:
        ready: True when Tier 2 conditions are met (always False for now).
        status: Human-readable status code.
        summary: Human-readable one-line summary.
    """

    ready: bool
    status: str
    summary: str = ""


# ============================================================================
# check_strategy_graduation (reserved interface)
# ============================================================================


def check_strategy_graduation(
    *,
    session_factory: Callable[[], object],
    db_engine: Engine,
    **kwargs: object,
) -> StrategyGraduationResult:
    """Evaluate Tier 2 strategy graduation conditions (reserved interface).

    This function accepts all Tier 2 parameters as keyword arguments but
    does not yet evaluate them. It always returns ``NOT_IMPLEMENTED``.

    Reserved parameters (all optional, accepted as **kwargs):
        walk_forward_years: float
        live_shadow_months: float
        lift_at_20: float
        llm_delta_lift: float
        ic_decay: float

    Args:
        session_factory: Zero-arg callable returning a SQLAlchemy Session.
        db_engine: SQLAlchemy Engine.
        **kwargs: Reserved Tier 2 parameters (accepted but not evaluated).

    Returns:
        :class:`StrategyGraduationResult` with ``ready=False`` and
        ``status="NOT_IMPLEMENTED"``.
    """
    result = StrategyGraduationResult(
        ready=False,
        status="NOT_IMPLEMENTED",
        summary="Tier 2 Strategy Graduation: NOT_IMPLEMENTED (reserved interface — "
        "Tier 2 conditions are not yet implemented)",
    )
    _logger.info("Strategy graduation: %s", result.summary)
    return result
