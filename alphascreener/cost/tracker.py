"""LLM cost tracker with 4-level circuit breaker.

Issue #108: Cost tracking & circuit breaker.
Reference: PRD 4.6.2 / 4.6.3.

Architecture:
  - CostTracker: reads/writes ``llm_cost_daily``, accumulates per-call costs.
  - CircuitBreaker: evaluates 4-level thresholds and returns CircuitStatus.
  - MODEL_PRICING: per-model USD-per-token pricing table.

Typical usage::

    from alphascreener.cost import CostTracker, CircuitBreaker
    from alphascreener.config import Settings

    settings = Settings()
    tracker = CostTracker(session_factory, thresholds={
        "l1_warning_daily": settings.cost_l1_warning_daily_usd,
        "l2_degrade_daily": settings.cost_l2_degrade_daily_usd,
        "l3_savings_monthly": settings.cost_l3_savings_monthly_usd,
        "l4_circuit_monthly": settings.cost_l4_circuit_monthly_usd,
    })

    # After each LLM call:
    tracker.record_call("refining", input_tokens=500, output_tokens=200, model="gpt-4o-mini")

    # Check before next batch:
    status = tracker.check_circuit()
    if status.level == CircuitLevel.L4_BREAKER:
        raise RuntimeError("Cost circuit breaker tripped")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import IntEnum

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alphascreener.db.models import LlmCostDaily
from alphascreener.logging import get_logger

_logger: logging.Logger = get_logger("screening")

# ============================================================================
# Model pricing (USD per token) — PRD 4.6.2
# ============================================================================

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":  {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
    "gpt-4o":       {"input": 2.500 / 1_000_000, "output": 10.000 / 1_000_000},
    "gpt-4.1":      {"input": 2.000 / 1_000_000, "output":  8.000 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.400 / 1_000_000, "output":  1.600 / 1_000_000},
}

MODEL_PRICING: dict[str, dict[str, float]] = _MODEL_PRICING

# ============================================================================
# Circuit levels
# ============================================================================


class CircuitLevel(IntEnum):
    """Escalation levels for cost circuit breaker (PRD 4.6.3)."""

    NORMAL = 0
    L1_WARNING = 1   # daily >= $0.80 -> batch size 3→2
    L2_DEGRADE = 2   # daily >= $1.00 -> pause fine screening, only coarse
    L3_SAVINGS = 3   # monthly >= $80  -> fine screening reduced to Top 10
    L4_BREAKER = 4   # monthly >= $95  -> completely stop LLM calls


_LEVEL_LABELS: dict[CircuitLevel, str] = {
    CircuitLevel.NORMAL: "NORMAL",
    CircuitLevel.L1_WARNING: "L1_WARNING",
    CircuitLevel.L2_DEGRADE: "L2_DEGRADE",
    CircuitLevel.L3_SAVINGS: "L3_SAVINGS",
    CircuitLevel.L4_BREAKER: "L4_BREAKER",
}


# ============================================================================
# Circuit status
# ============================================================================


@dataclass
class CircuitStatus:
    """Result of a circuit breaker check."""

    level: CircuitLevel
    daily_cost: float
    monthly_cost: float
    message: str = ""

    def is_blocked(self) -> bool:
        """True when all LLM calls must stop (L4)."""
        return self.level >= CircuitLevel.L4_BREAKER

    def fine_screening_allowed(self) -> bool:
        """True when fine screening (Bull/Bear/PM) is permitted."""
        return self.level < CircuitLevel.L2_DEGRADE

    @property
    def label(self) -> str:
        return _LEVEL_LABELS.get(self.level, "UNKNOWN")


# ============================================================================
# Default thresholds (overridable via config / Settings)
# ============================================================================

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "l1_warning_daily":    0.80,
    "l2_degrade_daily":    1.00,
    "l3_savings_monthly":  80.0,
    "l4_circuit_monthly":  95.0,
}


# ============================================================================
# CostTracker
# ============================================================================


class CostTracker:
    """Persist and query daily LLM cost records.

    Each day a row is upserted into ``llm_cost_daily`` with aggregated
    ``total_usd``, ``call_count``, and per-module breakdown JSON.

    Args:
        session_factory: Zero-arg callable returning a new SQLAlchemy ``Session``.
        thresholds: Optional dict overriding default circuit thresholds.
            Keys: ``l1_warning_daily``, ``l2_degrade_daily``,
            ``l3_savings_monthly``, ``l4_circuit_monthly``.
        model: Default LLM model name used for pricing lookups when
            :meth:`record_call` is called without an explicit *model*.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        thresholds: dict[str, float] | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._sf = session_factory
        self._model = model
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)

    # ------------------------------------------------------------------
    # Pricing helpers (public for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def calc_call_cost(
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o-mini",
    ) -> float:
        """Calculate USD cost for a single LLM call.

        Args:
            input_tokens: Number of prompt/input tokens consumed.
            output_tokens: Number of completion/output tokens generated.
            model: Model identifier.  Falls back to ``"gpt-4o-mini"`` pricing
                when the model is not in :data:`MODEL_PRICING`.

        Returns:
            Cost in USD (float).
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    # ------------------------------------------------------------------
    # Read daily cost
    # ------------------------------------------------------------------

    def get_daily_cost(self, cost_date: date | None = None) -> tuple[float, int]:
        """Return ``(total_usd, call_count)`` for *cost_date*.

        Args:
            cost_date: Date to query.  Defaults to today (UTC).

        Returns:
            ``(total_usd, call_count)`` — ``(0.0, 0)`` if no row exists.
        """
        if cost_date is None:
            cost_date = date.today()

        with self._sf() as session:
            row = session.get(LlmCostDaily, cost_date)
            if row is None:
                return (0.0, 0)
            return (row.total_usd, row.call_count)

    def get_monthly_cost(self, ref_date: date | None = None) -> float:
        """Sum ``total_usd`` for all days in the current calendar month.

        Args:
            ref_date: Reference date for the month.  Defaults to today (UTC).

        Returns:
            Total USD cost for the month.  ``0.0`` if no rows exist.
        """
        if ref_date is None:
            ref_date = date.today()
        month_start = ref_date.replace(day=1)

        with self._sf() as session:
            stmt = select(func.coalesce(func.sum(LlmCostDaily.total_usd), 0.0)).where(
                LlmCostDaily.cost_date >= month_start,
            )
            return session.scalar(stmt) or 0.0

    # ------------------------------------------------------------------
    # Write / accumulate cost
    # ------------------------------------------------------------------

    def record_call(
        self,
        module: str,
        input_tokens: int,
        output_tokens: int = 0,
        *,
        model: str | None = None,
        cost_date: date | None = None,
    ) -> float:
        """Record a single LLM call and upsert the daily aggregate row.

        Call this after every successful LLM invocation.

        Args:
            module: Logical module name (e.g. ``"bull"``, ``"bear"``,
                ``"pm"``, ``"refining"``).  Used for ``by_module_json``
                breakdown.
            input_tokens: Prompt/input tokens consumed.
            output_tokens: Completion/output tokens generated.
            model: Model name for pricing.  Uses the tracker default if None.
            cost_date: Date for the cost row.  Defaults to today (UTC).

        Returns:
            The USD cost of this individual call.
        """
        _model = model or self._model
        call_cost = self.calc_call_cost(input_tokens, output_tokens, model=_model)

        if cost_date is None:
            cost_date = date.today()

        with self._sf() as session:
            # Fetch or create daily row
            row = session.get(LlmCostDaily, cost_date)
            if row is None:
                row = LlmCostDaily(
                    cost_date=cost_date,
                    total_usd=0.0,
                    call_count=0,
                    by_module_json="{}",
                )
                session.add(row)
                # Flush to get a PK in case two concurrent writers collide
                session.flush()

            # Update totals
            row.total_usd = round(row.total_usd + call_cost, 6)
            row.call_count += 1

            # Update per-module breakdown
            breakdown: dict[str, float] = {}
            if row.by_module_json:
                try:
                    breakdown = json.loads(row.by_module_json)
                except (json.JSONDecodeError, TypeError):
                    breakdown = {}
            breakdown[module] = round(breakdown.get(module, 0.0) + call_cost, 6)
            row.by_module_json = json.dumps(breakdown, ensure_ascii=False)

            session.commit()

        _logger.debug(
            "Cost recorded: module=%s tokens_in=%d tokens_out=%d cost=$%.6f",
            module,
            input_tokens,
            output_tokens,
            call_cost,
        )
        return call_cost

    # ------------------------------------------------------------------
    # Monthly reset (called on 1st of each month)
    # ------------------------------------------------------------------

    def reset_monthly(self) -> None:
        """Log monthly total and optionally archive old rows.

        This does **not** delete rows — they are permanent for audit.
        The monthly counter naturally resets because :meth:`get_monthly_cost`
        queries from the 1st of the current month.

        Logging the monthly total allows operators to verify billing.
        """
        monthly = self.get_monthly_cost()
        _logger.info(
            "Monthly cost reset: previous month total=$%.4f. "
            "New month accumulation starts from $0.00.",
            monthly,
        )

    # ------------------------------------------------------------------
    # Circuit breaker check
    # ------------------------------------------------------------------

    def check_circuit(self) -> CircuitStatus:
        """Evaluate all 4 circuit levels and return the highest triggered.

        Returns:
            :class:`CircuitStatus` with the highest active level.
        """
        daily_cost, _ = self.get_daily_cost()
        monthly_cost = self.get_monthly_cost()

        level = CircuitLevel.NORMAL
        messages: list[str] = []

        # L4: monthly >= threshold → complete stop (checked first, highest priority)
        if monthly_cost >= self._thresholds["l4_circuit_monthly"]:
            level = CircuitLevel.L4_BREAKER
            messages.append(
                f"L4 BREAKER: monthly cost ${monthly_cost:.2f} >= "
                f"${self._thresholds['l4_circuit_monthly']:.2f} — "
                f"ALL LLM calls stopped"
            )

        # L3: monthly >= threshold → reduce fine screening to Top 10
        if (
            level < CircuitLevel.L4_BREAKER
            and monthly_cost >= self._thresholds["l3_savings_monthly"]
        ):
            level = CircuitLevel.L3_SAVINGS
            messages.append(
                f"L3 SAVINGS: monthly cost ${monthly_cost:.2f} >= "
                f"${self._thresholds['l3_savings_monthly']:.2f} — "
                f"fine screening reduced to Top 10"
            )

        # L2: daily >= threshold → pause fine screening, only coarse
        if level < CircuitLevel.L3_SAVINGS and daily_cost >= self._thresholds["l2_degrade_daily"]:
            level = CircuitLevel.L2_DEGRADE
            messages.append(
                f"L2 DEGRADE: daily cost ${daily_cost:.2f} >= "
                f"${self._thresholds['l2_degrade_daily']:.2f} — "
                f"fine screening paused, coarse only"
            )

        # L1: daily >= threshold → batch size 3→2
        if level < CircuitLevel.L2_DEGRADE and daily_cost >= self._thresholds["l1_warning_daily"]:
            level = CircuitLevel.L1_WARNING
            messages.append(
                f"L1 WARNING: daily cost ${daily_cost:.2f} >= "
                f"${self._thresholds['l1_warning_daily']:.2f} — "
                f"batch size 3→2"
            )

        if level != CircuitLevel.NORMAL:
            _logger.warning("Circuit breaker: %s", " | ".join(messages))

        return CircuitStatus(
            level=level,
            daily_cost=daily_cost,
            monthly_cost=monthly_cost,
            message="; ".join(messages) if messages else "OK",
        )


# ============================================================================
# Convenience: CircuitBreaker (lightweight wrapper)
# ============================================================================


class CircuitBreaker:
    """Lightweight callable for checking circuit status.

    Wraps a :class:`CostTracker` for use in pipeline decision points::

        breaker = CircuitBreaker(tracker)
        status = breaker()
        if not status.fine_screening_allowed():
            # degrade to coarse screening only
            ...
    """

    def __init__(self, tracker: CostTracker) -> None:
        self._tracker = tracker

    def __call__(self) -> CircuitStatus:
        return self._tracker.check_circuit()
