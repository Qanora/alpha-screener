"""Tests for LLM cost tracking & circuit breaker.

Issue #108: Cost tracking & circuit breaker.
Reference: PRD 4.6.2 / 4.6.3.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alphascreener.cost.tracker import (
    MODEL_PRICING,
    CircuitBreaker,
    CircuitLevel,
    CircuitStatus,
    CostTracker,
)
from alphascreener.db.models import Base, LlmCostDaily


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables created."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_factory(db_engine):
    """Factory returning a new SQLAlchemy Session."""

    def _sf() -> Session:
        return Session(db_engine)

    return _sf


@pytest.fixture
def tracker(session_factory):
    """Default CostTracker instance with fresh DB."""
    return CostTracker(session_factory)


# ============================================================================
# Unit tests: calc_call_cost
# ============================================================================


class TestCalcCallCost:
    """Unit tests for static cost calculation."""

    def test_gpt4o_mini_pricing(self):
        """GPT-4o-mini: $0.15/1M input, $0.60/1M output."""
        cost = CostTracker.calc_call_cost(1_000_000, 1_000_000, model="gpt-4o-mini")
        assert cost == pytest.approx(0.15 + 0.60, rel=1e-6)

    def test_gpt4o_pricing(self):
        """GPT-4o: $2.50/1M input, $10.00/1M output."""
        cost = CostTracker.calc_call_cost(1_000_000, 500_000, model="gpt-4o")
        assert cost == pytest.approx(2.50 + 5.00, rel=1e-6)

    def test_zero_tokens(self):
        """Zero tokens should yield zero cost."""
        cost = CostTracker.calc_call_cost(0, 0, model="gpt-4o-mini")
        assert cost == 0.0

    def test_unknown_model_fallback(self):
        """Unknown model falls back to gpt-4o-mini pricing."""
        cost_unknown = CostTracker.calc_call_cost(1_000_000, 0, model="nonexistent-model")
        cost_known = CostTracker.calc_call_cost(1_000_000, 0, model="gpt-4o-mini")
        assert cost_unknown == pytest.approx(cost_known, rel=1e-6)

    def test_small_token_counts(self):
        """Small token counts compute correctly."""
        cost = CostTracker.calc_call_cost(500, 200, model="gpt-4o-mini")
        expected = 500 * 0.15 / 1_000_000 + 200 * 0.60 / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-6)


# ============================================================================
# Integration tests: record_call
# ============================================================================


class TestRecordCall:
    """Tests for persisting LLM call costs."""

    def test_first_record_creates_row(self, tracker):
        """First call creates a new llm_cost_daily row."""
        tracker.record_call("refining", 500, 200, model="gpt-4o-mini")

        daily, count = tracker.get_daily_cost()
        assert count == 1
        assert daily > 0.0

    def test_multiple_calls_accumulate(self, tracker):
        """Multiple calls on the same day accumulate correctly."""
        tracker.record_call("bull", 1000, 0, model="gpt-4o-mini")
        tracker.record_call("bear", 1000, 0, model="gpt-4o-mini")
        tracker.record_call("pm", 500, 100, model="gpt-4o-mini")

        daily, count = tracker.get_daily_cost()
        assert count == 3
        # Verify sum
        expected = (
            CostTracker.calc_call_cost(1000, 0, "gpt-4o-mini") * 2
            + CostTracker.calc_call_cost(500, 100, "gpt-4o-mini")
        )
        assert daily == pytest.approx(expected, rel=1e-6)

    def test_by_module_breakdown(self, tracker, session_factory):
        """``by_module_json`` tracks per-module costs."""
        tracker.record_call("bull", 1000, 200, model="gpt-4o-mini")
        tracker.record_call("bear", 800, 150, model="gpt-4o-mini")
        tracker.record_call("bull", 500, 100, model="gpt-4o-mini")

        with session_factory() as s:
            row = s.get(LlmCostDaily, date.today())
            breakdown = json.loads(row.by_module_json)

        assert "bull" in breakdown
        assert "bear" in breakdown
        assert breakdown["bull"] > breakdown["bear"]

    def test_different_dates_get_separate_rows(self, tracker, session_factory):
        """Calls on different dates create separate rows."""
        yesterday = date.today() - timedelta(days=1)

        tracker.record_call("refining", 100, 50, model="gpt-4o-mini")  # today
        tracker.record_call(
            "refining", 200, 60, model="gpt-4o-mini", cost_date=yesterday
        )

        with session_factory() as s:
            today_row = s.get(LlmCostDaily, date.today())
            yesterday_row = s.get(LlmCostDaily, yesterday)

        assert today_row is not None
        assert yesterday_row is not None
        assert today_row.call_count == 1
        assert yesterday_row.call_count == 1
        assert today_row.total_usd != yesterday_row.total_usd

    def test_empty_db_returns_zero(self, tracker):
        """Fresh DB returns (0.0, 0) for daily cost."""
        daily, count = tracker.get_daily_cost()
        assert daily == 0.0
        assert count == 0

    def test_record_call_returns_cost(self, tracker):
        """record_call returns the USD cost of the individual call."""
        cost = tracker.record_call("pm", 1000000, 0, model="gpt-4o-mini")
        assert cost == pytest.approx(0.15, rel=1e-6)


# ============================================================================
# Unit tests: get_monthly_cost
# ============================================================================


class TestMonthlyCost:
    """Tests for monthly cost aggregation."""

    def test_empty_month_returns_zero(self, tracker):
        """Empty month returns 0.0."""
        assert tracker.get_monthly_cost() == 0.0

    def test_sums_all_days_in_month(self, tracker):
        """Monthly cost sums across multiple days in the same month."""
        today = date.today()
        # Avoid cross-month boundary on the 1st: ``today - 1 day`` would
        # land in the previous month and break the two-day assertion.
        if today.day == 1:
            today = today.replace(day=2)
        tracker.record_call("refining", 1_000_000, 0, model="gpt-4o-mini",
                            cost_date=today)
        tracker.record_call("refining", 1_000_000, 0, model="gpt-4o-mini",
                            cost_date=today - timedelta(days=1))
        # Day from previous month (should NOT be included)
        prev_month = today.replace(day=1) - timedelta(days=1)
        tracker.record_call("refining", 1_000_000, 0, model="gpt-4o-mini",
                            cost_date=prev_month)

        monthly = tracker.get_monthly_cost(ref_date=today)
        assert monthly == pytest.approx(0.30, rel=1e-6)  # only 2 days in current month

    def test_previous_month_not_included(self, tracker):
        """Records from previous months are NOT included in current monthly total."""
        today = date.today()
        # Record in previous month
        prev_month_day = today.replace(day=1) - timedelta(days=1)
        tracker.record_call("refining", 10_000_000, 0, model="gpt-4o-mini",
                            cost_date=prev_month_day)

        monthly = tracker.get_monthly_cost(ref_date=today)
        assert monthly == 0.0


# ============================================================================
# Unit tests: circuit breaker
# ============================================================================


class TestCircuitBreaker:
    """Tests for 4-level circuit breaker logic."""

    def test_normal_when_below_all_thresholds(self, tracker):
        """NORMAL when costs are below all thresholds."""
        status = tracker.check_circuit()
        assert status.level == CircuitLevel.NORMAL
        assert status.fine_screening_allowed()
        assert not status.is_blocked()

    def test_l1_warning_when_daily_above_warning(self, session_factory):
        """L1 triggered when daily cost >= $0.80."""
        tracker_low = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 0.10, "l2_degrade_daily": 999.0,
                         "l3_savings_monthly": 999.0, "l4_circuit_monthly": 999.0},
        )
        # Record a call that pushes daily cost above L1
        tracker_low.record_call(
            "refining", 1_000_000, 1_000_000, model="gpt-4o-mini",
        )  # ~$0.75 — above 0.10 L1
        status = tracker_low.check_circuit()
        assert status.level == CircuitLevel.L1_WARNING

    def test_l2_degrade_when_daily_above_degrade(self, session_factory):
        """L2 triggered when daily cost >= $1.00."""
        tracker_l2 = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 0.05, "l2_degrade_daily": 0.10,
                         "l3_savings_monthly": 999.0, "l4_circuit_monthly": 999.0},
        )
        tracker_l2.record_call("refining", 1_000_000, 1_000_000, model="gpt-4o-mini")  # ~$0.75
        status = tracker_l2.check_circuit()
        assert status.level == CircuitLevel.L2_DEGRADE
        # L2 DEGRADE blocks fine screening entirely.
        assert not status.fine_screening_allowed()

    def test_l3_savings_when_monthly_above_savings(self, session_factory):
        """L3 triggered when monthly cost >= threshold."""
        tracker_l3 = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 999.0, "l2_degrade_daily": 999.0,
                         "l3_savings_monthly": 0.10, "l4_circuit_monthly": 999.0},
        )
        tracker_l3.record_call("refining", 1_000_000, 1_000_000, model="gpt-4o-mini")  # ~$0.75
        status = tracker_l3.check_circuit()
        assert status.level == CircuitLevel.L3_SAVINGS

    def test_l4_breaker_stops_all(self, session_factory):
        """L4 triggered when monthly cost >= circuit threshold."""
        tracker_l4 = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 999.0, "l2_degrade_daily": 999.0,
                         "l3_savings_monthly": 0.10, "l4_circuit_monthly": 0.50},
        )
        tracker_l4.record_call("refining", 1_000_000, 1_000_000, model="gpt-4o-mini")  # ~$0.75
        status = tracker_l4.check_circuit()
        assert status.level == CircuitLevel.L4_BREAKER
        assert status.is_blocked()
        assert not status.fine_screening_allowed()

    def test_l4_takes_priority_over_all(self, session_factory):
        """L4 always wins, even when all lower levels also fire."""
        tracker = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 0.01, "l2_degrade_daily": 0.02,
                         "l3_savings_monthly": 0.03, "l4_circuit_monthly": 0.04},
        )
        tracker.record_call("refining", 1_000_000, 1_000_000, model="gpt-4o-mini")
        status = tracker.check_circuit()
        assert status.level == CircuitLevel.L4_BREAKER

    def test_circuit_status_label(self, session_factory):
        """CircuitStatus.label returns human-readable string."""
        tracker = CostTracker(session_factory)
        status = tracker.check_circuit()
        assert status.label == "NORMAL"

    def test_circuit_status_message(self, session_factory):
        """CircuitStatus.message is non-empty when any level trips."""
        tracker = CostTracker(
            session_factory,
            thresholds={"l1_warning_daily": 0.01, "l2_degrade_daily": 999.0,
                         "l3_savings_monthly": 999.0, "l4_circuit_monthly": 999.0},
        )
        tracker.record_call("refining", 1_000_000, 0, model="gpt-4o-mini")
        status = tracker.check_circuit()
        assert "L1" in status.message


# ============================================================================
# Integration: CircuitBreaker wrapper
# ============================================================================


class TestCircuitBreakerWrapper:
    """Tests for the CircuitBreaker convenience wrapper."""

    def test_callable_returns_circuit_status(self, tracker):
        """CircuitBreaker is callable and returns CircuitStatus."""
        breaker = CircuitBreaker(tracker)
        status = breaker()
        assert isinstance(status, CircuitStatus)
        assert status.level == CircuitLevel.NORMAL


# ============================================================================
# Integration: reset_monthly
# ============================================================================


class TestResetMonthly:
    """Tests for the monthly cost reset."""

    def test_reset_does_not_delete_rows(self, tracker, session_factory):
        """reset_monthly logs but does NOT delete data."""
        tracker.record_call("refining", 1_000_000, 0, model="gpt-4o-mini")
        daily_before, count_before = tracker.get_daily_cost()

        tracker.reset_monthly()

        daily_after, count_after = tracker.get_daily_cost()
        assert daily_after == daily_before
        assert count_after == count_before


# ============================================================================
# Pricing table validation
# ============================================================================


class TestModelPricing:
    """Verify MODEL_PRICING has expected structure."""

    def test_all_entries_have_input_output_keys(self):
        """Every pricing entry must have 'input' and 'output' keys."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing 'input'"
            assert "output" in pricing, f"{model} missing 'output'"
            assert pricing["input"] > 0, f"{model} input pricing must be > 0"
            assert pricing["output"] > 0, f"{model} output pricing must be > 0"

    def test_gpt4o_mini_is_default_fallback(self):
        """gpt-4o-mini must exist as the default fallback model."""
        assert "gpt-4o-mini" in MODEL_PRICING


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_corrupted_by_module_json(self, tracker, session_factory):
        """Corrupted ``by_module_json`` is handled gracefully."""
        # Manually insert a row with corrupted JSON
        with session_factory() as s:
            row = LlmCostDaily(
                cost_date=date.today(),
                total_usd=0.05,
                call_count=1,
                by_module_json="not-valid-json{{{",
            )
            s.add(row)
            s.commit()

        # Recording a new call should overwrite the broken JSON
        tracker.record_call("pm", 500, 100, model="gpt-4o-mini")

        with session_factory() as s:
            row = s.get(LlmCostDaily, date.today())
            breakdown = json.loads(row.by_module_json)
        assert "pm" in breakdown
        assert row.call_count == 2

    def test_negative_tokens_raises_error(self):
        """Negative token counts raise ValueError."""
        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            CostTracker.calc_call_cost(-100, -50, model="gpt-4o-mini")
