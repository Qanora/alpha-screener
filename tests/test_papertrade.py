"""Tests for Paper Trading tracker and graduation conditions.

Issue #102: Paper Trading tracker.
Reference: PRD 5.4 / 7.6.2.1.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alphascreener.db.models import Base, PaperTrade

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


# ============================================================================
# P&L calculation unit tests
# ============================================================================


class TestPnlCalculation:
    """Unit tests for P&L percentage calculation."""

    def test_profit_trade(self):
        """Positive return: entry 100, exit 110 -> +10%."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        pnl = calc_pnl_pct(100.0, 110.0)
        assert pnl == pytest.approx(10.0, rel=1e-6)

    def test_loss_trade(self):
        """Negative return: entry 100, exit 92 -> -8%."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        pnl = calc_pnl_pct(100.0, 92.0)
        assert pnl == pytest.approx(-8.0, rel=1e-6)

    def test_break_even(self):
        """Zero return: entry == exit -> 0%."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        pnl = calc_pnl_pct(50.0, 50.0)
        assert pnl == 0.0

    def test_entry_price_zero_raises(self):
        """Entry price of 0 should raise ValueError."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        with pytest.raises(ValueError, match="must be positive and finite"):
            calc_pnl_pct(0.0, 50.0)

    def test_entry_price_negative_raises(self):
        """Negative entry price should raise ValueError."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        with pytest.raises(ValueError, match="must be positive and finite"):
            calc_pnl_pct(-10.0, 50.0)

    def test_large_gain(self):
        """Large gain: entry 10, exit 50 -> +400%."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        pnl = calc_pnl_pct(10.0, 50.0)
        assert pnl == pytest.approx(400.0, rel=1e-6)

    def test_near_total_loss(self):
        """Near total loss: entry 100, exit 1 -> -99%."""
        from alphascreener.papertrade.tracker import calc_pnl_pct

        pnl = calc_pnl_pct(100.0, 1.0)
        assert pnl == pytest.approx(-99.0, rel=1e-6)


# ============================================================================
# Exit reason enum tests
# ============================================================================


class TestExitReason:
    """Tests for exit_reason enum validation."""

    def test_valid_exit_reasons(self):
        """Valid exit reasons are accepted."""
        from alphascreener.papertrade.tracker import ExitReason

        assert ExitReason.TIME == "time"
        assert ExitReason.STOP_LOSS == "stop_loss"
        assert ExitReason.HALT == "halt"

    def test_is_valid_exit_reason(self):
        """is_valid_exit_reason correctly validates reasons."""
        from alphascreener.papertrade.tracker import ExitReason, is_valid_exit_reason

        assert is_valid_exit_reason(ExitReason.TIME)
        assert is_valid_exit_reason(ExitReason.STOP_LOSS)
        assert is_valid_exit_reason(ExitReason.HALT)
        assert not is_valid_exit_reason("random")
        assert not is_valid_exit_reason("")
        assert not is_valid_exit_reason(None)


# ============================================================================
# PaperTradeTracker integration tests
# ============================================================================


class TestPaperTradeTracker:
    """Integration tests for trade entry/exit lifecycle."""

    def test_enter_trade_creates_record(self, session_factory):
        """Entering a trade persists a new paper_trades row."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        sig_date = date(2026, 5, 20)
        trade_id = tracker.enter_trade(
            signal_date=sig_date,
            ticker="AAPL",
            rating="Strong Buy",
            breakout_probability=0.78,
            entry_price=150.0,
            factor_version="1.0.0",
        )

        with session_factory() as s:
            row = s.get(PaperTrade, trade_id)
        assert row is not None
        assert row.ticker == "AAPL"
        assert row.rating == "Strong Buy"
        assert row.entry_price == 150.0
        assert row.exit_price is None
        assert row.exit_reason is None
        assert row.pnl_pct is None
        assert row.signal_date == sig_date

    def test_exit_trade_updates_record(self, session_factory):
        """Exiting an open trade sets exit_price, exit_reason, and pnl_pct."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="MSFT",
            rating="Buy",
            breakout_probability=0.65,
            entry_price=400.0,
            factor_version="1.0.0",
        )

        tracker.exit_trade(
            trade_id=trade_id,
            exit_price=440.0,
            exit_reason=ExitReason.TIME,
        )

        with session_factory() as s:
            row = s.get(PaperTrade, trade_id)
        assert row.exit_price == 440.0
        assert row.exit_reason == "time"
        assert row.pnl_pct == pytest.approx(10.0, rel=1e-6)

    def test_exit_trade_with_stop_loss(self, session_factory):
        """Exit with stop_loss reason computes correct P&L."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="GOOGL",
            rating="Buy",
            breakout_probability=0.60,
            entry_price=200.0,
            factor_version="1.0.0",
        )

        tracker.exit_trade(
            trade_id=trade_id,
            exit_price=184.0,
            exit_reason=ExitReason.STOP_LOSS,
        )

        with session_factory() as s:
            row = s.get(PaperTrade, trade_id)
        assert row.exit_reason == "stop_loss"
        assert row.pnl_pct == pytest.approx(-8.0, rel=1e-6)

    def test_exit_trade_with_halt(self, session_factory):
        """Exit with halt reason is accepted."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="XYZ",
            rating="Hold",
            breakout_probability=0.45,
            entry_price=50.0,
            factor_version="1.0.0",
        )

        tracker.exit_trade(
            trade_id=trade_id,
            exit_price=48.0,
            exit_reason=ExitReason.HALT,
        )

        with session_factory() as s:
            row = s.get(PaperTrade, trade_id)
        assert row.exit_reason == "halt"
        assert row.pnl_pct == pytest.approx(-4.0, rel=1e-6)

    def test_exit_nonexistent_trade_raises(self, session_factory):
        """Exiting a non-existent trade ID raises ValueError."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        with pytest.raises(ValueError, match="Trade 99999 not found"):
            tracker.exit_trade(
                trade_id=99999,
                exit_price=100.0,
                exit_reason="time",
            )

    def test_exit_already_closed_trade_raises(self, session_factory):
        """Exiting a trade that already has exit_price raises ValueError."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="NVDA",
            rating="Strong Buy",
            breakout_probability=0.85,
            entry_price=100.0,
            factor_version="1.0.0",
        )
        tracker.exit_trade(trade_id, 110.0, ExitReason.TIME)

        with pytest.raises(ValueError, match="already closed"):
            tracker.exit_trade(trade_id, 120.0, ExitReason.TIME)

    def test_invalid_exit_reason_raises(self, session_factory):
        """Invalid exit reason string raises ValueError."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="AMD",
            rating="Buy",
            breakout_probability=0.55,
            entry_price=80.0,
            factor_version="1.0.0",
        )

        with pytest.raises(ValueError, match="Invalid exit_reason"):
            tracker.exit_trade(trade_id, 85.0, "foobar")

    def test_get_open_trades(self, session_factory):
        """get_open_trades returns only trades without exit_price."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        tracker.enter_trade(
            signal_date=date(2026, 5, 18),
            ticker="AAPL",
            rating="Buy",
            breakout_probability=0.70,
            entry_price=150.0,
            factor_version="1.0.0",
        )
        tid2 = tracker.enter_trade(
            signal_date=date(2026, 5, 19),
            ticker="MSFT",
            rating="Buy",
            breakout_probability=0.65,
            entry_price=400.0,
            factor_version="1.0.0",
        )
        # Close one trade
        tracker.exit_trade(tid2, 420.0, ExitReason.TIME)

        open_trades = tracker.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].ticker == "AAPL"

    def test_get_open_trades_empty(self, session_factory):
        """get_open_trades returns empty list when no open trades."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        open_trades = tracker.get_open_trades()
        assert open_trades == []

    def test_get_trade_history(self, session_factory):
        """get_trade_history returns completed trades ordered by signal_date desc."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        tid1 = tracker.enter_trade(
            signal_date=date(2026, 5, 18),
            ticker="AAPL",
            rating="Buy",
            breakout_probability=0.70,
            entry_price=150.0,
            factor_version="1.0.0",
        )
        tracker.exit_trade(tid1, 155.0, ExitReason.TIME)

        tid2 = tracker.enter_trade(
            signal_date=date(2026, 5, 19),
            ticker="MSFT",
            rating="Strong Buy",
            breakout_probability=0.80,
            entry_price=400.0,
            factor_version="1.0.0",
        )
        tracker.exit_trade(tid2, 390.0, ExitReason.STOP_LOSS)

        # Open trade should NOT appear in history
        tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="GOOGL",
            rating="Buy",
            breakout_probability=0.60,
            entry_price=200.0,
            factor_version="1.0.0",
        )

        history = tracker.get_trade_history()
        assert len(history) == 2
        # Most recent signal_date first
        assert history[0].ticker == "MSFT"
        assert history[1].ticker == "AAPL"

    def test_enter_trade_without_entry_price(self, session_factory):
        """Entering a trade with entry_price=None (T+1 price unknown at signal time)."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="TSLA",
            rating="Buy",
            breakout_probability=0.72,
            entry_price=None,
            factor_version="1.0.0",
        )

        with session_factory() as s:
            row = s.get(PaperTrade, trade_id)
        assert row is not None
        assert row.entry_price is None
        assert row.ticker == "TSLA"

    def test_exit_trade_with_null_entry_price_raises(self, session_factory):
        """Cannot compute P&L when entry_price is None."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="TSLA",
            rating="Buy",
            breakout_probability=0.72,
            entry_price=None,
            factor_version="1.0.0",
        )

        with pytest.raises(ValueError, match="entry_price"):
            tracker.exit_trade(trade_id, 200.0, "time")

    def test_multiple_trades_same_ticker_different_dates(self, session_factory):
        """Same ticker can have multiple trades on different signal dates."""
        from alphascreener.papertrade.tracker import ExitReason, PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        tid1 = tracker.enter_trade(
            signal_date=date(2026, 5, 18),
            ticker="AAPL",
            rating="Buy",
            breakout_probability=0.70,
            entry_price=150.0,
            factor_version="1.0.0",
        )
        tid2 = tracker.enter_trade(
            signal_date=date(2026, 5, 19),
            ticker="AAPL",
            rating="Strong Buy",
            breakout_probability=0.82,
            entry_price=152.0,
            factor_version="1.0.0",
        )
        tracker.exit_trade(tid1, 155.0, ExitReason.TIME)
        tracker.exit_trade(tid2, 148.0, ExitReason.STOP_LOSS)

        with session_factory() as s:
            row1 = s.get(PaperTrade, tid1)
            row2 = s.get(PaperTrade, tid2)
        assert row1.pnl_pct == pytest.approx(3.33333, rel=1e-4)
        assert row2.pnl_pct == pytest.approx(-2.63157, rel=1e-4)


# ============================================================================
# Tier 1 Engineering Graduation conditions tests (PRD 7.6.2.1)
# ============================================================================


class TestEngineeringGraduation:
    """Tests for Tier 1 engineering graduation conditions."""

    def test_all_conditions_pass(self, session_factory, db_engine):
        """When all conditions are met, returns PASS."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        # Seed: >60 days of data (we simulate by providing manually)
        # Populate some data to make conditions pass
        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=65,
            l3_l4_events_last_30d=0,
            nan_rate=0.02,
            scheduler_success_rate=0.97,
            alt_source_alerts_monthly=2,
        )

        assert result.passed
        assert result.days_in_operation == 65
        assert result.l3_l4_event_count == 0
        assert result.nan_rate == pytest.approx(0.02)
        assert result.scheduler_success_rate == pytest.approx(0.97)
        assert result.alt_source_alerts_monthly == 2
        assert len(result.failed_checks) == 0

    def test_insufficient_days_fails(self, session_factory, db_engine):
        """Less than 60 days of operation fails."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=30,
            l3_l4_events_last_30d=0,
            nan_rate=0.0,
            scheduler_success_rate=1.0,
            alt_source_alerts_monthly=0,
        )

        assert not result.passed
        assert "days_in_operation" in result.failed_checks

    def test_exactly_60_days_passes(self, session_factory, db_engine):
        """Exactly 60 days is the boundary and should pass."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.0,
            scheduler_success_rate=1.0,
            alt_source_alerts_monthly=0,
        )

        assert result.passed

    def test_l3_l4_events_trigger_failure(self, session_factory, db_engine):
        """Any L3 or L4 circuit breaker event in last 30 days fails."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=1,
            nan_rate=0.0,
            scheduler_success_rate=1.0,
            alt_source_alerts_monthly=0,
        )

        assert not result.passed
        assert "l3_l4_events" in result.failed_checks

    def test_high_nan_rate_fails(self, session_factory, db_engine):
        """NaN rate >= 5% fails."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.07,
            scheduler_success_rate=1.0,
            alt_source_alerts_monthly=0,
        )

        assert not result.passed
        assert "nan_rate" in result.failed_checks

    def test_nan_rate_exactly_5_pct_fails(self, session_factory, db_engine):
        """NaN rate exactly at 5% should fail — PRD requires strictly <5%, so 5.0% is not <5%."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.05,
            scheduler_success_rate=1.0,
            alt_source_alerts_monthly=0,
        )

        # <0.05 means 5% fails. Let the test confirm which way the spec goes.
        # PRD says "<5%", so 0.05 should FAIL (equal to 5% is not <5%)
        assert not result.passed

    def test_low_scheduler_success_rate_fails(self, session_factory, db_engine):
        """Scheduler success rate < 95% fails."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.01,
            scheduler_success_rate=0.90,
            alt_source_alerts_monthly=0,
        )

        assert not result.passed
        assert "scheduler_success_rate" in result.failed_checks

    def test_scheduler_success_rate_exactly_95_passes(self, session_factory, db_engine):
        """Scheduler success rate >= 95% passes at exactly 0.95."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.01,
            scheduler_success_rate=0.95,
            alt_source_alerts_monthly=0,
        )

        assert result.passed

    def test_too_many_alt_source_alerts_fails(self, session_factory, db_engine):
        """More than 5 alt source diff alerts per month fails."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=60,
            l3_l4_events_last_30d=0,
            nan_rate=0.01,
            scheduler_success_rate=0.98,
            alt_source_alerts_monthly=7,
        )

        assert not result.passed
        assert "alt_source_alerts" in result.failed_checks

    def test_multiple_failures_all_reported(self, session_factory, db_engine):
        """When multiple conditions fail, all failures are reported."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=20,
            l3_l4_events_last_30d=2,
            nan_rate=0.08,
            scheduler_success_rate=0.80,
            alt_source_alerts_monthly=10,
        )

        assert not result.passed
        assert len(result.failed_checks) == 5
        assert "days_in_operation" in result.failed_checks
        assert "l3_l4_events" in result.failed_checks
        assert "nan_rate" in result.failed_checks
        assert "scheduler_success_rate" in result.failed_checks
        assert "alt_source_alerts" in result.failed_checks

    def test_result_summary_includes_all_details(self, session_factory, db_engine):
        """EngineeringGraduationResult.summary returns human-readable text."""
        from alphascreener.papertrade.graduation import check_engineering_graduation

        result = check_engineering_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            days_in_operation=40,
            l3_l4_events_last_30d=1,
            nan_rate=0.01,
            scheduler_success_rate=0.97,
            alt_source_alerts_monthly=2,
        )

        summary = result.summary
        assert isinstance(summary, str)
        assert "FAIL" in summary or "PASS" in summary
        assert "40" in summary  # days


# ============================================================================
# Tier 2 Strategy Graduation conditions tests (reserved interface)
# ============================================================================


class TestStrategyGraduation:
    """Tests for Tier 2 strategy graduation condition (reserved interface)."""

    def test_interface_returns_not_ready(self, session_factory, db_engine):
        """Tier 2 strategy graduation returns NOT_READY (not yet implemented)."""
        from alphascreener.papertrade.graduation import (
            StrategyGraduationResult,
            check_strategy_graduation,
        )

        result = check_strategy_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
        )

        assert isinstance(result, StrategyGraduationResult)
        assert not result.ready
        assert result.status == "NOT_IMPLEMENTED"
        assert "not yet implemented" in result.summary.lower()

    def test_interface_accepts_unused_kwargs(self, session_factory, db_engine):
        """The reserved interface accepts (and ignores) strategy parameters."""
        from alphascreener.papertrade.graduation import check_strategy_graduation

        result = check_strategy_graduation(
            session_factory=session_factory,
            db_engine=db_engine,
            walk_forward_years=2,
            live_shadow_months=6,
            lift_at_20=1.15,
            llm_delta_lift=0.06,
            ic_decay=0.30,
        )

        assert not result.ready
        assert result.status == "NOT_IMPLEMENTED"


# ============================================================================
# Edge cases
# ============================================================================


class TestPaperTradeEdgeCases:
    """Edge-case and robustness tests for paper trade tracker."""

    def test_zero_entry_price_raises(self, session_factory):
        """Entry price of 0.0 raises ValueError."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        with pytest.raises(ValueError, match="must be positive and finite"):
            tracker.enter_trade(
                signal_date=date(2026, 5, 20),
                ticker="BAD",
                rating="Buy",
                breakout_probability=0.5,
                entry_price=0.0,
                factor_version="1.0.0",
            )

    def test_negative_entry_price_raises(self, session_factory):
        """Negative entry price raises ValueError."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        with pytest.raises(ValueError, match="must be positive and finite"):
            tracker.enter_trade(
                signal_date=date(2026, 5, 20),
                ticker="BAD",
                rating="Buy",
                breakout_probability=0.5,
                entry_price=-50.0,
                factor_version="1.0.0",
            )

    def test_get_trade_history_empty(self, session_factory):
        """get_trade_history returns empty list when no closed trades."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        history = tracker.get_trade_history()
        assert history == []

    def test_get_trade_by_id(self, session_factory):
        """Can retrieve a specific trade by ID."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade_id = tracker.enter_trade(
            signal_date=date(2026, 5, 20),
            ticker="PLTR",
            rating="Avoid",
            breakout_probability=0.25,
            entry_price=15.0,
            factor_version="1.0.0",
        )

        trade = tracker.get_trade(trade_id)
        assert trade is not None
        assert trade.ticker == "PLTR"

    def test_get_trade_nonexistent(self, session_factory):
        """get_trade returns None for non-existent ID."""
        from alphascreener.papertrade.tracker import PaperTradeTracker

        tracker = PaperTradeTracker(session_factory)
        trade = tracker.get_trade(99999)
        assert trade is None
