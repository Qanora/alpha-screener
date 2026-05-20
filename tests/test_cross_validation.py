"""Tests for OHLCV cross-validation module.

Issue #91: Stooq fallback adapter + cross-validation.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alphascreener.cross_validation.comparator import (
    DIFF_THRESHOLD_PCT,
    OHLCVFieldDiffs,
    compare_ohlcv_dataframes,
    compute_diff_pct,
)
from alphascreener.cross_validation.diff_store import DiffStore
from alphascreener.cross_validation.health_monitor import (
    DEFAULT_FAILURE_THRESHOLD_PCT,
    DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS,
    DailyHealthRecord,
    YFinanceHealthMonitor,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def yfinance_ohlcv() -> pl.DataFrame:
    """Sample yfinance OHLCV DataFrame."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
            "dt": [
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 6),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ],
            "open": [150.0, 151.0, 153.0, 140.0, 141.5],
            "high": [152.0, 153.0, 155.0, 142.0, 143.0],
            "low": [149.0, 150.0, 152.5, 139.0, 140.5],
            "close": [151.5, 152.5, 154.5, 141.0, 142.0],
            "volume": [100000, 110000, 120000, 200000, 210000],
        },
    )


@pytest.fixture
def stooq_ohlcv_match() -> pl.DataFrame:
    """Stooq OHLCV that closely matches yfinance (within 0.5%)."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
            "dt": [
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 6),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ],
            "open": [150.0, 151.0, 153.0, 140.0, 141.5],
            "high": [152.0, 153.0, 155.0, 142.0, 143.0],
            "low": [149.0, 150.0, 152.5, 139.0, 140.5],
            "close": [151.5, 152.5, 154.5, 141.0, 142.0],
            "volume": [100000, 110000, 120000, 200000, 210000],
        },
    )


@pytest.fixture
def stooq_ohlcv_divergent() -> pl.DataFrame:
    """Stooq OHLCV with significant divergences (>0.5%) on some fields."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "GOOGL"],
            "dt": [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 2)],
            # AAPL Jan 2: close 151.5 → 145.0 = 4.3% diff (>0.5%)
            "open": [150.0, 151.0, 140.0],
            "high": [152.0, 153.0, 142.0],
            "low": [149.0, 150.0, 139.0],
            "close": [145.0, 155.0, 140.0],  # AAPL Jan 2 close differs 4.3%
            "volume": [100000, 110000, 200000],
        },
    )


@pytest.fixture
def empty_ohlcv() -> pl.DataFrame:
    """Empty OHLCV DataFrame with correct schema."""
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "dt": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )


@pytest.fixture
def diff_store() -> DiffStore:
    """Return a fresh DiffStore with in-memory SQLite."""
    return DiffStore(":memory:")


# ============================================================================
# compute_diff_pct
# ============================================================================


class TestComputeDiffPct:
    """Unit tests for relative difference computation."""

    def test_identical_values(self):
        assert compute_diff_pct(100.0, 100.0) == 0.0

    def test_zero_diff(self):
        assert compute_diff_pct(50.0, 50.0) == 0.0

    def test_small_diff_below_threshold(self):
        # 100 → 99.5 = 0.5% diff (exactly at threshold, not above)
        pct = compute_diff_pct(100.0, 99.5)
        assert pct == 0.5

    def test_diff_above_threshold(self):
        # 100 → 99.0 = 1.0% diff
        pct = compute_diff_pct(100.0, 99.0)
        assert pct == 1.0

    def test_large_diff(self):
        # 100 → 95 = 5.0% diff
        pct = compute_diff_pct(100.0, 95.0)
        assert pct == 5.0

    def test_both_zero(self):
        assert compute_diff_pct(0.0, 0.0) == 0.0

    def test_primary_zero_fallback_nonzero(self):
        # Primary is 0 but fallback is non-zero → infinite diff
        pct = compute_diff_pct(0.0, 100.0)
        assert pct == float("inf")

    def test_fallback_zero(self):
        # Primary 100, fallback 0 → 100% diff
        pct = compute_diff_pct(100.0, 0.0)
        assert pct == 100.0

    def test_negative_values(self):
        # Primary -100, fallback -95 → 5% diff
        pct = compute_diff_pct(-100.0, -95.0)
        assert pct == 5.0


# ============================================================================
# compare_ohlcv_dataframes
# ============================================================================


class TestCompareOhlcvDataframes:
    """Integration tests for field-level OHLCV comparison."""

    def test_matching_data_no_diffs(self, yfinance_ohlcv, stooq_ohlcv_match):
        result = compare_ohlcv_dataframes(yfinance_ohlcv, stooq_ohlcv_match)
        assert len(result.records) == 0
        assert len(result.diff_tickers) == 0

    def test_divergent_data_detects_diffs(self, yfinance_ohlcv, stooq_ohlcv_divergent):
        result = compare_ohlcv_dataframes(yfinance_ohlcv, stooq_ohlcv_divergent)
        # AAPL Jan 2 close: 151.5 → 145.0 = ~4.29%, should be flagged
        # AAPL Jan 3 close: 152.5 → 155.0 = ~1.64%, should be flagged
        assert len(result.records) > 0

        # Check that AAPL is in diff_tickers
        assert "AAPL" in result.diff_tickers

        # Verify records have correct schema
        for rec in result.records:
            assert "ticker" in rec
            assert "dt" in rec
            assert "field" in rec
            assert "primary_value" in rec
            assert "fallback_value" in rec
            assert "fallback_source" in rec
            assert "diff_pct" in rec
            assert rec["diff_pct"] > DIFF_THRESHOLD_PCT

    def test_empty_primary(self, empty_ohlcv, stooq_ohlcv_match):
        result = compare_ohlcv_dataframes(empty_ohlcv, stooq_ohlcv_match)
        assert len(result.records) == 0
        assert len(result.diff_tickers) == 0

    def test_empty_fallback(self, yfinance_ohlcv, empty_ohlcv):
        result = compare_ohlcv_dataframes(yfinance_ohlcv, empty_ohlcv)
        assert len(result.records) == 0

    def test_no_overlapping_dates(self, yfinance_ohlcv):
        """When there are no overlapping (ticker, dt) pairs, no diffs."""
        non_overlapping = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 2, 1)],  # Different date
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.5],
                "volume": [100000],
            }
        )
        result = compare_ohlcv_dataframes(yfinance_ohlcv, non_overlapping)
        assert len(result.records) == 0

    def test_custom_threshold(self, yfinance_ohlcv, stooq_ohlcv_divergent):
        # With a very high threshold (50%), nothing should be flagged
        result = compare_ohlcv_dataframes(yfinance_ohlcv, stooq_ohlcv_divergent, threshold_pct=50.0)
        assert len(result.records) == 0

        # With a very low threshold (0.001%), many diffs
        result2 = compare_ohlcv_dataframes(
            yfinance_ohlcv, stooq_ohlcv_divergent, threshold_pct=0.001
        )
        assert len(result2.records) > 0

    def test_custom_fallback_source(self, yfinance_ohlcv, stooq_ohlcv_divergent):
        result = compare_ohlcv_dataframes(
            yfinance_ohlcv, stooq_ohlcv_divergent, fallback_source="alpaca"
        )
        for rec in result.records:
            assert rec["fallback_source"] == "alpaca"

    def test_volume_diff_detected(self):
        """Volume diffs should also be flagged when above threshold."""
        primary = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 1, 2)],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.5],
                "volume": [100000],
            },
        )
        fallback = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 1, 2)],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.5],
                "volume": [200000],  # 100% diff in volume
            },
        )
        result = compare_ohlcv_dataframes(primary, fallback)
        assert len(result.records) > 0
        volume_recs = [r for r in result.records if r["field"] == "volume"]
        assert len(volume_recs) > 0

    def test_ohlcvfielddiffs_defaults(self):
        """OHLCVFieldDiffs dataclass should have sensible defaults."""
        diffs = OHLCVFieldDiffs()
        assert diffs.records == []
        assert diffs.diff_tickers == set()

    def test_compare_skips_nan_values(self):
        """NaN values should be skipped in comparison."""
        primary = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 1, 2)],
                "open": [150.0],
                "high": [float("nan")],
                "low": [149.0],
                "close": [151.5],
                "volume": [100000],
            },
        )
        fallback = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 1, 2)],
                "open": [150.0],
                "high": [200.0],  # Very different but NaN in primary → skipped
                "low": [149.0],
                "close": [151.5],
                "volume": [100000],
            },
        )
        result = compare_ohlcv_dataframes(primary, fallback)
        # High field should not be in records because primary is NaN
        high_recs = [r for r in result.records if r["field"] == "high"]
        assert len(high_recs) == 0


# ============================================================================
# DiffStore
# ============================================================================


class TestDiffStore:
    """Persistence tests for data_source_diff records."""

    def test_insert_diffs_empty(self, diff_store):
        """Inserting empty diffs is a no-op."""
        diffs = OHLCVFieldDiffs()
        count = diff_store.insert_diffs(diffs)
        assert count == 0

    def test_insert_and_count_diffs(self, diff_store):
        """Insert diffs and count them."""
        diffs = OHLCVFieldDiffs(
            records=[
                {
                    "ticker": "AAPL",
                    "dt": date(2025, 1, 2),
                    "field": "close",
                    "primary_value": 151.5,
                    "fallback_value": 145.0,
                    "fallback_source": "stooq",
                    "diff_pct": 4.29,
                },
                {
                    "ticker": "AAPL",
                    "dt": date(2025, 1, 3),
                    "field": "close",
                    "primary_value": 152.5,
                    "fallback_value": 155.0,
                    "fallback_source": "stooq",
                    "diff_pct": 1.64,
                },
                {
                    "ticker": "GOOGL",
                    "dt": date(2025, 1, 2),
                    "field": "open",
                    "primary_value": 140.0,
                    "fallback_value": 142.8,
                    "fallback_source": "stooq",
                    "diff_pct": 2.0,
                },
            ],
            diff_tickers={"AAPL", "GOOGL"},
        )
        count = diff_store.insert_diffs(diffs)
        assert count == 3

        # Count daily diffs for Jan 2
        n = diff_store.count_daily_diffs(date(2025, 1, 2))
        assert n == 2

        # Count distinct tickers
        n_tickers = diff_store.count_daily_diff_tickers(date(2025, 1, 2))
        assert n_tickers == 2

    def test_count_daily_diffs_none_found(self, diff_store):
        """Counting diffs on a date with no records returns 0."""
        n = diff_store.count_daily_diffs(date(2025, 12, 25))
        assert n == 0

    def test_count_daily_diff_tickers_none_found(self, diff_store):
        n = diff_store.count_daily_diff_tickers(date(2025, 12, 25))
        assert n == 0

    def test_mark_alerted(self, diff_store):
        """Marking alerted sets the alerted flag."""
        diffs = OHLCVFieldDiffs(
            records=[
                {
                    "ticker": "AAPL",
                    "dt": date(2025, 1, 2),
                    "field": "close",
                    "primary_value": 151.5,
                    "fallback_value": 145.0,
                    "fallback_source": "stooq",
                    "diff_pct": 4.29,
                },
            ],
        )
        diff_store.insert_diffs(diffs)

        # Mark alerted
        n_updated = diff_store.mark_alerted(date(2025, 1, 2))
        assert n_updated == 1

    def test_default_date_uses_today(self, diff_store):
        """count_daily_diffs with no args uses today's date."""
        from datetime import date as today_date

        today_records = [
            {
                "ticker": "AAPL",
                "dt": today_date.today(),
                "field": "close",
                "primary_value": 151.5,
                "fallback_value": 145.0,
                "fallback_source": "stooq",
                "diff_pct": 4.29,
            },
        ]
        diffs = OHLCVFieldDiffs(records=today_records)
        diff_store.insert_diffs(diffs)

        n = diff_store.count_daily_diffs()  # No date → today
        assert n == 1


# ============================================================================
# YFinanceHealthMonitor
# ============================================================================


class TestYFinanceHealthMonitor:
    """Tests for yfinance health monitoring and fallback switching."""

    @pytest.fixture
    def monitor(self) -> YFinanceHealthMonitor:
        """Fresh health monitor with default thresholds."""
        return YFinanceHealthMonitor()

    def test_default_values(self):
        monitor = YFinanceHealthMonitor()
        assert monitor.failure_threshold_pct == DEFAULT_FAILURE_THRESHOLD_PCT  # 30.0
        assert monitor.consecutive_days == DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS  # 3

    def test_custom_values(self):
        monitor = YFinanceHealthMonitor(
            failure_threshold_pct=50.0,
            consecutive_days=2,
        )
        assert monitor.failure_threshold_pct == 50.0
        assert monitor.consecutive_days == 2

    def test_initial_state(self, monitor):
        assert monitor.fallback_activated is False
        assert monitor.consecutive_exceeded == 0
        assert monitor.history == []

    def test_healthy_day_resets_counter(self, monitor):
        """A healthy day (<30% failure) resets the consecutive exceeded counter."""
        # Seed with one bad day
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=40)
        )
        assert monitor.consecutive_exceeded == 1

        # Healthy day
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 3), total_tickers=100, failed_tickers=5)
        )
        assert monitor.consecutive_exceeded == 0
        assert monitor.fallback_activated is False

    def test_consecutive_failures_trigger_switch(self, monitor):
        """3 consecutive days with ≥30% failure rate triggers full switch."""
        for day_offset in range(3):
            monitor.record_day(
                DailyHealthRecord(
                    date=date(2025, 1, 2 + day_offset),
                    total_tickers=100,
                    failed_tickers=40,  # 40% failure rate
                )
            )

        assert monitor.consecutive_exceeded == 3
        assert monitor.fallback_activated is True

    def test_switch_triggers_only_once(self, monitor):
        """Once fallback is activated, it stays activated."""
        for day_offset in range(3):
            monitor.record_day(
                DailyHealthRecord(
                    date=date(2025, 1, 2 + day_offset),
                    total_tickers=100,
                    failed_tickers=40,
                )
            )
        assert monitor.fallback_activated is True

        # Another bad day should not change state or double-trigger
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 5), total_tickers=100, failed_tickers=50)
        )
        assert monitor.fallback_activated is True
        assert monitor.consecutive_exceeded == 4

    def test_intermittent_failures_no_switch(self, monitor):
        """Intermittent bad days (not consecutive) should not trigger switch."""
        # Bad → Good → Bad → Good → Bad (never 3 consecutive)
        for i, failed in enumerate([40, 5, 35, 10, 45]):
            monitor.record_day(
                DailyHealthRecord(
                    date=date(2025, 1, 2 + i),
                    total_tickers=100,
                    failed_tickers=failed,
                )
            )
        assert monitor.fallback_activated is False

    def test_failure_rate_exactly_at_threshold(self, monitor):
        """Exactly 30% failure rate should count as exceeded threshold."""
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=30)
        )
        assert monitor.consecutive_exceeded == 1

    def test_failure_rate_below_threshold(self, monitor):
        """29% failure rate is below threshold, should not count."""
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=29)
        )
        assert monitor.consecutive_exceeded == 0

    def test_zero_tickers_recorded(self, monitor):
        """Zero total tickers → 0% failure rate."""
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 2), total_tickers=0, failed_tickers=0)
        )
        assert monitor.consecutive_exceeded == 0

    def test_record_today_convenience(self, monitor, monkeypatch):
        """record_today() uses Date.today() when not specified."""

        fake_today = date(2025, 3, 15)

        class _FakeDate(date):
            @classmethod
            def today(cls):
                return fake_today

        monkeypatch.setattr(
            "alphascreener.cross_validation.health_monitor.date",
            _FakeDate,
        )

        monitor.record_today(total_tickers=100, failed_tickers=50)
        assert len(monitor.history) == 1
        assert monitor.history[0].date == fake_today
        assert monitor.consecutive_exceeded == 1

    def test_reset(self, monitor):
        """reset() clears all state."""
        for _ in range(3):
            monitor.record_day(
                DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=40)
            )
        assert monitor.fallback_activated is True

        monitor.reset()
        assert monitor.fallback_activated is False
        assert monitor.consecutive_exceeded == 0
        assert monitor.history == []

    def test_daily_health_record_failure_rate(self):
        """DailyHealthRecord computes failure_rate_pct correctly."""
        record = DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=35)
        assert record.failure_rate_pct == 35.0

        record_zero = DailyHealthRecord(date=date(2025, 1, 2), total_tickers=0, failed_tickers=0)
        assert record_zero.failure_rate_pct == 0.0

    def test_history_is_copy(self, monitor):
        """history property returns a copy, not a reference."""
        monitor.record_day(
            DailyHealthRecord(date=date(2025, 1, 2), total_tickers=100, failed_tickers=10)
        )
        h = monitor.history
        h.clear()
        assert len(monitor.history) == 1  # Original unaffected


# ============================================================================
# DiffStore with module-level functions
# ============================================================================


class TestModuleLevelDiffStoreFunctions:
    """Test the stateless convenience functions in diff_store."""

    def test_count_daily_diffs_function(self, diff_store):
        from alphascreener.cross_validation.diff_store import count_daily_diffs

        diffs = OHLCVFieldDiffs(
            records=[
                {
                    "ticker": "AAPL",
                    "dt": date(2025, 1, 2),
                    "field": "close",
                    "primary_value": 151.5,
                    "fallback_value": 145.0,
                    "fallback_source": "stooq",
                    "diff_pct": 4.29,
                },
            ],
        )
        diff_store.insert_diffs(diffs)

        n = count_daily_diffs(diff_store._engine, date(2025, 1, 2))
        assert n == 1

    def test_mark_alerted_diffs_function(self, diff_store):
        from alphascreener.cross_validation.diff_store import mark_alerted_diffs

        diffs = OHLCVFieldDiffs(
            records=[
                {
                    "ticker": "AAPL",
                    "dt": date(2025, 1, 2),
                    "field": "close",
                    "primary_value": 151.5,
                    "fallback_value": 145.0,
                    "fallback_source": "stooq",
                    "diff_pct": 4.29,
                },
            ],
        )
        diff_store.insert_diffs(diffs)
        n = mark_alerted_diffs(diff_store._engine, date(2025, 1, 2))
        assert n == 1
