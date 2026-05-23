"""Tests for CUSUM fast-layer factor health monitoring.

Issue #103: CUSUM fast-layer monitoring.
Issue #189: CUSUM alert storm fixes — NaN guard, dedup, L3 cooldown.
Reference: PRD 6.1.1 / 6.3.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alphascreener.db.models import Alert, Base, FactorHealthDaily
from alphascreener.monitoring.cusum import (
    CUSUMConfig,
    CUSUMMonitor,
    _compute_cusum,
    _rolling_mean,
    _send_feishu_notification,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_engine(tmp_path: Path):
    """In-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def default_config() -> CUSUMConfig:
    return CUSUMConfig()


@pytest.fixture
def sample_dates() -> list[date]:
    """Generate 10 consecutive dates starting from 2025-01-01."""
    base = date(2025, 1, 1)
    return [base + timedelta(days=i) for i in range(10)]


# ============================================================================
# 1. CUSUM formula calculation
# ============================================================================


class TestCUSUMFormula:
    """S_t = max(0, S_{t-1} + IC_t - mu_IC - k)."""

    def test_cusum_below_threshold_stays_zero(self):
        """When IC is below mu+k, CUSUM stays at 0."""
        # mu=0.05, k=0.005, IC=0.04 => IC - mu - k = 0.04 - 0.05 - 0.005 = -0.015
        result = _compute_cusum(
            ic_t=0.04,
            s_prev=0.0,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == 0.0

    def test_cusum_accumulates_above_threshold(self):
        """When IC is above mu+k, CUSUM accumulates."""
        # mu=0.05, k=0.005, IC=0.07 => IC - mu - k = 0.07 - 0.05 - 0.005 = 0.015
        result = _compute_cusum(
            ic_t=0.07,
            s_prev=0.0,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == pytest.approx(0.015)

    def test_cusum_accumulates_from_previous(self):
        """S_t = S_{t-1} + IC_t - mu - k."""
        # S_{t-1}=0.03, mu=0.05, k=0.005, IC=0.08 => 0.03 + 0.08 - 0.05 - 0.005 = 0.055
        result = _compute_cusum(
            ic_t=0.08,
            s_prev=0.03,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == pytest.approx(0.055)

    def test_cusum_clamps_to_zero(self):
        """S_t cannot go below 0."""
        # S_{t-1}=0.01, mu=0.05, k=0.005, IC=-0.1 => 0.01 + (-0.1) - 0.05 - 0.005 = -0.145
        result = _compute_cusum(
            ic_t=-0.1,
            s_prev=0.01,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == 0.0

    def test_cusum_with_negative_ic_resets(self):
        """A large negative IC can reset accumulated CUSUM."""
        # S_{t-1}=0.04, mu=0.05, k=0.005, IC=-0.15 => 0.04 + (-0.15) - 0.05 - 0.005 = -0.165
        result = _compute_cusum(
            ic_t=-0.15,
            s_prev=0.04,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == 0.0

    def test_cusum_default_parameters(self):
        """With default config k=0.005."""
        result = _compute_cusum(ic_t=0.10, s_prev=0.0, mu_ic=0.05, k=0.005)
        # 0.10 - 0.05 - 0.005 = 0.045
        assert result == pytest.approx(0.045)


# ============================================================================
# 2. Rolling mean calculation
# ============================================================================


class TestRollingMean:
    """Rolling window mean computation for mu_IC."""

    def test_rolling_mean_exact_window(self):
        """Mean of exactly window-length values."""
        values = [0.05, 0.06, 0.04, 0.07, 0.03]
        result = _rolling_mean(values, window=90)
        # Mean of all 5 values: (0.05+0.06+0.04+0.07+0.03)/5 = 0.05
        assert result == pytest.approx(0.05)

    def test_rolling_mean_truncates_to_window(self):
        """Only uses last N values when list exceeds window."""
        values = [0.01] * 100 + [0.10] * 10  # first 100 are 0.01, last 10 are 0.10
        result = _rolling_mean(values, window=90)
        # Last 90 values: 80 x 0.01 + 10 x 0.10 = 0.8 + 1.0 = 1.8, / 90 = 0.02
        assert result == pytest.approx(1.8 / 90.0)

    def test_rolling_mean_empty_returns_zero(self):
        """Empty list returns 0.0."""
        assert _rolling_mean([], window=90) == 0.0

    def test_rolling_mean_single_value(self):
        """Single value returns that value."""
        assert _rolling_mean([0.05], window=90) == pytest.approx(0.05)


# ============================================================================
# 3. CUSUMConfig defaults
# ============================================================================


class TestCUSUMConfig:
    """CUSUMConfig holds all threshold constants."""

    def test_default_config_values(self):
        cfg = CUSUMConfig()
        assert cfg.k == 0.005
        assert cfg.h == 0.05
        assert cfg.rolling_window_days == 90
        assert cfg.l2_window_days == 5
        assert cfg.l2_trigger_count == 2
        assert cfg.l3_trigger_count == 5

    def test_config_can_override(self):
        cfg = CUSUMConfig(k=0.01, h=0.08, rolling_window_days=60)
        assert cfg.k == 0.01
        assert cfg.h == 0.08
        assert cfg.rolling_window_days == 60


# ============================================================================
# 4. L1 alert: single factor CUSUM > h
# ============================================================================


class TestL1Alert:
    """L1: Single factor CUSUM exceeds h threshold."""

    def test_cusum_below_h_no_alert(self, db_engine, default_config):
        """CUSUM value 0.03 < h=0.05 => no alert."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        assert not monitor._check_l1("MOM_5D", cusum_value=0.03)

    def test_cusum_above_h_triggers_l1(self, db_engine, default_config):
        """CUSUM value 0.06 >= h=0.05 => L1 alert triggered."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        assert monitor._check_l1("MOM_5D", cusum_value=0.06)

    def test_cusum_exactly_at_h_triggers_l1(self, db_engine, default_config):
        """CUSUM value equals h => L1 alert triggered (boundary)."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        assert monitor._check_l1("MOM_5D", cusum_value=0.05)


# ============================================================================
# 5. L2 suspend: factor triggers >= 2 times in 5-day window
# ============================================================================


class TestL2Suspend:
    """L2: Factor triggered >= 2 times within 5 calendar days => degraded."""

    def test_single_trigger_no_l2(self, db_engine, default_config):
        """One trigger in 5 days => no L2."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        # Simulate one alert today
        trigger_dates = [date.today()]
        assert not monitor._check_l2(trigger_dates)

    def test_two_triggers_outside_window_no_l2(self, db_engine, default_config):
        """Two triggers but more than 5 days apart => no L2."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        today = date.today()
        # 5 days apart (span = 5 >= window=5 => no L2)
        trigger_dates = [today - timedelta(days=5), today]
        assert not monitor._check_l2(trigger_dates)

    def test_two_triggers_within_5_days_l2(self, db_engine, default_config):
        """Two triggers within 5 days => L2 suspend."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        today = date.today()
        trigger_dates = [today - timedelta(days=3), today]
        assert monitor._check_l2(trigger_dates)

    def test_three_triggers_within_5_days_l2(self, db_engine, default_config):
        """Three triggers within 5 days => L2."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        today = date.today()
        trigger_dates = [
            today - timedelta(days=4),
            today - timedelta(days=2),
            today,
        ]
        assert monitor._check_l2(trigger_dates)

    def test_empty_triggers_no_l2(self, db_engine, default_config):
        """No triggers => no L2."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        assert not monitor._check_l2([])


# ============================================================================
# 6. L3 global downgrade: >= 5 factors trigger simultaneously
# ============================================================================


class TestL3GlobalDowngrade:
    """L3: >= 5 factors with CUSUM > h on the same day => critical."""

    def test_four_factors_no_l3(self, db_engine, default_config):
        """4 factors triggering => no L3."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        triggered = {"F1", "F2", "F3", "F4"}
        assert not monitor._check_l3(triggered)

    def test_five_factors_triggers_l3(self, db_engine, default_config):
        """5 factors triggering => L3 critical."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        triggered = {"F1", "F2", "F3", "F4", "F5"}
        assert monitor._check_l3(triggered)

    def test_many_factors_triggers_l3(self, db_engine, default_config):
        """10 factors triggering => L3."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        triggered = {f"F{i}" for i in range(10)}
        assert monitor._check_l3(triggered)

    def test_empty_no_l3(self, db_engine, default_config):
        """No factors => no L3."""
        monitor = CUSUMMonitor(
            session_factory=lambda: Session(db_engine),
            config=default_config,
        )
        assert not monitor._check_l3(set())


# ============================================================================
# 7. DB persistence: FactorHealthDaily records
# ============================================================================


class TestDBPersistence:
    """CUSUMMonitor persists FactorHealthDaily rows."""

    def test_persist_single_factor_record(self, db_engine, default_config):
        """Persist a FactorHealthDaily row and verify it can be read back."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor._persist_record(
            metric_date=date(2025, 1, 15),
            factor_name="MOM_5D",
            daily_ic=0.06,
            rolling_ic_mean=0.05,
            cusum_value=0.005,
            cusum_alert=False,
            consecutive_alerts=0,
        )

        with Session(db_engine) as s:
            row = (
                s.query(FactorHealthDaily)
                .filter_by(
                    metric_date=date(2025, 1, 15),
                    factor_name="MOM_5D",
                )
                .one()
            )
            assert row.daily_ic == pytest.approx(0.06)
            assert row.rolling_ic_mean_90d == pytest.approx(0.05)
            assert row.cusum_value == pytest.approx(0.005)
            assert row.cusum_alert is False
            assert row.consecutive_alerts == 0

    def test_persist_with_alert(self, db_engine, default_config):
        """Persist with cusum_alert=True and consecutive_alerts > 0."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor._persist_record(
            metric_date=date(2025, 2, 1),
            factor_name="PTH",
            daily_ic=0.02,
            rolling_ic_mean=0.04,
            cusum_value=0.07,
            cusum_alert=True,
            consecutive_alerts=3,
        )

        with Session(db_engine) as s:
            row = (
                s.query(FactorHealthDaily)
                .filter_by(
                    metric_date=date(2025, 2, 1),
                    factor_name="PTH",
                )
                .one()
            )
            assert row.cusum_alert is True
            assert row.consecutive_alerts == 3

    def test_upsert_replaces_existing(self, db_engine, default_config):
        """Writing the same (date, factor) twice updates the record."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        dt = date(2025, 3, 10)

        monitor._persist_record(dt, "F1", 0.01, 0.05, 0.001, False, 0)
        # Overwrite
        monitor._persist_record(dt, "F1", 0.08, 0.05, 0.035, True, 1)

        with Session(db_engine) as s:
            rows = s.query(FactorHealthDaily).filter_by(metric_date=dt, factor_name="F1").all()
            assert len(rows) == 1
            assert rows[0].daily_ic == pytest.approx(0.08)
            assert rows[0].cusum_alert is True


# ============================================================================
# 8. Previous CUSUM state retrieval
# ============================================================================


class TestPreviousState:
    """Retrieve S_{t-1} and IC history from the DB."""

    def test_get_previous_cusum_no_history(self, db_engine, default_config):
        """When no prior records exist, returns 0.0."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        result = monitor._get_previous_cusum("NEW_FACTOR", date(2025, 1, 1))
        assert result == 0.0

    def test_get_previous_cusum_returns_latest(self, db_engine, default_config):
        """Returns the cusum_value from the most recent prior record."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        # Insert some history
        monitor._persist_record(date(2025, 1, 1), "MOM_5D", 0.05, 0.04, 0.01, False, 0)
        monitor._persist_record(date(2025, 1, 2), "MOM_5D", 0.06, 0.04, 0.03, True, 1)

        result = monitor._get_previous_cusum("MOM_5D", date(2025, 1, 3))
        assert result == pytest.approx(0.03)

    def test_get_ic_history(self, db_engine, default_config):
        """Returns list of daily IC values for rolling window."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        # Insert history for a factor
        for i, ic in enumerate([0.04, 0.06, 0.05, 0.07, 0.03], start=1):
            monitor._persist_record(date(2025, 1, i), "F1", ic, 0.05, 0.0, False, 0)

        ic_history = monitor._get_ic_history("F1", before_date=date(2025, 1, 6))
        assert len(ic_history) == 5
        assert ic_history == [0.04, 0.06, 0.05, 0.07, 0.03]


# ============================================================================
# 9. Full run pipeline
# ============================================================================


class TestFullRun:
    """Integration test of CUSUMMonitor.run() with daily IC inputs."""

    def test_run_no_prior_history(self, db_engine, default_config):
        """First run with no prior history populates records correctly.

        With no prior IC history, mu_IC defaults to 0.0, so each factor's
        CUSUM value is IC_t - k (clamped to >= 0).
        """

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)

        daily_ics = {
            "MOM_5D": 0.06,
            "PTH": 0.04,
            "MOM_SLOPE": 0.07,
        }
        results = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)

        # Verify records were written for all factors
        with Session(db_engine) as s:
            rows = s.query(FactorHealthDaily).filter_by(metric_date=date(2025, 1, 15)).all()
            assert len(rows) == 3

            by_name = {r.factor_name: r for r in rows}

            # MOM_5D: IC=0.06, mu=0 (no history), delta=0.06-0.0-0.005=0.055
            assert by_name["MOM_5D"].cusum_value == pytest.approx(0.055)
            assert by_name["MOM_5D"].cusum_alert is True  # 0.055 >= h=0.05

            # PTH: IC=0.04, mu=0, delta=0.04-0.0-0.005=0.035
            assert by_name["PTH"].cusum_value == pytest.approx(0.035)
            assert by_name["PTH"].cusum_alert is False  # 0.035 < 0.05

            # MOM_SLOPE: IC=0.07, mu=0, delta=0.07-0.0-0.005=0.065
            assert by_name["MOM_SLOPE"].cusum_value == pytest.approx(0.065)
            assert by_name["MOM_SLOPE"].cusum_alert is True  # 0.065 >= 0.05

        # MOM_5D and MOM_SLOPE triggered L1
        assert results["l1_triggers"] == {"MOM_5D", "MOM_SLOPE"}
        assert results["l2_suspended"] == set()
        assert not results["l3_triggered"]

    def test_run_with_accumulated_cusum_triggers_l1(self, db_engine, default_config):
        """After building up CUSUM across days, L1 triggers.

        Day 1: IC=0.05 for both factors, no history => mu=0
               MOM_5D: S=0.045, PTH: S=0.045
        Day 2: MOM_5D IC=0.10, PTH IC=0.05
               MOM_5D: mu=0.05, S=0.045+0.10-0.05-0.005=0.09 >= 0.05 => L1!
               PTH: mu=0.05, S=0.045+0.05-0.05-0.005=0.04 => no L1
        """

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.005, h=0.05)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        # Day 1: All factors with moderate IC, no alerts
        results1 = monitor.run(
            metric_date=date(2025, 1, 15),
            daily_ics={"MOM_5D": 0.05, "PTH": 0.05},
        )
        assert results1["l1_triggers"] == set()

        # Day 2: MOM_5D gets high IC => CUSUM triggers
        results2 = monitor.run(
            metric_date=date(2025, 1, 16),
            daily_ics={"MOM_5D": 0.10, "PTH": 0.05},
        )
        # MOM_5D triggers L1: S=0.09 >= 0.05
        assert "MOM_5D" in results2["l1_triggers"]
        assert "PTH" not in results2["l1_triggers"]

    def test_run_with_five_triggers_l3(self, db_engine):
        """5 factors triggering L1 simultaneously => L3 global downgrade."""

        def sf():
            return Session(db_engine)

        # k=0, h=0.01: with no prior history mu=0, so IC=0.02 gives S=0.02 >= 0.01 => L1
        config = CUSUMConfig(k=0.0, h=0.01)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        daily_ics = {f"F{i}": 0.02 for i in range(6)}
        results = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)

        # All 6 factors trigger L1 (mu=0, IC-k=0.02 >= h=0.01)
        assert len(results["l1_triggers"]) == 6
        # 6 >= 5 => L3 global downgrade
        assert results["l3_triggered"] is True

    def test_run_with_actual_five_triggers_l3(self, db_engine):
        """5 factors each having CUSUM > h simultaneously => L3."""

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.0, h=0.005)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        # Each factor: IC=0.01, no prior history => mu=0.01
        # CUSUM = 0 + 0.01 - 0.01 - 0 = 0 => no trigger
        # Need to build up history first

        # Day 1: seed history with low IC
        monitor.run(
            metric_date=date(2025, 5, 1),
            daily_ics={f"F{i}": 0.01 for i in range(5)},
        )

        # Day 2: IC is high => CUSUM builds
        # S_prev=0, IC=0.02, mu=(0.01+0.02)/2=0.015 => 0+0.02-0.015-0=0.005 => >= h=0.005 => L1!
        results = monitor.run(
            metric_date=date(2025, 5, 2),
            daily_ics={f"F{i}": 0.02 for i in range(5)},
        )

        assert len(results["l1_triggers"]) == 5
        assert results["l3_triggered"] is True


# ============================================================================
# 10. Alert persistence in alerts table
# ============================================================================


class TestAlertPersistence:
    """Alerts are written to the alerts table."""

    def test_l1_alert_creates_alert_row(self, db_engine, default_config):
        """L1 alert writes a warning row to the alerts table."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor._write_alert(
            rule_name="cusum_l1",
            severity="warning",
            metric_value=0.06,
            notes="factor=MOM_5D cusum=0.06",
        )

        with Session(db_engine) as s:
            alerts = s.query(Alert).filter_by(rule_name="cusum_l1").all()
            assert len(alerts) == 1
            assert alerts[0].severity == "warning"
            assert alerts[0].metric_value == pytest.approx(0.06)
            assert "MOM_5D" in alerts[0].notes

    def test_l2_alert_creates_alert_row(self, db_engine, default_config):
        """L2 alert writes a warning row."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor._write_alert(
            rule_name="cusum_l2",
            severity="warning",
            metric_value=2.0,
            notes="factor=MOM_5D degraded after 2 triggers in 5 days",
        )

        with Session(db_engine) as s:
            alert = s.query(Alert).filter_by(rule_name="cusum_l2").one()
            assert alert.severity == "warning"

    def test_l3_alert_creates_critical_row(self, db_engine, default_config):
        """L3 alert writes a critical row."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor._write_alert(
            rule_name="cusum_l3",
            severity="critical",
            metric_value=5.0,
            notes="global low-activity mode: 5 factors triggered",
        )

        with Session(db_engine) as s:
            alert = s.query(Alert).filter_by(rule_name="cusum_l3").one()
            assert alert.severity == "critical"
            assert "low-activity" in alert.notes


# ============================================================================
# 11. Feishu notification
# ============================================================================


class TestFeishuNotification:
    """Feishu (Lark) notification sending."""

    def test_send_feishu_notification_logs_message(self):
        """_send_feishu_notification logs at warning level with the message."""
        with mock.patch("alphascreener.monitoring.cusum._logger") as mock_logger:
            _send_feishu_notification(
                "CUSUM L1: factor MOM_5D triggered, value=0.06",
                severity="warning",
            )
            mock_logger.warning.assert_called_once()
            # call_args[0] is tuple of positional args: (fmt, severity, message)
            call_args = mock_logger.warning.call_args[0]
            assert len(call_args) >= 3
            assert "MOM_5D" in call_args[2]

    def test_send_feishu_notification_critical(self):
        """Critical alerts are logged with the message content."""
        with mock.patch("alphascreener.monitoring.cusum._logger") as mock_logger:
            _send_feishu_notification(
                "CUSUM L3: global low-activity mode activated",
                severity="critical",
            )
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert len(call_args) >= 3
            assert "L3" in call_args[2]

    def test_feishu_disabled_no_log(self):
        """When feishu_push_enabled is False, no warning is sent.

        Uses a mock Settings object since pydantic-settings fields cannot be
        monkey-patched via mock.patch.object.
        """
        with (
            mock.patch(
                "alphascreener.monitoring.cusum.Settings",
                return_value=mock.MagicMock(feishu_push_enabled=False),
            ),
            mock.patch("alphascreener.monitoring.cusum._logger") as mock_logger,
        ):
            _send_feishu_notification("test message", severity="warning")
            mock_logger.warning.assert_not_called()


# ============================================================================
# 12. Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case handling for CUSUM monitor."""

    def test_empty_daily_ics(self, db_engine, default_config):
        """Empty IC dict returns empty results."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        results = monitor.run(metric_date=date(2025, 1, 15), daily_ics={})
        assert results["l1_triggers"] == set()
        assert results["l2_suspended"] == set()
        assert not results["l3_triggered"]

    def test_none_ic_treated_as_missing(self, db_engine, default_config):
        """None IC values are skipped."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        monitor.run(
            metric_date=date(2025, 1, 15),
            daily_ics={"MOM_5D": None, "PTH": 0.05},
        )
        # MOM_5D skipped (None IC), PTH processed normally
        with Session(db_engine) as s:
            rows = s.query(FactorHealthDaily).filter_by(metric_date=date(2025, 1, 15)).all()
            assert len(rows) == 1  # only PTH
            assert rows[0].factor_name == "PTH"

    def test_trigger_dates_retrieval(self, db_engine, default_config):
        """_get_trigger_dates returns recent trigger dates for a factor."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        today = date.today()

        # Insert records with alert
        monitor._persist_record(today - timedelta(days=3), "F1", 0.06, 0.05, 0.055, True, 1)
        monitor._persist_record(today - timedelta(days=1), "F1", 0.07, 0.05, 0.06, True, 2)

        trigger_dates = monitor._get_trigger_dates("F1", window_days=5, before_date=today)
        assert len(trigger_dates) == 2
        assert (today - timedelta(days=3)) in trigger_dates
        assert (today - timedelta(days=1)) in trigger_dates

    def test_consecutive_alerts_count(self, db_engine, default_config):
        """consecutive_alerts counts recent alert days before the reference date."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        today = date.today()

        # Insert consecutive alert days: today-4, today-3, today-2, today-1
        for i in range(4):
            monitor._persist_record(
                today - timedelta(days=i + 1), "F1", 0.06, 0.05, 0.06, True, i + 1
            )

        # before_date=today => counts dates < today within 5-day window
        # That's today-4, today-3, today-2, today-1 = 4 records
        count = monitor._count_recent_alerts("F1", window_days=5, before_date=today)
        assert count == 4


# ============================================================================
# 13. NaN IC handling (Issue #189)
# ============================================================================


class TestNaNHandling:
    """NaN IC values are guarded against in CUSUM computation and monitor."""

    def test_compute_cusum_with_nan_ic(self):
        """NaN IC input to _compute_cusum returns 0.0 (safe default)."""
        result = _compute_cusum(
            ic_t=float("nan"),
            s_prev=0.05,
            mu_ic=0.05,
            k=0.005,
        )
        assert result == 0.0

    def test_compute_cusum_with_nan_s_prev(self):
        """NaN S_prev input to _compute_cusum returns 0.0 (safe default)."""
        result = _compute_cusum(
            ic_t=0.06,
            s_prev=float("nan"),
            mu_ic=0.05,
            k=0.005,
        )
        assert result == 0.0

    def test_compute_cusum_with_nan_mu_ic(self):
        """NaN mu_ic input to _compute_cusum returns 0.0 (safe default)."""
        result = _compute_cusum(
            ic_t=0.06,
            s_prev=0.05,
            mu_ic=float("nan"),
            k=0.005,
        )
        assert result == 0.0

    def test_run_skips_nan_ic(self, db_engine, default_config):
        """Monitor.run() skips factors with NaN IC, treating them like None."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)
        daily_ics = {"MOM_5D": float("nan"), "PTH": 0.04}
        results = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)

        # PTH processed, MOM_5D skipped
        assert results["records_written"] == 1
        with Session(db_engine) as s:
            rows = s.query(FactorHealthDaily).filter_by(metric_date=date(2025, 1, 15)).all()
            factor_names = {r.factor_name for r in rows}
            assert "MOM_5D" not in factor_names
            assert "PTH" in factor_names


# ============================================================================
# 14. Alert deduplication (Issue #189)
# ============================================================================


class TestAlertDeduplication:
    """L1/L2/L3 alerts are deduplicated to prevent alert storms."""

    def test_l1_dedup_same_factor_same_day(self, db_engine, default_config):
        """L1 alert for same factor on same date only writes once."""

        def sf():
            return Session(db_engine)

        monitor = CUSUMMonitor(session_factory=sf, config=default_config)

        # High IC that triggers L1 for MOM_5D
        daily_ics = {"MOM_5D": 0.10}
        results1 = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)

        assert "MOM_5D" in results1["l1_triggers"]

        # Run again on same date with same factor => L1 should still trigger
        # but should NOT create a duplicate Alert row
        results2 = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)

        assert "MOM_5D" in results2["l1_triggers"]

        with Session(db_engine) as s:
            l1_alerts = s.query(Alert).filter_by(rule_name="cusum_l1").all()
            # Only 1 L1 alert should exist for this factor+date
            assert len(l1_alerts) == 1

    def test_l2_dedup_same_factor(self, db_engine):
        """L2 alert for same factor only writes once per window."""

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.0, h=0.01)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        # Day 1: Establish first L1 trigger for F0
        monitor.run(
            metric_date=date(2025, 1, 15),
            daily_ics={"F0": 0.05, "F1": 0.04},
        )

        # Day 2: F0 triggers L1 again (2nd in window -> L2 fires), F1 also now triggers
        results = monitor.run(
            metric_date=date(2025, 1, 16),
            daily_ics={"F0": 0.05, "F1": 0.05},
        )
        # F0 has 2 triggers in 2 days => L2 should fire for F0
        # F1 has only 1 trigger (first time today) => no L2 for F1 yet
        assert "F0" in results["l2_suspended"]

        # Day 3: F0 triggers again (3rd L1 in 3 days)
        # L2 conditions still met but should NOT create duplicate L2 alert
        monitor.run(
            metric_date=date(2025, 1, 17),
            daily_ics={"F0": 0.05, "F1": 0.05},
        )

        with Session(db_engine) as s:
            l2_alerts = s.query(Alert).filter_by(rule_name="cusum_l2").all()
            # Dedup: at most 1 L2 alert per factor
            by_factor: dict[str, list[Alert]] = {}
            for a in l2_alerts:
                if a.notes and "factor=" in a.notes:
                    factor = a.notes.split("factor=")[1].split(" ")[0]
                    by_factor.setdefault(factor, []).append(a)
            for factor, alerts in by_factor.items():
                assert len(alerts) == 1, f"Factor {factor} has {len(alerts)} L2 alerts"

    def test_l3_dedup_cooldown(self, db_engine):
        """L3 alert has a cooldown period - no duplicate within cooldown."""

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.0, h=0.01, l3_cooldown_hours=24)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        # 6 factors all trigger L1 -> L3 fires
        daily_ics = {f"F{i}": 0.05 for i in range(6)}
        results1 = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)
        assert results1["l3_triggered"] is True

        # Same day, same factors -> L3 should NOT create duplicate alert
        results2 = monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)
        assert results2["l3_triggered"] is True  # still detected, but no new alert

        with Session(db_engine) as s:
            l3_alerts = s.query(Alert).filter_by(rule_name="cusum_l3").all()
            assert len(l3_alerts) == 1

    def test_l3_dedup_different_days_within_cooldown(self, db_engine):
        """L3 on different days within cooldown still suppresses."""

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.0, h=0.01, l3_cooldown_hours=48)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        daily_ics = {f"F{i}": 0.05 for i in range(6)}

        # Day 1
        monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)
        with Session(db_engine) as s:
            l3_alerts = s.query(Alert).filter_by(rule_name="cusum_l3").all()
            assert len(l3_alerts) == 1

        # Day 2 (within 48h cooldown) -> no new L3 alert
        monitor.run(metric_date=date(2025, 1, 16), daily_ics=daily_ics)
        with Session(db_engine) as s:
            l3_alerts = s.query(Alert).filter_by(rule_name="cusum_l3").all()
            assert len(l3_alerts) == 1

    def test_l3_fires_again_after_cooldown(self, db_engine):
        """L3 fires again after cooldown period expires."""

        def sf():
            return Session(db_engine)

        config = CUSUMConfig(k=0.0, h=0.01, l3_cooldown_hours=0)
        monitor = CUSUMMonitor(session_factory=sf, config=config)

        daily_ics = {f"F{i}": 0.05 for i in range(6)}

        # Day 1
        monitor.run(metric_date=date(2025, 1, 15), daily_ics=daily_ics)
        with Session(db_engine) as s:
            l3_alerts = s.query(Alert).filter_by(rule_name="cusum_l3").all()
            assert len(l3_alerts) == 1

        # Day 2 (cooldown=0h means no suppression)
        monitor.run(metric_date=date(2025, 1, 16), daily_ics=daily_ics)
        with Session(db_engine) as s:
            l3_alerts = s.query(Alert).filter_by(rule_name="cusum_l3").all()
            assert len(l3_alerts) == 2


# ============================================================================
# 15. L3 cooldown config (Issue #189)
# ============================================================================


class TestL3CooldownConfig:
    """CUSUMConfig includes l3_cooldown_hours for alert suppression."""

    def test_default_cooldown(self):
        """Default L3 cooldown is 24 hours."""
        cfg = CUSUMConfig()
        assert cfg.l3_cooldown_hours == 24

    def test_custom_cooldown(self):
        """L3 cooldown can be overridden."""
        cfg = CUSUMConfig(l3_cooldown_hours=48)
        assert cfg.l3_cooldown_hours == 48

    def test_zero_cooldown_disables_dedup(self):
        """l3_cooldown_hours=0 means no suppression."""
        cfg = CUSUMConfig(l3_cooldown_hours=0)
        assert cfg.l3_cooldown_hours == 0
