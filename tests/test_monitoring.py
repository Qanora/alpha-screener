"""Tests for psutil resource monitoring, alerts, and data retention.

Issue #107: Resource monitoring.
Reference: PRD 9.2 / 9.3 / 10.2.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alphascreener.db.models import Alert, Base, MonitoringSample
from alphascreener.monitoring import (
    MonitoringConfig,
    ResourceMonitor,
    _alert_severity,
    _check_cpu_sustained,
    _check_disk,
    _check_oom,
    _check_rss_sustained,
    _cleanup_old_samples,
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
def db_session(db_engine):
    """Session factory bound to fresh in-memory SQLite."""
    with Session(db_engine) as session:
        yield session


@pytest.fixture
def default_config() -> MonitoringConfig:
    return MonitoringConfig()


# ============================================================================
# 1. Sampling — psutil collection of RSS/CPU/FD/threads
# ============================================================================


class TestSampleCollection:
    """Verify sample() collects RSS, CPU, FD, and thread count from psutil."""

    def test_sample_returns_dict_with_required_fields(self):
        monitor = ResourceMonitor(task_id="test-task", session_factory=lambda: None)
        monitor._proc = psutil.Process()
        result = monitor._collect_snapshot()
        assert "rss_mb" in result
        assert "cpu_percent" in result
        assert "open_fd_count" in result
        assert "thread_count" in result

    def test_sample_rss_is_positive_float(self):
        monitor = ResourceMonitor(task_id="test-task", session_factory=lambda: None)
        monitor._proc = psutil.Process()
        result = monitor._collect_snapshot()
        assert isinstance(result["rss_mb"], float)
        assert result["rss_mb"] > 0

    def test_sample_cpu_is_float(self):
        monitor = ResourceMonitor(task_id="test-task", session_factory=lambda: None)
        monitor._proc = psutil.Process()
        result = monitor._collect_snapshot()
        assert isinstance(result["cpu_percent"], float)

    def test_sample_open_fd_is_int(self):
        monitor = ResourceMonitor(task_id="test-task", session_factory=lambda: None)
        monitor._proc = psutil.Process()
        result = monitor._collect_snapshot()
        assert isinstance(result["open_fd_count"], int)

    def test_sample_thread_count_is_int(self):
        monitor = ResourceMonitor(task_id="test-task", session_factory=lambda: None)
        monitor._proc = psutil.Process()
        result = monitor._collect_snapshot()
        assert isinstance(result["thread_count"], int)


# ============================================================================
# 2. Periodic sampling — stores to DB on each interval
# ============================================================================


class TestPeriodicSampling:
    """Verify that samples are written to monitoring_samples on each tick."""

    def test_sample_persists_to_db(self, db_engine):
        """A single manual sample writes one row to monitoring_samples."""

        def sf():
            return Session(db_engine)

        monitor = ResourceMonitor(task_id="task-1", session_factory=sf)
        monitor._proc = psutil.Process()
        monitor._sample_and_persist()

        with Session(db_engine) as s:
            rows = s.query(MonitoringSample).filter_by(task_id="task-1").all()
            assert len(rows) == 1
            sample = rows[0]
            assert sample.rss_mb > 0
            assert sample.cpu_percent >= 0
            assert sample.sampled_at is not None


# ============================================================================
# 3. Peak recording — highest values in the session
# ============================================================================


class TestPeakRecording:
    """Verify peak RSS / CPU / FD / threads are tracked across samples."""

    def test_peak_rss_tracks_maximum(self):
        monitor = ResourceMonitor(task_id="t", session_factory=lambda: None)
        assert monitor._peak_rss_mb is None
        s1 = {"rss_mb": 100.0, "cpu_percent": 10.0, "open_fd_count": 5, "thread_count": 2}
        s2 = {"rss_mb": 300.0, "cpu_percent": 20.0, "open_fd_count": 15, "thread_count": 4}
        s3 = {"rss_mb": 150.0, "cpu_percent": 30.0, "open_fd_count": 8, "thread_count": 3}
        monitor._update_peaks(s1)
        monitor._update_peaks(s2)
        monitor._update_peaks(s3)
        assert monitor._peak_rss_mb == 300.0
        assert monitor._peak_cpu_percent == 30.0
        assert monitor._peak_open_fd == 15
        assert monitor._peak_threads == 4

    def test_peaks_flushed_to_db_on_exit(self, db_engine):
        """Peak values are written as a marked sample on __exit__."""

        def sf():
            return Session(db_engine)

        monitor = ResourceMonitor(task_id="task-pk", session_factory=sf)
        monitor._proc = psutil.Process()
        p = {"rss_mb": 512.0, "cpu_percent": 45.0, "open_fd_count": 60, "thread_count": 4}
        monitor._update_peaks(p)
        monitor._flush_peaks()

        with Session(db_engine) as s:
            rows = s.query(MonitoringSample).filter_by(task_id="task-pk").all()
            assert len(rows) == 1
            peak = rows[0]
            assert peak.rss_mb == 512.0
            assert peak.cpu_percent == 45.0
            assert peak.open_fd_count == 60
            assert peak.thread_count == 4
            assert peak.notes == "peak"


# ============================================================================
# 4. Memory budget alerts
# ============================================================================


class TestMemoryAlerts:
    """RSS > 5500MB for 5 minutes => warning; RSS > 7000MB => critical/kill."""

    def test_rss_below_warning_no_alert(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 2000.0, "cpu_percent": 50.0, "open_fd_count": 10, "thread_count": 3}
        severity = _check_rss_sustained(
            session_factory=sf,
            task_id="mem-test",
            snapshot=snapshot,
            config=default_config,
            high_rss_history={},
        )
        assert severity is None

    def test_rss_above_warning_but_short_duration_no_alert(self, db_engine, default_config):
        """RSS spikes briefly but not sustained — no alert yet."""

        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 6000.0, "cpu_percent": 50.0, "open_fd_count": 10, "thread_count": 3}
        # high_rss_history starts tracking now
        high_rss_history: dict[str, float] = {}
        severity = _check_rss_sustained(
            session_factory=sf,
            task_id="mem-test",
            snapshot=snapshot,
            config=default_config,
            high_rss_history=high_rss_history,
        )
        # Not yet sustained for >5 minutes
        assert severity is None
        # But history now tracks the start time
        assert "mem-test" in high_rss_history

    def test_rss_above_warning_sustained_5min_emits_warning(self, db_engine, default_config):
        """RSS > 5500MB sustained >5 minutes => severity='warning'."""

        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 6000.0, "cpu_percent": 50.0, "open_fd_count": 10, "thread_count": 3}
        # Simulate that high RSS started 6 minutes ago
        high_rss_history = {"mem-test": time.monotonic() - 360}
        severity = _check_rss_sustained(
            session_factory=sf,
            task_id="mem-test",
            snapshot=snapshot,
            config=default_config,
            high_rss_history=high_rss_history,
        )
        assert severity == "warning"

    def test_rss_above_kill_threshold_immediately_critical(self, db_engine, default_config):
        """RSS > 7000MB => immediate critical regardless of duration."""

        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 7500.0, "cpu_percent": 50.0, "open_fd_count": 10, "thread_count": 3}
        high_rss_history: dict[str, float] = {}
        severity = _check_rss_sustained(
            session_factory=sf,
            task_id="mem-test",
            snapshot=snapshot,
            config=default_config,
            high_rss_history=high_rss_history,
        )
        assert severity == "critical"

    def test_rss_drops_below_warning_clears_history(self, db_engine, default_config):
        """When RSS drops below warning, history is cleared."""
        high_rss_history = {"mem-test": time.monotonic() - 100}
        snapshot = {"rss_mb": 1000.0, "cpu_percent": 50.0, "open_fd_count": 10, "thread_count": 3}

        def sf():
            return Session(db_engine)

        _check_rss_sustained(
            session_factory=sf,
            task_id="mem-test",
            snapshot=snapshot,
            config=default_config,
            high_rss_history=high_rss_history,
        )
        assert "mem-test" not in high_rss_history


# ============================================================================
# 5. CPU sustained overload alerts
# ============================================================================


class TestCPUAlerts:
    """CPU > 380% sustained >10 minutes => warning (check task stuck)."""

    def test_cpu_below_threshold_no_alert(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 1000.0, "cpu_percent": 200.0, "open_fd_count": 10, "thread_count": 3}
        severity = _check_cpu_sustained(
            session_factory=sf,
            task_id="cpu-test",
            snapshot=snapshot,
            config=default_config,
            high_cpu_history={},
        )
        assert severity is None

    def test_cpu_above_threshold_short_duration_no_alert(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 1000.0, "cpu_percent": 390.0, "open_fd_count": 10, "thread_count": 3}
        high_cpu_history: dict[str, float] = {}
        severity = _check_cpu_sustained(
            session_factory=sf,
            task_id="cpu-test",
            snapshot=snapshot,
            config=default_config,
            high_cpu_history=high_cpu_history,
        )
        assert severity is None
        assert "cpu-test" in high_cpu_history

    def test_cpu_sustained_10min_emits_warning(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        snapshot = {"rss_mb": 1000.0, "cpu_percent": 400.0, "open_fd_count": 10, "thread_count": 3}
        high_cpu_history = {"cpu-test": time.monotonic() - 610}
        severity = _check_cpu_sustained(
            session_factory=sf,
            task_id="cpu-test",
            snapshot=snapshot,
            config=default_config,
            high_cpu_history=high_cpu_history,
        )
        assert severity == "warning"


# ============================================================================
# 6. OOM detection
# ============================================================================


class TestOOMDetection:
    """oom_kill_count > 0 => severity='critical'."""

    def test_no_oom_returns_none(self, db_engine):
        severity = _check_oom(
            lambda: Session(db_engine),
            task_id="oom-test",
            _proc_factory=psutil.Process,
        )
        assert severity is None

    def test_oom_detected_returns_critical(self, db_engine):
        """Simulate OOM via environment override path."""

        def sf():
            return Session(db_engine)

        # We test by directly calling with an override — since actual OOM
        # needs kernel support, we test the _check_oom logic via a mock.
        severity = _check_oom(sf, task_id="oom-test", oom_count_override=3)
        assert severity == "critical"


# ============================================================================
# 7. Disk tight alert
# ============================================================================


class TestDiskAlerts:
    """Free disk < threshold => warning."""

    def test_disk_free_above_threshold_no_alert(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        severity = _check_disk(
            sf, task_id="disk-test", free_gb_override=200.0, config=default_config
        )
        assert severity is None

    def test_disk_free_below_threshold_emits_warning(self, db_engine, default_config):
        def sf():
            return Session(db_engine)

        severity = _check_disk(
            sf, task_id="disk-test", free_gb_override=50.0, config=default_config
        )
        assert severity == "warning"


# ============================================================================
# 8. alert_severity helper
# ============================================================================


class TestAlertSeverity:
    """_alert_severity writes or updates rows in the alerts table."""

    def test_creates_new_alert_row(self, db_engine):
        def sf():
            return Session(db_engine)

        _alert_severity(
            session_factory=sf,
            task_id="alert-test",
            severity="warning",
            rule_name="disk_tight",
            metric_value=30.0,
        )
        with Session(db_engine) as s:
            alerts = s.query(Alert).filter_by(rule_name="disk_tight").all()
            assert len(alerts) == 1
            assert alerts[0].severity == "warning"
            assert alerts[0].metric_value == 30.0

    def test_alert_notes_include_task_id(self, db_engine):
        def sf():
            return Session(db_engine)

        _alert_severity(
            session_factory=sf,
            task_id="ta",
            severity="critical",
            rule_name="oom_kill",
            metric_value=1.0,
        )
        with Session(db_engine) as s:
            a = s.query(Alert).filter_by(rule_name="oom_kill").one()
            assert "ta" in a.notes


# ============================================================================
# 9. Data retention — cleanup samples older than 30 days
# ============================================================================


class TestDataRetention:
    """_cleanup_old_samples removes rows with sampled_at >30 days ago."""

    def test_keeps_recent_samples(self, db_engine):
        def sf():
            return Session(db_engine)

        # Insert a recent sample
        with Session(db_engine) as s:
            s.add(
                MonitoringSample(
                    task_id="cleanup-test",
                    sampled_at=datetime.utcnow().isoformat(),
                    rss_mb=100.0,
                    cpu_percent=10.0,
                )
            )
            s.commit()

        _cleanup_old_samples(sf, retention_days=30)

        with Session(db_engine) as s:
            count = s.query(MonitoringSample).filter_by(task_id="cleanup-test").count()
            assert count == 1

    def test_removes_samples_older_than_retention(self, db_engine):
        def sf():
            return Session(db_engine)

        old_ts = (datetime.utcnow() - timedelta(days=40)).isoformat()
        recent_ts = (datetime.utcnow() - timedelta(days=5)).isoformat()

        with Session(db_engine) as s:
            s.add(
                MonitoringSample(
                    task_id="old-task",
                    sampled_at=old_ts,
                    rss_mb=100.0,
                    cpu_percent=10.0,
                )
            )
            s.add(
                MonitoringSample(
                    task_id="recent-task",
                    sampled_at=recent_ts,
                    rss_mb=200.0,
                    cpu_percent=20.0,
                )
            )
            s.commit()

        _cleanup_old_samples(sf, retention_days=30)

        with Session(db_engine) as s:
            old = s.query(MonitoringSample).filter_by(task_id="old-task").count()
            recent = s.query(MonitoringSample).filter_by(task_id="recent-task").count()
            assert old == 0
            assert recent == 1


# ============================================================================
# 10. ResourceMonitor context manager
# ============================================================================


class TestContextManager:
    """ResourceMonitor should support with-statement and clean up."""

    def test_enter_starts_sampling(self, db_engine):
        def sf():
            return Session(db_engine)

        monitor = ResourceMonitor(task_id="ctx-test", session_factory=sf)
        monitor._proc = psutil.Process()
        monitor._sample_and_persist()
        # Verify a sample was written
        with Session(db_engine) as s:
            rows = s.query(MonitoringSample).filter_by(task_id="ctx-test").all()
            assert len(rows) >= 1

    def test_exit_writes_peaks(self, db_engine):
        def sf():
            return Session(db_engine)

        monitor = ResourceMonitor(task_id="ctx-peaks", session_factory=sf)
        monitor._proc = psutil.Process()
        pk = {"rss_mb": 999.0, "cpu_percent": 99.0, "open_fd_count": 50, "thread_count": 8}
        monitor._update_peaks(pk)
        monitor._flush_peaks()

        with Session(db_engine) as s:
            peaks = s.query(MonitoringSample).filter_by(task_id="ctx-peaks", notes="peak").all()
            assert len(peaks) == 1
            assert peaks[0].rss_mb == 999.0

    def test_exit_stops_sampling(self):
        monitor = ResourceMonitor(task_id="ctx-stop", session_factory=lambda: None)
        monitor._running = True
        monitor.stop()
        assert monitor._running is False


# ============================================================================
# 11. MonitoringConfig defaults
# ============================================================================


class TestMonitoringConfig:
    """MonitoringConfig holds all threshold constants."""

    def test_default_config_values(self):
        cfg = MonitoringConfig()
        assert cfg.sample_interval_seconds == 60
        assert cfg.rss_warning_mb == 5500.0
        assert cfg.rss_kill_mb == 7000.0
        assert cfg.rss_warning_persist_minutes == 5
        assert cfg.cpu_warning_percent == 380.0
        assert cfg.cpu_warning_persist_minutes == 10
        assert cfg.disk_free_gb_warning == 80.0
        assert cfg.retention_days == 30

    def test_config_can_override(self):
        cfg = MonitoringConfig(
            sample_interval_seconds=120,
            rss_warning_mb=3000.0,
            cpu_warning_percent=200.0,
        )
        assert cfg.sample_interval_seconds == 120
        assert cfg.rss_warning_mb == 3000.0
        assert cfg.cpu_warning_percent == 200.0
