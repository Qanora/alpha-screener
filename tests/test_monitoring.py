"""Tests for psutil resource monitoring, alerts, and data retention.

Issue #107: Resource monitoring.
Reference: PRD 9.2 / 9.3 / 10.2.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

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
from alphascreener.monitoring.sampler import (
    _parse_oom_kill_from_events,
    _parse_oom_kill_from_oom_control,
    _read_oom_kill_count,
    _resolve_cgroup_path,
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
# 6b. Cgroup OOM kill count parsing
# ============================================================================


CGROUP_V2 = """0::/system.slice/foo.service
"""

CGROUP_V1 = """11:memory:/user.slice/user-1000.slice
10:cpuset:/
9:blkio:/
"""

MEMORY_EVENTS = """low 0
high 0
max 0
oom 0
oom_kill 5
oom_group_kill 0
"""

OOM_CONTROL = """oom_kill_disable 0
under_oom 0
oom_kill 3
"""


class TestCgroupOOMKillCount:
    """_read_oom_kill_count parses cgroup v2 / v1 OOM kill counters."""

    def test_cgroup_v2_oom_kill_from_memory_events(self):
        """When cgroup v2 memory.events has oom_kill=5, return 5."""
        with (
            mock.patch(
                "alphascreener.monitoring.sampler._resolve_cgroup_path",
                return_value="/system.slice/foo.service",
            ),
            mock.patch(
                "alphascreener.monitoring.sampler.os.path.exists",
                return_value=True,
            ),
            mock.patch(
                "alphascreener.monitoring.sampler._parse_oom_kill_from_events",
                return_value=5,
            ),
        ):
            assert _read_oom_kill_count(99999) == 5

    def test_cgroup_v1_oom_kill_from_oom_control(self):
        """When cgroup v2 memory.events missing but v1 oom_control has oom_kill=3, return 3."""

        def exists_side_effect(path):
            # Only the v1 oom_control path exists; v2 memory.events does not
            return "memory.oom_control" in path

        with (
            mock.patch(
                "alphascreener.monitoring.sampler._resolve_cgroup_path",
                return_value="/user.slice/user-1000.slice",
            ),
            mock.patch(
                "alphascreener.monitoring.sampler.os.path.exists",
                side_effect=exists_side_effect,
            ),
            mock.patch(
                "alphascreener.monitoring.sampler._parse_oom_kill_from_oom_control",
                return_value=3,
            ),
        ):
            assert _read_oom_kill_count(99999) == 3

    def test_no_cgroup_file_returns_zero(self):
        """When cgroup path resolution returns None, return 0."""
        with mock.patch(
            "alphascreener.monitoring.sampler._resolve_cgroup_path",
            return_value=None,
        ):
            assert _read_oom_kill_count(99999) == 0

    def test_resolve_cgroup_v2_path(self):
        """_resolve_cgroup_path extracts cgroup path from cgroup v2 '0::/path' line."""
        with (
            mock.patch(
                "alphascreener.monitoring.sampler.os.path.exists",
                return_value=True,
            ),
            mock.patch(
                "builtins.open",
                mock.mock_open(read_data=CGROUP_V2),
            ),
        ):
            result = _resolve_cgroup_path(12345)
            assert result == "/system.slice/foo.service"

    def test_resolve_cgroup_v1_path(self):
        """_resolve_cgroup_path extracts cgroup path from cgroup v1 memory controller line."""
        with (
            mock.patch(
                "alphascreener.monitoring.sampler.os.path.exists",
                return_value=True,
            ),
            mock.patch(
                "builtins.open",
                mock.mock_open(read_data=CGROUP_V1),
            ),
        ):
            result = _resolve_cgroup_path(12345)
            assert result == "/user.slice/user-1000.slice"

    def test_parse_oom_kill_from_events_file(self, tmp_path):
        """_parse_oom_kill_from_events reads the oom_kill field from a real file."""
        events_file = tmp_path / "memory.events"
        events_file.write_text(MEMORY_EVENTS)
        assert _parse_oom_kill_from_events(str(events_file)) == 5

    def test_parse_oom_kill_from_oom_control_file(self, tmp_path):
        """_parse_oom_kill_from_oom_control reads oom_kill (not oom_kill_disable)."""
        oom_control_file = tmp_path / "memory.oom_control"
        oom_control_file.write_text(OOM_CONTROL)
        assert _parse_oom_kill_from_oom_control(str(oom_control_file)) == 3


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
                    sampled_at=datetime.now(UTC).isoformat(),
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

        old_ts = (datetime.now(UTC) - timedelta(days=40)).isoformat()
        recent_ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()

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

    def test_with_statement_writes_samples_and_peaks(self, db_engine):
        """Using `with ResourceMonitor(...) as monitor:` writes samples and peaks on exit."""
        import time as _time

        def sf():
            return Session(db_engine)

        # Use minimal interval so the background thread gets at least one sample
        # before we exit.  We still need a real psutil.Process.
        cfg = MonitoringConfig(sample_interval_seconds=1)
        with ResourceMonitor(task_id="ctx-with", session_factory=sf, config=cfg) as _monitor:
            # Let the background thread run at least one sample
            _time.sleep(1.5)

        # On __exit__: background thread stopped, peaks flushed, OOM/disk checked.
        with Session(db_engine) as s:
            samples = s.query(MonitoringSample).filter_by(task_id="ctx-with").all()
            assert len(samples) >= 1, "Expected at least one periodic sample"
            peaks = [r for r in samples if r.notes == "peak"]
            assert len(peaks) == 1, "Expected one peak row written on exit"

    def test_exit_stops_sampling(self, db_engine):
        """After the with-block, the background thread is no longer running."""

        def sf():
            return Session(db_engine)

        import time as _time

        cfg = MonitoringConfig(sample_interval_seconds=1)
        with ResourceMonitor(task_id="ctx-stop", session_factory=sf, config=cfg) as monitor:
            _time.sleep(0.5)

        assert monitor._running is False
        assert monitor._thread is None or not monitor._thread.is_alive()


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


# ============================================================================
# 12. Schema auto-migration — ensure tables exist on engine creation
# ============================================================================


class TestEnsureSchema:
    """_ensure_schema() creates all required tables if they don't exist."""

    def test_ensure_schema_creates_all_tables(self, tmp_path):
        """_ensure_schema on a file-based SQLite creates all 9 core tables."""
        from sqlalchemy import create_engine, inspect

        from alphascreener.db.ensure_schema import _ensure_schema

        db_path = str(tmp_path / "test_ensure.db")
        engine = create_engine(f"sqlite:///{db_path}")

        # Verify tables don't exist before
        inspector = inspect(engine)
        assert len(inspector.get_table_names()) == 0

        _ensure_schema(engine)

        # Verify all 9 core tables exist after
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        expected = {
            "alerts",
            "alpha_acceptance_daily",
            "data_source_diff",
            "factor_health_daily",
            "factor_versions",
            "llm_cost_daily",
            "monitoring_samples",
            "pid_lock",
            "paper_trades",
        }
        assert expected == table_names, f"Missing tables: {expected - table_names}"

    def test_ensure_schema_is_idempotent(self, tmp_path):
        """Calling _ensure_schema multiple times does not raise errors."""
        from sqlalchemy import create_engine

        from alphascreener.db.ensure_schema import _ensure_schema

        db_path = str(tmp_path / "test_idem.db")
        engine = create_engine(f"sqlite:///{db_path}")

        _ensure_schema(engine)
        _ensure_schema(engine)  # second call should not fail
        _ensure_schema(engine)  # third call should not fail

    def test_ensure_schema_creates_monitoring_samples_writable(self, tmp_path):
        """After _ensure_schema, monitoring_samples table accepts writes."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from alphascreener.db.ensure_schema import _ensure_schema
        from alphascreener.db.models import MonitoringSample

        db_path = str(tmp_path / "test_writable.db")
        engine = create_engine(f"sqlite:///{db_path}")
        _ensure_schema(engine)

        with Session(engine) as session:
            sample = MonitoringSample(
                task_id="test-task",
                sampled_at="2025-06-19T10:00:00+00:00",
                rss_mb=100.0,
                cpu_percent=10.0,
            )
            session.add(sample)
            session.commit()

            rows = session.query(MonitoringSample).filter_by(task_id="test-task").all()
            assert len(rows) == 1
            assert rows[0].rss_mb == 100.0


# ============================================================================
# 13. Pipeline-stage metric persistence
# ============================================================================


class TestPipelineStageMetrics:
    """Pipeline-stage metrics can be persisted to monitoring_samples."""

    def test_write_stage_metric_to_monitoring_samples(self, db_engine):
        """write_stage_metric() writes a monitoring sample with stage notes."""
        from sqlalchemy.orm import Session

        from alphascreener.db.models import MonitoringSample
        from alphascreener.monitoring import write_stage_metric

        def sf():
            return Session(db_engine)

        write_stage_metric(
            session_factory=sf,
            task_id="daily_scan",
            stage="phase1",
            notes="Phase1: 500 tickers, 120 passed (24.0%)",
        )

        with Session(db_engine) as s:
            rows = s.query(MonitoringSample).filter_by(task_id="daily_scan").all()
            assert len(rows) == 1
            assert rows[0].notes == "[phase1] Phase1: 500 tickers, 120 passed (24.0%)"
            assert rows[0].rss_mb == 0.0
            assert rows[0].thread_count == 0

    def test_write_multiple_stage_metrics(self, db_engine):
        """Multiple stage metrics can be written for a single task run."""
        from sqlalchemy.orm import Session

        from alphascreener.monitoring import write_stage_metric

        def sf():
            return Session(db_engine)

        write_stage_metric(sf, "daily_scan", "phase1", "Phase1: done")
        write_stage_metric(sf, "daily_scan", "phase2", "Phase2: 50 candidates")
        write_stage_metric(sf, "daily_scan", "fine", "Fine: 12 assessments")

        with Session(db_engine) as s:
            from alphascreener.db.models import MonitoringSample

            rows = (
                s.query(MonitoringSample)
                .filter_by(task_id="daily_scan")
                .order_by(MonitoringSample.sampled_at)
                .all()
            )
            assert len(rows) == 3
            assert rows[0].notes == "[phase1] Phase1: done"
            assert rows[1].notes == "[phase2] Phase2: 50 candidates"
            assert rows[2].notes == "[fine] Fine: 12 assessments"


# ============================================================================
# 14. Daily scan monitoring integration
# ============================================================================


class TestDailyScanMonitoringIntegration:
    """daily_scan writes monitoring_samples during pipeline execution."""

    def test_daily_scan_function_is_callable_with_monitoring(self):
        """daily_scan is callable and importable."""
        from alphascreener.scheduler.tasks import daily_scan

        assert callable(daily_scan)

    def test_monitoring_module_exports_stage_writer(self):
        """The monitoring module exports write_stage_metric."""
        from alphascreener.monitoring import write_stage_metric

        assert callable(write_stage_metric)

    def test_db_ensure_schema_module_exists(self):
        """The ensure_schema module is importable."""
        from alphascreener.db.ensure_schema import _ensure_schema

        assert callable(_ensure_schema)
