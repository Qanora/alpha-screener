"""psutil resource monitoring: sampling, alert thresholds, and data retention.

Issue #107: Resource monitoring.
Reference: PRD 9.2 / 9.3 / 10.2.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import psutil
from sqlalchemy.orm import Session

from alphascreener.db.models import Alert, MonitoringSample

logger = logging.getLogger("monitoring")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitoringConfig:
    """Threshold constants for resource monitoring.

    All values are defaults; override via constructor kwargs.
    """

    sample_interval_seconds: int = 60
    rss_warning_mb: float = 5500.0
    rss_kill_mb: float = 7000.0
    rss_warning_persist_minutes: int = 5
    cpu_warning_percent: float = 380.0
    cpu_warning_persist_minutes: int = 10
    disk_free_gb_warning: float = 80.0
    retention_days: int = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SessionFactory = Callable[[], Session]

# Per-task OOM kill count tracking — cgroup counters are cumulative so we
# only alert on delta increases.
_last_oom_count: dict[str, int] = {}


def _utcnow_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(UTC).isoformat()


def _with_session(
    session_factory: _SessionFactory,
    fn: Callable[[Session], T],
    *,
    rollback_value: T | None = None,
    operation: str = "unknown",
) -> T | None:
    """Execute *fn* inside a new session, commit, and close.

    On failure the session is rolled back and ``rollback_value`` is returned.
    The *operation* string is used in error logs to identify the failing write
    path (e.g. ``"monitoring_samples.insert"``, ``"alerts.insert"``).
    """
    session = session_factory()
    try:
        result = fn(session)
        session.commit()
        return result
    except Exception:
        logger.error(
            "Monitoring write failed [%s] — data will NOT be persisted. "
            "Check DB schema, disk space, and engine connectivity.",
            operation,
            exc_info=True,
        )
        session.rollback()
        return rollback_value
    finally:
        session.close()


def _now_monotonic() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Context manager that samples psutil metrics periodically and emits alerts.

    Usage::

        def make_session():
            return Session(engine)

        with ResourceMonitor(task_id="daily_scan", session_factory=make_session):
            # ... long-running task ...
            pass
        # On exit: peaks are flushed, background thread is stopped.
    """

    def __init__(
        self,
        task_id: str,
        session_factory: _SessionFactory,
        config: MonitoringConfig | None = None,
    ) -> None:
        self.task_id = task_id
        self._session_factory = session_factory
        self.config = config or MonitoringConfig()

        # background thread control
        self._running = False
        self._thread: threading.Thread | None = None

        # peak tracking
        self._peak_rss_mb: float | None = None
        self._peak_cpu_percent: float | None = None
        self._peak_open_fd: int | None = None
        self._peak_threads: int | None = None

        # sustained-threshold histories (keyed by task_id)
        self._high_rss_since: dict[str, float] = {}
        self._high_cpu_since: dict[str, float] = {}

        self._proc: psutil.Process | None = None

    # -- context manager protocol --------------------------------------------

    def __enter__(self) -> ResourceMonitor:
        self._proc = psutil.Process()
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.config.sample_interval_seconds + 5)
        self._flush_peaks()
        _check_oom(self._session_factory, self.task_id)
        _check_disk(self._session_factory, self.task_id, config=self.config)

    # -- sampling ------------------------------------------------------------

    def _sampling_loop(self) -> None:
        while self._running:
            self._sample_and_persist()
            time.sleep(self.config.sample_interval_seconds)

    def _sample_and_persist(self) -> None:
        snapshot = self._collect_snapshot()
        self._update_peaks(snapshot)
        self._evaluate_alerts(snapshot)
        self._persist_sample(snapshot)

    def _collect_snapshot(self) -> dict[str, Any]:
        if self._proc is None:
            return {"rss_mb": 0.0, "cpu_percent": 0.0, "open_fd_count": 0, "thread_count": 0}

        proc = self._proc
        with proc.oneshot():
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            cpu = proc.cpu_percent(interval=0.1)
            try:
                fd_count = proc.num_fds()
            except (psutil.AccessDenied, AttributeError):
                fd_count = -1
            thread_count = proc.num_threads()

        return {
            "rss_mb": round(rss_mb, 2),
            "cpu_percent": round(cpu, 2),
            "open_fd_count": fd_count,
            "thread_count": thread_count,
        }

    def _persist_sample(self, snapshot: dict[str, Any]) -> None:
        def _write(session: Session) -> None:
            session.add(
                MonitoringSample(
                    task_id=self.task_id,
                    sampled_at=_utcnow_iso(),
                    rss_mb=snapshot["rss_mb"],
                    cpu_percent=snapshot["cpu_percent"],
                    open_fd_count=snapshot["open_fd_count"],
                    thread_count=snapshot["thread_count"],
                )
            )

        _with_session(self._session_factory, _write, operation="monitoring_samples.insert")

    # -- peaks ---------------------------------------------------------------

    def _update_peaks(self, snapshot: dict[str, Any]) -> None:
        rss = snapshot["rss_mb"]
        cpu = snapshot["cpu_percent"]
        fd = snapshot["open_fd_count"]
        tc = snapshot["thread_count"]

        if self._peak_rss_mb is None or rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        if self._peak_cpu_percent is None or cpu > self._peak_cpu_percent:
            self._peak_cpu_percent = cpu
        if fd >= 0 and (self._peak_open_fd is None or fd > self._peak_open_fd):
            self._peak_open_fd = fd
        if self._peak_threads is None or tc > self._peak_threads:
            self._peak_threads = tc

    def _flush_peaks(self) -> None:
        if self._peak_rss_mb is None:
            return

        task_id = self.task_id
        rss = self._peak_rss_mb
        cpu = self._peak_cpu_percent or 0.0
        fd = self._peak_open_fd
        tc = self._peak_threads

        def _write(session: Session) -> None:
            session.add(
                MonitoringSample(
                    task_id=task_id,
                    sampled_at=_utcnow_iso(),
                    rss_mb=rss,
                    cpu_percent=cpu,
                    open_fd_count=fd,
                    thread_count=tc,
                    notes="peak",
                )
            )

        _with_session(self._session_factory, _write, operation="monitoring_samples.insert_peak")

    # -- alert evaluation ----------------------------------------------------

    def _evaluate_alerts(self, snapshot: dict[str, Any]) -> None:
        _check_rss_sustained(
            session_factory=self._session_factory,
            task_id=self.task_id,
            snapshot=snapshot,
            config=self.config,
            high_rss_history=self._high_rss_since,
        )
        _check_cpu_sustained(
            session_factory=self._session_factory,
            task_id=self.task_id,
            snapshot=snapshot,
            config=self.config,
            high_cpu_history=self._high_cpu_since,
        )

    # -- public helpers -------------------------------------------------------

    def stop(self) -> None:
        """Stop the background sampling thread (idempotent)."""
        self._running = False


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------


def _alert_severity(
    session_factory: _SessionFactory,
    task_id: str,
    severity: str,
    rule_name: str,
    metric_value: float | None = None,
) -> None:
    def _write(session: Session) -> None:
        session.add(
            Alert(
                triggered_at=_utcnow_iso(),
                severity=severity,
                rule_name=rule_name,
                metric_value=metric_value,
                notes=f"task_id={task_id}",
            )
        )

    _with_session(session_factory, _write, operation="alerts.insert")


def _check_sustained(
    session_factory: _SessionFactory,
    task_id: str,
    metric_value: float,
    threshold: float,
    persist_seconds: float,
    rule_name: str,
    severity: str,
    history: dict[str, float],
    *,
    rule_name_immediate: str | None = None,
    threshold_immediate: float | None = None,
    severity_immediate: str = "critical",
) -> str | None:
    """Generic sustained-threshold check with optional immediate-critical tier.

    - If ``threshold_immediate`` is set and *metric_value* exceeds it,
      fire an alert immediately and return ``severity_immediate``.
    - If *metric_value* exceeds *threshold* for ``persist_seconds``,
      fire an alert with the given ``severity`` and ``rule_name``.
    - When *metric_value* drops below *threshold*, the tracking history is cleared.
    """
    # Immediate critical tier
    if threshold_immediate is not None and metric_value > threshold_immediate:
        _alert_severity(
            session_factory,
            task_id,
            severity_immediate,
            rule_name_immediate or rule_name,
            metric_value,
        )
        return severity_immediate

    # Sustained warning tier
    if metric_value > threshold:
        if task_id not in history:
            history[task_id] = _now_monotonic()
            return None

        elapsed = _now_monotonic() - history[task_id]
        if elapsed >= persist_seconds:
            _alert_severity(session_factory, task_id, severity, rule_name, metric_value)
            history.pop(task_id, None)
            return severity
        return None

    history.pop(task_id, None)
    return None


def _check_rss_sustained(
    session_factory: _SessionFactory,
    task_id: str,
    snapshot: dict[str, Any],
    config: MonitoringConfig,
    high_rss_history: dict[str, float],
) -> str | None:
    return _check_sustained(
        session_factory=session_factory,
        task_id=task_id,
        metric_value=snapshot["rss_mb"],
        threshold=config.rss_warning_mb,
        persist_seconds=config.rss_warning_persist_minutes * 60,
        rule_name="rss_sustained_high",
        severity="warning",
        history=high_rss_history,
        threshold_immediate=config.rss_kill_mb,
        rule_name_immediate="rss_kill",
        severity_immediate="critical",
    )


def _check_cpu_sustained(
    session_factory: _SessionFactory,
    task_id: str,
    snapshot: dict[str, Any],
    config: MonitoringConfig,
    high_cpu_history: dict[str, float],
) -> str | None:
    return _check_sustained(
        session_factory=session_factory,
        task_id=task_id,
        metric_value=snapshot["cpu_percent"],
        threshold=config.cpu_warning_percent,
        persist_seconds=config.cpu_warning_persist_minutes * 60,
        rule_name="cpu_sustained_high",
        severity="warning",
        history=high_cpu_history,
    )


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------


def _check_oom(
    session_factory: _SessionFactory,
    task_id: str,
    _proc_factory: Callable[[], psutil.Process] | None = None,
    oom_count_override: int | None = None,
) -> str | None:
    if oom_count_override is not None:
        oom_count = oom_count_override
    else:
        try:
            proc = _proc_factory() if _proc_factory else psutil.Process()
            oom_count = _read_oom_kill_count(proc.pid)
        except Exception:
            logger.debug("Could not read OOM kill count", exc_info=True)
            return None

    prev = _last_oom_count.get(task_id, 0)
    _last_oom_count[task_id] = oom_count
    if oom_count > prev:
        _alert_severity(session_factory, task_id, "critical", "oom_kill", float(oom_count - prev))
        return "critical"
    return None


def _read_oom_kill_count(pid: int) -> int:
    """Read the OOM kill count for *pid* from cgroup memory events.

    Cgroup v2 path:  /sys/fs/cgroup/<cgroup_path>/memory.events  (field ``oom_kill``)
    Cgroup v1 fallback: /sys/fs/cgroup/memory/<cgroup_path>/memory.oom_control

    Returns 0 when cgroup files are missing, unreadable, or the field is absent.
    """
    try:
        cgroup_path = _resolve_cgroup_path(pid)
    except Exception:
        return 0
    if cgroup_path is None:
        return 0

    # cgroup v2
    events_path = f"/sys/fs/cgroup{cgroup_path}/memory.events"
    if os.path.exists(events_path):
        try:
            return _parse_oom_kill_from_events(events_path)
        except Exception:
            logger.debug("Failed to parse cgroup v2 memory.events", exc_info=True)
            return 0

    # cgroup v1 fallback
    oom_control_path = f"/sys/fs/cgroup/memory{cgroup_path}/memory.oom_control"
    if os.path.exists(oom_control_path):
        try:
            return _parse_oom_kill_from_oom_control(oom_control_path)
        except Exception:
            logger.debug("Failed to parse cgroup v1 memory.oom_control", exc_info=True)
            return 0

    return 0


def _resolve_cgroup_path(pid: int) -> str | None:
    """Return the cgroup path for *pid* from ``/proc/<pid>/cgroup``.

    For cgroup v2 the line is ``0::/path``.
    For cgroup v1 we look for the memory controller line.
    """
    cgroup_file = f"/proc/{pid}/cgroup"
    if not os.path.exists(cgroup_file):
        return None

    with open(cgroup_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # cgroup v2: "0::/system.slice/foo.service"
            if line.startswith("0::"):
                return line.split(":", 2)[2]
            # cgroup v1: "6:memory:/user.slice/user-1000.slice"
            parts = line.split(":")
            if len(parts) == 3 and "memory" in parts[1]:
                return parts[2]

    return None


def _parse_oom_kill_from_events(events_path: str) -> int:
    """Parse the ``oom_kill`` field from a cgroup v2 ``memory.events`` file."""
    with open(events_path) as fh:
        for line in fh:
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    return 0


def _parse_oom_kill_from_oom_control(oom_control_path: str) -> int:
    """Parse the oom_kill count from a cgroup v1 ``memory.oom_control`` file.

    Format (cgroup v1)::

        oom_kill_disable 0
        under_oom 0
        oom_kill 3
    """
    with open(oom_control_path) as fh:
        for line in fh:
            if line.startswith("oom_kill ") and "disable" not in line:
                return int(line.split()[1])
    return 0


# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------


def _check_disk(
    session_factory: _SessionFactory,
    task_id: str,
    config: MonitoringConfig | None = None,
    free_gb_override: float | None = None,
) -> str | None:
    cfg = config or MonitoringConfig()

    if free_gb_override is not None:
        free_gb = free_gb_override
    else:
        usage = psutil.disk_usage("/")
        free_gb = usage.free / (1024**3)

    if free_gb < cfg.disk_free_gb_warning:
        _alert_severity(session_factory, task_id, "warning", "disk_tight", round(free_gb, 2))
        return "warning"
    return None


# ---------------------------------------------------------------------------
# Pipeline-stage metric persistence
# ---------------------------------------------------------------------------


def write_stage_metric(
    session_factory: _SessionFactory,
    task_id: str,
    stage: str,
    notes: str = "",
) -> None:
    """Persist a pipeline-stage completion metric to ``monitoring_samples``.

    Uses the ``notes`` column to record stage-specific information (e.g.
    ticker counts, pass rates, assessment breakdowns).  RSS/CPU/FD/thread
    fields are set to 0.0 / 0 to distinguish these from resource samples.

    Args:
        session_factory: Callable returning a new SQLAlchemy Session.
        task_id: The scheduler task identifier (e.g. ``"daily_scan"``).
        stage: Pipeline stage label (e.g. ``"phase1"``, ``"phase2"``, ``"fine"``).
        notes: Human-readable metric summary.
    """
    full_notes = f"[{stage}] {notes}" if notes else f"[{stage}]"

    def _write(session: Session) -> None:
        session.add(
            MonitoringSample(
                task_id=task_id,
                sampled_at=_utcnow_iso(),
                rss_mb=0.0,
                cpu_percent=0.0,
                open_fd_count=0,
                thread_count=0,
                notes=full_notes,
            )
        )

    _with_session(session_factory, _write, operation="monitoring_samples.insert_stage")


# ---------------------------------------------------------------------------
# Data retention
# ---------------------------------------------------------------------------


def _cleanup_old_samples(
    session_factory: _SessionFactory,
    retention_days: int = 30,
) -> int:
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()

    def _delete(session: Session) -> int:
        return (
            session.query(MonitoringSample)
            .filter(MonitoringSample.sampled_at < cutoff)
            .delete(synchronize_session="fetch")
        )

    result = _with_session(
        session_factory,
        _delete,
        rollback_value=0,
        operation="monitoring_samples.cleanup",
    )
    return result if result is not None else 0
