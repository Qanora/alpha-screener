"""Resource monitoring utilities (lightweight, no DB persistence).

Issue #107: Resource monitoring.
Reference: PRD 9.2.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger("monitoring")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitoringConfig:
    """Threshold constants for resource monitoring."""

    sample_interval_seconds: int = 60
    rss_warning_mb: float = 5500.0
    rss_kill_mb: float = 7000.0
    retention_days: int = 30


# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------


def check_disk(free_gb_override: float | None = None) -> dict[str, Any]:
    """Return disk usage info; never persists."""
    if free_gb_override is not None:
        free_gb = free_gb_override
    else:
        usage = psutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
    return {
        "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
        "free_gb": round(free_gb, 2),
        "percent_used": psutil.disk_usage("/").percent,
    }


def check_memory() -> dict[str, Any]:
    """Return process RSS in MB."""
    proc = psutil.Process()
    with proc.oneshot():
        rss_mb = proc.memory_info().rss / (1024 * 1024)
    return {"rss_mb": round(rss_mb, 2)}


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------


def read_oom_kill_count(pid: int | None = None) -> int:
    """Read the OOM kill count from cgroup memory events.

    Cgroup v2 path:  /sys/fs/cgroup/<cgroup_path>/memory.events
    Cgroup v1 fallback: /sys/fs/cgroup/memory/<cgroup_path>/memory.oom_control

    Returns 0 when cgroup files are missing, unreadable, or the field is absent.
    """
    if pid is None:
        pid = os.getpid()
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
    cgroup_file = f"/proc/{pid}/cgroup"
    if not os.path.exists(cgroup_file):
        return None
    with open(cgroup_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("0::"):
                return line.split(":", 2)[2]
            parts = line.split(":")
            if len(parts) == 3 and "memory" in parts[1]:
                return parts[2]
    return None


def _parse_oom_kill_from_events(events_path: str) -> int:
    with open(events_path) as fh:
        for line in fh:
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    return 0


def _parse_oom_kill_from_oom_control(oom_control_path: str) -> int:
    with open(oom_control_path) as fh:
        for line in fh:
            if line.startswith("oom_kill ") and "disable" not in line:
                return int(line.split()[1])
    return 0


# ---------------------------------------------------------------------------
# Pipeline-stage metric helper (no DB — logs only)
# ---------------------------------------------------------------------------


def write_stage_metric(task_id: str, stage: str, notes: str = "") -> None:
    """Log a pipeline-stage completion metric."""
    full_notes = f"[{stage}] {notes}" if notes else f"[{stage}]"
    logger.info("stage_metric task=%s stage=%s %s", task_id, stage, full_notes)
