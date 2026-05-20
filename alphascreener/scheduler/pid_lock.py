"""Process-level mutex via pid_lock table with psutil deadlock recovery.

Issue #105: APScheduler + pid_lock + task orchestration.
Reference: PRD 7.7.2 — single-machine serial execution constraint.

Key behaviors:
- acquire() polls pid_lock until timeout_s, retrying every retry_interval_s.
- Deadlock recovery: if the lock-holding PID is dead (checked via psutil),
  the lock is force-released and a new lock is acquired.
- release() removes the lock row for the current PID.
- Context-manager support: ``with PidLockManager(...) as mgr:``
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import psutil
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from alphascreener.db.models import PidLock

_logger = logging.getLogger("scheduler")

# Default: 2 hours per PRD 7.7.2
DEFAULT_LOCK_TIMEOUT_S: int = 7200
# Poll interval between lock acquisition attempts
DEFAULT_RETRY_INTERVAL_S: float = 5.0
# Task timeout buffer added to expires_at (10 minutes per PRD 7.7.2)
EXPIRY_BUFFER_MINUTES: int = 10
# Expiry buffer for the lock itself beyond the task timeout
EXPIRE_AFTER_BUFFER_S: int = EXPIRY_BUFFER_MINUTES * 60


class PidLockManager:
    """Process-level mutex backed by the ``pid_lock`` table.

    Usage as context manager::

        with PidLockManager(session_factory, task_id="daily_scan") as mgr:
            # do work while holding the global lock
            ...

    Usage manual acquire/release::

        mgr = PidLockManager(session_factory, task_id="daily_scan")
        mgr.acquire()
        try:
            ...
        finally:
            mgr.release()
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        task_id: str = "unknown",
        lock_name: str = "global",
        timeout_s: int = DEFAULT_LOCK_TIMEOUT_S,
        retry_interval_s: float = DEFAULT_RETRY_INTERVAL_S,
    ) -> None:
        self._session_factory = session_factory
        self._task_id = task_id
        self._lock_name = lock_name
        self._timeout_s = timeout_s
        self._retry_interval_s = retry_interval_s
        self._pid = os.getpid()
        self._locked = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(
        self,
        task_id: str | None = None,
        timeout_s: int | None = None,
        meta: dict | None = None,
    ) -> bool:
        """Try to acquire the global lock, waiting up to *timeout_s*.

        Returns:
            True if the lock was acquired, False if the timeout expired.
        """
        tid = task_id or self._task_id
        timeout = timeout_s if timeout_s is not None else self._timeout_s

        deadline = time.monotonic() + timeout

        while True:
            # Open a fresh session for each attempt
            session = self._session_factory()
            try:
                # 1. Check existing lock
                existing = session.query(PidLock).filter_by(lock_name=self._lock_name).first()

                if existing is None:
                    # No lock — acquire it
                    self._insert_lock(session, tid, timeout, meta)
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        _logger.debug(
                            "IntegrityError acquiring lock '%s' (race), retrying",
                            self._lock_name,
                        )
                        continue
                    self._locked = True
                    _logger.info(
                        "Acquired lock '%s' for task '%s' (pid=%d)",
                        self._lock_name,
                        tid,
                        self._pid,
                    )
                    return True

                # 2. Lock exists — check if it can be recovered
                if self._can_recover(existing):
                    _logger.warning(
                        "Recovering dead/expired lock '%s' held by pid=%d (task=%s)",
                        self._lock_name,
                        existing.pid,
                        existing.task_id,
                    )
                    session.delete(existing)
                    session.commit()
                    # Retry immediately
                    continue

                # 3. Lock held by an alive process — wait and retry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _logger.warning(
                        "Lock acquire timed out after %.1fs (lock held by pid=%d, task=%s)",
                        timeout,
                        existing.pid,
                        existing.task_id,
                    )
                    return False

                _logger.debug(
                    "Lock '%s' held by pid=%d (task=%s), retrying in %.1fs (remaining=%.1fs)",
                    self._lock_name,
                    existing.pid,
                    existing.task_id,
                    self._retry_interval_s,
                    remaining,
                )
                session.close()
                time.sleep(min(self._retry_interval_s, remaining))

            finally:
                session.close()

    def release(self) -> None:
        """Release the lock if held by the current process."""
        if not self._locked:
            _logger.debug("Lock not held, release is no-op")
            return

        session = self._session_factory()
        try:
            row = session.query(PidLock).filter_by(lock_name=self._lock_name, pid=self._pid).first()
            if row is not None:
                session.delete(row)
                session.commit()
                _logger.info(
                    "Released lock '%s' (pid=%d, task=%s)",
                    self._lock_name,
                    self._pid,
                    row.task_id,
                )
            self._locked = False
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Deadlock recovery (PRD 7.7.2)
    # ------------------------------------------------------------------

    def _can_recover(self, existing: PidLock) -> bool:
        """Check whether the existing lock row can be recovered.

        Recovery is only allowed when the holding PID is no longer alive.
        An expired lock whose holder PID is still alive is NOT recoverable —
        subsequent jobs must skip to preserve global serial execution.
        """
        pid_alive = self._is_pid_alive(existing.pid)

        # If the PID is dead, the lock is always recoverable.
        if not pid_alive:
            return True

        # PID is alive — lock is NOT recoverable regardless of expiry.
        return False

    def _is_pid_alive(self, pid: int) -> bool:
        """Check whether a PID is alive using psutil.

        Returns False for pid <= 0 (sentinel/invalid) and for non-existent PIDs.
        """
        if pid <= 0:
            return False
        return psutil.pid_exists(pid)

    def _recover_dead_lock(self) -> bool:
        """Force-release the lock if the holding PID is dead.

        Returns:
            True if a dead lock was recovered, False if lock is valid or absent.
        """
        session = self._session_factory()
        try:
            existing = session.query(PidLock).filter_by(lock_name=self._lock_name).first()
            if existing is None:
                return False
            if self._can_recover(existing):
                session.delete(existing)
                session.commit()
                _logger.warning(
                    "Force-released dead/expired lock '%s' (pid=%d, task=%s)",
                    self._lock_name,
                    existing.pid,
                    existing.task_id,
                )
                return True
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_lock(
        self, session: Session, task_id: str, timeout_s: int, meta: dict | None
    ) -> None:
        """Insert a new pid_lock row for the current process."""
        now = datetime.now(UTC)
        # expires_at = now + timeout + expiry buffer
        expires_at = now + timedelta(seconds=timeout_s + EXPIRE_AFTER_BUFFER_S)

        meta_json = None
        if meta:
            meta_json = json.dumps(meta)

        lock = PidLock(
            lock_name=self._lock_name,
            pid=self._pid,
            task_id=task_id,
            acquired_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            meta_json=meta_json,
        )
        session.add(lock)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> PidLockManager:
        ok = self.acquire()
        if not ok:
            raise RuntimeError(
                f"Failed to acquire lock '{self._lock_name}' "
                f"for task '{self._task_id}' within {self._timeout_s}s"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        return None
