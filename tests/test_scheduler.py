"""Tests for APScheduler scheduling system, pid_lock, and task orchestration.

Issue #105: APScheduler + pid_lock + task orchestration.
Reference: PRD 7.7.1 / 7.7.2.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import psutil
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alphascreener.db.models import Base, PidLock
from alphascreener.scheduler.pid_lock import PidLockManager

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Fresh session for each test."""
    session = Session(db_engine)
    try:
        yield session
    finally:
        session.close()


def _session_factory(db_engine):
    """Return a session factory bound to the given engine."""

    def _factory():
        return Session(db_engine)

    return _factory


# ============================================================================
# 1. PidLockManager — acquire
# ============================================================================


class TestPidLockAcquire:
    """PidLockManager.acquire() obtains the global lock or raises."""

    def test_acquire_when_no_lock_exists(self, db_engine):
        """Acquire succeeds when the pid_lock table is empty."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        result = mgr.acquire(task_id="daily_scan", timeout_s=30)
        assert result is True

    def test_acquire_writes_row_to_db(self, db_engine):
        """After acquire, a row with lock_name='global' exists."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.acquire(task_id="daily_scan", timeout_s=30)

        with Session(db_engine) as s:
            row = s.query(PidLock).filter_by(lock_name="global").one()
            assert row.task_id == "daily_scan"
            assert row.pid > 0

    def test_acquire_sets_expires_at_in_future(self, db_engine):
        """expires_at should be in the future after acquire."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.acquire(task_id="daily_scan", timeout_s=30)

        with Session(db_engine) as s:
            row = s.query(PidLock).filter_by(lock_name="global").one()
            expires_dt = datetime.fromisoformat(row.expires_at)
            now = datetime.now(UTC)
            assert expires_dt > now

    def test_acquire_fails_when_lock_held_by_alive_process(self, db_engine):
        """If another (alive) process holds the lock, acquire returns False."""
        # Simulate a lock held by current PID (alive)
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        assert mgr.acquire(task_id="daily_scan", timeout_s=30) is True

        # Try to acquire again — should fail since lock is held by our PID (alive)
        mgr2 = PidLockManager(session_factory=_session_factory(db_engine))
        result = mgr2.acquire(task_id="daily_health_check", timeout_s=5)
        assert result is False

    def test_acquire_recovers_dead_lock(self, db_engine, monkeypatch):
        """When lock held by dead PID, acquire recovers and succeeds."""
        from unittest.mock import MagicMock
        dead_process = MagicMock()
        dead_process.is_running.return_value = False
        monkeypatch.setattr(psutil, "Process", lambda pid: dead_process)
        dead_pid = 99999
        with Session(db_engine) as s:
            s.add(
                PidLock(
                    lock_name="global",
                    pid=dead_pid,
                    task_id="stale_task",
                    acquired_at=datetime.now(UTC).isoformat(),
                    expires_at=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                )
            )
            s.commit()

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        result = mgr.acquire(task_id="daily_scan", timeout_s=30)
        assert result is True

    def test_acquire_refuses_expired_lock_when_pid_alive(self, db_engine):
        """Expired lock held by an alive PID is NOT recovered — preserves serial execution."""
        # Insert a lock that has already expired but whose PID is the current process (alive)
        with Session(db_engine) as s:
            s.add(
                PidLock(
                    lock_name="global",
                    pid=os.getpid(),
                    task_id="expired_task",
                    acquired_at=(datetime.now(UTC) - timedelta(hours=3)).isoformat(),
                    expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                )
            )
            s.commit()

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        result = mgr.acquire(task_id="daily_scan", timeout_s=0.5)
        assert result is False

    def test_acquire_uses_custom_timeout_s_for_expires_at(self, db_engine):
        """expires_at reflects the timeout_s passed to acquire(), not the class default."""
        custom_timeout = 45  # different from DEFAULT_LOCK_TIMEOUT_S (7200)
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.acquire(task_id="daily_scan", timeout_s=custom_timeout)

        with Session(db_engine) as s:
            row = s.query(PidLock).filter_by(lock_name="global").one()
            expires_dt = datetime.fromisoformat(row.expires_at)
            now = datetime.now(UTC)
            # expires_at should be ~now + custom_timeout + EXPIRE_AFTER_BUFFER_S (600)
            delta = (expires_dt - now).total_seconds()
            # Allow generous margin for test execution time
            expected = custom_timeout + 600
            assert abs(delta - expected) < 10, (
                f"Expected expires_at ~{expected}s from now, got {delta:.0f}s"
            )

    def test_acquire_handles_integrity_error_race(self, db_engine, monkeypatch):
        """When a concurrent insert causes IntegrityError, acquire retries successfully."""
        from sqlalchemy.exc import IntegrityError

        call_count = [0]
        original_commit = Session.commit

        def _mock_commit(session_self):
            call_count[0] += 1
            if call_count[0] == 1:
                session_self.rollback()
                raise IntegrityError("mock race condition", None, Exception("mock race condition"))
            # Second call proceeds normally
            original_commit(session_self)

        monkeypatch.setattr(Session, "commit", _mock_commit)

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        result = mgr.acquire(task_id="daily_scan", timeout_s=5)
        assert result is True
        assert call_count[0] == 2  # First attempt failed (race), second succeeded


# ============================================================================
# 2. PidLockManager — release
# ============================================================================


class TestPidLockRelease:
    """PidLockManager.release() removes the global lock."""

    def test_release_removes_lock_row(self, db_engine):
        """After release, no lock row remains."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.acquire(task_id="daily_scan", timeout_s=30)
        mgr.release()

        with Session(db_engine) as s:
            count = s.query(PidLock).filter_by(lock_name="global").count()
            assert count == 0

    def test_release_when_no_lock_is_noop(self, db_engine):
        """Calling release when no lock exists does not raise."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.release()  # should not raise

    def test_release_only_releases_own_pid(self, db_engine):
        """Release only removes the lock if the PID matches."""
        # Insert a lock held by a different PID
        with Session(db_engine) as s:
            s.add(
                PidLock(
                    lock_name="global",
                    pid=99999,
                    task_id="other_task",
                    acquired_at=datetime.now(UTC).isoformat(),
                    expires_at=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                )
            )
            s.commit()

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.release()  # should NOT release (different PID)

        with Session(db_engine) as s:
            count = s.query(PidLock).filter_by(lock_name="global").count()
            assert count == 1  # lock still held by PID 99999


# ============================================================================
# 3. PidLockManager — wait / timeout
# ============================================================================


class TestPidLockWait:
    """PidLockManager.acquire() waits up to timeout_s for the lock."""

    def test_acquire_waits_and_retries(self, db_engine):
        """When lock is held by an alive process, acquire retries until timeout."""
        # Hold the lock with our own PID (alive)
        mgr_holder = PidLockManager(session_factory=_session_factory(db_engine))
        assert mgr_holder.acquire(task_id="daily_scan", timeout_s=30) is True

        # Wait with a short timeout — should fail since lock held by us (alive)
        mgr_waiter = PidLockManager(
            session_factory=_session_factory(db_engine),
            retry_interval_s=0.1,
        )
        result = mgr_waiter.acquire(task_id="daily_health_check", timeout_s=1)
        assert result is False  # timeout expired

    def test_acquire_retries_on_busy_lock(self, db_engine):
        """acquire retries multiple times within the timeout window."""
        # Hold lock with our PID
        mgr_holder = PidLockManager(
            session_factory=_session_factory(db_engine),
            retry_interval_s=0.05,
        )
        assert mgr_holder.acquire(task_id="task_a", timeout_s=30) is True

        # Waiter with very short retry interval — polls until timeout
        mgr_waiter = PidLockManager(
            session_factory=_session_factory(db_engine),
            retry_interval_s=0.1,
        )
        start = time.monotonic()
        result = mgr_waiter.acquire(task_id="task_b", timeout_s=0.5)
        elapsed = time.monotonic() - start
        assert result is False
        # elapsed should be close to timeout (allowed margin for polling)
        assert elapsed >= 0.4


# ============================================================================
# 4. PidLockManager — deadlock recovery via psutil
# ============================================================================


class TestPidLockRecovery:
    """Deadlock recovery: psutil checks whether lock-holder PID is alive."""

    def test_is_pid_alive_returns_true_for_current_pid(self):
        """psutil.pid_exists returns True for the current process."""
        mgr = PidLockManager(session_factory=lambda: None)
        assert mgr._is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_returns_false_for_nonexistent_pid(self, monkeypatch):
        """psutil.NoSuchProcess means PID is not alive."""
        monkeypatch.setattr(psutil, "Process", lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)))
        mgr = PidLockManager(session_factory=lambda: None)
        assert mgr._is_pid_alive(99999) is False

    def test_recover_dead_lock_force_releases(self, db_engine, monkeypatch):
        """When lock-holder PID is dead, force_release removes the lock."""
        from unittest.mock import MagicMock
        dead_process = MagicMock()
        dead_process.is_running.return_value = False
        monkeypatch.setattr(psutil, "Process", lambda pid: dead_process)
        dead_pid = 99999
        with Session(db_engine) as s:
            s.add(
                PidLock(
                    lock_name="global",
                    pid=dead_pid,
                    task_id="stale_task",
                    acquired_at=datetime.now(UTC).isoformat(),
                    expires_at=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                )
            )
            s.commit()

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        recovered = mgr._recover_dead_lock()
        assert recovered is True

        with Session(db_engine) as s:
            count = s.query(PidLock).filter_by(lock_name="global").count()
            assert count == 0

    def test_recover_dead_lock_skips_when_pid_alive(self, db_engine):
        """When lock-holder PID is alive, recover does not remove the lock."""
        with Session(db_engine) as s:
            s.add(
                PidLock(
                    lock_name="global",
                    pid=os.getpid(),
                    task_id="running_task",
                    acquired_at=datetime.now(UTC).isoformat(),
                    expires_at=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                )
            )
            s.commit()

        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        recovered = mgr._recover_dead_lock()
        assert recovered is False

        with Session(db_engine) as s:
            count = s.query(PidLock).filter_by(lock_name="global").count()
            assert count == 1  # lock still held


# ============================================================================
# 5. PidLockManager — context manager
# ============================================================================


class TestPidLockContextManager:
    """PidLockManager supports with-statement for automatic acquire/release."""

    def test_context_manager_acquires_and_releases(self, db_engine):
        """Using `with PidLockManager(...) as mgr:` acquires and releases."""
        with PidLockManager(
            session_factory=_session_factory(db_engine),
            task_id="daily_scan",
            timeout_s=30,
        ) as mgr:
            with Session(db_engine) as s:
                count = s.query(PidLock).filter_by(lock_name="global").count()
                assert count == 1
                assert mgr._locked is True

        # After context exit, lock is released
        with Session(db_engine) as s:
            count = s.query(PidLock).filter_by(lock_name="global").count()
            assert count == 0

    def test_context_manager_raises_on_acquire_failure(self, db_engine):
        """When acquire fails, context manager raises RuntimeError."""
        # Pre-populate with an alive lock holder
        mgr_holder = PidLockManager(session_factory=_session_factory(db_engine))
        mgr_holder.acquire(task_id="busy_task", timeout_s=30)

        with pytest.raises(RuntimeError, match=r"(acquire|lock)"):
            with PidLockManager(
                session_factory=_session_factory(db_engine),
                task_id="daily_scan",
                timeout_s=0.5,
                retry_interval_s=0.1,
            ):
                pass  # should not reach here


# ============================================================================
# 6. PidLockManager — oversize lock handling (lock_name != 'global')
# ============================================================================


class TestPidLockCustomName:
    """PidLockManager can use non-default lock_name."""

    def test_custom_lock_name(self, db_engine):
        """Acquire with a custom lock_name stores it correctly."""
        mgr = PidLockManager(
            session_factory=_session_factory(db_engine),
            lock_name="monthly_backtest",
        )
        mgr.acquire(task_id="monthly_full_backtest", timeout_s=30)

        with Session(db_engine) as s:
            row = s.query(PidLock).filter_by(lock_name="monthly_backtest").one()
            assert row.task_id == "monthly_full_backtest"

        mgr.release()


# ============================================================================
# 7. Task definitions
# ============================================================================


class TestTaskDefinitionsStorage:
    """TASK_CRON map and TASK_IDS set exist and are correctly populated."""

    def test_task_cron_map_has_eleven_entries(self):
        """TASK_CRON has 11 tasks (includes weekly_case_library_rebuild from #190)."""
        from alphascreener.scheduler.tasks import TASK_CRON

        assert isinstance(TASK_CRON, dict)
        assert len(TASK_CRON) == 11

    def test_task_ids_set_is_complete(self):
        """TASK_IDS matches TASK_CRON keys."""
        from alphascreener.scheduler.tasks import TASK_CRON, TASK_IDS

        assert TASK_IDS == set(TASK_CRON.keys())

    def test_all_expected_task_ids_present(self):
        """All 11 task IDs (8 PRD 7.7.1 + 1 #103 + 1 #104 + 1 #190) are present."""
        from alphascreener.scheduler.tasks import TASK_CRON

        expected = {
            "monthly_cost_reset",
            "monthly_full_backtest",
            "monthly_isoforest_retrain",
            "biweekly_evolution",
            "monthly_universe_refresh",
            "daily_cusum_check",
            "daily_backtest_incremental",
            "daily_health_check",
            "daily_scan",
            "daily_feishu_push",
            "weekly_case_library_rebuild",
        }
        assert set(TASK_CRON.keys()) == expected

    def test_cron_expressions_match_prd(self):
        """Cron expressions match PRD 7.7.1 table."""
        from alphascreener.scheduler.tasks import TASK_CRON

        assert TASK_CRON["monthly_cost_reset"] == "0 0 1 * *"
        assert TASK_CRON["monthly_full_backtest"] == "5 0 1 * *"
        assert TASK_CRON["monthly_isoforest_retrain"] == "0 5 1 * *"
        assert TASK_CRON["biweekly_evolution"] == "30 5 1,15 * *"
        assert TASK_CRON["monthly_universe_refresh"] == "0 8 1 * *"
        assert TASK_CRON["daily_backtest_incremental"] == "0 11 * * 2-6"
        assert TASK_CRON["daily_health_check"] == "0 12 * * *"
        assert TASK_CRON["daily_scan"] == "0 23 * * 1-5"


class TestTaskCallables:
    """Each task ID maps to a callable."""

    def test_task_funcs_are_callable(self):
        """All tasks in TASK_FUNCS are callable."""
        from alphascreener.scheduler.tasks import TASK_FUNCS

        for task_id, func in TASK_FUNCS.items():
            assert callable(func), f"{task_id} is not callable"

    def test_task_funcs_accept_no_args(self):
        """Task functions accept no required arguments (APScheduler convention)."""
        import inspect

        from alphascreener.scheduler.tasks import TASK_FUNCS

        for task_id, func in TASK_FUNCS.items():
            sig = inspect.signature(func)
            for param in sig.parameters.values():
                if param.name == "self":
                    continue
                assert param.default is not param.empty, f"{task_id} has required arg: {param.name}"


# ============================================================================
# 8. Scheduler orchestrator
# ============================================================================


class TestSchedulerOrchestrator:
    """SchedulerApp sets up BlockingScheduler correctly."""

    def test_creates_blocking_scheduler(self, tmp_path):
        """SchedulerApp uses BlockingScheduler."""
        from alphascreener.scheduler.orchestrator import SchedulerApp

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}")
        from apscheduler.schedulers.blocking import BlockingScheduler

        assert isinstance(app.scheduler, BlockingScheduler)

    def test_timezone_is_utc(self, tmp_path):
        """Scheduler timezone is set to UTC."""
        from alphascreener.scheduler.orchestrator import SchedulerApp

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}")
        assert str(app.scheduler.timezone) == "UTC"

    def test_max_instances_is_one_per_job(self, tmp_path):
        """Scheduler job_defaults enforces max_instances=1 for all jobs."""
        from alphascreener.scheduler.orchestrator import SchedulerApp

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}")
        # APScheduler 3.x stores max_instances in job_defaults, not on Job objects
        job_defaults = app.scheduler._job_defaults
        assert job_defaults.get("max_instances") == 1, (
            f"max_instances should be 1, got {job_defaults.get('max_instances')}"
        )

    def test_eleven_jobs_registered(self, tmp_path):
        """All 11 tasks are registered as jobs."""
        from alphascreener.scheduler.orchestrator import SchedulerApp

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}")
        jobs = app.scheduler.get_jobs()
        job_ids = {job.id for job in jobs}
        assert len(job_ids) == 11
        from alphascreener.scheduler.tasks import TASK_IDS

        assert job_ids == TASK_IDS

    def test_each_job_has_correct_trigger(self, tmp_path):
        """Each registered job uses a CronTrigger matching the PRD."""
        from alphascreener.scheduler.orchestrator import SchedulerApp
        from alphascreener.scheduler.tasks import TASK_CRON

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}")
        jobs = {job.id: job for job in app.scheduler.get_jobs()}
        from apscheduler.triggers.cron import CronTrigger

        for task_id, cron_expr in TASK_CRON.items():
            job = jobs[task_id]
            assert isinstance(job.trigger, CronTrigger)
            # Parse the expected cron and compare field-by-field
            expected = CronTrigger.from_crontab(cron_expr)
            actual_fields = job.trigger.fields
            expected_fields = expected.fields
            field_names = ["minute", "hour", "day", "month", "day_of_week"]
            for i, (actual, exp) in enumerate(zip(actual_fields, expected_fields, strict=True)):
                assert str(actual) == str(exp), (
                    f"{task_id} field '{field_names[i]}': expected '{exp}', got '{actual}'"
                )


# ============================================================================
# 9. pid_lock database model integration
# ============================================================================


class TestPidLockModelIntegration:
    """The PidLock model fields align with PidLockManager usage."""

    def test_pid_lock_model_fields(self, db_engine):
        """PidLock model has all fields used by PidLockManager."""
        mgr = PidLockManager(session_factory=_session_factory(db_engine))
        mgr.acquire(task_id="daily_scan", timeout_s=30, meta={"trigger": "cron"})

        with Session(db_engine) as s:
            row = s.query(PidLock).filter_by(lock_name="global").one()
            assert row.lock_name == "global"
            assert isinstance(row.pid, int)
            assert row.task_id == "daily_scan"
            assert row.acquired_at is not None
            assert row.expires_at is not None
            assert row.meta_json is not None


# ============================================================================
# 10. SchedulerApp — lock_timeout_s configuration
# ============================================================================


class TestSchedulerTimeout:
    """Scheduler task timeout adheres to PRD 7.7.2: wait <= 2h, skip on timeout."""

    def test_default_lock_timeout_is_two_hours(self, tmp_path):
        """Default lock_timeout_s is 7200 (2 hours per PRD 7.7.2)."""
        from alphascreener.scheduler.orchestrator import DEFAULT_LOCK_TIMEOUT_S

        assert DEFAULT_LOCK_TIMEOUT_S == 7200

    def test_custom_lock_timeout(self, tmp_path):
        """SchedulerApp accepts custom lock_timeout_s."""
        from alphascreener.scheduler.orchestrator import SchedulerApp

        db_path = str(tmp_path / "test.db")
        app = SchedulerApp(db_url=f"sqlite:///{db_path}", lock_timeout_s=3600)
        assert app.lock_timeout_s == 3600
