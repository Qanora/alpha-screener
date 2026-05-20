"""APScheduler orchestrator: BlockingScheduler + SQLAlchemyJobStore + 8 cron jobs.

Issue #105: APScheduler + pid_lock + task orchestration.
Reference: PRD 7.7.1 / 7.7.2.

Key design decisions:
- BlockingScheduler (single-process, long-running daemon managed by systemd).
- SQLAlchemyJobStore for persistent job state.
- UTC timezone for all cron expressions.
- max_instances=1 per job (double insurance alongside pid_lock).
- pid_lock global mutex: each job acquires the lock before execution and
  releases it after.  Wait timeout = 2 hours per PRD 7.7.2.
- Task wrapper logs skipped executions when lock acquisition times out.
"""

from __future__ import annotations

import logging

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from alphascreener.scheduler.pid_lock import (
    DEFAULT_LOCK_TIMEOUT_S,
    PidLockManager,
)
from alphascreener.scheduler.tasks import TASK_CRON, TASK_FUNCS

_logger = logging.getLogger("scheduler")

# Default: 2 hours per PRD 7.7.2
DEFAULT_LOCK_TIMEOUT_S: int = DEFAULT_LOCK_TIMEOUT_S

# Number of executor threads (tasks are serial via pid_lock, but executor
# needs at least 1 thread to run the job + 1 for health check overlap avoidance)
EXECUTOR_MAX_WORKERS: int = 2


class SchedulerApp:
    """APScheduler application that registers all 8 cron jobs.

    Usage::

        app = SchedulerApp(db_url="sqlite:///path/to/alphabase.db")
        app.start()  # blocks forever

    Or for testing / inspection::

        app = SchedulerApp(db_url="sqlite:///:memory:")
        print(app.scheduler.get_jobs())
        app.shutdown()
    """

    def __init__(
        self,
        db_url: str,
        *,
        lock_timeout_s: int = DEFAULT_LOCK_TIMEOUT_S,
        job_defaults: dict | None = None,
    ) -> None:
        self.db_url = db_url
        self.lock_timeout_s = lock_timeout_s

        # Job store: SQLAlchemy for persistent job state
        jobstores = {
            "default": SQLAlchemyJobStore(url=db_url),
        }

        # Executor: ThreadPoolExecutor (sufficient for serial execution)
        executors = {
            "default": ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS),
        }

        # Job defaults: max_instances=1 for double safety alongside pid_lock
        if job_defaults is None:
            job_defaults = {
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 3600,  # 1 hour grace for misfires
            }

        self.scheduler = BlockingScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )

        self._register_jobs()

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register all 8 cron tasks with the scheduler.

        Each job is wrapped in a pid_lock acquire/release cycle.
        """
        for task_id, cron_expr in TASK_CRON.items():
            func = TASK_FUNCS[task_id]
            trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")

            # Build a closure that captures task_id so each job knows its
            # identity for pid_lock acquisition.
            def _make_job_wrapper(tid: str, fn: object):
                def _wrapped() -> None:
                    _logger.info("Job '%s' triggered", tid)
                    # Import lazily to avoid circular imports at module level.
                    from alphascreener.db.engine import create_db_engine

                    engine = create_db_engine(self.db_url.replace("sqlite:///", ""))
                    try:
                        from sqlalchemy.orm import Session

                        def _sf():
                            return Session(engine)

                        lock = PidLockManager(
                            session_factory=_sf,
                            task_id=tid,
                            timeout_s=self.lock_timeout_s,
                        )
                        ok = lock.acquire()
                        if not ok:
                            _logger.warning(
                                "Job '%s' SKIPPED: lock acquire timed out after %ds "
                                "(another task may still be running)",
                                tid,
                                self.lock_timeout_s,
                            )
                            return
                        try:
                            fn()
                        except Exception:
                            _logger.exception("Job '%s' failed with exception", tid)
                        finally:
                            lock.release()
                    finally:
                        engine.dispose()

                return _wrapped

            wrapped = _make_job_wrapper(task_id, func)

            self.scheduler.add_job(
                wrapped,
                trigger=trigger,
                id=task_id,
                name=task_id,
                replace_existing=True,
            )

            _logger.info(
                "Registered job '%s' with cron '%s'",
                task_id,
                cron_expr,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler (blocks forever)."""
        _logger.info("Starting APScheduler with %d jobs", len(self.scheduler.get_jobs()))
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            _logger.info("Scheduler shutting down")
            self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the scheduler gracefully."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            _logger.info("Scheduler shut down")
