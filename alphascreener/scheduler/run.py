"""Entrypoint for the APScheduler daemon (called by systemd).

Usage::

    python -m alphascreener.scheduler.run

Reads DB URL from settings (env ``DB_URL`` or derived from ``ALPHASCREENER_HOME``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic.command import upgrade
from alembic.config import Config

from alphascreener.config import Settings
from alphascreener.scheduler.orchestrator import SchedulerApp

_logger = logging.getLogger("scheduler")


def _run_migrations(db_url: str) -> None:
    """Run all pending Alembic migrations against the configured database."""
    alembic_ini = Path(__file__).parent.parent.parent / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    _logger.info("Starting alembic upgrade head")
    upgrade(alembic_cfg, "head")
    _logger.info("Alembic upgrade complete")


def main() -> None:
    """Run pending DB migrations and start the scheduler (blocks forever)."""
    settings = Settings()
    db_url = settings.get_db_url()

    # Auto-run all pending migrations so the schema is always current.
    _run_migrations(db_url)

    # Start the scheduler (blocks forever).
    app = SchedulerApp(db_url=db_url)
    app.start()


if __name__ == "__main__":
    main()
