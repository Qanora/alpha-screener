"""Entrypoint for the APScheduler daemon (called by systemd).

Usage::

    python -m alphascreener.scheduler.run

Reads DB path from environment or defaults to
``~/.alphascreener/alphabase.db``.
"""

from __future__ import annotations

from alphascreener.config import Settings
from alphascreener.db.engine import create_db_engine
from alphascreener.db.models import Base
from alphascreener.scheduler.orchestrator import SchedulerApp


def main() -> None:
    """Bootstrap the database schema and start the scheduler (blocks forever)."""
    settings = Settings()
    db_dir = settings.alphascreener_home / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "alphabase.db"

    # Bootstrap the database schema if needed.
    engine = create_db_engine(str(db_path))
    Base.metadata.create_all(engine)
    engine.dispose()

    # Start the scheduler (blocks forever).
    app = SchedulerApp(db_url=f"sqlite:///{db_path}")
    app.start()


if __name__ == "__main__":
    main()
