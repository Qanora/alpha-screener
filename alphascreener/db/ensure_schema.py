"""Schema auto-creation helper: ensures all tables exist on engine startup.

Issue #192: SQLite persistence layer — ensures the database schema is created
before any task writes monitoring metrics.

Usage::

    from alphascreener.db.ensure_schema import _ensure_schema

    engine = create_db_engine(db_url)
    _ensure_schema(engine)  # idempotent, creates tables if missing
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, inspect

from alphascreener.db.models import Base

_logger = logging.getLogger(__name__)


def _ensure_schema(engine: Engine) -> None:
    """Create all ORM tables in *engine* if they do not already exist.

    Uses SQLAlchemy's ``Base.metadata.create_all()`` which is idempotent —
    existing tables are left unchanged.  This is a lighter alternative to
    running full Alembic migrations at every task entry point; the scheduler
    daemon still runs ``alembic upgrade head`` on startup for proper migration
    history tracking.

    Args:
        engine: A SQLAlchemy Engine (typically SQLite file-based).
    """
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())

    if not existing:
        _logger.info("No tables found — creating all ORM tables")
        Base.metadata.create_all(engine)
        _logger.info("Tables created successfully")
    else:
        _logger.debug("Schema already exists (%d tables)", len(existing))
