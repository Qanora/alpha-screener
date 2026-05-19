"""SQLAlchemy engine factory with WAL mode and SQLite pragmas (Issue #85).

Reference: PRD 7.6.2 WAL mode requirement.
"""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event


def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable WAL mode and optimize SQLite settings on each new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    # NORMAL synchronous is safe with WAL and significantly faster
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Store temp files in memory for better performance
    cursor.execute("PRAGMA temp_store=MEMORY")
    # Foreign key enforcement (disabled by default in SQLite)
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_engine(db_path: str | Path, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine with SQLite WAL mode enabled.

    Args:
        db_path: Path to the SQLite database file.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy Engine configured for SQLite with WAL mode.
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)

    event.listen(engine, "connect", _set_sqlite_pragmas)

    return engine


# Default engine: uses in-memory SQLite for development / testing.
# Production code should call create_db_engine() with a file path.
engine: Engine = create_engine("sqlite://", echo=False)

# Register pragma listener on the default in-memory engine as well
event.listen(engine, "connect", _set_sqlite_pragmas)
