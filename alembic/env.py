"""Alembic environment configuration for Alpha Screener SQLite database.

Imports all ORM models so autogenerate can detect schema changes.
Enables WAL mode and foreign keys on each migration connection.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, event, pool

from alembic import context
from alphascreener.db.models import Base  # noqa: F401 — imports all models for autogenerate

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# All ORM models inherit from Base — use its metadata for autogenerate
target_metadata = Base.metadata


def _enable_wal_and_pragmas(dbapi_connection, connection_record):
    """Enable WAL mode and SQLite pragmas on each new connection during migrations."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emits SQL without a live database.

    Useful for generating SQL scripts for manual review.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database.

    Creates an engine, attaches WAL pragma listener, and executes all pending
    migrations.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Enable WAL mode on every migration connection
    event.listen(connectable, "connect", _enable_wal_and_pragmas)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
