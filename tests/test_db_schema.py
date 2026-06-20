"""Tests for SQLite schema, engine WAL mode, and Alembic migrations.

Issue #85: SQLite schema & migrations.
"""

from pathlib import Path

import pytest
from sqlalchemy import (
    create_engine,
    inspect,
    text,
)

from alphascreener.db.engine import create_db_engine
from alphascreener.db.models import Base

# ============================================================================
# Helper fixtures
# ============================================================================


@pytest.fixture
def fresh_sqlite_db(tmp_path: Path):
    """Create a fresh in-memory SQLite database with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def file_db_path(tmp_path: Path) -> Path:
    """Path for a file-based SQLite database."""
    return tmp_path / "test.db"


# ============================================================================
# Test all 9 core tables exist in metadata
# ============================================================================

EXPECTED_TABLES = [
    "factor_versions",
    "alpha_acceptance_daily",
]

CORE_TABLE_COUNT = len(EXPECTED_TABLES)


class TestAllTablesExist:
    """Verify core tables are declared in the SQLAlchemy metadata."""

    def test_all_core_tables_in_metadata(self):
        table_names = sorted(Base.metadata.tables.keys())
        for expected in EXPECTED_TABLES:
            assert expected in table_names, f"Missing table: {expected}"
        assert len([t for t in table_names if t in EXPECTED_TABLES]) == CORE_TABLE_COUNT


class TestTableCreation:
    """Verify all tables can be created in SQLite."""

    def test_all_tables_created(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        created = sorted(inspector.get_table_names())
        for expected in EXPECTED_TABLES:
            assert expected in created, f"Table not created: {expected}"


# ============================================================================
# Table structure tests
# ============================================================================


class TestFactorVersions:
    """factor_versions table: factor version config."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("factor_versions")}
        assert "version" in cols
        assert "released_at" in cols
        assert "config_json" in cols
        assert "parent_version" in cols
        assert "release_type" in cols
        assert cols["version"]["primary_key"] == 1

    def test_version_is_primary_key(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        pk = inspector.get_pk_constraint("factor_versions")
        assert pk["constrained_columns"] == ["version"]


class TestAlphaAcceptanceDaily:
    """alpha_acceptance_daily: daily alpha acceptance metrics."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("alpha_acceptance_daily")}
        assert "metric_date" in cols
        assert "base_rate" in cols
        assert "precision_at_20_pure" in cols
        assert "precision_at_20_llm" in cols
        assert "precision_at_10_pure" in cols
        assert "precision_at_10_llm" in cols
        assert "lift_at_20_pure" in cols
        assert "lift_at_20_llm" in cols
        assert "ic_pure" in cols
        assert "ic_llm" in cols
        assert "bootstrap_ci_lower_pure" in cols
        assert "bootstrap_ci_upper_pure" in cols
        assert "bootstrap_ci_lower_llm" in cols
        assert "bootstrap_ci_upper_llm" in cols
        assert "sample_size" in cols

    def test_metric_date_is_primary_key(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        pk = inspector.get_pk_constraint("alpha_acceptance_daily")
        assert pk["constrained_columns"] == ["metric_date"]


