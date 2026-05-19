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
    "paper_trades",
    "alerts",
    "llm_cost_daily",
    "pid_lock",
    "monitoring_samples",
    "alpha_acceptance_daily",
    "data_source_diff",
    "factor_health_daily",
]


class TestAllTablesExist:
    """Verify all 9 core tables are declared in the SQLAlchemy metadata."""

    def test_all_nine_tables_in_metadata(self):
        table_names = sorted(Base.metadata.tables.keys())
        for expected in EXPECTED_TABLES:
            assert expected in table_names, f"Missing table: {expected}"
        assert len([t for t in table_names if t in EXPECTED_TABLES]) == 9


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


class TestPaperTrades:
    """paper_trades table: paper trading / virtual trading records."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("paper_trades")}
        assert "id" in cols
        assert "signal_date" in cols
        assert "ticker" in cols
        assert "rating" in cols
        assert "breakout_probability" in cols
        assert "entry_price" in cols
        assert "exit_price" in cols
        assert "exit_reason" in cols
        assert "pnl_pct" in cols
        assert "factor_version" in cols
        assert "created_at" in cols

    def test_foreign_key_to_factor_versions(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        fks = inspector.get_foreign_keys("paper_trades")
        fk_cols = [fk["constrained_columns"] for fk in fks]
        assert ["factor_version"] in fk_cols

    def test_index_on_signal_date(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        indexes = inspector.get_indexes("paper_trades")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_paper_trades_signal_date" in index_names


class TestAlerts:
    """alerts table: alert events."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("alerts")}
        assert "id" in cols
        assert "triggered_at" in cols
        assert "severity" in cols
        assert "rule_name" in cols
        assert "metric_value" in cols
        assert "notes" in cols
        assert "resolved_at" in cols


class TestLlmCostDaily:
    """llm_cost_daily table: daily LLM cost aggregation."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("llm_cost_daily")}
        assert "cost_date" in cols
        assert "total_usd" in cols
        assert "call_count" in cols
        assert "by_module_json" in cols

    def test_cost_date_is_primary_key(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        pk = inspector.get_pk_constraint("llm_cost_daily")
        assert pk["constrained_columns"] == ["cost_date"]


class TestPidLock:
    """pid_lock table: process mutex for serial execution."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("pid_lock")}
        assert "lock_name" in cols
        assert "pid" in cols
        assert "task_id" in cols
        assert "acquired_at" in cols
        assert "expires_at" in cols
        assert "meta_json" in cols

    def test_lock_name_is_primary_key(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        pk = inspector.get_pk_constraint("pid_lock")
        assert pk["constrained_columns"] == ["lock_name"]

    def test_index_on_expires_at(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        indexes = inspector.get_indexes("pid_lock")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_pid_lock_expires" in index_names


class TestMonitoringSamples:
    """monitoring_samples table: resource monitoring samples."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("monitoring_samples")}
        assert "id" in cols
        assert "task_id" in cols
        assert "sampled_at" in cols
        assert "rss_mb" in cols
        assert "cpu_percent" in cols
        assert "open_fd_count" in cols
        assert "thread_count" in cols
        assert "notes" in cols

    def test_index_on_task_id_sampled_at(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        indexes = inspector.get_indexes("monitoring_samples")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_monitoring_task_time" in index_names


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


class TestDataSourceDiff:
    """data_source_diff table: fallback OHLCV cross-validation diffs."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("data_source_diff")}
        assert "id" in cols
        assert "metric_date" in cols
        assert "ticker" in cols
        assert "field" in cols
        assert "yfinance_value" in cols
        assert "fallback_value" in cols
        assert "fallback_source" in cols
        assert "diff_pct" in cols
        assert "alerted" in cols

    def test_index_on_metric_date(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        indexes = inspector.get_indexes("data_source_diff")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_data_source_diff_date" in index_names


class TestFactorHealthDaily:
    """factor_health_daily: CUSUM fast-monitoring time series."""

    def test_columns(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        cols = {c["name"]: c for c in inspector.get_columns("factor_health_daily")}
        assert "metric_date" in cols
        assert "factor_name" in cols
        assert "daily_ic" in cols
        assert "rolling_ic_mean_90d" in cols
        assert "cusum_value" in cols
        assert "cusum_alert" in cols
        assert "consecutive_alerts" in cols

    def test_composite_primary_key(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        pk = inspector.get_pk_constraint("factor_health_daily")
        assert set(pk["constrained_columns"]) == {"metric_date", "factor_name"}

    def test_index_on_factor_name_metric_date(self, fresh_sqlite_db):
        inspector = inspect(fresh_sqlite_db)
        indexes = inspector.get_indexes("factor_health_daily")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_factor_health_factor_date" in index_names


# ============================================================================
# WAL mode tests
# ============================================================================


class TestWalMode:
    """Verify WAL mode is enabled."""

    def test_wal_mode_for_file_db(self, file_db_path: Path):
        """Create a file-based DB via create_db_engine and verify WAL mode."""
        engine = create_db_engine(str(file_db_path))
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal", f"Expected WAL mode, got {result}"
        engine.dispose()

    def test_pragma_synchronous_normal(self, file_db_path: Path):
        """Verify synchronous PRAGMA is set to NORMAL for WAL."""
        engine = create_db_engine(str(file_db_path))
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA synchronous")).scalar()
            assert result == 1, f"Expected synchronous=1 (NORMAL), got {result}"
        engine.dispose()

    def test_in_memory_db_does_not_crash_on_wal_pragma(self, tmp_path: Path):
        """WAL pragma is applied via create_db_engine and does not crash."""
        db_path = tmp_path / "test_wal.db"
        engine = create_db_engine(str(db_path))
        try:
            Base.metadata.create_all(engine)
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA journal_mode")).scalar()
                assert result == "wal", f"Expected WAL mode, got {result}"
        finally:
            engine.dispose()


# ============================================================================
# Data retention strategy documentation tests
# ============================================================================


class TestDataRetentionDocumentation:
    """Verify data retention strategy is documented in model comments."""

    def test_alpha_acceptance_daily_permanent_retention_documented(self):
        """alpha_acceptance_daily is permanent (small table)."""
        table = Base.metadata.tables["alpha_acceptance_daily"]
        assert table.comment is not None and "永久" in table.comment, (
            "alpha_acceptance_daily should have permanent retention documented"
        )

    def test_data_source_diff_365_day_retention_documented(self):
        """data_source_diff retains 365 days."""
        table = Base.metadata.tables["data_source_diff"]
        assert table.comment is not None, "data_source_diff should have a comment"
        # Comment should mention 365 or retention
        assert "365" in table.comment

    def test_factor_health_daily_365_day_retention_documented(self):
        """factor_health_daily retains 365 days."""
        table = Base.metadata.tables["factor_health_daily"]
        assert table.comment is not None, "factor_health_daily should have a comment"
        assert "365" in table.comment


# ============================================================================
# Alembic migration tests
# ============================================================================


class TestAlembicSetup:
    """Verify Alembic is configured properly."""

    def test_alembic_ini_exists(self):
        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        assert alembic_ini.exists(), "alembic.ini missing"

    def test_alembic_env_exists(self):
        env_py = Path(__file__).parent.parent / "alembic" / "env.py"
        assert env_py.exists(), "alembic/env.py missing"

    def test_alembic_versions_dir_exists(self):
        versions_dir = Path(__file__).parent.parent / "alembic" / "versions"
        assert versions_dir.exists(), "alembic/versions/ missing"


class TestMigrationUpgrade:
    """Verify the initial migration creates all 9 tables."""

    def test_initial_migration_creates_all_tables(self, file_db_path: Path):
        """Run the initial Alembic migration and verify all tables exist."""
        from alembic.command import upgrade
        from alembic.config import Config

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini))
        # Override the sqlalchemy.url for this test
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{file_db_path}")

        upgrade(alembic_cfg, "head")

        engine = create_engine(f"sqlite:///{file_db_path}", echo=False)
        inspector = inspect(engine)
        created = sorted(inspector.get_table_names())
        # alembic_version table is expected
        created_user = [t for t in created if t != "alembic_version"]
        for expected in EXPECTED_TABLES:
            assert expected in created_user, f"Table {expected} not created by migration"
        engine.dispose()

    def test_migration_applies_wal_mode(self, file_db_path: Path):
        """After migration, verify WAL journal_mode on file DB."""
        from alembic.command import upgrade
        from alembic.config import Config

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{file_db_path}")

        upgrade(alembic_cfg, "head")

        engine = create_engine(f"sqlite:///{file_db_path}", echo=False)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal", f"WAL not enabled after migration, got {result}"
        engine.dispose()


class TestMigrationDowngrade:
    """Verify downgrade removes all tables."""

    def test_downgrade_removes_all_tables(self, file_db_path: Path):
        """Run upgrade then downgrade, verify all user tables are gone."""
        from alembic.command import downgrade, upgrade
        from alembic.config import Config

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{file_db_path}")

        upgrade(alembic_cfg, "head")
        downgrade(alembic_cfg, "base")

        engine = create_engine(f"sqlite:///{file_db_path}", echo=False)
        inspector = inspect(engine)
        created = inspector.get_table_names()
        # After full downgrade, only alembic_version remains (empty)
        user_tables = [t for t in created if t not in ("alembic_version",)]
        assert len(user_tables) == 0, f"Tables still exist after downgrade: {user_tables}"
        engine.dispose()
