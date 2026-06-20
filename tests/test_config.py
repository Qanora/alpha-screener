"""Tests for alphascreener.config module."""

from pathlib import Path

import pytest
from alphascreener.config import Settings


class TestSettingsDefaults:
    """Verify all default values match the PRD specification."""

    def test_default_data_source(self):
        s = Settings()
        assert s.primary_data_source == "yfinance"

    def test_default_screening_thresholds(self):
        s = Settings()
        assert s.mom_5d_min == 0.0
        assert s.atr_ratio_max == 0.8
        assert s.rsi_range_low == 25.0
        assert s.rsi_range_high == 75.0
        assert s.mfi_min_or_vol_anomaly == 40.0

    def test_default_sector_caps(self):
        s = Settings()
        assert s.sector_cap == 3
        assert s.industry_cap == 2

    def test_default_behavior_switches(self):
        s = Settings()
        assert s.evolution_weight_adjust_enabled is False
        assert s.llm_ablation_enabled is True

    def test_default_home_path_expanded(self):
        s = Settings()
        assert isinstance(s.alphascreener_home, Path)
        assert s.alphascreener_home == Path.home() / ".alphascreener"

    def test_home_path_with_custom_value_expanded(self):
        s = Settings(alphascreener_home="~/my_custom_path")
        assert s.alphascreener_home == Path.home() / "my_custom_path"


class TestSettingsEnvOverride:
    """Verify environment variables override defaults."""

    def test_override_via_constructor(self):
        s = Settings(mom_5d_min=0.05, sector_cap=5)
        assert s.mom_5d_min == 0.05
        assert s.sector_cap == 5

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("MOM_5D_MIN", "0.03")
        monkeypatch.setenv("SECTOR_CAP", "10")
        s = Settings()
        assert s.mom_5d_min == pytest.approx(0.03)
        assert s.sector_cap == 10

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MOM_5D_MIN", "0.01")
        s = Settings(mom_5d_min=0.02)
        assert s.mom_5d_min == pytest.approx(0.02)


    def test_behavior_switches_from_env(self, monkeypatch):
        monkeypatch.setenv("EVOLUTION_WEIGHT_ADJUST_ENABLED", "true")
        monkeypatch.setenv("LLM_ABLATION_ENABLED", "false")
        s = Settings()
        assert s.evolution_weight_adjust_enabled is True
        assert s.llm_ablation_enabled is False

    def test_extra_env_ignored(self, monkeypatch):
        """Unknown env vars should be silently ignored due to extra='ignore'."""
        monkeypatch.setenv("UNKNOWN_VAR", "should-be-ignored")
        s = Settings()
        assert not hasattr(s, "UNKNOWN_VAR")


class TestSettingsEnvFile:
    """Verify .env file loading."""

    def test_dotenv_file_loading(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("MOM_5D_MIN=0.01\nRSI_RANGE_LOW=20.0\n")

        # Clear env vars that may have been leaked by third-party packages
        # (e.g. tradingagents calls load_dotenv() at import time, which writes
        #  the projectʼs .env into os.environ).  Those real env vars take
        #  priority over _env_file in pydantic-settings, so we must remove
        #  them before the assertion.
        monkeypatch.delenv("MOM_5D_MIN", raising=False)
        monkeypatch.delenv("RSI_RANGE_LOW", raising=False)

        monkeypatch.chdir(tmp_path)
        s = Settings(_env_file=str(dotenv))
        assert s.mom_5d_min == pytest.approx(0.01)
        assert s.rsi_range_low == pytest.approx(20.0)

    def test_dotenv_file_with_home_path(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("ALPHASCREENER_HOME=/custom/data/path\n")

        monkeypatch.chdir(tmp_path)
        s = Settings(_env_file=str(dotenv))
        assert s.alphascreener_home == Path("/custom/data/path")


class TestGetDbUrl:
    """Tests for :meth:`Settings.get_db_url` with various legacy-DB states."""

    def test_explicit_db_url_takes_priority(self, tmp_path):
        home = tmp_path / "home"
        s = Settings(
            alphascreener_home=str(home),
            db_url="sqlite:///explicit/my.db",
        )
        assert s.get_db_url() == "sqlite:///explicit/my.db"

    def test_no_legacy_uses_new_path(self, tmp_path):
        home = tmp_path / "home"
        s = Settings(alphascreener_home=str(home))
        url = s.get_db_url()
        expected = f"sqlite:///{home / 'data' / 'alphascreener.db'}"
        assert url == expected

    def test_legacy_exists_but_empty_uses_new_path(self, tmp_path):
        import sqlite3

        home = tmp_path / "home"
        legacy = home / "alphabase.db"
        home.mkdir(parents=True, exist_ok=True)

        # Create an empty legacy DB with tables but 0 rows
        conn = sqlite3.connect(str(legacy))
        conn.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        s = Settings(alphascreener_home=str(home))
        url = s.get_db_url()
        expected = f"sqlite:///{home / 'data' / 'alphascreener.db'}"
        assert url == expected

    def test_legacy_exists_with_data_uses_legacy_path(self, tmp_path):
        import sqlite3

        home = tmp_path / "home"
        legacy = home / "alphabase.db"
        home.mkdir(parents=True, exist_ok=True)

        # Create a legacy DB with tables that actually have data
        conn = sqlite3.connect(str(legacy))
        conn.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY, msg TEXT)")
        conn.execute("INSERT INTO alerts VALUES (1, 'test alert')")
        conn.commit()
        conn.close()

        s = Settings(alphascreener_home=str(home))
        url = s.get_db_url()
        expected = f"sqlite:///{legacy}"
        assert url == expected

    def test_legacy_has_only_alembic_version_table_uses_new_path(self, tmp_path):
        """alembic_version is excluded from the data check."""
        import sqlite3

        home = tmp_path / "home"
        legacy = home / "alphabase.db"
        home.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(legacy))
        conn.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute(
            "INSERT INTO alembic_version VALUES ('abc123')"
        )
        conn.commit()
        conn.close()

        s = Settings(alphascreener_home=str(home))
        url = s.get_db_url()
        # alembic_version is excluded, so this counts as "no data"
        expected = f"sqlite:///{home / 'data' / 'alphascreener.db'}"
        assert url == expected

    def test_legacy_db_has_data_static_method(self, tmp_path):
        import sqlite3

        from alphascreener.config import Settings

        legacy = tmp_path / "test.db"

        # Empty DB -> False
        conn = sqlite3.connect(str(legacy))
        conn.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        assert Settings._legacy_db_has_data(legacy) is False

        # DB with data -> True
        conn = sqlite3.connect(str(legacy))
        conn.execute("INSERT INTO alerts VALUES (1)")
        conn.commit()
        conn.close()
        assert Settings._legacy_db_has_data(legacy) is True

    def test_legacy_db_has_data_nonexistent_file(self, tmp_path):
        from alphascreener.config import Settings

        nonexistent = tmp_path / "does_not_exist.db"
        assert Settings._legacy_db_has_data(nonexistent) is False
