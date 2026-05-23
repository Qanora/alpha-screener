"""Tests for alphascreener.config module."""

from pathlib import Path

from alphascreener.config import Settings


class TestSettingsDefaults:
    """Verify all default values match the PRD specification."""

    def test_default_data_source(self):
        s = Settings()
        assert s.primary_data_source == "yfinance"
        assert s.fallback_ohlcv_source == "stooq"
        assert s.fmp_api_key == ""
        assert s.fmp_tier == "free"
        assert s.fmp_daily_budget == 250
        assert s.stooq_base_url == "https://stooq.com/q/d/l/"

    def test_default_llm(self):
        s = Settings()
        assert s.openai_api_key == ""
        assert s.llm_model == "gpt-4o-mini"
        assert s.llm_rps == 5
        assert s.llm_batch_size == 3
        assert s.llm_max_concurrent_stage1 == 6

    def test_default_cost_thresholds(self):
        s = Settings()
        assert s.cost_l1_warning_daily_usd == 0.80
        assert s.cost_l2_degrade_daily_usd == 1.00
        assert s.cost_l3_savings_monthly_usd == 80.0
        assert s.cost_l4_circuit_monthly_usd == 95.0
        assert s.cost_budget_monthly_usd == 100

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

    def test_default_feishu(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "")
        monkeypatch.setenv("FEISHU_APP_SECRET", "")
        monkeypatch.setenv("FEISHU_TARGET_OPENID", "")
        s = Settings()
        assert s.feishu_app_id == ""
        assert s.feishu_app_secret == ""
        assert s.feishu_target_openid == ""
        assert s.feishu_push_enabled is True

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
        s = Settings(llm_model="gpt-4-turbo", llm_rps=10)
        assert s.llm_model == "gpt-4-turbo"
        assert s.llm_rps == 10

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_RPS", "3")
        s = Settings()
        assert s.llm_model == "gpt-4o"
        assert s.llm_rps == 3

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        s = Settings(llm_model="ctor-model")
        assert s.llm_model == "ctor-model"

    def test_fmp_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-fmp-key")
        s = Settings()
        assert s.fmp_api_key == "test-fmp-key"

    def test_openai_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        s = Settings()
        assert s.openai_api_key == "test-openai-key"

    def test_feishu_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "app-123")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret-abc")
        monkeypatch.setenv("FEISHU_TARGET_OPENID", "ou-xyz")
        monkeypatch.setenv("FEISHU_PUSH_ENABLED", "false")
        s = Settings()
        assert s.feishu_app_id == "app-123"
        assert s.feishu_app_secret == "secret-abc"
        assert s.feishu_target_openid == "ou-xyz"
        assert s.feishu_push_enabled is False

    def test_behavior_switches_from_env(self, monkeypatch):
        monkeypatch.setenv("EVOLUTION_WEIGHT_ADJUST_ENABLED", "true")
        monkeypatch.setenv("LLM_ABLATION_ENABLED", "false")
        s = Settings()
        assert s.evolution_weight_adjust_enabled is True
        assert s.llm_ablation_enabled is False

    def test_cost_budget_from_env(self, monkeypatch):
        monkeypatch.setenv("COST_BUDGET_MONTHLY_USD", "50")
        s = Settings()
        assert s.cost_budget_monthly_usd == 50

    def test_daily_budget_from_env(self, monkeypatch):
        monkeypatch.setenv("FMP_DAILY_BUDGET", "100")
        s = Settings()
        assert s.fmp_daily_budget == 100

    def test_extra_env_ignored(self, monkeypatch):
        """Unknown env vars should be silently ignored due to extra='ignore'."""
        monkeypatch.setenv("UNKNOWN_VAR", "should-be-ignored")
        s = Settings()
        assert not hasattr(s, "UNKNOWN_VAR")


class TestSettingsEnvFile:
    """Verify .env file loading."""

    def test_dotenv_file_loading(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("LLM_MODEL=gpt-4o-from-file\nLLM_RPS=8\n")

        monkeypatch.chdir(tmp_path)
        s = Settings(_env_file=str(dotenv))
        assert s.llm_model == "gpt-4o-from-file"
        assert s.llm_rps == 8

    def test_dotenv_file_with_home_path(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("ALPHASCREENER_HOME=/custom/data/path\n")

        monkeypatch.chdir(tmp_path)
        s = Settings(_env_file=str(dotenv))
        assert s.alphascreener_home == Path("/custom/data/path")
