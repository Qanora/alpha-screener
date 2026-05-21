"""Application configuration via pydantic-settings, loaded from environment / .env."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Alpha Screener configuration loaded from environment variables and .env file.

    All values have sensible defaults so the app can start without a .env file.
    Secrets (API keys) default to empty strings and must be provided at runtime.
    """

    # ====== 数据源 ======
    primary_data_source: str = "yfinance"
    fallback_ohlcv_source: str = "stooq"
    fmp_api_key: str = ""
    fmp_tier: str = "free"
    fmp_daily_budget: int = 250
    stooq_base_url: str = "https://stooq.com/q/d/l/"

    # ====== LLM ======
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_rps: int = 5
    llm_batch_size: int = 3
    llm_max_concurrent_stage1: int = 6

    # ====== 成本熔断阈值 ======
    cost_l1_warning_daily_usd: float = 0.80
    cost_l2_degrade_daily_usd: float = 1.00
    cost_l3_savings_monthly_usd: float = 80.0
    cost_l4_circuit_monthly_usd: float = 95.0
    cost_budget_monthly_usd: int = 100

    # ====== 粗筛阈值 ======
    mom_5d_min: float = 0.0
    atr_ratio_max: float = 0.8
    rsi_range_low: float = 25.0
    rsi_range_high: float = 75.0
    mfi_min_or_vol_anomaly: float = 40.0

    # ====== 行业去重 ======
    sector_cap: int = 3
    industry_cap: int = 2

    # ====== 飞书推送 ======
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_target_openid: str = ""
    feishu_push_enabled: bool = True

    # ====== 系统行为开关 ======
    evolution_weight_adjust_enabled: bool = False
    llm_ablation_enabled: bool = True

    # ====== 路径 ======
    alphascreener_home: Path = Path("~/.alphascreener")
    db_url: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("alphascreener_home", mode="before")
    @classmethod
    def _expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser()

    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database file."""
        return self.alphascreener_home / "alphabase.db"

    def get_db_url(self) -> str:
        """Return the SQLAlchemy database URL.

        Uses ``db_url`` if explicitly set (via env ``DB_URL``), otherwise
        derives a ``sqlite:///`` URL from ``alphascreener_home``.
        """
        if self.db_url:
            return self.db_url
        self.alphascreener_home.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.db_path}"
