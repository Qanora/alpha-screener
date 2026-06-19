"""Application configuration via pydantic-settings, loaded from environment / .env."""

import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

_logger = logging.getLogger(__name__)


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
    openai_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_rps: int = 5
    llm_batch_size: int = 3
    llm_max_concurrent_stage1: int = 6
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0

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
        return self.alphascreener_home / "data" / "alphascreener.db"

    @staticmethod
    def _legacy_db_has_data(legacy_path: Path) -> bool:
        """Check whether a legacy SQLite database contains any data in user tables.

        Returns ``True`` if at least one user table has one or more rows.
        Returns ``False`` when all user tables are empty or the file cannot
        be read.
        """
        import sqlite3

        try:
            conn = sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True)
        except Exception:
            _logger.debug(
                "Could not open legacy database at %s for inspection.",
                legacy_path,
                exc_info=True,
            )
            return False

        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
                "  AND name NOT LIKE 'sqlite_%'"
                "  AND name != 'alembic_version'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                return False
            for table in tables:
                # Use a quoted identifier so table names with special
                # characters do not break the query.
                row = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()
                if row and row[0] > 0:
                    return True
            return False
        except Exception:
            _logger.debug(
                "Could not inspect legacy database at %s.",
                legacy_path,
                exc_info=True,
            )
            return False
        finally:
            conn.close()

    def get_db_url(self) -> str:
        """Return the SQLAlchemy database URL.

        Uses ``db_url`` if explicitly set (via env ``DB_URL``), otherwise
        derives a ``sqlite:///`` URL from ``alphascreener_home``.

        When no explicit ``db_url`` is given, the old default location
        ``<home>/alphabase.db`` is checked first.  If it exists **and**
        contains data the legacy path is used so existing installations do
        not lose data, and a WARNING is logged advising the operator to
        migrate.  An empty legacy database (tables present but 0 rows) is
        silently ignored in favour of the new default path.
        """
        if self.db_url:
            return self.db_url

        legacy_path = self.alphascreener_home / "alphabase.db"
        if legacy_path.exists() and self._legacy_db_has_data(legacy_path):
            _logger.warning(
                "Legacy database found at %s with data.  "
                "The default path has moved to %s.  "
                "Move the file to the new location to suppress this warning.",
                legacy_path,
                self.db_path,
            )
            return f"sqlite:///{legacy_path}"

        if legacy_path.exists():
            _logger.info(
                "Legacy database found at %s but it is empty.  "
                "Using new default path %s.",
                legacy_path,
                self.db_path,
            )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.db_path}"
