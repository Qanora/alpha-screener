"""SQLAlchemy ORM models for all 9 core SQLite tables (Issue #85).

Reference: PRD 7.6.2 / 7.6.2.1.
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ============================================================================
# factor_versions — factor version config (PRD 6.5 factor JSON)
# ============================================================================


class FactorVersion(Base):
    __tablename__ = "factor_versions"
    __table_args__ = (
        CheckConstraint(
            "release_type IN ('MAJOR','MINOR','PATCH')",
            name="ck_factor_versions_release_type",
        ),
        {"comment": "因子版本配置；永久保留"},
    )

    version = mapped_column(Text, primary_key=True)  # e.g. "1.0.0"
    released_at = mapped_column(Text, nullable=False)  # TIMESTAMP
    config_json = mapped_column(Text, nullable=False)  # full factor config JSON
    parent_version = mapped_column(Text, nullable=True)  # previous version
    release_type = mapped_column(Text, nullable=True)  # MAJOR/MINOR/PATCH


# ============================================================================
# paper_trades — paper trading / virtual trading records
# ============================================================================


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    __table_args__ = (
        ForeignKeyConstraint(
            ["factor_version"],
            ["factor_versions.version"],
            name="fk_paper_trades_factor_version",
        ),
        Index("idx_paper_trades_signal_date", "signal_date"),
        {"comment": "Paper Trading / 实盘虚拟交易记录"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_date = mapped_column(Date, nullable=False)
    ticker = mapped_column(Text, nullable=False)
    rating = mapped_column(Text, nullable=False)  # Strong Buy/Buy/Hold/Avoid
    breakout_probability = mapped_column(Float, nullable=False)
    entry_price = mapped_column(Float, nullable=True)  # T+1 open buy price
    exit_price = mapped_column(Float, nullable=True)  # T+7 close or stop-loss
    exit_reason = mapped_column(Text, nullable=True)  # 'time' / 'stop_loss' / 'halt'
    pnl_pct = mapped_column(Float, nullable=True)
    factor_version = mapped_column(Text, nullable=False)
    created_at = mapped_column(Text, server_default=text("CURRENT_TIMESTAMP"))


# ============================================================================
# alerts — alert events (PRD 10.3 alert rules)
# ============================================================================


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('warning','critical')",
            name="ck_alerts_severity",
        ),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    triggered_at = mapped_column(Text, nullable=False)  # TIMESTAMP
    severity = mapped_column(Text, nullable=True)
    rule_name = mapped_column(Text, nullable=False)
    metric_value = mapped_column(Float, nullable=True)
    notes = mapped_column(Text, nullable=True)
    resolved_at = mapped_column(Text, nullable=True)


# ============================================================================
# llm_cost_daily — daily LLM cost aggregation
# ============================================================================


class LlmCostDaily(Base):
    __tablename__ = "llm_cost_daily"

    cost_date = mapped_column(Date, primary_key=True)
    total_usd = mapped_column(Float, nullable=False)
    call_count = mapped_column(Integer, nullable=False)
    by_module_json = mapped_column(Text, nullable=True)
    # {"refining": 0.05, "evolution": 0.01, ...}


# ============================================================================
# pid_lock — process mutex for serial execution (PRD 7.7.2)
# ============================================================================


class PidLock(Base):
    __tablename__ = "pid_lock"
    __table_args__ = (Index("idx_pid_lock_expires", "expires_at"),)

    lock_name = mapped_column(Text, primary_key=True)  # usually 'global'
    pid = mapped_column(Integer, nullable=False)
    task_id = mapped_column(Text, nullable=False)  # e.g. 'daily_scan'
    acquired_at = mapped_column(Text, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    expires_at = mapped_column(Text, nullable=False)  # acquired_at + task timeout + 10min buffer
    meta_json = mapped_column(Text, nullable=True)


# ============================================================================
# monitoring_samples — resource monitoring (RSS/CPU/FD samples)
# ============================================================================


class MonitoringSample(Base):
    __tablename__ = "monitoring_samples"
    __table_args__ = (
        Index("idx_monitoring_task_time", "task_id", "sampled_at"),
        {"comment": "资源监控采样；保留最近 30 天"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id = mapped_column(Text, nullable=False)  # aligns with pid_lock.task_id
    sampled_at = mapped_column(Text, nullable=False)  # TIMESTAMP
    rss_mb = mapped_column(Float, nullable=False)  # psutil resident memory (MB)
    cpu_percent = mapped_column(Float, nullable=False)  # 4-core normalized 0-400%
    open_fd_count = mapped_column(Integer, nullable=True)  # file descriptor count
    thread_count = mapped_column(Integer, nullable=True)
    notes = mapped_column(Text, nullable=True)


# ============================================================================
# alpha_acceptance_daily — alpha acceptance metrics (PRD 5.7)
# ============================================================================


class AlphaAcceptanceDaily(Base):
    __tablename__ = "alpha_acceptance_daily"
    __table_args__ = ({"comment": "Alpha 验收口径每日记录；永久保留（小表）"},)

    metric_date = mapped_column(Date, primary_key=True)
    base_rate = mapped_column(Float, nullable=False)
    precision_at_20_pure = mapped_column(Float, nullable=True)
    precision_at_20_llm = mapped_column(Float, nullable=True)
    precision_at_10_pure = mapped_column(Float, nullable=True)
    precision_at_10_llm = mapped_column(Float, nullable=True)
    lift_at_20_pure = mapped_column(Float, nullable=True)
    lift_at_20_llm = mapped_column(Float, nullable=True)
    ic_pure = mapped_column(Float, nullable=True)
    ic_llm = mapped_column(Float, nullable=True)
    bootstrap_ci_lower_pure = mapped_column(Float, nullable=True)
    bootstrap_ci_upper_pure = mapped_column(Float, nullable=True)
    bootstrap_ci_lower_llm = mapped_column(Float, nullable=True)
    bootstrap_ci_upper_llm = mapped_column(Float, nullable=True)
    sample_size = mapped_column(Integer, nullable=False)


# ============================================================================
# data_source_diff — fallback OHLCV cross-validation diffs (PRD 7.1)
# ============================================================================


class DataSourceDiff(Base):
    __tablename__ = "data_source_diff"
    __table_args__ = (
        CheckConstraint(
            "field IN ('open','high','low','close','volume')",
            name="ck_data_source_diff_field",
        ),
        CheckConstraint(
            "fallback_source IN ('stooq','alpaca','polygon')",
            name="ck_data_source_diff_fallback",
        ),
        Index("idx_data_source_diff_date", "metric_date"),
        {"comment": "备用 OHLCV 交叉校验差异；保留 365 天后归档至冷备份"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_date = mapped_column(Date, nullable=False)
    ticker = mapped_column(Text, nullable=False)
    field = mapped_column(Text, nullable=False)  # open/high/low/close/volume
    yfinance_value = mapped_column(Float, nullable=False)
    fallback_value = mapped_column(Float, nullable=False)
    fallback_source = mapped_column(Text, nullable=False)  # stooq/alpaca/polygon
    diff_pct = mapped_column(Float, nullable=False)
    alerted = mapped_column(Boolean, server_default=text("0"))


# ============================================================================
# factor_health_daily — CUSUM fast-monitoring time series (PRD 6.4.1)
# ============================================================================


class FactorHealthDaily(Base):
    __tablename__ = "factor_health_daily"
    __table_args__ = (
        Index("idx_factor_health_factor_date", "factor_name", "metric_date"),
        {"comment": "CUSUM 快速监控时序；保留 365 天后归档至冷备份"},
    )

    metric_date = mapped_column(Date, primary_key=True)
    factor_name = mapped_column(Text, primary_key=True)
    daily_ic = mapped_column(Float, nullable=True)
    rolling_ic_mean_90d = mapped_column(Float, nullable=True)
    cusum_value = mapped_column(Float, nullable=True)
    cusum_alert = mapped_column(Boolean, server_default=text("0"))
    consecutive_alerts = mapped_column(Integer, server_default=text("0"))
