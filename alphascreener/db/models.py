"""SQLAlchemy ORM models for all 9 core SQLite tables (Issue #85).

Reference: PRD 7.6.2 / 7.6.2.1.
"""

from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    Integer,
    Text,
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
    threshold = mapped_column(Float, nullable=True)


