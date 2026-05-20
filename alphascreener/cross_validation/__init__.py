"""Cross-validation module for OHLCV data sources.

Issue #91: Stooq fallback adapter + cross-validation.

Provides field-level comparison between primary (yfinance) and fallback (Stooq)
OHLCV data, diff persistence to the ``data_source_diff`` table, yfinance health
monitoring with automatic fallback switching, and Lark alert integration.

Reference: PRD 7.2.
"""

from alphascreener.cross_validation.comparator import (
    OHLCVFieldDiffs,
    compare_ohlcv_dataframes,
    compute_diff_pct,
)
from alphascreener.cross_validation.diff_store import (
    DiffStore,
    count_daily_diffs,
    mark_alerted_diffs,
)
from alphascreener.cross_validation.health_monitor import (
    DEFAULT_FAILURE_THRESHOLD_PCT,
    DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS,
    YFinanceHealthMonitor,
)

__all__ = [
    "OHLCVFieldDiffs",
    "compare_ohlcv_dataframes",
    "compute_diff_pct",
    "DiffStore",
    "count_daily_diffs",
    "mark_alerted_diffs",
    "YFinanceHealthMonitor",
    "DEFAULT_FAILURE_THRESHOLD_PCT",
    "DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS",
]
