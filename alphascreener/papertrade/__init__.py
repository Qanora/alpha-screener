"""Paper Trading tracker and graduation condition checks.

Issue #102: Paper Trading tracker.
Reference: PRD 5.4 / 7.6.2.1.
"""

from alphascreener.papertrade.graduation import (
    EngineeringGraduationResult,
    StrategyGraduationResult,
    check_engineering_graduation,
    check_strategy_graduation,
)
from alphascreener.papertrade.tracker import (
    ExitReason,
    PaperTradeTracker,
    calc_pnl_pct,
    is_valid_exit_reason,
)

__all__ = [
    "ExitReason",
    "PaperTradeTracker",
    "calc_pnl_pct",
    "is_valid_exit_reason",
    "EngineeringGraduationResult",
    "StrategyGraduationResult",
    "check_engineering_graduation",
    "check_strategy_graduation",
]
