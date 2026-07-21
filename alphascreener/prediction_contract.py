"""Immutable contract for the US-equity breakout prediction problem.

The screener ranks a liquid US-equity universe using information available on a
decision date.  It uses at most the previous 60 trading sessions to estimate
which candidates are most likely to have explosive positive returns over the
following 14 trading sessions.  This module intentionally contains no model
or data-provider logic: it is the shared definition that makes those later
components testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

INPUT_LOOKBACK_SESSIONS = 60
FORECAST_HORIZON_SESSIONS = 14
DEFAULT_TOP_K = 10
DEFAULT_ABSOLUTE_HIT_RETURN = 0.15
DEFAULT_CROSS_SECTION_HIT_QUANTILE = 0.95
STRATEGY_VERSION = "rank-v2"


@dataclass(frozen=True)
class ExplosionLabelSpec:
    """Pre-registered definition of a future 14-session breakout event.

    A stock is a hit only when its forward return clears both an economically
    meaningful absolute threshold and the configured cross-sectional tail.
    The latter prevents a calm market from labelling ordinary moves as
    explosive, while the former prevents a volatile market from labelling a
    small positive move as a hit.
    """

    horizon_sessions: int = FORECAST_HORIZON_SESSIONS
    absolute_return: float = DEFAULT_ABSOLUTE_HIT_RETURN
    cross_section_quantile: float = DEFAULT_CROSS_SECTION_HIT_QUANTILE

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive")
        if self.absolute_return <= 0:
            raise ValueError("absolute_return must be positive")
        if not 0.0 < self.cross_section_quantile < 1.0:
            raise ValueError("cross_section_quantile must be between 0 and 1")

    def threshold(self, forward_returns: Sequence[float]) -> float:
        """Return the larger of the absolute and empirical tail thresholds."""
        if not forward_returns:
            raise ValueError("forward_returns must not be empty")
        ordered = sorted(float(value) for value in forward_returns)
        index = max(0, ceil(len(ordered) * self.cross_section_quantile) - 1)
        return max(self.absolute_return, ordered[index])
