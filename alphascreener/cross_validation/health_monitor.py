"""YFinance health monitoring with automatic fallback switching.

Issue #91: Stooq fallback adapter + cross-validation.
Reference: PRD 7.2.

Tracks yfinance daily OHLCV failure rates and triggers a full switch to the
fallback source when failures exceed 30% for 3 consecutive days.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_THRESHOLD_PCT: float = 30.0
DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS: int = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DailyHealthRecord:
    """Health snapshot for a single day's yfinance OHLCV operations.

    Attributes:
        date: The trading date.
        total_tickers: Number of tickers requested.
        failed_tickers: Number of tickers that failed (no data / error).
        failure_rate_pct: Failure rate as a percentage (0-100).
    """

    date: date
    total_tickers: int = 0
    failed_tickers: int = 0

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.total_tickers < 0:
            raise ValueError(
                f"total_tickers must be >= 0, got {self.total_tickers}"
            )
        if self.failed_tickers < 0:
            raise ValueError(
                f"failed_tickers must be >= 0, got {self.failed_tickers}"
            )
        if self.failed_tickers > self.total_tickers:
            raise ValueError(
                f"failed_tickers ({self.failed_tickers}) must be <= "
                f"total_tickers ({self.total_tickers})"
            )

    @property
    def failure_rate_pct(self) -> float:
        """Failure rate as a percentage."""
        if self.total_tickers == 0:
            return 0.0
        return (self.failed_tickers / self.total_tickers) * 100.0


# ---------------------------------------------------------------------------
# Alert interface (reserved for Lark integration)
# ---------------------------------------------------------------------------


def _send_lark_alert(message: str, severity: str = "critical") -> None:
    """Send an alert via Lark (Feishu).

    This is a reserved interface for Lark/Feishu integration. Currently logs
    the alert; actual Lark API integration will be wired in a future issue.

    Args:
        message: Alert message text.
        severity: Alert severity level (warning/critical).
    """
    logger = get_logger("screening")
    logger.warning("Lark alert (%s): %s", severity, message)
    # Future: call feishu webhook API here


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------


@dataclass
class YFinanceHealthMonitor:
    """Monitor yfinance daily OHLCV health and trigger fallback switching.

    Tracks per-day failure rates and detects when failures exceed the configured
    threshold for consecutive days, indicating a systemic yfinance outage.

    When the full-switch condition is met, a critical alert is dispatched and
    the ``fallback_activated`` flag is set to ``True``, signaling downstream
    consumers to route all OHLCV requests through the fallback adapter instead
    of yfinance.

    Args:
        failure_threshold_pct: Failure rate percentage threshold (default 30%).
        consecutive_days: Number of consecutive days above threshold before
            triggering full switch (default 3).
    """

    failure_threshold_pct: float = DEFAULT_FAILURE_THRESHOLD_PCT
    consecutive_days: int = DEFAULT_FULL_SWITCH_CONSECUTIVE_DAYS

    # -- Internal state ----------------------------------------------------------

    _daily_history: list[DailyHealthRecord] = field(default_factory=list, repr=False)
    _fallback_activated: bool = field(default=False, repr=False)
    _consecutive_exceeded: int = field(default=0, repr=False)
    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("screening"), repr=False
    )

    def __post_init__(self) -> None:
        """Validate constructor parameters."""
        if self.consecutive_days <= 0:
            raise ValueError(
                f"consecutive_days must be > 0, got {self.consecutive_days}"
            )
        if not (0.0 <= self.failure_threshold_pct <= 100.0):
            raise ValueError(
                f"failure_threshold_pct must be in [0, 100], "
                f"got {self.failure_threshold_pct}"
            )

    # -- Properties --------------------------------------------------------------

    @property
    def fallback_activated(self) -> bool:
        """Whether full fallback switch has been activated."""
        return self._fallback_activated

    @property
    def consecutive_exceeded(self) -> int:
        """Number of consecutive days the failure threshold has been exceeded."""
        return self._consecutive_exceeded

    # -- Daily recording ---------------------------------------------------------

    def record_day(self, record: DailyHealthRecord) -> None:
        """Record a day's yfinance health statistics.

        Updates the consecutive-exceeded counter based on whether the failure
        rate exceeds the threshold. Consecutive counting uses **calendar dates**:
        only records whose ``date`` is exactly 1 day after the previous record
        count as consecutive. A gap > 1 day resets the counter before judging
        the current record. If the counter reaches ``consecutive_days``, the
        full fallback switch is activated and a critical alert is sent.

        Args:
            record: Daily health statistics.
        """
        self._daily_history.append(record)

        # Determine calendar gap from the previous record
        if len(self._daily_history) >= 2:
            prev_date = self._daily_history[-2].date  # second-to-last
            delta_days = (record.date - prev_date).days
        else:
            delta_days = 1  # first record — treat as the start of a fresh day

        if record.failure_rate_pct > self.failure_threshold_pct:
            if delta_days == 0:
                # Same calendar date — don't double-count
                pass
            elif delta_days == 1:
                self._consecutive_exceeded += 1
            else:
                # Gap > 1 day — reset and start a new streak
                self._consecutive_exceeded = 1
            self._logger.warning(
                "yfinance failure rate %.1f%% (≥ %.1f%%) for %s — "
                "consecutive days exceeded: %d/%d",
                record.failure_rate_pct,
                self.failure_threshold_pct,
                record.date.isoformat(),
                self._consecutive_exceeded,
                self.consecutive_days,
            )
        else:
            # Reset counter on a new healthy day (skip same-day repeats)
            if delta_days >= 1 and self._consecutive_exceeded > 0:
                self._logger.info(
                    "yfinance recovered — resetting consecutive exceeded counter (was %d)",
                    self._consecutive_exceeded,
                )
            if delta_days >= 1:
                self._consecutive_exceeded = 0

        # Check if full switch should be triggered
        if self._consecutive_exceeded >= self.consecutive_days and not self._fallback_activated:
            self._fallback_activated = True
            self._logger.warning(
                "FULL SWITCH: yfinance failure rate ≥ %.1f%% for %d consecutive days — "
                "routing all OHLCV to fallback source",
                self.failure_threshold_pct,
                self.consecutive_days,
            )
            _send_lark_alert(
                f"yfinance 全量切换备用源：连续 {self.consecutive_days} 天"
                f"失败率 ≥ {self.failure_threshold_pct}%",
                severity="critical",
            )

    def record_today(
        self,
        total_tickers: int,
        failed_tickers: int,
        today: date | None = None,
    ) -> None:
        """Convenience method to record today's health stats.

        Args:
            total_tickers: Total tickers requested today.
            failed_tickers: Number that failed.
            today: date to record. Defaults to today (local time).
        """
        if today is None:
            today = date.today()
        self.record_day(
            DailyHealthRecord(
                date=today, total_tickers=total_tickers, failed_tickers=failed_tickers
            )
        )

    # -- Reset -------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state (for testing or fresh sessions)."""
        self._daily_history.clear()
        self._fallback_activated = False
        self._consecutive_exceeded = 0
        self._logger.info("Health monitor reset")

    # -- History -----------------------------------------------------------------

    @property
    def history(self) -> list[DailyHealthRecord]:
        """Return a copy of the daily history."""
        return list(self._daily_history)
