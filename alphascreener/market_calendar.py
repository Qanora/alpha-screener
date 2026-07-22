"""Stable New York Stock Exchange session calendar."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import exchange_calendars as xcals
import polars as pl


@lru_cache(maxsize=1)
def _xnys_calendar():
    """Return the process-wide NYSE calendar instance."""
    return xcals.get_calendar("XNYS")


def market_dates_between(start: date, end: date) -> list[date]:
    """Return stable NYSE session labels in an inclusive date range."""
    if end < start:
        return []
    sessions = _xnys_calendar().sessions_in_range(start.isoformat(), end.isoformat())
    return [value.date() for value in sessions.to_pydatetime()]


def infer_market_dates(ohlcv: pl.DataFrame) -> list[date]:
    """Return NYSE sessions spanning the observed OHLCV date range.

    The name is retained for internal callers, but the dates are not inferred
    from panel size or SPY availability.  This makes historical horizons stable
    as the cache and listed universe change later.
    """
    required = {"ticker", "dt"}
    if missing := required - set(ohlcv.columns):
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    bounds = ohlcv.select(
        pl.col("dt").cast(pl.Date).min().alias("start"),
        pl.col("dt").cast(pl.Date).max().alias("end"),
    ).row(0, named=True)
    if bounds["start"] is None or bounds["end"] is None:
        raise ValueError("market calendar unavailable")
    dates = market_dates_between(bounds["start"], bounds["end"])
    if not dates:
        raise ValueError("market calendar unavailable")
    return dates


def future_market_date(decision_date: date, sessions: int) -> date:
    """Return the date exactly ``sessions`` NYSE sessions after a decision."""
    if sessions < 0:
        raise ValueError("sessions must be non-negative")
    calendar = _xnys_calendar()
    if not calendar.is_session(decision_date.isoformat()):
        raise ValueError(f"{decision_date} is not an NYSE session")
    window = calendar.sessions_window(decision_date.isoformat(), sessions + 1)
    return window[-1].date()


def latest_completed_market_date(now: datetime | None = None) -> date:
    """Return the latest NYSE session whose regular close has passed."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(UTC)
    calendar = _xnys_calendar()
    candidates = market_dates_between(
        current.date() - timedelta(days=14),
        current.date() + timedelta(days=1),
    )
    for session in reversed(candidates):
        close = calendar.session_close(session.isoformat()).to_pydatetime()
        if close <= current:
            return session
    raise ValueError("no completed NYSE session available")


def require_spy(ohlcv: pl.DataFrame) -> None:
    """Require at least one SPY observation without using it as the calendar."""
    if ohlcv.filter(pl.col("ticker") == "SPY").is_empty():
        raise ValueError("SPY market calendar unavailable")
