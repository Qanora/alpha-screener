"""Paper Trade tracker: trade lifecycle management and P&L calculation.

Issue #102: Paper Trading tracker.
Reference: PRD 5.4.

Provides:
  - ExitReason: enum of valid exit reasons (time / stop_loss / halt).
  - calc_pnl_pct: static P&L calculation.
  - PaperTradeTracker: read/write paper_trades table, manage T+1 entry / T+7 exit
    lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from alphascreener.db.models import PaperTrade
from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ============================================================================
# ExitReason enum
# ============================================================================

_VALID_EXIT_REASONS: frozenset[str] = frozenset({"time", "stop_loss", "halt"})


class ExitReason:
    """Valid exit reasons for paper trades (PRD 5.4).

    Attributes:
        TIME: Trade closed at T+7 holding period expiry.
        STOP_LOSS: Trade stopped out at 8% loss.
        HALT: Trade closed due to trading halt / delisting.
    """

    TIME: str = "time"
    STOP_LOSS: str = "stop_loss"
    HALT: str = "halt"


def is_valid_exit_reason(reason: str | None) -> bool:
    """Check whether an exit_reason string is valid."""
    if reason is None:
        return False
    return reason in _VALID_EXIT_REASONS


# ============================================================================
# P&L calculation
# ============================================================================


def calc_pnl_pct(entry_price: float, exit_price: float) -> float:
    """Calculate P&L as a percentage.

    Formula: ``(exit_price - entry_price) / entry_price * 100``.

    Args:
        entry_price: T+1 open buy price.
        exit_price: T+7 close or stop-loss price.

    Returns:
        P&L percentage (e.g. 10.0 = +10%, -8.0 = -8%).

    Raises:
        ValueError: If *entry_price* is not positive.
    """
    if entry_price <= 0:
        raise ValueError(f"Entry price must be positive, got {entry_price}")
    return (exit_price - entry_price) / entry_price * 100.0


# ============================================================================
# PaperTradeTracker
# ============================================================================


class PaperTradeTracker:
    """Manage paper trading records in the ``paper_trades`` SQLite table.

    Provides lifecycle methods for entering and exiting simulated trades,
    querying open/closed positions, and computing P&L.

    Args:
        session_factory: Zero-arg callable returning a new SQLAlchemy ``Session``.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self._sf = session_factory

    # ------------------------------------------------------------------
    # Enter trade
    # ------------------------------------------------------------------

    def enter_trade(
        self,
        signal_date: date,
        ticker: str,
        rating: str,
        breakout_probability: float,
        factor_version: str,
        *,
        entry_price: float | None = None,
    ) -> int:
        """Create a new paper trade record.

        Args:
            signal_date: Date the signal fired (T).
            ticker: Stock ticker symbol.
            rating: Analyst rating (e.g. Strong Buy, Buy, Hold, Avoid).
            breakout_probability: Model-predicted breakout probability (0-1).
            factor_version: Factor version identifier (e.g. "1.0.0").
            entry_price: T+1 open buy price.  May be None if the price is
                not yet known at signal time; can be updated later by calling
                :meth:`set_entry_price`.

        Returns:
            The auto-generated ``id`` of the new trade record.

        Raises:
            ValueError: If *entry_price* is provided and is not positive.
        """
        if entry_price is not None and entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        with self._sf() as session:
            trade = PaperTrade(
                signal_date=signal_date,
                ticker=ticker,
                rating=rating,
                breakout_probability=breakout_probability,
                entry_price=entry_price,
                factor_version=factor_version,
            )
            session.add(trade)
            session.flush()  # populate auto-increment id
            trade_id: int = trade.id
            session.commit()

        _logger.info(
            "Paper trade entered: id=%d ticker=%s rating=%s signal_date=%s",
            trade_id,
            ticker,
            rating,
            signal_date.isoformat(),
        )
        return trade_id

    # ------------------------------------------------------------------
    # Set entry price (for deferred T+1 fill)
    # ------------------------------------------------------------------

    def set_entry_price(self, trade_id: int, entry_price: float) -> None:
        """Update the entry price of an existing trade.

        Used when the T+1 open price was not known at signal time.

        Args:
            trade_id: The trade record ID.
            entry_price: T+1 open buy price.

        Raises:
            ValueError: If trade not found or entry_price is not positive.
        """
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        with self._sf() as session:
            trade = session.get(PaperTrade, trade_id)
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            trade.entry_price = entry_price
            session.commit()
        _logger.info("Paper trade %d: entry_price set to %0.2f", trade_id, entry_price)

    # ------------------------------------------------------------------
    # Exit trade
    # ------------------------------------------------------------------

    def exit_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> float:
        """Close an open trade, recording exit_price, exit_reason, and pnl_pct.

        Args:
            trade_id: The trade record ID.
            exit_price: T+7 close price or stop-loss trigger price.
            exit_reason: One of ``"time"``, ``"stop_loss"``, ``"halt"``.

        Returns:
            The computed P&L percentage.

        Raises:
            ValueError: If trade not found, trade already closed, entry_price
                is None, or exit_reason is invalid.
        """
        if not is_valid_exit_reason(exit_reason):
            raise ValueError(
                f"Invalid exit_reason: {exit_reason!r}. "
                f"Must be one of: {sorted(_VALID_EXIT_REASONS)}"
            )

        with self._sf() as session:
            trade = session.get(PaperTrade, trade_id)
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            if trade.exit_price is not None:
                _existing_exit = trade.exit_price
                _existing_reason = trade.exit_reason
                raise ValueError(
                    f"Trade {trade_id} already closed "
                    f"(exit_price={_existing_exit}, reason={_existing_reason})"
                )
            if trade.entry_price is None:
                raise ValueError(
                    f"Trade {trade_id} has no entry_price; call set_entry_price() before closing"
                )

            pnl = calc_pnl_pct(trade.entry_price, exit_price)
            _ticker = trade.ticker
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.pnl_pct = pnl
            session.commit()

        _logger.info(
            "Paper trade exited: id=%d ticker=%s exit_reason=%s pnl=%0.2f%%",
            trade_id,
            _ticker,
            exit_reason,
            pnl,
        )
        return pnl

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_trade(self, trade_id: int) -> PaperTrade | None:
        """Retrieve a single trade by ID.

        Returns:
            The :class:`PaperTrade` ORM instance, or None if not found.
        """
        with self._sf() as session:
            return session.get(PaperTrade, trade_id)

    def get_open_trades(self) -> list[PaperTrade]:
        """Return all trades that have not yet been closed.

        Returns:
            List of :class:`PaperTrade` instances with ``exit_price IS NULL``,
            ordered by ``signal_date DESC``.
        """
        with self._sf() as session:
            stmt = (
                select(PaperTrade)
                .where(PaperTrade.exit_price.is_(None))
                .order_by(PaperTrade.signal_date.desc())
            )
            return list(session.scalars(stmt).all())

    def get_trade_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[PaperTrade]:
        """Return completed (closed) trades.

        Args:
            limit: Optional maximum number of trades to return.

        Returns:
            List of :class:`PaperTrade` instances with ``exit_price IS NOT NULL``,
            ordered by ``signal_date DESC``.
        """
        with self._sf() as session:
            stmt = (
                select(PaperTrade)
                .where(PaperTrade.exit_price.is_not(None))
                .order_by(PaperTrade.signal_date.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def get_pnl_summary(self) -> dict[str, float | int]:
        """Compute summary statistics across all closed trades.

        Returns:
            Dict with keys: ``n_trades``, ``avg_pnl_pct``, ``win_rate``,
            ``total_pnl_pct``.
        """
        with self._sf() as session:
            stmt = select(PaperTrade).where(PaperTrade.pnl_pct.is_not(None))
            trades: list[PaperTrade] = list(session.scalars(stmt).all())

        if not trades:
            return {
                "n_trades": 0,
                "avg_pnl_pct": 0.0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
            }

        pnls = [t.pnl_pct for t in trades]
        n = len(pnls)
        avg = sum(pnls) / n
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)

        return {
            "n_trades": n,
            "avg_pnl_pct": round(avg, 4),
            "win_rate": round(wins / n, 4),
            "total_pnl_pct": round(total, 4),
        }
