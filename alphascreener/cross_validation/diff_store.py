"""Persistence layer for ``data_source_diff`` records.

Issue #91: Stooq fallback adapter + cross-validation.
Reference: PRD 7.2.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import func, select

from alphascreener.cross_validation.comparator import OHLCVFieldDiffs
from alphascreener.db.engine import create_db_engine
from alphascreener.db.models import Base, DataSourceDiff
from alphascreener.logging import get_logger


class DiffStore:
    """Persist and query ``data_source_diff`` records.

    Writes field-level OHLCV diffs to the ``data_source_diff`` table and
    provides aggregate query methods for alert threshold evaluation.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._engine = create_db_engine(db_path)
        Base.metadata.create_all(self._engine)
        self._logger: logging.Logger = get_logger("screening")

    # -- Write ------------------------------------------------------------------

    def insert_diffs(self, diffs: OHLCVFieldDiffs) -> int:
        """Insert diff records into the ``data_source_diff`` table.

        Args:
            diffs: Container with diff records from ``compare_ohlcv_dataframes``.

        Returns:
            Number of rows inserted.
        """
        if not diffs.records:
            return 0

        from sqlalchemy.orm import Session

        count = 0
        with Session(self._engine) as session:
            for rec in diffs.records:
                row = DataSourceDiff(
                    metric_date=rec["dt"],
                    ticker=rec["ticker"],
                    field=rec["field"],
                    yfinance_value=rec["primary_value"],
                    fallback_value=rec["fallback_value"],
                    fallback_source=rec["fallback_source"],
                    diff_pct=rec["diff_pct"],
                )
                session.add(row)
                count += 1
            session.commit()

        self._logger.info("Inserted %d data_source_diff records", count)
        return count

    # -- Query ------------------------------------------------------------------

    def count_daily_diffs(self, metric_date: date | None = None) -> int:
        """Count the number of ``data_source_diff`` records on a given date.

        Args:
            metric_date: date to query. Defaults to today (local time).

        Returns:
            Number of diff records for that date.
        """
        from datetime import date as today_date

        if metric_date is None:
            metric_date = today_date.today()

        from sqlalchemy.orm import Session

        with Session(self._engine) as session:
            count = session.scalar(
                select(func.count())
                .select_from(DataSourceDiff)
                .where(DataSourceDiff.metric_date == metric_date)
            )
            return count or 0

    def count_daily_diff_tickers(self, metric_date: date | None = None) -> int:
        """Count distinct tickers with diffs on a given date.

        Args:
            metric_date: date to query. Defaults to today.

        Returns:
            Number of distinct tickers that have at least one diff on that date.
        """
        from datetime import date as today_date

        if metric_date is None:
            metric_date = today_date.today()

        from sqlalchemy.orm import Session

        with Session(self._engine) as session:
            result = session.execute(
                select(func.count(func.distinct(DataSourceDiff.ticker)))
                .select_from(DataSourceDiff)
                .where(DataSourceDiff.metric_date == metric_date)
            )
            return result.scalar() or 0

    def mark_alerted(self, metric_date: date | None = None) -> int:
        """Mark all diffs on a given date as alerted.

        Args:
            metric_date: date to query. Defaults to today.

        Returns:
            Number of rows updated.
        """
        from datetime import date as today_date

        if metric_date is None:
            metric_date = today_date.today()

        from sqlalchemy.orm import Session

        with Session(self._engine) as session:
            result = (
                session.query(DataSourceDiff)
                .filter(
                    DataSourceDiff.metric_date == metric_date,
                    DataSourceDiff.alerted == False,  # noqa: E712
                )
                .update({"alerted": True})
            )
            session.commit()
            return result


# ---------------------------------------------------------------------------
# Module-level convenience functions (stateless, caller provides engine)
# ---------------------------------------------------------------------------


def count_daily_diffs(
    engine: Any,
    metric_date: date | None = None,
) -> int:
    """Count ``data_source_diff`` records for a given date.

    Stateless convenience function — the caller provides the SQLAlchemy engine.
    """
    from datetime import date as today_date

    if metric_date is None:
        metric_date = today_date.today()

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        count = session.scalar(
            select(func.count())
            .select_from(DataSourceDiff)
            .where(DataSourceDiff.metric_date == metric_date)
        )
        return count or 0


def mark_alerted_diffs(
    engine: Any,
    metric_date: date | None = None,
) -> int:
    """Mark all diffs on a given date as alerted.

    Stateless convenience function.
    """
    from datetime import date as today_date

    if metric_date is None:
        metric_date = today_date.today()

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        result = (
            session.query(DataSourceDiff)
            .filter(
                DataSourceDiff.metric_date == metric_date,
                DataSourceDiff.alerted == False,  # noqa: E712
            )
            .update({"alerted": True})
        )
        session.commit()
        return result
