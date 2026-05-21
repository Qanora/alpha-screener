"""CUSUM fast-layer factor health monitoring (Issue #103).

Reference: PRD 6.1.1 / 6.3.

Daily CUSUM calculation for each factor:
  S_t = max(0, S_{t-1} + IC_t - mu_IC - k)

Three alert levels:
  L1: Single factor CUSUM > h  =>  Feishu warning + factor_health_daily record
  L2: Factor triggers >= 2 times in 5-day window  =>  active -> degraded (weight freeze)
  L3: >= 5 factors trigger simultaneously  =>  low-activity mode + critical alert

Environment: EVOLUTION_WEIGHT_ADJUST_ENABLED=false (MVP — no weight adjustment).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from alphascreener.config import Settings
from alphascreener.db.models import Alert, FactorHealthDaily
from alphascreener.logging import get_logger

_logger = get_logger("monitoring")

# Session factory type
_SessionFactory = Callable[[], Session]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CUSUMConfig:
    """CUSUM threshold constants.

    All values are defaults from PRD 6.1.1; override via constructor kwargs.
    """

    k: float = 0.005  # allowance (slack) parameter
    h: float = 0.05  # decision interval (threshold)
    rolling_window_days: int = 90  # mu_IC rolling window
    l2_window_days: int = 5  # L2 lookback window
    l2_trigger_count: int = 2  # triggers needed in window for L2
    l3_trigger_count: int = 5  # simultaneous factor triggers for L3


# ---------------------------------------------------------------------------
# Core CUSUM formula
# ---------------------------------------------------------------------------


def _compute_cusum(
    ic_t: float,
    s_prev: float,
    mu_ic: float,
    k: float = 0.005,
) -> float:
    """Compute one-step CUSUM: S_t = max(0, S_{t-1} + IC_t - mu_IC - k).

    Args:
        ic_t: Current day's Information Coefficient for the factor.
        s_prev: Previous CUSUM value S_{t-1} (0.0 if first day).
        mu_ic: Rolling mean of IC (over ``rolling_window_days``).
        k: Allowance parameter (default 0.005).

    Returns:
        Updated CUSUM value S_t >= 0.
    """
    raw = s_prev + ic_t - mu_ic - k
    return max(0.0, raw)


def _rolling_mean(values: list[float], window: int) -> float:
    """Compute mean of the last *window* values.

    Args:
        values: Ordered list of values (oldest first).
        window: Number of most recent values to include.

    Returns:
        Mean of the last ``min(len(values), window)`` values.
        Returns 0.0 if *values* is empty.
    """
    if not values:
        return 0.0
    recent = values[-window:] if len(values) > window else values
    return sum(recent) / len(recent)


# ---------------------------------------------------------------------------
# Feishu notification stub
# ---------------------------------------------------------------------------


def _send_feishu_notification(message: str, severity: str = "warning") -> None:
    """Send an alert via Feishu (Lark).

    Currently logs the alert. Actual Feishu API integration is reserved for
    Issue #104 (Feishu daily card push).

    Args:
        message: Alert message text.
        severity: Alert severity level (warning/critical).
    """
    settings = Settings()
    if not settings.feishu_push_enabled:
        _logger.debug("Feishu push disabled, skipping notification")
        return
    _logger.warning("Feishu alert (%s): %s", severity, message)
    # Future: call feishu webhook / message API here (Issue #104)


# ---------------------------------------------------------------------------
# CUSUM Monitor
# ---------------------------------------------------------------------------


class CUSUMMonitor:
    """Daily CUSUM factor health monitor with 3-tier alerting.

    Usage::

        def make_session():
            return Session(engine)

        monitor = CUSUMMonitor(session_factory=make_session)
        results = monitor.run(
            metric_date=date.today(),
            daily_ics={"MOM_5D": 0.06, "PTH": 0.04, ...},
        )
        # results["l1_triggers"]  -> set of factor names that triggered L1
        # results["l2_suspended"] -> set of factor names that triggered L2
        # results["l3_triggered"] -> bool, global downgrade
    """

    def __init__(
        self,
        session_factory: _SessionFactory,
        config: CUSUMConfig | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.config = config or CUSUMConfig()

    # -- public API ------------------------------------------------------------

    def run(
        self,
        metric_date: date,
        daily_ics: dict[str, float | None],
    ) -> dict[str, Any]:
        """Run daily CUSUM computation for all factors.

        For each factor:
          1. Retrieve previous CUSUM value S_{t-1} from DB.
          2. Retrieve rolling IC history, compute mu_IC.
          3. Compute S_t = max(0, S_{t-1} + IC_t - mu_IC - k).
          4. Check L1: S_t >= h  =>  alert + record.
          5. Per factor, check L2: >= 2 L1 triggers in 5-day window.
          6. Check L3: >= 5 simultaneous L1 triggers => critical.

        Args:
            metric_date: The observation date for this CUSUM run.
            daily_ics: Dict mapping factor_name -> daily IC value.
                None IC values are silently skipped.

        Returns:
            Dict with keys ``l1_triggers`` (set[str]), ``l2_suspended``
            (set[str]), ``l3_triggered`` (bool), and ``records_written`` (int).
        """
        l1_triggers: set[str] = set()
        l2_suspended: set[str] = set()
        records_written: int = 0

        _logger.info(
            "CUSUM monitor: processing %d factors for %s",
            len(daily_ics),
            metric_date.isoformat(),
        )

        for factor_name, ic_t in daily_ics.items():
            if ic_t is None:
                _logger.debug("CUSUM: skipping %s (null IC)", factor_name)
                continue

            # 1. Retrieve S_{t-1}
            s_prev = self._get_previous_cusum(factor_name, metric_date)

            # 2. Retrieve IC history and compute rolling mean
            ic_history = self._get_ic_history(factor_name, before_date=metric_date)
            mu_ic = _rolling_mean(ic_history, self.config.rolling_window_days)

            # 3. Compute S_t
            s_t = _compute_cusum(
                ic_t=ic_t,
                s_prev=s_prev,
                mu_ic=mu_ic,
                k=self.config.k,
            )

            # 4. L1 check
            l1_triggered = s_t >= self.config.h

            # 5. Count consecutive alerts for L2
            consecutive = self._count_recent_alerts(
                factor_name,
                window_days=self.config.l2_window_days,
                before_date=metric_date,
            )
            if l1_triggered:
                consecutive += 1
                l1_triggers.add(factor_name)

            # 6. Persist record
            self._persist_record(
                metric_date=metric_date,
                factor_name=factor_name,
                daily_ic=ic_t,
                rolling_ic_mean=mu_ic,
                cusum_value=s_t,
                cusum_alert=l1_triggered,
                consecutive_alerts=consecutive,
            )
            records_written += 1

            # 7. L2 check (after persist so trigger_dates includes today)
            if l1_triggered:
                trigger_dates = self._get_trigger_dates(
                    factor_name,
                    window_days=self.config.l2_window_days,
                    before_date=metric_date,
                )
                if self._check_l2(trigger_dates):
                    l2_suspended.add(factor_name)
                    self._write_alert(
                        rule_name="cusum_l2",
                        severity="warning",
                        metric_value=float(len(trigger_dates)),
                        notes=(
                            f"factor={factor_name} degraded after "
                            f"{len(trigger_dates)} triggers in "
                            f"{self.config.l2_window_days} days"
                        ),
                    )
                    _send_feishu_notification(
                        f"CUSUM L2: factor {factor_name} degraded "
                        f"(weight freeze), {len(trigger_dates)} triggers "
                        f"in {self.config.l2_window_days}d",
                        severity="warning",
                    )

                # L1 feishu notification
                _send_feishu_notification(
                    f"CUSUM L1: factor {factor_name} cusum={s_t:.4f} >= h={self.config.h}",
                    severity="warning",
                )
                self._write_alert(
                    rule_name="cusum_l1",
                    severity="warning",
                    metric_value=s_t,
                    notes=f"factor={factor_name} cusum={s_t:.4f}",
                )

        # 8. L3 global check
        l3_triggered = self._check_l3(l1_triggers)
        if l3_triggered:
            self._write_alert(
                rule_name="cusum_l3",
                severity="critical",
                metric_value=float(len(l1_triggers)),
                notes=(
                    f"global low-activity mode: {len(l1_triggers)} factors "
                    f"triggered simultaneously on {metric_date.isoformat()}"
                ),
            )
            _send_feishu_notification(
                f"CUSUM L3: global low-activity mode activated — "
                f"{len(l1_triggers)} factors triggered simultaneously",
                severity="critical",
            )

        _logger.info(
            "CUSUM monitor complete: %d records, L1=%d, L2=%d, L3=%s",
            records_written,
            len(l1_triggers),
            len(l2_suspended),
            l3_triggered,
        )

        return {
            "l1_triggers": l1_triggers,
            "l2_suspended": l2_suspended,
            "l3_triggered": l3_triggered,
            "records_written": records_written,
        }

    # -- alert level checks ----------------------------------------------------

    def _check_l1(self, factor_name: str, cusum_value: float) -> bool:
        """Check L1: CUSUM exceeds threshold h.

        Args:
            factor_name: Factor name (for logging).
            cusum_value: Current CUSUM value S_t.

        Returns:
            True if L1 alert is triggered.
        """
        triggered = cusum_value >= self.config.h
        if triggered:
            _logger.info(
                "CUSUM L1 triggered: factor=%s cusum=%.4f h=%.4f",
                factor_name,
                cusum_value,
                self.config.h,
            )
        return triggered

    def _check_l2(self, trigger_dates: list[date]) -> bool:
        """Check L2: factor has >= config.l2_trigger_count triggers within
        config.l2_window_days calendar days.

        Both count AND date-span (max - min) must satisfy the thresholds.

        Args:
            trigger_dates: Dates of recent L1 triggers for a factor, sorted.

        Returns:
            True if L2 degrade should be applied.
        """
        if len(trigger_dates) < self.config.l2_trigger_count:
            return False
        # Check that triggers fall within the window span
        if len(trigger_dates) >= 2:
            span_days = (trigger_dates[-1] - trigger_dates[0]).days
            if span_days >= self.config.l2_window_days:
                return False
        _logger.info(
            "CUSUM L2 triggered: %d triggers in %d-day window (threshold=%d)",
            len(trigger_dates),
            self.config.l2_window_days,
            self.config.l2_trigger_count,
        )
        return True

    def _check_l3(self, triggered_factors: set[str]) -> bool:
        """Check L3: >= config.l3_trigger_count factors triggered simultaneously.

        Args:
            triggered_factors: Set of factor names that triggered L1 today.

        Returns:
            True if global low-activity mode should be activated.
        """
        triggered = len(triggered_factors) >= self.config.l3_trigger_count
        if triggered:
            _logger.warning(
                "CUSUM L3 triggered: %d factors (threshold=%d)",
                len(triggered_factors),
                self.config.l3_trigger_count,
            )
        return triggered

    # -- DB helpers ------------------------------------------------------------

    def _get_previous_cusum(
        self,
        factor_name: str,
        before_date: date,
    ) -> float:
        """Retrieve S_{t-1} from the most recent prior FactorHealthDaily record.

        Args:
            factor_name: Factor name.
            before_date: Upper bound (exclusive) for the query date.

        Returns:
            Previous CUSUM value, or 0.0 if no prior record exists.
        """

        def _query(session: Session) -> float:
            row = (
                session.query(FactorHealthDaily)
                .filter(
                    FactorHealthDaily.factor_name == factor_name,
                    FactorHealthDaily.metric_date < before_date,
                )
                .order_by(FactorHealthDaily.metric_date.desc())
                .first()
            )
            if row is None:
                return 0.0
            return row.cusum_value if row.cusum_value is not None else 0.0

        return _with_session(self._session_factory, _query, rollback_value=0.0)

    def _get_ic_history(
        self,
        factor_name: str,
        before_date: date,
    ) -> list[float]:
        """Retrieve historical daily IC values for rolling mean computation.

        Args:
            factor_name: Factor name.
            before_date: Upper bound (exclusive).

        Returns:
            List of daily_ic values in chronological order.
        """

        def _query(session: Session) -> list[float]:
            rows = (
                session.query(FactorHealthDaily.daily_ic)
                .filter(
                    FactorHealthDaily.factor_name == factor_name,
                    FactorHealthDaily.metric_date < before_date,
                )
                .order_by(FactorHealthDaily.metric_date.asc())
                .all()
            )
            return [r.daily_ic for r in rows if r.daily_ic is not None]

        return _with_session(self._session_factory, _query, rollback_value=[])

    def _get_trigger_dates(
        self,
        factor_name: str,
        window_days: int,
        before_date: date,
    ) -> list[date]:
        """Retrieve dates of recent L1 triggers for a factor.

        Args:
            factor_name: Factor name.
            window_days: Lookback window in days.
            before_date: Reference date. Dates from ``before_date - window_days``
                to ``before_date`` (inclusive) are considered.

        Returns:
            Sorted list of trigger dates.
        """
        cutoff = before_date - timedelta(days=window_days - 1)

        def _query(session: Session) -> list[date]:
            rows = (
                session.query(FactorHealthDaily.metric_date)
                .filter(
                    FactorHealthDaily.factor_name == factor_name,
                    FactorHealthDaily.cusum_alert == True,  # noqa: E712
                    FactorHealthDaily.metric_date >= cutoff,
                    FactorHealthDaily.metric_date <= before_date,
                )
                .order_by(FactorHealthDaily.metric_date.asc())
                .all()
            )
            return [r.metric_date for r in rows]

        return _with_session(self._session_factory, _query, rollback_value=[])

    def _count_recent_alerts(
        self,
        factor_name: str,
        window_days: int,
        before_date: date,
    ) -> int:
        """Count recent alert days for a factor within the lookback window.

        Args:
            factor_name: Factor name.
            window_days: Lookback window.
            before_date: Reference date (exclusive upper bound).

        Returns:
            Number of days with alerts in the window.
        """
        cutoff = before_date - timedelta(days=window_days - 1)

        def _query(session: Session) -> int:
            return (
                session.query(FactorHealthDaily)
                .filter(
                    FactorHealthDaily.factor_name == factor_name,
                    FactorHealthDaily.cusum_alert == True,  # noqa: E712
                    FactorHealthDaily.metric_date >= cutoff,
                    FactorHealthDaily.metric_date < before_date,
                )
                .count()
            )

        return _with_session(self._session_factory, _query, rollback_value=0)

    def _persist_record(
        self,
        metric_date: date,
        factor_name: str,
        daily_ic: float,
        rolling_ic_mean: float,
        cusum_value: float,
        cusum_alert: bool,
        consecutive_alerts: int,
    ) -> None:
        """Persist a FactorHealthDaily record (upsert semantics).

        If a record for the same (metric_date, factor_name) already exists,
        it is updated in place.
        """

        def _write(session: Session) -> None:
            existing = session.get(
                FactorHealthDaily,
                {"metric_date": metric_date, "factor_name": factor_name},
            )
            if existing is not None:
                existing.daily_ic = daily_ic
                existing.rolling_ic_mean_90d = rolling_ic_mean
                existing.cusum_value = cusum_value
                existing.cusum_alert = cusum_alert
                existing.consecutive_alerts = consecutive_alerts
            else:
                session.add(
                    FactorHealthDaily(
                        metric_date=metric_date,
                        factor_name=factor_name,
                        daily_ic=daily_ic,
                        rolling_ic_mean_90d=rolling_ic_mean,
                        cusum_value=cusum_value,
                        cusum_alert=cusum_alert,
                        consecutive_alerts=consecutive_alerts,
                    )
                )

        _with_session(self._session_factory, _write, suppress_exception=False)

    def _write_alert(
        self,
        rule_name: str,
        severity: str,
        metric_value: float | None = None,
        notes: str | None = None,
    ) -> None:
        """Write an alert row to the alerts table."""

        def _write(session: Session) -> None:
            from datetime import UTC, datetime

            session.add(
                Alert(
                    triggered_at=datetime.now(UTC).isoformat(),
                    severity=severity,
                    rule_name=rule_name,
                    metric_value=metric_value,
                    notes=notes,
                )
            )

        _with_session(self._session_factory, _write, suppress_exception=False)


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------


def _with_session(
    session_factory: _SessionFactory,
    fn: Callable[[Session], Any],
    *,
    rollback_value: Any = None,
    suppress_exception: bool = True,
) -> Any:
    """Execute *fn* inside a new session, commit, and close.

    Args:
        session_factory: Callable that returns a new SQLAlchemy Session.
        fn: Function to execute with the session.
        rollback_value: Value returned on failure when *suppress_exception* is True.
        suppress_exception: If True (default), exceptions are logged and
            ``rollback_value`` is returned.  If False, the exception is logged,
            the session is rolled back, and the exception is re-raised so the
            caller can handle it (intended for write paths where silent
            swallowing would hide data-loss bugs).
    """
    session = session_factory()
    try:
        result = fn(session)
        session.commit()
        return result
    except Exception:
        _logger.exception("CUSUM session operation failed")
        session.rollback()
        if suppress_exception:
            return rollback_value
        raise
    finally:
        session.close()
