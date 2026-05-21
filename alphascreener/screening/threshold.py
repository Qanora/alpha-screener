"""Dynamic threshold adjustment for Phase 1 hard filtering (Issue #94).

Auto-adjusts must-satisfy condition thresholds based on daily filter rate.

Rules (PRD 3.2.2):
  - Filter rate 80-92%: normal, no adjustment
  - Filter rate 92-95%: tight, alert (no auto adjustment)
  - Filter rate 95-98%: over-tight, auto-widen conditions
  - Filter rate > 98%: extreme, full-widen 10% + regime diagnosis
  - Filter rate < 70%: over-loose, auto-tighten conditions

Widening direction (PRD 3.2.2 table):
  - MOM_5D > X  →  lower X (-Δ)
  - ATR_RATIO < X  →  raise X (+Δ)
  - RSI in [a,b]  →  [a-Δ, b+Δ]
  - MFI_14 > X  →  lower X (-Δ)

Tightening reverses these directions.

Constraints:
  - Single-step magnitude: ±10% of base threshold
  - Cooldown: >= 3 trading days between adjustments
  - Cumulative relaxation cap: <= 30% of base threshold
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Default thresholds (PRD 3.2.1)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "MOM_5D": 0.0,
    "MFI_14": 40.0,
    "ATR_RATIO": 0.80,
    "RSI_LOW": 25.0,
    "RSI_HIGH": 75.0,
}

# ---------------------------------------------------------------------------
# Adjustable parameter keys (subset of DEFAULT_THRESHOLDS that can change)
# ---------------------------------------------------------------------------

_ADJUSTABLE_KEYS: frozenset[str] = frozenset(DEFAULT_THRESHOLDS.keys())

# ---------------------------------------------------------------------------
# Reference magnitudes for step-size calculation
# When the base threshold is 0 or very small, use a domain-appropriate floor
# so that the 10% step is meaningful.
# ---------------------------------------------------------------------------

_REFERENCE_MAGNITUDE: dict[str, float] = {
    "MOM_5D": 0.05,  # 5% return reference => step = 0.005 (0.5pp)
    "MFI_14": 40.0,  # step = 4.0
    "ATR_RATIO": 0.80,  # step = 0.08
    "RSI_LOW": 25.0,  # step = 2.5
    "RSI_HIGH": 75.0,  # step = 7.5
}

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

STEP_PCT: float = 0.10  # 10% per adjustment
COOLDOWN_DAYS: int = 3  # minimum trading days between adjustments
MAX_RELAXATION_PCT: float = 0.30  # 30% cumulative relaxation cap

# Filter rate bands (exclusive upper bound, inclusive lower bound)
FILTER_RATE_BANDS: dict[str, tuple[float, float]] = {
    "normal": (0.80, 0.92),
    "tight": (0.92, 0.95),
    "over_tight": (0.95, 0.98),
    "extreme": (0.98, 1.0),
    "over_loose": (0.0, 0.70),
}

# ---------------------------------------------------------------------------
# Adjustment direction per threshold key
# '>' means: X > threshold, so widen = -Δ, tighten = +Δ
# '<' means: X < threshold, so widen = +Δ, tighten = -Δ
# ---------------------------------------------------------------------------

_COMPARISON_DIRECTION: dict[str, str] = {
    "MOM_5D": ">",
    "MFI_14": ">",
    "ATR_RATIO": "<",
}


def _business_days_between(start: date, end: date) -> int:
    """Count weekdays between *start* (exclusive) and *end* (inclusive)."""
    days = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


class DynamicThreshold:
    """Manages dynamic threshold adjustments for Phase 1 hard filtering.

    Tracks current thresholds, adjustment history, cooldown state, and
    cumulative relaxation.  Call :meth:`adjust` daily with the observed
    filter rate to get updated thresholds.

    Usage::

        dt = DynamicThreshold()
        dt.adjust(0.96)  # filter rate too high, auto-widen
        custom_th = dt.thresholds
        result = hard_filter(df, thresholds=custom_th)
    """

    def __init__(self, *, cooldown_days: int = COOLDOWN_DAYS) -> None:
        self._cooldown_days = cooldown_days
        self._thresholds: dict[str, float] = copy.deepcopy(DEFAULT_THRESHOLDS)
        self._last_adjustment_date: date | None = None
        self._history: list[dict[str, Any]] = []
        # Track cumulative delta per key (absolute, not signed)
        # for relaxation cap enforcement
        self._cumulative_relaxation: dict[str, float] = {k: 0.0 for k in _ADJUSTABLE_KEYS}

    # -- public properties --------------------------------------------------------

    @property
    def thresholds(self) -> dict[str, float]:
        """Current thresholds dict (read-only view)."""
        return dict(self._thresholds)

    @property
    def history(self) -> list[dict[str, Any]]:
        """List of adjustment records, most recent first."""
        return list(self._history)

    # -- status classification ----------------------------------------------------

    def _classify(self, filter_rate: float) -> str:
        """Classify filter rate into a band label."""
        low_n, hi_n = FILTER_RATE_BANDS["normal"]
        if low_n <= filter_rate < hi_n:
            return "normal"
        low_t, hi_t = FILTER_RATE_BANDS["tight"]
        if low_t <= filter_rate < hi_t:
            return "tight"
        low_ot, hi_ot = FILTER_RATE_BANDS["over_tight"]
        if low_ot <= filter_rate <= hi_ot:
            return "over_tight"
        # extreme: >98% (strictly greater than over_tight upper bound)
        if filter_rate > hi_ot:
            return "extreme"
        _low_ol, hi_ol = FILTER_RATE_BANDS["over_loose"]
        if filter_rate < hi_ol:
            return "over_loose"
        return "normal"

    def _should_adjust(self, band: str) -> bool:
        """Return True if the given band triggers an adjustment."""
        return band in ("over_tight", "extreme", "over_loose")

    # -- step calculation ---------------------------------------------------------

    def _step_delta(self, key: str, direction: int, *, force_full: bool = False) -> float:
        """Compute the signed delta for a single adjustment step.

        Args:
            key: Threshold key (MOM_5D, MFI_14, ATR_RATIO, RSI_LOW, RSI_HIGH).
            direction: +1 for widen, -1 for tighten.
            force_full: If True (extreme band), force a full 10% step regardless
                of cumulative cap (cap still applies to total but one step won't
                be zero).

        Returns:
            Signed delta value.
        """
        base = _REFERENCE_MAGNITUDE[key]
        raw_step = STEP_PCT * base

        # Apply direction
        if key in _COMPARISON_DIRECTION:
            comp = _COMPARISON_DIRECTION[key]
            if comp == ">":
                # widen → lower threshold (-Δ), tighten → raise (+Δ)
                signed_step = -raw_step if direction > 0 else +raw_step
            else:  # '<'
                # widen → raise threshold (+Δ), tighten → lower (-Δ)
                signed_step = +raw_step if direction > 0 else -raw_step
        elif key == "RSI_LOW":
            # widen → lower bound (-Δ), tighten → raise bound (+Δ)
            signed_step = -raw_step if direction > 0 else +raw_step
        elif key == "RSI_HIGH":
            # widen → raise bound (+Δ), tighten → lower bound (-Δ)
            signed_step = +raw_step if direction > 0 else -raw_step
        else:
            signed_step = raw_step * direction

        # Cap cumulative relaxation: total widening (positive direction)
        # cannot exceed MAX_RELAXATION_PCT of base magnitude
        if direction > 0:
            ref_mag = _REFERENCE_MAGNITUDE[key]
            max_delta = MAX_RELAXATION_PCT * ref_mag
            current_cumulative = self._cumulative_relaxation.get(key, 0.0)
            allowed = max_delta - current_cumulative
            if allowed <= 0:
                return 0.0
            abs_step = abs(signed_step)
            if abs_step > allowed:
                signed_step = allowed if signed_step > 0 else -allowed

        return signed_step

    # -- adjustment ---------------------------------------------------------------

    def _apply_adjustment(self, band: str) -> dict[str, Any]:
        """Apply threshold adjustments for the given filter band.

        Returns an adjustment record dict.
        """
        direction = 1 if band in ("over_tight", "extreme") else -1
        force_full = band == "extreme"

        record: dict[str, Any] = {
            "band": band,
            "direction": "widen" if direction > 0 else "tighten",
            "changes": {},
            "thresholds_before": dict(self._thresholds),
        }

        for key in _ADJUSTABLE_KEYS:
            delta = self._step_delta(key, direction, force_full=force_full)
            if delta == 0.0:
                continue
            old_val = self._thresholds[key]
            new_val = old_val + delta

            # Special guard: RSI_LOW cannot go below 0, RSI_HIGH cannot exceed 100
            if key == "RSI_LOW":
                new_val = max(0.0, new_val)
            elif key == "RSI_HIGH":
                new_val = min(100.0, new_val)

            applied_delta = new_val - old_val
            if applied_delta != 0.0:
                self._thresholds[key] = new_val
                record["changes"][key] = {"old": old_val, "new": new_val, "delta": applied_delta}

            # Update cumulative relaxation tracker (only for widening)
            if direction > 0 and applied_delta != 0.0:
                prev = self._cumulative_relaxation.get(key, 0.0)
                self._cumulative_relaxation[key] = prev + abs(applied_delta)

        # RSI bounds crossover guard: after tightening, if LOW > HIGH,
        # reset both to the midpoint (50.0) to keep the interval valid.
        if direction < 0 and self._thresholds["RSI_LOW"] > self._thresholds["RSI_HIGH"]:
            mid = 50.0
            self._thresholds["RSI_LOW"] = mid
            self._thresholds["RSI_HIGH"] = mid
            record["changes"]["RSI_LOW"] = {
                "old": record["thresholds_before"]["RSI_LOW"],
                "new": mid,
                "delta": mid - record["thresholds_before"]["RSI_LOW"],
            }
            record["changes"]["RSI_HIGH"] = {
                "old": record["thresholds_before"]["RSI_HIGH"],
                "new": mid,
                "delta": mid - record["thresholds_before"]["RSI_HIGH"],
            }

        record["thresholds_after"] = dict(self._thresholds)
        return record

    def adjust(self, filter_rate: float, *, ref_date: date | None = None) -> str:
        """Evaluate filter rate and adjust thresholds if needed.

        Args:
            filter_rate: Fraction of tickers filtered out, in [0, 1].
            ref_date: Observation date. If None, uses today.  Used for cooldown
                enforcement.

        Returns:
            Status string: one of 'normal', 'tight', 'over_tight', 'extreme',
            'over_loose', suffixed with ': cooldown' when adjustment is skipped,
            or ': adjusted' when thresholds change.
        """
        if ref_date is None:
            ref_date = date.today()

        band = self._classify(filter_rate)

        # Only adjust on actionable bands
        if not self._should_adjust(band):
            return band

        # Cooldown check (trading days, excluding weekends)
        if self._last_adjustment_date is not None:
            days_since = _business_days_between(self._last_adjustment_date, ref_date)
            if days_since < self._cooldown_days:
                return f"{band}: cooldown"

        # Apply adjustment
        record = self._apply_adjustment(band)
        record["filter_rate"] = filter_rate
        record["date"] = ref_date.isoformat()
        self._last_adjustment_date = ref_date
        self._history.append(record)

        return f"{band}: adjusted"

    # -- status / reset -----------------------------------------------------------

    def get_status(self) -> str:
        """Return a human-readable status string summarising current state."""
        th = self._thresholds
        lines = [
            f"MOM_5D > {th['MOM_5D']:.4f}",
            f"MFI_14 > {th['MFI_14']:.2f}",
            f"ATR_RATIO < {th['ATR_RATIO']:.2f}",
            f"RSI_14 ∈ [{th['RSI_LOW']:.1f}, {th['RSI_HIGH']:.1f}]",
        ]
        if self._history:
            last = self._history[-1]
            lines.append(
                f"last_adjustment={last['date']} "
                f"filter_rate={last['filter_rate']:.2%} "
                f"band={last['band']}"
            )
        else:
            lines.append("no adjustments yet")
        return " | ".join(lines)

    def reset(self) -> None:
        """Reset thresholds to defaults and clear history."""
        self._thresholds = copy.deepcopy(DEFAULT_THRESHOLDS)
        self._last_adjustment_date = None
        self._history.clear()
        self._cumulative_relaxation = {k: 0.0 for k in _ADJUSTABLE_KEYS}
