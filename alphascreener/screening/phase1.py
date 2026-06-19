"""Phase 1 hard filtering rules (Issue #94).

Must-satisfy conditions (all 4 must pass):
  1. MOM_5D > threshold[MOM_5D]      (default > 0)
  2. VOL_ANOMALY = 1 OR MFI_14 > threshold[MFI_14]   (default MFI > 40)
  3. ATR_RATIO < threshold[ATR_RATIO]   (default < 0.8)
  4. RSI_14 in [threshold[RSI_LOW], threshold[RSI_HIGH]]   (default [25, 75])

Optional bonus conditions (5, count only):
  - BB_SQUEEZE = 1
  - PTH > 0.90
  - CMF_21 > 0
  - PEAD_FLAG = 1
  - INSIDER_BUY = 1

Reference: PRD 3.2.1.
"""

from __future__ import annotations

import polars as pl

from alphascreener.logging import get_logger
from alphascreener.screening.threshold import DEFAULT_THRESHOLDS

_logger = get_logger("screening")

# ---------------------------------------------------------------------------
# Required factor columns (must be present in input DataFrame)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "ticker",
        "MOM_5D",
        "VOL_ANOMALY",
        "MFI_14",
        "ATR_RATIO",
        "RSI_14",
        "BB_SQUEEZE",
        "PTH",
        "CMF_21",
        "PEAD_FLAG",
        "INSIDER_BUY",
    }
)

# ---------------------------------------------------------------------------
# Individual must-satisfy condition checks
# ---------------------------------------------------------------------------


def _check_mom_5d(df: pl.DataFrame, threshold: float = 0.0) -> pl.DataFrame:
    """Add ``_mom_ok`` column: True if MOM_5D > threshold."""
    return df.with_columns((pl.col("MOM_5D") > threshold).alias("_mom_ok"))


def _check_vol_mfi(df: pl.DataFrame, mfi_threshold: float = 40.0) -> pl.DataFrame:
    """Add ``_volmfi_ok`` column: True if VOL_ANOMALY == 1 OR MFI_14 > mfi_threshold."""
    return df.with_columns(
        ((pl.col("VOL_ANOMALY") == 1) | (pl.col("MFI_14") > mfi_threshold)).alias("_volmfi_ok")
    )


def _check_atr_ratio(df: pl.DataFrame, threshold: float = 0.8) -> pl.DataFrame:
    """Add ``_atr_ok`` column: True if ATR_RATIO < threshold."""
    return df.with_columns((pl.col("ATR_RATIO") < threshold).alias("_atr_ok"))


def _check_rsi_range(df: pl.DataFrame, low: float = 25.0, high: float = 75.0) -> pl.DataFrame:
    """Add ``_rsi_ok`` column: True if RSI_14 in [low, high] (inclusive)."""
    col = pl.col("RSI_14")
    return df.with_columns(((col >= low) & (col <= high)).alias("_rsi_ok"))


# ---------------------------------------------------------------------------
# Optional bonus counting
# ---------------------------------------------------------------------------

_BONUS_PTH_THRESHOLD: float = 0.90


def _count_bonuses(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``bonus_count`` column: sum of 5 optional bonus flags."""
    bonus = pl.lit(0, dtype=pl.Int32)
    bonus = bonus + pl.when(pl.col("BB_SQUEEZE") == 1).then(1).otherwise(0)
    bonus = bonus + pl.when(pl.col("PTH") > _BONUS_PTH_THRESHOLD).then(1).otherwise(0)
    bonus = bonus + pl.when(pl.col("CMF_21") > 0.0).then(1).otherwise(0)
    bonus = bonus + pl.when(pl.col("PEAD_FLAG") == 1).then(1).otherwise(0)
    bonus = bonus + pl.when(pl.col("INSIDER_BUY") == 1).then(1).otherwise(0)
    return df.with_columns(bonus.alias("bonus_count"))


# ---------------------------------------------------------------------------
# Combined hard filter
# ---------------------------------------------------------------------------


def _validate_columns(df: pl.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Phase 1 hard filter requires columns: {sorted(missing)}")


def _resolve_thresholds(thresholds: dict | None) -> dict:
    """Merge user-supplied thresholds with defaults.

    Acceptable keys: MOM_5D, MFI_14, ATR_RATIO, RSI_LOW, RSI_HIGH.
    """
    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for k in ("MOM_5D", "MFI_14", "ATR_RATIO", "RSI_LOW", "RSI_HIGH"):
            if k in thresholds:
                merged[k] = float(thresholds[k])
    return merged


def hard_filter(df: pl.DataFrame, *, thresholds: dict | None = None) -> pl.DataFrame:
    """Apply Phase 1 hard filtering to a factor DataFrame.

    Args:
        df: Factor DataFrame with columns: ticker, MOM_5D, VOL_ANOMALY,
            MFI_14, ATR_RATIO, RSI_14, BB_SQUEEZE, PTH, CMF_21,
            PEAD_FLAG, INSIDER_BUY.
        thresholds: Optional dict overriding must-satisfy condition thresholds.
            Keys: MOM_5D, MFI_14, ATR_RATIO, RSI_LOW, RSI_HIGH.

    Returns:
        DataFrame with all input columns plus ``pass_phase1`` (bool) and
        ``bonus_count`` (i32) columns appended.
    """
    if df.height == 0:
        return df.with_columns(
            pl.lit(True, dtype=pl.Boolean).alias("pass_phase1"),
            pl.lit(0, dtype=pl.Int32).alias("bonus_count"),
        )

    _validate_columns(df)
    th = _resolve_thresholds(thresholds)

    result = (
        df.pipe(_check_mom_5d, th["MOM_5D"])
        .pipe(_check_vol_mfi, th["MFI_14"])
        .pipe(_check_atr_ratio, th["ATR_RATIO"])
        .pipe(_check_rsi_range, th["RSI_LOW"], th["RSI_HIGH"])
        .pipe(_count_bonuses)
    )

    pass_all = pl.col("_mom_ok") & pl.col("_volmfi_ok") & pl.col("_atr_ok") & pl.col("_rsi_ok")

    result = result.with_columns(pass_all.alias("pass_phase1"))
    # Drop internal columns
    result = result.drop(["_mom_ok", "_volmfi_ok", "_atr_ok", "_rsi_ok"])
    return result


# ---------------------------------------------------------------------------
# Filter rate computation
# ---------------------------------------------------------------------------


def compute_filter_rate(df: pl.DataFrame, *, thresholds: dict | None = None) -> float:
    """Compute Phase 1 filter rate = 1 - (pass_count / total_count).

    Args:
        df: Factor DataFrame (see :func:`hard_filter` for schema).
        thresholds: Optional threshold overrides.

    Returns:
        Filter rate as a fraction in [0, 1]. Returns 0.0 if df is empty.
    """
    if df.height == 0:
        return 0.0

    result = hard_filter(df, thresholds=thresholds)
    passed = result["pass_phase1"].sum()
    return 1.0 - (passed / df.height)


# ---------------------------------------------------------------------------
# Auto-relaxation: hard_filter with fallback for over-strict filtering (Issue #219)
# ---------------------------------------------------------------------------

# Trigger relaxation when pass_rate falls below this fraction.
_RELAX_PASS_RATE_THRESHOLD: float = 0.05
# Also trigger when fewer than this many tickers pass (for small/medium universes).
_RELAX_MIN_PASSERS: int = 5
# Maximum number of consecutive relaxation steps before giving up.
_RELAX_MAX_STEPS: int = 3
# Maximum cumulative threshold relaxation as a fraction of reference magnitude.
# Matches ``threshold.MAX_RELAXATION_PCT``.  Enforced across all relaxation steps.
_RELAX_MAX_CUMULATIVE: float = 0.30

# Per-threshold relaxation delta magnitudes (one-step).  These match the
# 10% step defined in DynamicThreshold._step_delta for the "over_tight" band.
# Sign convention: positive = widen (let more tickers pass).
_RELAX_DELTA: dict[str, float] = {
    "MOM_5D": -0.005,  # lower MOM_5D threshold (MOM_5D > X → relax by -Δ)
    "MFI_14": -4.0,  # lower MFI_14 threshold
    "ATR_RATIO": +0.08,  # raise ATR_RATIO threshold (ATR_RATIO < X → relax by +Δ)
    "RSI_LOW": -2.5,  # lower RSI low bound
    "RSI_HIGH": +7.5,  # raise RSI high bound
}

# Reference magnitudes for each threshold key (used for cumulative cap).
# These match DynamicThreshold._reference_magnitude values.
_RELAX_REFERENCE_MAGNITUDE: dict[str, float] = {
    "MOM_5D": 0.05,
    "MFI_14": 40.0,
    "ATR_RATIO": 0.80,
    "RSI_LOW": 25.0,
    "RSI_HIGH": 75.0,
}


def _build_relaxed_thresholds(
    thresholds: dict,
    *,
    step: int = 1,
    original: dict | None = None,
) -> dict:
    """Return a copy of *thresholds* with relaxation applied.

    When *original* is provided, the cumulative relaxation from *original* to
    the new thresholds is capped at ``_RELAX_MAX_CUMULATIVE`` of the reference
    magnitude per key to prevent unbounded threshold drift.

    Args:
        thresholds: Current threshold dict (keys: MOM_5D, MFI_14, ATR_RATIO,
            RSI_LOW, RSI_HIGH).
        step: Number of relaxation steps to apply (default 1).
        original: Original (pre-relaxation) thresholds dict for cap enforcement.
            When None, no cumulative cap is enforced.

    Returns:
        Relaxed thresholds dict.
    """
    relaxed = dict(thresholds)

    for key, delta in _RELAX_DELTA.items():
        # Apply step-worth of deltas
        new_val = thresholds[key] + delta * step

        # Enforce cumulative relaxation cap relative to original defaults
        if original is not None and key in _RELAX_REFERENCE_MAGNITUDE:
            ref = _RELAX_REFERENCE_MAGNITUDE[key]
            max_abs_delta = ref * _RELAX_MAX_CUMULATIVE
            # Determine widen direction sign
            if key in ("MOM_5D", "MFI_14", "RSI_LOW"):
                # These use > comparison → widen = lower threshold
                cap_val = original[key] - max_abs_delta
                new_val = max(cap_val, new_val)
            elif key in ("ATR_RATIO",):
                # Uses < comparison → widen = raise threshold
                cap_val = original[key] + max_abs_delta
                new_val = min(cap_val, new_val)
            elif key == "RSI_HIGH":
                # RSI upper bound: widen = raise
                cap_val = original[key] + max_abs_delta
                new_val = min(cap_val, new_val)

        # Defensive absolute bounds (hard floor/ceiling)
        if key == "RSI_LOW":
            new_val = max(0.0, new_val)
        elif key == "RSI_HIGH":
            new_val = min(100.0, new_val)
        elif key == "MFI_14":
            new_val = max(10.0, new_val)
        elif key == "ATR_RATIO":
            new_val = min(1.5, new_val)

        relaxed[key] = new_val

    return relaxed


def hard_filter_with_fallback(
    df: pl.DataFrame,
    *,
    thresholds: dict | None = None,
) -> tuple[pl.DataFrame, bool]:
    """Apply Phase 1 hard filtering with iterative threshold relaxation.

    When the pass rate falls below ``_RELAX_PASS_RATE_THRESHOLD`` (5 %) or
    fewer than ``_RELAX_MIN_PASSERS`` tickers pass, thresholds are
    progressively relaxed up to ``_RELAX_MAX_STEPS`` times (cumulative cap
    ``_RELAX_MAX_CUMULATIVE``) to prevent the pipeline from stalling on a
    single ticker.

    Args:
        df: Factor DataFrame (see :func:`hard_filter` for schema).
        thresholds: Optional base threshold overrides.

    Returns:
        ``(dataframe, was_relaxed)`` tuple.  The dataframe has
        ``pass_phase1`` and ``bonus_count`` columns appended (same shape as
        :func:`hard_filter`).  ``was_relaxed`` is True when relaxation was
        applied.
    """
    th = _resolve_thresholds(thresholds)
    original_th = dict(th)  # snapshot for cumulative cap

    result = hard_filter(df, thresholds=th)
    n_pass = result["pass_phase1"].sum()
    n_total = result.height

    if n_total == 0:
        return result, False

    pass_rate = n_pass / n_total
    should_relax = pass_rate < _RELAX_PASS_RATE_THRESHOLD or (
        n_total >= _RELAX_MIN_PASSERS and n_pass < _RELAX_MIN_PASSERS
    )

    if not should_relax:
        return result, False

    _logger.warning(
        "Phase 1 pass_rate too low (%.1f%%, %d/%d). "
        "Starting iterative threshold relaxation to prevent pipeline stall.",
        pass_rate * 100,
        n_pass,
        n_total,
    )

    # Iterative relaxation: try up to _RELAX_MAX_STEPS
    n_pass_relaxed = n_pass  # track best result across steps
    for step in range(1, _RELAX_MAX_STEPS + 1):
        relaxed_th = _build_relaxed_thresholds(original_th, step=step, original=original_th)

        _logger.info(
            "Phase 1 relaxation step %d/%d: MOM_5D > %.4f, MFI_14 > %.1f, "
            "ATR_RATIO < %.2f, RSI in [%.1f, %.1f]",
            step,
            _RELAX_MAX_STEPS,
            relaxed_th["MOM_5D"],
            relaxed_th["MFI_14"],
            relaxed_th["ATR_RATIO"],
            relaxed_th["RSI_LOW"],
            relaxed_th["RSI_HIGH"],
        )

        step_result = hard_filter(df, thresholds=relaxed_th)
        step_n_pass = step_result["pass_phase1"].sum()
        step_pass_rate = step_n_pass / n_total

        _logger.info(
            "Phase 1 relaxation step %d result: %d/%d tickers passed (%.1f%%)",
            step,
            step_n_pass,
            n_total,
            step_pass_rate * 100,
        )

        # Keep the best result
        if step_n_pass > n_pass_relaxed:
            n_pass_relaxed = step_n_pass
            result = step_result

        # Stop if enough tickers pass
        if step_pass_rate >= _RELAX_PASS_RATE_THRESHOLD and step_n_pass >= _RELAX_MIN_PASSERS:
            _logger.info(
                "Phase 1 relaxation succeeded at step %d: %d/%d tickers passed (%.1f%%)",
                step,
                step_n_pass,
                n_total,
                step_pass_rate * 100,
            )
            break
    else:
        # All steps exhausted — log the best result we got
        _logger.warning(
            "Phase 1 iterative relaxation exhausted (%d steps): "
            "best result %d/%d tickers passed (%.1f%%). "
            "Cumulative cap reached or data quality too poor.",
            _RELAX_MAX_STEPS,
            n_pass_relaxed,
            n_total,
            n_pass_relaxed / n_total * 100,
        )

    return result, True
