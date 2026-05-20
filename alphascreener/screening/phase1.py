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

from alphascreener.screening.threshold import DEFAULT_THRESHOLDS

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
