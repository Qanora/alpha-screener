"""Phase 2 weighted scoring + GICS industry dedup (Issue #95).

Reference: PRD 3.2.3 / 3.3.

Pipeline:
  1. Breakout_Score = Σ(w_i × z_capped_i) weighted composite
  2. Sort descending → Top 30
  3. GICS Sector ≤ 3 / Industry ≤ 2 dedup
  4. Fallback to original scoring if < 20
  5. Output Top 20 for LLM fine screening
"""

from __future__ import annotations

import polars as pl

from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ---------------------------------------------------------------------------
# MVP factor weights from PRD 3.1.1 (normalised to sum to ~1.0)
# ---------------------------------------------------------------------------

MVP_WEIGHTS: dict[str, float] = {
    "MOM_5D": 0.14,
    "PTH": 0.12,
    "MOM_SLOPE": 0.10,
    "BB_SQUEEZE": 0.13,
    "ATR_RATIO": 0.09,
    "MFI_14": 0.095,
    "CMF_21": 0.085,
    "VOL_ANOMALY": 0.045,
    "RSI_OVERSOLD": 0.045,
    "MACD_CROSS": 0.035,
    "GOLDEN_CROSS": 0.035,
    "INSIDER_BUY": 0.045,
    "REV_ACCEL": 0.035,
}

# ---------------------------------------------------------------------------
# Factor signal direction metadata (Issue #191)
# ---------------------------------------------------------------------------
# +1: higher raw factor value -> stronger alpha signal (positive contribution)
# -1: higher raw factor value -> weaker alpha signal (invert in composite)
#
# Directions are aligned with Phase 1 filter constraints:
#   - ATR_RATIO < threshold: low vol contraction is bullish -> invert (-1)
#   - All momentum/money-flow/technical factors: higher is better (+1)

_SIGNAL_DIRECTION: dict[str, int] = {
    "MOM_5D": 1,
    "PTH": 1,
    "MOM_SLOPE": 1,
    "BB_SQUEEZE": 1,
    "ATR_RATIO": -1,  # Phase 1: ATR_RATIO < 0.8 -> low is good
    "MFI_14": 1,
    "CMF_21": 1,
    "VOL_ANOMALY": 1,
    "RSI_OVERSOLD": 1,
    "REV_ACCEL": 1,
}

# ---------------------------------------------------------------------------
# Factor classification for scoring
# ---------------------------------------------------------------------------

# Continuous factors with z_capped columns (from normalize_factors in engine.py)
_Z_CAPPED_FACTORS: tuple[str, ...] = (
    "MOM_5D",
    "PTH",
    "MOM_SLOPE",
    "BB_SQUEEZE",
    "ATR_RATIO",
    "MFI_14",
    "CMF_21",
    "VOL_ANOMALY",
    "RSI_OVERSOLD",
    "REV_ACCEL",
)

# Binary factors that contribute full weight when flag == 1
# PEAD_FLAG weight = 0 in coarse screening (PRD: 0% 粗筛)
_BINARY_FACTORS: tuple[str, ...] = ("MACD_CROSS", "GOLDEN_CROSS", "INSIDER_BUY")

# ---------------------------------------------------------------------------
# Default caps
# ---------------------------------------------------------------------------

DEFAULT_TOP_N: int = 30
DEFAULT_FINAL_N: int = 20
DEFAULT_SECTOR_CAP: int = 3
DEFAULT_INDUSTRY_CAP: int = 2


# ---------------------------------------------------------------------------
# Weighted breakout score
# ---------------------------------------------------------------------------


def compute_breakout_score(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Breakout_Score = Σ(w_i × z_capped_i) for each ticker.

    Continuous factors contribute w_i × z_capped_i (clipped z-score).
    Binary factors contribute full weight when flag == 1, otherwise 0.
    PEAD_FLAG weight is 0 in coarse screening per PRD 3.1.1.

    Args:
        df: DataFrame with factor columns and ``z_capped_{factor}`` columns
            (output of :func:`alphascreener.factors.engine.normalize_factors`).

    Returns:
        DataFrame with ``breakout_score`` (f64) column appended.
    """
    if df.height == 0:
        return df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

    score = pl.lit(0.0, dtype=pl.Float64)

    for fname in _Z_CAPPED_FACTORS:
        raw_col = fname
        if raw_col in df.columns:
            w = MVP_WEIGHTS[fname]
            direction = _SIGNAL_DIRECTION.get(fname, 1)
            # Cross-sectional z-score normalization (mean 0, std 1)
            # Clipped to [-3, +3] for outlier robustness
            mu = pl.col(raw_col).mean()
            sigma = pl.col(raw_col).std(ddof=1)
            z_score = pl.when(sigma > 1e-12).then((pl.col(raw_col) - mu) / sigma).otherwise(0.0)
            z_score = pl.when(pl.col(raw_col).is_null()).then(0.0).otherwise(z_score)
            z_capped = z_score.clip(-3.0, 3.0)
            score = score + z_capped * w * direction

    for fname in _BINARY_FACTORS:
        if fname in df.columns:
            w = MVP_WEIGHTS[fname]
            score = score + pl.when(pl.col(fname) == 1).then(w).otherwise(0.0)

    return df.with_columns(score.alias("breakout_score"))


# ---------------------------------------------------------------------------
# Industry dedup
# ---------------------------------------------------------------------------


def apply_industry_dedup(
    df: pl.DataFrame,
    *,
    sector_cap: int = DEFAULT_SECTOR_CAP,
    industry_cap: int = DEFAULT_INDUSTRY_CAP,
    sector_count: dict[str, int] | None = None,
    industry_count: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Apply GICS sector/industry dedup to a sorted candidate list.

    Greedy algorithm: walks candidates from top score downward, selecting
    tickers while enforcing per-sector and per-industry caps.  Tickers with
    null sector/industry are always selected (no cap applied).

    The input DataFrame is assumed already sorted by breakout score descending.

    Args:
        df: DataFrame sorted by score, with columns ``ticker``, ``sector``
            (optional), ``industry`` (optional).
        sector_cap: Max tickers per GICS Sector.
        industry_cap: Max tickers per GICS Industry.
        sector_count: Optional pre-populated sector counts (for incremental
            dedup). Mutated in place.
        industry_count: Optional pre-populated industry counts (for
            incremental dedup). Mutated in place.

    Returns:
        Dedup-selected rows, preserving relative sort order.
    """
    if df.height == 0:
        return df

    rows = df.to_dicts()
    selected: list[dict] = []
    if sector_count is None:
        sector_count = {}
    if industry_count is None:
        industry_count = {}

    for row in rows:
        sector = row.get("sector")
        industry = row.get("industry")

        sector_key = str(sector) if sector is not None else None
        industry_key = str(industry) if industry is not None else None

        # Enforce caps
        if sector_key is not None and sector_count.get(sector_key, 0) >= sector_cap:
            continue
        if industry_key is not None and industry_count.get(industry_key, 0) >= industry_cap:
            continue

        selected.append(row)
        if sector_key is not None:
            sector_count[sector_key] = sector_count.get(sector_key, 0) + 1
        if industry_key is not None:
            industry_count[industry_key] = industry_count.get(industry_key, 0) + 1

    if not selected:
        return df.clear()

    return pl.DataFrame(selected, schema=df.schema)


# ---------------------------------------------------------------------------
# Full Phase 2 pipeline
# ---------------------------------------------------------------------------


def phase2_pipeline(
    df: pl.DataFrame,
    *,
    n_top: int = DEFAULT_TOP_N,
    n_final: int = DEFAULT_FINAL_N,
    sector_cap: int = DEFAULT_SECTOR_CAP,
    industry_cap: int = DEFAULT_INDUSTRY_CAP,
) -> pl.DataFrame:
    """Run the full Phase 2 pipeline: weighted scoring + industry dedup.

    Pipeline steps:
      1. Compute ``breakout_score`` for all tickers.
      2. Sort by ``breakout_score`` descending → take Top *n_top*.
      3. Apply GICS sector/industry caps (greedy dedup) to the Top *n_top*.
      4. If dedup result < *n_final*, fill from remaining candidates (by
         original breakout score, also deduped) until *n_final* is reached
         or the pool is exhausted.
      5. Return at most *n_final* tickers, sorted by ``breakout_score``
         descending.

    Args:
        df: Factor DataFrame from Phase 1 (must include ``pass_phase1``
            column to filter).  Must have ``z_capped_{factor}`` columns
            from ``normalize_factors``.  Optionally has ``sector`` and
            ``industry`` columns (joined from ``universe_meta.parquet``).
        n_top: Number of top-scoring candidates to consider before dedup.
        n_final: Target output count.
        sector_cap: Max tickers per GICS Sector.
        industry_cap: Max tickers per GICS Industry.

    Returns:
        DataFrame of at most *n_final* tickers with breakout_score, sector,
        industry, and all input columns.
    """
    if df.height == 0:
        return compute_breakout_score(df)

    # 1. Compute breakout score
    df = compute_breakout_score(df)

    # 2. Sort by breakout_score descending
    df = df.sort("breakout_score", descending=True)

    # 3. Top N candidates
    cand = df.head(n_top)

    # 4. Apply industry dedup to top N
    deduped = apply_industry_dedup(cand, sector_cap=sector_cap, industry_cap=industry_cap)

    # 5. Fallback: if dedup result < n_final, fill from remaining pool
    if deduped.height < n_final:
        # Build sector/industry counts from already-selected tickers so
        # incremental backfill does not break the global caps.
        sc: dict[str, int] = {}
        ic: dict[str, int] = {}
        for row in deduped.to_dicts():
            s = row.get("sector")
            if s is not None:
                sk = str(s)
                sc[sk] = sc.get(sk, 0) + 1
            ind = row.get("industry")
            if ind is not None:
                ik = str(ind)
                ic[ik] = ic.get(ik, 0) + 1

        shortage = n_final - deduped.height
        existing_tickers = set(deduped["ticker"].to_list())

        # Remaining candidates: top N dedup-skipped + beyond top N
        remainder = df.filter(~pl.col("ticker").is_in(existing_tickers))

        if remainder.height > 0:
            fill = apply_industry_dedup(
                remainder,
                sector_cap=sector_cap,
                industry_cap=industry_cap,
                sector_count=sc,
                industry_count=ic,
            ).head(shortage)

            if fill.height > 0:
                deduped = pl.concat([deduped, fill], how="diagonal_relaxed")

        # If still short (e.g. caps too restrictive), relax caps and fill
        if deduped.height < n_final:
            shortage2 = n_final - deduped.height
            existing_tickers = set(deduped["ticker"].to_list())
            remainder2 = df.filter(~pl.col("ticker").is_in(existing_tickers))

            if remainder2.height > 0:
                # Fill without caps (raw score order)
                fill2 = remainder2.head(shortage2)
                deduped = pl.concat([deduped, fill2], how="diagonal_relaxed")

    _logger.info(
        "Phase 2: scored %d tickers → top %d → dedup %d → output %d "
        "(sector_cap=%d, industry_cap=%d)",
        df.height,
        min(df.height, n_top),
        min(deduped.height, n_final) if deduped.height >= n_final else deduped.height,
        min(deduped.height, n_final),
        sector_cap,
        industry_cap,
    )

    return deduped.sort("breakout_score", descending=True).head(n_final)
