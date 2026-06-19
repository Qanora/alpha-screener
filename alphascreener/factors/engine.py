"""Factor computation engine with chunked streaming (Issue #93).

Processes OHLCV data in configurable chunks (default 4 batches x 500 symbols),
computes 14 factors, applies z-score normalisation, and writes results to the
Hive-partitioned Parquet store via :func:`alphascreener.data.io.write_parquet`.

Reference: PRD 3.1.3 (normalisation) and 3.1.4 (missing-data handling).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import polars as pl

from alphascreener.data.io import write_parquet
from alphascreener.factors.formulas import (
    FACTOR_NAMES,
    compute_all_technical_factors,
    compute_insider_buy,
    compute_pead_flag,
    compute_rev_accel,
)
from alphascreener.logging import get_logger

_logger = get_logger("screening")

# -- constants ---------------------------------------------------------------

# Default chunking: 4 batches x 500 symbols = up to 2000 symbols total
DEFAULT_BATCH_SIZE: int = 500
DEFAULT_N_BATCHES: int = 4

# Normalisation boundaries
Z_SCORE_CAP: float = 3.0  # clip z-scores to [-3, +3]
DISPLAY_SCORE_MIN: float = 0.0
DISPLAY_SCORE_NEUTRAL: float = 50.0
DISPLAY_SCORE_MAX: float = 100.0

# Missing-data threshold
MISSING_FACTOR_RATE_MAX: float = 0.30  # exclude ticker if > 30% factors missing


# -- normalisation -----------------------------------------------------------


def normalize_factors(df: pl.DataFrame) -> pl.DataFrame:
    """Apply cross-sectional z-score normalisation, clipping, and display scoring.

    Step 1: z_i = (f_i - mu) / sigma   (cross-sectional per factor)
    Step 2: z_capped_i = clip(z_i, -3, +3)
    Step 3: score_i = 50 + z_capped_i * (50/3)   mapped to [0, 100]

    Non-technical factors (PEAD_FLAG, INSIDER_BUY, MACD_CROSS, etc.) are binary
    (0/1) indicators and are NOT z-scored; they flow through un-normalised.
    Display scores for binary factors are: 0 -> 0, 1 -> 100.

    The output DataFrame retains all original columns and adds:
      - ``z_{factor}``  for each continuous factor (z-score, null if missing)
      - ``z_capped_{factor}``  (clipped z-score)
      - ``score_{factor}``  (display score [0, 100], 50 if missing)
      - ``final_score``  (sum of z_capped_i for continuous factors)

    Args:
        df: DataFrame with factor columns as produced by :func:`process_chunk`.

    Returns:
        DataFrame with normalisation columns appended.
    """
    # Continuous factors that should be z-scored
    continuous = [
        "MOM_5D",
        "PTH",
        "MOM_SLOPE",
        "BB_SQUEEZE",  # binary in output, continuous in raw form
        "ATR_RATIO",
        "MFI_14",
        "CMF_21",
        "VOL_ANOMALY",  # binary in output, but volume_z used internally
        "RSI_OVERSOLD",
        "REV_ACCEL",
    ]

    # Binary / flag factors (not z-scored)
    binary_factors = [
        "MACD_CROSS",
        "GOLDEN_CROSS",
        "PEAD_FLAG",
        "INSIDER_BUY",
    ]

    result = df.clone()
    z_capped_cols: list[pl.Expr] = []

    for fname in continuous:
        if fname not in result.columns:
            continue
        col = pl.col(fname)
        mu = col.mean()
        sigma = col.std(ddof=1)  # sample std
        # Avoid division by zero: if sigma == 0, z_score = 0
        z_score = pl.when(sigma > 1e-12).then((col - mu) / sigma).otherwise(0.0)
        # Missing values keep z_score = 0; display score = 50
        z_score = pl.when(col.is_null()).then(0.0).otherwise(z_score)
        z_capped = z_score.clip(-Z_SCORE_CAP, Z_SCORE_CAP)

        display_score = DISPLAY_SCORE_NEUTRAL + z_capped * (50.0 / Z_SCORE_CAP)

        result = result.with_columns(
            z_score.alias(f"z_{fname}"),
            z_capped.alias(f"z_capped_{fname}"),
            display_score.alias(f"score_{fname}"),
        )
        z_capped_cols.append(z_capped)

    # Binary factors: 0 -> score 0, 1 -> score 100, null -> 50
    # They contribute 0 to final_score (no z_capped contribution)
    for fname in binary_factors:
        if fname not in result.columns:
            continue
        col = pl.col(fname)
        display_score = (
            pl.when(col.is_null())
            .then(DISPLAY_SCORE_NEUTRAL)
            .when(col == 1)
            .then(100.0)
            .otherwise(0.0)
        )
        result = result.with_columns(display_score.alias(f"score_{fname}"))

    # Final composite score: sum of z_capped for continuous factors
    if z_capped_cols:
        final_score = z_capped_cols[0]
        for expr in z_capped_cols[1:]:
            final_score = final_score + expr
        result = result.with_columns(final_score.alias("final_score"))
    else:
        result = result.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("final_score"))

    return result


# -- missing-data validation -------------------------------------------------


def _validate_missing_data(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Tag tickers with insufficient data and neutralise individual missing factors.

    Per PRD 3.1.4:
      - Price/volume missing -> ticker excluded upstream (done by caller).
      - Single factor missing -> z-score = 0, display score = 50.
      - Factor missing rate > 30% -> mark ``data_sufficient = False``.

    Adds column ``data_sufficient`` (bool) and ``missing_rate`` (f64).
    """
    # Factor columns (original, not the score/z ones)
    factor_cols_in_df = [f for f in FACTOR_NAMES if f in df.columns]
    if not factor_cols_in_df:
        return df.with_columns(
            pl.lit(True, dtype=pl.Boolean).alias("data_sufficient"),
            pl.lit(0.0, dtype=pl.Float64).alias("missing_rate"),
        )

    # Count nulls per row across factor columns only
    null_counts = pl.lit(0, dtype=pl.Int32)
    for f in factor_cols_in_df:
        null_counts = null_counts + pl.col(f).is_null().cast(pl.Int32)

    n_factors = len(factor_cols_in_df)
    missing_rate = null_counts / n_factors

    sufficient = missing_rate <= MISSING_FACTOR_RATE_MAX

    return df.with_columns(
        sufficient.alias("data_sufficient"),
        missing_rate.alias("missing_rate"),
    )


# -- chunking ----------------------------------------------------------------


def _chunk_tickers(tickers: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[str]]:
    """Split a list of tickers into fixed-size batches."""
    chunks = []
    for i in range(0, len(tickers), batch_size):
        chunks.append(list(tickers[i : i + batch_size]))
    return chunks


# -- per-chunk processing ----------------------------------------------------


def process_chunk(
    df: pl.DataFrame,
    *,
    reference_date: date | None = None,
    earnings_dates: dict[str, list[date]] | None = None,
    insider_ratio: dict[str, float] | None = None,
    revenue_growth: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """Compute all 13 factors for a single chunk of OHLCV data.

    Pipeline:
      1. Compute 11 technical factors from OHLCV.
      2. Apply fundamental factors (PEAD_FLAG, INSIDER_BUY, REV_ACCEL).
      3. Validate missing data.

    Normalisation must be applied by the caller after concatenating all
    chunks (see :func:`compute_factors` or :meth:`FactorEngine.run`).

    Args:
        df: OHLCV DataFrame with columns ``ticker, dt, open, high, low, close, volume``.
        reference_date: Observation date (for PEAD_FLAG window).
        earnings_dates: ``{ticker: [earnings_dates]}``.
        insider_ratio: ``{ticker: insider_buy_ratio}``.
        revenue_growth: ``{ticker: [revenue_growth_rates]}``.

    Returns:
        DataFrame with factor, score, and metadata columns appended.
    """
    if df.height == 0:
        return df

    required = {"ticker", "dt", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    _logger.debug("Processing chunk: %d rows, %d tickers", df.height, df["ticker"].n_unique())

    # 1. Technical factors (OHLCV only)
    df = compute_all_technical_factors(df)

    # 2. Fundamental factors
    df = compute_pead_flag(df, earnings_dates=earnings_dates, reference_date=reference_date)
    df = compute_insider_buy(df, insider_ratio=insider_ratio)
    df = compute_rev_accel(df, revenue_growth=revenue_growth)

    # 3. Missing-data tags
    df = _validate_missing_data(df)

    return df


# -- engine ------------------------------------------------------------------


class FactorEngine:
    """Compute and persist factor scores using chunked streaming.

    The engine reads OHLCV data for a given date, splits tickers into batches,
    computes factors + normalisation per batch, and writes results to the
    Hive-partitioned Parquet store under the ``factors`` category.

    Usage::

        engine = FactorEngine(batch_size=500, n_batches=4)
        engine.run(date="2025-01-15")
    """

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_batches: int = DEFAULT_N_BATCHES,
    ) -> None:
        self.batch_size = batch_size
        self.n_batches = n_batches

    def run(
        self,
        dt: date | str,
        *,
        earnings_dates: dict[str, list[date]] | None = None,
        insider_ratio: dict[str, float] | None = None,
        revenue_growth: dict[str, list[float]] | None = None,
    ) -> pl.DataFrame:
        """Run the full factor computation pipeline for a single date.

        Args:
            dt: Observation date.
            earnings_dates: ``{ticker: [earnings_dates]}`` for PEAD_FLAG.
            insider_ratio: ``{ticker: ratio}`` for INSIDER_BUY.
            revenue_growth: ``{ticker: [growth_rates]}`` for REV_ACCEL.

        Returns:
            Combined DataFrame with all factor and score columns for all tickers.
        """
        from datetime import date as date_type

        from alphascreener.data.io import scan_parquet

        if isinstance(dt, str):
            dt = date_type.fromisoformat(dt)

        _logger.info("Factor engine: starting computation for %s", dt.isoformat())

        # Read OHLCV data for the target date
        try:
            lf = scan_parquet("ohlcv", date_filter=dt)
        except FileNotFoundError:
            _logger.warning("No OHLCV data found for %s", dt.isoformat())
            return pl.DataFrame()

        # Collect all tickers and chunk them
        tickers_all = sorted(lf.select("ticker").unique().collect().get_column("ticker").to_list())
        if not tickers_all:
            _logger.warning("No tickers in OHLCV data for %s", dt.isoformat())
            return pl.DataFrame()

        chunks = _chunk_tickers(tickers_all, self.batch_size)
        _logger.info(
            "Splitting %d tickers into %d chunks (batch_size=%d)",
            len(tickers_all),
            len(chunks),
            self.batch_size,
        )

        # Limit to n_batches
        results: list[pl.DataFrame] = []
        for idx, chunk_tickers in enumerate(chunks[: self.n_batches]):
            _logger.debug(
                "Chunk %d/%d: %d tickers",
                idx + 1,
                len(chunks[: self.n_batches]),
                len(chunk_tickers),
            )

            # Filter scan to this chunk's tickers and materialise
            chunk_df = lf.filter(pl.col("ticker").is_in(chunk_tickers)).collect()

            if chunk_df.height == 0:
                _logger.debug("Chunk %d: empty, skipping", idx + 1)
                continue

            processed = process_chunk(
                chunk_df,
                reference_date=dt,
                earnings_dates=earnings_dates,
                insider_ratio=insider_ratio,
                revenue_growth=revenue_growth,
            )
            results.append(processed)

        if not results:
            _logger.warning("All chunks empty for %s", dt.isoformat())
            return pl.DataFrame()

        # Combine all chunks
        combined = pl.concat(results, how="diagonal_relaxed")
        combined = normalize_factors(combined)
        _logger.info(
            "Factor computation complete: %d rows, %d tickers",
            combined.height,
            combined["ticker"].n_unique(),
        )

        # Persist to Parquet
        _persist_factors(combined)

        return combined


# -- convenience function ----------------------------------------------------


def compute_factors(
    df: pl.DataFrame,
    *,
    dt: date | str | None = None,
    earnings_dates: dict[str, list[date]] | None = None,
    insider_ratio: dict[str, float] | None = None,
    revenue_growth: dict[str, list[float]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pl.DataFrame:
    """Convenience function: compute all factors on an in-memory DataFrame.

    Splits *df* into chunks by ticker and processes each through
    :func:`process_chunk`.

    Args:
        df: OHLCV DataFrame with columns ``ticker, dt, open, high, low, close,
            volume``.
        dt: Optional observation date (for PEAD_FLAG).
        earnings_dates: Optional earnings calendar.
        insider_ratio: Optional insider buy data.
        revenue_growth: Optional revenue growth data.
        batch_size: Number of tickers per chunk.

    Returns:
        DataFrame with factor and score columns appended.
    """
    from datetime import date as date_type

    if isinstance(dt, str):
        dt = date_type.fromisoformat(dt)

    if df.height == 0:
        return df

    # Defensive dedup: keep last occurrence of duplicate (ticker, dt) rows,
    # then sort so downstream time-series ops (shift, rolling_*) see correct order.
    df = df.unique(subset=["ticker", "dt"], keep="last", maintain_order=True).sort(["ticker", "dt"])

    tickers = sorted(df["ticker"].unique().to_list())
    chunks = _chunk_tickers(tickers, batch_size)

    results: list[pl.DataFrame] = []
    for chunk_tickers in chunks:
        chunk_df = df.filter(pl.col("ticker").is_in(chunk_tickers))
        processed = process_chunk(
            chunk_df,
            reference_date=dt,
            earnings_dates=earnings_dates,
            insider_ratio=insider_ratio,
            revenue_growth=revenue_growth,
        )
        results.append(processed)

    combined = pl.concat(results, how="diagonal_relaxed")
    combined = normalize_factors(combined)
    return combined


# -- persistence -------------------------------------------------------------


def _persist_factors(df: pl.DataFrame) -> None:
    """Write factor scores to the Parquet store under the ``factors`` category."""
    if df.height == 0:
        return
    try:
        write_parquet(df, "factors")
        _logger.info("Persisted %d factor rows to Parquet", df.height)
    except Exception:
        _logger.exception("Failed to persist factor data")
        raise
