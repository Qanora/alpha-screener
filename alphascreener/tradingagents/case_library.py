"""Breakout case library builder — populates ``cases.parquet`` for the faiss retriever.

Issue #190: The breakout case library was never initialised, so the
BreakoutCaseRetriever always returns ``[]`` and the breakout analyst has
no similar-historical-case context.

This module provides:
  - :class:`CaseLibraryBuilder` — reads historical factor + OHLCV data,
    identifies positive breakout samples (high breakout_score
    AND T+7 forward return >= 10%), and writes them to
    ``~/.alphascreener/data/case_library/cases.parquet``.
  - :func:`rebuild_case_library` — convenience function called from the CLI
    and optionally from the daily-scan post-processing hook.

Schema of ``cases.parquet``:
  ============= ====== ===================================================
  ticker        str    Stock symbol
  date          str    Observation date (YYYY-MM-DD)
  actual_pnl    f64    T+7 forward return (used as the "label" for similarity)
  f_mom_5d      f64    z_capped_MOM_5D
  f_pth         f64    z_capped_PTH
  f_mom_slope   f64    z_capped_MOM_SLOPE
  f_bb_squeeze  f64    z_capped_BB_SQUEEZE
  f_atr_ratio   f64    z_capped_ATR_RATIO
  f_mfi_14      f64    z_capped_MFI_14
  f_cmf_21      f64    z_capped_CMF_21
  f_vol_anomaly f64    z_capped_VOL_ANOMALY
  f_rsi_ovs     f64    z_capped_RSI_OVERSOLD
  f_rev_accel   f64    z_capped_REV_ACCEL
  ============= ====== ===================================================
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import timedelta
from pathlib import Path

import polars as pl

from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CASE_LIBRARY_DIR: Path = Path.home() / ".alphascreener" / "data" / "case_library"
_CASES_PARQUET: Path = _CASE_LIBRARY_DIR / "cases.parquet"

# Factor vector columns in factor Parquet (z_capped_{name}) mapped to
# the ``f_*`` column names expected by BreakoutCaseRetriever.
_FACTOR_VECTOR_MAP: list[tuple[str, str]] = [
    ("z_capped_MOM_5D", "f_mom_5d"),
    ("z_capped_PTH", "f_pth"),
    ("z_capped_MOM_SLOPE", "f_mom_slope"),
    ("z_capped_BB_SQUEEZE", "f_bb_squeeze"),
    ("z_capped_ATR_RATIO", "f_atr_ratio"),
    ("z_capped_MFI_14", "f_mfi_14"),
    ("z_capped_CMF_21", "f_cmf_21"),
    ("z_capped_VOL_ANOMALY", "f_vol_anomaly"),
    ("z_capped_RSI_OVERSOLD", "f_rsi_ovs"),
    ("z_capped_REV_ACCEL", "f_rev_accel"),
]

# Thresholds for positive breakout cases
DEFAULT_BREAKOUT_SCORE_PERCENTILE: float = 0.75  # Top 25% by breakout_score
DEFAULT_MIN_RETURN: float = 0.10  # T+7 return >= 10% (matches alpha acceptance)
DEFAULT_FORWARD_DAYS: int = 7  # Calendar days forward for return computation


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class CaseLibraryBuilder:
    """Build the breakout case library from historical factor + OHLCV data.

    Usage::

        builder = CaseLibraryBuilder()
        n_added = builder.rebuild()
        # n_added == number of positive cases written to cases.parquet
    """

    def __init__(
        self,
        *,
        breakout_score_pct: float = DEFAULT_BREAKOUT_SCORE_PERCENTILE,
        min_return: float = DEFAULT_MIN_RETURN,
        forward_days: int = DEFAULT_FORWARD_DAYS,
        output_path: Path | None = None,
    ) -> None:
        if not 0.0 <= breakout_score_pct <= 1.0:
            raise ValueError("breakout_score_pct must be within [0.0, 1.0]")
        if not 0.0 <= min_return <= 1.0:
            raise ValueError("min_return must be within [0.0, 1.0]")
        if forward_days <= 0:
            raise ValueError("forward_days must be > 0")
        self._breakout_score_pct = breakout_score_pct
        self._min_return = min_return
        self._forward_days = forward_days
        self._output_path = output_path or _CASES_PARQUET

    # -- public API ------------------------------------------------------------

    def rebuild(self) -> int:
        """Rebuild cases.parquet from all available historical factor data.

        Returns:
            Number of positive breakout cases written.
        """
        # 1. Load factor data
        factors_df = self._load_all_factors()
        if factors_df is None or factors_df.height == 0:
            self._write_empty_library()
            _logger.warning("No factor data found — case library remains empty")
            return 0

        _logger.info(
            "Loaded %d factor rows across %d unique (ticker, dt) pairs",
            factors_df.height,
            factors_df.select(["ticker", "dt"]).unique().height,
        )

        # 2. Ensure breakout_score column exists (compute if missing)
        if "breakout_score" not in factors_df.columns:
            factors_df = self._compute_breakout_score(factors_df)

        # 3. Compute forward returns
        factors_df = self._compute_forward_returns(factors_df)

        # 4. Select positive breakout cases
        cases = self._select_positive_cases(factors_df)
        if cases.height == 0:
            self._write_empty_library()
            _logger.warning(
                "No positive breakout cases found (score_pct=%.0f, min_return=%.0f%%) — "
                "case library remains empty",
                self._breakout_score_pct * 100,
                self._min_return * 100,
            )
            return 0

        # 5. Map to case library schema
        case_rows = self._to_case_schema(cases).unique(
            subset=["ticker", "date"], keep="last", maintain_order=True
        )

        # 6. Write
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        case_rows.write_parquet(str(self._output_path))

        _logger.info(
            "Wrote %d positive breakout cases to %s",
            case_rows.height,
            self._output_path,
        )
        return case_rows.height

    def append_date(self, dt: date_type, *, df: pl.DataFrame | None = None) -> int:
        """Append cases for a single date to the existing case library.

        Reads factor data for *dt* from the Parquet store, computes forward
        returns, selects positive cases, and appends them to ``cases.parquet``.
        Existing cases are NOT overwritten; duplicate (ticker, date) rows are
        deduplicated (keep last).

        Args:
            dt: Observation date.
            df: Optional pre-loaded factor DataFrame. When provided, this is
                used instead of reading from disk (useful for testing).

        Returns:
            Number of *new* cases appended.
        """
        if df is not None:
            factors_df = df
        else:
            from alphascreener.data.io import scan_parquet

            try:
                factors_df = scan_parquet("factors", date_filter=dt).collect()
            except FileNotFoundError:
                _logger.warning("No factor data found for %s", dt.isoformat())
                return 0

        if factors_df.height == 0:
            return 0

        if "breakout_score" not in factors_df.columns:
            factors_df = self._compute_breakout_score(factors_df)

        factors_df = self._compute_forward_returns(factors_df)
        cases = self._select_positive_cases(factors_df)
        if cases.height == 0:
            return 0

        new_rows = self._to_case_schema(cases)

        # Merge with existing cases (upsert: keep last for duplicate keys)
        existing: pl.DataFrame | None = None
        if self._output_path.exists():
            try:
                existing = pl.read_parquet(str(self._output_path))
            except Exception:
                _logger.exception("Failed to read existing case library, cannot append safely")
                raise

        n_before = existing.height if existing is not None and existing.height > 0 else 0
        if existing is not None and existing.height > 0:
            merged = pl.concat([existing, new_rows], how="diagonal_relaxed")
            merged = merged.unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        else:
            merged = new_rows

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(str(self._output_path))
        n_new = max(0, merged.height - n_before)
        _logger.info("Appended %d cases for %s (total: %d)", n_new, dt.isoformat(), merged.height)
        return n_new

    def status(self) -> dict:
        """Return status information about the case library.

        Returns:
            Dict with keys ``path``, ``exists``, ``n_cases``,
            ``n_unique_tickers``, ``date_range``.
        """
        info: dict = {
            "path": str(self._output_path),
            "exists": self._output_path.exists(),
            "n_cases": 0,
            "n_unique_tickers": 0,
            "date_range": None,
        }
        if not self._output_path.exists():
            return info

        try:
            df = pl.read_parquet(str(self._output_path))
        except Exception as e:
            _logger.warning("Failed to read case library for status", exc_info=True)
            info["error"] = str(e)
            info["corrupt"] = True
            info["exists"] = False
            return info

        if df.height == 0:
            return info

        info["n_cases"] = df.height
        info["n_unique_tickers"] = df["ticker"].n_unique() if "ticker" in df.columns else 0

        if "date" in df.columns:
            dates = df["date"].sort()
            info["date_range"] = (dates[0], dates[-1])

        return info

    # -- internal --------------------------------------------------------------

    def _write_empty_library(self) -> None:
        """Write an empty case library with the canonical schema to disk.

        Ensures any stale ``cases.parquet`` from a previous build is replaced
        with a clean empty file so that the on-disk state matches the result of
        :meth:`rebuild` (zero cases).
        """
        schema: dict[str, type] = {
            "ticker": str,
            "date": str,
            "actual_pnl": float,
        }
        for _, f_col in _FACTOR_VECTOR_MAP:
            schema[f_col] = float
        empty_df = pl.DataFrame(schema=schema)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.write_parquet(str(self._output_path))
        _logger.debug("Wrote empty case library to %s", self._output_path)

    def _load_all_factors(self) -> pl.DataFrame | None:
        """Load all historical factor data from the Parquet store."""
        from alphascreener.data.io import scan_parquet

        try:
            return scan_parquet("factors").collect()
        except FileNotFoundError:
            return None

    def _compute_breakout_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute breakout_score via Phase 2 weights if not present."""
        from alphascreener.screening.phase2 import compute_breakout_score

        _logger.debug("Computing breakout_score for %d rows", df.height)
        return compute_breakout_score(df)

    def _compute_forward_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``t7_return`` column: (close_{t+N} - close_t) / close_t.

        Reads OHLCV data for the forward date to get the future close price.
        Tickers whose forward close is unavailable get a null t7_return and
        are excluded from positive case selection.

        If ``t7_return`` column already exists, it is preserved as-is
        (the caller has already computed forward returns).
        """
        if df.height == 0:
            return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("t7_return"))

        # If t7_return already present, skip re-computation
        if "t7_return" in df.columns:
            return df

        # Collect all unique observation dates
        dates = sorted(df["dt"].unique().to_list())
        if not dates:
            return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("t7_return"))

        _logger.debug("Computing forward returns for %d dates", len(dates))

        # For each date, read OHLCV data at t+N and merge
        forward_map: dict[tuple[str, date_type], float] = {}

        from alphascreener.data.io import scan_parquet

        for obs_date in dates:
            if isinstance(obs_date, str):
                obs_date = date_type.fromisoformat(obs_date)
            elif type(obs_date) is not date_type:
                obs_date = date_type.fromisoformat(str(obs_date)[:10])
            fwd_date = obs_date + timedelta(days=self._forward_days)

            try:
                ohlcv = scan_parquet("ohlcv", date_filter=fwd_date).collect()
            except FileNotFoundError:
                _logger.debug("No OHLCV data for forward date %s", fwd_date.isoformat())
                continue

            if ohlcv.height == 0:
                continue

            # Build price lookup: ticker -> close
            for row in ohlcv.select(["ticker", "close"]).iter_rows(named=True):
                ticker = str(row["ticker"])
                close_val = row["close"]
                if close_val is None:
                    continue
                forward_map[(ticker, obs_date)] = float(close_val)

        if not forward_map:
            _logger.warning("No forward price data available — all t7_return will be null")
            return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("t7_return"))

        # Build lookup DataFrame
        fwd_rows = []
        for (ticker, dt_val), fwd_close in forward_map.items():
            fwd_rows.append({"ticker": ticker, "dt": dt_val, "_fwd_close": fwd_close})
        fwd_df = pl.DataFrame(fwd_rows)

        # Left-join to factor data
        result = df.join(fwd_df, on=["ticker", "dt"], how="left")

        # Compute t7_return (guard against close <= 0 to avoid Inf/-Inf)
        if "close" in result.columns:
            result = result.with_columns(
                pl.when(
                    pl.col("_fwd_close").is_not_null()
                    & pl.col("close").is_not_null()
                    & (pl.col("close") > 0)
                )
                .then((pl.col("_fwd_close") - pl.col("close")) / pl.col("close"))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("t7_return")
            )
        else:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("t7_return"))

        # Drop helper column
        if "_fwd_close" in result.columns:
            result = result.drop("_fwd_close")

        n_with_return = result["t7_return"].drop_nulls().len()
        _logger.debug(
            "Computed forward returns: %d / %d rows have t7_return", n_with_return, result.height
        )
        return result

    def _select_positive_cases(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select positive breakout cases.

        A positive case: breakout_score >= percentile threshold
        AND t7_return >= min_return AND t7_return is not null
        AND the factor vector is complete (no nulls in z_capped cols).
        """
        if df.height == 0:
            return df

        required_z = [z_col for z_col, _ in _FACTOR_VECTOR_MAP]
        missing_required = [c for c in required_z if c not in df.columns]
        if missing_required:
            raise ValueError(
                f"Missing required z_capped columns ({len(missing_required)}): "
                + ", ".join(missing_required)
            )
        z_cols_present = required_z

        # Compute score threshold (top percentile)
        if "breakout_score" in df.columns:
            score_thresh = df["breakout_score"].quantile(self._breakout_score_pct)
        else:
            score_thresh = 0.0

        _logger.debug(
            "Breakout score threshold (pct=%.0f): %.4f",
            self._breakout_score_pct * 100,
            score_thresh,
        )

        # Filter
        mask = pl.col("breakout_score") >= score_thresh

        if "t7_return" in df.columns:
            mask = mask & pl.col("t7_return").is_not_null()
            mask = mask & (pl.col("t7_return") >= self._min_return)

        for zc in z_cols_present:
            mask = mask & pl.col(zc).is_not_null()

        return df.filter(mask)

    def _to_case_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map factor DataFrame columns to the case library schema."""
        # Build the output schema
        selects: list[pl.Expr] = [
            pl.col("ticker"),
            pl.col("dt").cast(pl.String).alias("date"),
        ]

        # actual_pnl: use t7_return if available, otherwise 0.0
        if "t7_return" in df.columns:
            selects.append(pl.col("t7_return").alias("actual_pnl"))
        else:
            selects.append(pl.lit(0.0, dtype=pl.Float64).alias("actual_pnl"))

        # Map z_capped_* columns to f_* columns
        # All required columns are guaranteed present by _select_positive_cases
        for z_col, f_col in _FACTOR_VECTOR_MAP:
            selects.append(pl.col(z_col).cast(pl.Float64).alias(f_col))

        return df.select(selects)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def rebuild_case_library(
    *,
    breakout_score_pct: float = DEFAULT_BREAKOUT_SCORE_PERCENTILE,
    min_return: float = DEFAULT_MIN_RETURN,
    forward_days: int = DEFAULT_FORWARD_DAYS,
) -> int:
    """Rebuild the breakout case library from all available historical data.

    Args:
        breakout_score_pct: Percentile threshold for breakout_score (0.0-1.0).
        min_return: Minimum T+7 forward return to qualify as positive.
        forward_days: Number of calendar days forward for return computation.

    Returns:
        Number of positive breakout cases written.
    """
    builder = CaseLibraryBuilder(
        breakout_score_pct=breakout_score_pct,
        min_return=min_return,
        forward_days=forward_days,
    )
    return builder.rebuild()


def case_library_status() -> dict:
    """Return status information about the existing case library."""
    builder = CaseLibraryBuilder()
    return builder.status()
