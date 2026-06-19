"""Data sync orchestrator — coordinates yfinance, Stooq, FMP data sources.

Orchestrates the daily data sync pipeline (PRD 7.4 step 1):

1. yfinance downloads OHLCV + fundamentals + news for the full universe
2. Stooq cross-validates OHLCV for the top 100 + top 20 names
3. FMP supplements with analyst estimates, insider trading, grades
4. Write Parquet partitions under ohlcv/dt=YYYY-MM-DD/
5. Integrity check: NaN rate per field must be < 5% of universe
6. Continuity alert: yfinance 3 consecutive days failure rate > 30%

Issue #92: Data sync orchestrator.
Reference: PRD 7.1 / 7.2 / 7.4.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from alphascreener.data import write_parquet
from alphascreener.logging import get_logger
from alphascreener.sources.fmp_adapter import FmpAdapter, FmpBudgetExhaustedError
from alphascreener.sources.stooq_adapter import StooqAdapter
from alphascreener.sources.yfinance_adapter import YFinanceAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many trading days to pull for incremental updates
DEFAULT_LOOKBACK_DAYS: int = 5

# Cross-validation sample sizes per PRD 7.2
STOOQ_VALIDATE_TOP_N: int = 120  # top 100 + top 20

# Integrity: max allowed NaN fraction per field (daily average across tickers)
MAX_NAN_FRACTION: float = 0.05

# Continuity: yfinance failure rate threshold over 3 consecutive days
CONTINUITY_WINDOW_DAYS: int = 3
CONTINUITY_FAILURE_RATE_THRESHOLD: float = 0.30

# diff threshold for Stooq cross-validation (relative difference > 0.5%)
DIFF_THRESHOLD_PCT: float = 0.005

# Auto-recovery: health probe tickers (Issue #224)
# These well-known liquid tickers are used to test whether yfinance is healthy.
HEALTH_PROBE_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Auto-recovery: minimum successful ticker fraction to consider primary source healthy
HEALTH_PROBE_SUCCESS_THRESHOLD: float = 0.80

# Auto-recovery: maximum consecutive days in fallback mode before forced health probe
MAX_FALLBACK_DAYS: int = 1


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    """Summary of a single data sync run.

    Attributes:
        tickers_total: Total number of tickers requested.
        tickers_succeeded: Number of tickers with at least partial data.
        tickers_failed: Number of tickers with no data at all.
        ohlcv_rows: Total OHLCV rows written to Parquet.
        stooq_validated: Number of tickers validated against Stooq.
        stooq_diff_count: Number of field-level diffs exceeding threshold.
        fmp_enriched: Number of tickers enriched with FMP data.
        fmp_budget_exhausted: Whether FMP daily budget was exhausted.
        integrity: IntegrityReport if integrity check ran, None otherwise.
        elapsed_s: Wall-clock time for this sync run.
        errors: List of non-fatal error messages encountered.
    """

    tickers_total: int = 0
    tickers_succeeded: int = 0
    tickers_failed: int = 0
    ohlcv_rows: int = 0
    stooq_validated: int = 0
    stooq_diff_count: int = 0
    fmp_enriched: int = 0
    fmp_budget_exhausted: bool = False
    integrity: IntegrityReport | None = None
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class IntegrityReport:
    """Result of an OHLCV data integrity check.

    Attributes:
        total_tickers: Number of tickers in the dataset.
        nan_counts: Dict mapping field name → total NaN count.
        nan_fractions: Dict mapping field name → NaN fraction (0.0–1.0).
        passed: Whether all fields are below MAX_NAN_FRACTION.
    """

    total_tickers: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    nan_fractions: dict[str, float] = field(default_factory=dict)
    passed: bool = True


# ---------------------------------------------------------------------------
# SyncOrchestrator
# ---------------------------------------------------------------------------


@dataclass
class SyncOrchestrator:
    """Coordinates yfinance, Stooq cross-validation, and FMP enrichment.

    Usage::

        orch = SyncOrchestrator(yfinance=YFinanceAdapter(),
                                stooq=StooqAdapter(),
                                fmp=FmpAdapter(api_key="..."))
        report = await orch.sync(tickers=["AAPL", "GOOGL", ...])

    To persist cross-validation diffs to the monitoring database, pass a
    :class:`~alphascreener.cross_validation.diff_store.DiffStore` via
    *diff_store*::

        from alphascreener.cross_validation import DiffStore
        from alphascreener.config import Settings

        settings = Settings()
        store = DiffStore(settings.get_db_url())
        orch = SyncOrchestrator(..., diff_store=store)
    """

    yfinance: YFinanceAdapter
    stooq: StooqAdapter | None = None
    fmp: FmpAdapter | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    stooq_validate_n: int = STOOQ_VALIDATE_TOP_N
    max_nan_fraction: float = MAX_NAN_FRACTION
    diff_store: Any | None = None  # DiffStore for persisting data_source_diff records
    health_probe_tickers: list[str] = field(
        default_factory=lambda: list(HEALTH_PROBE_TICKERS)
    )

    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("screening"), repr=False, init=False
    )

    # -- Continuity tracker (persistent across sync runs) ---------------------

    _daily_failure_rates: list[float] = field(default_factory=list, repr=False, init=False)

    # -- Auto-recovery state (Issue #224) --------------------------------------

    _primary_healthy: bool = field(default=True, repr=False, init=False)
    _consecutive_fallback_days: int = field(default=0, repr=False, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def sync(
        self,
        tickers: list[str],
        *,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        incremental: bool = True,
    ) -> SyncReport:
        """Execute a full data sync cycle.

        Steps:
          1. Compute date range (incremental or explicit).
          2. Download OHLCV via yfinance.
          3. Cross-validate top-N OHLCV against Stooq.
          4. Enrich tickers via FMP (analyst estimates, insider trading).
          5. Write OHLCV to Parquet partitions.
          6. Run integrity check.
          7. Update continuity tracker.

        Args:
            tickers: List of ticker symbols to sync.
            start_date: Start date (inclusive).  If *None* and *incremental*
                is True, defaults to *lookback_days* ago.
            end_date: End date (inclusive).  Defaults to today UTC.
            incremental: When True (default) and no explicit *start_date*,
                only the last *lookback_days* calendar days are pulled.

        Returns:
            SyncReport summarizing the run.
        """
        t0 = datetime.now(UTC)
        report = SyncReport(tickers_total=len(tickers))

        if not tickers:
            report.elapsed_s = (datetime.now(UTC) - t0).total_seconds()
            return report

        # -- Step 0: compute date range --------------------------------------
        if end_date is None:
            end = date.today()
        elif isinstance(end_date, str):
            end = date.fromisoformat(end_date)
        else:
            end = end_date

        if start_date is not None:
            if isinstance(start_date, str):
                start = date.fromisoformat(start_date)
            else:
                start = start_date
        elif incremental:
            start = end - timedelta(days=self.lookback_days)
        else:
            start = end - timedelta(days=365)  # full year fallback

        self._logger.info(
            "Data sync: %d tickers, %s → %s (incremental=%s)",
            len(tickers),
            start.isoformat(),
            end.isoformat(),
            incremental,
        )

        # -- Step 1: yfinance OHLCV download (primary source) -----------------
        ohlcv_df = await self._download_primary(tickers, start, end, report)

        # Per-ticker failure tracking
        succeeded_tickers: set[str] = set()
        if ohlcv_df.height > 0:
            succeeded_tickers = set(ohlcv_df["ticker"].unique().to_list())
        report.tickers_failed = len(tickers) - len(succeeded_tickers)

        # -- Step 1b: OHLCV fallback via Stooq for failed tickers (Issue #224) -
        if self.stooq is not None and report.tickers_failed > 0:
            failed_tickers = [t for t in tickers if t not in succeeded_tickers]
            self._logger.warning(
                "yfinance missed %d/%d tickers — falling back to Stooq OHLCV",
                len(failed_tickers),
                len(tickers),
            )
            fallback_df = await self._download_fallback(failed_tickers, start, end)
            if fallback_df.height > 0:
                fallback_succeeded = set(fallback_df["ticker"].unique().to_list())
                report.tickers_failed -= len(fallback_succeeded)
                self._logger.info(
                    "Stooq fallback covered %d additional tickers (%d still missing)",
                    len(fallback_succeeded),
                    report.tickers_failed,
                )
                # Merge fallback results into the main OHLCV DataFrame
                combined = [ohlcv_df] if ohlcv_df.height > 0 else []
                combined.append(fallback_df)
                ohlcv_df = pl.concat(combined)
                report.ohlcv_rows = ohlcv_df.height
                report.tickers_succeeded = (
                    len(ohlcv_df["ticker"].unique()) if ohlcv_df.height > 0 else 0
                )

        # -- Step 2: Stooq cross-validation (top N) --------------------------
        if self.stooq is not None and ohlcv_df.height > 0:
            report.stooq_validated, report.stooq_diff_count = await self._cross_validate(
                ohlcv_df, start, end
            )

        # -- Step 3: FMP enrichment (analyst estimates, insider trading) -----
        if self.fmp is not None:
            try:
                enriched = await self._enrich_fmp(tickers, report)
                report.fmp_enriched = enriched
            except FmpBudgetExhaustedError:
                report.fmp_budget_exhausted = True
                self._logger.warning("FMP daily budget exhausted — skipping enrichment")
            except Exception as e:
                self._logger.warning("FMP enrichment failed: %s", e)
                report.errors.append(f"FMP enrichment: {e}")

        # -- Step 4: Write OHLCV to Parquet partitions ------------------------
        if ohlcv_df.height > 0:
            try:
                write_parquet(ohlcv_df, "ohlcv")
                self._logger.info(
                    "Wrote %d OHLCV rows to Parquet (%d unique trading days)",
                    ohlcv_df.height,
                    len(ohlcv_df["dt"].unique()),
                )
            except Exception as e:
                self._logger.error("Failed to write OHLCV Parquet: %s", e)
                report.errors.append(f"Parquet write: {e}")

        # -- Step 5: Stooq diff persistence (data_source_diff) ----------------
        # Performed inside _cross_validate → the diffs are returned; the DB
        # write is left to the caller or a separate step for testability.

        # -- Step 6: Integrity check ------------------------------------------
        if ohlcv_df.height > 0:
            report.integrity = self._check_integrity(ohlcv_df, len(tickers))

        # -- Step 7: Continuity tracker + auto-recovery update (Issue #224) -----
        if len(tickers) > 0:
            failure_rate = report.tickers_failed / len(tickers)
            self._update_continuity(failure_rate, report)
            # Auto-recovery: track fallback days and probe primary source health
            self._update_auto_recovery(failure_rate)

        report.elapsed_s = (datetime.now(UTC) - t0).total_seconds()
        self._logger.info(
            "Data sync complete: %d/%d tickers succeeded, %d OHLCV rows, %.1fs",
            report.tickers_succeeded,
            report.tickers_total,
            report.ohlcv_rows,
            report.elapsed_s,
        )

        return report

    # ------------------------------------------------------------------
    # Primary / fallback OHLCV download (Issue #224)
    # ------------------------------------------------------------------

    async def _download_primary(
        self,
        tickers: list[str],
        start: date,
        end: date,
        report: SyncReport,
    ) -> pl.DataFrame:
        """Download OHLCV from the primary source (yfinance).

        When the primary source is known to be unhealthy (auto-recovery has
        flagged it), a health probe is attempted first to see if the source
        has recovered. If the probe succeeds, circuit breakers are reset
        and the primary source is used normally.
        """
        if not self._primary_healthy:
            self._logger.warning(
                "Primary source (yfinance) was flagged unhealthy — probing health"
            )
            healthy = await self._probe_primary_health()
            if healthy:
                self._logger.info(
                    "Primary source health probe PASSED — resetting circuit breakers "
                    "and resuming normal operation"
                )
                self.yfinance.reset_circuit_breakers()
                self._primary_healthy = True
                self._consecutive_fallback_days = 0
            else:
                self._logger.warning(
                    "Primary source health probe FAILED — continuing in fallback mode"
                )

        try:
            ohlcv_df = await self.yfinance.download_ohlcv(tickers, start, end)
            report.ohlcv_rows = ohlcv_df.height
            report.tickers_succeeded = (
                len(ohlcv_df["ticker"].unique()) if ohlcv_df.height > 0 else 0
            )
            return ohlcv_df
        except Exception as e:
            self._logger.error("yfinance OHLCV download failed: %s", e)
            report.errors.append(f"yfinance download: {e}")
            return _empty_ohlcv_df()

    async def _download_fallback(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Download OHLCV from the fallback source (Stooq) for failed tickers.

        This is invoked when yfinance fails to return data for some tickers.
        Stooq is queried ticker-by-ticker and the results are returned in
        the standard OHLCV schema for merging into the main DataFrame.
        """
        if self.stooq is None or not tickers:
            return _empty_ohlcv_df()

        self._logger.info(
            "Fallback: downloading OHLCV from Stooq for %d tickers",
            len(tickers),
        )
        try:
            return await self.stooq.download_ohlcv(tickers, start, end)
        except Exception as e:
            self._logger.error("Stooq fallback download failed: %s", e)
            return _empty_ohlcv_df()

    # ------------------------------------------------------------------
    # Cross-validation (PRD 7.2)
    # ------------------------------------------------------------------

    async def _cross_validate(
        self,
        yf_df: pl.DataFrame,
        start: date,
        end: date,
    ) -> tuple[int, int]:
        """Cross-validate yfinance OHLCV against Stooq for top-N tickers.

        Returns:
            (validated_count, diff_count) tuple.
        """
        if self.stooq is None:
            return 0, 0

        # Select top-N tickers by number of rows (most active first)
        ticker_counts = yf_df.group_by("ticker").agg(pl.len().alias("n")).sort("n", descending=True)
        top_tickers: list[str] = ticker_counts.head(self.stooq_validate_n)["ticker"].to_list()

        # Map standard tickers to Stooq format
        stooq_tickers = [t.lower() for t in top_tickers]

        try:
            stooq_df = await self.stooq.download_ohlcv(
                stooq_tickers, start_date=start, end_date=end
            )
        except Exception as e:
            self._logger.warning("Stooq cross-validation failed: %s", e)
            return 0, 0

        if stooq_df.height == 0:
            self._logger.warning("Stooq returned no data for cross-validation")
            return 0, 0

        # Normalize Stooq tickers back to standard format for comparison
        def _strip_suffix(t: str) -> str:
            return t.rsplit(".", 1)[0] if "." in t else t

        stooq_df = stooq_df.with_columns(
            pl.col("ticker")
            .map_elements(_strip_suffix, return_dtype=pl.Utf8)
            .str.to_uppercase()
            .alias("ticker")
        )

        # Compare field-by-field for each (ticker, dt) pair
        fields = ["open", "high", "low", "close"]
        diff_count = 0
        diff_records: list[dict[str, Any]] = []

        joined = yf_df.filter(pl.col("ticker").is_in(top_tickers)).join(
            stooq_df,
            on=["ticker", "dt"],
            how="inner",
            suffix="_stooq",
        )
        validated_count = joined["ticker"].n_unique() if joined.height > 0 else 0

        from alphascreener.cross_validation.comparator import compute_diff_pct

        for col_name in fields:
            yf_col = col_name
            sq_col = f"{col_name}_stooq"
            if sq_col not in joined.columns:
                continue

            diffs = joined.filter(
                (pl.col(yf_col).abs() > 0)
                & (
                    (pl.col(yf_col) - pl.col(sq_col)).abs() / pl.col(yf_col).abs()
                    > DIFF_THRESHOLD_PCT
                )
            )

            for row in diffs.iter_rows(named=True):
                yf_val = float(row.get(yf_col, 0.0))
                sq_val = float(row.get(sq_col, 0.0))
                diff_pct = compute_diff_pct(yf_val, sq_val)

                self._logger.debug(
                    "Stooq diff: %s %s %s: yf=%s stooq=%s",
                    row.get("ticker"),
                    row.get("dt"),
                    col_name,
                    yf_val,
                    sq_val,
                )
                diff_records.append({
                    "ticker": str(row.get("ticker", "")),
                    "dt": row.get("dt"),
                    "field": col_name,
                    "primary_value": yf_val,
                    "fallback_value": sq_val,
                    "fallback_source": "stooq",
                    "diff_pct": diff_pct,
                })
                diff_count += 1

        if diff_count > 0:
            self._logger.warning(
                "Stooq cross-validation found %d field-level diffs (>%.1f%%)",
                diff_count,
                DIFF_THRESHOLD_PCT * 100,
            )

        # -- Persist diffs to data_source_diff (Issue #216) ------------------
        if diff_records and self.diff_store is not None:
            try:
                from alphascreener.cross_validation.comparator import OHLCVFieldDiffs

                diffs_container = OHLCVFieldDiffs(records=diff_records)
                inserted = self.diff_store.insert_diffs(diffs_container)
                self._logger.info(
                    "Persisted %d data_source_diff records (of %d detected)",
                    inserted,
                    diff_count,
                )
            except Exception:
                self._logger.error(
                    "Failed to persist data_source_diff records (%d records)",
                    len(diff_records),
                    exc_info=True,
                )

        self._logger.info(
            "Stooq cross-validation: %d tickers checked, %d diffs found",
            validated_count,
            diff_count,
        )
        return validated_count, diff_count

    # ------------------------------------------------------------------
    # FMP enrichment
    # ------------------------------------------------------------------

    async def _enrich_fmp(self, tickers: list[str], report: SyncReport) -> int:
        """Enrich tickers with FMP data (analyst estimates, insider trading).

        FMP is called only for the subset of tickers that need detailed
        enrichment.  The orchestrator prioritises tickers in a round-robin
        fashion until the budget is exhausted.

        Returns:
            Number of tickers enriched with FMP data.
        """
        if self.fmp is None or self.fmp.is_budget_exhausted:
            return 0

        enriched = 0
        for ticker in tickers:
            if self.fmp.is_budget_exhausted:
                report.fmp_budget_exhausted = True
                break
            try:
                # Fetch analyst estimates + insider trading for each ticker
                analyst_task = self.fmp.fetch_analyst_estimates(ticker)
                insider_task = self.fmp.fetch_insider_trading(ticker)

                analyst_df, insider_df = await asyncio.gather(
                    analyst_task, insider_task, return_exceptions=True
                )

                if isinstance(analyst_df, Exception):
                    if isinstance(analyst_df, FmpBudgetExhaustedError):
                        report.fmp_budget_exhausted = True
                        break
                    analyst_df = None
                if isinstance(insider_df, Exception):
                    if isinstance(insider_df, FmpBudgetExhaustedError):
                        report.fmp_budget_exhausted = True
                        break
                    insider_df = None

                if analyst_df is None and insider_df is None:
                    continue

                enriched += 1
            except FmpBudgetExhaustedError:
                report.fmp_budget_exhausted = True
                break
            except Exception as e:
                self._logger.warning("FMP enrichment failed for %s: %s", ticker, e)
                report.errors.append(f"FMP enrichment ({ticker}): {e}")
                continue

        self._logger.info("FMP enrichment: %d tickers enriched", enriched)
        return enriched

    # ------------------------------------------------------------------
    # Integrity check (NaN inspection)
    # ------------------------------------------------------------------

    def _check_integrity(self, ohlcv_df: pl.DataFrame, total_tickers: int) -> IntegrityReport:
        """Check OHLCV data integrity: NaN fractions per field.

        The daily average NaN count per field must be < MAX_NAN_FRACTION
        of the total universe.  For example, with 2,000 tickers, no more
        than 100 NaN values per field per day on average.

        Args:
            ohlcv_df: The OHLCV DataFrame to inspect.
            total_tickers: Total number of tickers in the universe (used
                as the denominator for fraction calculation).

        Returns:
            IntegrityReport with per-field NaN statistics.
        """
        fields = ["open", "high", "low", "close", "volume"]
        nan_counts: dict[str, int] = {}
        nan_fractions: dict[str, float] = {}

        for f in fields:
            if f not in ohlcv_df.columns:
                continue
            nan_count = ohlcv_df.filter(pl.col(f).is_null() | pl.col(f).is_nan()).height
            nan_counts[f] = nan_count
            # Normalize by number of trading days present
            if ohlcv_df.height > 0:
                unique_days = len(ohlcv_df["dt"].unique())
            else:
                unique_days = 1
            daily_avg_nan = nan_count / unique_days if unique_days > 0 else nan_count
            nan_fractions[f] = daily_avg_nan / total_tickers if total_tickers > 0 else 1.0

        passed = all(fr < self.max_nan_fraction for fr in nan_fractions.values())

        report = IntegrityReport(
            total_tickers=total_tickers,
            nan_counts=nan_counts,
            nan_fractions=nan_fractions,
            passed=passed,
        )

        if not passed:
            self._logger.warning(
                "Integrity check FAILED: nan fractions=%s",
                {k: f"{v:.4f}" for k, v in nan_fractions.items()},
            )
        else:
            self._logger.debug(
                "Integrity check passed: nan fractions=%s",
                {k: f"{v:.4f}" for k, v in nan_fractions.items()},
            )

        return report

    # ------------------------------------------------------------------
    # Continuity tracker (PRD 7.2)
    # ------------------------------------------------------------------

    def _update_continuity(self, failure_rate: float, report: SyncReport) -> None:
        """Record today's failure rate and check for continuity alert.

        Alert condition (PRD 7.2):
          yfinance failure rate > 30% for 3 consecutive days
        """
        self._daily_failure_rates.append(failure_rate)

        if len(self._daily_failure_rates) < CONTINUITY_WINDOW_DAYS:
            return

        # Keep only the most recent N days
        recent = self._daily_failure_rates[-CONTINUITY_WINDOW_DAYS:]

        if all(r > CONTINUITY_FAILURE_RATE_THRESHOLD for r in recent):
            msg = (
                f"CONTINUITY ALERT: yfinance failure rate exceeded "
                f"{CONTINUITY_FAILURE_RATE_THRESHOLD * 100:.0f}% for "
                f"{CONTINUITY_WINDOW_DAYS} consecutive days: "
                f"{[f'{r * 100:.1f}%' for r in recent]}"
            )
            self._logger.warning(msg)
            report.errors.append(msg)

    def reset_continuity(self) -> None:
        """Reset the continuity tracker (for testing)."""
        self._daily_failure_rates.clear()

    # ------------------------------------------------------------------
    # Auto-recovery (Issue #224)
    # ------------------------------------------------------------------

    def _update_auto_recovery(self, failure_rate: float) -> None:
        """Track fallback days and trigger health probe when threshold reached.

        When the yfinance failure rate exceeds the continuity threshold,
        the primary source is flagged as unhealthy. The system then:
          1. Increments the consecutive fallback days counter
          2. After MAX_FALLBACK_DAYS days, probes yfinance health
          3. If the probe succeeds, resets circuit breakers and resumes
          4. If the probe fails, stays in fallback mode and retries next cycle

        Args:
            failure_rate: Fraction of tickers that yfinance failed to cover
                (0.0 = all succeeded, 1.0 = all failed).
        """
        if failure_rate > CONTINUITY_FAILURE_RATE_THRESHOLD:
            if self._primary_healthy:
                self._logger.warning(
                    "Primary source (yfinance) failure rate %.1f%% > threshold %.0f%% "
                    "— flagging as unhealthy",
                    failure_rate * 100,
                    CONTINUITY_FAILURE_RATE_THRESHOLD * 100,
                )
                self._primary_healthy = False
            self._consecutive_fallback_days += 1
            self._logger.warning(
                "Fallback mode: day %d/%d (failure rate %.1f%%)",
                self._consecutive_fallback_days,
                MAX_FALLBACK_DAYS,
                failure_rate * 100,
            )
        else:
            # Failure rate is acceptable — if we were in fallback mode,
            # consider this a recovery signal.
            if not self._primary_healthy:
                self._logger.info(
                    "Primary source failure rate dropped to %.1f%% — "
                    "marking as healthy",
                    failure_rate * 100,
                )
                self._primary_healthy = True
                self._consecutive_fallback_days = 0

    async def _probe_primary_health(self) -> bool:
        """Probe whether the primary source (yfinance) is healthy.

        Downloads a small set of well-known liquid tickers and checks that
        at least HEALTH_PROBE_SUCCESS_THRESHOLD fraction succeed.

        Returns:
            True if the primary source appears healthy.
        """
        today = date.today()
        start = today - timedelta(days=2)  # 2-day lookback for probe

        self._logger.info(
            "Health probe: testing yfinance with %d tickers",
            len(self.health_probe_tickers),
        )
        try:
            df = await self.yfinance.download_ohlcv(
                self.health_probe_tickers, start, today
            )
        except Exception as e:
            self._logger.warning("Health probe failed with exception: %s", e)
            return False

        if df.height == 0:
            self._logger.warning("Health probe returned no data")
            return False

        succeeded = len(df["ticker"].unique())
        success_rate = succeeded / len(self.health_probe_tickers)
        healthy = success_rate >= HEALTH_PROBE_SUCCESS_THRESHOLD

        self._logger.info(
            "Health probe: %d/%d tickers succeeded (%.0f%%) — %s",
            succeeded,
            len(self.health_probe_tickers),
            success_rate * 100,
            "HEALTHY" if healthy else "UNHEALTHY",
        )
        return healthy

    def reset_primary_health(self) -> None:
        """Reset auto-recovery state (for testing)."""
        self._primary_healthy = True
        self._consecutive_fallback_days = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_ohlcv_df() -> pl.DataFrame:
    """Return an empty DataFrame with the standard OHLCV schema."""
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "dt": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
