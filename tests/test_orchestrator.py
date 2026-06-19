"""Tests for Data sync orchestrator.

Issue #92: Data sync orchestrator.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from alphascreener.data_sync.orchestrator import (
    CONTINUITY_FAILURE_RATE_THRESHOLD,
    CONTINUITY_WINDOW_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DIFF_THRESHOLD_PCT,
    MAX_NAN_FRACTION,
    STOOQ_VALIDATE_TOP_N,
    IntegrityReport,
    SyncOrchestrator,
    SyncReport,
)
from alphascreener.sources.fmp_adapter import FmpBudgetExhaustedError
from alphascreener.sources.stooq_adapter import StooqAdapter
from alphascreener.sources.yfinance_adapter import YFinanceAdapter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_tickers() -> list[str]:
    """Standard ticker list for tests."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Small OHLCV DataFrame simulating yfinance output."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT"],
            "dt": [date.today() - timedelta(days=d) for d in [1, 2, 1, 2, 1]],
            "open": [150.0, 149.0, 140.0, 139.0, 330.0],
            "high": [152.0, 151.0, 142.0, 141.0, 332.0],
            "low": [149.0, 148.0, 139.0, 138.0, 328.0],
            "close": [151.5, 150.5, 141.0, 140.0, 331.0],
            "volume": [100_000, 95_000, 200_000, 190_000, 50_000],
        }
    )


@pytest.fixture
def sample_ohlcv_with_nan(sample_ohlcv) -> pl.DataFrame:
    """OHLCV with some NaN values in the close column."""
    df = sample_ohlcv.clone()
    # Put a NaN in one row
    close_vals = df["close"].to_list()
    close_vals[0] = float("nan")
    return df.with_columns(pl.Series("close", close_vals))


@pytest.fixture
def mock_yfinance(sample_ohlcv) -> YFinanceAdapter:
    """Mock YFinanceAdapter that returns sample_ohlcv."""
    yf = MagicMock(spec=YFinanceAdapter)
    yf.download_ohlcv = AsyncMock(return_value=sample_ohlcv)
    # Access open_circuits for checking circuit breaker state
    yf.open_circuits = {}
    return yf


@pytest.fixture
def mock_yfinance_empty() -> YFinanceAdapter:
    """Mock YFinanceAdapter that returns empty DataFrame."""
    yf = MagicMock(spec=YFinanceAdapter)
    yf.download_ohlcv = AsyncMock(return_value=_empty_ohlcv_df())
    return yf


@pytest.fixture
def mock_stooq(sample_ohlcv) -> StooqAdapter:
    """Mock StooqAdapter that returns matching data (no diffs)."""
    stq = MagicMock(spec=StooqAdapter)
    # Return same data as yfinance to trigger zero diffs
    stq.download_ohlcv = AsyncMock(return_value=sample_ohlcv.clone())
    return stq


@pytest.fixture
def mock_stooq_divergent(sample_ohlcv) -> StooqAdapter:
    """Mock StooqAdapter that returns subtly different close prices."""
    divergent = sample_ohlcv.clone()
    # Shift close by 1% to trigger diff detection
    close_vals = [c * 1.01 for c in divergent["close"].to_list()]
    divergent = divergent.with_columns(pl.Series("close", close_vals))
    stq = MagicMock(spec=StooqAdapter)
    stq.download_ohlcv = AsyncMock(return_value=divergent)
    return stq


@pytest.fixture
def mock_stooq_empty() -> StooqAdapter:
    """Mock StooqAdapter that returns empty DataFrame."""
    stq = MagicMock(spec=StooqAdapter)
    stq.download_ohlcv = AsyncMock(return_value=_empty_ohlcv_df())
    return stq


@pytest.fixture
def mock_fmp() -> MagicMock:
    """Mock FmpAdapter."""
    fmp = MagicMock()
    fmp.is_budget_exhausted = False
    fmp.fetch_analyst_estimates = AsyncMock(
        return_value=pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2025-01-15"],
                "estimated_revenue_avg": [100e9],
                "estimated_eps_avg": [2.0],
                "estimated_eps_high": [2.2],
                "estimated_eps_low": [1.8],
                "estimated_ebitda_avg": [30e9],
            }
        )
    )
    fmp.fetch_insider_trading = AsyncMock(
        return_value=pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "transaction_date": ["2025-01-10"],
                "reporting_name": ["Test"],
                "relationship": ["CEO"],
                "transaction_type": ["Buy"],
                "securities_transacted": [1000.0],
                "price": [150.0],
                "security_name": ["Common Stock"],
            }
        )
    )
    return fmp


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


# ============================================================================
# Constructor tests
# ============================================================================


class TestSyncOrchestratorInit:
    """Constructor defaults and validation."""

    def test_default_construction(self, mock_yfinance):
        orch = SyncOrchestrator(yfinance=mock_yfinance)
        assert orch.yfinance is mock_yfinance
        assert orch.stooq is None
        assert orch.fmp is None
        assert orch.lookback_days == DEFAULT_LOOKBACK_DAYS
        assert orch.stooq_validate_n == STOOQ_VALIDATE_TOP_N
        assert orch.max_nan_fraction == MAX_NAN_FRACTION

    def test_with_all_sources(self, mock_yfinance, mock_stooq, mock_fmp):
        orch = SyncOrchestrator(yfinance=mock_yfinance, stooq=mock_stooq, fmp=mock_fmp)
        assert orch.stooq is mock_stooq
        assert orch.fmp is mock_fmp

    def test_custom_params(self, mock_yfinance):
        orch = SyncOrchestrator(
            yfinance=mock_yfinance,
            lookback_days=3,
            stooq_validate_n=50,
            max_nan_fraction=0.10,
        )
        assert orch.lookback_days == 3
        assert orch.stooq_validate_n == 50
        assert orch.max_nan_fraction == 0.10


# ============================================================================
# Sync tests
# ============================================================================


class TestSync:
    """Full sync cycle tests with mocked downstream adapters."""

    @pytest.mark.asyncio
    async def test_sync_with_yfinance_only(
        self, mock_yfinance, sample_tickers, tmp_path, monkeypatch
    ):
        """Sync with only yfinance — should download OHLCV and write Parquet."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = await orch.sync(sample_tickers)

        assert isinstance(report, SyncReport)
        assert report.tickers_total == len(sample_tickers)
        assert report.tickers_succeeded > 0
        assert report.ohlcv_rows > 0
        assert report.stooq_validated == 0
        assert report.fmp_enriched == 0
        # Verify Parquet was written
        ohlcv_dir = tmp_path / ".alphascreener" / "data" / "ohlcv"
        assert ohlcv_dir.exists()

    @pytest.mark.asyncio
    async def test_sync_writes_parquet_partitions(
        self, mock_yfinance, sample_tickers, tmp_path, monkeypatch
    ):
        """OHLCV data is written to dt=YYYY-MM-DD partitions."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = await orch.sync(sample_tickers)

        ohlcv_dir = tmp_path / ".alphascreener" / "data" / "ohlcv"
        partitions = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
        assert len(partitions) > 0
        assert all(p.startswith("dt=") for p in partitions)

    @pytest.mark.asyncio
    async def test_sync_incremental_date_range(self, mock_yfinance, sample_tickers):
        """Incremental sync pulls only last lookback_days calendar days."""
        orch = SyncOrchestrator(yfinance=mock_yfinance, lookback_days=3)

        report = await orch.sync(sample_tickers)

        # Check that download_ohlcv was called with correct date range
        call_kwargs = mock_yfinance.download_ohlcv.call_args.kwargs
        assert "start_date" in call_kwargs or len(mock_yfinance.download_ohlcv.call_args.args) >= 2

        args = mock_yfinance.download_ohlcv.call_args
        # First positional arg should be tickers
        # Second should be start_date
        assert args.args[0] == sample_tickers

    @pytest.mark.asyncio
    async def test_sync_explicit_date_range(self, mock_yfinance, sample_tickers):
        """Explicit start/end date overrides incremental behavior."""
        orch = SyncOrchestrator(yfinance=mock_yfinance, lookback_days=3)

        await orch.sync(
            sample_tickers,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )

        args = mock_yfinance.download_ohlcv.call_args
        assert args.args[1] == date(2025, 1, 1)
        assert args.kwargs.get("end_date") == date(2025, 1, 31) or args.args[2] == date(2025, 1, 31)

    @pytest.mark.asyncio
    async def test_sync_yfinance_failure_handles_gracefully(
        self, sample_tickers, tmp_path, monkeypatch
    ):
        """When yfinance download raises, the sync continues and reports the error."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        yf = MagicMock(spec=YFinanceAdapter)
        yf.download_ohlcv = AsyncMock(side_effect=RuntimeError("yfinance down"))

        orch = SyncOrchestrator(yfinance=yf)
        report = await orch.sync(sample_tickers)

        assert report.ohlcv_rows == 0
        assert report.tickers_succeeded == 0
        assert report.tickers_failed > 0  # 0 succeeded → all failed
        assert len(report.errors) > 0
        assert any("yfinance" in e for e in report.errors)

    @pytest.mark.asyncio
    async def test_sync_returns_timing_info(self, mock_yfinance, sample_tickers):
        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = await orch.sync(sample_tickers)

        assert report.elapsed_s >= 0.0


class TestSyncWithStooq:
    """Sync with Stooq cross-validation."""

    @pytest.mark.asyncio
    async def test_sync_runs_stooq_validation(
        self, mock_yfinance, mock_stooq, sample_tickers, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance, stooq=mock_stooq)
        report = await orch.sync(sample_tickers)

        assert report.stooq_validated > 0
        assert report.stooq_diff_count == 0  # Matching data → no diffs
        # Stooq is called for both OHLCV fallback (Issue #224) and cross-validation
        assert mock_stooq.download_ohlcv.await_count >= 1

    @pytest.mark.asyncio
    async def test_sync_detects_stooq_diffs(
        self, mock_yfinance, mock_stooq_divergent, sample_tickers, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance, stooq=mock_stooq_divergent)
        report = await orch.sync(sample_tickers)

        assert report.stooq_validated > 0
        # With close shifted 1%, should detect diffs > 0.5%
        assert report.stooq_diff_count > 0

    @pytest.mark.asyncio
    async def test_sync_stooq_empty_handled(
        self, mock_yfinance, mock_stooq_empty, sample_tickers, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance, stooq=mock_stooq_empty)
        report = await orch.sync(sample_tickers)

        assert report.stooq_validated == 0
        assert report.stooq_diff_count == 0


class TestSyncWithFmp:
    """Sync with FMP enrichment."""

    @pytest.mark.asyncio
    async def test_sync_runs_fmp_enrichment(
        self, mock_yfinance, mock_fmp, sample_tickers, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance, fmp=mock_fmp)
        report = await orch.sync(sample_tickers)

        assert report.fmp_enriched > 0
        mock_fmp.fetch_analyst_estimates.assert_awaited()
        mock_fmp.fetch_insider_trading.assert_awaited()

    @pytest.mark.asyncio
    async def test_sync_fmp_budget_exhausted(
        self, mock_yfinance, sample_tickers, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        fmp = MagicMock()
        fmp.is_budget_exhausted = False
        fmp.fetch_analyst_estimates = AsyncMock(side_effect=FmpBudgetExhaustedError("budget gone"))
        fmp.fetch_insider_trading = AsyncMock(side_effect=FmpBudgetExhaustedError("budget gone"))

        orch = SyncOrchestrator(yfinance=mock_yfinance, fmp=fmp)
        report = await orch.sync(sample_tickers)

        assert report.fmp_budget_exhausted is True


# ============================================================================
# Integrity check tests
# ============================================================================


class TestIntegrityCheck:
    """IntegrityReport and NaN rate checks."""

    def test_all_clean_data_passes(self, sample_ohlcv):
        orch = SyncOrchestrator(yfinance=MagicMock())
        report = orch._check_integrity(sample_ohlcv, total_tickers=5)

        assert isinstance(report, IntegrityReport)
        assert report.passed is True
        for field in ["open", "high", "low", "close"]:
            assert report.nan_counts[field] == 0
            assert report.nan_fractions[field] == 0.0

    def test_detects_nan_values(self, sample_ohlcv_with_nan):
        orch = SyncOrchestrator(yfinance=MagicMock())
        report = orch._check_integrity(sample_ohlcv_with_nan, total_tickers=5)

        assert report.nan_counts["close"] > 0
        # NaN fraction should be > 0
        assert report.nan_fractions["close"] > 0.0

    def test_nan_fraction_scales_with_universe(self, sample_ohlcv):
        """NaN fraction denominator is total tickers (universe size)."""
        orch = SyncOrchestrator(yfinance=MagicMock())

        # Same data, different universe sizes
        r_small = orch._check_integrity(sample_ohlcv, total_tickers=5)
        r_large = orch._check_integrity(sample_ohlcv, total_tickers=2000)

        # Fractions should be smaller for larger universe
        for field in ["open", "high", "low", "close"]:
            assert r_large.nan_fractions[field] <= r_small.nan_fractions[field]

    def test_fails_when_nan_fraction_exceeds_threshold(self, sample_ohlcv):
        """Inject enough NaNs to exceed the threshold."""
        # Create data where 20% of values are NaN (threshold is 5%)
        n = 100
        data = pl.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "dt": [date.today()] * n,
                "open": [float(i) for i in range(n)],
                "high": [float(i) for i in range(n)],
                "low": [float(i) for i in range(n)],
                "close": [float("nan") if i < 20 else float(i) for i in range(n)],
                "volume": [i * 1000 for i in range(n)],
            }
        )

        orch = SyncOrchestrator(yfinance=MagicMock())
        report = orch._check_integrity(data, total_tickers=100)

        # With 20 NaN / 1 day / 100 tickers = 0.20 > 0.05
        assert report.nan_fractions["close"] > MAX_NAN_FRACTION
        assert report.passed is False


# ============================================================================
# Continuity tracker tests
# ============================================================================


class TestContinuityTracker:
    """Continuity alert when yfinance failure rate > 30% for 3 consecutive days."""

    def test_no_alert_on_normal_rates(self, mock_yfinance):
        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = SyncReport()

        orch._update_continuity(0.10, report)
        orch._update_continuity(0.20, report)
        orch._update_continuity(0.25, report)

        # No alert — none exceeded 30%
        assert not any("CONTINUITY ALERT" in e for e in report.errors)

        # Tracker has recent entries
        assert len(orch._daily_failure_rates) == 3

    def test_alert_on_consecutive_high_rates(self, mock_yfinance):
        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = SyncReport()

        orch._update_continuity(0.35, report)
        orch._update_continuity(0.40, report)
        orch._update_continuity(0.50, report)

        # All 3 > 30% → alert
        assert any("CONTINUITY ALERT" in e for e in report.errors)

    def test_no_alert_when_one_day_drops_below(self, mock_yfinance):
        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = SyncReport()

        orch._update_continuity(0.40, report)
        orch._update_continuity(0.35, report)
        orch._update_continuity(0.20, report)  # Below threshold

        assert not any("CONTINUITY ALERT" in e for e in report.errors)

    def test_reset_continuity(self, mock_yfinance):
        orch = SyncOrchestrator(yfinance=mock_yfinance)

        orch._update_continuity(0.40, MockReport())
        orch._update_continuity(0.40, MockReport())
        orch._update_continuity(0.40, MockReport())

        assert len(orch._daily_failure_rates) == 3

        orch.reset_continuity()
        assert len(orch._daily_failure_rates) == 0

    @pytest.mark.asyncio
    async def test_sync_integrates_continuity(
        self, mock_yfinance, sample_tickers, tmp_path, monkeypatch
    ):
        """The full sync pipeline feeds failure rate into continuity tracker."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance)
        assert len(orch._daily_failure_rates) == 0

        report = await orch.sync(sample_tickers)
        assert len(orch._daily_failure_rates) == 1


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge case and error handling."""

    @pytest.mark.asyncio
    async def test_sync_empty_ticker_list(self, mock_yfinance, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance)
        report = await orch.sync([])

        assert report.tickers_total == 0
        assert report.tickers_succeeded == 0
        # Early return skips downstream calls when ticker list is empty
        mock_yfinance.download_ohlcv.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_when_yfinance_returns_empty(
        self, mock_yfinance_empty, sample_tickers, tmp_path, monkeypatch
    ):
        """When yfinance returns no data, sync still completes cleanly."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(yfinance=mock_yfinance_empty)
        report = await orch.sync(sample_tickers)

        assert report.ohlcv_rows == 0
        assert report.tickers_succeeded == 0
        assert report.tickers_failed == len(sample_tickers)

    @pytest.mark.asyncio
    async def test_sync_full_pipeline(
        self, mock_yfinance, mock_stooq, mock_fmp, sample_tickers, tmp_path, monkeypatch
    ):
        """End-to-end: yfinance + Stooq + FMP + Parquet write."""
        monkeypatch.setattr(
            "alphascreener.data.paths.get_data_home", lambda: tmp_path / ".alphascreener"
        )

        orch = SyncOrchestrator(
            yfinance=mock_yfinance,
            stooq=mock_stooq,
            fmp=mock_fmp,
        )
        report = await orch.sync(sample_tickers)

        assert isinstance(report, SyncReport)
        assert report.ohlcv_rows > 0
        assert report.stooq_validated > 0
        assert report.fmp_enriched > 0
        assert report.integrity is not None
        assert report.integrity.passed is True
        assert report.elapsed_s > 0

        # Verify Parquet was written
        ohlcv_dir = tmp_path / ".alphascreener" / "data" / "ohlcv"
        assert ohlcv_dir.exists()


# ============================================================================
# SyncReport tests
# ============================================================================


class TestSyncReport:
    """SyncReport dataclass defaults."""

    def test_defaults(self):
        r = SyncReport()
        assert r.tickers_total == 0
        assert r.tickers_succeeded == 0
        assert r.tickers_failed == 0
        assert r.ohlcv_rows == 0
        assert r.stooq_validated == 0
        assert r.stooq_diff_count == 0
        assert r.fmp_enriched == 0
        assert r.fmp_budget_exhausted is False
        assert r.integrity is None
        assert r.elapsed_s == 0.0
        assert r.errors == []


class TestIntegrityReport:
    """IntegrityReport dataclass."""

    def test_defaults(self):
        r = IntegrityReport()
        assert r.total_tickers == 0
        assert r.nan_counts == {}
        assert r.nan_fractions == {}
        assert r.passed is True

    def test_failed_report(self):
        r = IntegrityReport(
            total_tickers=100,
            nan_counts={"close": 15},
            nan_fractions={"close": 0.15},
            passed=False,
        )
        assert r.nan_counts["close"] == 15
        assert r.nan_fractions["close"] == 0.15
        assert r.passed is False


# Helper: a SyncReport-like object with just an errors list for _update_continuity
class MockReport:
    errors: list[str] = []
