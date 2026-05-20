"""Tests for FMP data source adapter.

Issue #90: FMP adapter.
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import Mock, patch

import httpx
import polars as pl
import pytest

from alphascreener.sources.fmp_adapter import (
    FMP_BASE_URL,
    FMP_DAILY_BUDGET,
    FMP_DEFAULT_RPS,
    FMP_MAX_RETRIES,
    FMP_RETRY_WAIT_INIT_S,
    FMP_RETRY_WAIT_MAX_S,
    FmpAdapter,
    FmpBudgetExhaustedError,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def adapter() -> FmpAdapter:
    """Return a fresh FmpAdapter with default settings (test api key)."""
    return FmpAdapter(api_key="test_key")


@pytest.fixture
def adapter_fast() -> FmpAdapter:
    """Return a FmpAdapter configured for fast tests."""
    return FmpAdapter(api_key="test_key", rps=10, daily_budget=10)


@pytest.fixture
def sample_tickers() -> list[str]:
    """Standard ticker list for tests."""
    return ["AAPL", "GOOGL", "MSFT"]


# ============================================================================
# FMP API response fixtures (matching real FMP JSON shapes)
# ============================================================================


@pytest.fixture
def sample_analyst_estimates_json() -> list[dict]:
    """Typical FMP /v3/analyst-estimates response."""
    return [
        {
            "symbol": "AAPL",
            "date": "2025-01-15",
            "estimatedRevenueLow": 90000000000,
            "estimatedRevenueHigh": 95000000000,
            "estimatedRevenueAvg": 92500000000,
            "estimatedEbitdaLow": 30000000000,
            "estimatedEbitdaHigh": 32000000000,
            "estimatedEbitdaAvg": 31000000000,
            "estimatedEpsLow": 1.8,
            "estimatedEpsHigh": 2.2,
            "estimatedEpsAvg": 2.0,
        },
        {
            "symbol": "AAPL",
            "date": "2025-04-15",
            "estimatedRevenueLow": 85000000000,
            "estimatedRevenueHigh": 90000000000,
            "estimatedRevenueAvg": 87500000000,
            "estimatedEbitdaLow": 28000000000,
            "estimatedEbitdaHigh": 30000000000,
            "estimatedEbitdaAvg": 29000000000,
            "estimatedEpsLow": 1.5,
            "estimatedEpsHigh": 1.9,
            "estimatedEpsAvg": 1.7,
        },
    ]


@pytest.fixture
def sample_insider_trading_json() -> list[dict]:
    """Typical FMP /v4/insider-trading response."""
    return [
        {
            "symbol": "AAPL",
            "transactionDate": "2025-01-20",
            "reportingCik": "0001234567",
            "transactionType": "P-Purchase",
            "securitiesOwned": 50000,
            "securitiesTransacted": 10000,
            "price": 150.0,
            "securityName": "Common Stock",
            "reportingName": "Tim Cook",
            "relationship": "CEO",
        },
        {
            "symbol": "AAPL",
            "transactionDate": "2025-01-15",
            "reportingCik": "0007654321",
            "transactionType": "S-Sale",
            "securitiesOwned": 20000,
            "securitiesTransacted": 5000,
            "price": 152.0,
            "securityName": "Common Stock",
            "reportingName": "Jane Doe",
            "relationship": "CFO",
        },
    ]


@pytest.fixture
def sample_grade_json() -> list[dict]:
    """Typical FMP /v3/grade response."""
    return [
        {
            "symbol": "AAPL",
            "date": "2025-01-10",
            "gradingCompany": "JP Morgan",
            "previousGrade": "Overweight",
            "newGrade": "Overweight",
        },
        {
            "symbol": "AAPL",
            "date": "2025-01-05",
            "gradingCompany": "Goldman Sachs",
            "previousGrade": "Buy",
            "newGrade": "Buy",
        },
    ]


@pytest.fixture
def sample_stock_news_json() -> list[dict]:
    """Typical FMP /v3/stock_news response."""
    return [
        {
            "symbol": "AAPL",
            "publishedDate": "2025-01-20T10:30:00.000Z",
            "title": "Apple Announces New Product",
            "text": "Apple Inc. announced a new product today...",
            "url": "https://example.com/1",
            "site": "Bloomberg",
        },
        {
            "symbol": "AAPL",
            "publishedDate": "2025-01-19T14:00:00.000Z",
            "title": "Apple Beats Earnings",
            "text": "Apple reported better than expected earnings...",
            "url": "https://example.com/2",
            "site": "Reuters",
        },
    ]


@pytest.fixture
def sample_earning_calendar_json() -> list[dict]:
    """Typical FMP /v3/earning_calendar response."""
    return [
        {
            "symbol": "AAPL",
            "date": "2025-01-30",
            "fiscalDateEnding": "2024-12-31",
            "time": "after-market",
            "updatedFromDate": "2025-01-25",
        },
        {
            "symbol": "GOOGL",
            "date": "2025-02-05",
            "fiscalDateEnding": "2024-12-31",
            "time": "after-market",
            "updatedFromDate": "2025-01-30",
        },
    ]


@pytest.fixture
def sample_historical_earning_json() -> list[dict]:
    """Typical FMP /v3/historical-earning-calendar response."""
    return [
        {
            "symbol": "AAPL",
            "date": "2024-10-31",
            "eps": 1.64,
            "epsEstimated": 1.60,
            "revenue": 94900000000,
            "revenueEstimated": 94500000000,
            "fiscalDateEnding": "2024-09-30",
            "time": "after-market",
        },
    ]


# ============================================================================
# Constructor & defaults
# ============================================================================


class TestFmpAdapterInit:
    """Test FmpAdapter constructor, defaults, and validation."""

    def test_default_values(self, adapter):
        assert adapter.api_key == "test_key"
        assert adapter.daily_budget == FMP_DAILY_BUDGET  # 250
        assert adapter.rps == FMP_DEFAULT_RPS  # 2
        assert adapter.base_url == FMP_BASE_URL
        assert adapter.max_retries == FMP_MAX_RETRIES
        assert adapter.retry_wait_init_s == FMP_RETRY_WAIT_INIT_S
        assert adapter.retry_wait_max_s == FMP_RETRY_WAIT_MAX_S

    def test_custom_values(self):
        a = FmpAdapter(
            api_key="custom_key",
            daily_budget=100,
            rps=5,
            base_url="https://custom.api.com",
            max_retries=5,
            retry_wait_init_s=3.0,
            retry_wait_max_s=30.0,
        )
        assert a.api_key == "custom_key"
        assert a.daily_budget == 100
        assert a.rps == 5
        assert a.base_url == "https://custom.api.com/"
        assert a.max_retries == 5
        assert a.retry_wait_init_s == 3.0
        assert a.retry_wait_max_s == 30.0

    def test_initial_budget_state(self, adapter):
        """Daily request counter starts at 0."""
        assert adapter.requests_used_today == 0
        assert not adapter.is_budget_exhausted

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key must not be empty"):
            FmpAdapter(api_key="")

    def test_api_key_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="api_key must not be empty"):
            FmpAdapter(api_key="   ")

    def test_daily_budget_zero_raises(self):
        with pytest.raises(ValueError, match="daily_budget must be >= 1"):
            FmpAdapter(api_key="k", daily_budget=0)

    def test_daily_budget_negative_raises(self):
        with pytest.raises(ValueError, match="daily_budget must be >= 1"):
            FmpAdapter(api_key="k", daily_budget=-5)

    def test_rps_zero_raises(self):
        with pytest.raises(ValueError, match="rps must be >= 1"):
            FmpAdapter(api_key="k", rps=0)

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            FmpAdapter(api_key="k", max_retries=0)

    def test_retry_wait_init_s_zero_raises(self):
        with pytest.raises(ValueError, match="retry_wait_init_s must be > 0"):
            FmpAdapter(api_key="k", retry_wait_init_s=0.0)

    def test_retry_wait_max_s_less_than_init_raises(self):
        with pytest.raises(ValueError, match="retry_wait_max_s"):
            FmpAdapter(api_key="k", retry_wait_init_s=5.0, retry_wait_max_s=3.0)


# ============================================================================
# Budget control
# ============================================================================


class TestBudgetControl:
    """Daily budget tracking and exhaustion behavior."""

    async def test_requests_used_today_increments(self, adapter_fast):
        await adapter_fast._reserve_budget()
        assert adapter_fast.requests_used_today == 1
        await adapter_fast._reserve_budget()
        assert adapter_fast.requests_used_today == 2

    async def test_budget_exhausted_when_limit_reached(self, adapter_fast):
        """After using daily_budget requests, budget is exhausted."""
        for _ in range(adapter_fast.daily_budget):
            await adapter_fast._reserve_budget()
        assert adapter_fast.requests_used_today == 10
        assert adapter_fast.is_budget_exhausted

    async def test_budget_not_exhausted_below_limit(self, adapter_fast):
        for _ in range(adapter_fast.daily_budget - 1):
            await adapter_fast._reserve_budget()
        assert not adapter_fast.is_budget_exhausted

    async def test_budget_exhaustion_raises_error(self, adapter_fast):
        """_reserve_budget raises FmpBudgetExhaustedError when exhausted."""
        for _ in range(adapter_fast.daily_budget):
            await adapter_fast._reserve_budget()
        with pytest.raises(FmpBudgetExhaustedError, match="FMP daily budget exhausted"):
            await adapter_fast._reserve_budget()

    async def test_reset_budget(self, adapter_fast):
        """reset_budget clears the counter."""
        for _ in range(adapter_fast.daily_budget):
            await adapter_fast._reserve_budget()
        assert adapter_fast.is_budget_exhausted
        adapter_fast.reset_budget()
        assert adapter_fast.requests_used_today == 0
        assert not adapter_fast.is_budget_exhausted

    async def test_concurrent_budget_reserve(self, adapter_fast):
        """Concurrent _reserve_budget calls must not exceed budget atomically."""
        n = min(5, adapter_fast.daily_budget)

        async def reserve_one():
            await adapter_fast._reserve_budget()

        tasks = [reserve_one() for _ in range(n)]
        await asyncio.gather(*tasks)

        assert adapter_fast.requests_used_today == n


# ============================================================================
# Polars conversion helpers
# ============================================================================


class TestAnalystEstimatesToPolars:
    """FMP analyst-estimates JSON → polars DataFrame."""

    def test_standard_estimates(self, adapter, sample_analyst_estimates_json):
        result = adapter._analyst_estimates_to_polars(sample_analyst_estimates_json)
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert result["estimated_eps_avg"].to_list() == [2.0, 1.7]
        assert "estimated_revenue_avg" in result.columns

    def test_empty_estimates(self, adapter):
        result = adapter._analyst_estimates_to_polars([])
        assert result.height == 0


class TestInsiderTradingToPolars:
    """FMP insider-trading JSON → polars DataFrame."""

    def test_standard_insider(self, adapter, sample_insider_trading_json):
        result = adapter._insider_trading_to_polars(sample_insider_trading_json)
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert result["reporting_name"].to_list() == ["Tim Cook", "Jane Doe"]
        assert result["transaction_type"].to_list() == ["P-Purchase", "S-Sale"]

    def test_empty_insider(self, adapter):
        result = adapter._insider_trading_to_polars([])
        assert result.height == 0


class TestGradeToPolars:
    """FMP grade JSON → polars DataFrame."""

    def test_standard_grade(self, adapter, sample_grade_json):
        result = adapter._grade_to_polars(sample_grade_json)
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert "JP Morgan" in result["grading_company"].to_list()

    def test_empty_grade(self, adapter):
        result = adapter._grade_to_polars([])
        assert result.height == 0


class TestStockNewsToPolars:
    """FMP stock_news JSON → polars DataFrame."""

    def test_standard_news(self, adapter, sample_stock_news_json):
        result = adapter._stock_news_to_polars(sample_stock_news_json)
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert "Bloomberg" in result["site"].to_list()
        assert "published_at" in result.columns
        assert "url" in result.columns

    def test_empty_news(self, adapter):
        result = adapter._stock_news_to_polars([])
        assert result.height == 0


class TestEarningCalendarToPolars:
    """FMP earning_calendar JSON → polars DataFrame."""

    def test_standard_calendar(self, adapter, sample_earning_calendar_json):
        result = adapter._earning_calendar_to_polars(sample_earning_calendar_json)
        assert result.height == 2
        assert set(result["ticker"].to_list()) == {"AAPL", "GOOGL"}
        assert result["time"].to_list() == ["after-market", "after-market"]

    def test_empty_calendar(self, adapter):
        result = adapter._earning_calendar_to_polars([])
        assert result.height == 0


class TestHistoricalEarningToPolars:
    """FMP historical-earning-calendar JSON → polars DataFrame."""

    def test_standard_historical(self, adapter, sample_historical_earning_json):
        result = adapter._historical_earning_to_polars(sample_historical_earning_json)
        assert result.height == 1
        assert result["ticker"].to_list() == ["AAPL"]
        assert result["eps"].to_list() == [1.64]
        assert result["eps_estimated"].to_list() == [1.60]

    def test_empty_historical(self, adapter):
        result = adapter._historical_earning_to_polars([])
        assert result.height == 0


# ============================================================================
# FmpBudgetExhaustedError
# ============================================================================


class TestFmpBudgetExhaustedError:
    """Custom exception for FMP daily budget exhaustion."""

    def test_is_runtime_error(self):
        try:
            raise FmpBudgetExhaustedError("budget gone")
        except RuntimeError:
            pass

    def test_message_preserved(self):
        with pytest.raises(FmpBudgetExhaustedError, match="250/250"):
            raise FmpBudgetExhaustedError("FMP daily budget exhausted: 250/250 requests used")


# ============================================================================
# HTTP client
# ============================================================================


class TestHttpClient:
    """HTTP client creation and configuration."""

    @pytest.mark.asyncio
    async def test_client_has_api_key_header(self, adapter):
        client = adapter._create_client()
        try:
            assert client.headers.get("x-api-key") == "test_key"
            assert client.base_url == adapter.base_url
        finally:
            await client.aclose()

    def test_client_base_url_trailing_slash(self):
        """Base URL should always end with a single slash."""
        a = FmpAdapter(api_key="k", base_url="https://api.fmp.com/api")
        assert a.base_url == "https://api.fmp.com/api/"


# ============================================================================
# Async endpoint tests (mocked HTTP)
# ============================================================================


@pytest.mark.asyncio
class TestFetchAnalystEstimates:
    """Async analyst-estimates fetch with mocked HTTP."""

    async def test_fetch_single_ticker(self, adapter_fast, sample_analyst_estimates_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_analyst_estimates_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_analyst_estimates("AAPL")
            assert result.height == 2
            assert result["estimated_eps_avg"].to_list() == [2.0, 1.7]

    async def test_fetch_empty_result(self, adapter_fast, monkeypatch):
        """When FMP returns empty data and yfinance also returns empty."""
        import alphascreener.sources.yfinance_adapter as yf_mod

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()

        async def mock_yf_empty(self, tickers):
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "earnings_date": pl.Utf8,
                    "eps_estimate": pl.Float64,
                    "reported_eps": pl.Float64,
                    "surprise_pct": pl.Float64,
                }
            )

        monkeypatch.setattr(yf_mod.YFinanceAdapter, "download_earnings_dates", mock_yf_empty)

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_analyst_estimates("AAPL")
            assert result.height == 0

    async def test_budget_exhausted_returns_empty(self, adapter_fast):
        """When budget is exhausted, fetch returns empty DataFrame."""
        for _ in range(adapter_fast.daily_budget):
            await adapter_fast._reserve_budget()

        result = await adapter_fast.fetch_analyst_estimates("AAPL")
        assert result.height == 0


@pytest.mark.asyncio
class TestFetchInsiderTrading:
    """Async insider-trading fetch with mocked HTTP."""

    async def test_fetch_single_ticker(self, adapter_fast, sample_insider_trading_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_insider_trading_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_insider_trading("AAPL")
            assert result.height == 2
            assert result["reporting_name"].to_list() == ["Tim Cook", "Jane Doe"]


@pytest.mark.asyncio
class TestFetchGrade:
    """Async grade fetch with mocked HTTP."""

    async def test_fetch_single_ticker(self, adapter_fast, sample_grade_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_grade_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_grade("AAPL")
            assert result.height == 2
            assert "JP Morgan" in result["grading_company"].to_list()


@pytest.mark.asyncio
class TestFetchStockNews:
    """Async stock_news fetch with mocked HTTP."""

    async def test_fetch_with_limit(self, adapter_fast, sample_stock_news_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_stock_news_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_stock_news(["AAPL"], limit=10)
            assert result.height == 2
            assert "published_at" in result.columns


@pytest.mark.asyncio
class TestFetchEarningCalendar:
    """Async earning_calendar fetch with mocked HTTP."""

    async def test_fetch_calendar(self, adapter_fast, sample_earning_calendar_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_earning_calendar_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_earning_calendar("2025-01-01", "2025-02-01")
            assert result.height == 2
            assert set(result["ticker"].to_list()) == {"AAPL", "GOOGL"}


@pytest.mark.asyncio
class TestFetchHistoricalEarningCalendar:
    """Async historical-earning-calendar fetch with mocked HTTP."""

    async def test_fetch_single_ticker(self, adapter_fast, sample_historical_earning_json):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_historical_earning_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_historical_earning_calendar("AAPL")
            assert result.height == 1
            assert result["eps"].to_list() == [1.64]


# ============================================================================
# Budget tracking across requests
# ============================================================================


@pytest.mark.asyncio
class TestBudgetTrackingAcrossRequests:
    """Verify budget counter increments across multiple endpoint calls."""

    async def test_counter_increments_across_calls(
        self, adapter_fast, sample_analyst_estimates_json
    ):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_analyst_estimates_json
        mock_response.raise_for_status = Mock()

        async def mock_get(*args, **kwargs):
            await adapter_fast._reserve_budget()
            return mock_response

        with patch.object(adapter_fast, "_get", side_effect=mock_get):
            assert adapter_fast.requests_used_today == 0
            await adapter_fast.fetch_analyst_estimates("AAPL")
            assert adapter_fast.requests_used_today == 1
            await adapter_fast.fetch_analyst_estimates("GOOGL")
            assert adapter_fast.requests_used_today == 2

    async def test_budget_exhausted_halts_further_fetches(
        self, adapter_fast, sample_analyst_estimates_json
    ):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_analyst_estimates_json
        mock_response.raise_for_status = Mock()

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            # Use up budget completely
            for _ in range(adapter_fast.daily_budget):
                await adapter_fast._reserve_budget()
            assert adapter_fast.is_budget_exhausted

            result = await adapter_fast.fetch_analyst_estimates("AAPL")
            # Returns empty, counter does NOT increment past budget
            assert result.height == 0
            assert adapter_fast.requests_used_today == adapter_fast.daily_budget


# ============================================================================
# Semaphore rate limiting
# ============================================================================


@pytest.mark.asyncio
class TestRateLimiting:
    """Verify that FMP rate limiting slots are acquired and released."""

    async def test_semaphore_limits_concurrency(self, adapter):
        # Trigger semaphore creation via _acquire_slot
        await adapter._acquire_slot()
        sem = adapter._semaphore
        assert sem is not None

        # Acquire remaining slots
        for _ in range(adapter.rps - 1):
            await sem.acquire()
        assert sem.locked()

        # Release all
        for _ in range(adapter.rps):
            sem.release()

    async def test_release_after_delay_releases(self, adapter):
        await adapter._acquire_slot()
        sem = adapter._semaphore
        for _ in range(adapter.rps - 1):
            await sem.acquire()
        assert sem.locked()

        adapter._release_after_delay()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (1 / adapter.rps) + 0.5
        while sem.locked() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert not sem.locked()


# ============================================================================
# Retry policy
# ============================================================================


class TestRetryPolicy:
    """tenacity retry policy for FMP transient errors."""

    def test_policy_retries_on_httpx_error(self):
        from alphascreener.sources.fmp_adapter import _default_retry_policy

        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            raise httpx.ConnectError("transient")

        with pytest.raises(httpx.ConnectError):
            flaky()
        assert call_count[0] == FMP_MAX_RETRIES


# ============================================================================
# YFinance fallback integration
# ============================================================================


@pytest.mark.asyncio
class TestYFinanceFallback:
    """When FMP is unavailable, fallback to yfinance should work."""

    async def test_fmp_unavailable_fallback_to_yfinance(self, adapter_fast, monkeypatch):
        """When FMP HTTP calls fail, fallback to yfinance for earnings."""
        import alphascreener.sources.yfinance_adapter as yf_mod

        # Make FMP _get always fail
        async def failing_get(*args, **kwargs):
            raise httpx.ConnectError("FMP down")

        monkeypatch.setattr(adapter_fast, "_get", failing_get)

        # Mock yfinance earnings fetch
        sample_earnings_df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-01-30"],
                "eps_estimate": [1.5],
                "reported_eps": [1.6],
                "surprise_pct": [6.7],
            }
        )
        call_args = []

        async def mock_yf_earnings(self, tickers):
            call_args.append(tickers)
            return sample_earnings_df

        monkeypatch.setattr(yf_mod.YFinanceAdapter, "download_earnings_dates", mock_yf_earnings)

        result = await adapter_fast.fetch_analyst_estimates("AAPL")
        # Fallback was exercised
        assert len(call_args) == 1
        assert call_args[0] == ["AAPL"]
        # Result uses FMP schema after conversion
        assert result.height == 1
        assert result["ticker"].to_list() == ["AAPL"]
        assert result["date"].to_list() == ["2025-01-30"]
        assert result["estimated_eps_avg"].to_list() == [1.5]

    async def test_fmp_returns_empty_uses_yfinance_fallback(self, adapter_fast, monkeypatch):
        """When FMP returns empty results, fallback to yfinance."""
        import alphascreener.sources.yfinance_adapter as yf_mod

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()

        # Mock yfinance fallback
        sample_earnings_df = pl.DataFrame(
            {
                "ticker": ["GOOGL"],
                "earnings_date": ["2025-02-05"],
                "eps_estimate": [2.1],
                "reported_eps": [2.3],
                "surprise_pct": [9.5],
            }
        )
        call_args = []

        async def mock_yf_earnings(self, tickers):
            call_args.append(tickers)
            return sample_earnings_df

        monkeypatch.setattr(yf_mod.YFinanceAdapter, "download_earnings_dates", mock_yf_earnings)

        with patch.object(adapter_fast, "_get", return_value=mock_response):
            result = await adapter_fast.fetch_analyst_estimates("AAPL")
            # Fallback was exercised
            assert len(call_args) == 1
            assert call_args[0] == ["AAPL"]
            # Result uses FMP schema after conversion
            assert result.height == 1
            assert result["ticker"].to_list() == ["GOOGL"]
            assert result["date"].to_list() == ["2025-02-05"]
            assert result["estimated_eps_avg"].to_list() == [2.1]


# ============================================================================
# Date normalization
# ============================================================================


class TestDateNormalization:
    """FMP dates should be normalized to standard formats."""

    def test_parse_fmp_date(self, adapter):
        """FMP dates like '2025-01-20' parse correctly."""
        result = adapter._parse_fmp_date("2025-01-20")
        assert isinstance(result, date)
        assert result == date(2025, 1, 20)

    def test_parse_fmp_datetime(self, adapter):
        """FMP datetime strings like '2025-01-20T10:30:00.000Z' parse correctly."""
        result = adapter._parse_fmp_datetime("2025-01-20T10:30:00.000Z")
        assert isinstance(result, str)
        assert result.startswith("2025-01-20")
        assert "10:30:00" in result
