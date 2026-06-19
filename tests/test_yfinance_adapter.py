"""Tests for yfinance data source adapter.

Issue #89: yfinance adapter.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from alphascreener.sources.yfinance_adapter import (
    BATCH_SIZE_MAX,
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_RPS,
    MAX_RETRIES,
    RETRY_WAIT_INIT_S,
    RETRY_WAIT_MAX_S,
    CircuitBreakerOpenError,
    YFinanceAdapter,
    _default_retry_policy,
    _earnings_to_polars,
    _info_to_dict,
    _insider_to_polars,
    _news_to_polars,
    _ohlcv_to_polars,
    _safe_float,
    _safe_int,
    _utc_today,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def adapter() -> YFinanceAdapter:
    """Return a fresh YFinanceAdapter with default settings."""
    return YFinanceAdapter()


@pytest.fixture
def adapter_fast() -> YFinanceAdapter:
    """Return a YFinanceAdapter configured for fast tests (small batch, low RPS)."""
    return YFinanceAdapter(batch_size=2, rps=2)


@pytest.fixture
def sample_tickers() -> list[str]:
    """Standard ticker list for tests."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]


@pytest.fixture
def single_ticker_ohlcv() -> pd.DataFrame:
    """Mimic yf.download output for a single ticker."""
    idx = pd.date_range("2025-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0, 152.0],
            "High": [152.0, 153.0, 154.0],
            "Low": [149.0, 150.0, 151.0],
            "Close": [151.5, 152.5, 153.5],
            "Volume": [100000, 120000, 110000],
        },
        index=idx,
    )


@pytest.fixture
def multi_ticker_ohlcv() -> pd.DataFrame:
    """Mimic yf.download output for two tickers (MultiIndex columns)."""
    idx = pd.date_range("2025-01-02", periods=2, freq="B")
    arrays = {
        "Open": {"AAPL": [150.0, 151.0], "GOOGL": [140.0, 141.5]},
        "High": {"AAPL": [152.0, 153.0], "GOOGL": [142.0, 143.0]},
        "Low": {"AAPL": [149.0, 150.0], "GOOGL": [139.0, 140.5]},
        "Close": {"AAPL": [151.5, 152.5], "GOOGL": [141.0, 142.0]},
        "Volume": {"AAPL": [100000, 120000], "GOOGL": [200000, 210000]},
    }
    cols = pd.MultiIndex.from_tuples(
        [(price, ticker) for price, tickers in arrays.items() for ticker in tickers]
    )
    data = {}
    for price, ticker in cols:
        data[(price, ticker)] = arrays[price][ticker]
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def sample_info() -> dict:
    """Typical Ticker.info dict."""
    return {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000,
        "forwardEps": 7.5,
        "trailingEps": 6.8,
        "pegRatio": 2.1,
        "dividendYield": 0.005,
        "beta": 1.2,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 120.0,
        "regularMarketTime": 1735689600,
    }


@pytest.fixture
def sample_earnings() -> pd.DataFrame:
    """Typical Ticker.earnings_dates DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-30"), pd.Timestamp("2025-04-30")])
    return pd.DataFrame(
        {
            "EPS Estimate": [1.5, 1.6],
            "Reported EPS": [1.6, None],
            "Surprise(%)": [6.7, None],
        },
        index=idx,
    )


@pytest.fixture
def sample_insider() -> pd.DataFrame:
    """Typical Ticker.insider_transactions DataFrame."""
    return pd.DataFrame(
        {
            "Insider": ["Tim Cook", "Jane Doe"],
            "Title": ["CEO", "CFO"],
            "Transaction": ["Buy", "Sell"],
            "Shares": [10000, -5000],
            "Value": [1500000, -750000],
            "Start Date": ["2025-01-15", "2025-01-20"],
        }
    )


@pytest.fixture
def sample_news() -> list[dict]:
    """Typical Ticker.news list."""
    return [
        {
            "title": "Apple Announces New Product",
            "link": "https://example.com/1",
            "publisher": "Bloomberg",
            "providerPublishTime": 1735689600,
            "type": "STORY",
        },
        {
            "title": "Apple Beats Earnings",
            "link": "https://example.com/2",
            "publisher": "Reuters",
            "providerPublishTime": 1735776000,
            "type": "STORY",
        },
    ]


# ============================================================================
# Constructor & defaults
# ============================================================================


class TestYFinanceAdapterInit:
    """Test YFinanceAdapter constructor and defaults."""

    def test_default_values(self, adapter):
        assert adapter.batch_size == BATCH_SIZE_MAX  # 50
        assert adapter.rps == DEFAULT_RPS  # 5
        assert adapter.max_retries == MAX_RETRIES  # 3
        assert adapter.retry_wait_init_s == RETRY_WAIT_INIT_S  # 2.0
        assert adapter.retry_wait_max_s == RETRY_WAIT_MAX_S  # 60.0

    def test_custom_values(self):
        a = YFinanceAdapter(
            batch_size=10, rps=3, max_retries=5, retry_wait_init_s=1.0, retry_wait_max_s=30.0
        )
        assert a.batch_size == 10
        assert a.rps == 3
        assert a.max_retries == 5
        assert a.retry_wait_init_s == 1.0
        assert a.retry_wait_max_s == 30.0

    def test_initial_state_empty(self, adapter):
        """Circuit breakers start empty."""
        assert adapter._failures == {}
        assert adapter._skip_until == {}
        assert adapter.open_circuits == {}


# ============================================================================
# Batch splitting
# ============================================================================


class TestSplitBatches:
    """Ticker list → batches of ≤ batch_size."""

    def test_equal_batches(self, adapter_fast):
        batches = adapter_fast._split_batches(["A", "B", "C", "D"])
        assert batches == [["A", "B"], ["C", "D"]]

    def test_partial_last_batch(self, adapter_fast):
        batches = adapter_fast._split_batches(["A", "B", "C"])
        assert batches == [["A", "B"], ["C"]]

    def test_single_ticker(self, adapter_fast):
        batches = adapter_fast._split_batches(["A"])
        assert batches == [["A"]]

    def test_empty_list(self, adapter_fast):
        batches = adapter_fast._split_batches([])
        assert batches == []

    def test_batch_size_default(self, adapter):
        """Default batch_size is 50, so 55 tickers → 2 batches."""
        tickers = [f"T{i}" for i in range(55)]
        batches = adapter._split_batches(tickers)
        assert len(batches) == 2
        assert len(batches[0]) == 50
        assert len(batches[1]) == 5


# ============================================================================
# Circuit breaker
# ============================================================================


class TestCircuitBreaker:
    """Circuit breaker: CIRCUIT_BREAKER_THRESHOLD consecutive failures → skip for the day."""

    def test_open_after_threshold_failures(self, adapter):
        today = _utc_today()
        ticker = "FAIL"
        assert not adapter._is_circuit_open(ticker, today)
        for i in range(CIRCUIT_BREAKER_THRESHOLD - 1):
            adapter._record_failure(ticker, today)
            assert not adapter._is_circuit_open(ticker, today), f"still closed at failure {i+1}"
        adapter._record_failure(ticker, today)
        assert adapter._is_circuit_open(ticker, today)

    def test_reset_on_success(self, adapter):
        today = _utc_today()
        ticker = "RECOVER"
        adapter._record_failure(ticker, today)
        adapter._record_failure(ticker, today)
        assert not adapter._is_circuit_open(ticker, today)
        adapter._record_success(ticker)
        # Failures reset
        assert adapter._failures.get(ticker, 0) == 0

    def test_ttl_expiry(self, adapter):
        today = _utc_today()
        ticker = "EXPIRE"
        # Force 3 failures
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            adapter._record_failure(ticker, today)
        assert adapter._is_circuit_open(ticker, today)
        # Check yesterday (before failures) → would be open if it was the same day
        # but since skip_until = today + 1, the check is based on the date passed
        later_today = today  # same day, still open
        assert adapter._is_circuit_open(ticker, later_today)
        # Check tomorrow → TTL is based on <, so tomorrow >= skip_until → closed
        tomorrow = today + timedelta(days=1)
        assert not adapter._is_circuit_open(ticker, tomorrow)
        # Check day after tomorrow → should be released
        day_after = today + timedelta(days=2)
        assert not adapter._is_circuit_open(ticker, day_after)

    def test_open_circuits_property(self, adapter):
        today = _utc_today()
        ticker = "PROP"
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            adapter._record_failure(ticker, today)
        circuits = adapter.open_circuits
        assert ticker in circuits
        assert circuits[ticker] == today + timedelta(days=1)

    def test_open_circuits_cleans_up_expired_entries(self, adapter):
        """open_circuits must clean up expired breakers before returning.

        Issue #121 — the property must call _is_circuit_open on each ticker
        (which internally cleans up expired entries) so callers never see
        stale breakers.
        """
        yesterday = _utc_today() - timedelta(days=1)
        # Manually inject an expired _skip_until entry
        adapter._skip_until["EXPIRED"] = yesterday
        # Also inject a valid (not yet expired) entry
        tomorrow = _utc_today() + timedelta(days=1)
        adapter._skip_until["ACTIVE"] = tomorrow

        circuits = adapter.open_circuits
        # Expired entry must be cleaned up
        assert "EXPIRED" not in circuits
        # Active entry must still be present
        assert "ACTIVE" in circuits
        assert circuits["ACTIVE"] == tomorrow

    def test_reset_all(self, adapter):
        today = _utc_today()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            adapter._record_failure("A", today)
            adapter._record_failure("B", today)
        assert len(adapter.open_circuits) == 2
        adapter.reset_circuit_breakers()
        assert adapter.open_circuits == {}
        assert adapter._failures == {}


# ============================================================================
# Polars conversion helpers
# ============================================================================


class TestOhlcvToPolars:
    """yf.download DataFrame → polars OHLCV tidy format."""

    def test_single_ticker(self, single_ticker_ohlcv):
        result = _ohlcv_to_polars(single_ticker_ohlcv, fallback_ticker="AAPL")
        assert result.height == 3
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}
        assert result["ticker"].to_list() == ["AAPL", "AAPL", "AAPL"]
        assert result["close"].to_list() == [151.5, 152.5, 153.5]

    def test_multi_ticker(self, multi_ticker_ohlcv):
        result = _ohlcv_to_polars(multi_ticker_ohlcv)
        assert result.height == 4  # 2 dates x 2 tickers
        tickers = set(result["ticker"].to_list())
        assert tickers == {"AAPL", "GOOGL"}
        # Check that dt is a date type
        assert result.schema["dt"] == pl.Date

    def test_empty_dataframe(self):
        empty = pd.DataFrame()
        result = _ohlcv_to_polars(empty)
        assert result.height == 0
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}


class TestInfoToDict:
    """Extract relevant fields from Ticker.info."""

    def test_standard_info(self, sample_info):
        result = _info_to_dict(sample_info)
        assert result["symbol"] == "AAPL"
        assert result["sector"] == "Technology"
        assert result["marketCap"] == 3000000000000
        assert result["forwardEps"] == 7.5
        # Unknown keys absent
        assert "unknown_field" not in result

    def test_empty_info(self):
        result = _info_to_dict({})
        assert result["symbol"] == ""
        assert result["sector"] is None
        assert result["marketCap"] is None


class TestEarningsToPolars:
    """Ticker.earnings_dates DataFrame → polars."""

    def test_standard_earnings(self, sample_earnings):
        result = _earnings_to_polars(sample_earnings, "AAPL")
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert result["eps_estimate"].to_list() == [1.5, 1.6]

    def test_none_earnings(self):
        result = _earnings_to_polars(pd.DataFrame(), "AAPL")
        assert result.height == 0


class TestInsiderToPolars:
    """Ticker.insider_transactions → polars."""

    def test_standard_insider(self, sample_insider):
        result = _insider_to_polars(sample_insider, "AAPL")
        assert result.height == 2
        assert result["insider_name"].to_list() == ["Tim Cook", "Jane Doe"]
        assert result["transaction_type"].to_list() == ["Buy", "Sell"]

    def test_none_insider(self):
        result = _insider_to_polars(pd.DataFrame(), "AAPL")
        assert result.height == 0


class TestNewsToPolars:
    """Ticker.news list → polars."""

    def test_standard_news(self, sample_news):
        result = _news_to_polars(sample_news, "AAPL")
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "AAPL"]
        assert "Bloomberg" in result["publisher"].to_list()

    def test_empty_news(self):
        result = _news_to_polars([], "AAPL")
        assert result.height == 0


# ============================================================================
# Retry policy
# ============================================================================


class TestRetryPolicy:
    """tenacity retry policy for transient errors."""

    def test_policy_is_callable(self):
        """_default_retry_policy returns a decorator that can wrap a function."""
        policy = _default_retry_policy()

        @policy
        def foo():
            pass

        # Function was decorated successfully
        assert callable(foo)

    def test_policy_retries_on_connection_error(self):
        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            raise ConnectionError("transient")

        with pytest.raises(ConnectionError):
            flaky()
        assert call_count[0] == MAX_RETRIES  # 3 attempts


# ============================================================================
# Async download tests (mocked yfinance)
# ============================================================================


@pytest.mark.asyncio
class TestDownloadOhlcvAsync:
    """Async OHLCV download with mocked yfinance."""

    async def test_download_single_ticker(self, adapter_fast, single_ticker_ohlcv, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda tickers, start, end: single_ticker_ohlcv)

        result = await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", "2025-01-05")
        assert result.height == 3
        assert "close" in result.columns

    async def test_download_multiple_batches(self, adapter_fast, single_ticker_ohlcv, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda tickers, start, end: single_ticker_ohlcv)

        result = await adapter_fast.download_ohlcv(["A", "B", "C"], "2025-01-01", "2025-01-05")
        # With individual fallback (Issue #224): when a ticker is missing from
        # a batch result, the adapter retries it individually.  The mock returns
        # plain-column data, so batch["A","B"] produces empty-ticker rows and
        # triggers individual fallback for both A and B (3 rows each), plus
        # batch["C"] succeeds normally (3 rows) = 3+3+3+3 = 12 rows total.
        assert result.height == 12  # 4 downloads x 3 rows each

    async def test_download_empty_tickers(self, adapter_fast):
        result = await adapter_fast.download_ohlcv([], "2025-01-01", "2025-01-05")
        assert result.height == 0
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}


@pytest.mark.asyncio
class TestDownloadFundamentalsAsync:
    """Async fundamentals download with mocked yfinance."""

    async def test_download_fundamentals(self, adapter_fast, sample_info, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_fetch_ticker_info", lambda t: sample_info)

        results = await adapter_fast.download_fundamentals(["AAPL"])
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["sector"] == "Technology"

    async def test_skips_failed_tickers(self, adapter_fast, sample_info, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        def flaky_info(ticker):
            if ticker == "BAD":
                raise RuntimeError("no data")
            return sample_info

        monkeypatch.setattr(mod, "_fetch_ticker_info", flaky_info)

        results = await adapter_fast.download_fundamentals(["AAPL", "BAD", "MSFT"])
        assert len(results) == 2  # BAD skipped


@pytest.mark.asyncio
class TestDownloadEarningsAsync:
    """Async earnings_dates download with mocked yfinance."""

    async def test_download_earnings(self, adapter_fast, sample_earnings, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_fetch_earnings_dates", lambda t: sample_earnings)

        result = await adapter_fast.download_earnings_dates(["AAPL"])
        assert result.height == 2
        assert result["eps_estimate"].to_list() == [1.5, 1.6]


@pytest.mark.asyncio
class TestDownloadInsiderAsync:
    """Async insider_transactions download with mocked yfinance."""

    async def test_download_insider(self, adapter_fast, sample_insider, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_fetch_insider_transactions", lambda t: sample_insider)

        result = await adapter_fast.download_insider_transactions(["AAPL"])
        assert result.height == 2
        assert result["transaction_type"].to_list() == ["Buy", "Sell"]


@pytest.mark.asyncio
class TestDownloadNewsAsync:
    """Async news download with mocked yfinance."""

    async def test_download_news(self, adapter_fast, sample_news, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_fetch_news", lambda t: sample_news)

        result = await adapter_fast.download_news(["AAPL"])
        assert result.height == 2
        assert "Bloomberg" in result["publisher"].to_list()


@pytest.mark.asyncio
class TestDownloadAllAsync:
    """Bulk download_all combines all data types."""

    async def test_download_all(
        self,
        adapter_fast,
        single_ticker_ohlcv,
        sample_info,
        sample_earnings,
        sample_insider,
        sample_news,
        monkeypatch,
    ):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda tickers, start, end: single_ticker_ohlcv)
        monkeypatch.setattr(mod, "_fetch_ticker_info", lambda t: sample_info)
        monkeypatch.setattr(mod, "_fetch_earnings_dates", lambda t: sample_earnings)
        monkeypatch.setattr(mod, "_fetch_insider_transactions", lambda t: sample_insider)
        monkeypatch.setattr(mod, "_fetch_news", lambda t: sample_news)

        result = await adapter_fast.download_all(["AAPL"], "2025-01-01", "2025-01-05")
        assert "ohlcv" in result
        assert "fundamentals" in result
        assert "earnings_dates" in result
        assert "insider_transactions" in result
        assert "news" in result
        assert result["ohlcv"].height == 3
        assert len(result["fundamentals"]) == 1


# ============================================================================
# Semaphore rate limiting
# ============================================================================


@pytest.mark.asyncio
class TestRateLimiting:
    """Verify that rate limiting slots are acquired and released."""

    async def test_semaphore_limits_concurrency(self, adapter):
        """Semaphore with rps=5 allows up to 5 concurrent acquires."""
        # Trigger semaphore creation via _acquire_slot
        await adapter._acquire_slot()
        sem = adapter._semaphore
        assert sem is not None

        # We already acquired 1 slot; 4 more should be available
        for _ in range(4):
            await sem.acquire()
        # All 5 slots taken → semaphore is locked
        assert sem.locked()

        # Release all slots
        for _ in range(5):
            sem.release()

    async def test_release_after_delay_schedules_release(self, adapter):
        """_release_after_delay schedules semaphore release after 1s."""
        # Acquire all 5 slots so semaphore is locked
        await adapter._acquire_slot()
        sem = adapter._semaphore
        for _ in range(4):
            await sem.acquire()
        assert sem.locked()  # all 5 slots taken

        adapter._release_after_delay()

        # Wait for the 1s delay + small buffer
        await asyncio.sleep(1.1)
        assert not sem.locked()  # one slot released


# ============================================================================
# CircuitBreakerOpenError
# ============================================================================


class TestCircuitBreakerOpenError:
    """Custom exception for circuit breaker."""

    def test_is_runtime_error(self):
        """CircuitBreakerOpenError is a RuntimeError subclass."""
        try:
            raise CircuitBreakerOpenError("test")
        except RuntimeError:
            pass  # Expected

    def test_message_preserved(self):
        with pytest.raises(CircuitBreakerOpenError, match="AAPL circuit open"):
            raise CircuitBreakerOpenError("AAPL circuit open")


# ============================================================================
# Date handling in download_ohlcv
# ============================================================================


@pytest.mark.asyncio
class TestDateHandling:
    """start_date / end_date accept both str and date objects."""

    async def test_date_objects_accepted(self, adapter_fast, single_ticker_ohlcv, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda tickers, start, end: single_ticker_ohlcv)

        result = await adapter_fast.download_ohlcv(
            ["AAPL"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 5),
        )
        assert result.height == 3

    async def test_end_date_defaults_to_today(self, adapter_fast, single_ticker_ohlcv, monkeypatch):
        import alphascreener.sources.yfinance_adapter as mod

        captured_end = []

        def capture(tickers, start, end):
            captured_end.append(end)
            return single_ticker_ohlcv

        monkeypatch.setattr(mod, "_download_batch", capture)

        await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01")
        assert len(captured_end) == 1
        assert captured_end[0] == (_utc_today() + timedelta(days=1)).isoformat()

    async def test_end_date_adjusted_for_inclusivity(
        self, adapter_fast, single_ticker_ohlcv, monkeypatch
    ):
        """end_date is documented as inclusive; yfinance treats it as exclusive.
        The adapter must add 1 day before passing to _download_batch."""
        import alphascreener.sources.yfinance_adapter as mod

        captured: list[tuple] = []

        def capture(tickers, start, end):
            captured.append((start, end))
            return single_ticker_ohlcv

        monkeypatch.setattr(mod, "_download_batch", capture)

        end = date(2025, 1, 5)
        await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", end)
        assert len(captured) == 1
        assert captured[0][0] == "2025-01-01"
        # yfinance expects exclusive end — so Jan 5 inclusive → Jan 6
        assert captured[0][1] == "2025-01-06"

        # Also test with string end_date
        captured.clear()
        await adapter_fast.download_ohlcv(["AAPL"], "2025-03-01", "2025-03-10")
        assert len(captured) == 1
        assert captured[0][0] == "2025-03-01"
        assert captured[0][1] == "2025-03-11"


# ============================================================================
# __post_init__ validation — Issue #120 round 2
# ============================================================================


class TestPostInitValidation:
    """Constructor must reject illegal parameter values with ValueError."""

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be 1"):
            YFinanceAdapter(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be 1"):
            YFinanceAdapter(batch_size=-5)

    def test_batch_size_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="batch_size must be 1"):
            YFinanceAdapter(batch_size=51)

    def test_rps_zero_raises(self):
        with pytest.raises(ValueError, match="rps must be >= 1"):
            YFinanceAdapter(rps=0)

    def test_rps_negative_raises(self):
        with pytest.raises(ValueError, match="rps must be >= 1"):
            YFinanceAdapter(rps=-1)

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            YFinanceAdapter(max_retries=0)

    def test_retry_wait_init_s_zero_raises(self):
        with pytest.raises(ValueError, match="retry_wait_init_s must be > 0"):
            YFinanceAdapter(retry_wait_init_s=0.0)

    def test_retry_wait_init_s_negative_raises(self):
        with pytest.raises(ValueError, match="retry_wait_init_s must be > 0"):
            YFinanceAdapter(retry_wait_init_s=-1.0)

    def test_retry_wait_max_s_less_than_init_raises(self):
        with pytest.raises(ValueError, match="retry_wait_max_s"):
            YFinanceAdapter(retry_wait_init_s=5.0, retry_wait_max_s=3.0)

    def test_valid_custom_values_accepted(self):
        """Ensure valid values do not raise."""
        a = YFinanceAdapter(
            batch_size=10,
            rps=3,
            max_retries=5,
            retry_wait_init_s=1.0,
            retry_wait_max_s=30.0,
        )
        assert a.batch_size == 10
        assert a.rps == 3

    def test_retry_wait_max_s_equals_init_accepted(self):
        """retry_wait_max_s == retry_wait_init_s is allowed."""
        a = YFinanceAdapter(retry_wait_init_s=5.0, retry_wait_max_s=5.0)
        assert a.retry_wait_max_s == 5.0


# ============================================================================
# Partial batch success — per-ticker tracking
# ============================================================================


@pytest.mark.asyncio
class TestPartialBatchTracking:
    """When a batch download succeeds but returns data for only some tickers,
    only those tickers get _record_success; the missing ones get _record_failure."""

    async def test_missing_ticker_gets_failure_not_success(self, adapter_fast, monkeypatch):
        """Batch with AAPL+GOOGL, but yfinance only returns AAPL data.

        With individual fallback (Issue #224), the missing ticker (GOOGL)
        gets retried individually. If the individual download also fails,
        it is recorded as a circuit-breaker failure.
        """
        import alphascreener.sources.yfinance_adapter as mod

        # Build a MultiIndex DataFrame that only has AAPL
        idx = pd.date_range("2025-01-02", periods=2, freq="B")
        arrays = [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ]
        data = {
            col: vals
            for col, vals in zip(
                arrays,
                [[150.0, 151.0], [152.0, 153.0], [149.0, 150.0], [151.5, 152.5], [100000, 120000]],
                strict=True,
            )
        }
        cols = pd.MultiIndex.from_tuples(arrays)
        aapl_only_df = pd.DataFrame(data, index=idx, columns=cols)

        # Mock: batch download returns only AAPL; individual GOOGL download fails
        def selective_download(tickers, start, end):
            if "GOOGL" in tickers and len(tickers) == 1:
                raise RuntimeError("no data for GOOGL")
            return aapl_only_df

        monkeypatch.setattr(mod, "_download_batch", selective_download)

        # AAPL should succeed, GOOGL should fail after individual fallback exhausted
        await adapter_fast.download_ohlcv(["AAPL", "GOOGL"], "2025-01-01", "2025-01-05")

        # AAPL: success recorded → no failure counter
        assert "AAPL" not in adapter_fast._failures
        # GOOGL: individual fallback exhausted → failure recorded
        assert adapter_fast._failures.get("GOOGL", 0) == 1

    async def test_all_tickers_present_all_succeed(
        self, adapter_fast, multi_ticker_ohlcv, monkeypatch
    ):
        """When all tickers are present in the result, all get _record_success."""
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda t, s, e: multi_ticker_ohlcv)

        # Pre-seed a failure to make sure success clears it
        adapter_fast._failures["AAPL"] = 2
        adapter_fast._failures["GOOGL"] = 1

        await adapter_fast.download_ohlcv(["AAPL", "GOOGL"], "2025-01-01", "2025-01-05")

        assert "AAPL" not in adapter_fast._failures
        assert "GOOGL" not in adapter_fast._failures

    async def test_no_synthetic_batch_key_in_circuit_state(
        self, adapter_fast, single_ticker_ohlcv, monkeypatch
    ):
        """Verify batch downloads don't leave synthetic keys in circuit state."""
        import alphascreener.sources.yfinance_adapter as mod

        monkeypatch.setattr(mod, "_download_batch", lambda t, s, e: single_ticker_ohlcv)

        await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", "2025-01-05")

        # No synthetic batch keys should appear
        for key in list(adapter_fast._failures.keys()):
            assert not key.startswith("batch:")
        for key in adapter_fast._skip_until:
            assert not key.startswith("batch:")

    async def test_retry_reacquires_semaphore_slot(self, adapter_fast, monkeypatch):
        """Each retry must go through _acquire_slot to respect RPS limits.

        Issue #121 — the retry policy is applied at the async level so that
        every tenacity retry calls _call_with_slot, which acquires a fresh
        semaphore slot.
        """
        acquire_count = 0
        call_count = 0

        async def counting_acquire(_self):
            nonlocal acquire_count
            acquire_count += 1
            if _self._semaphore is None:
                _self._semaphore = asyncio.Semaphore(_self.rps)
            await _self._semaphore.acquire()

        def counting_release(_self):
            loop = asyncio.get_running_loop()
            loop.call_later(0.05, _self._semaphore.release)

        monkeypatch.setattr(YFinanceAdapter, "_acquire_slot", counting_acquire)
        monkeypatch.setattr(YFinanceAdapter, "_release_after_delay", counting_release)

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return {"symbol": "AAPL"}

        result = await adapter_fast._rate_limited_call("AAPL", flaky_func, track_circuit=True)
        assert result == {"symbol": "AAPL"}
        assert call_count == 3  # 2 failures + 1 success
        assert acquire_count == 3  # each attempt acquired a slot


# ============================================================================
# _safe_float / _safe_int — Issue #121 NaN-safe conversion helpers
# ============================================================================


class TestSafeFloat:
    """_safe_float returns float(v) with NaN/None safety."""

    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(42) == 42.0

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=1.0) == 1.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float(float("nan"), default=-1.0) == -1.0

    def test_pandas_na_returns_default(self):
        assert _safe_float(pd.NA) == 0.0


class TestSafeInt:
    """_safe_int returns int(v) with NaN/None safety."""

    def test_valid_int(self):
        assert _safe_int(42) == 42
        assert _safe_int(3.14) == 3

    def test_none_returns_default(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, default=5) == 5

    def test_nan_returns_default(self):
        assert _safe_int(float("nan")) == 0
        assert _safe_int(float("nan"), default=-1) == -1

    def test_pandas_na_returns_default(self):
        assert _safe_int(pd.NA) == 0


# ============================================================================
# _utc_today — Issue #121 UTC date helper
# ============================================================================


class TestUtcToday:
    """_utc_today returns today's date in UTC, not local timezone."""

    def test_returns_date_object(self):
        result = _utc_today()
        assert isinstance(result, date)
        # Must not be a datetime (should be date only)
        assert not isinstance(result, datetime)

    def test_near_utc_now(self):
        """_utc_today should be within 1 day of actual UTC now."""
        utc_now = datetime.now(UTC).date()
        result = _utc_today()
        assert abs((result - utc_now).days) <= 1
