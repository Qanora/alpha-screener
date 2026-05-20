"""Tests for Stooq OHLCV data source adapter.

Issue #91: Stooq fallback adapter + cross-validation.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import polars as pl
import pytest

from alphascreener.sources.stooq_adapter import (
    STOOQ_BASE_URL,
    STOOQ_DEFAULT_RPS,
    STOOQ_MAX_RETRIES,
    STOOQ_RETRY_WAIT_INIT_S,
    STOOQ_RETRY_WAIT_MAX_S,
    StooqAdapter,
    _default_retry_policy,
    _is_retryable_http_status,
    _parse_stooq_csv,
    _safe_float,
    _safe_int,
    _utc_today,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def adapter() -> StooqAdapter:
    """Return a fresh StooqAdapter with default settings."""
    return StooqAdapter()


@pytest.fixture
def adapter_fast() -> StooqAdapter:
    """Return a StooqAdapter configured for fast tests."""
    return StooqAdapter(rps=5)


@pytest.fixture
def sample_tickers() -> list[str]:
    """Standard ticker list for tests."""
    return ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture
def sample_csv() -> str:
    """Typical Stooq CSV response for a single ticker."""
    return (
        "Date,Open,High,Low,Close,Volume\r\n"
        "2025-01-02,150.0,152.0,149.0,151.5,100000\r\n"
        "2025-01-03,151.5,153.5,150.5,153.0,110000\r\n"
        "2025-01-06,153.0,155.0,152.5,154.5,120000\r\n"
    )


@pytest.fixture
def empty_csv() -> str:
    """Stooq response with only header (no data)."""
    return "Date,Open,High,Low,Close,Volume\r\n"


@pytest.fixture
def partial_csv() -> str:
    """Stooq response with some rows containing empty or invalid values."""
    return (
        "Date,Open,High,Low,Close,Volume\r\n"
        "2025-01-02,150.0,152.0,149.0,151.5,100000\r\n"
        "2025-01-03,N/A,N/A,N/A,N/A,N/A\r\n"
        "2025-01-06,153.0,155.0,,154.5,\r\n"
    )


# ============================================================================
# Constructor & defaults
# ============================================================================


class TestStooqAdapterInit:
    """Test StooqAdapter constructor and defaults."""

    def test_default_values(self, adapter):
        assert adapter.base_url == STOOQ_BASE_URL
        assert adapter.rps == STOOQ_DEFAULT_RPS  # 2
        assert adapter.max_retries == STOOQ_MAX_RETRIES  # 3
        assert adapter.retry_wait_init_s == STOOQ_RETRY_WAIT_INIT_S  # 2.0
        assert adapter.retry_wait_max_s == STOOQ_RETRY_WAIT_MAX_S  # 30.0

    def test_custom_values(self):
        a = StooqAdapter(
            base_url="https://example.com/",
            rps=3,
            max_retries=5,
            retry_wait_init_s=1.0,
            retry_wait_max_s=15.0,
        )
        assert a.base_url == "https://example.com/"
        assert a.rps == 3
        assert a.max_retries == 5
        assert a.retry_wait_init_s == 1.0
        assert a.retry_wait_max_s == 15.0

    def test_default_adapter_has_no_client_initially(self, adapter):
        """_client should be None until first HTTP call."""
        assert adapter._client is None


# ============================================================================
# __post_init__ validation
# ============================================================================


class TestPostInitValidation:
    """Constructor parameter validation."""

    def test_rps_zero_raises(self):
        with pytest.raises(ValueError, match="rps must be >= 1"):
            StooqAdapter(rps=0)

    def test_rps_negative_raises(self):
        with pytest.raises(ValueError, match="rps must be >= 1"):
            StooqAdapter(rps=-1)

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            StooqAdapter(max_retries=0)

    def test_retry_wait_init_s_zero_raises(self):
        with pytest.raises(ValueError, match="retry_wait_init_s must be > 0"):
            StooqAdapter(retry_wait_init_s=0.0)

    def test_retry_wait_init_s_negative_raises(self):
        with pytest.raises(ValueError, match="retry_wait_init_s must be > 0"):
            StooqAdapter(retry_wait_init_s=-1.0)

    def test_retry_wait_max_s_less_than_init_raises(self):
        with pytest.raises(ValueError, match="retry_wait_max_s"):
            StooqAdapter(retry_wait_init_s=5.0, retry_wait_max_s=3.0)

    def test_retry_wait_max_s_equals_init_accepted(self):
        a = StooqAdapter(retry_wait_init_s=5.0, retry_wait_max_s=5.0)
        assert a.retry_wait_max_s == 5.0

    def test_base_url_normalized_with_trailing_slash(self):
        a = StooqAdapter(base_url="https://example.com/api")
        assert a.base_url == "https://example.com/api/"


# ============================================================================
# URL building
# ============================================================================


class TestUrlBuilding:
    """Stooq URL construction for different tickers and date ranges."""

    def test_build_url_lowercases_ticker(self, adapter):
        url = adapter._build_url("AAPL")
        assert "s=aapl" in url
        assert "i=d" in url

    def test_build_url_plain_ticker(self, adapter):
        url = adapter._build_url("TSLA")
        assert "s=tsla" in url

    def test_build_date_url(self, adapter):
        start = date(2025, 1, 2)
        end = date(2025, 1, 10)
        url = adapter._build_date_url("AAPL", start, end)
        assert "s=aapl" in url
        assert "d1=20250102" in url
        assert "d2=20250110" in url
        assert "i=d" in url

    def test_build_date_url_single_day(self, adapter):
        d = date(2025, 3, 15)
        url = adapter._build_date_url("MSFT", d, d)
        assert "d1=20250315" in url
        assert "d2=20250315" in url


# ============================================================================
# CSV parsing
# ============================================================================


class TestParseStooqCsv:
    """Parse Stooq CSV text into polars DataFrame."""

    def test_standard_csv(self, sample_csv):
        result = _parse_stooq_csv(sample_csv, "AAPL")
        assert result.height == 3
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}
        assert result.schema["dt"] == pl.Date
        assert result["ticker"].to_list() == ["AAPL", "AAPL", "AAPL"]
        assert result["close"].to_list() == [151.5, 153.0, 154.5]
        assert result["volume"].to_list() == [100000, 110000, 120000]

    def test_empty_csv(self, empty_csv):
        result = _parse_stooq_csv(empty_csv, "AAPL")
        assert result.height == 0
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}

    def test_empty_string(self):
        result = _parse_stooq_csv("", "AAPL")
        assert result.height == 0

    def test_header_only_whitespace(self):
        result = _parse_stooq_csv("Date,Open,High,Low,Close,Volume\n", "AAPL")
        assert result.height == 0

    def test_partial_data(self, partial_csv):
        """Rows with N/A or empty values should use defaults (0.0 / 0)."""
        result = _parse_stooq_csv(partial_csv, "AAPL")
        # Rows with all N/A should still be included with 0.0 values
        assert result.height == 3
        # First row should have valid data
        first_row = result.row(0, named=True)
        assert first_row["close"] == 151.5
        # Second row values become 0.0 (N/A → safe default)
        second_row = result.row(1, named=True)
        assert second_row["open"] == 0.0  # N/A → safe_float default

    def test_csv_with_crlf_line_endings(self):
        """Stooq can return CRLF or LF line endings."""
        csv_text = (
            "Date,Open,High,Low,Close,Volume\r\n"
            "2025-01-02,150.0,152.0,149.0,151.5,100000\r\n"
        )
        result = _parse_stooq_csv(csv_text, "AAPL")
        assert result.height == 1

    def test_date_conversion(self, sample_csv):
        result = _parse_stooq_csv(sample_csv, "AAPL")
        dt_vals = result["dt"].to_list()
        assert dt_vals[0] == date(2025, 1, 2)
        assert dt_vals[2] == date(2025, 1, 6)


# ============================================================================
# Safe conversion helpers
# ============================================================================


class TestSafeFloat:
    """_safe_float returns float(v) with safety for None/empty strings."""

    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(42) == 42.0

    def test_string_float(self):
        assert _safe_float("3.14") == 3.14

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=1.0) == 1.0

    def test_empty_string_returns_default(self):
        assert _safe_float("") == 0.0
        assert _safe_float("", default=-1.0) == -1.0

    def test_invalid_string_returns_default(self):
        assert _safe_float("N/A") == 0.0
        assert _safe_float("abc") == 0.0


class TestSafeInt:
    """_safe_int returns int(v) with safety for None/empty strings."""

    def test_valid_int(self):
        assert _safe_int(42) == 42
        assert _safe_int(3.14) == 3

    def test_string_int(self):
        assert _safe_int("42") == 42

    def test_none_returns_default(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, default=5) == 5

    def test_empty_string_returns_default(self):
        assert _safe_int("") == 0

    def test_invalid_string_returns_default(self):
        assert _safe_int("N/A") == 0


# ============================================================================
# Retry policy
# ============================================================================


class TestRetryPolicy:
    """tenacity retry policy for transient HTTP errors."""

    def test_policy_is_callable(self):
        policy = _default_retry_policy()

        @policy
        def foo():
            pass

        assert callable(foo)

    def test_policy_retries_on_connection_error(self):
        import httpx

        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            raise httpx.ConnectError("transient")

        with pytest.raises(httpx.ConnectError):
            flaky()
        assert call_count[0] == STOOQ_MAX_RETRIES  # 3 attempts

    def test_policy_retries_on_http_429(self):
        import httpx

        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            resp = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
            raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            flaky()
        assert call_count[0] == STOOQ_MAX_RETRIES

    def test_policy_retries_on_http_5xx(self):
        import httpx

        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            resp = httpx.Response(503, request=httpx.Request("GET", "https://example.com"))
            raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            flaky()
        assert call_count[0] == STOOQ_MAX_RETRIES

    def test_policy_no_retry_on_http_4xx_except_429(self):
        """HTTP 404 (client error) should NOT be retried."""
        import httpx

        call_count = [0]

        @_default_retry_policy()
        def flaky():
            call_count[0] += 1
            resp = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))
            raise httpx.HTTPStatusError("not found", request=resp.request, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            flaky()
        assert call_count[0] == 1  # No retry on 404


class TestIsRetryableHttpStatus:
    """Unit tests for _is_retryable_http_status predicate."""

    def test_429_is_retryable(self):
        import httpx

        resp = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is True

    def test_503_is_retryable(self):
        import httpx

        resp = httpx.Response(503, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is True

    def test_500_is_retryable(self):
        import httpx

        resp = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("internal error", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is True

    def test_599_is_retryable(self):
        import httpx

        resp = httpx.Response(599, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("network error", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is True

    def test_404_is_not_retryable(self):
        import httpx

        resp = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("not found", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is False

    def test_400_is_not_retryable(self):
        import httpx

        resp = httpx.Response(400, request=httpx.Request("GET", "https://example.com"))
        exc = httpx.HTTPStatusError("bad request", request=resp.request, response=resp)
        assert _is_retryable_http_status(exc) is False

    def test_connect_error_is_not_http_status(self):
        import httpx

        exc = httpx.ConnectError("connection failed")
        assert _is_retryable_http_status(exc) is False


# ============================================================================
# Async download tests (mocked HTTP)
# ============================================================================


@pytest.mark.asyncio
class TestDownloadOhlcvAsync:
    """Async OHLCV download with mocked Stooq HTTP responses."""

    async def test_download_single_ticker(self, adapter_fast, sample_csv, monkeypatch):

        async def mock_fetch(_self, url):
            return sample_csv

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", mock_fetch)

        result = await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", "2025-01-10")
        assert result.height == 3
        assert "close" in result.columns
        assert result["ticker"].to_list() == ["AAPL", "AAPL", "AAPL"]

    async def test_download_multiple_tickers(self, adapter_fast, sample_csv, monkeypatch):
        async def mock_fetch(_self, url):
            # Return ticker-specific data based on URL
            if "aapl" in url.lower():
                return (
                    "Date,Open,High,Low,Close,Volume\r\n"
                    "2025-01-02,150.0,152.0,149.0,151.5,100000\r\n"
                )
            elif "googl" in url.lower():
                return (
                    "Date,Open,High,Low,Close,Volume\r\n"
                    "2025-01-02,140.0,142.0,139.0,141.0,200000\r\n"
                )
            return "Date,Open,High,Low,Close,Volume\r\n"

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", mock_fetch)

        result = await adapter_fast.download_ohlcv(
            ["AAPL", "GOOGL"], "2025-01-01", "2025-01-05"
        )
        assert result.height == 2
        tickers = set(result["ticker"].to_list())
        assert tickers == {"AAPL", "GOOGL"}

    async def test_download_empty_tickers(self, adapter_fast):
        result = await adapter_fast.download_ohlcv([], "2025-01-01", "2025-01-05")
        assert result.height == 0
        assert set(result.columns) == {"ticker", "dt", "open", "high", "low", "close", "volume"}

    async def test_download_handles_empty_response(self, adapter_fast, empty_csv, monkeypatch):
        async def mock_fetch(_self, url):
            return empty_csv

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", mock_fetch)

        result = await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", "2025-01-05")
        assert result.height == 0  # Empty CSV → empty DataFrame

    async def test_download_skips_failed_tickers(self, adapter_fast, sample_csv, monkeypatch):
        async def flaky_fetch(_self, url):
            if "bad" in url.lower():
                raise RuntimeError("network error")
            return sample_csv

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", flaky_fetch)

        result = await adapter_fast.download_ohlcv(
            ["AAPL", "BAD", "MSFT"], "2025-01-01", "2025-01-05"
        )
        # AAPL and MSFT should succeed; BAD should be silently skipped
        assert result.height > 0
        tickers = set(result["ticker"].to_list())
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "BAD" not in tickers

    async def test_download_date_objects_accepted(self, adapter_fast, sample_csv, monkeypatch):
        async def mock_fetch(_self, url):
            return sample_csv

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", mock_fetch)

        result = await adapter_fast.download_ohlcv(
            ["AAPL"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )
        assert result.height == 3

    async def test_download_end_date_defaults_to_today(self, adapter_fast, monkeypatch):
        captured_urls: list[str] = []

        async def capture_url(_self, url):
            captured_urls.append(url)
            return (
                "Date,Open,High,Low,Close,Volume\r\n"
                "2025-01-02,150.0,152.0,149.0,151.5,100000\r\n"
            )

        monkeypatch.setattr(StooqAdapter, "_fetch_csv", capture_url)

        await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01")
        assert len(captured_urls) == 1
        # Should have both d1 and d2 in the URL
        assert "d1=" in captured_urls[0]
        assert "d2=" in captured_urls[0]


# ============================================================================
# Rate limiting
# ============================================================================


@pytest.mark.asyncio
class TestRateLimiting:
    """Verify rate limiting slots are acquired and released."""

    async def test_semaphore_limits_concurrency(self, adapter):
        await adapter._acquire_slot()
        sem = adapter._semaphore
        assert sem is not None

        # Acquire remaining slot (default rps=2)
        await sem.acquire()
        assert sem.locked()

        # Release both
        sem.release()
        sem.release()

    async def test_release_after_delay(self, adapter):
        await adapter._acquire_slot()
        sem = adapter._semaphore
        await sem.acquire()  # lock it
        assert sem.locked()

        adapter._release_after_delay()

        await asyncio.sleep(1.1)
        assert not sem.locked()


# ============================================================================
# close() cleanup
# ============================================================================


@pytest.mark.asyncio
class TestClose:
    """close() cleans up the httpx client."""

    async def test_close_cleans_up_client(self, adapter_fast, sample_csv, monkeypatch):
        """close() must clean up the internal httpx client.

        We mock at the httpx.AsyncClient.get level so the adapter's own
        _get_client() is exercised and the client is created.
        """
        import httpx

        class MockResponse:
            @property
            def status_code(self):
                return 200

            @property
            def text(self):
                return sample_csv

            def raise_for_status(self):
                pass

        async def mock_get(_self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        # Trigger client creation
        await adapter_fast.download_ohlcv(["AAPL"], "2025-01-01", "2025-01-05")
        assert adapter_fast._client is not None

        await adapter_fast.close()
        assert adapter_fast._client is None

    async def test_close_idempotent(self, adapter):
        """close() on a fresh adapter should not raise."""
        await adapter.close()  # Should not raise
        await adapter.close()  # Double-close safe


# ============================================================================
# _utc_today
# ============================================================================


class TestUtcToday:
    """_utc_today returns today's date in UTC."""

    def test_returns_date_object(self):
        result = _utc_today()
        assert isinstance(result, date)
        assert not isinstance(result, datetime)

    def test_near_utc_now(self):
        from datetime import UTC as _UTC

        utc_now = datetime.now(_UTC).date()
        result = _utc_today()
        assert abs((result - utc_now).days) <= 1
