"""FMP (Financial Modeling Prep) data source adapter with budget control and rate-limiting.

Issue #90: FMP adapter.
Reference: PRD 7.1 / 7.1.2.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FMP_BASE_URL: str = "https://financialmodelingprep.com/api/"
FMP_DAILY_BUDGET: int = 250
FMP_DEFAULT_RPS: int = 2
FMP_MAX_RETRIES: int = 3
FMP_RETRY_WAIT_INIT_S: float = 2.0
FMP_RETRY_WAIT_MAX_S: float = 30.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FmpBudgetExhaustedError(RuntimeError):
    """Raised when the FMP daily budget (≤250 req/day for Free tier) is exhausted."""

    pass


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def _default_retry_policy(
    max_retries: int = FMP_MAX_RETRIES,
    wait_init: float = FMP_RETRY_WAIT_INIT_S,
    wait_max: float = FMP_RETRY_WAIT_MAX_S,
):
    """Exponential backoff: 2s, 4s, 8s, up to 30s.

    Retries on transient HTTP / network errors.  Does NOT retry on 429
    (rate-limit) — that is handled separately with a configurable wait.
    """
    return retry(
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, OSError)
        ),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=wait_init, max=wait_max),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# FmpAdapter
# ---------------------------------------------------------------------------


@dataclass
class FmpAdapter:
    """Async FMP (Financial Modeling Prep) Free tier adapter.

    Features (PRD 7.1 FMP Free tier 调用约束):
      - ≤250 req/day budget control
      - Rate limit: asyncio.Semaphore (default 2 RPS, conservative for Free tier)
      - Retry: tenacity exponential backoff (initial 2s, max 30s, 3 attempts)
      - Endpoints: analyst-estimates, insider-trading, grade, stock_news,
        earning_calendar, historical-earning-calendar
      - FMP unavailable → fallback to yfinance

    Reference: PRD 7.1 数据源架构 / 7.1.2 数据源→字段精确映射.
    """

    api_key: str

    daily_budget: int = FMP_DAILY_BUDGET
    rps: int = FMP_DEFAULT_RPS
    base_url: str = FMP_BASE_URL
    max_retries: int = FMP_MAX_RETRIES
    retry_wait_init_s: float = FMP_RETRY_WAIT_INIT_S
    retry_wait_max_s: float = FMP_RETRY_WAIT_MAX_S

    # -- Internal state (not user-configurable) ----------------------------------

    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False, init=False)
    _request_count: int = field(default=0, repr=False, init=False)
    _budget_date: date | None = field(default=None, repr=False, init=False)
    _budget_lock: asyncio.Lock | None = field(default=None, repr=False, init=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False, init=False)
    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("screening"), repr=False, init=False
    )

    def __post_init__(self) -> None:
        """Validate constructor parameters."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.daily_budget < 1:
            raise ValueError(f"daily_budget must be >= 1, got {self.daily_budget}")
        if self.rps < 1:
            raise ValueError(f"rps must be >= 1, got {self.rps}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if self.retry_wait_init_s <= 0:
            raise ValueError(f"retry_wait_init_s must be > 0, got {self.retry_wait_init_s}")
        if self.retry_wait_max_s < self.retry_wait_init_s:
            raise ValueError(
                f"retry_wait_max_s ({self.retry_wait_max_s}) must be >= "
                f"retry_wait_init_s ({self.retry_wait_init_s})"
            )
        # Normalize base_url to end with a single slash
        self.base_url = self.base_url.rstrip("/") + "/"

    # -- Budget control ---------------------------------------------------------

    def _check_date(self) -> None:
        """Reset counter if a new day has started."""
        today = date.today()
        if self._budget_date is None or self._budget_date != today:
            self._request_count = 0
            self._budget_date = today

    async def _reserve_budget(self) -> None:
        """Atomically check budget and increment the daily request counter.

        Raises :class:`FmpBudgetExhaustedError` if the daily budget is exhausted.

        Uses an :class:`asyncio.Lock` so that concurrent callers cannot race
        past the daily budget limit.
        """
        if self._budget_lock is None:
            self._budget_lock = asyncio.Lock()
        async with self._budget_lock:
            self._check_date()
            if self._request_count >= self.daily_budget:
                msg = f"FMP daily budget exhausted: {self._request_count}/{self.daily_budget}"
                raise FmpBudgetExhaustedError(msg)
            self._request_count += 1

    @property
    def requests_used_today(self) -> int:
        """Number of requests made today."""
        self._check_date()
        return self._request_count

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether the daily budget has been reached."""
        self._check_date()
        return self._request_count >= self.daily_budget

    def reset_budget(self) -> None:
        """Reset the daily request counter (for testing)."""
        self._request_count = 0
        self._budget_date = None

    # -- Semaphore helpers -------------------------------------------------------

    async def _acquire_slot(self) -> None:
        """Acquire a rate-limit slot."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.rps)
        await self._semaphore.acquire()

    def _release_after_delay(self) -> None:
        """Schedule semaphore release after 1s (1 slot = 1 RPS)."""
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, self._semaphore.release)

    # -- HTTP client -------------------------------------------------------------

    def _create_client(self) -> httpx.AsyncClient:
        """Create a new httpx.AsyncClient with API key header."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"x-api-key": self.api_key},
            timeout=httpx.Timeout(30.0),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Core execution wrapper --------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Perform an HTTP GET with rate limiting, budget check, and retry.

        Args:
            path: API path relative to base_url (e.g. ``"v3/analyst-estimates/AAPL"``).
            params: Optional query parameters (api key is injected automatically).

        Returns:
            The httpx Response object.
        """
        await self._reserve_budget()

        await self._acquire_slot()
        try:
            client = await self._get_client()

            retried_get = _default_retry_policy(
                max_retries=self.max_retries,
                wait_init=self.retry_wait_init_s,
                wait_max=self.retry_wait_max_s,
            )(client.get)

            merged_params = params.copy() if params else {}
            merged_params["apikey"] = self.api_key

            response = await retried_get(path, params=merged_params)

            # Handle rate limiting (429)
            if response.status_code == 429:
                # Retry with a fixed 60s wait (Free tier rate limit window)
                self._logger.warning("FMP rate limit (429) — waiting 60s before retry")
                await asyncio.sleep(60)
                response = await retried_get(path, params=merged_params)
                if response.status_code == 429:
                    raise RuntimeError("FMP rate limit (429) persisted after 60s wait")
                response.raise_for_status()

            response.raise_for_status()
            return response
        finally:
            self._release_after_delay()

    # -- Polars conversion helpers -----------------------------------------------

    @staticmethod
    def _analyst_estimates_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP analyst-estimates JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "estimated_revenue_avg": pl.Float64,
                    "estimated_eps_avg": pl.Float64,
                    "estimated_eps_high": pl.Float64,
                    "estimated_eps_low": pl.Float64,
                    "estimated_ebitda_avg": pl.Float64,
                }
            )
        records = []
        for item in data:
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "estimated_revenue_avg": float(item.get("estimatedRevenueAvg", 0) or 0),
                    "estimated_eps_avg": float(item.get("estimatedEpsAvg", 0) or 0),
                    "estimated_eps_high": float(item.get("estimatedEpsHigh", 0) or 0),
                    "estimated_eps_low": float(item.get("estimatedEpsLow", 0) or 0),
                    "estimated_ebitda_avg": float(item.get("estimatedEbitdaAvg", 0) or 0),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _insider_trading_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP insider-trading JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "transaction_date": pl.Utf8,
                    "reporting_name": pl.Utf8,
                    "relationship": pl.Utf8,
                    "transaction_type": pl.Utf8,
                    "securities_transacted": pl.Float64,
                    "price": pl.Float64,
                    "security_name": pl.Utf8,
                }
            )
        records = []
        for item in data:
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "transaction_date": item.get("transactionDate", ""),
                    "reporting_name": item.get("reportingName", ""),
                    "relationship": item.get("relationship", ""),
                    "transaction_type": item.get("transactionType", ""),
                    "securities_transacted": float(item.get("securitiesTransacted", 0) or 0),
                    "price": float(item.get("price", 0) or 0),
                    "security_name": item.get("securityName", ""),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _grade_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP grade JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "grading_company": pl.Utf8,
                    "previous_grade": pl.Utf8,
                    "new_grade": pl.Utf8,
                }
            )
        records = []
        for item in data:
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "grading_company": item.get("gradingCompany", ""),
                    "previous_grade": item.get("previousGrade", ""),
                    "new_grade": item.get("newGrade", ""),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _stock_news_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP stock_news JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "published_at": pl.Utf8,
                    "title": pl.Utf8,
                    "url": pl.Utf8,
                    "site": pl.Utf8,
                }
            )
        records = []
        for item in data:
            pub_date = item.get("publishedDate", "")
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "published_at": str(pub_date),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "site": item.get("site", ""),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _earning_calendar_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP earning_calendar JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "fiscal_date_ending": pl.Utf8,
                    "time": pl.Utf8,
                }
            )
        records = []
        for item in data:
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "fiscal_date_ending": item.get("fiscalDateEnding", ""),
                    "time": item.get("time", ""),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _historical_earning_to_polars(data: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert FMP historical-earning-calendar JSON to polars DataFrame."""
        if not data:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "eps": pl.Float64,
                    "eps_estimated": pl.Float64,
                    "revenue": pl.Float64,
                    "revenue_estimated": pl.Float64,
                    "fiscal_date_ending": pl.Utf8,
                    "time": pl.Utf8,
                }
            )
        records = []
        for item in data:
            records.append(
                {
                    "ticker": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "eps": float(item.get("eps", 0) or 0),
                    "eps_estimated": float(item.get("epsEstimated", 0) or 0),
                    "revenue": float(item.get("revenue", 0) or 0),
                    "revenue_estimated": float(item.get("revenueEstimated", 0) or 0),
                    "fiscal_date_ending": item.get("fiscalDateEnding", ""),
                    "time": item.get("time", ""),
                }
            )
        return pl.DataFrame(records)

    @staticmethod
    def _yfinance_earnings_to_analyst_estimates(yf_df: pl.DataFrame, ticker: str) -> pl.DataFrame:
        """Convert yfinance earnings_dates DataFrame to FMP analyst_estimates schema.

        yfinance earnings_dates columns:
          ticker, earnings_date, eps_estimate, reported_eps, surprise_pct

        FMP analyst_estimates columns:
          ticker, date, estimated_revenue_avg, estimated_eps_avg,
          estimated_eps_high, estimated_eps_low, estimated_ebitda_avg

        Columns that have no yfinance equivalent are filled with null/None.
        """
        if yf_df.height == 0:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "estimated_revenue_avg": pl.Float64,
                    "estimated_eps_avg": pl.Float64,
                    "estimated_eps_high": pl.Float64,
                    "estimated_eps_low": pl.Float64,
                    "estimated_ebitda_avg": pl.Float64,
                }
            )
        records = []
        for row in yf_df.iter_rows(named=True):
            records.append(
                {
                    "ticker": row.get("ticker", ticker),
                    "date": str(row.get("earnings_date", "")),
                    "estimated_revenue_avg": None,
                    "estimated_eps_avg": float(row.get("eps_estimate", 0) or 0),
                    "estimated_eps_high": None,
                    "estimated_eps_low": None,
                    "estimated_ebitda_avg": None,
                }
            )
        return pl.DataFrame(records)

    # -- Date helpers ------------------------------------------------------------

    @staticmethod
    def _parse_fmp_date(date_str: str) -> date:
        """Parse an FMP date string like '2025-01-20' into a date object."""
        return date.fromisoformat(date_str)

    @staticmethod
    def _parse_fmp_datetime(dt_str: str) -> str:
        """Normalize an FMP datetime string to ISO format."""
        if not dt_str:
            return ""
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.isoformat()
        except (ValueError, TypeError):
            return dt_str

    # -- Async endpoint methods --------------------------------------------------

    async def fetch_analyst_estimates(self, ticker: str, period: str = "quarter") -> pl.DataFrame:
        """Fetch analyst consensus EPS estimates.

        Endpoint: ``GET /v3/analyst-estimates/{ticker}?period=quarter``

        Args:
            ticker: Ticker symbol.
            period: Estimate period (``"quarter"`` or ``"annual"``).

        Returns:
            polars DataFrame with estimated EPS, revenue, EBITDA fields.
        """
        if self.is_budget_exhausted:
            self._logger.warning("FMP budget exhausted — skipping analyst estimates for %s", ticker)
            return self._analyst_estimates_to_polars([])

        try:
            response = await self._get(f"v3/analyst-estimates/{ticker}", params={"period": period})
            data: list[dict[str, Any]] = response.json()
            if data:
                return self._analyst_estimates_to_polars(data)
            # FMP returned empty results — fall through to yfinance fallback
        except FmpBudgetExhaustedError:
            return self._analyst_estimates_to_polars([])
        except Exception as e:
            self._logger.warning("FMP analyst-estimates fetch for %s failed: %s", ticker, e)
            # FMP failed — fall through to yfinance fallback

        # Fallback to yfinance when FMP is unavailable or returns empty data
        try:
            from alphascreener.sources.yfinance_adapter import YFinanceAdapter

            yf = YFinanceAdapter()
            result = await yf.download_earnings_dates([ticker])
            if result.height > 0:
                self._logger.info(
                    "FMP fallback: yfinance returned %d earnings rows for %s",
                    result.height,
                    ticker,
                )
                return self._yfinance_earnings_to_analyst_estimates(result, ticker)
            self._logger.warning("FMP fallback: yfinance returned empty earnings for %s", ticker)
            return self._analyst_estimates_to_polars([])
        except Exception as e:
            self._logger.warning(
                "FMP fallback: yfinance earnings fetch for %s also failed: %s", ticker, e
            )
            return self._analyst_estimates_to_polars([])

    async def fetch_insider_trading(self, ticker: str, limit: int = 50) -> pl.DataFrame:
        """Fetch detailed insider trading transactions.

        Endpoint: ``GET /v4/insider-trading?symbol={ticker}&limit={limit}``

        Args:
            ticker: Ticker symbol.
            limit: Maximum number of transactions to return.

        Returns:
            polars DataFrame with insider name, relationship, transaction type, shares, price.
        """
        if self.is_budget_exhausted:
            self._logger.warning("FMP budget exhausted — skipping insider trading for %s", ticker)
            return self._insider_trading_to_polars([])

        try:
            response = await self._get(
                "v4/insider-trading", params={"symbol": ticker, "limit": limit}
            )
            data: list[dict[str, Any]] = response.json()
            return self._insider_trading_to_polars(data)
        except FmpBudgetExhaustedError:
            return self._insider_trading_to_polars([])
        except Exception as e:
            self._logger.warning("FMP insider-trading fetch for %s failed: %s", ticker, e)
            return self._insider_trading_to_polars([])

    async def fetch_grade(self, ticker: str, limit: int = 20) -> pl.DataFrame:
        """Fetch analyst rating grades.

        Endpoint: ``GET /v3/grade/{ticker}?limit={limit}``

        Args:
            ticker: Ticker symbol.
            limit: Maximum number of grade changes to return.

        Returns:
            polars DataFrame with grading company, previous grade, new grade.
        """
        if self.is_budget_exhausted:
            self._logger.warning("FMP budget exhausted — skipping grade for %s", ticker)
            return self._grade_to_polars([])

        try:
            response = await self._get(f"v3/grade/{ticker}", params={"limit": limit})
            data: list[dict[str, Any]] = response.json()
            return self._grade_to_polars(data)
        except FmpBudgetExhaustedError:
            return self._grade_to_polars([])
        except Exception as e:
            self._logger.warning("FMP grade fetch for %s failed: %s", ticker, e)
            return self._grade_to_polars([])

    async def fetch_stock_news(self, tickers: list[str], limit: int = 50) -> pl.DataFrame:
        """Fetch stock news for one or more tickers.

        Endpoint: ``GET /v3/stock_news?tickers={ticker1,ticker2,...}&limit={limit}``

        Args:
            tickers: List of ticker symbols (max ~5 for Free tier).
            limit: Maximum number of articles to return.

        Returns:
            polars DataFrame with title, URL, site, published_at.
        """
        if self.is_budget_exhausted:
            self._logger.warning("FMP budget exhausted — skipping stock news")
            return self._stock_news_to_polars([])

        try:
            ticker_str = ",".join(tickers)
            response = await self._get(
                "v3/stock_news", params={"tickers": ticker_str, "limit": limit}
            )
            data: list[dict[str, Any]] = response.json()
            return self._stock_news_to_polars(data)
        except FmpBudgetExhaustedError:
            return self._stock_news_to_polars([])
        except Exception as e:
            self._logger.warning("FMP stock_news fetch for %s failed: %s", tickers, e)
            return self._stock_news_to_polars([])

    async def fetch_earning_calendar(
        self, start_date: str | date, end_date: str | date
    ) -> pl.DataFrame:
        """Fetch earnings calendar for a date range.

        Endpoint: ``GET /v3/earning_calendar?from={from}&to={to}``

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            polars DataFrame with ticker, date, fiscal_date_ending, time.
        """
        if self.is_budget_exhausted:
            self._logger.warning("FMP budget exhausted — skipping earning calendar")
            return self._earning_calendar_to_polars([])

        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()

        try:
            response = await self._get(
                "v3/earning_calendar", params={"from": start_date, "to": end_date}
            )
            data: list[dict[str, Any]] = response.json()
            return self._earning_calendar_to_polars(data)
        except FmpBudgetExhaustedError:
            return self._earning_calendar_to_polars([])
        except Exception as e:
            self._logger.warning(
                "FMP earning_calendar fetch (%s → %s) failed: %s", start_date, end_date, e
            )
            return self._earning_calendar_to_polars([])

    async def fetch_historical_earning_calendar(self, ticker: str, limit: int = 8) -> pl.DataFrame:
        """Fetch historical earnings calendar for a ticker.

        Endpoint: ``GET /v3/historical-earning-calendar/{ticker}?limit={limit}``

        Args:
            ticker: Ticker symbol.
            limit: Maximum number of historical earnings to return.

        Returns:
            polars DataFrame with EPS actual, estimated, revenue, and date fields.
        """
        if self.is_budget_exhausted:
            self._logger.warning(
                "FMP budget exhausted — skipping historical earning for %s", ticker
            )
            return self._historical_earning_to_polars([])

        try:
            response = await self._get(
                f"v3/historical-earning-calendar/{ticker}", params={"limit": limit}
            )
            data: list[dict[str, Any]] = response.json()
            return self._historical_earning_to_polars(data)
        except FmpBudgetExhaustedError:
            return self._historical_earning_to_polars([])
        except Exception as e:
            self._logger.warning(
                "FMP historical-earning-calendar fetch for %s failed: %s", ticker, e
            )
            return self._historical_earning_to_polars([])

    # -- Bulk fetch ---------------------------------------------------------------

    async def fetch_all_for_ticker(self, ticker: str) -> dict[str, Any]:
        """Fetch all FMP data for a single ticker (used for precision screening Top 20).

        Returns:
            Dict with keys: analyst_estimates, insider_trading, grade,
            historical_earnings.
        """
        analyst, insider, grade, historical = await asyncio.gather(
            self.fetch_analyst_estimates(ticker),
            self.fetch_insider_trading(ticker),
            self.fetch_grade(ticker),
            self.fetch_historical_earning_calendar(ticker),
        )
        return {
            "analyst_estimates": analyst,
            "insider_trading": insider,
            "grade": grade,
            "historical_earnings": historical,
        }
