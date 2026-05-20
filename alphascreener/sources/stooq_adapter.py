"""Stooq free OHLCV data source adapter with rate-limiting and retry.

Issue #91: Stooq fallback adapter + cross-validation.
Reference: PRD 7.1.1 / 7.2.

Stooq provides free end-of-day OHLCV CSV downloads without authentication.
URL template: https://stooq.com/q/d/l/?s={ticker}&d1={YYYYMMDD}&d2={YYYYMMDD}&i=d
"""

from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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

STOOQ_BASE_URL: str = "https://stooq.com/q/d/l/"
STOOQ_DEFAULT_RPS: int = 2
STOOQ_MAX_RETRIES: int = 3
STOOQ_RETRY_WAIT_INIT_S: float = 2.0
STOOQ_RETRY_WAIT_MAX_S: float = 30.0

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def _default_retry_policy(
    max_retries: int = STOOQ_MAX_RETRIES,
    wait_init: float = STOOQ_RETRY_WAIT_INIT_S,
    wait_max: float = STOOQ_RETRY_WAIT_MAX_S,
):
    """Exponential backoff: 2s, 4s, 8s, up to 30s.

    Retries on transient HTTP / network errors.
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
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _parse_stooq_csv(csv_text: str, ticker: str) -> pl.DataFrame:
    """Parse Stooq CSV response into a polars OHLCV DataFrame.

    Stooq CSV format::

        Date,Open,High,Low,Close,Volume
        2025-01-02,150.0,152.0,149.0,151.5,100000

    Returns a DataFrame with columns: ticker, dt, open, high, low, close, volume.
    Returns an empty DataFrame with the correct schema when CSV has no data rows.
    """
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        # No data rows (only header or empty)
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

    reader = csv.DictReader(lines)
    records: list[dict[str, Any]] = []
    for row in reader:
        try:
            dt_val = date.fromisoformat(row["Date"])
        except (ValueError, KeyError):
            continue
        records.append(
            {
                "ticker": ticker,
                "dt": dt_val,
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": _safe_int(row.get("Volume")),
            }
        )

    if not records:
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

    return pl.DataFrame(records).with_columns(pl.col("dt").cast(pl.Date))


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Return float(v) or *default* when *v* is None or empty."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    """Return int(v) or *default* when *v* is None or empty."""
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _utc_today() -> date:
    """Return today's date in UTC."""
    return datetime.now(UTC).date()


# ---------------------------------------------------------------------------
# StooqAdapter
# ---------------------------------------------------------------------------


@dataclass
class StooqAdapter:
    """Async Stooq OHLCV data source adapter with rate-limiting and retry.

    Features (PRD 7.1.1 免费备用数据源):
      - Free end-of-day OHLCV CSV download (no API key required)
      - Rate limit: asyncio.Semaphore (default 2 RPS, conservative for free tier)
      - Retry: tenacity exponential backoff (initial 2s, max 30s, 3 attempts)
      - Per-ticker download with empty-result handling

    Stooq is the first-priority fallback when yfinance is unavailable.

    Reference: PRD 7.1 数据源架构 / 7.2 数据交叉校验.
    """

    base_url: str = STOOQ_BASE_URL
    rps: int = STOOQ_DEFAULT_RPS
    max_retries: int = STOOQ_MAX_RETRIES
    retry_wait_init_s: float = STOOQ_RETRY_WAIT_INIT_S
    retry_wait_max_s: float = STOOQ_RETRY_WAIT_MAX_S

    # -- Internal state (not user-configurable) ----------------------------------

    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False, init=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False, init=False)
    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("screening"), repr=False, init=False
    )

    def __post_init__(self) -> None:
        """Validate constructor parameters."""
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
        """Create a new httpx.AsyncClient."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
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

    @staticmethod
    def _build_url(ticker: str) -> str:
        """Build the Stooq download URL for a ticker.

        The URL format is: https://stooq.com/q/d/l/?s={ticker}&i=d

        The ticker is lowercased for Stooq compatibility (US equities).
        """
        return f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"

    @staticmethod
    def _build_date_url(ticker: str, start_date: date, end_date: date) -> str:
        """Build the Stooq download URL with explicit date range.

        d1 and d2 use YYYYMMDD format.
        """
        d1 = start_date.strftime("%Y%m%d")
        d2 = end_date.strftime("%Y%m%d")
        return (
            f"https://stooq.com/q/d/l/?s={ticker.lower()}"
            f"&d1={d1}&d2={d2}&i=d"
        )

    async def _fetch_csv(self, url: str) -> str:
        """Fetch CSV data from a Stooq URL with rate limiting and retry.

        Args:
            url: Full Stooq download URL.

        Returns:
            Raw CSV text.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses (after retries exhausted).
        """
        await self._acquire_slot()
        try:
            client = await self._get_client()

            retried_get = _default_retry_policy(
                max_retries=self.max_retries,
                wait_init=self.retry_wait_init_s,
                wait_max=self.retry_wait_max_s,
            )(client.get)

            response = await retried_get(url)
            response.raise_for_status()
            return response.text
        finally:
            self._release_after_delay()

    # -- OHLCV download ----------------------------------------------------------

    async def download_ohlcv(
        self,
        tickers: list[str],
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> pl.DataFrame:
        """Download OHLCV data for a list of tickers from Stooq.

        Each ticker is fetched individually with rate limiting and retry.
        Stooq does not support batch downloads.

        Args:
            tickers: List of ticker symbols (e.g. ``["AAPL", "GOOGL"]``).
            start_date: Start date (inclusive), ISO string or ``date``.
            end_date: End date (inclusive). Defaults to today.

        Returns:
            polars DataFrame with columns: ticker, dt, open, high, low, close, volume.
        """
        today = _utc_today()

        # -- Normalize dates ---------------------------------------------------
        if isinstance(start_date, date):
            start = start_date
        else:
            start = date.fromisoformat(start_date)

        if end_date is None:
            end = today
        elif isinstance(end_date, date):
            end = end_date
        else:
            end = date.fromisoformat(end_date)

        if not tickers:
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

        self._logger.info(
            "Downloading OHLCV from Stooq for %d tickers (%s → %s)",
            len(tickers),
            start.isoformat(),
            end.isoformat(),
        )

        results: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                url = self._build_date_url(ticker, start, end)
                csv_text = await self._fetch_csv(url)
                pl_df = _parse_stooq_csv(csv_text, ticker)
                if pl_df.height > 0:
                    results.append(pl_df)
                else:
                    self._logger.debug("Stooq returned empty data for %s", ticker)
            except Exception as e:
                self._logger.warning("Stooq OHLCV fetch for %s failed: %s", ticker, e)

        if not results:
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

        combined = pl.concat(results)
        return combined.with_columns(pl.col("dt").cast(pl.Date))
