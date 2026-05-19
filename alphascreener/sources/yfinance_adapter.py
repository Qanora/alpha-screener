"""yfinance data source adapter with rate-limiting, retry, and circuit breaker.

Issue #89: yfinance adapter.
Reference: PRD 7.1 / 7.3.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl
import yfinance as yf
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

BATCH_SIZE_MAX: int = 50
DEFAULT_RPS: int = 5
MAX_RETRIES: int = 3
RETRY_WAIT_INIT_S: float = 2.0
RETRY_WAIT_MAX_S: float = 60.0
CIRCUIT_BREAKER_THRESHOLD: int = 3
CIRCUIT_BREAKER_TTL_DAYS: int = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a ticker's circuit breaker is open (3 consecutive failures)."""

    pass


# ---------------------------------------------------------------------------
# Retry policy for tenacity
# ---------------------------------------------------------------------------


def _default_retry_policy(
    max_retries: int = MAX_RETRIES,
    wait_init: float = RETRY_WAIT_INIT_S,
    wait_max: float = RETRY_WAIT_MAX_S,
):
    """Exponential backoff: 2s, 4s, 8s, up to 60s.

    Parameters are exposed so callers (e.g. ``YFinanceAdapter._rate_limited_call``)
    can override with per-instance values.
    """
    return retry(
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError, RuntimeError)
        ),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=wait_init, max=wait_max),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Synchronous helpers (run in executor)
# ---------------------------------------------------------------------------


def _download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for a batch of tickers.

    Wrapped with ``@_default_retry_policy`` for automatic retry on transient failures.
    """
    ticker_str = " ".join(tickers)
    df = yf.download(
        ticker_str,
        start=start,
        end=end,
        threads=False,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for batch of {len(tickers)} tickers")
    # yf.download returns MultiIndex columns when multiple tickers: (Open, AAPL) etc.
    # Single ticker: plain columns: Open, High, Low, Close, Volume.
    return df  # type: ignore[return-value]


def _fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Fetch fundamentals via Ticker.info."""
    t = yf.Ticker(ticker)
    info = t.info
    if not info or info.get("regularMarketPreviousClose") is None:
        raise RuntimeError(f"Empty or invalid info for {ticker}")
    return info


def _fetch_earnings_dates(ticker: str) -> pd.DataFrame | None:
    """Fetch earnings dates via Ticker.earnings_dates."""
    t = yf.Ticker(ticker)
    df = t.earnings_dates
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    return df  # type: ignore[return-value]


def _fetch_insider_transactions(ticker: str) -> pd.DataFrame | None:
    """Fetch insider transactions via Ticker.insider_transactions."""
    t = yf.Ticker(ticker)
    df = t.insider_transactions
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    return df  # type: ignore[return-value]


def _fetch_news(ticker: str) -> list[dict[str, Any]]:
    """Fetch news via Ticker.news.

    Returns a list of dicts with keys: title, link, publisher, providerPublishTime, type.
    """
    t = yf.Ticker(ticker)
    return t.news or []  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# NaN-safe conversion helpers
# ---------------------------------------------------------------------------


def _safe_float(v, default=0.0):
    """Return float(v) or *default* when *v* is None or NaN."""
    if v is None or pd.isna(v):
        return default
    return float(v)


def _safe_int(v, default=0):
    """Return int(v) or *default* when *v* is None or NaN."""
    if v is None or pd.isna(v):
        return default
    return int(v)


def _utc_today() -> date:
    """Return today's date in UTC, avoiding local timezone discrepancies."""
    return datetime.now(UTC).date()


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def _ohlcv_to_polars(pd_df: pd.DataFrame, fallback_ticker: str | None = None) -> pl.DataFrame:
    """Convert yfinance download output to a tidy polars DataFrame.

    Handles both single-ticker (plain columns) and multi-ticker (MultiIndex columns)
    outputs from ``yf.download()``.

    Args:
        pd_df: Raw DataFrame from ``yf.download()``.
        fallback_ticker: Ticker symbol used in the ``"ticker"`` column when
            ``pd_df`` has plain (non-MultiIndex) columns.  Required for correct
            output when the caller downloads a single ticker.
    """
    records: list[dict[str, Any]] = []
    columns = pd_df.columns

    if isinstance(columns, pd.MultiIndex):
        # Multi-ticker: columns are (Price, TICKER)
        tickers = sorted(set(c[1] for c in columns))
        for ticker in tickers:
            try:
                sub = pd_df.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                continue
            sub = sub.dropna(how="all")
            for idx, row in sub.iterrows():
                dt_val = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
                records.append(
                    {
                        "ticker": ticker,
                        "dt": dt_val,
                        "open": _safe_float(row.get("Open", 0)),
                        "high": _safe_float(row.get("High", 0)),
                        "low": _safe_float(row.get("Low", 0)),
                        "close": _safe_float(row.get("Close", 0)),
                        "volume": _safe_int(row.get("Volume", 0)),
                    }
                )
    else:
        # Single ticker: plain column names
        sub = pd_df.dropna(how="all")
        for idx, row in sub.iterrows():
            dt_val = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
            records.append(
                {
                    "ticker": fallback_ticker or "",
                    "dt": dt_val,
                    "open": _safe_float(row.get("Open", 0)),
                    "high": _safe_float(row.get("High", 0)),
                    "low": _safe_float(row.get("Low", 0)),
                    "close": _safe_float(row.get("Close", 0)),
                    "volume": _safe_int(row.get("Volume", 0)),
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


def _info_to_dict(info: dict[str, Any]) -> dict[str, Any]:
    """Extract key fundamental fields from Ticker.info."""
    return {
        "symbol": info.get("symbol", ""),
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "forwardEps": info.get("forwardEps"),
        "trailingEps": info.get("trailingEps"),
        "pegRatio": info.get("pegRatio"),
        "dividendYield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "regularMarketTime": info.get("regularMarketTime"),
    }


def _earnings_to_polars(pd_df: pd.DataFrame, ticker: str) -> pl.DataFrame:
    """Convert earnings_dates DataFrame to polars."""
    if pd_df is None or pd_df.empty:
        return pl.DataFrame()
    df = pd_df.reset_index()
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "ticker": ticker,
                "earnings_date": str(row.get("Earnings Date", "")),
                "eps_estimate": float(row.get("EPS Estimate", 0) or 0),
                "reported_eps": float(row.get("Reported EPS", 0) or 0),
                "surprise_pct": float(row.get("Surprise(%)", 0) or 0),
            }
        )
    return pl.DataFrame(records)


def _insider_to_polars(pd_df: pd.DataFrame, ticker: str) -> pl.DataFrame:
    """Convert insider_transactions DataFrame to polars."""
    if pd_df is None or pd_df.empty:
        return pl.DataFrame()
    records = []
    for _, row in pd_df.iterrows():
        shares_val = row.get("Shares", 0)
        shares = 0 if pd.isna(shares_val) else int(shares_val)
        value_val = row.get("Value", 0)
        value = 0.0 if pd.isna(value_val) else float(value_val)
        records.append(
            {
                "ticker": ticker,
                "insider_name": str(row.get("Insider", "")),
                "title": str(row.get("Title", "")),
                "transaction_type": str(row.get("Transaction", "")),
                "shares": shares,
                "value": value,
                "start_date": str(row.get("Start Date", "")),
            }
        )
    return pl.DataFrame(records)


def _news_to_polars(news_list: list[dict[str, Any]], ticker: str) -> pl.DataFrame:
    """Convert Ticker.news list to polars DataFrame."""
    if not news_list:
        return pl.DataFrame()
    records = []
    for item in news_list:
        ts = item.get("providerPublishTime", 0)
        if ts is None or pd.isna(ts):
            dt_str = ""
        else:
            dt_str = datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
        records.append(
            {
                "ticker": ticker,
                "title": str(item.get("title", "")),
                "link": str(item.get("link", "")),
                "publisher": str(item.get("publisher", "")),
                "published_at": dt_str,
                "news_type": str(item.get("type", "")),
            }
        )
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# YFinanceAdapter
# ---------------------------------------------------------------------------


@dataclass
class YFinanceAdapter:
    """Async yfinance data source adapter with rate-limiting, retry, and circuit breaker.

    Features (PRD 7.3 yfinance 调用约束):
      - Batch download: ≤50 stocks/batch, ``threads=False``
      - Rate limit: ``asyncio.Semaphore`` 5 RPS
      - Retry: tenacity exponential backoff (initial 2s, max 60s, 3 attempts)
      - Circuit breaker: single ticker 3 consecutive failures → skipped for the day

    Reference: PRD 7.1 数据源架构 / 7.3 yfinance 调用约束.
    """

    batch_size: int = BATCH_SIZE_MAX
    rps: int = DEFAULT_RPS
    max_retries: int = MAX_RETRIES
    retry_wait_init_s: float = RETRY_WAIT_INIT_S
    retry_wait_max_s: float = RETRY_WAIT_MAX_S

    # -- Internal state (not user-configurable) ----------------------------------

    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Validate constructor parameters."""
        if not (1 <= self.batch_size <= BATCH_SIZE_MAX):
            raise ValueError(f"batch_size must be 1-{BATCH_SIZE_MAX}, got {self.batch_size}")
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

    _failures: dict[str, int] = field(default_factory=dict, repr=False, init=False)
    _skip_until: dict[str, date] = field(default_factory=dict, repr=False, init=False)
    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("screening"), repr=False, init=False
    )

    # -- Semaphore helpers -------------------------------------------------------

    async def _acquire_slot(self) -> None:
        """Acquire a rate-limit slot (enforcing ≤5 RPS)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.rps)
        await self._semaphore.acquire()

    def _release_after_delay(self) -> None:
        """Schedule semaphore release after 1s (1 slot = 1 RPS)."""
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, self._semaphore.release)

    # -- Circuit breaker ---------------------------------------------------------

    def _is_circuit_open(self, ticker: str, today: date | None = None) -> bool:
        """Check whether a ticker's circuit breaker is open."""
        if today is None:
            today = _utc_today()
        if ticker in self._skip_until:
            if today < self._skip_until[ticker]:
                return True
            # TTL expired — reset
            del self._skip_until[ticker]
            self._failures.pop(ticker, None)
        return False

    def _record_failure(self, ticker: str, today: date | None = None) -> None:
        """Record a failure for a ticker; open circuit breaker on threshold."""
        if today is None:
            today = _utc_today()
        self._failures[ticker] = self._failures.get(ticker, 0) + 1
        if self._failures[ticker] >= CIRCUIT_BREAKER_THRESHOLD:
            skip_until = today + timedelta(days=CIRCUIT_BREAKER_TTL_DAYS)
            self._skip_until[ticker] = skip_until
            self._logger.warning(
                "Circuit breaker open for %s until %s (3 consecutive failures)",
                ticker,
                skip_until.isoformat(),
            )

    def _record_success(self, ticker: str) -> None:
        """Reset failure counter on success."""
        self._failures.pop(ticker, None)
        self._skip_until.pop(ticker, None)

    # -- Core execution wrapper --------------------------------------------------

    async def _rate_limited_call(
        self,
        ticker: str,
        func,
        *args,
        track_circuit: bool = True,
        **kwargs,
    ) -> Any:
        """Execute a synchronous function with rate limiting and circuit breaker.

        The function ``func(*args, **kwargs)`` runs in a thread executor.
        Circuit breaker is checked before invocation and updated on result.

        Args:
            ticker: Ticker symbol used for circuit-breaker tracking.
            func: Synchronous callable to execute in a thread.
            track_circuit: When False, circuit-breaker success/failure tracking
                is skipped (caller handles per-ticker tracking itself).
        """
        today = _utc_today()
        if track_circuit and self._is_circuit_open(ticker, today):
            raise CircuitBreakerOpenError(
                f"Circuit breaker open for {ticker} — skipping for the day"
            )

        retried_func = _default_retry_policy(
            max_retries=self.max_retries,
            wait_init=self.retry_wait_init_s,
            wait_max=self.retry_wait_max_s,
        )(func)

        await self._acquire_slot()
        try:
            result = await asyncio.to_thread(retried_func, *args, **kwargs)
            if track_circuit:
                self._record_success(ticker)
            return result
        except Exception:
            if track_circuit:
                self._record_failure(ticker, today)
            raise
        finally:
            self._release_after_delay()

    # -- Batched OHLCV -----------------------------------------------------------

    def _split_batches(self, tickers: list[str]) -> list[list[str]]:
        """Split ticker list into batches of ≤ ``self.batch_size``."""
        return [tickers[i : i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]

    async def download_ohlcv(
        self,
        tickers: list[str],
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> pl.DataFrame:
        """Download OHLCV data for a list of tickers.

        Tickers are split into batches of ≤ ``batch_size``. Each batch is
        downloaded via ``yf.download(threads=False)`` with rate limiting and
        exponential-backoff retry.

        Circuit breaker state is tracked **per ticker** so that a healthy ticker
        is never blocked because it shared a batch with a failing ticker.

        Args:
            tickers: List of ticker symbols (e.g. ``["AAPL", "GOOGL"]``).
            start_date: Start date (inclusive), ISO string or ``date``.
            end_date: End date (inclusive). Defaults to today.

        Returns:
            polars DataFrame with columns: ticker, dt, open, high, low, close, volume.
        """
        today = _utc_today()

        # -- Normalize end_date to a date object -------------------------------
        if end_date is None:
            end_date = today
        elif isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        # yfinance treats ``end`` as exclusive, but our API contract says
        # inclusive.  Shift forward by 1 day to keep the contract.
        _end: str = (end_date + timedelta(days=1)).isoformat()

        if isinstance(start_date, date):
            start_date = start_date.isoformat()

        # -- Per-ticker circuit breaker filtering -------------------------------
        active_tickers: list[str] = []
        for t in tickers:
            if not self._is_circuit_open(t, today):
                active_tickers.append(t)
        skipped = len(tickers) - len(active_tickers)
        if skipped > 0:
            self._logger.warning("Skipping %d tickers with open circuit breakers", skipped)

        if not active_tickers:
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

        batches = self._split_batches(active_tickers)
        self._logger.info(
            "Downloading OHLCV for %d tickers in %d batches (batch_size=%d)",
            len(active_tickers),
            len(batches),
            self.batch_size,
        )

        results: list[pl.DataFrame] = []
        for batch in batches:
            # Use a real ticker (not a synthetic key) for rate-limiting.
            # Circuit-breaker tracking is handled per-ticker below so we
            # skip it inside _rate_limited_call.
            try:
                pd_df = await self._rate_limited_call(
                    batch[0],
                    _download_batch,
                    batch,
                    start_date,
                    _end,
                    track_circuit=False,
                )
                fallback = batch[0] if len(batch) == 1 else None
                pl_df = _ohlcv_to_polars(pd_df, fallback_ticker=fallback)
                if pl_df.height > 0:
                    results.append(pl_df)
                # Per-ticker tracking: only mark success for tickers that
                # actually appear in the returned data.
                present_tickers: set[str] = (
                    set(pl_df["ticker"].unique().to_list()) if pl_df.height > 0 else set()
                )
                for t in batch:
                    if t in present_tickers:
                        self._record_success(t)
                    else:
                        self._record_failure(t, today)
            except CircuitBreakerOpenError as e:
                self._logger.warning(
                    "OHLCV batch %s skipped (circuit breaker): %s",
                    batch[0],
                    e,
                )
            except Exception as e:
                self._logger.warning(
                    "OHLCV batch %s failed: %s",
                    batch[0],
                    e,
                )
                # Per-ticker failure tracking
                for t in batch:
                    self._record_failure(t, today)

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

    async def download_fundamentals(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Download fundamental data (Ticker.info) for a list of tickers.

        Each ticker is fetched individually with rate limiting, retry, and
        circuit breaker. Tickers whose breakers are open are silently skipped.

        Args:
            tickers: List of ticker symbols.

        Returns:
            List of dicts with key fundamental fields (symbol, sector, marketCap, etc.).
        """
        self._logger.info("Downloading fundamentals for %d tickers", len(tickers))
        results: list[dict[str, Any]] = []
        for ticker in tickers:
            try:
                info = await self._rate_limited_call(ticker, _fetch_ticker_info, ticker)
                results.append(_info_to_dict(info))
            except CircuitBreakerOpenError:
                pass  # silently skip tickers whose breaker is open
            except Exception as e:
                self._logger.warning("Fundamental fetch for %s failed: %s", ticker, e)
        return results

    async def download_earnings_dates(self, tickers: list[str]) -> pl.DataFrame:
        """Download earnings dates for a list of tickers.

        Each ticker is fetched individually with rate limiting, retry, and
        circuit breaker.

        Returns:
            polars DataFrame with columns: ticker, earnings_date, eps_estimate,
            reported_eps, surprise_pct.
        """
        self._logger.info("Downloading earnings dates for %d tickers", len(tickers))
        results: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                pd_df = await self._rate_limited_call(ticker, _fetch_earnings_dates, ticker)
                pl_df = _earnings_to_polars(pd_df, ticker)
                if pl_df.height > 0:
                    results.append(pl_df)
            except CircuitBreakerOpenError:
                pass  # silently skip tickers whose breaker is open
            except Exception as e:
                self._logger.warning("Earnings fetch for %s failed: %s", ticker, e)
        if not results:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "earnings_date": pl.Utf8,
                    "eps_estimate": pl.Float64,
                    "reported_eps": pl.Float64,
                    "surprise_pct": pl.Float64,
                }
            )
        return pl.concat(results)

    async def download_insider_transactions(self, tickers: list[str]) -> pl.DataFrame:
        """Download insider transactions for a list of tickers.

        Returns:
            polars DataFrame with columns: ticker, insider_name, title,
            transaction_type, shares, value, start_date.
        """
        self._logger.info("Downloading insider transactions for %d tickers", len(tickers))
        results: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                pd_df = await self._rate_limited_call(ticker, _fetch_insider_transactions, ticker)
                pl_df = _insider_to_polars(pd_df, ticker)
                if pl_df.height > 0:
                    results.append(pl_df)
            except CircuitBreakerOpenError:
                pass  # silently skip tickers whose breaker is open
            except Exception as e:
                self._logger.warning("Insider fetch for %s failed: %s", ticker, e)
        if not results:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "insider_name": pl.Utf8,
                    "title": pl.Utf8,
                    "transaction_type": pl.Utf8,
                    "shares": pl.Int64,
                    "value": pl.Float64,
                    "start_date": pl.Utf8,
                }
            )
        return pl.concat(results)

    async def download_news(self, tickers: list[str]) -> pl.DataFrame:
        """Download recent news for a list of tickers.

        Returns:
            polars DataFrame with columns: ticker, title, link, publisher,
            published_at, news_type.
        """
        self._logger.info("Downloading news for %d tickers", len(tickers))
        results: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                news_list = await self._rate_limited_call(ticker, _fetch_news, ticker)
                pl_df = _news_to_polars(news_list, ticker)
                if pl_df.height > 0:
                    results.append(pl_df)
            except CircuitBreakerOpenError:
                pass  # silently skip tickers whose breaker is open
            except Exception as e:
                self._logger.warning("News fetch for %s failed: %s", ticker, e)
        if not results:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "title": pl.Utf8,
                    "link": pl.Utf8,
                    "publisher": pl.Utf8,
                    "published_at": pl.Utf8,
                    "news_type": pl.Utf8,
                }
            )
        return pl.concat(results)

    # -- Bulk download -----------------------------------------------------------

    async def download_all(
        self,
        tickers: list[str],
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> dict[str, Any]:
        """Download all data (OHLCV + fundamentals + earnings + insiders + news).

        Args:
            tickers: List of ticker symbols.
            start_date: Start date (inclusive).
            end_date: End date (inclusive). Defaults to today.

        Returns:
            Dict with keys: ``ohlcv`` (pl.DataFrame), ``fundamentals`` (list[dict]),
            ``earnings_dates`` (pl.DataFrame), ``insider_transactions`` (pl.DataFrame),
            ``news`` (pl.DataFrame).
        """
        ohlcv, fundamentals, earnings, insiders, news = await asyncio.gather(
            self.download_ohlcv(tickers, start_date, end_date),
            self.download_fundamentals(tickers),
            self.download_earnings_dates(tickers),
            self.download_insider_transactions(tickers),
            self.download_news(tickers),
        )
        return {
            "ohlcv": ohlcv,
            "fundamentals": fundamentals,
            "earnings_dates": earnings,
            "insider_transactions": insiders,
            "news": news,
        }

    # -- Circuit breaker management ----------------------------------------------

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breaker state (for testing or fresh sessions)."""
        self._failures.clear()
        self._skip_until.clear()
        self._logger.info("All circuit breakers reset")

    @property
    def open_circuits(self) -> dict[str, date]:
        """Return a copy of currently open circuit breakers (ticker → skip_until date)."""
        return dict(self._skip_until)
