"""Download a broad US-equity universe and incrementally synchronize OHLCV."""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.request import Request, urlopen

import numpy as np
import polars as pl
import yfinance as yf

from alphascreener.data.io import scan_ohlcv, write_ohlcv
from alphascreener.prediction_contract import REQUIRED_HISTORY_SESSIONS

_logger = logging.getLogger(__name__)

_BATCH_SIZE = 50
MIN_SYNC_COVERAGE = 0.90
_SYMBOL_DIRECTORY_URLS = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
)
_NON_EQUITY_NAME_MARKERS = (
    " warrant",
    " right",
    " - unit",
    " unit ",
    " units",
    " preferred stock",
    " preferred shares",
    " preferred ",
    " preference share",
    " preference shares",
    " debt",
    " debenture",
    " notes due",
    " bond",
)


@dataclass(frozen=True)
class SyncResult:
    """Observable result of one synchronization attempt."""

    rows_written: int
    requested_tickers: int
    downloaded_tickers: int
    failed_tickers: tuple[str, ...]
    ready_tickers: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        """Return the fraction of requested tickers with usable rows."""
        if not self.requested_tickers:
            return 1.0
        return self.downloaded_tickers / self.requested_tickers


def _download_symbol_directory(url: str) -> str:
    request = Request(url, headers={"User-Agent": "alpha-screener/0.1"})
    with urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8")


def _parse_symbol_directory(contents: str) -> set[str]:
    """Extract ordinary US-listed equities and ADRs from a Nasdaq directory."""
    tickers: set[str] = set()
    for row in csv.DictReader(io.StringIO(contents), delimiter="|"):
        if row.get("Test Issue") != "N" or row.get("ETF") != "N":
            continue
        if row.get("NextShares") == "Y":
            continue
        name = f" {row.get('Security Name', '').lower()}"
        if any(marker in name for marker in _NON_EQUITY_NAME_MARKERS):
            continue
        if "$" in row.get("ACT Symbol", ""):
            continue
        ticker = row.get("NASDAQ Symbol") or row.get("Symbol") or ""
        ticker = ticker.strip().replace(".", "-")
        if ticker and not any(character in ticker for character in "^/"):
            tickers.add(ticker)
    return tickers


def _default_universe() -> list[str]:
    """Return US-listed equities from the official Nasdaq Trader directories."""
    tickers: set[str] = set()
    failures: list[str] = []
    for url in _SYMBOL_DIRECTORY_URLS:
        try:
            directory_tickers = _parse_symbol_directory(_download_symbol_directory(url))
            if not directory_tickers:
                raise ValueError("symbol directory contained no eligible equities")
            tickers.update(directory_tickers)
        except Exception as exc:
            _logger.warning("Could not fetch symbol directory %s: %s", url, exc)
            failures.append(url)
    if failures:
        raise RuntimeError(
            "complete US-equity universe unavailable; failed directories: "
            + ", ".join(failures)
        )
    tickers.add("SPY")
    return sorted(tickers)


def last_sync_date() -> date | None:
    """Return the most recent date in the local OHLCV store, or None."""
    try:
        frame = scan_ohlcv().select("dt").collect()
        return frame["dt"].max() if frame.height else None
    except Exception:
        return None


def sync_ohlcv(
    tickers: list[str] | None = None,
    *,
    start: date | None = None,
) -> SyncResult:
    """Checkpoint OHLCV batches and report decision-ready ticker coverage."""
    requested = tuple(dict.fromkeys(tickers if tickers is not None else _default_universe()))
    end = date.today()
    existing = _stored_history()
    stats = _history_stats(existing)
    if start is not None:
        plans = [(requested, start)]
    else:
        benchmark_date = _ticker_last_date(stats, "SPY")
        ready = _ready_tickers(stats, requested, benchmark_date)
        if (
            requested
            and benchmark_date is not None
            and (date.today() - benchmark_date).days <= 1
            and len(ready) / len(requested) >= MIN_SYNC_COVERAGE
        ):
            return _sync_result(0, requested, ready)
        backfill = tuple(
            ticker
            for ticker in requested
            if _ticker_sessions(stats, ticker) < REQUIRED_HISTORY_SESSIONS
        )
        refresh_all = benchmark_date is None or (date.today() - benchmark_date).days > 1
        refresh = tuple(
            ticker
            for ticker in requested
            if ticker not in backfill
            and (
                refresh_all
                or _ticker_last_date(stats, ticker) != benchmark_date
            )
        )
        plans = [
            (backfill, date.today() - timedelta(days=120)),
            (refresh, (benchmark_date or date.today()) - timedelta(days=7)),
        ]

    rows_written = 0
    for plan_tickers, plan_start in plans:
        if not plan_tickers:
            continue
        ordered = tuple(sorted(plan_tickers, key=lambda ticker: (ticker != "SPY", ticker)))
        _logger.info("Syncing %d tickers from %s to %s", len(ordered), plan_start, end)
        for offset in range(0, len(ordered), _BATCH_SIZE):
            batch = ordered[offset : offset + _BATCH_SIZE]
            batch_number = offset // _BATCH_SIZE + 1
            records = _download_batch(batch, start=plan_start, end=end, batch_number=batch_number)
            if records:
                write_ohlcv(pl.DataFrame(records))
                rows_written += len(records)

    stored = _stored_history()
    ready_stats = _history_stats(stored)
    benchmark_date = _coverage_date(ready_stats, requested)
    ready = _ready_tickers(ready_stats, requested, benchmark_date)
    result = _sync_result(rows_written, requested, ready)
    _logger.info(
        "Sync complete: %d rows, %.1f%% decision-ready coverage",
        result.rows_written,
        result.coverage * 100,
    )
    return result


def _sync_result(
    rows_written: int,
    requested: tuple[str, ...],
    ready: set[str],
) -> SyncResult:
    """Build a stable synchronization result from the ready universe."""
    return SyncResult(
        rows_written,
        len(requested),
        len(ready),
        tuple(sorted(set(requested) - ready)),
        tuple(sorted(ready)),
    )


def _download_batch(
    batch: tuple[str, ...],
    *,
    start: date,
    end: date,
    batch_number: int,
) -> list[dict[str, object]]:
    """Return validated records for one independently committable batch."""
    try:
        data = yf.download(
            list(batch),
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=4,
        )
    except Exception as exc:
        _logger.warning("Batch %d download failed: %s", batch_number, exc)
        return []
    if data.empty:
        return []

    batch_records: list[dict[str, object]] = []
    is_multi = hasattr(data.columns, "levels") and len(data.columns.levels) > 1
    for ticker in batch:
        try:
            if is_multi:
                if ticker not in data.columns.get_level_values(1):
                    continue
                ticker_data = data.xs(ticker, level=1, axis=1)
            elif len(batch) == 1:
                ticker_data = data.copy()
            else:
                continue
        except (KeyError, AttributeError):
            continue

        ticker_data = ticker_data.dropna(how="all")
        if ticker_data.empty:
            continue
        column_map: dict[object, str] = {}
        for column in ticker_data.columns:
            lowered = str(column).lower()
            for field in ("open", "high", "low", "close", "volume"):
                if field in lowered:
                    column_map[column] = field
                    break
        ticker_data = ticker_data.rename(columns=column_map)
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(ticker_data.columns):
            continue
        ticker_data = ticker_data[list(required_columns)]
        try:
            ticker_data = ticker_data.astype({column: "float64" for column in required_columns})
        except (TypeError, ValueError):
            continue
        finite_prices = np.isfinite(ticker_data[["open", "high", "low", "close"]]).all(axis=1)
        finite_volume = np.isfinite(ticker_data["volume"])
        positive_prices = (ticker_data[["open", "high", "low", "close"]] > 0).all(axis=1)
        non_negative_volume = ticker_data["volume"] >= 0
        ticker_data = ticker_data.loc[
            finite_prices & finite_volume & positive_prices & non_negative_volume
        ]
        for index, row in ticker_data.iterrows():
            batch_records.append({
                "ticker": ticker,
                "dt": index.date() if hasattr(index, "date") else index,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })
    return batch_records


def _stored_history() -> pl.DataFrame:
    try:
        return scan_ohlcv().collect()
    except FileNotFoundError:
        return pl.DataFrame(schema={"ticker": pl.String, "dt": pl.Date})


def _history_stats(history: pl.DataFrame) -> dict[str, tuple[int, date]]:
    if history.is_empty():
        return {}
    return {
        ticker: (sessions, last_date)
        for ticker, sessions, last_date in history.group_by("ticker").agg(
            pl.len().alias("sessions"), pl.col("dt").max().alias("last_date")
        ).iter_rows()
    }


def _ticker_sessions(stats: dict[str, tuple[int, date]], ticker: str) -> int:
    return stats.get(ticker, (0, date.min))[0]


def _ticker_last_date(stats: dict[str, tuple[int, date]], ticker: str) -> date | None:
    value = stats.get(ticker)
    return value[1] if value else None


def _ready_tickers(
    stats: dict[str, tuple[int, date]],
    requested: tuple[str, ...],
    benchmark_date: date | None,
) -> set[str]:
    if benchmark_date is None:
        return set()
    return {
        ticker
        for ticker in requested
        if _ticker_sessions(stats, ticker) >= REQUIRED_HISTORY_SESSIONS
        and _ticker_last_date(stats, ticker) == benchmark_date
    }


def _coverage_date(
    stats: dict[str, tuple[int, date]], requested: tuple[str, ...]
) -> date | None:
    spy_date = _ticker_last_date(stats, "SPY")
    if spy_date is not None:
        return spy_date
    dates = [_ticker_last_date(stats, ticker) for ticker in requested]
    return max((value for value in dates if value is not None), default=None)
