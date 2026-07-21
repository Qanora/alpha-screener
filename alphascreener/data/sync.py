"""Download a broad US-equity universe and incrementally synchronize OHLCV."""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.request import Request, urlopen

import polars as pl
import yfinance as yf

from alphascreener.data.io import scan_ohlcv, write_ohlcv

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
    " debt",
    " notes due",
    " bond",
)
_FALLBACK_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
    "XOM", "NFLX", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AVGO",
    "COST", "PEP", "KO", "MRK", "ABBV", "PFE", "LLY", "TMO", "ABT",
    "NKE", "MCD", "SBUX", "ORCL", "CSCO", "IBM", "CVX", "WFC", "GS",
    "MS", "CAT", "BA", "GE", "MMM", "RTX", "LMT", "SPY",
}


@dataclass(frozen=True)
class SyncResult:
    """Observable result of one synchronization attempt."""

    rows_written: int
    requested_tickers: int
    downloaded_tickers: int
    failed_tickers: tuple[str, ...]

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
        ticker = row.get("NASDAQ Symbol") or row.get("Symbol") or ""
        ticker = ticker.strip().replace(".", "-")
        if ticker and not any(character in ticker for character in "^/"):
            tickers.add(ticker)
    return tickers


def _default_universe() -> list[str]:
    """Return US-listed equities from the official Nasdaq Trader directories."""
    tickers: set[str] = set()
    for url in _SYMBOL_DIRECTORY_URLS:
        try:
            tickers.update(_parse_symbol_directory(_download_symbol_directory(url)))
        except Exception as exc:
            _logger.warning("Could not fetch symbol directory %s: %s", url, exc)
    if not tickers:
        _logger.warning("No symbol directory was available; using the fallback universe")
        tickers.update(_FALLBACK_TICKERS)
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
    """Download OHLCV rows, merge them safely, and report ticker coverage."""
    requested = tuple(dict.fromkeys(tickers if tickers is not None else _default_universe()))
    if start is None:
        last = last_sync_date()
        if last is not None:
            try:
                existing = scan_ohlcv().collect()
                data_span = (last - existing["dt"].min()).days if existing.height else 0
            except Exception:
                data_span = 0
            start = (
                date.today() - timedelta(days=120)
                if data_span < 90
                else last - timedelta(days=7)
            )
        else:
            start = date.today() - timedelta(days=120)

    end = date.today()
    _logger.info("Syncing %d tickers from %s to %s", len(requested), start, end)

    all_records: list[dict[str, object]] = []
    downloaded: set[str] = set()
    for offset in range(0, len(requested), _BATCH_SIZE):
        batch = requested[offset : offset + _BATCH_SIZE]
        batch_number = offset // _BATCH_SIZE + 1
        try:
            data = yf.download(
                list(batch),
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            _logger.warning("Batch %d download failed: %s", batch_number, exc)
            continue
        if data.empty:
            continue

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
            keep_columns = [
                field for field in ("open", "high", "low", "close", "volume")
                if field in column_map.values()
            ]
            ticker_data = ticker_data[keep_columns]
            if ticker_data.empty or "close" not in ticker_data.columns:
                continue

            ticker_rows = []
            for index, row in ticker_data.iterrows():
                ticker_rows.append({
                    "ticker": ticker,
                    "dt": index.date() if hasattr(index, "date") else index,
                    "open": float(row.get("open", 0) or 0),
                    "high": float(row.get("high", 0) or 0),
                    "low": float(row.get("low", 0) or 0),
                    "close": float(row.get("close", 0) or 0),
                    "volume": int(row.get("volume", 0) or 0),
                })
            if ticker_rows:
                downloaded.add(ticker)
                all_records.extend(ticker_rows)

    if all_records:
        write_ohlcv(pl.DataFrame(all_records))
    failed = tuple(sorted(set(requested) - downloaded))
    result = SyncResult(len(all_records), len(requested), len(downloaded), failed)
    _logger.info(
        "Sync complete: %d rows, %.1f%% ticker coverage",
        result.rows_written,
        result.coverage * 100,
    )
    return result
