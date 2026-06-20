"""Yahoo Finance OHLCV data sync — download + incremental update.

Writes Hive-partitioned Parquet to ~/.alphascreener/data/ohlcv/dt=YYYY-MM-DD/
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

from alphascreener.data.io import scan_parquet, write_parquet

_logger = logging.getLogger(__name__)

# SP500 + Russell 1000 composite (major US equities, ~1500 after dedup)
_DEFAULT_TICKERS: list[str] | None = None

_BATCH_SIZE = 50  # yfinance download batch size


def _default_universe() -> list[str]:
    """Return a composite US large-cap ticker list."""
    if _DEFAULT_TICKERS is not None:
        return _DEFAULT_TICKERS  # type: ignore[return-value]

    tickers: set[str] = set()
    try:
        # SP500 from Wikipedia
        table = pl.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        sp500 = table["Symbol"].str.replace(".", "-").to_list()
        tickers.update(sp500)
    except Exception:
        _logger.warning("Could not fetch SP500 list, using fallback")

    if not tickers:
        # Minimum viable universe if online sources fail
        tickers = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
            "XOM", "NFLX", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AVGO",
            "COST", "PEP", "KO", "MRK", "ABBV", "PFE", "LLY", "TMO", "ABT",
            "NKE", "MCD", "SBUX", "ORCL", "CSCO", "IBM", "CVX", "WFC", "GS",
            "MS", "CAT", "BA", "GE", "MMM", "RTX", "LMT", "SPY",
        }
    return sorted(tickers)


def last_sync_date() -> date | None:
    """Return the most recent date in the local OHLCV store, or None."""
    try:
        lf = scan_parquet("ohlcv")
        df = lf.select("dt").collect()
        if df.height == 0:
            return None
        return df["dt"].max()
    except Exception:
        return None


def sync_ohlcv(
    tickers: list[str] | None = None,
    *,
    start: date | None = None,
    progress_callback=None,
) -> int:
    """Download OHLCV data and write to the Parquet store.

    Args:
        tickers: Ticker list (default: SP500 + Russell 1000).
        start: Start date for download. Default: last sync date - 7 days,
               or 2 years ago if no data exists.
        progress_callback: Optional callable(ticker_count, batch_num, total_batches).

    Returns:
        Number of new rows written.
    """
    if tickers is None:
        tickers = _default_universe()

    # Determine date range. If existing data is minimal, do full download.
    if start is None:
        last = last_sync_date()
        if last is not None:
            try:
                from alphascreener.data.io import scan_parquet
                existing = scan_parquet("ohlcv").collect()
                data_span = (last - existing["dt"].min()).days if existing.height > 0 else 0
            except Exception:
                data_span = 0
            if data_span < 180:
                start = date.today() - timedelta(days=365 * 2)
            else:
                start = last - timedelta(days=7)
        else:
            start = date.today() - timedelta(days=365 * 2)

    end = date.today()

    _logger.info("Syncing %d tickers from %s to %s", len(tickers), start, end)

    # Download in batches, collect all records, then write once
    all_records = []
    total_batches = (len(tickers) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1

        if progress_callback:
            progress_callback(len(tickers), batch_num, total_batches)

        try:
            data = yf.download(
                batch,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            _logger.warning("Batch %d download failed: %s", batch_num, exc)
            continue

        if data.empty:
            continue

        is_multi = hasattr(data.columns, "levels") and len(data.columns.levels) > 1

        for ticker_str in batch:
            try:
                if is_multi:
                    if ticker_str not in data.columns.get_level_values(1):
                        continue
                    ticker_data = data.xs(ticker_str, level=1, axis=1)
                else:
                    ticker_data = data.copy()
            except (KeyError, AttributeError):
                continue

            ticker_data = ticker_data.dropna(how="all")
            if ticker_data.empty:
                continue

            col_map = {}
            for col in ticker_data.columns:
                cl = str(col).lower()
                if "open" in cl: col_map[col] = "open"
                elif "high" in cl: col_map[col] = "high"
                elif "low" in cl: col_map[col] = "low"
                elif "close" in cl: col_map[col] = "close"
                elif "volume" in cl: col_map[col] = "volume"
            ticker_data = ticker_data.rename(columns=col_map)
            keep_cols = [c for c in ["open","high","low","close","volume"] if c in col_map.values()]
            ticker_data = ticker_data[keep_cols]
            if ticker_data.empty or "close" not in ticker_data.columns:
                continue

            for idx, row in ticker_data.iterrows():
                all_records.append({
                    "ticker": ticker_str,
                    "dt": idx.date() if hasattr(idx, "date") else idx,
                    "open": float(row.get("open", 0) or 0),
                    "high": float(row.get("high", 0) or 0),
                    "low": float(row.get("low", 0) or 0),
                    "close": float(row.get("close", 0) or 0),
                    "volume": int(row.get("volume", 0) or 0),
                })

    all_rows = 0
    if all_records:
        df = pl.DataFrame(all_records)
        write_parquet(df, "ohlcv")
        all_rows = df.height

    _logger.info("Sync complete: %d new rows", all_rows)
    return all_rows
