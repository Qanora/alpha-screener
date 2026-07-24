"""Download a broad US-equity universe and incrementally synchronize OHLCV."""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yfinance as yf

from alphascreener.data.io import scan_ohlcv, write_ohlcv
from alphascreener.data.universe import (
    parse_symbol_directories,
    save_universe_snapshot,
)
from alphascreener.market_calendar import (
    latest_completed_market_date,
    market_dates_between,
)
from alphascreener.prediction_contract import (
    BACKTEST_HISTORY_SESSIONS,
    PREDICTION_HISTORY_SESSIONS,
)

_logger = logging.getLogger(__name__)
_NEW_YORK = ZoneInfo("America/New_York")

_BATCH_SIZE = 50
_NETWORK_BATCHES_PER_CHECKPOINT = 20
_BACKFILL_CALENDAR_DAYS = 210
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
    requested_symbols: tuple[str, ...] = ()
    as_of_date: date | None = None
    is_fresh: bool = True

    @property
    def coverage(self) -> float:
        """Return the fraction of requested tickers with usable rows."""
        if not self.requested_tickers:
            return 1.0
        return self.downloaded_tickers / self.requested_tickers


def _download_symbol_directory(url: str) -> str:
    request = Request(url, headers={"User-Agent": "alpha-screener/0.2"})
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
    payloads: dict[str, str] = {}
    for url in _SYMBOL_DIRECTORY_URLS:
        try:
            payload = _download_symbol_directory(url)
            payloads[url.rsplit("/", 1)[-1].removesuffix(".txt")] = payload
            directory_tickers = _parse_symbol_directory(payload)
            if not directory_tickers:
                raise ValueError("symbol directory contained no eligible equities")
            tickers.update(directory_tickers)
        except Exception as exc:
            _logger.warning("Could not fetch symbol directory %s: %s", url, exc)
            failures.append(url)
    if failures:
        raise RuntimeError(
            "complete US-equity universe unavailable; failed directories: " + ", ".join(failures)
        )
    observed_at = datetime.now(UTC)
    normalized = parse_symbol_directories(
        payloads["nasdaqlisted"],
        payloads["otherlisted"],
        available_at=observed_at,
    )
    creation_dates = {
        value.astimezone(_NEW_YORK).date()
        for value in normalized["file_creation_time"].unique().to_list()
    }
    if len(creation_dates) != 1:
        raise RuntimeError(
            "complete US-equity universe unavailable; symbol directories "
            "have different creation dates"
        )
    save_universe_snapshot(
        payloads["nasdaqlisted"],
        payloads["otherlisted"],
        as_of=creation_dates.pop(),
        available_at=observed_at,
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
    outcome_requirements: tuple[tuple[str, date, date], ...] = (),
) -> SyncResult:
    """Checkpoint OHLCV and report coverage for the current official universe.

    ``outcome_requirements`` keeps exact immutable-ledger result dates
    recoverable, but never adds those symbols to today's official universe or
    its coverage denominator.
    """
    requested = tuple(dict.fromkeys(tickers if tickers is not None else _default_universe()))
    expected_market_date = latest_completed_market_date()
    outcome_work = tuple(
        requirement
        for requirement in set(outcome_requirements)
        if requirement[2] <= expected_market_date or requirement[0] not in requested
    )
    # yfinance treats ``end`` as exclusive.  After a same-calendar-day US
    # close, requesting through today would otherwise omit the required bar.
    end = expected_market_date + timedelta(days=1)
    recent_history_dates = market_dates_between(
        expected_market_date - timedelta(days=BACKTEST_HISTORY_SESSIONS * 3),
        expected_market_date,
    )[-BACKTEST_HISTORY_SESSIONS:]
    recent_history_start = recent_history_dates[0] if recent_history_dates else expected_market_date
    history_start = None if start is not None else recent_history_start
    existing = _stored_history(start=history_start).filter(pl.col("dt") <= expected_market_date)
    stats = _history_stats(existing)
    if start is not None:
        plans = [(requested, start)]
    else:
        benchmark_date = _ticker_last_date(stats, "SPY")
        ready = _ready_tickers(existing, requested, benchmark_date)
        history_ready = _ready_tickers(
            existing,
            requested,
            benchmark_date,
            required_sessions=BACKTEST_HISTORY_SESSIONS,
        )
        if (
            requested
            and benchmark_date is not None
            and benchmark_date == expected_market_date
            and "SPY" in ready
            and "SPY" in history_ready
            and len(ready) / len(requested) >= MIN_SYNC_COVERAGE
            and len(history_ready) / len(requested) >= MIN_SYNC_COVERAGE
            and not outcome_work
        ):
            return _sync_result(0, requested, ready, benchmark_date)
        backfill = tuple(ticker for ticker in requested if ticker not in history_ready)
        refresh_all = benchmark_date != expected_market_date
        refresh = tuple(
            ticker
            for ticker in requested
            if ticker not in backfill
            and (refresh_all or _ticker_last_date(stats, ticker) != benchmark_date)
        )
        plans = [
            (backfill, date.today() - timedelta(days=_BACKFILL_CALENDAR_DAYS)),
            (refresh, (benchmark_date or date.today()) - timedelta(days=7)),
        ]

    if outcome_work:
        outcome_start = min(requirement[1] for requirement in outcome_work) - timedelta(days=7)
        outcome_symbols = tuple(sorted({requirement[0] for requirement in outcome_work}))
        plans.append((outcome_symbols, outcome_start))

    rows_written = 0
    for plan_tickers, plan_start in plans:
        if not plan_tickers:
            continue
        ordered = tuple(sorted(plan_tickers, key=lambda ticker: (ticker != "SPY", ticker)))
        _logger.info("Syncing %d tickers from %s to %s", len(ordered), plan_start, end)
        checkpoint_records: list[dict[str, object]] = []
        for offset in range(0, len(ordered), _BATCH_SIZE):
            batch = ordered[offset : offset + _BATCH_SIZE]
            batch_number = offset // _BATCH_SIZE + 1
            records = _download_batch(batch, start=plan_start, end=end, batch_number=batch_number)
            checkpoint_records.extend(records)
            if batch_number % _NETWORK_BATCHES_PER_CHECKPOINT == 0:
                rows_written += _write_checkpoint(checkpoint_records)
                checkpoint_records = []
        rows_written += _write_checkpoint(checkpoint_records)

    stored = _stored_history(start=history_start).filter(pl.col("dt") <= expected_market_date)
    ready_stats = _history_stats(stored)
    benchmark_date = _coverage_date(ready_stats, requested)
    ready = _ready_tickers(stored, requested, benchmark_date)
    result = _sync_result(rows_written, requested, ready, benchmark_date)
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
    benchmark_date: date | None,
) -> SyncResult:
    """Build a stable synchronization result from the ready universe."""
    return SyncResult(
        rows_written,
        len(requested),
        len(ready),
        tuple(sorted(set(requested) - ready)),
        tuple(sorted(ready)),
        tuple(sorted(requested)),
        benchmark_date,
        benchmark_date == latest_completed_market_date(),
    )


def _download_batch(
    batch: tuple[str, ...],
    *,
    start: date,
    end: date,
    batch_number: int,
) -> list[dict[str, object]]:
    """Return validated records for one network batch."""
    try:
        data = yf.download(
            list(batch),
            start=str(start),
            end=str(end),
            auto_adjust=False,
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
            lowered = str(column).strip().lower().replace("_", " ")
            if "adj close" in lowered:
                column_map[column] = "close"
            elif lowered == "close":
                column_map[column] = "raw_close"
            elif lowered in {"open", "high", "low", "volume"}:
                column_map[column] = lowered
        ticker_data = ticker_data.rename(columns=column_map)
        if "close" not in ticker_data.columns and "raw_close" in ticker_data.columns:
            ticker_data["close"] = ticker_data["raw_close"]
        required_columns = ("open", "high", "low", "close", "raw_close", "volume")
        if not set(required_columns).issubset(ticker_data.columns):
            continue
        ticker_data = ticker_data[list(required_columns)]
        try:
            ticker_data = ticker_data.astype({column: "float64" for column in required_columns})
        except (TypeError, ValueError):
            continue
        finite_prices = np.isfinite(ticker_data[["open", "high", "low", "close", "raw_close"]]).all(
            axis=1
        )
        finite_volume = np.isfinite(ticker_data["volume"])
        positive_prices = (ticker_data[["open", "high", "low", "close", "raw_close"]] > 0).all(
            axis=1
        )
        non_negative_volume = ticker_data["volume"] >= 0
        ticker_data = ticker_data.loc[
            finite_prices & finite_volume & positive_prices & non_negative_volume
        ]
        for index, row in ticker_data.iterrows():
            batch_records.append(
                {
                    "ticker": ticker,
                    "dt": index.date() if hasattr(index, "date") else index,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "raw_close": float(row["raw_close"]),
                    "volume": int(row["volume"]),
                }
            )
    return batch_records


def _write_checkpoint(records: list[dict[str, object]]) -> int:
    """Commit accumulated successful downloads and return their row count."""
    if not records:
        return 0
    write_ohlcv(pl.DataFrame(records))
    return len(records)


def _stored_history(*, start: date | None = None) -> pl.DataFrame:
    try:
        history = scan_ohlcv()
        if start is not None:
            history = history.filter(pl.col("dt") >= start)
        return history.collect()
    except FileNotFoundError:
        return pl.DataFrame(schema={"ticker": pl.String, "dt": pl.Date})


def _history_stats(history: pl.DataFrame) -> dict[str, tuple[int, date]]:
    if history.is_empty():
        return {}
    return {
        ticker: (sessions, last_date)
        for ticker, sessions, last_date in history.group_by("ticker")
        .agg(pl.len().alias("sessions"), pl.col("dt").max().alias("last_date"))
        .iter_rows()
    }


def _ticker_last_date(stats: dict[str, tuple[int, date]], ticker: str) -> date | None:
    value = stats.get(ticker)
    return value[1] if value else None


def _ready_tickers(
    history: pl.DataFrame,
    requested: tuple[str, ...],
    benchmark_date: date | None,
    *,
    required_sessions: int = PREDICTION_HISTORY_SESSIONS,
) -> set[str]:
    if benchmark_date is None or history.is_empty() or not requested:
        return set()
    expected_dates = market_dates_between(
        benchmark_date - timedelta(days=required_sessions * 3),
        benchmark_date,
    )[-required_sessions:]
    if len(expected_dates) < required_sessions:
        return set()
    eligible = history.filter(
        pl.col("ticker").is_in(requested) & pl.col("dt").is_in(expected_dates)
    )
    if "raw_close" in eligible.columns:
        eligible = eligible.filter(pl.col("raw_close").is_not_null())
    return set(
        eligible.group_by("ticker")
        .agg(pl.col("dt").n_unique().alias("sessions"))
        .filter(pl.col("sessions") == required_sessions)["ticker"]
        .to_list()
    )


def _coverage_date(stats: dict[str, tuple[int, date]], requested: tuple[str, ...]) -> date | None:
    spy_date = _ticker_last_date(stats, "SPY")
    if spy_date is not None:
        return spy_date
    dates = [_ticker_last_date(stats, ticker) for ticker in requested]
    return max((value for value in dates if value is not None), default=None)
