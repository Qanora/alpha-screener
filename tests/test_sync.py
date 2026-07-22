from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest

from alphascreener.data.io import scan_ohlcv, write_ohlcv
from alphascreener.data.sync import (
    _default_universe,
    _download_batch,
    _parse_symbol_directory,
    sync_ohlcv,
)
from alphascreener.market_calendar import (
    latest_completed_market_date,
    market_dates_between,
)


def test_symbol_directory_keeps_equities_and_excludes_other_instruments() -> None:
    header = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares"
    )
    contents = "\n".join([
        header,
        "GOOD|Good Corporation - Common Stock|Q|N|N|100|N|N",
        "ADR|Global Company - American Depositary Shares|Q|N|N|100|N|N",
        "FUND|Example ETF|Q|N|N|100|Y|N",
        "WARR|Example Warrant|Q|N|N|100|N|N",
        "PREF|Example Depositary Shares representing Preference Shares|Q|N|N|100|N|N",
        "TEST|Test Security|Q|Y|N|100|N|N",
    ])

    assert _parse_symbol_directory(contents) == {"GOOD", "ADR"}


def test_default_universe_combines_official_directories_and_adds_spy(monkeypatch) -> None:
    nasdaq = "\n".join([
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares",
        "GOOD|Good Corporation - Common Stock|Q|N|N|100|N|N",
    ])
    other = "\n".join([
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol",
        "OTHER|Other Corporation Common Stock|N|OTHER|N|100|N|OTHER",
    ])
    payloads = {"nasdaqlisted.txt": nasdaq, "otherlisted.txt": other}
    monkeypatch.setattr(
        "alphascreener.data.sync._download_symbol_directory",
        lambda url: next(value for suffix, value in payloads.items() if url.endswith(suffix)),
    )

    assert _default_universe() == ["GOOD", "OTHER", "SPY"]


def test_otherlisted_parser_rejects_dollar_suffixed_securities() -> None:
    contents = "\n".join([
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol",
        "GOOD|Good Corporation Common Stock|N|GOOD|N|100|N|GOOD",
        "PREF$A|Issuer Depositary Shares|N|PREFpA|N|100|N|PREF-A",
    ])

    assert _parse_symbol_directory(contents) == {"GOOD"}


def test_default_universe_rejects_a_missing_directory(monkeypatch) -> None:
    nasdaq = "\n".join([
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares",
        "GOOD|Good Corporation - Common Stock|Q|N|N|100|N|N",
    ])

    def download(url):
        if url.endswith("otherlisted.txt"):
            raise OSError("directory unavailable")
        return nasdaq

    monkeypatch.setattr("alphascreener.data.sync._download_symbol_directory", download)

    with pytest.raises(RuntimeError, match="complete US-equity universe unavailable"):
        _default_universe()


def test_partial_sync_reports_coverage_and_preserves_failed_ticker_history(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    monkeypatch.setattr("alphascreener.data.sync._BATCH_SIZE", 1)
    existing = pl.DataFrame({
        "ticker": ["FAIL"],
        "dt": [date(2025, 1, 2)],
        "open": [20.0],
        "high": [20.0],
        "low": [20.0],
        "close": [20.0],
        "volume": [1000],
    })
    write_ohlcv(existing)

    def download(tickers, **kwargs):
        if tickers == ["FAIL"]:
            raise RuntimeError("simulated batch failure")
        dates = pd.to_datetime(
            market_dates_between(date(2025, 1, 2), date(2025, 6, 30))[:76]
        )
        return pd.DataFrame(
            {
                "Open": [10.0] * 76,
                "High": [11.0] * 76,
                "Low": [9.0] * 76,
                "Close": [10.5] * 76,
                "Volume": [2000] * 76,
            },
            index=dates,
        )

    monkeypatch.setattr("alphascreener.data.sync.yf.download", download)

    result = sync_ohlcv(["GOOD", "FAIL"], start=date(2025, 1, 1))

    assert result.coverage == 0.5
    assert result.failed_tickers == ("FAIL",)
    assert result.requested_symbols == ("FAIL", "GOOD")
    stored = scan_ohlcv().collect().sort("ticker")
    assert stored["ticker"].unique().sort().to_list() == ["FAIL", "GOOD"]
    assert stored.filter(pl.col("ticker") == "FAIL").item(0, "close") == 20.0


def test_sync_does_not_count_non_finite_rows_as_downloaded(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphascreener.data.sync.yf.download",
        lambda tickers, **kwargs: pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.0],
                "Close": [float("nan")],
                "Volume": [2_000],
            },
            index=pd.to_datetime(["2025-01-02"]),
        ),
    )

    result = sync_ohlcv(["BAD"], start=date(2025, 1, 1))

    assert result.downloaded_tickers == 0
    assert result.failed_tickers == ("BAD",)


def test_download_keeps_adjusted_and_raw_close_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphascreener.data.sync.yf.download",
        lambda tickers, **kwargs: pd.DataFrame(
            {
                "Open": [10.0],
                "High": [13.0],
                "Low": [9.0],
                "Close": [12.0],
                "Adj Close": [6.0],
                "Volume": [2_000],
            },
            index=pd.to_datetime(["2025-01-02"]),
        ),
    )

    row = _download_batch(
        ("SPLIT",),
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        batch_number=1,
    )[0]

    assert row["close"] == 6.0
    assert row["raw_close"] == 12.0


def test_network_batches_are_committed_in_bounded_groups_and_keep_successes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    monkeypatch.setattr("alphascreener.data.sync._BATCH_SIZE", 1)
    monkeypatch.setattr("alphascreener.data.sync._NETWORK_BATCHES_PER_CHECKPOINT", 2)
    network_calls = []

    def download(batch, *, start, end, batch_number):
        network_calls.append((batch, batch_number))
        ticker = batch[0]
        if ticker == "B":
            return []
        return [{
            "ticker": ticker,
            "dt": date(2025, 1, 2),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 2_000,
        }]

    checkpoints = []
    monkeypatch.setattr("alphascreener.data.sync._download_batch", download)
    monkeypatch.setattr(
        "alphascreener.data.sync.write_ohlcv", lambda frame: checkpoints.append(frame)
    )

    result = sync_ohlcv(["A", "B", "C", "D", "E"], start=date(2025, 1, 1))

    assert network_calls == [
        (("A",), 1),
        (("B",), 2),
        (("C",), 3),
        (("D",), 4),
        (("E",), 5),
    ]
    assert [frame["ticker"].to_list() for frame in checkpoints] == [
        ["A"],
        ["C", "D"],
        ["E"],
    ]
    assert result.rows_written == 4


def test_ledger_outcome_ticker_is_refreshed_but_excluded_from_current_coverage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    calls = []
    starts = []

    def download(batch, *, start, end, batch_number):
        calls.extend(batch)
        starts.append((batch, start))
        return [
            {
                "ticker": ticker,
                "dt": date.today(),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 2_000_000,
            }
            for ticker in batch
        ]

    monkeypatch.setattr("alphascreener.data.sync._download_batch", download)

    decision_date = date.today() - timedelta(days=400)
    result_date = decision_date + timedelta(days=21)
    result = sync_ohlcv(
        ["SPY", "CURRENT"],
        start=date.today(),
        outcome_requirements=(("DELISTED", decision_date, result_date),),
    )

    assert set(calls) == {"SPY", "CURRENT", "DELISTED"}
    assert result.requested_tickers == 2
    assert result.requested_symbols == ("CURRENT", "SPY")
    assert "DELISTED" not in result.ready_tickers
    assert (("DELISTED",), decision_date - timedelta(days=7)) in starts
    assert scan_ohlcv().collect()["ticker"].unique().sort().to_list() == [
        "CURRENT",
        "DELISTED",
        "SPY",
    ]


def test_download_end_includes_a_market_session_completed_today(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    monkeypatch.setattr(
        "alphascreener.data.sync.latest_completed_market_date",
        lambda: date.today(),
    )
    ends = []
    monkeypatch.setattr(
        "alphascreener.data.sync._download_batch",
        lambda batch, *, start, end, batch_number: ends.append(end) or [],
    )

    sync_ohlcv(["SPY"], start=date.today() - timedelta(days=7))

    assert ends == [date.today() + timedelta(days=1)]


def _stored_rows(ticker: str, sessions: int) -> list[dict[str, object]]:
    end = latest_completed_market_date()
    dates = market_dates_between(
        end - timedelta(days=sessions * 3),
        end,
    )[-sessions:]
    return [
        {
            "ticker": ticker,
            "dt": session_date,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 2_000,
        }
        for session_date in dates
    ]


def test_fresh_backtest_ready_universe_skips_redundant_downloads(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    rows = []
    for ticker in ("SPY", "GOOD"):
        rows.extend(_stored_rows(ticker, 118))
    write_ohlcv(pl.DataFrame(rows))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("a fresh decision-ready universe must not be downloaded again")

    monkeypatch.setattr("alphascreener.data.sync.yf.download", unexpected_download)

    result = sync_ohlcv(["SPY", "GOOD"])

    assert result.rows_written == 0
    assert result.coverage == 1.0
    assert result.ready_tickers == ("GOOD", "SPY")


def test_fresh_spy_gap_forces_a_repair_download_even_above_90pct_coverage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    tickers = ["SPY", *(f"M{index}" for index in range(9))]
    rows = [
        row
        for ticker in tickers
        for row in _stored_rows(ticker, 118)
    ]
    spy_dates = [row["dt"] for row in rows if row["ticker"] == "SPY"]
    missing_date = spy_dates[-30]
    rows = [
        row
        for row in rows
        if not (row["ticker"] == "SPY" and row["dt"] == missing_date)
    ]
    write_ohlcv(pl.DataFrame(rows))
    calls = []
    monkeypatch.setattr(
        "alphascreener.data.sync._download_batch",
        lambda batch, **kwargs: calls.append(batch) or [],
    )

    result = sync_ohlcv(tickers)

    assert calls == [("SPY",)]
    assert result.coverage == 0.9
    assert "SPY" not in result.ready_tickers


def test_fresh_prediction_ready_cache_is_expanded_for_backtests(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    write_ohlcv(
        pl.DataFrame([
            row
            for ticker in ("SPY", "GOOD")
            for row in _stored_rows(ticker, 76)
        ])
    )
    calls = []

    def observe_download(batch, *, start, end, batch_number):
        calls.append((batch, start, end, batch_number))
        return []

    monkeypatch.setattr("alphascreener.data.sync._download_batch", observe_download)

    result = sync_ohlcv(["SPY", "GOOD"])

    assert calls == [
        (
            ("SPY", "GOOD"),
            date.today() - timedelta(days=210),
            latest_completed_market_date() + timedelta(days=1),
            1,
        )
    ]
    assert result.coverage == 1.0
    assert result.ready_tickers == ("GOOD", "SPY")


def test_stale_complete_cache_remains_stale_when_refresh_downloads_fail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    stale_end = latest_completed_market_date() - timedelta(days=10)
    rows = []
    for ticker in ("SPY", "GOOD"):
        rows.extend(
            {
                **row,
                "dt": stale_end - timedelta(days=117 - index),
            }
            for index, row in enumerate(_stored_rows(ticker, 118))
        )
    write_ohlcv(pl.DataFrame(rows))
    calls = []
    monkeypatch.setattr(
        "alphascreener.data.sync._download_batch",
        lambda *args, **kwargs: calls.append(True) or [],
    )

    result = sync_ohlcv(["SPY", "GOOD"])

    assert calls
    assert result.coverage == 1.0
    assert result.as_of_date == stale_end
    assert result.is_fresh is False


def test_prediction_ready_ipo_is_not_excluded_by_backtest_history_requirement(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    mature_tickers = ["SPY", *[f"M{index}" for index in range(8)]]
    rows = [
        row
        for ticker in mature_tickers
        for row in _stored_rows(ticker, 118)
    ]
    rows.extend(_stored_rows("IPO", 60))
    write_ohlcv(pl.DataFrame(rows))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("90% backtest history should permit a fresh-cache fast path")

    monkeypatch.setattr("alphascreener.data.sync._download_batch", unexpected_download)

    result = sync_ohlcv([*mature_tickers, "IPO"])

    assert result.rows_written == 0
    assert result.coverage == 1.0
    assert "IPO" in result.ready_tickers
