from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest

from alphascreener.data.io import scan_ohlcv, write_ohlcv
from alphascreener.data.sync import _default_universe, _parse_symbol_directory, sync_ohlcv


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
        dates = pd.date_range("2025-01-01", periods=76, freq="D")
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


def test_fresh_ready_universe_skips_redundant_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    rows = []
    for ticker in ("SPY", "GOOD"):
        for offset in range(76):
            rows.append({
                "ticker": ticker,
                "dt": date.today() - timedelta(days=75 - offset),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 2_000,
            })
    write_ohlcv(pl.DataFrame(rows))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("a fresh decision-ready universe must not be downloaded again")

    monkeypatch.setattr("alphascreener.data.sync.yf.download", unexpected_download)

    result = sync_ohlcv(["SPY", "GOOD"])

    assert result.rows_written == 0
    assert result.coverage == 1.0
    assert result.ready_tickers == ("GOOD", "SPY")
