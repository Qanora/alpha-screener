from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alphascreener.data.io import scan_ohlcv, write_ohlcv


def _ohlcv(tickers: list[str], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": tickers,
        "dt": [date(2025, 1, 2)] * len(tickers),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000] * len(tickers),
    })


def test_ohlcv_storage_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    data = _ohlcv(["AAPL"], [100.0])

    write_ohlcv(data)

    assert scan_ohlcv().collect().equals(data)


def test_ohlcv_storage_preserves_unaffected_tickers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    initial = _ohlcv(["AAPL", "MSFT"], [100.0, 200.0])
    update = _ohlcv(["AAPL"], [101.0])

    write_ohlcv(initial)
    write_ohlcv(update)

    stored = scan_ohlcv().collect().sort("ticker")
    assert stored["ticker"].to_list() == ["AAPL", "MSFT"]
    assert stored["close"].to_list() == [101.0, 200.0]


def test_ohlcv_storage_keeps_original_partition_when_replace_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    initial = _ohlcv(["AAPL"], [100.0])
    write_ohlcv(initial)

    def fail_replace(self, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    update = initial.with_columns(pl.lit(101.0).alias("close"))
    with pytest.raises(OSError, match="simulated replace failure"):
        write_ohlcv(update)

    assert scan_ohlcv().collect().equals(initial)


def test_ohlcv_storage_rejects_non_finite_prices(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)

    with pytest.raises(ValueError, match="finite positive prices"):
        write_ohlcv(_ohlcv(["BAD"], [float("nan")]))


def test_concurrent_ohlcv_writes_preserve_both_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    write_ohlcv(_ohlcv(["BASE"], [50.0]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write_ohlcv, [_ohlcv(["AAPL"], [100.0]), _ohlcv(["MSFT"], [200.0])]))

    stored = scan_ohlcv().collect().sort("ticker")
    assert stored["ticker"].to_list() == ["AAPL", "BASE", "MSFT"]
