from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alphascreener.data.io import scan_ohlcv, write_ohlcv


def test_ohlcv_storage_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    data = pl.DataFrame({
        "ticker": ["AAPL"], "dt": [date(2025, 1, 2)], "close": [100.0],
    })

    write_ohlcv(data)

    assert scan_ohlcv().collect().equals(data)


def test_ohlcv_storage_preserves_unaffected_tickers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    initial = pl.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "dt": [date(2025, 1, 2), date(2025, 1, 2)],
        "close": [100.0, 200.0],
    })
    update = pl.DataFrame({
        "ticker": ["AAPL"],
        "dt": [date(2025, 1, 2)],
        "close": [101.0],
    })

    write_ohlcv(initial)
    write_ohlcv(update)

    stored = scan_ohlcv().collect().sort("ticker")
    assert stored["ticker"].to_list() == ["AAPL", "MSFT"]
    assert stored["close"].to_list() == [101.0, 200.0]


def test_ohlcv_storage_keeps_original_partition_when_replace_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    initial = pl.DataFrame({
        "ticker": ["AAPL"], "dt": [date(2025, 1, 2)], "close": [100.0],
    })
    write_ohlcv(initial)

    def fail_replace(self, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    update = initial.with_columns(pl.lit(101.0).alias("close"))
    with pytest.raises(OSError, match="simulated replace failure"):
        write_ohlcv(update)

    assert scan_ohlcv().collect().equals(initial)
