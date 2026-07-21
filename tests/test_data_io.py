from __future__ import annotations

from datetime import date

import polars as pl

from alphascreener.data.io import scan_ohlcv, write_ohlcv


def test_ohlcv_storage_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: tmp_path)
    data = pl.DataFrame({
        "ticker": ["AAPL"], "dt": [date(2025, 1, 2)], "close": [100.0],
    })

    write_ohlcv(data)

    assert scan_ohlcv().collect().equals(data)
