"""Tests for low-degree features that need no more than 60 sessions."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from alphascreener.features import compute_60d_features


def _ohlcv() -> pl.DataFrame:
    rows = []
    for ticker, growth, volume in [("SPY", 1.001, 1_000_000), ("WIN", 1.01, 2_000_000)]:
        for index in range(60):
            rows.append({"ticker": ticker, "dt": date(2025, 1, 1) + timedelta(days=index),
                         "close": 100.0 * growth**index, "volume": volume + index * 1_000})
    return pl.DataFrame(rows)


def test_features_use_no_window_longer_than_60_sessions() -> None:
    result = compute_60d_features(_ohlcv())
    win = result.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == date(2025, 3, 1)))

    assert win.item(0, "return_20d") > 0
    assert win.item(0, "distance_to_60d_high") == pytest.approx(0.0)
    assert win.item(0, "relative_strength_20d") > 0
    assert win.item(0, "volume_zscore_20") > 0


def test_features_reject_missing_price_or_volume_data() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        compute_60d_features(pl.DataFrame({"ticker": ["AAPL"], "dt": [date(2025, 1, 1)]}))
