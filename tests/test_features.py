"""Tests for low-degree features that need no more than 60 sessions."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alphascreener.features import compute_60d_features
from alphascreener.market_calendar import market_dates_between


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2025, 1, 2), date(2025, 12, 31))
    assert len(dates) >= count
    return dates[:count]


def _ohlcv() -> pl.DataFrame:
    rows = []
    market_dates = _market_dates(60)
    for ticker, growth, volume in [("SPY", 1.001, 1_000_000), ("WIN", 1.01, 2_000_000)]:
        for index, market_date in enumerate(market_dates):
            rows.append(
                {
                    "ticker": ticker,
                    "dt": market_date,
                    "close": 100.0 * growth**index,
                    "volume": volume + index * 1_000,
                }
            )
    return pl.DataFrame(rows)


def test_features_use_no_window_longer_than_60_sessions() -> None:
    data = _ohlcv()
    result = compute_60d_features(data)
    win = result.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == _market_dates(60)[-1]))

    assert win.item(0, "return_20d") > 0
    assert win.item(0, "distance_to_60d_high") == pytest.approx(0.0)
    assert win.item(0, "relative_strength_20d") > 0
    assert win.item(0, "volume_zscore_20") > 0
    assert win.item(0, "history_complete_60d") is True
    assert win.item(0, "valid_price_volume_sessions_prior_20d") == 20

    win_rows = data.filter(pl.col("ticker") == "WIN").sort("dt")
    expected_prior_median = (
        win_rows.slice(39, 20).select((pl.col("close") * pl.col("volume")).median()).item()
    )
    assert win.item(0, "median_dollar_volume_prior_20d") == pytest.approx(expected_prior_median)


def test_stable_liquidity_features_exclude_the_decision_session() -> None:
    data = _ohlcv()
    latest_date = _market_dates(60)[-1]
    before = compute_60d_features(data).filter(
        (pl.col("ticker") == "WIN") & (pl.col("dt") == latest_date)
    )
    after = compute_60d_features(
        data.with_columns(
            pl.when((pl.col("ticker") == "WIN") & (pl.col("dt") == latest_date))
            .then(pl.lit(1_000_000_000))
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
    ).filter((pl.col("ticker") == "WIN") & (pl.col("dt") == latest_date))

    assert after.item(0, "average_dollar_volume_20d") > before.item(0, "average_dollar_volume_20d")
    assert after.item(0, "median_dollar_volume_prior_20d") == before.item(
        0, "median_dollar_volume_prior_20d"
    )
    assert after.item(0, "valid_price_volume_sessions_prior_20d") == 20


def test_features_reject_missing_price_or_volume_data() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        compute_60d_features(pl.DataFrame({"ticker": ["AAPL"], "dt": [date(2025, 1, 1)]}))


def test_feature_history_requires_consecutive_spy_market_sessions() -> None:
    market_dates = _market_dates(61)
    rows = [
        {"ticker": "SPY", "dt": market_date, "close": 100.0, "volume": 1_000_000}
        for market_date in market_dates
    ]
    rows.extend(
        {"ticker": "GAP", "dt": market_date, "close": 20.0, "volume": 1_000_000}
        for index, market_date in enumerate(market_dates)
        if index != 30
    )

    result = compute_60d_features(pl.DataFrame(rows))
    latest = result.filter((pl.col("ticker") == "GAP") & (pl.col("dt") == market_dates[-1]))

    assert latest.item(0, "history_complete_60d") is False


def test_spy_history_is_incomplete_when_broad_market_exposes_a_gap() -> None:
    market_dates = _market_dates(61)
    rows = []
    for ticker in ("SPY", "WIN", "AUX"):
        rows.extend(
            {
                "ticker": ticker,
                "dt": market_date,
                "close": 100.0,
                "volume": 2_000_000,
            }
            for index, market_date in enumerate(market_dates)
            if ticker != "SPY" or index != 30
        )

    result = compute_60d_features(pl.DataFrame(rows))
    latest_date = market_dates[-1]

    assert (
        result.filter((pl.col("ticker") == "SPY") & (pl.col("dt") == latest_date)).item(
            0, "history_complete_60d"
        )
        is False
    )
    assert (
        result.filter((pl.col("ticker") == "WIN") & (pl.col("dt") == latest_date)).item(
            0, "history_complete_60d"
        )
        is True
    )
