"""Tests for the preregistered 60-session research feature set."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alphascreener.market_calendar import market_dates_between
from alphascreener.research_features import (
    LIGHTGBM_FEATURES,
    RESEARCH_MARKET_FEATURES,
    RESEARCH_STOCK_FEATURES,
    add_cross_sectional_ranks,
    compute_research_features,
    cross_sectional_feature_name,
)


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2024, 1, 2), date(2026, 12, 31))
    assert len(dates) >= count
    return dates[:count]


def _ohlcv(sessions: int = 75) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, growth, phase in [
        ("SPY", 1.001, 0.0),
        ("A", 1.004, 0.2),
        ("B", 0.999, 0.4),
    ]:
        for index, market_date in enumerate(_market_dates(sessions)):
            raw_close = 80.0 * growth**index + phase
            rows.append({
                "ticker": ticker,
                "dt": market_date,
                "open": raw_close * 0.995,
                "high": raw_close * 1.015,
                "low": raw_close * 0.985,
                "close": raw_close,
                "raw_close": raw_close,
                "volume": 2_000_000 + index * 1_000,
            })
    return pl.DataFrame(rows)


def test_research_features_are_finite_after_exactly_60_sessions() -> None:
    result = compute_research_features(_ohlcv(60))
    latest = result.filter(pl.col("dt") == _market_dates(60)[-1])

    assert latest.filter(pl.col("ticker") == "A").item(0, "return_59d") == pytest.approx(
        _ohlcv(60)
        .filter(pl.col("ticker") == "A")
        .sort("dt")["close"][-1]
        / _ohlcv(60).filter(pl.col("ticker") == "A").sort("dt")["close"][0]
        - 1.0
    )
    invalid = latest.select(
        pl.any_horizontal([
            pl.col(column).is_null() | ~pl.col(column).is_finite()
            for column in RESEARCH_STOCK_FEATURES
        ]).sum()
    ).item()
    assert invalid == 0


def test_observation_outside_the_60_session_window_cannot_change_features() -> None:
    data = _ohlcv(61)
    decision_date = _market_dates(61)[-1]
    changed = data.with_columns(
        pl.when((pl.col("ticker") == "A") & (pl.col("dt") == _market_dates(61)[0]))
        .then(pl.col("close") * 100.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    before = compute_research_features(data).filter(
        (pl.col("ticker") == "A") & (pl.col("dt") == decision_date)
    ).select(RESEARCH_STOCK_FEATURES)
    after = compute_research_features(changed).filter(
        (pl.col("ticker") == "A") & (pl.col("dt") == decision_date)
    ).select(RESEARCH_STOCK_FEATURES)

    for feature in RESEARCH_STOCK_FEATURES:
        assert before.item(0, feature) == pytest.approx(
            after.item(0, feature), rel=1e-10, abs=1e-12
        )


def test_cross_sectional_ranks_are_computed_within_the_filtered_date() -> None:
    features = pl.DataFrame({
        "ticker": ["A", "B", "C"],
        "dt": [date(2025, 1, 2)] * 3,
        **{
            feature: [1.0, 2.0, 3.0]
            for feature in RESEARCH_STOCK_FEATURES
        },
        **{feature: [0.1, 0.1, 0.1] for feature in RESEARCH_MARKET_FEATURES},
    })

    ranked = add_cross_sectional_ranks(features)

    column = cross_sectional_feature_name("return_1d")
    assert ranked[column].to_list() == [-1.0, 0.0, 1.0]
    assert set(LIGHTGBM_FEATURES).issubset(ranked.columns)
