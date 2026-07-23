"""Tests for strict research labels, splits, and promotion statistics."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from alphascreener.market_calendar import market_dates_between
from alphascreener.research import (
    INCUMBENT_NAME,
    LIGHTGBM_NAME,
    BootstrapInterval,
    ResearchConfig,
    ResearchData,
    _complete_pool_thresholds,
    build_research_data,
    make_walk_forward_folds,
    paired_date_block_bootstrap,
    promotion_decision,
)


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2020, 1, 2), date(2027, 7, 23))
    assert len(dates) >= count
    return dates[:count]


def _ohlcv(sessions: int = 80) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    tickers = [("SPY", 1.001)] + [
        (f"T{index:02d}", 1.02 if index == 11 else 1.0 + index / 10_000.0)
        for index in range(1, 12)
    ]
    for ticker, growth in tickers:
        for index, market_date in enumerate(_market_dates(sessions)):
            raw_close = 100.0 * growth**index
            rows.append({
                "ticker": ticker,
                "dt": market_date,
                "open": raw_close * 0.995,
                "high": raw_close * 1.01,
                "low": raw_close * 0.99,
                "close": raw_close,
                "raw_close": raw_close,
                "volume": 2_000_000,
            })
    return pl.DataFrame(rows)


def test_research_data_invalidates_a_date_with_any_missing_pool_outcome() -> None:
    market_dates = _market_dates(80)
    data = _ohlcv()
    incomplete = data.filter(
        ~((pl.col("ticker") == "T11") & (pl.col("dt") == market_dates[-1]))
    )

    result = build_research_data(incomplete)
    latest = result.date_quality.filter(
        pl.col("decision_date") == market_dates[-15]
    ).row(0, named=True)

    assert latest["outcome_coverage"] == pytest.approx(10 / 11)
    assert latest["date_valid"] is False
    assert latest["invalid_reason"] == "complete_universe_outcomes_required"
    affected = result.panel.filter(pl.col("decision_date") == market_dates[-15])
    assert affected["hit_threshold"].null_count() == affected.height
    assert affected["is_explosion"].null_count() == affected.height


def test_research_threshold_uses_the_contracts_exact_nearest_rank() -> None:
    decision_date = date(2025, 1, 2)
    candidates = pl.DataFrame({
        "ticker": [f"T{index:02d}" for index in range(20)],
        "dt": [decision_date] * 20,
        "forward_return": [index / 100.0 for index in range(20)],
    })
    quality = pl.DataFrame({
        "dt": [decision_date],
        "universe_size": [20],
        "date_valid": [True],
    })

    threshold = _complete_pool_thresholds(candidates, quality)

    assert threshold.item(0, "hit_threshold") == pytest.approx(0.18)


def test_chunking_preserves_the_same_daily_panel_and_labels() -> None:
    data = _ohlcv(150)

    small_chunks = build_research_data(data.lazy(), chunk_decision_dates=7)
    one_chunk = build_research_data(data.lazy(), chunk_decision_dates=500)

    assert_frame_equal(small_chunks.date_quality, one_chunk.date_quality)
    assert_frame_equal(
        small_chunks.panel,
        one_chunk.panel,
        check_exact=False,
        rel_tol=1e-9,
        abs_tol=1e-11,
    )


def _fold_data(valid_count: int = 930) -> ResearchData:
    market_dates = _market_dates(valid_count + 14)
    valid_dates = tuple(market_dates[:valid_count])
    quality = pl.DataFrame({
        "decision_date": valid_dates,
        "result_date": market_dates[14 : valid_count + 14],
        "universe_size": [100] * valid_count,
        "outcome_count": [100] * valid_count,
        "outcome_coverage": [1.0] * valid_count,
        "date_valid": [True] * valid_count,
        "invalid_reason": [None] * valid_count,
    })
    return ResearchData(
        panel=pl.DataFrame(),
        date_quality=quality,
        valid_dates=valid_dates,
        snapshot_id="snapshot",
        feature_digest="features",
    )


def test_walk_forward_folds_only_use_labels_known_before_each_model_date() -> None:
    config = ResearchConfig(bootstrap_replications=100)

    folds = make_walk_forward_folds(_fold_data(), config)

    assert len(folds) == 4
    assert sum(len(fold.test_dates) for fold in folds) == 252
    for fold in folds:
        assert len(fold.training_dates) >= 504
        assert len(fold.validation_dates) == 126
        assert fold.maximum_training_result_date < fold.validation_dates[0]
        assert fold.maximum_training_result_date < fold.model_as_of
        assert max(fold.validation_dates) < fold.model_as_of


def _daily(strategy: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({
        "strategy": [strategy] * len(values),
        "decision_date": _market_dates(len(values)),
        "precision_at_10": values,
    })


def test_paired_block_bootstrap_is_deterministic_and_zero_for_identical_scores() -> None:
    values = [0.0, 0.1, 0.2] * 28

    first = paired_date_block_bootstrap(
        _daily(LIGHTGBM_NAME, values),
        _daily(INCUMBENT_NAME, values),
        block_dates=14,
        replications=500,
        seed=7,
    )
    second = paired_date_block_bootstrap(
        _daily(LIGHTGBM_NAME, values),
        _daily(INCUMBENT_NAME, values),
        block_dates=14,
        replications=500,
        seed=7,
    )

    assert first == second
    assert first.estimate == 0.0
    assert first.lower_95 == 0.0
    assert first.upper_95 == 0.0


def _summary(*, incumbent: float, challenger: float) -> pl.DataFrame:
    return pl.DataFrame({
        "strategy": [INCUMBENT_NAME, LIGHTGBM_NAME],
        "valid_dates": [252, 252],
        "hits_at_10": [round(incumbent * 2520), round(challenger * 2520)],
        "precision_at_10": [incumbent, challenger],
        "mean_base_explosion_rate": [0.05, 0.05],
        "passing_date_rate": [0.60, 0.61],
    })


def _stable_segments() -> pl.DataFrame:
    return pl.DataFrame({
        "segment": [
            "test_block_1",
            "test_block_2",
            "test_block_3",
            "test_block_4",
            "spy_20d_nonnegative",
            "spy_20d_negative",
        ],
        "dates": [63, 63, 63, 63, 180, 72],
        "precision_uplift": [0.01] * 6,
    })


def test_promotion_requires_both_two_points_and_a_positive_lower_bound() -> None:
    weak_interval = BootstrapInterval(0.03, 0.0, 0.06, 0.95, 252)
    strong_interval = BootstrapInterval(0.02, 0.001, 0.04, 0.98, 252)

    weak_passed, _ = promotion_decision(
        _summary(incumbent=0.10, challenger=0.13),
        weak_interval,
        recent_45_uplift=0.01,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments(),
    )
    small_passed, _ = promotion_decision(
        _summary(incumbent=0.10, challenger=0.119),
        strong_interval,
        recent_45_uplift=0.01,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments(),
    )
    strong_passed, reasons = promotion_decision(
        _summary(incumbent=0.10, challenger=0.12),
        strong_interval,
        recent_45_uplift=0.0,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments(),
    )
    unstable_passed, _ = promotion_decision(
        _summary(incumbent=0.10, challenger=0.12),
        strong_interval,
        recent_45_uplift=0.0,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments().with_columns(
            pl.when(pl.col("segment") == "test_block_1")
            .then(-0.001)
            .otherwise(pl.col("precision_uplift"))
            .alias("precision_uplift")
        ),
    )

    assert weak_passed is False
    assert small_passed is False
    assert strong_passed is True
    assert unstable_passed is False
    assert reasons == ("all preregistered gates passed",)
