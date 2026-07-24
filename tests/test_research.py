"""Tests for strict research labels, splits, and promotion statistics."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
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
    RidgeRanker,
    _complete_pool_thresholds,
    add_official_research_features,
    build_research_data,
    choose_boom_blend,
    choose_risk_lambda,
    date_equal_weights,
    evaluate_daily_scores,
    make_walk_forward_folds,
    paired_date_block_bootstrap,
    paired_tail_risk_block_bootstrap,
    promotion_decision,
    rank_blend_scores,
    risk_adjusted_scores,
    summarize_daily_scores,
    top_n_score_candidates,
)
from alphascreener.research_features import LINEAR_FEATURES


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2020, 1, 2), date(2027, 7, 23))
    assert len(dates) >= count
    return dates[:count]


def _ohlcv(sessions: int = 80) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    tickers = [("SPY", 1.001)] + [
        (f"T{index:02d}", 1.02 if index == 11 else 1.0 + index / 10_000.0) for index in range(1, 12)
    ]
    for ticker, growth in tickers:
        for index, market_date in enumerate(_market_dates(sessions)):
            raw_close = 100.0 * growth**index
            rows.append(
                {
                    "ticker": ticker,
                    "dt": market_date,
                    "open": raw_close * 0.995,
                    "high": raw_close * 1.01,
                    "low": raw_close * 0.99,
                    "close": raw_close,
                    "raw_close": raw_close,
                    "volume": 2_000_000,
                }
            )
    return pl.DataFrame(rows)


def test_research_data_invalidates_a_date_with_any_missing_pool_outcome() -> None:
    market_dates = _market_dates(80)
    data = _ohlcv()
    incomplete = data.filter(~((pl.col("ticker") == "T11") & (pl.col("dt") == market_dates[-1])))

    result = build_research_data(incomplete)
    latest = result.date_quality.filter(pl.col("decision_date") == market_dates[-15]).row(
        0, named=True
    )

    assert latest["outcome_coverage"] == pytest.approx(10 / 11)
    assert latest["date_valid"] is False
    assert latest["invalid_reason"] == "complete_universe_outcomes_required"
    affected = result.panel.filter(pl.col("decision_date") == market_dates[-15])
    assert affected["hit_threshold"].null_count() == affected.height
    assert affected["is_explosion"].null_count() == affected.height
    assert affected["is_severe_downside"].null_count() == affected.height
    assert affected["is_catastrophic_loss"].null_count() == affected.height
    assert affected["ranking_relevance"].null_count() == affected.height


def test_research_threshold_uses_the_contracts_exact_nearest_rank() -> None:
    decision_date = date(2025, 1, 2)
    candidates = pl.DataFrame(
        {
            "ticker": [f"T{index:02d}" for index in range(20)],
            "dt": [decision_date] * 20,
            "forward_return": [index / 100.0 for index in range(20)],
        }
    )
    quality = pl.DataFrame(
        {
            "dt": [decision_date],
            "universe_size": [20],
            "date_valid": [True],
        }
    )

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
    assert all(small_chunks.panel.schema[column] == pl.Float32 for column in LINEAR_FEATURES)


def test_ridge_ranker_matches_the_weighted_closed_form_in_batches(
    monkeypatch,
) -> None:
    first, second = _market_dates(2)
    labels = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    rows: dict[str, object] = {
        "decision_date": [first, first, first, second, second],
        "is_explosion": labels.astype(bool),
    }
    for feature_index, feature in enumerate(LINEAR_FEATURES):
        rows[feature] = np.linspace(-1.0, 1.0, len(labels)) + feature_index / 100.0
    frame = pl.DataFrame(rows)
    l2 = 2.0
    monkeypatch.setattr("alphascreener.research.RIDGE_FIT_BATCH_ROWS", 2)

    model = RidgeRanker(l2=l2)
    model.fit(frame)

    features = frame.select(LINEAR_FEATURES).to_numpy()
    design = np.column_stack([np.ones(frame.height), features])
    weights = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2])
    positive_weight = weights[labels == 1].sum()
    negative_weight = weights[labels == 0].sum()
    weights[labels == 1] *= 0.5 / positive_weight
    weights[labels == 0] *= 0.5 / negative_weight
    weights *= frame.height
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    expected_coefficients = np.linalg.solve(
        design.T @ (design * weights[:, None]) + penalty,
        design.T @ (weights * labels),
    )

    assert np.allclose(
        model.score(frame),
        design @ expected_coefficients,
        rtol=1e-10,
        atol=1e-10,
    )


def _fold_data(valid_count: int = 930) -> ResearchData:
    market_dates = _market_dates(valid_count + 14)
    valid_dates = tuple(market_dates[:valid_count])
    quality = pl.DataFrame(
        {
            "decision_date": valid_dates,
            "result_date": market_dates[14 : valid_count + 14],
            "universe_size": [100] * valid_count,
            "outcome_count": [100] * valid_count,
            "outcome_coverage": [1.0] * valid_count,
            "date_valid": [True] * valid_count,
            "invalid_reason": [None] * valid_count,
        }
    )
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


def test_official_feature_attachment_uses_panel_keys_and_volume() -> None:
    decision_date = date(2026, 7, 20)
    data = ResearchData(
        panel=pl.DataFrame(
            {
                "ticker": ["ABC"],
                "decision_date": [decision_date],
                "average_volume_20d": [1_000.0],
            }
        ),
        date_quality=pl.DataFrame(),
        valid_dates=(decision_date,),
        snapshot_id="ohlcv",
        feature_digest="base",
    )

    def loader(decisions, **kwargs):
        assert decisions.to_dicts() == [
            {
                "ticker": "ABC",
                "decision_date": decision_date,
                "average_volume_20d": 1_000.0,
            }
        ]
        assert kwargs["end"] == decision_date
        assert kwargs["average_daily_volume_column"] == "average_volume_20d"
        assert kwargs["shares_outstanding_column"] is None
        return SimpleNamespace(
            snapshot_digest="official",
            features=pl.DataFrame(
                {
                    "ticker": ["ABC"],
                    "decision_date": [decision_date],
                    "sec_coverage": ["complete"],
                    "filings_coverage": ["complete"],
                    "insider_coverage": ["complete"],
                    "current_report_count": [1],
                    "recent_offering_risk": [False],
                    "late_filing_risk": [False],
                    "new_13d": [False],
                    "form4_open_market_buy_usd": [0.0],
                    "form4_non_10b5_1_buy_usd": [0.0],
                    "distinct_insider_buyers": [0],
                    "cluster_buy": [False],
                    "short_interest_age_sessions": [2],
                    "short_interest": [100],
                    "short_interest_delta": [0.1],
                    "days_to_cover": [1.0],
                }
            ),
        )

    enriched = add_official_research_features(data, loader=loader)

    assert enriched.snapshot_id != data.snapshot_id
    assert enriched.extra_model_features
    assert enriched.external_coverage is not None


def test_risk_training_weights_balance_dates_without_changing_class_prior() -> None:
    first, second = _market_dates(2)
    frame = pl.DataFrame(
        {
            "decision_date": [first, first, first, second, second],
            "is_severe_downside": [False, False, True, False, True],
        }
    )

    weights = date_equal_weights(frame)
    weighted = frame.with_columns(pl.Series("weight", weights))

    assert weighted.filter(pl.col("decision_date") == first)["weight"].sum() == (
        pytest.approx(weighted.filter(pl.col("decision_date") == second)["weight"].sum())
    )
    assert weights.sum() == pytest.approx(frame.height)
    assert weights.mean() == pytest.approx(1.0)


def test_risk_training_candidates_are_only_each_boom_top_30() -> None:
    frame = _scored_validation(dates=2)
    ranks = [int(ticker.removeprefix("T")) for ticker in frame["ticker"].to_list()]

    selected = top_n_score_candidates(
        frame,
        -pl.Series(ranks, dtype=pl.Float64).to_numpy(),
        top_n=30,
    )

    assert selected.height == 60
    assert selected.group_by("decision_date").len()["len"].unique().to_list() == [30]
    assert set(selected["ticker"].unique().to_list()) == {f"T{rank:02d}" for rank in range(1, 31)}


def _daily(strategy: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": [strategy] * len(values),
            "decision_date": _market_dates(len(values)),
            "precision_at_10": values,
        }
    )


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
    return pl.DataFrame(
        {
            "strategy": [INCUMBENT_NAME, LIGHTGBM_NAME],
            "valid_dates": [252, 252],
            "hits_at_10": [round(incumbent * 2520), round(challenger * 2520)],
            "precision_at_10": [incumbent, challenger],
            "mean_base_explosion_rate": [0.05, 0.05],
            "passing_date_rate": [0.60, 0.61],
            "downside_at_10": [0.05, 0.04],
            "catastrophic_loss_at_10": [0.01, 0.01],
            "expected_shortfall_10": [-0.10, -0.09],
        }
    )


def _stable_segments() -> pl.DataFrame:
    return pl.DataFrame(
        {
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
            "downside_change": [-0.01] * 6,
            "catastrophic_loss_change": [0.0] * 6,
            "expected_shortfall_change": [0.01] * 6,
        }
    )


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


def _scored_validation(dates: int = 2, tickers: int = 40) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for decision_date in _market_dates(dates):
        for rank in range(1, tickers + 1):
            severe = rank == 10
            rows.append(
                {
                    "ticker": f"T{rank:02d}",
                    "decision_date": decision_date,
                    "result_date": decision_date,
                    "universe_size": tickers,
                    "forward_return": -0.12 if severe else (0.20 if rank == 1 else 0.0),
                    "is_explosion": rank == 1,
                    "is_severe_downside": severe,
                    "is_catastrophic_loss": False,
                    "spy_return_20d": 0.01,
                }
            )
    return pl.DataFrame(rows).sort(["decision_date", "ticker"])


def test_risk_overlay_only_reorders_the_original_boom_top_30() -> None:
    frame = _scored_validation(dates=1)
    boom_scores = -pl.Series(range(1, 41)).to_numpy()
    risk_scores = pl.Series(
        [100.0 if rank == 1 else float(rank) for rank in range(1, 41)]
    ).to_numpy()

    unchanged = risk_adjusted_scores(
        frame,
        boom_scores,
        risk_scores,
        risk_lambda=0,
    )
    adjusted = risk_adjusted_scores(
        frame,
        boom_scores,
        risk_scores,
        risk_lambda=3,
    )
    unchanged_order = (
        frame.with_columns(pl.Series("score", unchanged))
        .sort("score", descending=True)["ticker"]
        .to_list()
    )
    adjusted_order = (
        frame.with_columns(pl.Series("score", adjusted))
        .sort("score", descending=True)["ticker"]
        .to_list()
    )

    assert unchanged_order == [f"T{rank:02d}" for rank in range(1, 41)]
    assert adjusted_order[0] != "T01"
    assert set(adjusted_order[:30]) == {f"T{rank:02d}" for rank in range(1, 31)}
    assert adjusted_order[30:] == [f"T{rank:02d}" for rank in range(31, 41)]
    original_positions = {ticker: index for index, ticker in enumerate(unchanged_order, start=1)}
    adjusted_positions = {ticker: index for index, ticker in enumerate(adjusted_order, start=1)}
    assert (
        max(
            abs(adjusted_positions[ticker] - original_positions[ticker])
            for ticker in adjusted_order
        )
        <= 3
    )


def test_one_position_risk_budget_can_swap_the_top_10_boundary() -> None:
    frame = _scored_validation(dates=1)
    boom_scores = -pl.Series(range(1, 41), dtype=pl.Float64).to_numpy()
    risk_scores = pl.Series(
        [100.0 if rank == 10 else (-100.0 if rank == 11 else float(rank)) for rank in range(1, 41)],
        dtype=pl.Float64,
    ).to_numpy()

    adjusted = risk_adjusted_scores(
        frame,
        boom_scores,
        risk_scores,
        risk_lambda=1,
    )
    adjusted_order = (
        frame.with_columns(pl.Series("score", adjusted))
        .sort("score", descending=True)["ticker"]
        .to_list()
    )

    assert adjusted_order[9:11] == ["T11", "T10"]


def test_boom_blend_keeps_incumbent_when_challenger_adds_tail_risk() -> None:
    frame = _scored_validation(dates=10).with_columns(
        pl.Series(
            "rank_v6_score",
            [
                -float(int(ticker.removeprefix("T")))
                for ticker in _scored_validation(dates=10)["ticker"].to_list()
            ],
        )
    )
    challenger_scores = np.array(
        [
            100.0 if ticker == "T10" else -float(int(ticker.removeprefix("T")))
            for ticker in frame["ticker"].to_list()
        ]
    )

    weight, choice = choose_boom_blend(frame, challenger_scores)

    assert weight == 0.0
    assert choice["downside_at_10"] == pytest.approx(0.10)
    incumbent = rank_blend_scores(
        frame,
        frame["rank_v6_score"].to_numpy(),
        challenger_scores,
        challenger_weight=0.0,
    )
    assert np.isfinite(incumbent).all()


def test_validation_chooses_lowest_downside_with_near_best_precision() -> None:
    frame = _scored_validation()
    ranks = [int(ticker.removeprefix("T")) for ticker in frame["ticker"].to_list()]
    boom_scores = -pl.Series(ranks, dtype=pl.Float64).to_numpy()
    risk_scores = pl.Series([100.0 if rank == 10 else float(rank) for rank in ranks]).to_numpy()

    risk_lambda, choice = choose_risk_lambda(
        frame,
        boom_scores,
        risk_scores,
    )

    assert risk_lambda > 0
    assert choice["precision_at_10"] == pytest.approx(0.10)
    assert choice["downside_at_10"] == 0.0


def test_daily_metrics_include_downside_basket_return_and_es10() -> None:
    rows: list[dict[str, object]] = []
    for day_index, decision_date in enumerate(_market_dates(70)):
        forward_return = -0.20 if day_index < 7 else 0.10
        for rank in range(1, 11):
            rows.append(
                {
                    "ticker": f"T{rank:02d}",
                    "decision_date": decision_date,
                    "result_date": decision_date,
                    "universe_size": 10,
                    "forward_return": forward_return,
                    "is_explosion": forward_return >= 0.15,
                    "is_severe_downside": forward_return <= -0.10,
                    "is_catastrophic_loss": forward_return <= -0.20,
                    "spy_return_20d": 0.0,
                    "score": -float(rank),
                }
            )
    daily = evaluate_daily_scores(pl.DataFrame(rows), "score", "test")
    summary = summarize_daily_scores(daily).row(0, named=True)

    assert (
        daily.filter(pl.col("decision_date") == _market_dates(1)[0]).item(0, "downside_at_10")
        == 1.0
    )
    assert summary["downside_at_10"] == pytest.approx(0.10)
    assert summary["catastrophic_loss_at_10"] == pytest.approx(0.10)
    assert summary["mean_basket_return"] == pytest.approx(0.07)
    assert summary["expected_shortfall_10"] == pytest.approx(-0.20)

    tail = paired_tail_risk_block_bootstrap(
        daily,
        daily,
        block_dates=14,
        replications=100,
        seed=7,
    )
    assert tail.downside_change.estimate == 0.0
    assert tail.catastrophic_loss_change.estimate == 0.0
    assert tail.expected_shortfall_change.estimate == 0.0


def test_promotion_rejects_worse_tail_risk() -> None:
    interval = BootstrapInterval(0.02, 0.001, 0.04, 0.98, 252)
    worse_risk = _summary(incumbent=0.10, challenger=0.12).with_columns(
        pl.when(pl.col("strategy") == LIGHTGBM_NAME)
        .then(0.051)
        .otherwise(pl.col("downside_at_10"))
        .alias("downside_at_10")
    )

    passed, reasons = promotion_decision(
        worse_risk,
        interval,
        recent_45_uplift=0.0,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments(),
    )

    assert passed is False
    assert any("Downside@10 increased" in reason for reason in reasons)


def test_promotion_requires_a_material_tail_risk_improvement() -> None:
    interval = BootstrapInterval(0.02, 0.001, 0.04, 0.98, 252)
    immaterial = _summary(incumbent=0.10, challenger=0.12).with_columns(
        pl.when(pl.col("strategy") == LIGHTGBM_NAME)
        .then(0.049)
        .otherwise(pl.col("downside_at_10"))
        .alias("downside_at_10"),
        pl.when(pl.col("strategy") == LIGHTGBM_NAME)
        .then(-0.099)
        .otherwise(pl.col("expected_shortfall_10"))
        .alias("expected_shortfall_10"),
    )

    passed, reasons = promotion_decision(
        immaterial,
        interval,
        recent_45_uplift=0.0,
        minimum_uplift=0.02,
        required_dates=252,
        stability=_stable_segments(),
    )

    assert passed is False
    assert any("no material tail-risk improvement" in reason for reason in reasons)
