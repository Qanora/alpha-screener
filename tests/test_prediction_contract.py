"""Tests for the frozen future-14-session breakout prediction contract."""

from __future__ import annotations

import pytest

from alphascreener.prediction_contract import (
    BACKTEST_HISTORY_SESSIONS,
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_TOP_K,
    FORECAST_HORIZON_SESSIONS,
    INPUT_LOOKBACK_SESSIONS,
    MAX_BACKTEST_DAYS,
    MAX_CANDIDATES,
    MAX_RISK_RERANK_POSITIONS,
    MIN_AVERAGE_DOLLAR_VOLUME,
    MIN_CANDIDATE_CLOSE,
    MIN_MEDIAN_DOLLAR_VOLUME_PRIOR_20D,
    MIN_VALID_PRICE_VOLUME_SESSIONS_PRIOR_20D,
    PREDICTION_HISTORY_SESSIONS,
    RISK_RERANK_CANDIDATES,
    STRATEGY_VERSION,
    ExplosionLabelSpec,
    RiskLabelSpec,
)


def test_contract_uses_60_sessions_to_predict_14_sessions() -> None:
    assert INPUT_LOOKBACK_SESSIONS == 60
    assert FORECAST_HORIZON_SESSIONS == 14
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_BACKTEST_DAYS == 30
    assert MAX_BACKTEST_DAYS == 45
    assert MIN_CANDIDATE_CLOSE == 5.0
    assert MIN_AVERAGE_DOLLAR_VOLUME == 10_000_000.0
    assert MIN_MEDIAN_DOLLAR_VOLUME_PRIOR_20D == 5_000_000.0
    assert MIN_VALID_PRICE_VOLUME_SESSIONS_PRIOR_20D == 18
    assert MAX_CANDIDATES == 2_000
    assert RISK_RERANK_CANDIDATES == 30
    assert MAX_RISK_RERANK_POSITIONS == 3
    assert PREDICTION_HISTORY_SESSIONS == 60
    assert BACKTEST_HISTORY_SESSIONS == 118
    assert STRATEGY_VERSION == "rank-v7-guardrails"


def test_explosion_threshold_requires_absolute_and_cross_sectional_tail() -> None:
    spec = ExplosionLabelSpec(absolute_return=0.15, cross_section_quantile=0.8)

    assert spec.threshold([0.01, 0.02, 0.03, 0.04, 0.05]) == 0.15
    assert spec.threshold([0.01, 0.10, 0.20, 0.30, 0.40]) == 0.30


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_sessions": 0},
        {"absolute_return": 0.0},
        {"cross_section_quantile": 1.0},
    ],
)
def test_invalid_label_contract_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ExplosionLabelSpec(**kwargs)


def test_risk_contract_uses_asymmetric_downside_and_ten_percent_es() -> None:
    spec = RiskLabelSpec()

    assert spec.severe_return == -0.10
    assert spec.catastrophic_return == -0.20
    assert spec.adverse_path_return == -0.15
    assert spec.expected_shortfall_quantile == 0.10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_sessions": 0},
        {"severe_return": 0.0},
        {"catastrophic_return": -0.05},
        {"adverse_path_return": -0.25},
        {"expected_shortfall_quantile": 0.5},
    ],
)
def test_invalid_risk_contract_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RiskLabelSpec(**kwargs)
