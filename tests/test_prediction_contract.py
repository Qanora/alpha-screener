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
    MIN_AVERAGE_DOLLAR_VOLUME,
    MIN_CANDIDATE_CLOSE,
    PREDICTION_HISTORY_SESSIONS,
    STRATEGY_VERSION,
    ExplosionLabelSpec,
)


def test_contract_uses_60_sessions_to_predict_14_sessions() -> None:
    assert INPUT_LOOKBACK_SESSIONS == 60
    assert FORECAST_HORIZON_SESSIONS == 14
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_BACKTEST_DAYS == 30
    assert MAX_BACKTEST_DAYS == 45
    assert MIN_CANDIDATE_CLOSE == 5.0
    assert MIN_AVERAGE_DOLLAR_VOLUME == 10_000_000.0
    assert MAX_CANDIDATES == 2_000
    assert PREDICTION_HISTORY_SESSIONS == 60
    assert BACKTEST_HISTORY_SESSIONS == 118
    assert STRATEGY_VERSION == "rank-v6"


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
