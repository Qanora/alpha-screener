"""Tests for the frozen future-14-session breakout prediction contract."""

from __future__ import annotations

import pytest

from alphascreener.prediction_contract import (
    DEFAULT_TOP_K,
    FORECAST_HORIZON_SESSIONS,
    INPUT_LOOKBACK_SESSIONS,
    ExplosionLabelSpec,
    assert_purged_time_split,
)


def test_contract_uses_60_sessions_to_predict_14_sessions() -> None:
    assert INPUT_LOOKBACK_SESSIONS == 60
    assert FORECAST_HORIZON_SESSIONS == 14
    assert DEFAULT_TOP_K == 10


def test_explosion_threshold_requires_absolute_and_cross_sectional_tail() -> None:
    spec = ExplosionLabelSpec(absolute_return=0.15, cross_section_quantile=0.8)

    assert spec.threshold([0.01, 0.02, 0.03, 0.04, 0.05]) == 0.15
    assert spec.threshold([0.01, 0.10, 0.20, 0.30, 0.40]) == 0.30


def test_explosion_hit_uses_same_date_cross_section() -> None:
    spec = ExplosionLabelSpec(absolute_return=0.10, cross_section_quantile=0.8)
    returns = [0.01, 0.05, 0.11, 0.15, 0.30]

    assert spec.is_hit(0.30, returns)
    assert not spec.is_hit(0.11, returns)


def test_purged_split_requires_a_full_14_session_gap() -> None:
    assert_purged_time_split(
        train_last_decision_session=100,
        test_first_decision_session=115,
    )

    with pytest.raises(ValueError, match="training label horizon"):
        assert_purged_time_split(
            train_last_decision_session=100,
            test_first_decision_session=114,
        )


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
