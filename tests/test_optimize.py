"""Tests for factor weight optimization (grid search + TPE Bayesian)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from alphascreener.optimize import (
    OptimizeReport,
    _build_rolling_windows,
    _normalize_weights,
    _perturb_weights,
    optimize_weights,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic_ohlcv(
    n_days: int = 200,
    n_tickers: int = 5,
    start: date = date(2020, 1, 1),
    seed: int = 42,
) -> pl.DataFrame:
    """Build synthetic OHLCV data for testing."""
    import numpy as np

    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        for t_idx in range(n_tickers):
            ticker = f"TEST{t_idx}"
            base = 100.0 + t_idx * 10.0
            noise = rng.randn() * 2.0
            close = base + noise + i * 0.05  # slight upward drift
            rows.append(
                {
                    "dt": d,
                    "ticker": ticker,
                    "open": close - rng.random() * 0.5,
                    "high": close + rng.random() * 1.0,
                    "low": close - rng.random() * 1.0,
                    "close": close,
                    "volume": float(rng.randint(100000, 1000000)),
                }
            )
    return pl.DataFrame(rows)


# ── unit tests ───────────────────────────────────────────────────────────────


class TestNormalizeWeights:
    def test_normalize_sums_to_one(self):
        w = {"a": 0.3, "b": 0.3, "c": 0.4}
        n = _normalize_weights(w)
        assert abs(sum(n.values()) - 1.0) < 1e-9

    def test_normalize_arbitrary_values(self):
        w = {"a": 3.0, "b": 7.0}
        n = _normalize_weights(w)
        assert n["a"] == pytest.approx(0.3)
        assert n["b"] == pytest.approx(0.7)

    def test_normalize_zero_sum(self):
        w = {"a": 0.0, "b": 0.0}
        n = _normalize_weights(w)
        assert n == w

    def test_normalize_single_factor(self):
        w = {"x": 5.0}
        n = _normalize_weights(w)
        assert n["x"] == pytest.approx(1.0)


class TestBuildRollingWindows:
    def test_returns_correct_number(self):
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2025, 1, 1),
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=10,
        )
        assert len(ws) > 0
        assert len(ws) <= 10

    def test_window_order_monotonic(self):
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2025, 1, 1),
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=5,
        )
        for i in range(1, len(ws)):
            assert ws[i][0] > ws[i - 1][0]  # train_start is monotonic increasing

    def test_train_before_test(self):
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2025, 1, 1),
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=3,
        )
        for tr_s, tr_e, te_s, te_e in ws:
            assert tr_s < tr_e
            assert te_s == tr_e  # test starts where train ends
            assert te_s < te_e

    def test_adaptive_shortens_when_data_insufficient(self):
        """When data span is too short for train_years, fall back to shorter."""
        # Only 1 year of data, but request 2-year training + 6-month test
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2021, 1, 1),
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=10,
        )
        # Should still produce at least one window by shortening train_years
        assert len(ws) > 0, "Expected at least 1 window when train_years is adaptively shortened"

    def test_adaptive_returns_all_possible_windows(self):
        """Once a working train length is found, fill up to max_windows."""
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2024, 1, 1),  # 4 years span
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=50,
        )
        # With 4 years data, 2+0.5 years needed per window, should get several
        assert len(ws) > 1

    def test_adaptive_minimal_data_produces_one_window(self):
        """With just barely enough data, adaptive should find one window."""
        # ~14 months of data: train=0.5yr + test=6mo ≈ 12mo, barely fits
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2021, 3, 1),
            train_years=3,
            test_months=6,
            step_months=6,
            max_windows=10,
        )
        assert len(ws) >= 1

    def test_adaptive_too_little_data_returns_empty(self):
        """When even minimum training period can't produce a window, return []."""
        # Only 6 months of data — even 0.5yr train + 6mo test won't fit
        ws = _build_rolling_windows(
            date(2020, 1, 1),
            date(2020, 7, 1),
            train_years=2,
            test_months=6,
            step_months=6,
            max_windows=10,
        )
        assert len(ws) == 0


class TestPerturbWeights:
    def test_produces_variants(self):
        w = {"a": 0.5, "b": 0.3, "c": 0.2}
        variants = _perturb_weights(w, step_size=0.02)
        # 2 per factor (up/down)
        assert len(variants) == 6

    def test_variants_normalized(self):
        w = {"a": 0.5, "b": 0.3, "c": 0.2}
        variants = _perturb_weights(w, step_size=0.02)
        for name, vw in variants:
            assert abs(sum(vw.values()) - 1.0) < 1e-9

    def test_up_down_names(self):
        w = {"a": 0.5, "b": 0.5}
        variants = _perturb_weights(w, step_size=0.02)
        names = {n for n, _ in variants}
        assert "a+" in names
        assert "a-" in names
        assert "b+" in names
        assert "b-" in names


# ── integration tests ────────────────────────────────────────────────────────


class TestOptimizeWeightsGridSearch:
    """Tests for the grid search optimization strategy."""

    def test_returns_optimize_report(self):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.25, "rsi_oversold": 0.25, "vol_anomaly": 0.25, "rev_accel": 0.25}
        report = optimize_weights(df, weights, strategy="grid_search", max_windows=3)
        assert isinstance(report, OptimizeReport)
        assert report.initial_weights == weights
        assert len(report.final_weights) == len(weights)

    def test_final_weights_sum_to_one(self):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.25, "rsi_oversold": 0.25, "vol_anomaly": 0.25, "rev_accel": 0.25}
        report = optimize_weights(df, weights, strategy="grid_search", max_windows=3)
        assert abs(sum(report.final_weights.values()) - 1.0) < 1e-9


class TestOptimizeWeightsTPE:
    """Tests for the TPE Bayesian optimization strategy."""

    def test_returns_optimize_report(self):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.25, "rsi_oversold": 0.25, "vol_anomaly": 0.25, "rev_accel": 0.25}
        report = optimize_weights(df, weights, strategy="tpe", max_windows=3, n_trials=5)
        assert isinstance(report, OptimizeReport)
        assert report.initial_weights == weights
        assert len(report.final_weights) == len(weights)

    def test_final_weights_sum_to_one(self):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.25, "rsi_oversold": 0.25, "vol_anomaly": 0.25, "rev_accel": 0.25}
        report = optimize_weights(df, weights, strategy="tpe", max_windows=3, n_trials=5)
        assert abs(sum(report.final_weights.values()) - 1.0) < 1e-9

    def test_tpe_default_strategy(self):
        """TPE should be the default strategy."""
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.3, "rsi_oversold": 0.3, "vol_anomaly": 0.4}
        report = optimize_weights(df, weights, max_windows=3, n_trials=3)
        assert isinstance(report, OptimizeReport)
        # Should use TPE by default (no error)
        assert len(report.final_weights) == 3


class TestOptimizeWeightsErrors:
    """Edge case and error tests."""

    def test_invalid_strategy_raises(self):
        df = _make_synthetic_ohlcv(n_days=100, n_tickers=3)
        weights = {"mom_5d": 0.5, "rsi_oversold": 0.5}
        with pytest.raises(ValueError, match="strategy"):
            optimize_weights(df, weights, strategy="invalid_strategy")

    def test_empty_weights_handled(self):
        df = _make_synthetic_ohlcv(n_days=100, n_tickers=3)
        report = optimize_weights(df, {}, strategy="grid_search")
        assert report.final_weights == {}

    def test_tpe_with_few_windows(self):
        """TPE should work even with small data."""
        df = _make_synthetic_ohlcv(n_days=400, n_tickers=5)
        weights = {"mom_5d": 0.5, "rsi_oversold": 0.5}
        report = optimize_weights(df, weights, strategy="tpe", max_windows=2, n_trials=3)
        assert isinstance(report, OptimizeReport)
        assert len(report.final_weights) == 2
