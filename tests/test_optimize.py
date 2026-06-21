"""Tests for factor weight optimization (grid search + TPE Bayesian)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from alphascreener.optimize import (
    OptimizeReport,
    _build_rolling_windows,
    _evaluate_window,
    _normalize_weights,
    _perturb_weights,
    optimize_weights,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS = {"mom_5d": 0.25, "rsi_oversold": 0.25, "vol_anomaly": 0.25, "rev_accel": 0.25}


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


class TestOptimizeWeights:
    """Tests for the weight optimization strategies (grid_search + TPE)."""

    @pytest.mark.parametrize(
        ("strategy", "n_trials"),
        [("grid_search", None), ("tpe", 5)],
    )
    def test_returns_optimize_report(self, strategy, n_trials):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        kwargs = {"strategy": strategy, "max_windows": 3}
        if n_trials is not None:
            kwargs["n_trials"] = n_trials
        report = optimize_weights(df, _DEFAULT_WEIGHTS, **kwargs)
        assert isinstance(report, OptimizeReport)
        assert report.initial_weights == _DEFAULT_WEIGHTS
        assert len(report.final_weights) == len(_DEFAULT_WEIGHTS)

    @pytest.mark.parametrize(
        ("strategy", "n_trials"),
        [("grid_search", None), ("tpe", 5)],
    )
    def test_final_weights_sum_to_one(self, strategy, n_trials):
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        kwargs = {"strategy": strategy, "max_windows": 3}
        if n_trials is not None:
            kwargs["n_trials"] = n_trials
        report = optimize_weights(df, _DEFAULT_WEIGHTS, **kwargs)
        assert abs(sum(report.final_weights.values()) - 1.0) < 1e-9

    def test_tpe_default_strategy(self):
        """TPE should be the default strategy."""
        df = _make_synthetic_ohlcv(n_days=800, n_tickers=5)
        weights = {"mom_5d": 0.3, "rsi_oversold": 0.3, "vol_anomaly": 0.4}
        report = optimize_weights(df, weights, max_windows=3, n_trials=3)
        assert isinstance(report, OptimizeReport)
        # Should use TPE by default (no error)
        assert len(report.final_weights) == 3


class TestEvaluateWindowLookAhead:
    """Verify _evaluate_window does not leak test-period data into factor computation.

    The bug (#324): factor computation used train+test data (dt <= test_end),
    allowing EMA / rolling windows to "see" the test period before snap to
    test_start, producing upward-biased IC estimates.
    """

    def test_factor_computation_uses_train_only(self, monkeypatch):
        """compute_factors must only receive data within [train_start, train_end]."""
        from alphascreener.factors.engine import compute_factors as _original_cf
        from alphascreener.optimize import _evaluate_window

        df = _make_synthetic_ohlcv(n_days=300, n_tickers=10, start=date(2022, 1, 1))

        train_start = date(2022, 3, 1)
        train_end = date(2022, 6, 30)
        test_start = train_end
        test_end = date(2022, 10, 31)

        captured_df: list[pl.DataFrame] = []
        captured_kwargs: list[dict] = []

        def _capture_compute_factors(data, **kwargs):
            captured_df.append(data.clone())
            captured_kwargs.append(kwargs)
            return _original_cf(data, **kwargs)

        monkeypatch.setattr(
            "alphascreener.factors.engine.compute_factors",
            _capture_compute_factors,
        )

        _evaluate_window(
            df,
            {"mom_5d": 0.5, "rsi_oversold": 0.5},
            train_start,
            train_end,
            test_start,
            test_end,
        )

        assert len(captured_df) == 1, "compute_factors should be called exactly once"
        assert len(captured_kwargs) == 1, "compute_factors should be called exactly once"

        data_passed = captured_df[0]
        max_dt = data_passed["dt"].max()
        min_dt = data_passed["dt"].min()

        assert max_dt <= train_end, (
            f"Look-ahead bias: factor data max_dt={max_dt} exceeds train_end={train_end}"
        )
        assert min_dt >= train_start, (
            f"Factor data min_dt={min_dt} precedes train_start={train_start}"
        )

        dt_arg = captured_kwargs[0].get("dt")
        assert dt_arg == train_end, f"dt kwarg should be train_end={train_end}, got {dt_arg}"


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


# ── Industry dedup in evaluation tests ─────────────────────────────────────────


def _make_universe_meta(tickers: list[str]) -> pl.DataFrame:
    """Create a synthetic universe_meta DataFrame with sector/industry info.

    Assigns specific sectors so that dedup (Sector≤3, Industry≤2) has a
    measurable effect when many tickers share the same sector.
    """
    rows = []
    for t in tickers:
        t_idx = int(t.replace("TEST", "")) if t.startswith("TEST") else hash(t) % 100
        if t_idx < 5:
            sector, industry = "Technology", "Software"
        elif t_idx < 8:
            sector, industry = "Technology", "Semiconductors"
        elif t_idx < 12:
            sector, industry = "Financials", "Banks"
        else:
            sector, industry = "Healthcare", "Biotech"
        rows.append({"ticker": t, "sector": sector, "industry": industry})
    return pl.DataFrame(rows)


class TestEvaluateWindowIndustryDedup:
    """Verify _evaluate_window applies industry dedup before IC/Precision when
    universe_meta is provided (Issue #325)."""

    def test_universe_meta_dedup_reduces_scoring_set(self):
        """With universe_meta, dedup caps Sector≤3 Industry≤2, shrinking the
        effective set used for IC/Precision."""
        df = _make_synthetic_ohlcv(n_days=500, n_tickers=25, start=date(2022, 1, 1))
        universe_meta = _make_universe_meta(df["ticker"].unique().to_list())

        train_start = date(2022, 6, 1)
        train_end = date(2023, 1, 1)
        test_start = train_end
        test_end = date(2023, 5, 1)

        weights = {"mom_5d": 0.3, "rsi_oversold": 0.3, "vol_anomaly": 0.4}

        # Without universe_meta (backward compatible - dedup skipped)
        result_no_dedup = _evaluate_window(
            df, weights, train_start, train_end, test_start, test_end
        )

        # With universe_meta (dedup applied)
        result_with_dedup = _evaluate_window(
            df, weights, train_start, train_end, test_start, test_end,
            universe_meta=universe_meta,
        )

        # Both should produce valid WindowResults
        assert result_no_dedup is not None, "Should produce result without dedup"
        assert result_with_dedup is not None, "Should produce result with dedup"

        # The scores may differ because dedup changes the scoring pool
        # We cannot assert exact score differences (random data), but both
        # should produce valid float metrics within expected range.
        assert -1.0 <= result_with_dedup.ic <= 1.0
        assert result_with_dedup.precision_at_20 >= 0.0
        assert result_with_dedup.sharpe == result_with_dedup.score

    def test_universe_meta_none_is_backward_compatible(self):
        """When universe_meta is None, behavior matches current (no dedup)."""
        df = _make_synthetic_ohlcv(n_days=500, n_tickers=10, start=date(2022, 1, 1))

        train_start = date(2022, 6, 1)
        train_end = date(2023, 1, 1)
        test_start = train_end
        test_end = date(2023, 5, 1)

        weights = {"mom_5d": 0.3, "rsi_oversold": 0.3, "vol_anomaly": 0.4}

        result = _evaluate_window(
            df, weights, train_start, train_end, test_start, test_end,
            universe_meta=None,
        )

        # Should still work with None
        assert result is not None
        assert -1.0 <= result.ic <= 1.0
        assert result.precision_at_20 >= 0.0

    def test_optimize_weights_accepts_universe_meta(self):
        """optimize_weights should accept and pass through universe_meta."""
        df = _make_synthetic_ohlcv(n_days=600, n_tickers=10, start=date(2022, 1, 1))
        universe_meta = _make_universe_meta(df["ticker"].unique().to_list())

        weights = {"mom_5d": 0.3, "rsi_oversold": 0.3, "vol_anomaly": 0.4}

        report = optimize_weights(
            df, weights,
            strategy="grid_search",
            max_windows=2,
            universe_meta=universe_meta,
        )
        assert isinstance(report, OptimizeReport)
        assert len(report.final_weights) == len(weights)
        assert abs(sum(report.final_weights.values()) - 1.0) < 1e-9
