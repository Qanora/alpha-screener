"""Tests for alpha acceptance metrics computation (Issue #101).

Reference: PRD 5.5.

Covers:
  - base_rate: fraction of full-market tickers with T+7 return >= 10%
  - Precision@K (K=10,20): fraction of top K tickers by score that are hits
  - Lift@K = Precision@K / base_rate
  - Recall@K: top K hits / total market hits
  - IC: Spearman rank correlation between scores and returns
  - Block-bootstrap 95% CI: block size=5 trading days x 1000 resamples
  - write_alpha_acceptance: persist metrics to alpha_acceptance_daily table
  - Edge cases: empty DataFrame, zero base_rate, NaN/inf values, single ticker
"""

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sqlalchemy import create_engine

from alphascreener.db.models import AlphaAcceptanceDaily, Base

# ============================================================================
# Helpers -- build test DataFrames
# ============================================================================


def _make_alpha_df(
    tickers: list[str],
    breakout_scores: list[float],
    refined_scores: list[float | None],
    t7_returns: list[float],
) -> pl.DataFrame:
    """Build a minimal DataFrame for alpha acceptance testing.

    Columns: ticker, breakout_score, refined_score, t7_return.
    """
    return pl.DataFrame(
        {
            "ticker": tickers,
            "breakout_score": breakout_scores,
            "refined_score": refined_scores,
            "t7_return": t7_returns,
        }
    )


def _make_random_alpha_df(
    n_tickers: int = 100,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic alpha data with roughly 10% hit rate."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:04d}" for i in range(n_tickers)]
    breakout_scores = list(rng.normal(0.0, 1.0, n_tickers).astype(float))
    refined_scores = list(rng.normal(0.1, 0.9, n_tickers).astype(float))
    # T+7 returns: ~10% hit rate (>= 0.10 threshold)
    t7_returns = list(rng.normal(0.02, 0.12, n_tickers).astype(float))
    return pl.DataFrame(
        {
            "ticker": tickers,
            "breakout_score": breakout_scores,
            "refined_score": refined_scores,
            "t7_return": t7_returns,
        }
    )


# ============================================================================
# 1. base_rate tests
# ============================================================================


class TestBaseRate:
    """Compute base_rate = fraction of tickers with T+7 return >= 10%."""

    def test_base_rate_normal(self):
        """10 tickers, 2 hits => base_rate = 0.2."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series(
            "t7_return",
            [0.05, 0.12, 0.10, 0.08, -0.02, 0.15, 0.03, 0.00, 0.20, -0.10],
        )
        result = compute_base_rate(returns)
        # hits: 0.12, 0.10, 0.15, 0.20 => 4 out of 10
        assert result == pytest.approx(0.4)

    def test_base_rate_no_hits(self):
        """No ticker reaches 10% => base_rate = 0.0."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [0.05, 0.08, 0.03, -0.02])
        result = compute_base_rate(returns)
        assert result == 0.0

    def test_base_rate_all_hits(self):
        """All tickers reach 10% => base_rate = 1.0."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [0.10, 0.15, 0.20, 0.11])
        result = compute_base_rate(returns)
        assert result == 1.0

    def test_base_rate_empty(self):
        """Empty returns => base_rate = 0.0."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [], dtype=pl.Float64)
        result = compute_base_rate(returns)
        assert result == 0.0

    def test_base_rate_with_nulls(self):
        """Null returns are excluded from denominator."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [0.15, None, 0.05, None, 0.12])
        result = compute_base_rate(returns)
        # hits: 0.15, 0.12 => 2 out of 3 (nulls excluded)
        assert result == pytest.approx(2.0 / 3.0)

    def test_base_rate_boundary_at_10_pct(self):
        """T+7 return == 0.10 is a hit (>= threshold)."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [0.10, 0.0999])
        result = compute_base_rate(returns)
        assert result == 0.5

    def test_base_rate_negative_returns(self):
        """Very negative returns still count as non-hits."""
        from alphascreener.alpha_acceptance import compute_base_rate

        returns = pl.Series("t7_return", [-0.50, -0.30, -0.10])
        result = compute_base_rate(returns)
        assert result == 0.0


# ============================================================================
# 2. Precision@K tests
# ============================================================================


class TestPrecisionAtK:
    """Compute Precision@K = fraction of top K tickers by score that are hits."""

    def test_precision_at_10_normal(self):
        """10 tickers in top 10, 4 hits => precision = 0.4."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        # 10 tickers total, K=10 means all tickers
        hits = pl.Series("hits", [True, True, True, True, False, False, False, False, False, False])
        scores = pl.Series("score", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        result = compute_precision_at_k(scores, hits, k=10)
        assert result == pytest.approx(0.4)

    def test_precision_at_20_perfect(self):
        """Top 20 all hits => precision = 1.0."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        hits_vals = [True] * 20 + [False] * 80
        scores = list(range(100, 0, -1))
        hits = pl.Series("hits", hits_vals)
        scores_series = pl.Series("score", scores, dtype=pl.Float64)
        result = compute_precision_at_k(scores_series, hits, k=20)
        assert result == 1.0

    def test_precision_at_10_zero(self):
        """Top 10 are all non-hits => precision = 0.0."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        hits = pl.Series("hits", [False] * 10 + [True] * 90)
        scores = pl.Series("score", list(range(100, 0, -1)), dtype=pl.Float64)
        result = compute_precision_at_k(scores, hits, k=10)
        assert result == 0.0

    def test_precision_at_k_k_larger_than_n(self):
        """K larger than total tickers => uses all tickers (K = n)."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        hits = pl.Series("hits", [True, False, True])
        scores = pl.Series("score", [0.9, 0.5, 0.8], dtype=pl.Float64)
        result = compute_precision_at_k(scores, hits, k=100)
        # Top K effectively = 3
        assert result == pytest.approx(2.0 / 3.0)

    def test_precision_at_k_empty(self):
        """Empty input => precision = 0.0."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        hits = pl.Series("hits", [], dtype=pl.Boolean)
        scores = pl.Series("score", [], dtype=pl.Float64)
        result = compute_precision_at_k(scores, hits, k=20)
        assert result == 0.0

    def test_precision_at_k_tie_breaking(self):
        """Tied scores: all tied tickers included (conservative)."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        # 15 tickers with same score, k=10, 9 hits total
        hits = pl.Series("hits", [True] * 9 + [False] * 6)
        scores = pl.Series("score", [1.0] * 15, dtype=pl.Float64)
        result = compute_precision_at_k(scores, hits, k=10)
        # With ties, we take all tied tickers at the boundary
        # So effectively selects all 15, precision = 9/15
        assert result == pytest.approx(9.0 / 15.0)

    def test_precision_at_k_with_nulls(self):
        """Null scores are treated as lowest rank."""
        from alphascreener.alpha_acceptance import compute_precision_at_k

        hits = pl.Series("hits", [True, False, True, True, False])
        scores = pl.Series("score", [0.9, None, 0.7, 0.8, None], dtype=pl.Float64)
        result = compute_precision_at_k(scores, hits, k=3)
        # Top 3 by score: idx 0 (0.9, True), idx 3 (0.8, True), idx 2 (0.7, True)
        assert result == pytest.approx(1.0)


# ============================================================================
# 3. Lift@K tests
# ============================================================================


class TestLiftAtK:
    """Compute Lift@K = Precision@K / base_rate."""

    def test_lift_normal(self):
        """Precision=0.4, base_rate=0.2 => lift=2.0."""
        from alphascreener.alpha_acceptance import compute_lift_at_k

        result = compute_lift_at_k(precision=0.4, base_rate=0.2)
        assert result == pytest.approx(2.0)

    def test_lift_equal_one(self):
        """Precision = base_rate => lift = 1.0 (random)."""
        from alphascreener.alpha_acceptance import compute_lift_at_k

        result = compute_lift_at_k(precision=0.3, base_rate=0.3)
        assert result == pytest.approx(1.0)

    def test_lift_less_than_one(self):
        """Precision < base_rate => lift < 1.0."""
        from alphascreener.alpha_acceptance import compute_lift_at_k

        result = compute_lift_at_k(precision=0.1, base_rate=0.3)
        assert result < 1.0

    def test_lift_zero_base_rate(self):
        """base_rate = 0 => lift = inf, handled gracefully as None."""
        from alphascreener.alpha_acceptance import compute_lift_at_k

        result = compute_lift_at_k(precision=0.1, base_rate=0.0)
        assert result is None


# ============================================================================
# 4. Recall@K tests
# ============================================================================


class TestRecallAtK:
    """Compute Recall@K = top K hits / total market hits."""

    def test_recall_normal(self):
        """20 total hits, top 10 has 8 => recall = 0.4."""
        from alphascreener.alpha_acceptance import compute_recall_at_k

        hits = pl.Series("hits", [True] * 8 + [False] * 2 + [True] * 12 + [False] * 78)
        scores = pl.Series("score", list(range(100, 0, -1)), dtype=pl.Float64)
        result = compute_recall_at_k(scores, hits, k=10)
        # Top 10 hits: 8, total hits: 20 => 8/20
        assert result == pytest.approx(8.0 / 20.0)

    def test_recall_perfect(self):
        """All market hits are in top K => recall = 1.0."""
        from alphascreener.alpha_acceptance import compute_recall_at_k

        hits = pl.Series("hits", [True] * 5 + [False] * 95)
        scores = pl.Series("score", list(range(100, 0, -1)), dtype=pl.Float64)
        result = compute_recall_at_k(scores, hits, k=10)
        # Top 10: 5 hits, total: 5 hits => 1.0
        assert result == 1.0

    def test_recall_zero(self):
        """No hits in top K => recall = 0.0."""
        from alphascreener.alpha_acceptance import compute_recall_at_k

        hits = pl.Series("hits", [False] * 10 + [True] * 90)
        scores = pl.Series("score", list(range(100, 0, -1)), dtype=pl.Float64)
        result = compute_recall_at_k(scores, hits, k=10)
        assert result == 0.0

    def test_recall_no_hits_anywhere(self):
        """No hits in the entire market => recall = 0.0."""
        from alphascreener.alpha_acceptance import compute_recall_at_k

        hits = pl.Series("hits", [False] * 100)
        scores = pl.Series("score", list(range(100, 0, -1)), dtype=pl.Float64)
        result = compute_recall_at_k(scores, hits, k=10)
        assert result == 0.0

    def test_recall_empty(self):
        """Empty input => recall = 0.0."""
        from alphascreener.alpha_acceptance import compute_recall_at_k

        hits = pl.Series("hits", [], dtype=pl.Boolean)
        scores = pl.Series("score", [], dtype=pl.Float64)
        result = compute_recall_at_k(scores, hits, k=10)
        assert result == 0.0


# ============================================================================
# 5. IC (Spearman rank correlation) tests
# ============================================================================


class TestInformationCoefficient:
    """Compute IC = Spearman rank correlation between scores and returns."""

    def test_ic_perfect_positive(self):
        """Perfectly monotonic => IC = 1.0."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.1, 0.2, 0.3, 0.4, 0.5], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.02, 0.03, 0.04, 0.05], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert result == pytest.approx(1.0)

    def test_ic_perfect_negative(self):
        """Perfectly inverse monotonic => IC = -1.0."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.5, 0.4, 0.3, 0.2, 0.1], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.02, 0.03, 0.04, 0.05], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert result == pytest.approx(-1.0)

    def test_ic_zero(self):
        """No rank correlation => IC ~ 0."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series(
            "score",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=pl.Float64,
        )
        returns = pl.Series(
            "t7_return",
            [0.05, -0.02, 0.08, -0.03, 0.01, 0.06, -0.05, 0.03, -0.01, 0.02],
            dtype=pl.Float64,
        )
        result = compute_ic(scores, returns)
        # With random-like data, IC should be near 0
        assert -0.8 < result < 0.8

    def test_ic_tied_ranks(self):
        """Tied values use average rank (handled by scipy)."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.5, 0.5, 0.5, 0.1, 0.9], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.01, 0.01, 0.00, 0.05], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert -1.0 <= result <= 1.0

    def test_ic_too_few_points(self):
        """Less than 3 data points => IC = 0.0 (not enough for rank correlation)."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.5, 0.6], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.02], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert result == 0.0

    def test_ic_with_nulls(self):
        """Null pairs are dropped before computing IC."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.1, 0.2, None, 0.4, 0.5], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.02, 0.03, None, 0.05], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert -1.0 <= result <= 1.0

    def test_ic_all_nulls(self):
        """All null => IC = 0.0."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [None, None, None], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.01, 0.02, 0.03], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        assert result == 0.0

    def test_ic_constant_returns(self):
        """All returns equal => no variance => IC = 0.0 (or nan handled)."""
        from alphascreener.alpha_acceptance import compute_ic

        scores = pl.Series("score", [0.1, 0.2, 0.3, 0.4, 0.5], dtype=pl.Float64)
        returns = pl.Series("t7_return", [0.05, 0.05, 0.05, 0.05, 0.05], dtype=pl.Float64)
        result = compute_ic(scores, returns)
        # All returns equal => ranks all same => spearmanr returns nan
        assert result == 0.0 or math.isnan(result)


# ============================================================================
# 6. Block-bootstrap 95% CI tests
# ============================================================================


class TestBlockBootstrapCI:
    """Compute block-bootstrap 95% confidence interval."""

    def test_bootstrap_ci_bounds(self):
        """CI lower < CI upper and both are finite."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.random.default_rng(42).normal(0.0, 1.0, 100)
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=500, ci=95.0)
        assert lower <= upper
        assert math.isfinite(lower)
        assert math.isfinite(upper)

    def test_bootstrap_ci_with_block_structure(self):
        """CI computation handles block_size > 1 correctly."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=3, n_samples=500, ci=95.0)
        assert lower <= upper
        # Mean of data = 5.5, CI should contain it
        assert lower <= 5.5 <= upper

    def test_bootstrap_ci_negative_data(self):
        """CI works with negative values."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.random.default_rng(42).normal(-0.05, 0.15, 100)
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=500, ci=95.0)
        # Mean should be negative, CI should cover it
        assert lower <= np.mean(data) <= upper

    def test_bootstrap_ci_small_data(self):
        """Very small data (< 2 * block_size) still produces results."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.array([1.0, 2.0, 3.0])
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=100, ci=95.0)
        assert lower <= upper

    def test_bootstrap_ci_empty_data(self):
        """Empty data returns (0.0, 0.0)."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.array([])
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=100, ci=95.0)
        assert lower == 0.0 and upper == 0.0

    def test_bootstrap_ci_constant_data(self):
        """Constant data: all bootstrap samples give same statistic => CI narrow."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.array([5.0] * 50)
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=200, ci=95.0)
        # Mean = 5.0, CI should be very tight
        assert lower == pytest.approx(5.0, abs=0.1)
        assert upper == pytest.approx(5.0, abs=0.1)

    def test_bootstrap_ci_custom_statistic(self):
        """Works with a custom statistic function (e.g., precision)."""
        from alphascreener.alpha_acceptance import block_bootstrap_ci

        data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        lower, upper = block_bootstrap_ci(data, np.mean, block_size=5, n_samples=500, ci=95.0)
        # Mean = 11/20 = 0.55
        assert lower <= 0.55 <= upper


# ============================================================================
# 7. compute_all_alpha_metrics integration tests
# ============================================================================


class TestComputeAllAlphaMetrics:
    """End-to-end metric computation from a single DataFrame."""

    def test_returns_all_expected_keys(self):
        """Result dict contains all expected metric keys."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(100)
        metrics = compute_all_alpha_metrics(df)

        expected_keys = [
            "base_rate",
            "precision_at_20_pure",
            "precision_at_20_llm",
            "precision_at_10_pure",
            "precision_at_10_llm",
            "lift_at_20_pure",
            "lift_at_20_llm",
            "recall_at_20_pure",
            "recall_at_20_llm",
            "recall_at_10_pure",
            "recall_at_10_llm",
            "ic_pure",
            "ic_llm",
            "bootstrap_ci_lower_pure",
            "bootstrap_ci_upper_pure",
            "bootstrap_ci_lower_llm",
            "bootstrap_ci_upper_llm",
            "sample_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_sample_size_matches_input(self):
        """sample_size equals number of tickers in input."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(73)
        metrics = compute_all_alpha_metrics(df)
        assert metrics["sample_size"] == 73

    def test_metrics_in_range(self):
        """All ratio metrics are in [0, 1]."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(200)
        metrics = compute_all_alpha_metrics(df)

        ratio_keys = [
            "base_rate",
            "precision_at_20_pure",
            "precision_at_20_llm",
            "precision_at_10_pure",
            "precision_at_10_llm",
            "recall_at_20_pure",
            "recall_at_20_llm",
            "recall_at_10_pure",
            "recall_at_10_llm",
        ]
        for key in ratio_keys:
            val = metrics[key]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key} = {val} out of range [0,1]"

    def test_ic_in_range(self):
        """IC values are in [-1, 1]."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(200)
        metrics = compute_all_alpha_metrics(df)
        assert -1.0 <= metrics["ic_pure"] <= 1.0
        assert -1.0 <= metrics["ic_llm"] <= 1.0

    def test_bootstrap_ci_ordering(self):
        """CI lower <= CI upper."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(200)
        metrics = compute_all_alpha_metrics(df)
        assert metrics["bootstrap_ci_lower_pure"] <= metrics["bootstrap_ci_upper_pure"]
        assert metrics["bootstrap_ci_lower_llm"] <= metrics["bootstrap_ci_upper_llm"]

    def test_empty_dataframe(self):
        """Empty DataFrame returns zero-valued metrics."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "breakout_score": pl.Float64,
                "refined_score": pl.Float64,
                "t7_return": pl.Float64,
            },
        )
        metrics = compute_all_alpha_metrics(df)
        assert metrics["sample_size"] == 0
        assert metrics["base_rate"] == 0.0

    def test_missing_score_column(self):
        """Missing score column => raises ValueError."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = pl.DataFrame(
            {
                "ticker": ["A"],
                "t7_return": [0.05],
            }
        )
        with pytest.raises(ValueError, match="breakout_score"):
            compute_all_alpha_metrics(df)

    def test_missing_return_column(self):
        """Missing return column => raises ValueError."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = pl.DataFrame(
            {
                "ticker": ["A"],
                "breakout_score": [1.0],
                "refined_score": [1.0],
            }
        )
        with pytest.raises(ValueError, match="t7_return"):
            compute_all_alpha_metrics(df)

    def test_single_ticker(self):
        """Single ticker: metrics computed without crashing."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_alpha_df(["A"], [1.0], [2.0], [0.15])
        metrics = compute_all_alpha_metrics(df)
        assert metrics["sample_size"] == 1
        assert metrics["base_rate"] == 1.0
        # With 1 ticker, precision@K = 1.0 if it's a hit
        assert metrics["precision_at_20_pure"] == 1.0
        assert metrics["recall_at_20_pure"] == 1.0

    def test_custom_k_values(self):
        """Custom K values are supported."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_random_alpha_df(100)
        metrics = compute_all_alpha_metrics(df, k_values=(5, 15))
        assert "precision_at_5_pure" in metrics
        assert "precision_at_15_pure" in metrics
        assert "precision_at_10_pure" not in metrics

    def test_lift_zero_base_rate_handling(self):
        """When base_rate=0, lift values are None."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        df = _make_alpha_df(
            ["A", "B", "C", "D", "E"],
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.05, 0.03, 0.01, -0.02, -0.05],
        )
        metrics = compute_all_alpha_metrics(df)
        assert metrics["base_rate"] == 0.0
        assert metrics["lift_at_20_pure"] is None
        assert metrics["lift_at_20_llm"] is None

    def test_pure_and_llm_tracks_independent(self):
        """Pure and LLM scores produce different metrics (two tracks)."""
        from alphascreener.alpha_acceptance import compute_all_alpha_metrics

        # Pure scores rank T1 highest, LLM scores rank T2 highest
        df = _make_alpha_df(
            ["T1", "T2", "T3", "T4", "T5"],
            [0.9, 0.3, 0.5, 0.2, 0.1],  # pure
            [0.1, 0.9, 0.5, 0.3, 0.2],  # llm
            [0.15, 0.05, 0.03, 0.12, 0.08],  # returns
        )
        metrics = compute_all_alpha_metrics(df, k_values=(2,))
        # T1 hit (0.15), T2 non-hit (0.05)
        # Pure Top 2: T1, T3 => 1/2 = 0.5
        # LLM Top 2: T2, T3 => 0/2 = 0.0
        assert metrics["precision_at_2_pure"] == 0.5
        assert metrics["precision_at_2_llm"] == 0.0


# ============================================================================
# 8. Database write tests
# ============================================================================


class TestWriteAlphaAcceptance:
    """Write metrics to alpha_acceptance_daily table."""

    @pytest.fixture
    def fresh_db(self, tmp_path: Path):
        """Create a fresh in-memory SQLite database with alpha_acceptance_daily table."""
        engine = create_engine("sqlite://", echo=False)
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def sample_metrics(self) -> dict:
        """Return a complete metrics dict for testing."""
        return {
            "metric_date": date(2025, 3, 15),
            "base_rate": 0.15,
            "precision_at_20_pure": 0.30,
            "precision_at_20_llm": 0.35,
            "precision_at_10_pure": 0.40,
            "precision_at_10_llm": 0.45,
            "lift_at_20_pure": 2.0,
            "lift_at_20_llm": 2.33,
            "recall_at_20_pure": 0.25,
            "recall_at_20_llm": 0.28,
            "recall_at_10_pure": 0.15,
            "recall_at_10_llm": 0.18,
            "ic_pure": 0.12,
            "ic_llm": 0.15,
            "bootstrap_ci_lower_pure": 0.10,
            "bootstrap_ci_upper_pure": 0.50,
            "bootstrap_ci_lower_llm": 0.12,
            "bootstrap_ci_upper_llm": 0.55,
            "sample_size": 500,
        }

    def test_write_single_record(self, fresh_db, sample_metrics):
        """Write a single metrics record and verify it can be read back."""
        from alphascreener.alpha_acceptance import write_alpha_acceptance

        write_alpha_acceptance(sample_metrics, fresh_db)

        from sqlalchemy import select
        from sqlalchemy.orm import Session

        with Session(fresh_db) as session:
            row = session.execute(
                select(AlphaAcceptanceDaily).where(
                    AlphaAcceptanceDaily.metric_date == date(2025, 3, 15)
                )
            ).scalar_one()
            assert row.base_rate == pytest.approx(0.15)
            assert row.precision_at_20_pure == pytest.approx(0.30)
            assert row.ic_pure == pytest.approx(0.12)
            assert row.sample_size == 500

    def test_write_nullable_fields_handled(self, fresh_db):
        """Nullable fields (lift, precision) can be None."""
        from alphascreener.alpha_acceptance import write_alpha_acceptance

        metrics = {
            "metric_date": date(2025, 3, 16),
            "base_rate": 0.0,
            "precision_at_20_pure": None,
            "precision_at_20_llm": None,
            "precision_at_10_pure": None,
            "precision_at_10_llm": None,
            "lift_at_20_pure": None,
            "lift_at_20_llm": None,
            "recall_at_20_pure": None,
            "recall_at_20_llm": None,
            "recall_at_10_pure": None,
            "recall_at_10_llm": None,
            "ic_pure": None,
            "ic_llm": None,
            "bootstrap_ci_lower_pure": None,
            "bootstrap_ci_upper_pure": None,
            "bootstrap_ci_lower_llm": None,
            "bootstrap_ci_upper_llm": None,
            "sample_size": 0,
        }
        write_alpha_acceptance(metrics, fresh_db)

        from sqlalchemy import select
        from sqlalchemy.orm import Session

        with Session(fresh_db) as session:
            row = session.execute(
                select(AlphaAcceptanceDaily).where(
                    AlphaAcceptanceDaily.metric_date == date(2025, 3, 16)
                )
            ).scalar_one()
            assert row.base_rate == 0.0
            assert row.precision_at_20_pure is None

    def test_write_upsert_overwrites_existing(self, fresh_db, sample_metrics):
        """Writing to the same date twice upserts (replaces) the record."""
        from alphascreener.alpha_acceptance import write_alpha_acceptance

        write_alpha_acceptance(sample_metrics, fresh_db)

        # Write again with different values for the same date
        updated = dict(sample_metrics)
        updated["base_rate"] = 0.20
        updated["sample_size"] = 600
        write_alpha_acceptance(updated, fresh_db)

        from sqlalchemy import select
        from sqlalchemy.orm import Session

        with Session(fresh_db) as session:
            rows = session.execute(select(AlphaAcceptanceDaily)).scalars().all()
            assert len(rows) == 1  # Only one record
            assert rows[0].base_rate == pytest.approx(0.20)
            assert rows[0].sample_size == 600

    def test_write_missing_metric_date_raises(self, fresh_db):
        """Missing metric_date in metrics dict raises KeyError."""
        from alphascreener.alpha_acceptance import write_alpha_acceptance

        with pytest.raises(KeyError):
            write_alpha_acceptance({"base_rate": 0.1, "sample_size": 10}, fresh_db)
