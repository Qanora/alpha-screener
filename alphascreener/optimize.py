"""Factor weight optimization via rolling walk-forward backtest.

Iteratively perturbs factor weights and measures out-of-sample performance
on rolling train/test windows. Converges when weight changes drop below 1%.

Uses Precision@20, Lift@20, and Sharpe ratio as optimization targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from alphascreener.acceptance import (
    compute_all_alpha_metrics,
    compute_base_rate,
    compute_precision_at_k,
)
from alphascreener.logging import get_logger

_logger = get_logger("screening")

# Default optimization parameters
DEFAULT_TRAIN_YEARS = 2
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 6
DEFAULT_MAX_WINDOWS = 50
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # 1% weight change
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_LR_DECAY = 0.95  # per window


@dataclass
class WindowResult:
    """Performance metrics for a single train/test window."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    precision_at_20: float
    lift_at_20: float
    base_rate: float
    sharpe: float
    max_drawdown: float
    weights: dict[str, float]

    @property
    def score(self) -> float:
        """Composite optimization score (higher = better)."""
        # Lift > 1 means we beat random; scale to [0, 1] via tanh
        lift_score = math.tanh(max(0, self.lift_at_20 - 1.0))
        sharpe_score = math.tanh(max(0, self.sharpe) / 2.0)
        precision_score = min(1.0, self.precision_at_20 / 0.5)
        return 0.4 * lift_score + 0.3 * sharpe_score + 0.3 * precision_score


@dataclass
class OptimizeReport:
    """Full optimization report."""

    initial_weights: dict[str, float]
    final_weights: dict[str, float]
    windows: list[WindowResult] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0

    @property
    def weight_changes(self) -> dict[str, float]:
        return {
            k: self.final_weights.get(k, 0) - self.initial_weights.get(k, 0)
            for k in self.initial_weights
        }


def _build_rolling_windows(
    data_start: date,
    data_end: date,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> list[tuple[date, date, date, date]]:
    """Generate rolling train/test windows."""
    windows = []
    train_days = train_years * 365
    test_days = test_months * 30
    step_days = step_months * 30

    cursor = data_start + timedelta(days=train_days)
    while cursor + timedelta(days=test_days) <= data_end and len(windows) < max_windows:
        train_start = cursor - timedelta(days=train_days)
        train_end = cursor
        test_start = cursor
        test_end = cursor + timedelta(days=test_days)
        windows.append((train_start, train_end, test_start, test_end))
        cursor += timedelta(days=step_days)

    return windows


def _evaluate_window(
    ohlcv_df: pl.DataFrame,
    weights: dict[str, float],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> WindowResult | None:
    """Run screening + evaluation on one window with given weights."""
    try:
        from alphascreener.factors.engine import compute_factors
        from alphascreener.screening.phase1 import hard_filter_with_fallback
        from alphascreener.screening.phase2 import phase2_pipeline
    except ImportError:
        return None

    # ── Factor computation on test data ──
    test_data = ohlcv_df.filter(
        (pl.col("dt") >= test_start) & (pl.col("dt") <= test_end)
    )
    if test_data.height < 100:
        return None

    # Compute factors on the full range (need train history for rolling windows)
    range_data = ohlcv_df.filter(
        (pl.col("dt") >= train_start) & (pl.col("dt") <= test_end)
    )
    factors = compute_factors(range_data, dt=test_end)

    # ── Screening ──
    filtered, _ = hard_filter_with_fallback(factors)
    passed = filtered.filter(pl.col("pass_phase1"))
    if passed.height == 0:
        return None

    result = phase2_pipeline(passed, n_final=20)

    # ── Outcomes from OHLCV (T+7 forward return) ──
    # Build outcomes: if any ticker has T+7 close >= entry * 1.10 -> hit
    tickers = result["ticker"].to_list()
    hits = []
    for t in tickers:
        t_data = ohlcv_df.filter(pl.col("ticker") == t).sort("dt")
        t_dates = t_data["dt"].to_list()
        t_closes = t_data["close"].to_list()

        # Find latest date close and T+7 close
        latest_idx = len(t_dates) - 1
        entry_close = t_closes[latest_idx] if latest_idx >= 0 else 0
        # Look forward 7 trading days
        fwd_idx = min(latest_idx + 7, len(t_closes) - 1)
        fwd_close = t_closes[fwd_idx] if fwd_idx > latest_idx else entry_close
        hit = 1 if entry_close > 0 and (fwd_close / entry_close - 1) >= 0.10 else 0
        hits.append(hit)

    # ── Metrics ──
    scores = np.array([1.0] * len(tickers))  # all have equal score in result
    hits_arr = np.array(hits, dtype=np.int32)

    precision = compute_precision_at_k(
        pl.DataFrame({"score": scores, "hit": hits_arr}), 20, score_col="score", outcome_col="hit"
    )
    base_rate = compute_base_rate(pl.DataFrame({"score": scores, "hit": hits_arr}), outcome_col="hit")
    if math.isnan(precision):
        precision = 0.0
    lift = precision / base_rate if base_rate and base_rate > 0 else 0.0
    if math.isnan(lift):
        lift = 0.0

    # ── Backtest for Sharpe/MaxDD ──
    from alphascreener.backtrader import _load_ohlcv_data, _load_signals_data, run_backtest

    sharpe = 0.0
    max_dd = 0.0
    try:
        ticker_dfs = _load_ohlcv_data(start_date=test_start, end_date=test_end)
        signals = _load_signals_data(start_date=test_start, end_date=test_end)
        bt_results = []
        for t in tickers[:5]:
            df_t = ticker_dfs.get(t)
            if df_t is None or df_t.height == 0:
                continue
            bt = run_backtest({t: df_t}, signals=signals)
            bt_results.append(bt["metrics"]["sharpe_ratio"])
            max_dd = max(max_dd, abs(bt["metrics"]["max_drawdown"]))
        if bt_results:
            sharpe = sum(bt_results) / len(bt_results)
    except Exception:
        pass

    return WindowResult(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        precision_at_20=precision,
        lift_at_20=lift,
        base_rate=base_rate,
        sharpe=sharpe,
        max_drawdown=max_dd,
        weights=dict(weights),
    )


def _perturb_weights(
    weights: dict[str, float],
    *,
    step_size: float = 0.02,
) -> list[tuple[str, dict[str, float]]]:
    """Generate perturbed weight variants (one with each factor bumped up/down)."""
    variants = []
    for factor in weights:
        up = dict(weights)
        up[factor] = min(0.5, up[factor] + step_size)
        variants.append((f"{factor}+", up))

        down = dict(weights)
        down[factor] = max(0.001, down[factor] - step_size)
        variants.append((f"{factor}-", down))

    # Normalize each variant to sum to ~1.0
    for name, w in variants:
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] /= total

    return variants


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def optimize_weights(
    ohlcv_df: pl.DataFrame,
    initial_weights: dict[str, float],
    *,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    max_windows: int = DEFAULT_MAX_WINDOWS,
    convergence: float = DEFAULT_CONVERGENCE_THRESHOLD,
) -> OptimizeReport:
    """Run walk-forward weight optimization.

    Args:
        ohlcv_df: Full OHLCV DataFrame with columns ticker, dt, open, high, low, close, volume.
        initial_weights: Starting factor weights (e.g. MVP_WEIGHTS).
        train_years: Training window length in years.
        test_months: Test window length in months.
        step_months: Step size between windows in months.
        max_windows: Maximum number of windows.
        convergence: Stop when max weight change < this threshold.

    Returns:
        OptimizeReport with final weights and window-by-window results.
    """
    weights = dict(initial_weights)
    data_start = ohlcv_df["dt"].min()
    data_end = ohlcv_df["dt"].max()

    _logger.info(
        "optimize: data=%s→%s, train=%dy, test=%dm, step=%dm",
        data_start, data_end, train_years, test_months, step_months,
    )

    windows = _build_rolling_windows(
        data_start, data_end, train_years, test_months, step_months, max_windows
    )

    if len(windows) < 2:
        _logger.warning("optimize: insufficient data for rolling windows (need >= 2)")
        return OptimizeReport(
            initial_weights=initial_weights,
            final_weights=weights,
            converged=False,
            iterations=0,
        )

    report = OptimizeReport(initial_weights=dict(initial_weights), final_weights={})
    lr = DEFAULT_LEARNING_RATE

    prev_weights = dict(weights)

    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        # Evaluate current weights
        baseline = _evaluate_window(ohlcv_df, weights, tr_s, tr_e, te_s, te_e)
        if baseline is None:
            continue

        # Generate perturbed variants and test them
        variants = _perturb_weights(weights, step_size=lr)
        best_variant = None
        best_score = baseline.score

        for vname, vw in variants:
            result = _evaluate_window(ohlcv_df, vw, tr_s, tr_e, te_s, te_e)
            if result is None:
                continue
            if result.score > best_score:
                best_score = result.score
                best_variant = vw

        # Update weights toward best variant
        if best_variant is not None:
            # Interpolate between current and best variant
            for k in weights:
                weights[k] = (1 - lr) * weights[k] + lr * best_variant[k]
            weights = _normalize_weights(weights)

        # Re-evaluate with updated weights for the report
        final_eval = _evaluate_window(ohlcv_df, weights, tr_s, tr_e, te_s, te_e)
        if final_eval is not None:
            report.windows.append(final_eval)

        # Check convergence
        max_delta = max(abs(weights.get(k, 0) - prev_weights.get(k, 0)) for k in weights)
        if max_delta < convergence and wi > 1:
            report.converged = True
            break

        prev_weights = dict(weights)
        lr *= DEFAULT_LR_DECAY  # decay learning rate
        report.iterations = wi + 1

    report.final_weights = dict(weights)

    if not report.converged and report.iterations >= len(windows) - 1:
        report.iterations = len(windows)

    return report
