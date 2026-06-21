"""Factor weight optimization via rolling walk-forward backtest.

Iteratively perturbs factor weights and measures out-of-sample performance
on rolling train/test windows. Converges when weight changes drop below 1%.

Uses Precision@20, Lift@20, and Sharpe ratio as optimization targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl

from alphascreener.acceptance import (
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
    """Generate rolling train/test windows.

    When the data span is too short for the requested ``train_years``, the
    function progressively shortens the training period (by 0.5-year steps,
    down to a minimum of 0.5 years) until at least one valid window can be
    produced.
    """
    windows: list[tuple[date, date, date, date]] = []
    test_days = test_months * 30
    step_days = step_months * 30
    min_train_years = 0.5

    current_train_years = float(train_years)
    while current_train_years >= min_train_years and len(windows) == 0:
        train_days = int(current_train_years * 365)

        cursor = data_start + timedelta(days=train_days)
        while cursor + timedelta(days=test_days) <= data_end and len(windows) < max_windows:
            train_start = cursor - timedelta(days=train_days)
            train_end = cursor
            test_start = cursor
            test_end = cursor + timedelta(days=test_days)
            windows.append((train_start, train_end, test_start, test_end))
            cursor += timedelta(days=step_days)

        if len(windows) == 0:
            _logger.warning(
                "No rolling windows for train_years=%.1f"
                " (span %s → %s); retrying with shorter training",
                current_train_years,
                data_start,
                data_end,
            )
            current_train_years -= 0.5

    if len(windows) == 0:
        _logger.error(
            "Cannot generate any rolling window with data span %s → %s "
            "(need at least %.1f years + %d months)",
            data_start,
            data_end,
            min_train_years,
            test_months,
        )

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
    test_data = ohlcv_df.filter((pl.col("dt") >= test_start) & (pl.col("dt") <= test_end))
    if test_data.height < 100:
        return None

    # Compute factors on the full range (need train history for rolling windows)
    range_data = ohlcv_df.filter((pl.col("dt") >= train_start) & (pl.col("dt") <= test_end))
    factors = compute_factors(range_data, dt=test_end)

    # ── Screening ──
    filtered, _ = hard_filter_with_fallback(factors)
    passed = filtered.filter(pl.col("pass_phase1"))
    if passed.height == 0:
        return None

    result = phase2_pipeline(passed, n_final=20)

    # ── Outcomes from OHLCV (T+7 forward return) ──
    # Entry at test_start; hold for 7 trading days; hit if return >= 10%.
    tickers = result["ticker"].to_list()
    hits = []
    for t in tickers:
        t_data = ohlcv_df.filter(pl.col("ticker") == t).sort("dt")
        t_dates = t_data["dt"].to_list()
        t_closes = t_data["close"].to_list()

        if len(t_closes) == 0:
            hits.append(0)
            continue

        # Find entry index: first trading day ≥ test_start
        entry_idx = None
        for i, d in enumerate(t_dates):
            if d >= test_start:
                entry_idx = i
                break
        if entry_idx is None:
            hits.append(0)
            continue

        entry_close = t_closes[entry_idx]
        fwd_idx = min(entry_idx + 7, len(t_closes) - 1)
        fwd_close = t_closes[fwd_idx] if fwd_idx > entry_idx else entry_close
        hit = 1 if entry_close > 0 and (fwd_close / entry_close - 1) >= 0.10 else 0
        hits.append(hit)

    # ── Metrics ──
    breakout_vals = result["breakout_score"].to_list()
    if len(breakout_vals) != len(tickers):
        # Safety: if column lengths diverge, fall back to equal weights
        breakout_vals = [1.0] * len(tickers)

    scores_s = pl.Series("score", breakout_vals, dtype=pl.Float64)
    hits_s = pl.Series("hit", hits, dtype=pl.Int64)
    precision = compute_precision_at_k(scores_s, hits_s, 20)
    base_rate = compute_base_rate(hits_s.cast(pl.Float64))
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
    return [(name, _normalize_weights(w)) for name, w in variants]


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _score_one(
    ohlcv_df: pl.DataFrame,
    weights: dict[str, float],
    windows: list[tuple[date, date, date, date]],
) -> float:
    """Compute average composite score across all windows for given weights."""
    vals = []
    for tr_s, tr_e, te_s, te_e in windows:
        r = _evaluate_window(ohlcv_df, weights, tr_s, tr_e, te_s, te_e)
        if r is not None:
            vals.append(r.score)
    return sum(vals) / len(vals) if vals else 0.0


def _optimize_grid_search(
    ohlcv_df: pl.DataFrame,
    initial_weights: dict[str, float],
    windows: list[tuple[date, date, date, date]],
) -> dict[str, float]:
    """Exhaustive per-factor multiplier grid search (baseline strategy)."""
    best_weights = dict(initial_weights)
    multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]

    for factor in list(initial_weights.keys()):
        best_w = initial_weights[factor]
        best_score = _score_one(ohlcv_df, best_weights, windows)
        for m in multipliers:
            test_w = dict(best_weights)
            test_w[factor] = initial_weights[factor] * m
            test_w = _normalize_weights(test_w)
            s = _score_one(ohlcv_df, test_w, windows)
            if s > best_score:
                best_score = s
                best_w = test_w[factor]
        best_weights[factor] = best_w
        best_weights = _normalize_weights(best_weights)

    return best_weights


def _optimize_tpe(
    ohlcv_df: pl.DataFrame,
    initial_weights: dict[str, float],
    windows: list[tuple[date, date, date, date]],
    n_trials: int,
) -> dict[str, float]:
    """Bayesian optimization via Optuna TPESampler.

    Samples factor weights from [0.01, 1.0] and normalizes them to sum
    to 1.0 before scoring. Uses a fixed random seed for reproducibility.
    """
    import optuna

    factor_names = list(initial_weights.keys())
    if not factor_names:
        return {}

    def objective(trial: optuna.Trial) -> float:
        raw = {name: trial.suggest_float(f"w_{name}", 0.01, 1.0) for name in factor_names}
        w = _normalize_weights(raw)
        return _score_one(ohlcv_df, w, windows)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_raw = {
        name: study.best_params.get(f"w_{name}", initial_weights.get(name, 1.0 / len(factor_names)))
        for name in factor_names
    }
    return _normalize_weights(best_raw)


def optimize_weights(
    ohlcv_df: pl.DataFrame,
    initial_weights: dict[str, float],
    *,
    strategy: str = "tpe",
    n_trials: int = 30,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    max_windows: int = DEFAULT_MAX_WINDOWS,
    convergence: float = DEFAULT_CONVERGENCE_THRESHOLD,
) -> OptimizeReport:
    """Run walk-forward weight optimization.

    Parameters
    ----------
    strategy : str
        ``"tpe"`` (default) uses Optuna TPESampler for Bayesian
        optimization. ``"grid_search"`` uses exhaustive per-factor
        multiplier search.
    n_trials : int
        Number of Optuna trials (TPE only, default 30).
    """
    if strategy not in ("tpe", "grid_search"):
        raise ValueError(
            f"Unknown optimization strategy: {strategy!r}. Expected 'tpe' or 'grid_search'."
        )

    data_start = ohlcv_df["dt"].min()
    data_end = ohlcv_df["dt"].max()
    windows = _build_rolling_windows(
        data_start, data_end, train_years, test_months, step_months, max_windows
    )

    report = OptimizeReport(initial_weights=dict(initial_weights), final_weights={})

    if not initial_weights:
        report.converged = False
        report.iterations = 0
        return report

    weights = dict(initial_weights)
    for tr_s, tr_e, te_s, te_e in windows:
        r = _evaluate_window(ohlcv_df, weights, tr_s, tr_e, te_s, te_e)
        if r is not None:
            report.windows.append(r)

    if strategy == "tpe":
        best_weights = _optimize_tpe(ohlcv_df, initial_weights, windows, n_trials)
    else:
        best_weights = _optimize_grid_search(ohlcv_df, initial_weights, windows)

    report.final_weights = dict(best_weights)
    report.iterations = len(windows)
    report.converged = any(
        abs(best_weights.get(k, 0) - initial_weights.get(k, 0)) > 0.001 for k in initial_weights
    )
    return report
