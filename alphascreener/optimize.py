"""Factor weight optimization via rolling walk-forward backtest.

Uses portfolio-level Sharpe ratio from multi-ticker backtrader runs as the
optimization target, eliminating the proxy-objective bias of the previous
composite score (IC + Lift + QuantileSpread + single-stock Sharpe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl

from alphascreener.acceptance import (
    compute_base_rate,
    compute_ic,
    compute_precision_at_k,
)
from alphascreener.logging import get_logger
from alphascreener.screening.phase2 import apply_industry_dedup

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
    """Performance metrics for a single train/test window.

    ic : Spearman rank IC between breakout_score and T+7 returns.
    quantile_spread : Mean T+7 return of top vs bottom quintile.
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    precision_at_20: float
    lift_at_20: float
    base_rate: float
    ic: float
    quantile_spread: float
    sharpe: float
    max_drawdown: float
    weights: dict[str, float]

    @property
    def score(self) -> float:
        """Portfolio-level Sharpe ratio as the optimization target.

        Replaces the previous composite proxy (IC+Lift+QuantileSpread+
        single-stock Sharpe) with the true portfolio Sharpe from a
        multi-ticker backtrader run.  This eliminates proxy-objective bias:
        the optimiser now maximises the same risk-adjusted return that the
        strategy will actually deliver.
        """
        return self.sharpe


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
    *,
    universe_meta: pl.DataFrame | None = None,
) -> WindowResult | None:
    """Run screening + evaluation on one window with given weights.

    Factors are computed strictly on train-only data
    (``train_start..train_end``) to avoid look-ahead bias from EMA and
    rolling windows.  The result is then snapped to the latest row per
    ticker on or before *test_start*.  Custom *weights* are passed
    through to ``compute_breakout_score`` so the optimiser can measure
    their true impact on ranking quality.

    When *universe_meta* is provided (DataFrame with ``ticker``,
    ``sector``, ``industry`` columns), industry deduplication
    (Sector≤3, Industry≤2) is applied to the scored candidates before
    computing IC, Precision, Lift, and QuantileSpread.  This makes the
    optimisation metrics reflect the real Phase 2 pipeline constraints
    instead of overestimating performance on an unconstrained candidate
    pool.
    """
    try:
        from alphascreener.factors.engine import compute_factors
        from alphascreener.screening.phase1 import hard_filter_with_fallback
        from alphascreener.screening.phase2 import compute_breakout_score as _cbs
    except ImportError:
        return None

    # ── Factor computation on train-only range (avoid look-ahead bias) ──
    range_data = ohlcv_df.filter((pl.col("dt") >= train_start) & (pl.col("dt") <= train_end))
    if range_data.height < 100:
        return None

    factors_all = compute_factors(range_data, dt=train_end)

    # ── Snap to test_start: one row per ticker (latest ≤ test_start) ──
    snap = (
        factors_all.filter(pl.col("dt") <= test_start)
        .sort("dt", descending=True)
        .unique(subset=["ticker"], keep="first")
    )
    if snap.height < 10:
        return None

    # ── Phase 1 hard filter ──
    filtered, _ = hard_filter_with_fallback(snap)
    passed = filtered.filter(pl.col("pass_phase1"))
    if passed.height < 5:
        return None

    # ── Phase 2: score ALL passed tickers with custom weights ──
    scored_all = _cbs(passed, weights=weights)

    # ── Industry dedup (Issue #325): apply real Phase 2 constraints ──
    deduped = scored_all
    if universe_meta is not None and universe_meta.height > 0:
        try:
            # Pre-selected columns avoid per-window .select() overhead
            joined = scored_all.join(
                universe_meta, on="ticker", how="left"
            )
            deduped = (
                joined
                .sort("breakout_score", descending=True)
                .pipe(apply_industry_dedup)
            )
        except pl.PolarsError:
            _logger.warning(
                "Industry dedup failed for window %s→%s; falling back to undeduped set",
                test_start,
                test_end,
                exc_info=True,
            )
            deduped = scored_all

    # ── T+7 forward returns for DEDUPED tickers ──
    t7_map: dict[str, float] = {}
    hit_map: dict[str, int] = {}
    deduped_tickers = set(deduped["ticker"].to_list())
    for t in deduped_tickers:
        t_data = ohlcv_df.filter(pl.col("ticker") == t).sort("dt")
        t_dates = t_data["dt"].to_list()
        t_closes = t_data["close"].to_list()
        if len(t_closes) == 0:
            t7_map[t] = 0.0
            hit_map[t] = 0
            continue
        entry_idx = None
        for i, d in enumerate(t_dates):
            if d >= test_start:
                entry_idx = i
                break
        if entry_idx is None:
            t7_map[t] = 0.0
            hit_map[t] = 0
            continue
        entry_close = t_closes[entry_idx]
        fwd_idx = min(entry_idx + 7, len(t_closes) - 1)
        fwd_close = t_closes[fwd_idx] if fwd_idx > entry_idx else entry_close
        if entry_close > 0:
            t7_map[t] = fwd_close / entry_close - 1.0
        else:
            t7_map[t] = 0.0
        hit_map[t] = 1 if t7_map[t] >= 0.10 else 0

    # ── Build score / return arrays (from deduped set) ──
    score_vals: list[float] = []
    return_vals: list[float] = []
    hit_vals: list[int] = []
    for row in deduped.iter_rows(named=True):
        t = row["ticker"]
        score_vals.append(float(row["breakout_score"]))
        return_vals.append(t7_map.get(t, 0.0))
        hit_vals.append(hit_map.get(t, 0))

    scores_s = pl.Series("score", score_vals, dtype=pl.Float64)
    returns_s = pl.Series("t7_return", return_vals, dtype=pl.Float64)
    hits_s = pl.Series("hit", hit_vals, dtype=pl.Int64)

    # ── IC: Spearman rank correlation (primary metric) ──
    ic = compute_ic(scores_s, returns_s)
    if math.isnan(ic):
        ic = 0.0

    # ── Quantile spread ──
    quantile_spread = 0.0
    n_effective = deduped.height
    if n_effective >= 10:
        n_q = max(1, n_effective // 5)
        order = np.argsort(score_vals)[::-1]
        top_r = [return_vals[i] for i in order[:n_q]]
        bot_r = [return_vals[i] for i in order[-n_q:]]
        top_mean = float(np.mean(top_r)) if top_r else 0.0
        bot_mean = float(np.mean(bot_r)) if bot_r else 0.0
        quantile_spread = top_mean - bot_mean

    # ── Precision@20 / Lift@20 ──
    precision = compute_precision_at_k(scores_s, hits_s, 20)
    base_rate = compute_base_rate(hits_s.cast(pl.Float64))
    if math.isnan(precision):
        precision = 0.0
    lift = precision / base_rate if base_rate and base_rate > 0 else 0.0
    if math.isnan(lift):
        lift = 0.0

    # ── Portfolio-level backtest for Sharpe/MaxDD ──
    from alphascreener.backtrader import run_backtest

    sharpe = 0.0
    max_dd = 0.0
    try:
        # Select top N tickers by breakout_score for portfolio construction
        n_positions = min(20, n_effective)
        top = sorted(
            zip(score_vals, deduped["ticker"].to_list()),
            key=lambda x: x[0],
            reverse=True,
        )[:n_positions]
        top_names = [t for _, t in top]

        # Build ticker_dfs from ohlcv_df for the test period
        ticker_dfs: dict[str, pl.DataFrame] = {}
        for t in top_names:
            t_data = ohlcv_df.filter(
                (pl.col("ticker") == t) & (pl.col("dt") >= test_start) & (pl.col("dt") <= test_end)
            ).sort("dt")
            if t_data.height >= 7:  # need at least enough bars for T+7
                ticker_dfs[t] = t_data

        if len(ticker_dfs) >= 1:
            # Single signal per ticker at test_start so the strategy enters
            # at T+1 and holds for the full test window.
            signal_rows = [
                {"ticker": t, "dt": test_start, "refined_score": 1.0} for t in ticker_dfs
            ]
            signals_df = pl.DataFrame(signal_rows)
            bt = run_backtest(ticker_dfs, signals=signals_df)
            sharpe = bt["metrics"]["sharpe_ratio"]
            max_dd = abs(bt["metrics"]["max_drawdown"])
    except Exception:
        _logger.warning(
            "Portfolio backtest failed for window %s→%s; Sharpe set to 0",
            test_start,
            test_end,
        )

    return WindowResult(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        precision_at_20=precision,
        lift_at_20=lift,
        base_rate=base_rate,
        ic=ic,
        quantile_spread=quantile_spread,
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
    *,
    universe_meta: pl.DataFrame | None = None,
) -> float:
    """Compute average composite score across all windows for given weights."""
    vals = []
    for tr_s, tr_e, te_s, te_e in windows:
        r = _evaluate_window(
            ohlcv_df, weights, tr_s, tr_e, te_s, te_e, universe_meta=universe_meta
        )
        if r is not None:
            vals.append(r.score)
    return sum(vals) / len(vals) if vals else 0.0


def _optimize_grid_search(
    ohlcv_df: pl.DataFrame,
    initial_weights: dict[str, float],
    windows: list[tuple[date, date, date, date]],
    *,
    universe_meta: pl.DataFrame | None = None,
) -> dict[str, float]:
    """Exhaustive per-factor multiplier grid search (baseline strategy)."""
    best_weights = dict(initial_weights)
    multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]

    for factor in list(initial_weights.keys()):
        best_w = initial_weights[factor]
        best_score = _score_one(ohlcv_df, best_weights, windows, universe_meta=universe_meta)
        for m in multipliers:
            test_w = dict(best_weights)
            test_w[factor] = initial_weights[factor] * m
            test_w = _normalize_weights(test_w)
            s = _score_one(ohlcv_df, test_w, windows, universe_meta=universe_meta)
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
    *,
    universe_meta: pl.DataFrame | None = None,
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
        return _score_one(ohlcv_df, w, windows, universe_meta=universe_meta)

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
    universe_meta: pl.DataFrame | None = None,
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
    universe_meta : DataFrame or None
        Optional DataFrame with ``ticker``, ``sector``, ``industry``
        columns.  When provided, industry deduplication (Sector≤3,
        Industry≤2) is applied in each evaluation window so the
        optimisation metrics reflect the real Phase 2 pipeline
        constraints.
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

    # Pre-select universe_meta columns once (avoid per-window .select() overhead)
    if universe_meta is not None and universe_meta.height > 0:
        universe_meta = universe_meta.select(["ticker", "sector", "industry"])

    report = OptimizeReport(initial_weights=dict(initial_weights), final_weights={})

    if not initial_weights:
        report.converged = False
        report.iterations = 0
        return report

    weights = dict(initial_weights)
    for tr_s, tr_e, te_s, te_e in windows:
        r = _evaluate_window(
            ohlcv_df, weights, tr_s, tr_e, te_s, te_e, universe_meta=universe_meta
        )
        if r is not None:
            report.windows.append(r)

    if strategy == "tpe":
        best_weights = _optimize_tpe(
            ohlcv_df, initial_weights, windows, n_trials, universe_meta=universe_meta
        )
    else:
        best_weights = _optimize_grid_search(
            ohlcv_df, initial_weights, windows, universe_meta=universe_meta
        )

    report.final_weights = dict(best_weights)
    report.iterations = len(windows)
    report.converged = any(
        abs(best_weights.get(k, 0) - initial_weights.get(k, 0)) > 0.001 for k in initial_weights
    )
    return report
