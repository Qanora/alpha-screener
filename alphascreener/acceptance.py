"""Alpha acceptance metrics computation (Issue #101).

Reference: PRD 5.5.

Computes the following metrics for alpha model validation:
  - base_rate: Fraction of full-market tickers with T+7 return >= 10%
  - Precision@K: Fraction of top K tickers by score that are hits
  - Lift@K: Precision@K / base_rate
  - Recall@K: Top K hits / total market hits
  - IC: Spearman rank correlation between scores and actual returns
  - Block-bootstrap 95% CI: block size=5 trading days x 1000 resamples

Writes results to the ``acceptance_daily`` SQLite table.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from alphascreener.db.models import AlphaAcceptanceDaily
from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HIT_THRESHOLD: float = 0.10  # T+7 return >= 10%
DEFAULT_K_VALUES: tuple[int, ...] = (10, 20)
DEFAULT_BLOCK_SIZE: int = 5  # trading days
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_CI_PCT: float = 95.0

# ---------------------------------------------------------------------------
# 1. base_rate
# ---------------------------------------------------------------------------


def compute_base_rate(
    t7_returns: pl.Series,
    threshold: float = DEFAULT_HIT_THRESHOLD,
) -> float:
    """Compute base_rate = fraction of tickers with T+7 return >= threshold.

    Args:
        t7_returns: Series of T+7 forward returns per ticker.
        threshold: Hit threshold (default 0.10 = 10%).

    Returns:
        Fraction in [0, 1]. Null values are excluded from denominator.
    """
    valid = t7_returns.drop_nulls()
    if len(valid) == 0:
        return 0.0
    hits = (valid >= threshold).sum()
    return float(hits / len(valid))


# ---------------------------------------------------------------------------
# Shared top-K selection (used by Precision@K and Recall@K)
# ---------------------------------------------------------------------------


def _select_top_k(
    scores: pl.Series,
    hits: pl.Series,
    k: int,
) -> pl.DataFrame:
    """Return the subset of rows corresponding to the top K by score.

    Ties at the boundary cause all tied tickers to be included, so the
    effective count may exceed k.  Null scores are placed at the bottom.
    """
    if k <= 0:
        raise ValueError(
            f"k must be a positive integer, got {k}. "
            "Adjust compute_all_alpha_metrics / k_values to pass k >= 1."
        )
    n = len(scores)
    if n == 0:
        return pl.DataFrame(
            schema={
                "score": pl.Float64,
                "hit": pl.Int32,
                "_sort_score": pl.Float64,
            },
        )

    df = pl.DataFrame({"score": scores, "hit": hits.cast(pl.Int32)})
    df = df.with_columns(pl.col("score").fill_null(-float("inf")).alias("_sort_score"))
    df = df.sort("_sort_score", descending=True)

    effective_k = min(k, df.height)
    if effective_k == 0:
        return df.clear()

    threshold_score = df["_sort_score"][effective_k - 1]
    return df.filter(pl.col("_sort_score") >= threshold_score)


# ---------------------------------------------------------------------------
# 2. Precision@K
# ---------------------------------------------------------------------------


def compute_precision_at_k(
    scores: pl.Series,
    hits: pl.Series,
    k: int,
) -> float:
    """Compute Precision@K = fraction of top K tickers by score that are hits.

    Ties at the boundary (k-th position) cause all tied tickers to be
    included, which may produce effective K > k.

    Null scores are treated as the lowest possible rank (placed at bottom).

    Args:
        scores: Score per ticker (higher = better).
        hits: Boolean series: True if T+7 return >= hit threshold.
        k: Number of top tickers to consider.

    Returns:
        Fraction in [0, 1]. Returns 0.0 if input is empty.
    """
    top_k = _select_top_k(scores, hits, k)
    if top_k.height == 0:
        return 0.0
    n_hits = top_k["hit"].sum()
    return float(n_hits / top_k.height)


# ---------------------------------------------------------------------------
# 3. Lift@K
# ---------------------------------------------------------------------------


def compute_lift_at_k(precision: float, base_rate: float) -> float | None:
    """Compute Lift@K = Precision@K / base_rate.

    Args:
        precision: Precision@K value in [0, 1].
        base_rate: Base rate in [0, 1].

    Returns:
        Lift value >= 0. Returns None if base_rate is 0 (undefined).
    """
    if base_rate == 0.0:
        return None
    return precision / base_rate


# ---------------------------------------------------------------------------
# 4. Recall@K
# ---------------------------------------------------------------------------


def compute_recall_at_k(
    scores: pl.Series,
    hits: pl.Series,
    k: int,
) -> float:
    """Compute Recall@K = top K hits / total market hits.

    Args:
        scores: Score per ticker (higher = better).
        hits: Boolean series: True if T+7 return >= hit threshold.
        k: Number of top tickers to consider.

    Returns:
        Fraction in [0, 1]. Returns 0.0 if no hits exist or input is empty.
    """
    total_hits = int(hits.sum())
    if total_hits == 0:
        return 0.0

    top_k = _select_top_k(scores, hits, k)
    if top_k.height == 0:
        return 0.0

    top_k_hits = top_k["hit"].sum()
    return float(top_k_hits / total_hits)


# ---------------------------------------------------------------------------
# 5. IC (Spearman rank correlation)
# ---------------------------------------------------------------------------


def compute_ic(
    scores: pl.Series,
    t7_returns: pl.Series,
) -> float:
    """Compute IC = Spearman rank correlation between scores and T+7 returns.

    Nulls in either series cause the pair to be dropped.
    Requires at least 3 valid data points.

    Args:
        scores: Score per ticker (higher = better).
        t7_returns: T+7 forward return per ticker.

    Returns:
        Spearman rank correlation in [-1, 1]. Returns 0.0 if fewer than 3
        valid data points or all values are constant.
    """
    valid_mask = ~scores.is_null() & ~t7_returns.is_null()
    valid_scores = scores.filter(valid_mask)
    valid_returns = t7_returns.filter(valid_mask)

    n = len(valid_scores)
    if n < 3:
        return 0.0

    score_arr = valid_scores.to_numpy()
    ret_arr = valid_returns.to_numpy()

    # Check for zero variance (all values equal)
    if np.std(score_arr) < 1e-15 or np.std(ret_arr) < 1e-15:
        return 0.0

    result = scipy_stats.spearmanr(score_arr, ret_arr)
    corr = result.correlation

    if corr is None or math.isnan(corr):
        return 0.0

    return float(corr)


# ---------------------------------------------------------------------------
# 6. Block-bootstrap 95% CI
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_samples: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI_PCT,
) -> tuple[float, float]:
    """Compute block-bootstrap confidence interval for a statistic.

    Uses the stationary block bootstrap: resamples contiguous blocks of
    length *block_size* with replacement to preserve serial dependence.

    Args:
        data: 1-D numpy array of observations.
        statistic_fn: Callable that computes a scalar statistic from a sample.
        block_size: Number of contiguous observations per block.
        n_samples: Number of bootstrap resamples.
        ci: Confidence level in percent (default 95.0).

    Returns:
        ``(lower, upper)`` bounds of the confidence interval.
        Returns ``(0.0, 0.0)`` if data is empty.
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0

    rng = np.random.default_rng()
    boot_stats = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        sample = _block_resample(data, block_size, rng)
        boot_stats[i] = statistic_fn(sample)

    alpha = (100.0 - ci) / 100.0
    lower = np.percentile(boot_stats, alpha / 2.0 * 100.0)
    upper = np.percentile(boot_stats, (1.0 - alpha / 2.0) * 100.0)

    return float(lower), float(upper)


def _block_resample(
    data: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single block-bootstrap resample.

    Draws contiguous blocks of length *block_size* with replacement until
    the resample length equals or exceeds len(data), then truncates to len(data).
    """
    n = len(data)
    if n == 0:
        return data.copy()

    effective_block = min(block_size, n)
    n_blocks = int(math.ceil(n / effective_block))

    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, n - effective_block + 1)
        result.append(data[start : start + effective_block])

    resampled = np.concatenate(result)
    return resampled[:n]


# ---------------------------------------------------------------------------
# 7. compute_all_alpha_metrics
# ---------------------------------------------------------------------------


def compute_all_alpha_metrics(
    df: pl.DataFrame,
    *,
    score_col_pure: str = "breakout_score",
    score_col_llm: str = "refined_score",
    return_col: str = "t7_return",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_pct: float = DEFAULT_CI_PCT,
) -> dict[str, float | None | int]:
    """Compute all alpha acceptance metrics from a single DataFrame.

    The DataFrame must contain score columns (pure and LLM) and a T+7 return
    column. The caller is responsible for joining scores with returns.

    The bootstrap CI is computed on the sequence of per-ticker hits for each
    score track: for a given score, we compute precision@20_nohit, then the
    CI is reported for that metric. We use the precision at K=20 as the
    primary statistic for the bootstrap CI.

    Args:
        df: DataFrame with score and return columns.
        score_col_pure: Column name for pure (quant) score.
        score_col_llm: Column name for LLM-refined score.
        return_col: Column name for T+7 forward return.
        k_values: K values for Precision/Recall/Lift computation.
        hit_threshold: T+7 return threshold for a hit (default 0.10).
        block_size: Block size for bootstrap (trading days).
        n_bootstrap: Number of bootstrap resamples.
        ci_pct: Confidence interval level in percent.

    Returns:
        Dict with all metric keys matching the acceptance_daily schema.

    Raises:
        ValueError: If required columns are missing from *df*.
    """
    required = {score_col_pure, score_col_llm, return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. Input has: {sorted(df.columns)}"
        )

    n_total = df.height
    _logger.info("Computing alpha acceptance metrics for %d tickers", n_total)

    # --- base_rate (full market) ---
    t7 = df[return_col]
    base_rate = compute_base_rate(t7, threshold=hit_threshold)

    # --- Hit mask ---
    hits = t7 >= hit_threshold

    # --- Per-track metrics ---
    results: dict[str, float | None | int] = {
        "base_rate": base_rate,
        "sample_size": n_total,
        "threshold": hit_threshold,
    }

    for track, score_col in [("pure", score_col_pure), ("llm", score_col_llm)]:
        scores = df[score_col]

        # Precision and Recall at each K
        for k in k_values:
            prec = compute_precision_at_k(scores, hits, k)
            rec = compute_recall_at_k(scores, hits, k)
            lift = compute_lift_at_k(prec, base_rate)

            results[f"precision_at_{k}_{track}"] = prec
            results[f"recall_at_{k}_{track}"] = rec
            results[f"lift_at_{k}_{track}"] = lift

        # IC
        ic_val = compute_ic(scores, t7)
        results[f"ic_{track}"] = ic_val

        # Bootstrap CI on precision@K=20 (primary metric)
        # Build a per-ticker hit array ordered by score for bootstrap
        # Actually, compute the precision statistic via bootstrap on paired data
        if n_total > 0:
            hit_arr = hits.to_numpy().astype(np.float64)
            score_arr = scores.fill_null(-float("inf")).to_numpy()

            def precision20_fn(data_2d):
                """Compute precision@20 on a bootstrap resample (block-bootstrap
                on the rows)."""
                # data_2d is shape (n, 2): [score, hit]
                s = data_2d[:, 0]
                h = data_2d[:, 1]
                # Find top 20 by score (descending)
                k = min(20, len(s))
                if k == 0:
                    return 0.0
                # Get indices of top K scores
                idx = np.argpartition(-s, k - 1)[:k]
                # Tie-breaking: include all >= threshold score
                threshold = s[idx[k - 1]]
                top_mask = s >= threshold
                top_hits = h[top_mask]
                if len(top_hits) == 0:
                    return 0.0
                return float(top_hits.sum() / len(top_hits))

            paired = np.column_stack([score_arr, hit_arr])
            ci_lower, ci_upper = block_bootstrap_ci(
                paired,
                precision20_fn,
                block_size=block_size,
                n_samples=n_bootstrap,
                ci=ci_pct,
            )
        else:
            ci_lower, ci_upper = 0.0, 0.0

        results[f"bootstrap_ci_lower_{track}"] = ci_lower
        results[f"bootstrap_ci_upper_{track}"] = ci_upper

    _logger.info(
        "Alpha metrics: base_rate=%.4f, ic_pure=%.4f, precision@20_pure=%.4f, n=%d",
        base_rate,
        results.get("ic_pure", 0.0),
        results.get("precision_at_20_pure", 0.0),
        n_total,
    )

    return results


# ---------------------------------------------------------------------------
# 8. Database write
# ---------------------------------------------------------------------------


def write_acceptance(
    metrics: dict[str, float | None | int | date],
    db_engine: Engine,
) -> None:
    """Write alpha acceptance metrics to the ``acceptance_daily`` table.

    Uses upsert semantics (merge): if a record for the same ``metric_date``
    already exists, it is updated in place.

    Args:
        metrics: Metric dict as returned by :func:`compute_all_alpha_metrics`,
            with an additional ``metric_date`` key.
        db_engine: SQLAlchemy Engine for the target SQLite database.

    Raises:
        KeyError: If ``metric_date`` is missing from *metrics*.
    """
    metric_date = metrics["metric_date"]
    if isinstance(metric_date, str):
        metric_date = date.fromisoformat(metric_date)

    _logger.info("Writing alpha acceptance metrics for %s", metric_date.isoformat())

    with Session(db_engine) as session:
        existing = session.get(AlphaAcceptanceDaily, metric_date)
        if existing is not None:
            # Update existing
            _update_model(existing, metrics)
        else:
            # Insert new
            record = AlphaAcceptanceDaily(metric_date=metric_date)
            _update_model(record, metrics)
            session.add(record)

        session.commit()

    _logger.info("Alpha acceptance metrics persisted for %s", metric_date.isoformat())


def _update_model(record: AlphaAcceptanceDaily, metrics: dict) -> None:
    """Populate an AlphaAcceptanceDaily model instance from a metrics dict."""
    record.base_rate = float(metrics["base_rate"])
    record.precision_at_20_pure = _maybe_float(metrics.get("precision_at_20_pure"))
    record.precision_at_20_llm = _maybe_float(metrics.get("precision_at_20_llm"))
    record.precision_at_10_pure = _maybe_float(metrics.get("precision_at_10_pure"))
    record.precision_at_10_llm = _maybe_float(metrics.get("precision_at_10_llm"))
    record.lift_at_20_pure = _maybe_float(metrics.get("lift_at_20_pure"))
    record.lift_at_20_llm = _maybe_float(metrics.get("lift_at_20_llm"))
    record.ic_pure = _maybe_float(metrics.get("ic_pure"))
    record.ic_llm = _maybe_float(metrics.get("ic_llm"))
    record.bootstrap_ci_lower_pure = _maybe_float(metrics.get("bootstrap_ci_lower_pure"))
    record.bootstrap_ci_upper_pure = _maybe_float(metrics.get("bootstrap_ci_upper_pure"))
    record.bootstrap_ci_lower_llm = _maybe_float(metrics.get("bootstrap_ci_lower_llm"))
    record.bootstrap_ci_upper_llm = _maybe_float(metrics.get("bootstrap_ci_upper_llm"))
    record.sample_size = int(metrics["sample_size"])
    record.threshold = _maybe_float(metrics.get("threshold"))


def _maybe_float(value: object) -> float | None:
    """Convert a value to float, returning None if the value is None."""
    if value is None:
        return None
    return float(value)
