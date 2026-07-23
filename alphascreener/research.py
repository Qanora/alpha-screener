"""Internal, preregistered walk-forward qualification for ranking algorithms.

This module is deliberately absent from the installed ``asc`` command.  It
compares challengers with the frozen production score on identical daily
universes and exact future outcomes; it never writes the prediction ledger.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from math import ceil
from typing import Any

import numpy as np
import polars as pl

from alphascreener.data.io import scan_ohlcv
from alphascreener.data.sync import MIN_SYNC_COVERAGE, sync_ohlcv
from alphascreener.features import compute_60d_features
from alphascreener.market_calendar import market_dates_between
from alphascreener.prediction_contract import (
    DEFAULT_ABSOLUTE_HIT_RETURN,
    DEFAULT_CROSS_SECTION_HIT_QUANTILE,
    DEFAULT_TOP_K,
    FORECAST_HORIZON_SESSIONS,
    INPUT_LOOKBACK_SESSIONS,
)
from alphascreener.ranking import (
    score_rank_v6,
    select_eligible_candidate_features,
)
from alphascreener.research_features import (
    LIGHTGBM_FEATURES,
    LINEAR_FEATURES,
    RESEARCH_MARKET_FEATURES,
    RESEARCH_RANK_FEATURES,
    RESEARCH_RAW_TREE_FEATURES,
    RESEARCH_STOCK_FEATURES,
    add_cross_sectional_ranks,
    compute_research_features,
)

EVIDENCE_TYPE = "CURRENT_SURVIVOR_UNIVERSE_RESEARCH_DIAGNOSTIC"
INCUMBENT_NAME = "rank-v6"
LINEAR_NAME = "ridge-rank-v1"
LIGHTGBM_NAME = "lambdamart-rank-v1"
FEATURE_VERSION = "ohlcv-60d-v1"
RESEARCH_CHUNK_DECISION_DATES = 63

_REQUIRED_OHLCV_COLUMNS = {
    "ticker",
    "dt",
    "open",
    "high",
    "low",
    "close",
    "raw_close",
    "volume",
}
_MODEL_COLUMNS = (
    *RESEARCH_RANK_FEATURES,
    *RESEARCH_RAW_TREE_FEATURES,
    *RESEARCH_MARKET_FEATURES,
)
_logger = logging.getLogger(__name__)

LGBM_RANK_V1_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [DEFAULT_TOP_K],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": DEFAULT_TOP_K + 3,
    "lambdarank_norm": True,
    "boosting": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 2_000,
    "lambda_l2": 10.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_by_query": True,
    "seed": 20_260_723,
    "data_random_seed": 20_260_723,
    "feature_fraction_seed": 20_260_723,
    "bagging_seed": 20_260_723,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}


@dataclass(frozen=True)
class ResearchConfig:
    """Frozen validation protocol; command-line flags cannot tune it."""

    minimum_training_dates: int = 504
    validation_dates: int = 126
    locked_test_dates: int = 252
    retrain_interval: int = 63
    bootstrap_block_dates: int = FORECAST_HORIZON_SESSIONS
    bootstrap_replications: int = 10_000
    random_seed: int = 20_260_723
    minimum_uplift: float = 0.02
    linear_l2: float = 10.0
    num_boost_round: int = 1_000
    early_stopping_rounds: int = 50
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self) -> None:
        positive = {
            "minimum_training_dates": self.minimum_training_dates,
            "validation_dates": self.validation_dates,
            "locked_test_dates": self.locked_test_dates,
            "retrain_interval": self.retrain_interval,
            "bootstrap_block_dates": self.bootstrap_block_dates,
            "bootstrap_replications": self.bootstrap_replications,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        if invalid := [name for name, value in positive.items() if value <= 0]:
            raise ValueError(f"research config values must be positive: {invalid}")
        if self.bootstrap_block_dates < FORECAST_HORIZON_SESSIONS:
            raise ValueError("bootstrap blocks cannot be shorter than the label horizon")
        if self.minimum_uplift <= 0 or self.linear_l2 <= 0:
            raise ValueError("uplift and linear regularization must be positive")


@dataclass(frozen=True)
class ResearchData:
    """Separated features/outcomes and their immutable local snapshot identity."""

    panel: pl.DataFrame
    date_quality: pl.DataFrame
    valid_dates: tuple[date, ...]
    snapshot_id: str
    feature_digest: str


@dataclass(frozen=True)
class WalkForwardFold:
    """One anchored, point-in-time model fit and its following test block."""

    model_as_of: date
    training_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    maximum_training_result_date: date


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower_95: float
    upper_95: float
    probability_positive: float
    paired_dates: int


@dataclass(frozen=True)
class ResearchReport:
    """All evidence needed to accept or reject the frozen challenger."""

    daily: pl.DataFrame
    summary: pl.DataFrame
    folds: pl.DataFrame
    stability: pl.DataFrame
    bootstrap: BootstrapInterval
    recent_45_uplift: float
    promotion_passed: bool
    promotion_reasons: tuple[str, ...]
    snapshot_id: str
    feature_digest: str
    config_digest: str


def build_research_data(
    ohlcv: pl.DataFrame | pl.LazyFrame,
    *,
    chunk_decision_dates: int = RESEARCH_CHUNK_DECISION_DATES,
) -> ResearchData:
    """Build paired research rows in bounded date chunks.

    Every chunk includes the exact 59 earlier sessions needed by a 60-session
    feature and the 14 later sessions needed by the outcome.  Chunking changes
    only peak memory; daily universes, features, labels, and ordering are
    identical to a full-history calculation.
    """
    if chunk_decision_dates <= 0:
        raise ValueError("chunk_decision_dates must be positive")
    source = ohlcv.lazy() if isinstance(ohlcv, pl.DataFrame) else ohlcv
    if missing := _REQUIRED_OHLCV_COLUMNS - set(source.collect_schema().names()):
        raise ValueError(f"research OHLCV data missing columns: {sorted(missing)}")
    source = source.select(sorted(_REQUIRED_OHLCV_COLUMNS))
    bounds = source.select(
        pl.col("dt").cast(pl.Date).min().alias("start"),
        pl.col("dt").cast(pl.Date).max().alias("end"),
    ).collect().row(0, named=True)
    if bounds["start"] is None or bounds["end"] is None:
        raise ValueError("research OHLCV data is empty")
    market_dates = market_dates_between(bounds["start"], bounds["end"])
    first_decision_index = INPUT_LOOKBACK_SESSIONS - 1
    last_decision_index = len(market_dates) - FORECAST_HORIZON_SESSIONS
    if last_decision_index <= first_decision_index:
        raise ValueError("not enough market sessions to build matured research samples")
    decision_dates = market_dates[first_decision_index:last_decision_index]
    date_positions = {value: index for index, value in enumerate(market_dates)}
    panel_chunks: list[pl.DataFrame] = []
    quality_chunks: list[pl.DataFrame] = []
    total_chunks = ceil(len(decision_dates) / chunk_decision_dates)
    for offset in range(0, len(decision_dates), chunk_decision_dates):
        chunk_dates = decision_dates[offset : offset + chunk_decision_dates]
        _logger.info(
            "Building research panel chunk %d/%d (%s through %s)",
            offset // chunk_decision_dates + 1,
            total_chunks,
            chunk_dates[0],
            chunk_dates[-1],
        )
        first_position = date_positions[chunk_dates[0]]
        last_position = date_positions[chunk_dates[-1]]
        window_dates = market_dates[
            first_position - INPUT_LOOKBACK_SESSIONS + 1 :
            last_position + FORECAST_HORIZON_SESSIONS + 1
        ]
        window = (
            source.filter(pl.col("dt").is_in(window_dates))
            .collect()
            .with_columns(pl.col("dt").cast(pl.Date))
            .unique(subset=["ticker", "dt"], keep="last")
            .sort(["ticker", "dt"])
        )
        result_calendar = pl.DataFrame({
            "dt": chunk_dates,
            "result_date": [
                market_dates[date_positions[value] + FORECAST_HORIZON_SESSIONS]
                for value in chunk_dates
            ],
        })
        built = _build_research_chunk(window, chunk_dates, result_calendar)
        if built is not None:
            panel_chunk, quality_chunk = built
            panel_chunks.append(panel_chunk)
            quality_chunks.append(quality_chunk)
        del window
        gc.collect()
    if not panel_chunks:
        raise ValueError("no eligible research candidates")

    panel = pl.concat(panel_chunks, rechunk=False)
    observed_quality = pl.concat(quality_chunks, rechunk=False).drop("result_date")
    del panel_chunks, quality_chunks
    gc.collect()
    decision_calendar = pl.DataFrame({
        "dt": decision_dates,
        "result_date": [
            market_dates[date_positions[value] + FORECAST_HORIZON_SESSIONS]
            for value in decision_dates
        ],
    })
    quality = decision_calendar.join(
        observed_quality,
        on="dt",
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("universe_size").is_null().alias("_missing_universe")
    ).with_columns(
        pl.col("universe_size").fill_null(0),
        pl.col("outcome_count").fill_null(0),
        pl.col("outcome_coverage").fill_null(0.0),
        pl.col("date_valid").fill_null(False),
        pl.when(pl.col("_missing_universe"))
        .then(pl.lit(f"eligible_universe_below_top_{DEFAULT_TOP_K}"))
        .otherwise(pl.col("invalid_reason"))
        .alias("invalid_reason"),
    ).drop("_missing_universe").sort("dt")
    valid_dates = tuple(
        quality.filter(pl.col("date_valid"))["dt"].sort().to_list()
    )
    if not valid_dates:
        raise ValueError("no research dates have complete eligible-universe outcomes")

    snapshot_id = _snapshot_id(source)
    feature_digest = _digest_json({
        "version": FEATURE_VERSION,
        "chunk_decision_dates": chunk_decision_dates,
        "stock": RESEARCH_STOCK_FEATURES,
        "linear": LINEAR_FEATURES,
        "lightgbm": LIGHTGBM_FEATURES,
    })
    return ResearchData(
        panel=panel,
        date_quality=quality.rename({"dt": "decision_date"}),
        valid_dates=valid_dates,
        snapshot_id=snapshot_id,
        feature_digest=feature_digest,
    )


def _build_research_chunk(
    data: pl.DataFrame,
    decision_dates: list[date],
    result_calendar: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Build one bounded panel chunk with the same global daily contract."""
    base_features = compute_60d_features(data)
    eligible = select_eligible_candidate_features(base_features, decision_dates)
    if eligible.is_empty():
        return None
    incumbent = score_rank_v6(eligible).rename({
        "decision_date": "dt",
        "score": "rank_v6_score",
        "rank": "rank_v6_rank",
    })
    eligible_keys = eligible.select("ticker", "dt")
    research_tickers = eligible["ticker"].unique().to_list()
    if "SPY" not in research_tickers:
        research_tickers.append("SPY")
    del base_features, eligible

    research_features = compute_research_features(
        data.filter(pl.col("ticker").is_in(research_tickers))
    )
    candidates = research_features.join(
        eligible_keys,
        on=["ticker", "dt"],
        how="semi",
    )
    del research_features, eligible_keys
    candidates = add_cross_sectional_ranks(candidates).join(
        incumbent,
        on=["ticker", "dt"],
        how="inner",
        validate="1:1",
    )
    future_prices = data.select(
        "ticker",
        pl.col("dt").alias("result_date"),
        pl.col("close").alias("future_close"),
    )
    candidates = candidates.join(result_calendar, on="dt", how="inner").join(
        future_prices,
        on=["ticker", "result_date"],
        how="left",
        validate="m:1",
    ).with_columns(
        (pl.col("future_close") / pl.col("close") - 1.0).alias("forward_return")
    )
    quality = candidates.group_by("dt").agg(
        pl.col("result_date").first(),
        pl.len().cast(pl.Int64).alias("universe_size"),
        pl.col("forward_return").is_not_null().sum().cast(pl.Int64).alias(
            "outcome_count"
        ),
    ).with_columns(
        (pl.col("outcome_count") / pl.col("universe_size")).alias(
            "outcome_coverage"
        ),
        (
            (pl.col("universe_size") >= DEFAULT_TOP_K)
            & (pl.col("outcome_count") == pl.col("universe_size"))
        ).alias("date_valid"),
    ).with_columns(
        pl.when(pl.col("universe_size") < DEFAULT_TOP_K)
        .then(pl.lit(f"eligible_universe_below_top_{DEFAULT_TOP_K}"))
        .when(pl.col("outcome_count") != pl.col("universe_size"))
        .then(pl.lit("complete_universe_outcomes_required"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("invalid_reason")
    ).sort("dt")
    thresholds = _complete_pool_thresholds(candidates, quality)
    candidates = candidates.join(
        quality.select(
            "dt",
            "universe_size",
            "outcome_coverage",
            "date_valid",
        ),
        on="dt",
        how="inner",
        validate="m:1",
    ).join(
        thresholds,
        on="dt",
        how="left",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("date_valid"))
        .then(pl.col("forward_return") >= pl.col("hit_threshold"))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("is_explosion")
    ).rename({"dt": "decision_date"})
    _require_finite_features(
        candidates.filter(pl.col("date_valid")),
        list(_MODEL_COLUMNS),
    )
    panel = candidates.select(
        "ticker",
        "decision_date",
        "result_date",
        "universe_size",
        "outcome_coverage",
        "date_valid",
        "forward_return",
        "hit_threshold",
        "is_explosion",
        "rank_v6_score",
        "rank_v6_rank",
        *_MODEL_COLUMNS,
    ).sort(["decision_date", "ticker"])
    return panel, quality


def make_walk_forward_folds(
    data: ResearchData,
    config: ResearchConfig,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding fits with purged training and a final locked test year."""
    valid_dates = data.valid_dates
    minimum = (
        config.minimum_training_dates
        + config.validation_dates
        + config.locked_test_dates
    )
    if len(valid_dates) < minimum:
        raise ValueError(
            f"research needs at least {minimum} complete mature dates; "
            f"found {len(valid_dates)}"
        )
    locked_dates = valid_dates[-config.locked_test_dates :]
    quality_by_date = {
        row["decision_date"]: row["result_date"]
        for row in data.date_quality.filter(pl.col("date_valid")).iter_rows(named=True)
    }
    folds: list[WalkForwardFold] = []
    for offset in range(0, len(locked_dates), config.retrain_interval):
        test_dates = locked_dates[offset : offset + config.retrain_interval]
        model_as_of = test_dates[0]
        matured_dates = tuple(
            decision_date
            for decision_date in valid_dates
            if decision_date < model_as_of
            and quality_by_date[decision_date] < model_as_of
        )
        if len(matured_dates) < config.validation_dates:
            raise ValueError(f"not enough matured dates before {model_as_of}")
        validation_dates = matured_dates[-config.validation_dates :]
        validation_start = validation_dates[0]
        training_dates = tuple(
            decision_date
            for decision_date in matured_dates[: -config.validation_dates]
            if quality_by_date[decision_date] < validation_start
        )
        if len(training_dates) < config.minimum_training_dates:
            raise ValueError(
                f"model as of {model_as_of} has {len(training_dates)} purged training "
                f"dates; {config.minimum_training_dates} required"
            )
        maximum_training_result_date = max(
            quality_by_date[decision_date] for decision_date in training_dates
        )
        if maximum_training_result_date >= validation_start:
            raise AssertionError("training outcome overlaps the validation period")
        if max(quality_by_date[value] for value in validation_dates) >= model_as_of:
            raise AssertionError("validation outcome was unavailable at model_as_of")
        folds.append(
            WalkForwardFold(
                model_as_of=model_as_of,
                training_dates=training_dates,
                validation_dates=validation_dates,
                test_dates=test_dates,
                maximum_training_result_date=maximum_training_result_date,
            )
        )
    return tuple(folds)


class RidgeRanker:
    """Small deterministic regularized linear probability rank baseline."""

    def __init__(self, *, l2: float) -> None:
        self._l2 = l2
        self._coefficients: np.ndarray | None = None

    def fit(self, frame: pl.DataFrame) -> None:
        features = frame.select(LINEAR_FEATURES).to_numpy().astype(np.float64)
        labels = frame["is_explosion"].cast(pl.Int8).to_numpy().astype(np.float64)
        date_counts = frame.group_by("decision_date").len().sort("decision_date")
        group_sizes = date_counts["len"].to_numpy()
        weights = np.concatenate([
            np.full(int(size), 1.0 / float(size), dtype=np.float64)
            for size in group_sizes
        ])
        positive_weight = float(weights[labels == 1].sum())
        negative_weight = float(weights[labels == 0].sum())
        if positive_weight <= 0 or negative_weight <= 0:
            raise ValueError("linear training data must contain both label classes")
        weights[labels == 1] *= 0.5 / positive_weight
        weights[labels == 0] *= 0.5 / negative_weight
        design = np.column_stack([np.ones(features.shape[0]), features])
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_labels = labels * np.sqrt(weights)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self._l2
        penalty[0, 0] = 0.0
        gram = weighted_design.T @ weighted_design + penalty
        target = weighted_design.T @ weighted_labels
        try:
            self._coefficients = np.linalg.solve(gram, target)
        except np.linalg.LinAlgError:
            self._coefficients = np.linalg.lstsq(gram, target, rcond=None)[0]

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("linear ranker has not been fitted")
        features = frame.select(LINEAR_FEATURES).to_numpy().astype(np.float64)
        design = np.column_stack([np.ones(features.shape[0]), features])
        return design @ self._coefficients


def run_walk_forward_research(
    data: ResearchData,
    *,
    config: ResearchConfig = ResearchConfig(),
) -> ResearchReport:
    """Train frozen challengers and compare them on one locked test year."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required; run `uv sync --extra research` first"
        ) from exc

    folds = make_walk_forward_folds(data, config)
    scored_blocks: list[pl.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold in folds:
        train = _frame_for_dates(data.panel, fold.training_dates)
        validation = _frame_for_dates(data.panel, fold.validation_dates)
        test = _frame_for_dates(data.panel, fold.test_dates)
        if test.is_empty():
            raise ValueError(f"locked test block beginning {fold.model_as_of} is empty")

        linear = RidgeRanker(l2=config.linear_l2)
        linear.fit(train)
        linear_scores = linear.score(test)

        train_groups = _group_sizes(train)
        validation_groups = _group_sizes(validation)
        train_set = lgb.Dataset(
            train.select(LIGHTGBM_FEATURES),
            label=train["is_explosion"].cast(pl.Int8),
            group=train_groups,
            feature_name=list(LIGHTGBM_FEATURES),
            free_raw_data=True,
        )
        validation_set = lgb.Dataset(
            validation.select(LIGHTGBM_FEATURES),
            label=validation["is_explosion"].cast(pl.Int8),
            group=validation_groups,
            feature_name=list(LIGHTGBM_FEATURES),
            reference=train_set,
            free_raw_data=True,
        )
        model = lgb.train(
            LGBM_RANK_V1_PARAMS,
            train_set,
            num_boost_round=config.num_boost_round,
            valid_sets=[validation_set],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                    min_delta=config.early_stopping_min_delta,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        lightgbm_scores = model.predict(
            test.select(LIGHTGBM_FEATURES),
            num_iteration=model.best_iteration,
        )
        scored_blocks.append(
            test.select(
                "ticker",
                "decision_date",
                "result_date",
                "universe_size",
                "is_explosion",
                "rank_v6_score",
                "spy_return_20d",
            ).with_columns(
                pl.Series("linear_score", linear_scores),
                pl.Series("lightgbm_score", lightgbm_scores),
            )
        )
        audits.append({
            "model_as_of": fold.model_as_of,
            "maximum_training_result_date": fold.maximum_training_result_date,
            "training_dates": len(fold.training_dates),
            "training_rows": train.height,
            "validation_dates": len(fold.validation_dates),
            "validation_rows": validation.height,
            "test_dates": len(fold.test_dates),
            "test_rows": test.height,
            "lightgbm_best_iteration": int(model.best_iteration),
        })
        del train, validation, test, train_set, validation_set, model
        gc.collect()

    scored = pl.concat(scored_blocks).sort(["decision_date", "ticker"])
    daily = pl.concat([
        evaluate_daily_scores(scored, "rank_v6_score", INCUMBENT_NAME),
        evaluate_daily_scores(scored, "linear_score", LINEAR_NAME),
        evaluate_daily_scores(scored, "lightgbm_score", LIGHTGBM_NAME),
    ]).sort(["strategy", "decision_date"])
    summary = summarize_daily_scores(daily)
    stability = paired_stability(daily, folds)
    bootstrap = paired_date_block_bootstrap(
        daily.filter(pl.col("strategy") == LIGHTGBM_NAME),
        daily.filter(pl.col("strategy") == INCUMBENT_NAME),
        block_dates=config.bootstrap_block_dates,
        replications=config.bootstrap_replications,
        seed=config.random_seed,
    )
    recent = daily.filter(
        pl.col("decision_date").is_in(list(data.valid_dates[-45:]))
    )
    recent_precision = {
        row["strategy"]: row["precision_at_10"]
        for row in summarize_daily_scores(recent).iter_rows(named=True)
    }
    recent_45_uplift = (
        recent_precision[LIGHTGBM_NAME] - recent_precision[INCUMBENT_NAME]
    )
    promotion_passed, reasons = promotion_decision(
        summary,
        bootstrap,
        recent_45_uplift=recent_45_uplift,
        minimum_uplift=config.minimum_uplift,
        required_dates=config.locked_test_dates,
        stability=stability,
    )
    config_digest = _digest_json({
        "protocol": asdict(config),
        "lightgbm": LGBM_RANK_V1_PARAMS,
        "lightgbm_version": lgb.__version__,
    })
    return ResearchReport(
        daily=daily,
        summary=summary,
        folds=pl.DataFrame(audits),
        stability=stability,
        bootstrap=bootstrap,
        recent_45_uplift=recent_45_uplift,
        promotion_passed=promotion_passed,
        promotion_reasons=reasons,
        snapshot_id=data.snapshot_id,
        feature_digest=data.feature_digest,
        config_digest=config_digest,
    )


def evaluate_daily_scores(
    scored: pl.DataFrame,
    score_column: str,
    strategy: str,
) -> pl.DataFrame:
    """Evaluate a score on complete paired daily groups with deterministic ties."""
    required = {
        "ticker",
        "decision_date",
        "result_date",
        "universe_size",
        "is_explosion",
        "spy_return_20d",
        score_column,
    }
    if missing := required - set(scored.columns):
        raise ValueError(f"scored frame missing columns: {sorted(missing)}")
    ordered = scored.sort(
        ["decision_date", score_column, "ticker"],
        descending=[False, True, False],
    ).with_columns(
        pl.col("ticker").cum_count().over("decision_date").alias("_model_rank")
    )
    return ordered.group_by("decision_date").agg(
        pl.col("result_date").first(),
        pl.col("universe_size").first(),
        pl.col("spy_return_20d").first(),
        pl.col("is_explosion").mean().alias("base_explosion_rate"),
        pl.when(pl.col("_model_rank") <= DEFAULT_TOP_K)
        .then(pl.col("is_explosion").cast(pl.Int64))
        .otherwise(0)
        .sum()
        .cast(pl.Int64)
        .alias("hits_at_10"),
    ).with_columns(
        pl.lit(strategy).alias("strategy"),
        (pl.col("hits_at_10") / DEFAULT_TOP_K).alias("precision_at_10"),
    ).with_columns(
        (
            (pl.col("precision_at_10") >= 0.10)
            & (pl.col("precision_at_10") > pl.col("base_explosion_rate"))
        ).alias("passed")
    ).select(
        "strategy",
        "decision_date",
        "result_date",
        "universe_size",
        "spy_return_20d",
        "hits_at_10",
        "precision_at_10",
        "base_explosion_rate",
        "passed",
    ).sort("decision_date")


def summarize_daily_scores(daily: pl.DataFrame) -> pl.DataFrame:
    """Return equal-date pooled Precision@10 and pass rates by strategy."""
    if daily.is_empty():
        raise ValueError("daily research metrics are empty")
    return daily.group_by("strategy").agg(
        pl.len().cast(pl.Int64).alias("valid_dates"),
        pl.col("hits_at_10").sum().cast(pl.Int64).alias("hits_at_10"),
        (pl.col("hits_at_10").sum() / (pl.len() * DEFAULT_TOP_K)).alias(
            "precision_at_10"
        ),
        pl.col("base_explosion_rate").mean().alias("mean_base_explosion_rate"),
        pl.col("passed").mean().alias("passing_date_rate"),
    ).sort("strategy")


def paired_date_block_bootstrap(
    challenger: pl.DataFrame,
    incumbent: pl.DataFrame,
    *,
    block_dates: int,
    replications: int,
    seed: int,
) -> BootstrapInterval:
    """Moving-block bootstrap of paired daily Precision@10 differences."""
    if block_dates < FORECAST_HORIZON_SESSIONS:
        raise ValueError("bootstrap block is shorter than the outcome horizon")
    if replications <= 0:
        raise ValueError("bootstrap replications must be positive")
    left = challenger.select(
        "decision_date",
        pl.col("precision_at_10").alias("challenger_precision"),
    )
    right = incumbent.select(
        "decision_date",
        pl.col("precision_at_10").alias("incumbent_precision"),
    )
    if set(left["decision_date"].to_list()) != set(right["decision_date"].to_list()):
        raise ValueError("challenger and incumbent dates must match exactly")
    paired = left.join(right, on="decision_date", how="inner", validate="1:1").sort(
        "decision_date"
    )
    deltas = (
        paired["challenger_precision"] - paired["incumbent_precision"]
    ).to_numpy()
    count = len(deltas)
    if count < block_dates * 5:
        raise ValueError("at least five complete bootstrap blocks are required")
    rng = np.random.default_rng(seed)
    blocks_per_sample = ceil(count / block_dates)
    starts = rng.integers(
        0,
        count - block_dates + 1,
        size=(replications, blocks_per_sample),
    )
    offsets = np.arange(block_dates)
    indices = (starts[..., None] + offsets).reshape(replications, -1)[:, :count]
    sampled = deltas[indices].mean(axis=1)
    return BootstrapInterval(
        estimate=float(deltas.mean()),
        lower_95=float(np.quantile(sampled, 0.025)),
        upper_95=float(np.quantile(sampled, 0.975)),
        probability_positive=float((sampled > 0).mean()),
        paired_dates=count,
    )


def paired_stability(
    daily: pl.DataFrame,
    folds: tuple[WalkForwardFold, ...],
) -> pl.DataFrame:
    """Report paired uplift in every test block and both simple market regimes."""
    challenger = daily.filter(pl.col("strategy") == LIGHTGBM_NAME).select(
        "decision_date",
        "spy_return_20d",
        pl.col("precision_at_10").alias("challenger_precision"),
    )
    incumbent = daily.filter(pl.col("strategy") == INCUMBENT_NAME).select(
        "decision_date",
        pl.col("precision_at_10").alias("incumbent_precision"),
    )
    paired = challenger.join(
        incumbent,
        on="decision_date",
        how="inner",
        validate="1:1",
    ).with_columns(
        (pl.col("challenger_precision") - pl.col("incumbent_precision")).alias(
            "uplift"
        )
    )
    segments: list[dict[str, object]] = []
    for index, fold in enumerate(folds, start=1):
        subset = paired.filter(pl.col("decision_date").is_in(list(fold.test_dates)))
        segments.append({
            "segment": f"test_block_{index}",
            "dates": subset.height,
            "precision_uplift": float(subset["uplift"].mean()),
        })
    for name, predicate in (
        ("spy_20d_nonnegative", pl.col("spy_return_20d") >= 0),
        ("spy_20d_negative", pl.col("spy_return_20d") < 0),
    ):
        subset = paired.filter(predicate)
        if subset.is_empty():
            raise ValueError(f"locked test contains no dates for regime {name}")
        segments.append({
            "segment": name,
            "dates": subset.height,
            "precision_uplift": float(subset["uplift"].mean()),
        })
    return pl.DataFrame(segments).sort("segment")


def promotion_decision(
    summary: pl.DataFrame,
    bootstrap: BootstrapInterval,
    *,
    recent_45_uplift: float,
    minimum_uplift: float,
    required_dates: int,
    stability: pl.DataFrame,
) -> tuple[bool, tuple[str, ...]]:
    """Apply all preregistered gates without discretionary interpretation."""
    rows = {row["strategy"]: row for row in summary.iter_rows(named=True)}
    if INCUMBENT_NAME not in rows or LIGHTGBM_NAME not in rows:
        raise ValueError("promotion summary is missing incumbent or LightGBM")
    incumbent = rows[INCUMBENT_NAME]
    challenger = rows[LIGHTGBM_NAME]
    uplift = challenger["precision_at_10"] - incumbent["precision_at_10"]
    failures: list[str] = []
    if challenger["valid_dates"] != required_dates:
        failures.append(
            f"locked valid dates {challenger['valid_dates']} != {required_dates}"
        )
    if uplift + 1e-12 < minimum_uplift:
        failures.append(f"Precision@10 uplift {uplift:.2%} < {minimum_uplift:.2%}")
    if bootstrap.lower_95 <= 0:
        failures.append(f"paired 95% lower bound {bootstrap.lower_95:.2%} <= 0")
    if challenger["passing_date_rate"] < incumbent["passing_date_rate"]:
        failures.append("passing-date rate declined")
    if recent_45_uplift < 0:
        failures.append(f"recent-45 Precision@10 declined by {-recent_45_uplift:.2%}")
    unstable = stability.filter(pl.col("precision_uplift") < -1e-12)
    if not unstable.is_empty():
        segments = ", ".join(
            f"{row['segment']}={row['precision_uplift']:.2%}"
            for row in unstable.iter_rows(named=True)
        )
        failures.append(f"segment Precision@10 declined: {segments}")
    return not failures, tuple(failures or ["all preregistered gates passed"])


def render_report(report: ResearchReport) -> None:
    """Print a compact, auditable qualification report."""
    print("Alpha Screener algorithm qualification")
    print(f"Evidence: {EVIDENCE_TYPE}")
    print(f"Snapshot: {report.snapshot_id}")
    print(f"Feature digest: {report.feature_digest}")
    print(f"Config digest: {report.config_digest}")
    print("\nLocked walk-forward summary")
    for row in report.summary.iter_rows(named=True):
        print(
            f"  {row['strategy']}: dates={row['valid_dates']} "
            f"P@10={row['precision_at_10']:.2%} "
            f"pass_rate={row['passing_date_rate']:.2%} "
            f"mean_base={row['mean_base_explosion_rate']:.2%}"
        )
    interval = report.bootstrap
    print(
        "\nLambdaMART vs rank-v6: "
        f"uplift={interval.estimate:.2%}, "
        f"paired block 95% CI=[{interval.lower_95:.2%}, {interval.upper_95:.2%}], "
        f"P(uplift>0)={interval.probability_positive:.1%}, "
        f"recent45={report.recent_45_uplift:.2%}"
    )
    print("\nStability")
    for row in report.stability.iter_rows(named=True):
        print(
            f"  {row['segment']}: dates={row['dates']} "
            f"uplift={row['precision_uplift']:.2%}"
        )
    print("\nLeakage audit")
    for row in report.folds.iter_rows(named=True):
        print(
            f"  model_as_of={row['model_as_of']} "
            f"max_training_result={row['maximum_training_result_date']} "
            f"train_dates={row['training_dates']} validation_dates={row['validation_dates']} "
            f"test_dates={row['test_dates']} trees={row['lightgbm_best_iteration']}"
        )
    print(f"\nPromotion: {'PASS' if report.promotion_passed else 'REJECT'}")
    for reason in report.promotion_reasons:
        print(f"  - {reason}")
    print(
        "  Current-survivor-universe evidence contains survivorship bias; even a PASS "
        "only qualifies a new strategy for prospective ledger validation."
    )


def _frame_for_dates(panel: pl.DataFrame, dates: tuple[date, ...]) -> pl.DataFrame:
    frame = panel.filter(
        pl.col("date_valid") & pl.col("decision_date").is_in(list(dates))
    ).sort(["decision_date", "ticker"])
    if frame["is_explosion"].null_count():
        raise AssertionError("valid training or test rows contain missing labels")
    return frame


def _complete_pool_thresholds(
    candidates: pl.DataFrame,
    quality: pl.DataFrame,
) -> pl.DataFrame:
    """Implement the contract's nearest-rank P95 exactly, only on complete dates."""
    ordered = candidates.filter(pl.col("forward_return").is_not_null()).sort(
        ["dt", "forward_return", "ticker"]
    ).with_columns(
        pl.col("ticker").cum_count().over("dt").alias("_return_order")
    ).join(
        quality.select("dt", "universe_size", "date_valid"),
        on="dt",
        how="inner",
        validate="m:1",
    )
    threshold_order = (
        pl.col("universe_size") * DEFAULT_CROSS_SECTION_HIT_QUANTILE
    ).ceil().cast(pl.Int64)
    return ordered.filter(
        pl.col("date_valid") & (pl.col("_return_order") == threshold_order)
    ).select(
        "dt",
        pl.max_horizontal(
            pl.lit(DEFAULT_ABSOLUTE_HIT_RETURN),
            pl.col("forward_return"),
        ).alias("hit_threshold"),
    )


def _group_sizes(frame: pl.DataFrame) -> list[int]:
    sizes = (
        frame.group_by("decision_date")
        .len()
        .sort("decision_date")["len"]
        .to_list()
    )
    if sum(sizes) != frame.height:
        raise AssertionError("LightGBM query groups do not cover the ordered frame")
    return sizes


def _require_finite_features(frame: pl.DataFrame, columns: list[str]) -> None:
    if missing := set(columns) - set(frame.columns):
        raise ValueError(f"model features missing columns: {sorted(missing)}")
    invalid = frame.select(
        pl.any_horizontal([
            pl.col(column).is_null() | ~pl.col(column).is_finite()
            for column in columns
        ]).sum()
    ).item()
    if invalid:
        raise ValueError(f"model feature matrix contains {invalid} non-finite rows")


def _snapshot_id(data: pl.DataFrame | pl.LazyFrame) -> str:
    source = data.lazy() if isinstance(data, pl.DataFrame) else data
    stats = source.select(
        pl.len().alias("rows"),
        pl.col("ticker").n_unique().alias("tickers"),
        pl.col("dt").min().alias("first_date"),
        pl.col("dt").max().alias("last_date"),
        pl.col("close").sum().alias("close_checksum"),
        pl.col("volume").sum().alias("volume_checksum"),
    ).collect().row(0, named=True)
    return _digest_json(stats)


def _digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Synchronize an optional long backfill and run the frozen research protocol."""
    parser = argparse.ArgumentParser(
        description="Internal fixed-protocol ranking algorithm qualification"
    )
    parser.add_argument(
        "--backfill-start",
        type=_parse_date,
        help="Download the current official universe from this date before validation",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sync = sync_ohlcv(start=args.backfill_start)
    if not sync.is_fresh:
        raise RuntimeError(f"market data is stale at {sync.as_of_date}")
    if sync.coverage < MIN_SYNC_COVERAGE:
        raise RuntimeError(f"current-directory coverage is only {sync.coverage:.1%}")
    current = scan_ohlcv().filter(
        pl.col("ticker").is_in(list(sync.requested_symbols))
    )
    if sync.as_of_date is not None:
        current = current.filter(pl.col("dt") <= sync.as_of_date)
    research_data = build_research_data(current)
    del current
    gc.collect()
    report = run_walk_forward_research(research_data)
    render_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
