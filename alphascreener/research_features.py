"""Frozen 60-session feature set used only for algorithm qualification."""

from __future__ import annotations

import polars as pl

from alphascreener.features import compute_60d_features

_VOLATILITY_EPSILON = 1e-6

RESEARCH_STOCK_FEATURES = (
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_40d",
    "return_59d",
    "momentum_20d_ex_last_5d",
    "momentum_40d_ex_last_5d",
    "positive_day_fraction_20d",
    "trend_efficiency_20d",
    "distance_to_20d_high",
    "distance_to_60d_high",
    "distance_to_20d_low",
    "range_position_60d",
    "realized_volatility_5d",
    "realized_volatility_20d",
    "realized_volatility_59d",
    "downside_volatility_20d",
    "max_daily_return_20d",
    "min_daily_return_20d",
    "jump_ratio_20d",
    "normalized_atr_5d",
    "normalized_atr_20d",
    "parkinson_range_20d",
    "intraday_return_mean_5d",
    "overnight_gap_mean_5d",
    "close_location_mean_5d",
    "log_average_dollar_volume_20d",
    "dollar_volume_ratio_5d_20d",
    "dollar_volume_zscore_20d",
    "zero_volume_fraction_20d",
    "amihud_20d",
    "return_dollar_volume_correlation_20d",
    "excess_return_5d",
    "excess_return_20d",
    "beta_40d",
    "residual_momentum_20d",
    "idiosyncratic_volatility_20d",
)

RESEARCH_MARKET_FEATURES = (
    "spy_return_20d",
    "spy_realized_volatility_20d",
    "spy_distance_to_60d_high",
)

RESEARCH_RAW_TREE_FEATURES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_59d",
    "momentum_20d_ex_last_5d",
    "distance_to_60d_high",
    "realized_volatility_20d",
    "downside_volatility_20d",
    "jump_ratio_20d",
    "normalized_atr_20d",
    "dollar_volume_ratio_5d_20d",
    "log_average_dollar_volume_20d",
    "beta_40d",
    "residual_momentum_20d",
    "idiosyncratic_volatility_20d",
)


def cross_sectional_feature_name(feature: str) -> str:
    """Return the stable column name for a daily percentile transform."""
    return f"xrank_{feature}"


RESEARCH_RANK_FEATURES = tuple(
    cross_sectional_feature_name(feature) for feature in RESEARCH_STOCK_FEATURES
)
LIGHTGBM_FEATURES = (
    *RESEARCH_RANK_FEATURES,
    *RESEARCH_RAW_TREE_FEATURES,
    *RESEARCH_MARKET_FEATURES,
)
LINEAR_FEATURES = RESEARCH_RANK_FEATURES


def compute_research_features(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """Compute the preregistered OHLCV features without reading beyond 60 sessions."""
    required = {"open", "high", "low", "raw_close"}
    if missing := required - set(ohlcv.columns):
        raise ValueError(f"research OHLCV data missing columns: {sorted(missing)}")

    data = compute_60d_features(ohlcv).with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_adjustment_factor"),
        (pl.col("raw_close") * pl.col("volume")).alias("_dollar_volume"),
    ).with_columns(
        (pl.col("open") * pl.col("_adjustment_factor")).alias("_adjusted_open"),
        (pl.col("high") * pl.col("_adjustment_factor")).alias("_adjusted_high"),
        (pl.col("low") * pl.col("_adjustment_factor")).alias("_adjusted_low"),
        pl.col("_dollar_volume").log1p().alias("_log_dollar_volume"),
    )

    previous_close = pl.col("close").shift(1).over("ticker")
    log_return = (pl.col("close") / previous_close).log()
    data = data.with_columns(
        log_return.alias("_log_return_1d"),
        (
            pl.col("_log_dollar_volume")
            - pl.col("_log_dollar_volume").shift(1).over("ticker")
        ).alias("_log_dollar_volume_change"),
        (pl.col("close") / pl.col("_adjusted_open") - 1.0).alias(
            "_intraday_return"
        ),
        (pl.col("_adjusted_open") / previous_close - 1.0).alias("_overnight_gap"),
        pl.when((pl.col("_adjusted_high") - pl.col("_adjusted_low")).abs() > 1e-12)
        .then(
            (pl.col("close") - pl.col("_adjusted_low"))
            / (pl.col("_adjusted_high") - pl.col("_adjusted_low"))
        )
        .otherwise(0.5)
        .alias("_close_location"),
        (
            pl.max_horizontal(
                pl.col("_adjusted_high") - pl.col("_adjusted_low"),
                (pl.col("_adjusted_high") - previous_close).abs(),
                (pl.col("_adjusted_low") - previous_close).abs(),
            )
            / previous_close
        ).alias("_normalized_true_range"),
        (pl.col("_adjusted_high") / pl.col("_adjusted_low")).log().pow(2).alias(
            "_parkinson_range"
        ),
    )

    log_return_20_sum = pl.col("_log_return_1d").rolling_sum(20).over("ticker")
    absolute_log_return_20_sum = (
        pl.col("_log_return_1d").abs().rolling_sum(20).over("ticker")
    )
    vol_20 = pl.col("_log_return_1d").rolling_std(20).over("ticker")
    max_abs_return_20 = (
        pl.col("_log_return_1d").abs().rolling_max(20).over("ticker")
    )
    dollar_volume_mean_5 = pl.col("_dollar_volume").rolling_mean(5).over("ticker")
    dollar_volume_mean_20 = pl.col("_dollar_volume").rolling_mean(20).over("ticker")
    dollar_volume_std_20 = pl.col("_dollar_volume").rolling_std(20).over("ticker")
    return_mean_20 = pl.col("_log_return_1d").rolling_mean(20).over("ticker")
    dollar_volume_change_mean_20 = (
        pl.col("_log_dollar_volume_change").rolling_mean(20).over("ticker")
    )
    covariance_20 = (
        (pl.col("_log_return_1d") * pl.col("_log_dollar_volume_change"))
        .rolling_mean(20)
        .over("ticker")
        - return_mean_20 * dollar_volume_change_mean_20
    )
    return_std_20 = pl.col("_log_return_1d").rolling_std(20).over("ticker")
    dollar_volume_change_std_20 = (
        pl.col("_log_dollar_volume_change").rolling_std(20).over("ticker")
    )
    high_20 = pl.col("close").rolling_max(20).over("ticker")
    low_20 = pl.col("close").rolling_min(20).over("ticker")
    high_60 = pl.col("close").rolling_max(60).over("ticker")
    low_60 = pl.col("close").rolling_min(60).over("ticker")

    data = data.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias(
            "return_1d"
        ),
        (pl.col("close") / pl.col("close").shift(2).over("ticker") - 1.0).alias(
            "return_2d"
        ),
        (pl.col("close") / pl.col("close").shift(10).over("ticker") - 1.0).alias(
            "return_10d"
        ),
        (pl.col("close") / pl.col("close").shift(40).over("ticker") - 1.0).alias(
            "return_40d"
        ),
        (pl.col("close") / pl.col("close").shift(59).over("ticker") - 1.0).alias(
            "return_59d"
        ),
        (
            pl.col("close").shift(5).over("ticker")
            / pl.col("close").shift(20).over("ticker")
            - 1.0
        ).alias("momentum_20d_ex_last_5d"),
        (
            pl.col("close").shift(5).over("ticker")
            / pl.col("close").shift(40).over("ticker")
            - 1.0
        ).alias("momentum_40d_ex_last_5d"),
        (pl.col("_log_return_1d") > 0)
        .cast(pl.Float64)
        .rolling_mean(20)
        .over("ticker")
        .alias("positive_day_fraction_20d"),
        pl.when(absolute_log_return_20_sum > 1e-12)
        .then(log_return_20_sum.abs() / absolute_log_return_20_sum)
        .otherwise(0.0)
        .alias("trend_efficiency_20d"),
        (pl.col("close") / high_20 - 1.0).alias("distance_to_20d_high"),
        (pl.col("close") / low_20 - 1.0).alias("distance_to_20d_low"),
        pl.when((high_60 - low_60).abs() > 1e-12)
        .then((pl.col("close") - low_60) / (high_60 - low_60))
        .otherwise(0.5)
        .alias("range_position_60d"),
        pl.when(
            pl.col("_log_return_1d").rolling_std(5).over("ticker")
            > _VOLATILITY_EPSILON
        )
        .then(pl.col("_log_return_1d").rolling_std(5).over("ticker"))
        .otherwise(0.0)
        .alias("realized_volatility_5d"),
        pl.when(vol_20 > _VOLATILITY_EPSILON)
        .then(vol_20)
        .otherwise(0.0)
        .alias("realized_volatility_20d"),
        pl.when(
            pl.col("_log_return_1d").rolling_std(59).over("ticker")
            > _VOLATILITY_EPSILON
        )
        .then(pl.col("_log_return_1d").rolling_std(59).over("ticker"))
        .otherwise(0.0)
        .alias("realized_volatility_59d"),
        pl.when(pl.col("_log_return_1d") < 0)
        .then(pl.col("_log_return_1d").pow(2))
        .otherwise(0.0)
        .rolling_mean(20)
        .over("ticker")
        .sqrt()
        .alias("downside_volatility_20d"),
        pl.col("_log_return_1d")
        .rolling_max(20)
        .over("ticker")
        .alias("max_daily_return_20d"),
        pl.col("_log_return_1d")
        .rolling_min(20)
        .over("ticker")
        .alias("min_daily_return_20d"),
        pl.when(vol_20 > _VOLATILITY_EPSILON)
        .then(max_abs_return_20 / vol_20)
        .otherwise(0.0)
        .alias("jump_ratio_20d"),
        pl.col("_normalized_true_range")
        .rolling_mean(5)
        .over("ticker")
        .alias("normalized_atr_5d"),
        pl.col("_normalized_true_range")
        .rolling_mean(20)
        .over("ticker")
        .alias("normalized_atr_20d"),
        pl.col("_parkinson_range")
        .rolling_mean(20)
        .over("ticker")
        .alias("parkinson_range_20d"),
        pl.col("_intraday_return")
        .rolling_mean(5)
        .over("ticker")
        .alias("intraday_return_mean_5d"),
        pl.col("_overnight_gap")
        .rolling_mean(5)
        .over("ticker")
        .alias("overnight_gap_mean_5d"),
        pl.col("_close_location")
        .rolling_mean(5)
        .over("ticker")
        .alias("close_location_mean_5d"),
        dollar_volume_mean_20.log1p().alias("log_average_dollar_volume_20d"),
        pl.when(dollar_volume_mean_20 > 0)
        .then(dollar_volume_mean_5 / dollar_volume_mean_20 - 1.0)
        .otherwise(0.0)
        .alias("dollar_volume_ratio_5d_20d"),
        pl.when(dollar_volume_std_20 > 1e-6)
        .then(
            (pl.col("_dollar_volume") - dollar_volume_mean_20)
            / dollar_volume_std_20
        )
        .otherwise(0.0)
        .alias("dollar_volume_zscore_20d"),
        (pl.col("volume") == 0)
        .cast(pl.Float64)
        .rolling_mean(20)
        .over("ticker")
        .alias("zero_volume_fraction_20d"),
        (pl.col("_log_return_1d").abs() / pl.col("_dollar_volume") * 1_000_000.0)
        .rolling_mean(20)
        .over("ticker")
        .alias("amihud_20d"),
        pl.when((return_std_20 * dollar_volume_change_std_20).abs() > 1e-12)
        .then(covariance_20 / (return_std_20 * dollar_volume_change_std_20))
        .otherwise(0.0)
        .alias("return_dollar_volume_correlation_20d"),
    )

    market = data.filter(pl.col("ticker") == "SPY").select(
        "dt",
        pl.col("_log_return_1d").alias("_spy_log_return_1d"),
        pl.col("return_5d").alias("_spy_return_5d"),
        pl.col("return_20d").alias("spy_return_20d"),
        pl.col("return_40d").alias("_spy_return_40d"),
        pl.col("realized_volatility_20d").alias("spy_realized_volatility_20d"),
        pl.col("distance_to_60d_high").alias("spy_distance_to_60d_high"),
    )
    data = data.join(market, on="dt", how="left")

    stock_mean_40 = pl.col("_log_return_1d").rolling_mean(40).over("ticker")
    market_mean_40 = pl.col("_spy_log_return_1d").rolling_mean(40).over("ticker")
    market_second_moment_40 = (
        pl.col("_spy_log_return_1d").pow(2).rolling_mean(40).over("ticker")
    )
    cross_moment_40 = (
        (pl.col("_log_return_1d") * pl.col("_spy_log_return_1d"))
        .rolling_mean(40)
        .over("ticker")
    )
    data = data.with_columns(
        (cross_moment_40 - stock_mean_40 * market_mean_40).alias("_market_cov_40d"),
        (market_second_moment_40 - market_mean_40.pow(2)).alias("_market_var_40d"),
    ).with_columns(
        pl.when(pl.col("_market_var_40d") > 1e-12)
        .then(pl.col("_market_cov_40d") / pl.col("_market_var_40d"))
        .otherwise(0.0)
        .alias("beta_40d"),
        (pl.col("return_5d") - pl.col("_spy_return_5d")).alias("excess_return_5d"),
        (pl.col("return_20d") - pl.col("spy_return_20d")).alias("excess_return_20d"),
    ).with_columns(
        (
            pl.col("_log_return_1d")
            - pl.col("beta_40d") * pl.col("_spy_log_return_1d")
        ).alias("_residual_return_1d")
    ).with_columns(
        pl.col("_residual_return_1d")
        .rolling_sum(20)
        .over("ticker")
        .alias("residual_momentum_20d"),
        pl.when(
            pl.col("_residual_return_1d").rolling_std(20).over("ticker")
            > _VOLATILITY_EPSILON
        )
        .then(pl.col("_residual_return_1d").rolling_std(20).over("ticker"))
        .otherwise(0.0)
        .alias("idiosyncratic_volatility_20d"),
    )
    return data


def add_cross_sectional_ranks(candidates: pl.DataFrame) -> pl.DataFrame:
    """Rank stock features only after the exact daily prefilter has run."""
    if missing := set(RESEARCH_STOCK_FEATURES) - set(candidates.columns):
        raise ValueError(f"research features missing columns: {sorted(missing)}")
    candidates = candidates.with_columns([
        pl.col(feature).round(12).alias(feature)
        for feature in (*RESEARCH_STOCK_FEATURES, *RESEARCH_MARKET_FEATURES)
    ])
    group_size = pl.len().over("dt")
    return candidates.with_columns([
        pl.when(group_size > 1)
        .then(
            2.0
            * (pl.col(feature).rank("average").over("dt") - 1.0)
            / (group_size - 1.0)
            - 1.0
        )
        .otherwise(0.0)
        .alias(cross_sectional_feature_name(feature))
        for feature in RESEARCH_STOCK_FEATURES
    ])
