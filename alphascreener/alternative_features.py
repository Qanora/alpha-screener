"""Frozen transforms for point-in-time SEC and FINRA research signals.

The source modules retain auditable dates and coverage.  This module converts
their human-scale outputs into finite model inputs and joins them to an
existing OHLCV research panel.  It does not fetch data and cannot change the
production ranker by itself.
"""

from __future__ import annotations

import hashlib
import json

import polars as pl

from alphascreener.research import ResearchData

ALTERNATIVE_FEATURE_VERSION = "sec-finra-60d-v3"

ALTERNATIVE_MODEL_FEATURES = (
    "alt_sec_coverage",
    "alt_filing_coverage",
    "alt_insider_coverage",
    "alt_current_report_count",
    "alt_offering_risk",
    "alt_late_filing_risk",
    "alt_new_13d",
    "alt_form4_buy_log",
    "alt_form4_non_plan_buy_log",
    "alt_distinct_insider_buyers",
    "alt_cluster_buy",
    "alt_short_interest_coverage",
    "alt_short_interest_log",
    "alt_short_interest_delta",
    "alt_days_to_cover_log",
    "alt_short_interest_age",
    "alt_xrank_form4_buy",
    "alt_xrank_short_interest",
    "alt_xrank_short_interest_delta",
    "alt_xrank_days_to_cover",
)

_KEY_COLUMNS = ("ticker", "decision_date")
_SEC_INPUTS = {
    "sec_coverage",
    "filings_coverage",
    "insider_coverage",
    "current_report_count",
    "recent_offering_risk",
    "late_filing_risk",
    "new_13d",
    "form4_open_market_buy_usd",
    "form4_non_10b5_1_buy_usd",
    "distinct_insider_buyers",
    "cluster_buy",
}
_FINRA_INPUTS = {
    "short_interest_age_sessions",
    "short_interest",
    "short_interest_delta",
    "days_to_cover",
}
ALTERNATIVE_SOURCE_COLUMNS = (
    *_KEY_COLUMNS,
    *sorted(_SEC_INPUTS),
    *sorted(_FINRA_INPUTS),
)


def enrich_research_data(
    data: ResearchData,
    source_features: pl.DataFrame,
    *,
    source_snapshot_id: str,
) -> ResearchData:
    """Return ``data`` with frozen external features joined point-in-time.

    Missing FINRA and SEC history is represented by source-specific coverage
    flags plus neutral finite values.  This prevents unknown observations from
    being silently presented as known zeros.
    """
    if not source_snapshot_id.strip():
        raise ValueError("source_snapshot_id must not be empty")
    required = {*_KEY_COLUMNS, *_SEC_INPUTS, *_FINRA_INPUTS}
    if missing := required - set(source_features.columns):
        raise ValueError(f"alternative source features missing columns: {sorted(missing)}")
    if source_features.select(_KEY_COLUMNS).n_unique() != source_features.height:
        raise ValueError("alternative source features contain duplicate ticker/date rows")

    panel_keys = data.panel.select(_KEY_COLUMNS)
    extra_keys = source_features.select(_KEY_COLUMNS)
    unexpected = extra_keys.join(
        panel_keys,
        on=list(_KEY_COLUMNS),
        how="anti",
    )
    if not unexpected.is_empty():
        raise ValueError("alternative source features contain rows outside the research panel")

    transformed = _transform_source_features(source_features)
    panel = data.panel.join(
        transformed,
        on=list(_KEY_COLUMNS),
        how="left",
        validate="1:1",
    ).with_columns(
        [pl.col(feature).fill_null(0.0).cast(pl.Float32) for feature in ALTERNATIVE_MODEL_FEATURES]
    )
    invalid = panel.select(
        [
            (~pl.col(feature).is_finite()).any().alias(feature)
            for feature in ALTERNATIVE_MODEL_FEATURES
        ]
    ).row(0, named=True)
    if non_finite := [name for name, present in invalid.items() if present]:
        raise ValueError(f"alternative model features are non-finite: {non_finite}")

    snapshot_id = _digest(
        {
            "ohlcv_snapshot": data.snapshot_id,
            "alternative_snapshot": source_snapshot_id,
        }
    )
    feature_digest = _digest(
        {
            "base": data.feature_digest,
            "version": ALTERNATIVE_FEATURE_VERSION,
            "features": ALTERNATIVE_MODEL_FEATURES,
        }
    )
    external_coverage = (
        panel.group_by("decision_date")
        .agg(
            pl.len().cast(pl.Int64).alias("rows"),
            pl.col("alt_filing_coverage").mean().alias("filing_coverage_rate"),
            pl.col("alt_insider_coverage").mean().alias("insider_coverage_rate"),
            pl.col("alt_short_interest_coverage").mean().alias("short_interest_coverage_rate"),
        )
        .sort("decision_date")
    )
    return ResearchData(
        panel=panel,
        date_quality=data.date_quality,
        valid_dates=data.valid_dates,
        snapshot_id=snapshot_id,
        feature_digest=feature_digest,
        extra_model_features=ALTERNATIVE_MODEL_FEATURES,
        external_coverage=external_coverage,
    )


def _transform_source_features(source: pl.DataFrame) -> pl.DataFrame:
    sec_complete = (pl.col("sec_coverage") == "complete").cast(pl.Float64)
    short_present = pl.col("short_interest").is_not_null().cast(pl.Float64)
    transformed = source.with_columns(
        sec_complete.alias("alt_sec_coverage"),
        (pl.col("filings_coverage") == "complete").cast(pl.Float64).alias("alt_filing_coverage"),
        (pl.col("insider_coverage") == "complete").cast(pl.Float64).alias("alt_insider_coverage"),
        pl.col("current_report_count")
        .cast(pl.Float64)
        .clip(0.0, 20.0)
        .log1p()
        .alias("alt_current_report_count"),
        pl.col("recent_offering_risk").cast(pl.Float64).alias("alt_offering_risk"),
        pl.col("late_filing_risk").cast(pl.Float64).alias("alt_late_filing_risk"),
        pl.col("new_13d").cast(pl.Float64).alias("alt_new_13d"),
        pl.col("form4_open_market_buy_usd")
        .cast(pl.Float64)
        .clip(0.0, None)
        .log1p()
        .alias("alt_form4_buy_log"),
        pl.col("form4_non_10b5_1_buy_usd")
        .cast(pl.Float64)
        .clip(0.0, None)
        .log1p()
        .alias("alt_form4_non_plan_buy_log"),
        (pl.col("distinct_insider_buyers").cast(pl.Float64).clip(0.0, 5.0) / 5.0).alias(
            "alt_distinct_insider_buyers"
        ),
        pl.col("cluster_buy").cast(pl.Float64).alias("alt_cluster_buy"),
        short_present.alias("alt_short_interest_coverage"),
        pl.col("short_interest")
        .cast(pl.Float64)
        .clip(0.0, None)
        .log1p()
        .alias("alt_short_interest_log"),
        pl.col("short_interest_delta")
        .cast(pl.Float64)
        .clip(-1.0, 5.0)
        .alias("alt_short_interest_delta"),
        pl.col("days_to_cover")
        .cast(pl.Float64)
        .clip(0.0, 50.0)
        .log1p()
        .alias("alt_days_to_cover_log"),
        (pl.col("short_interest_age_sessions").cast(pl.Float64).clip(0.0, 60.0) / 60.0).alias(
            "alt_short_interest_age"
        ),
    )
    transformed = transformed.with_columns(
        _daily_rank("alt_form4_buy_log", "alt_xrank_form4_buy"),
        _daily_rank("alt_short_interest_log", "alt_xrank_short_interest"),
        _daily_rank(
            "alt_short_interest_delta",
            "alt_xrank_short_interest_delta",
        ),
        _daily_rank("alt_days_to_cover_log", "alt_xrank_days_to_cover"),
    )
    return transformed.select(*_KEY_COLUMNS, *ALTERNATIVE_MODEL_FEATURES)


def _daily_rank(column: str, output: str) -> pl.Expr:
    observed = pl.col(column).is_not_null().sum().over("decision_date")
    return (
        pl.when(pl.col(column).is_null() | (observed <= 1))
        .then(0.0)
        .otherwise(
            2.0 * (pl.col(column).rank("average").over("decision_date") - 1.0) / (observed - 1.0)
            - 1.0
        )
        .alias(output)
    )


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
