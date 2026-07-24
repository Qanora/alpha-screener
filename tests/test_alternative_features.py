from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alphascreener.alternative_features import (
    ALTERNATIVE_MODEL_FEATURES,
    enrich_research_data,
)
from alphascreener.research import ResearchData


def _research_data() -> ResearchData:
    panel = pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "decision_date": [date(2026, 7, 20)] * 2,
            "date_valid": [True, True],
        }
    )
    return ResearchData(
        panel=panel,
        date_quality=pl.DataFrame(),
        valid_dates=(date(2026, 7, 20),),
        snapshot_id="ohlcv",
        feature_digest="base",
    )


def _source_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "decision_date": [date(2026, 7, 20)] * 2,
            "sec_coverage": ["complete", "missing"],
            "filings_coverage": ["complete", "missing"],
            "insider_coverage": ["complete", "missing"],
            "days_since_8k_earnings": [2, None],
            "current_report_count": [2, None],
            "material_event_count": [3, None],
            "recent_offering_risk": [False, None],
            "late_filing_risk": [False, None],
            "new_13d": [True, None],
            "form4_open_market_buy_usd": [100_000.0, None],
            "form4_non_10b5_1_buy_usd": [100_000.0, None],
            "distinct_insider_buyers": [2, None],
            "cluster_buy": [True, None],
            "short_interest_age_sessions": [4, None],
            "short_interest": [1_000_000, None],
            "short_interest_delta": [0.25, None],
            "days_to_cover": [4.0, None],
        }
    )


def test_enrichment_is_finite_and_keeps_missingness_explicit() -> None:
    result = enrich_research_data(
        _research_data(),
        _source_features(),
        source_snapshot_id="official",
    )

    assert result.extra_model_features == ALTERNATIVE_MODEL_FEATURES
    assert result.external_coverage is not None
    assert result.snapshot_id != "ohlcv"
    assert result.feature_digest != "base"
    aaa = result.panel.filter(pl.col("ticker") == "AAA").row(0, named=True)
    bbb = result.panel.filter(pl.col("ticker") == "BBB").row(0, named=True)
    assert aaa["alt_sec_coverage"] == 1.0
    assert aaa["alt_filing_coverage"] == 1.0
    assert aaa["alt_insider_coverage"] == 1.0
    assert aaa["alt_short_interest_coverage"] == 1.0
    assert aaa["alt_cluster_buy"] == 1.0
    assert bbb["alt_sec_coverage"] == 0.0
    assert bbb["alt_short_interest_coverage"] == 0.0
    assert all(
        value is not None for value in result.panel.select(ALTERNATIVE_MODEL_FEATURES).row(1)
    )


def test_enrichment_rejects_duplicate_or_out_of_panel_rows() -> None:
    source = _source_features()
    duplicate = pl.concat([source, source.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        enrich_research_data(
            _research_data(),
            duplicate,
            source_snapshot_id="official",
        )

    outside = source.with_columns(
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit("CCC"))
        .otherwise(pl.col("ticker"))
        .alias("ticker")
    )
    with pytest.raises(ValueError, match="outside"):
        enrich_research_data(
            _research_data(),
            outside,
            source_snapshot_id="official",
        )


def test_enrichment_requires_a_source_identity() -> None:
    with pytest.raises(ValueError, match="source_snapshot_id"):
        enrich_research_data(
            _research_data(),
            _source_features(),
            source_snapshot_id=" ",
        )
