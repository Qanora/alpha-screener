"""Tests for ablation dual-track recording and Delta-Lift@20 monitoring.

Issue #99: Ablation dual-track recording.
Reference: PRD 4.7.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from unittest import mock as umock

import polars as pl
import pytest

from alphascreener.tradingagents.ablation import (
    AblationConfig,
    AblationDecision,
    AblationEntry,
    AblationTracker,
    build_outcomes_from_ohlcv,
    compute_ablation_decision,
    compute_base_rate,
    compute_delta_lift,
    compute_lift_at_k,
    compute_precision_at_k,
    compute_refined_score_llm,
    compute_refined_score_pure,
    compute_risk_filter,
    create_ablation_tracker,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_scores_df(
    scores: list[float],
    hits: list[int],
) -> pl.DataFrame:
    """Build a DataFrame with *scores* and *hits* columns."""
    return pl.DataFrame({
        "score": scores,
        "hit": hits,
    })


def _make_aligned_df(
    pure_scores: list[float],
    llm_scores: list[float],
    hits: list[int],
) -> pl.DataFrame:
    """Build a pre-joined ablation DataFrame."""
    return pl.DataFrame({
        "ticker": [f"T{i}" for i in range(len(pure_scores))],
        "dt": date(2025, 6, 1),
        "refined_score_pure": pure_scores,
        "refined_score": llm_scores,
        "hit": hits,
    })


def _make_ohlcv_df(
    tickers: list[str],
    dates: list[date],
    closes: list[float],
) -> pl.DataFrame:
    """Build a minimal OHLCV DataFrame."""
    return pl.DataFrame({
        "ticker": tickers,
        "dt": dates,
        "close": closes,
    })


# ============================================================================
# 1. compute_risk_filter
# ============================================================================


class TestComputeRiskFilter:
    """PRD 4.4 risk_filter: 0 when hard-block tag present or data conflict."""

    def test_no_risk_no_conflict_returns_one(self):
        assert compute_risk_filter([], data_conflict_detected=False) == 1.0
        assert compute_risk_filter(["momentum_breakdown"], data_conflict_detected=False) == 1.0

    def test_delisting_risk_returns_zero(self):
        assert compute_risk_filter(["delisting_risk"], data_conflict_detected=False) == 0.0
        assert compute_risk_filter(
            ["momentum_breakdown", "delisting_risk"], data_conflict_detected=False,
        ) == 0.0

    def test_data_conflict_returns_zero(self):
        assert compute_risk_filter([], data_conflict_detected=True) == 0.0
        assert compute_risk_filter(["volume_divergence"], data_conflict_detected=True) == 0.0

    def test_none_tags_treated_as_empty(self):
        assert compute_risk_filter(None, data_conflict_detected=False) == 1.0
        assert compute_risk_filter(None, data_conflict_detected=True) == 0.0

    def test_both_triggers_returns_zero(self):
        assert compute_risk_filter(
            ["delisting_risk"], data_conflict_detected=True,
        ) == 0.0


# ============================================================================
# 2. Refined score functions
# ============================================================================


class TestComputeRefinedScorePure:
    """A-track: Refined_Score_pure = Coarse_Final_Score."""

    def test_identity(self):
        assert compute_refined_score_pure(0.0) == 0.0
        assert compute_refined_score_pure(1.5) == 1.5
        assert compute_refined_score_pure(-3.0) == -3.0


class TestComputeRefinedScoreLLM:
    """B-track: Refined_Score = Coarse_Final_Score x score_correction x risk_filter."""

    def test_neutral_correction_no_risk(self):
        assert compute_refined_score_llm(2.0, 1.0, 1.0) == 2.0

    def test_upward_correction(self):
        assert compute_refined_score_llm(2.0, 1.05, 1.0) == 2.1

    def test_downward_correction(self):
        assert compute_refined_score_llm(2.0, 0.90, 1.0) == 1.8

    def test_risk_filter_zeros_score(self):
        assert compute_refined_score_llm(2.0, 1.05, 0.0) == 0.0
        assert compute_refined_score_llm(0.0, 1.0, 0.0) == 0.0


# ============================================================================
# 3. AblationEntry
# ============================================================================


class TestAblationEntry:
    """Entry construction, field integrity, and serialisation."""

    def test_basic_entry_post_init(self):
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        assert entry.refined_score_pure == 2.0
        assert entry.refined_score_llm == 2.0  # default correction=1.0, filter=1.0

    def test_entry_with_correction_and_risk(self):
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1),
            coarse_final_score=2.0, score_correction=1.05,
            risk_filter=0.0, risk_tags=["delisting_risk"],
            data_conflict_detected=False,
        )
        assert entry.refined_score_pure == 2.0
        assert entry.refined_score_llm == 0.0

    def test_to_dict_pure(self):
        entry = AblationEntry(
            ticker="MSFT", dt=date(2025, 7, 1),
            coarse_final_score=1.5, phase1_pass=True,
        )
        d = entry.to_dict_pure()
        assert d["ticker"] == "MSFT"
        assert d["coarse_final_score"] == 1.5
        assert d["refined_score_pure"] == 1.5
        assert d["phase1_pass"] is True

    def test_to_dict_llm(self):
        entry = AblationEntry(
            ticker="MSFT", dt=date(2025, 7, 1),
            coarse_final_score=1.5, score_correction=0.95,
            risk_filter=1.0, risk_tags=["overbought"],
            data_conflict_detected=False, phase1_pass=True,
        )
        d = entry.to_dict_llm()
        assert d["refined_score"] == 1.5 * 0.95
        assert d["score_correction"] == 0.95
        assert "overbought" in d["risk_tags"]

    def test_from_assessment_normal(self):
        entry = AblationEntry.from_assessment(
            ticker="GOOG", dt=date(2025, 8, 1),
            coarse_final_score=3.0, score_correction=1.02,
            risk_tags=["momentum_breakdown"], data_conflict_detected=False,
        )
        assert entry.risk_filter == 1.0
        assert entry.refined_score_llm == 3.0 * 1.02

    def test_from_assessment_data_conflict(self):
        entry = AblationEntry.from_assessment(
            ticker="GOOG", dt=date(2025, 8, 1),
            coarse_final_score=3.0, score_correction=1.02,
            risk_tags=[], data_conflict_detected=True,
        )
        assert entry.risk_filter == 0.0
        assert entry.refined_score_llm == 0.0

    def test_from_assessment_delisting(self):
        entry = AblationEntry.from_assessment(
            ticker="GOOG", dt=date(2025, 8, 1),
            coarse_final_score=3.0, score_correction=1.02,
            risk_tags=["delisting_risk"], data_conflict_detected=False,
        )
        assert entry.risk_filter == 0.0
        assert entry.refined_score_llm == 0.0

    def test_from_assessment_none_tags(self):
        entry = AblationEntry.from_assessment(
            ticker="GOOG", dt=date(2025, 8, 1),
            coarse_final_score=3.0, score_correction=1.0,
            risk_tags=[], data_conflict_detected=False,
        )
        assert entry.risk_filter == 1.0


# ============================================================================
# 4. Precision@K
# ============================================================================


class TestComputePrecisionAtK:
    """Precision@K = hits_in_top_k / k."""

    def test_all_hits(self):
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        )
        assert compute_precision_at_k(df, 5, score_col="score", outcome_col="hit") == 1.0

    def test_partial_hits(self):
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        assert compute_precision_at_k(df, 5, score_col="score", outcome_col="hit") == 0.4

    def test_zero_hits(self):
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        assert compute_precision_at_k(df, 5, score_col="score", outcome_col="hit") == 0.0

    def test_too_few_rows_returns_nan(self):
        df = _make_scores_df(scores=[10, 9, 8], hits=[1, 0, 0])
        result = compute_precision_at_k(df, 5, score_col="score", outcome_col="hit")
        assert math.isnan(result)

    def test_null_scores_excluded(self):
        df = pl.DataFrame({
            "score": [10, 9, 8, None, 7, 6, 5, 4, 3, 2],
            "hit": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        # 3 valid scores >= k=3, top 3: 10(hit=1), 9(hit=1), 8(hit=0) -> 2/3
        result = compute_precision_at_k(df, 3, score_col="score", outcome_col="hit")
        assert result == pytest.approx(2.0 / 3.0)


# ============================================================================
# 5. Base rate
# ============================================================================


class TestComputeBaseRate:
    """base_rate = hits / total."""

    def test_half_base_rate(self):
        df = _make_scores_df(scores=[1, 2, 3, 4], hits=[1, 0, 1, 0])
        assert compute_base_rate(df, outcome_col="hit") == 0.5

    def test_zero_base_rate(self):
        df = _make_scores_df(scores=[1, 2], hits=[0, 0])
        assert compute_base_rate(df, outcome_col="hit") == 0.0

    def test_empty_dataframe_returns_nan(self):
        df = pl.DataFrame(schema={"score": pl.Float64, "hit": pl.Int64})
        assert math.isnan(compute_base_rate(df, outcome_col="hit"))


# ============================================================================
# 6. Lift@K
# ============================================================================


class TestComputeLiftAtK:
    """Lift@K = Precision@K / base_rate."""

    def test_lift_equals_two(self):
        # Precision=0.6, base_rate=0.3 -> lift=2.0
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        # precision@5 = 3/5 = 0.6, base_rate = 3/10 = 0.3
        result = compute_lift_at_k(df, 5, score_col="score", outcome_col="hit")
        assert result == pytest.approx(2.0)

    def test_lift_one_when_precision_equals_base_rate(self):
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        result = compute_lift_at_k(df, 5, score_col="score", outcome_col="hit")
        assert result == pytest.approx(1.0)

    def test_nan_when_base_rate_zero(self):
        df = _make_scores_df(
            scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        result = compute_lift_at_k(df, 5, score_col="score", outcome_col="hit")
        assert math.isnan(result)

    def test_nan_when_too_few_rows(self):
        df = _make_scores_df(scores=[10, 9], hits=[1, 0])
        result = compute_lift_at_k(df, 5, score_col="score", outcome_col="hit")
        assert math.isnan(result)


# ============================================================================
# 7. Delta-Lift
# ============================================================================


class TestComputeDeltaLift:
    """Delta-Lift = Lift(B) - Lift(A)."""

    def test_zero_delta_when_identical_scores(self):
        df = _make_aligned_df(
            pure_scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            llm_scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        result = compute_delta_lift(
            df, df, 5,
            score_col_pure="refined_score_pure",
            score_col_llm="refined_score",
            outcome_col="hit",
        )
        assert result == pytest.approx(0.0)

    def test_positive_delta_when_llm_better(self):
        # base_rate = 3/10 = 0.3
        # A-track top-5: T0-T4 scores all 5,
        #   hits [1,1,0,0,0] -> prec=0.4 -> lift=1.333
        # B-track top-5: sorted by higher scores
        #   hits [1,1,1,0,0] -> prec=0.6 -> lift=2.0 -> delta=0.667
        pure_df = pl.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "dt": date(2025, 6, 1),
            "refined_score_pure": [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "hit": [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        })
        llm_df = pl.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "dt": date(2025, 6, 1),
            "refined_score": [10.0, 9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "hit": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })
        result = compute_delta_lift(
            pure_df, llm_df, 5,
            score_col_pure="refined_score_pure",
            score_col_llm="refined_score",
            outcome_col="hit",
        )
        # LLM better ranking -> positive delta
        assert result > 0.0

    def test_nan_when_one_track_fails(self):
        pure_df = pl.DataFrame({
            "ticker": ["T0"], "dt": date(2025, 6, 1),
            "refined_score_pure": [5.0], "hit": [1],
        })
        llm_df = pl.DataFrame({
            "ticker": ["T0"], "dt": date(2025, 6, 1),
            "refined_score": [6.0], "hit": [1],
        })
        # k=5 but only 1 row -> NaN
        result = compute_delta_lift(
            pure_df, llm_df, 5,
            score_col_pure="refined_score_pure",
            score_col_llm="refined_score",
            outcome_col="hit",
        )
        assert math.isnan(result)


# ============================================================================
# 8. AblationDecision
# ============================================================================


class TestComputeAblationDecision:
    """PRD 4.7 threshold mapping."""

    def test_pass_threshold_and_above(self):
        assert compute_ablation_decision(0.05) == AblationDecision.PASS
        assert compute_ablation_decision(0.10) == AblationDecision.PASS
        assert compute_ablation_decision(1.0) == AblationDecision.PASS

    def test_borderline(self):
        assert compute_ablation_decision(0.0) == AblationDecision.BORDERLINE
        assert compute_ablation_decision(0.01) == AblationDecision.BORDERLINE
        assert compute_ablation_decision(0.0499) == AblationDecision.BORDERLINE

    def test_fail(self):
        assert compute_ablation_decision(-0.01) == AblationDecision.FAIL
        assert compute_ablation_decision(-1.0) == AblationDecision.FAIL


# ============================================================================
# 9. AblationTracker — record & dedup
# ============================================================================


class TestAblationTrackerRecord:
    """Tracker records entries and prevents duplicates."""

    def test_record_single_entry(self):
        tracker = AblationTracker()
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(entry)
        assert tracker.n_records == 1

    def test_record_duplicate_skipped(self):
        tracker = AblationTracker()
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(entry)
        tracker.record(entry)
        assert tracker.n_records == 1

    def test_record_batch(self):
        tracker = AblationTracker()
        entries = [
            AblationEntry(ticker=f"T{i}", dt=date(2025, 6, 1), coarse_final_score=float(i))
            for i in range(5)
        ]
        tracker.record_batch(entries)
        assert tracker.n_records == 5

    def test_disabled_tracker_no_op(self):
        config = AblationConfig(enabled=False)
        tracker = AblationTracker(config)
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(entry)
        assert tracker.n_records == 0

    def test_different_dates_same_ticker_allowed(self):
        tracker = AblationTracker()
        e1 = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        e2 = AblationEntry(ticker="AAPL", dt=date(2025, 6, 2), coarse_final_score=2.0)
        tracker.record(e1)
        tracker.record(e2)
        assert tracker.n_records == 2

    def test_same_date_different_tickers_allowed(self):
        tracker = AblationTracker()
        e1 = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        e2 = AblationEntry(ticker="MSFT", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(e1)
        tracker.record(e2)
        assert tracker.n_records == 2


# ============================================================================
# 10. AblationTracker — DataFrames
# ============================================================================


class TestAblationTrackerDf:
    """pure_df() and llm_df() build correct DataFrames."""

    def test_pure_df_columns(self):
        tracker = AblationTracker()
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(entry)
        df = tracker.pure_df()
        expected = {"ticker", "dt", "coarse_final_score",
                    "refined_score_pure", "phase1_pass"}
        assert set(df.columns) == expected

    def test_llm_df_columns(self):
        tracker = AblationTracker()
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1),
            coarse_final_score=2.0, score_correction=0.95,
            risk_tags=["overbought"],
        )
        tracker.record(entry)
        df = tracker.llm_df()
        assert "refined_score" in df.columns
        assert "score_correction" in df.columns
        assert "risk_tags" in df.columns
        assert "risk_filter" in df.columns
        assert "data_conflict_detected" in df.columns

    def test_empty_pure_df_has_correct_schema(self):
        tracker = AblationTracker()
        df = tracker.pure_df()
        assert df.height == 0
        assert "refined_score_pure" in df.columns

    def test_empty_llm_df_has_correct_schema(self):
        tracker = AblationTracker()
        df = tracker.llm_df()
        assert df.height == 0
        assert "refined_score" in df.columns

    def test_pure_df_values_correct(self):
        tracker = AblationTracker()
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1),
            coarse_final_score=3.0, score_correction=1.05,
            risk_filter=1.0,
        )
        tracker.record(entry)
        pdf = tracker.pure_df()
        assert pdf["refined_score_pure"][0] == 3.0

    def test_llm_df_values_correct(self):
        tracker = AblationTracker()
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1),
            coarse_final_score=3.0, score_correction=1.05,
            risk_filter=1.0,
        )
        tracker.record(entry)
        ldf = tracker.llm_df()
        assert ldf["refined_score"][0] == 3.0 * 1.05


# ============================================================================
# 11. AblationTracker — delta_lift & decide
# ============================================================================


class TestAblationTrackerDeltaLift:
    """delta_lift() and decide() integration."""

    def test_delta_lift_from_aligned(self):
        tracker = AblationTracker()
        df = _make_aligned_df(
            pure_scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            llm_scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            hits=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        dl = tracker.delta_lift_from_aligned(df, k=5)
        assert dl == pytest.approx(0.0)

    def test_delta_lift_with_outcomes(self):
        tracker = AblationTracker()
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=5.0,
            score_correction=1.05, risk_filter=1.0,
        )
        tracker.record(entry)
        outcomes = pl.DataFrame({
            "ticker": ["AAPL"], "dt": [date(2025, 6, 1)], "hit": [1],
        })
        # Only 1 row, k=20 -> NaN
        dl = tracker.delta_lift(outcomes, k=20)
        assert math.isnan(dl)

    def test_delta_lift_no_records_returns_nan(self):
        tracker = AblationTracker()
        outcomes = pl.DataFrame({
            "ticker": ["AAPL"], "dt": [date(2025, 6, 1)], "hit": [1],
        })
        dl = tracker.delta_lift(outcomes)
        assert math.isnan(dl)

    def test_decide_returns_none_without_delta(self):
        tracker = AblationTracker()
        assert tracker.decide() is None

    def test_decide_pass(self):
        tracker = AblationTracker()
        assert tracker.decide(0.05) == AblationDecision.PASS

    def test_decide_borderline(self):
        tracker = AblationTracker()
        assert tracker.decide(0.0) == AblationDecision.BORDERLINE

    def test_decide_fail(self):
        tracker = AblationTracker()
        assert tracker.decide(-0.1) == AblationDecision.FAIL

    def test_full_roundtrip(self):
        """Simulate a realistic ablation flow."""
        tracker = AblationTracker()
        # Record 10 entries with aligned scores and outcomes
        for i in range(10):
            entry = AblationEntry(
                ticker=f"T{i}", dt=date(2025, 6, 1),
                coarse_final_score=float(10 - i),
                score_correction=1.05 if i < 5 else 0.95,
            )
            tracker.record(entry)

        # Build aligned df with hits
        hits = [1 if i < 3 else 0 for i in range(10)]
        df = tracker.pure_df().join(
            tracker.llm_df().select("ticker", "dt", "refined_score"),
            on=["ticker", "dt"],
        ).with_columns(pl.Series("hit", hits))

        dl = tracker.delta_lift_from_aligned(df, k=5)
        decision = tracker.decide(dl)
        assert decision in AblationDecision
        assert dl is not None


# ============================================================================
# 12. AblationTracker — flush
# ============================================================================


class TestAblationTrackerFlush:
    """flush() persists both tracks as Parquet under the signals category."""

    def test_flush_empty_no_op(self, tmp_path: Path):
        """Flushing an empty tracker does not create directories."""
        tracker = AblationTracker()
        with umock.patch(
            "alphascreener.tradingagents.ablation.get_data_dir",
            return_value=tmp_path,
        ):
            tracker.flush()
        # No dt= subdirectories created
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subdirs) == 0

    def test_flush_writes_pure_and_llm(self, tmp_path: Path):
        """Flushing one entry creates the partition dir with both files."""
        tracker = AblationTracker()
        entry = AblationEntry(
            ticker="AAPL", dt=date(2025, 6, 1),
            coarse_final_score=2.0, risk_tags=["momentum_breakdown"],
        )
        tracker.record(entry)

        with umock.patch(
            "alphascreener.tradingagents.ablation.get_data_dir",
            return_value=tmp_path,
        ):
            tracker.flush()

        # Partition directory
        part_dir = tmp_path / "dt=2025-06-01"
        assert part_dir.is_dir()

        files = sorted(part_dir.iterdir())
        assert len(files) == 2
        names = [f.name for f in files]
        assert any(n.startswith("pure_") for n in names)
        assert any(n.startswith("llm_") for n in names)

        # Read back and verify
        for f in files:
            df = pl.read_parquet(f)
            assert df.height == 1
            if f.name.startswith("pure_"):
                assert "refined_score_pure" in df.columns
            else:
                assert "refined_score" in df.columns

    def test_flush_multiple_dates(self, tmp_path: Path):
        """Entries from different dates land in separate partitions."""
        tracker = AblationTracker()
        tracker.record(AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=1.0))
        tracker.record(AblationEntry(ticker="MSFT", dt=date(2025, 6, 2), coarse_final_score=2.0))

        with umock.patch(
            "alphascreener.tradingagents.ablation.get_data_dir",
            return_value=tmp_path,
        ):
            tracker.flush()

        assert (tmp_path / "dt=2025-06-01").is_dir()
        assert (tmp_path / "dt=2025-06-02").is_dir()

    def test_clear_does_not_delete_files(self, tmp_path: Path):
        """clear() only resets in-memory state."""
        tracker = AblationTracker()
        entry = AblationEntry(ticker="AAPL", dt=date(2025, 6, 1), coarse_final_score=2.0)
        tracker.record(entry)

        with umock.patch(
            "alphascreener.tradingagents.ablation.get_data_dir",
            return_value=tmp_path,
        ):
            tracker.flush()

        tracker.clear()
        assert tracker.n_records == 0

        # Files still exist
        part_dir = tmp_path / "dt=2025-06-01"
        assert part_dir.is_dir()
        assert len(list(part_dir.iterdir())) == 2


# ============================================================================
# 13. build_outcomes_from_ohlcv
# ============================================================================


class TestBuildOutcomesFromOhlcv:
    """Forward return computation from OHLCV data."""

    def test_basic_forward_return(self):
        df = _make_ohlcv_df(
            tickers=["AAPL"] * 10,
            dates=[date(2025, 6, d) for d in range(1, 11)],
            closes=[100.0, 101.0, 102.0, 103.0, 104.0,
                     105.0, 106.0, 107.0, 108.0, 109.0],
        )
        outcomes = build_outcomes_from_ohlcv(df, return_threshold=0.10, holding_days=7)
        # fwd_return for day 1: (108-100)/100 = 0.08 < 0.10 -> hit=0
        assert outcomes.filter(pl.col("dt") == date(2025, 6, 1))["hit"][0] == 0

    def test_hit_when_return_above_threshold(self):
        df = _make_ohlcv_df(
            tickers=["AAPL"] * 10,
            dates=[date(2025, 6, d) for d in range(1, 11)],
            closes=[100.0, 105.0, 110.0, 115.0, 120.0,
                     125.0, 130.0, 135.0, 140.0, 145.0],
        )
        # day 1: close=100, close_fwd (day 8) = 135 -> (135-100)/100 = 0.35 > 0.10
        outcomes = build_outcomes_from_ohlcv(df, return_threshold=0.10, holding_days=7)
        hit = outcomes.filter(pl.col("dt") == date(2025, 6, 1))["hit"][0]
        assert hit == 1

    def test_rows_at_end_dropped(self):
        """Last *holding_days* rows per ticker have null forward return and are dropped."""
        df = _make_ohlcv_df(
            tickers=["AAPL"] * 10,
            dates=[date(2025, 6, d) for d in range(1, 11)],
            closes=[100.0] * 10,
        )
        outcomes = build_outcomes_from_ohlcv(df, holding_days=7)
        # 10 - 7 = 3 valid rows per ticker
        assert outcomes.height == 3

    def test_empty_input(self):
        df = pl.DataFrame(schema={"ticker": pl.Utf8, "dt": pl.Date, "close": pl.Float64})
        outcomes = build_outcomes_from_ohlcv(df)
        assert outcomes.height == 0
        assert "hit" in outcomes.columns

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"ticker": ["AAPL"], "dt": [date(2025, 6, 1)]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_outcomes_from_ohlcv(df)  # type: ignore[arg-type]

    def test_multiple_tickers_independent(self):
        """Each ticker's forward return is computed independently."""
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "dt": [date(2025, 6, d) for d in range(1, 6)] * 2,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0,
                      200.0, 210.0, 215.0, 220.0, 225.0],
        })
        outcomes = build_outcomes_from_ohlcv(df, holding_days=2)
        # Each ticker: 5 - 2 = 3 valid rows -> 6 total
        assert outcomes.height == 6
        # AAPL dt=2025-06-01: close=100, close_fwd=2d later=102 -> return=0.02 -> hit=0
        aapl_row = outcomes.filter(
            (pl.col("ticker") == "AAPL") & (pl.col("dt") == date(2025, 6, 1))
        )
        assert aapl_row["hit"][0] == 0


# ============================================================================
# 14. create_ablation_tracker factory
# ============================================================================


class TestCreateAblationTracker:
    """Factory reads LLM_ABLATION_ENABLED from settings."""

    def test_default_enabled(self):
        tracker = create_ablation_tracker()
        assert tracker.enabled is True

    def test_explicit_disabled(self):
        tracker = create_ablation_tracker(enabled=False)
        assert tracker.enabled is False

    def test_explicit_k(self):
        tracker = create_ablation_tracker(k=10)
        assert tracker._config.k == 10
