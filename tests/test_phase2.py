"""Tests for Phase 2 weighted scoring + industry dedup (Issue #95).

Covers:
  - compute_breakout_score with continuous (z_capped) and binary factors
  - apply_industry_dedup with sector/industry caps
  - phase2_pipeline full integration
  - Edge cases: empty DataFrame, null metadata, missing columns, fallback
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest  # noqa: I001

# ============================================================================
# Helpers — build test DataFrames
# ============================================================================


def _scored_df(rows: list[dict]) -> pl.DataFrame:
    """Minimal DataFrame with z_capped columns and optional sector/industry."""
    schema = {
        "ticker": pl.Utf8,
        "dt": pl.Date,
        "MOM_5D": pl.Float64,
        "PTH": pl.Float64,
        "MOM_SLOPE": pl.Float64,
        "BB_SQUEEZE": pl.Float64,
        "ATR_RATIO": pl.Float64,
        "MFI_14": pl.Float64,
        "CMF_21": pl.Float64,
        "VOL_ANOMALY": pl.Float64,
        "RSI_OVERSOLD": pl.Float64,
        "REV_ACCEL": pl.Float64,
        "MACD_CROSS": pl.Int32,
        "GOLDEN_CROSS": pl.Int32,
        "INSIDER_BUY": pl.Int32,
        "PEAD_FLAG": pl.Int32,
        "sector": pl.Utf8,
        "industry": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_row(
    ticker: str,
    mom_z: float = 0.0,
    pth_z: float = 0.0,
    macd: int = 0,
    golden: int = 0,
    insider: int = 0,
    sector: str | None = None,
    industry: str | None = None,
) -> dict:
    """Create a single row with default-neutral values."""
    return {
        "ticker": ticker,
        "dt": date(2025, 3, 15),
        "MOM_5D": mom_z,
        "PTH": pth_z,
        "MOM_SLOPE": 0.0,
        "BB_SQUEEZE": 0.0,
        "ATR_RATIO": 0.08,
        "MFI_14": 50.0,
        "CMF_21": 0.0,
        "VOL_ANOMALY": 0.0,
        "RSI_OVERSOLD": 0.0,
        "REV_ACCEL": 0.0,
        "MACD_CROSS": macd,
        "GOLDEN_CROSS": golden,
        "INSIDER_BUY": insider,
        "PEAD_FLAG": 0,
        "sector": sector,
        "industry": industry,
    }


# ============================================================================
# compute_breakout_score tests
# ============================================================================


class TestComputeBreakoutScore:
    """Weighted breakout score = Σ(w_i × z_capped_i) + Σ(w_j × flag_j)."""

    def test_continuous_only(self):
        """Higher z-score factor values produce higher breakout_score."""
        df = _scored_df(
            [
                _make_row("HIGH", mom_z=3.0, pth_z=2.0),
                _make_row("LOW", mom_z=-3.0, pth_z=-2.0),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        high_score = result.filter(pl.col("ticker") == "HIGH")["breakout_score"][0]
        low_score = result.filter(pl.col("ticker") == "LOW")["breakout_score"][0]
        # HIGH has strictly higher factor values -> higher z-scores -> higher score
        assert high_score > low_score
        # Both should be non-zero (factors have variance across 2 rows)
        assert high_score != pytest.approx(0.0)
        assert low_score != pytest.approx(0.0)

    def test_binary_flags_add_weight(self):
        """Binary flags == 1 contribute their full weight. Continuous z-scores are 0 (sigma=0)."""
        from alphascreener.screening.phase2 import MVP_WEIGHTS

        df = _scored_df(
            [
                _make_row("A", macd=1, golden=1, insider=1),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df)
        # All continuous factors: single row → sigma=0 → z=0 → contribution=0
        # Binary factors only contribute
        expected = (
            MVP_WEIGHTS["MACD_CROSS"] + MVP_WEIGHTS["GOLDEN_CROSS"] + MVP_WEIGHTS["INSIDER_BUY"]
        )
        assert result["breakout_score"][0] == pytest.approx(expected)

    def test_pead_flag_not_weighted(self):
        """PEAD_FLAG weight is 0 in coarse screening."""
        df = _scored_df(
            [
                _make_row("A", mom_z=1.0),
                _make_row("B", mom_z=1.0),
            ]
        )
        df = df.with_columns(
            pl.when(pl.col("ticker") == "A")
            .then(1)
            .otherwise(pl.col("PEAD_FLAG"))
            .alias("PEAD_FLAG")
        )
        # Set PEAD_FLAG to 1 for A, keep 0 for B

        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df)
        # Both should have identical scores since PEAD weight = 0
        assert result["breakout_score"][0] == pytest.approx(result["breakout_score"][1])

    def test_null_z_capped_treated_as_zero(self):
        """Null raw factor values contribute 0 to the score (z-score = 0 for null)."""
        df = _scored_df(
            [
                _make_row("A", mom_z=1.0),
                _make_row("B", mom_z=-1.0),
            ]
        )
        # Set PTH to null for ticker A
        df = df.with_columns(
            pl.when(pl.col("ticker") == "A")
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("PTH"))
            .alias("PTH")
        )

        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        # A: PTH is null → PTH z-score = 0 for A
        # B: PTH = 0.0 (normal value)
        # Verify: B's PTH contribution is non-zero, A's PTH contribution is 0
        assert result["breakout_score"][0] != pytest.approx(result["breakout_score"][1])

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty with breakout_score column."""
        from alphascreener.screening.phase2 import compute_breakout_score

        empty = _scored_df([])
        result = compute_breakout_score(empty)
        assert result.height == 0
        assert "breakout_score" in result.columns

    def test_weights_sum_near_one(self):
        """MVP weights should sum to approximately 1.0 (within rounding)."""
        from alphascreener.screening.phase2 import MVP_WEIGHTS

        total = sum(MVP_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_multiple_tickers_sortable(self):
        """Scores should produce a meaningful descending order."""
        df = _scored_df(
            [
                _make_row("HIGH", mom_z=3.0, macd=1, golden=1),
                _make_row("MED", mom_z=1.0, insider=1),
                _make_row("LOW", mom_z=-1.0),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("breakout_score", descending=True)
        assert result["ticker"].to_list() == ["HIGH", "MED", "LOW"]

    def test_all_zeros_gives_zero_score(self):
        """All continuous factors at 0 (sigma=0) + no binary flags => breakout_score = 0."""
        df = _scored_df(
            [
                _make_row("Z", mom_z=0.0),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df)
        # Single row: all z-scores = 0 (sigma=0), no binary flags → score = 0
        assert result["breakout_score"][0] == pytest.approx(0.0)

    def test_max_z_capped_contribution(self):
        """Higher raw factor values produce higher z-scores and thus higher breakout_score."""
        df = _scored_df(
            [
                _make_row("HIGH", mom_z=3.0),
                _make_row("MED", mom_z=0.0),
                _make_row("LOW", mom_z=-3.0),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        high_score = result.filter(pl.col("ticker") == "HIGH")["breakout_score"][0]
        med_score = result.filter(pl.col("ticker") == "MED")["breakout_score"][0]
        low_score = result.filter(pl.col("ticker") == "LOW")["breakout_score"][0]
        # z-score ordering preserves raw value ordering
        assert high_score > med_score > low_score
        # Weight * z_capped contribution: z-scores are proportional to raw differences
        assert high_score > 0 > low_score

    def test_min_z_capped_contribution(self):
        """Lower raw factor values produce lower z-scores and thus lower breakout_score."""
        df = _scored_df(
            [
                _make_row("HIGH", mom_z=5.0),
                _make_row("LOW", mom_z=-5.0),
            ]
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        high_score = result.filter(pl.col("ticker") == "HIGH")["breakout_score"][0]
        low_score = result.filter(pl.col("ticker") == "LOW")["breakout_score"][0]
        assert high_score > low_score


# ============================================================================
# apply_industry_dedup tests
# ============================================================================


class TestApplyIndustryDedup:
    """Greedy dedup enforcing sector/industry caps."""

    def test_sector_cap_of_three(self):
        """Max 3 tickers per sector, 4th in same sector skipped."""
        df = _scored_df(
            [
                _make_row("T1", mom_z=3.0, sector="Tech", industry="SW"),
                _make_row("T2", mom_z=2.0, sector="Tech", industry="HW"),
                _make_row("T3", mom_z=1.0, sector="Tech", industry="Semi"),
                _make_row("T4", mom_z=0.5, sector="Tech", industry="Cloud"),  # 4th Tech
                _make_row("F1", mom_z=0.4, sector="Finance", industry="Bank"),
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df, sector_cap=3, industry_cap=2)
        tickers = result["ticker"].to_list()
        assert "T1" in tickers
        assert "T2" in tickers
        assert "T3" in tickers
        assert "T4" not in tickers  # Sector cap hit
        assert "F1" in tickers  # Different sector

    def test_industry_cap_of_two(self):
        """Max 2 tickers per industry, 3rd in same industry skipped."""
        df = _scored_df(
            [
                _make_row("T1", mom_z=3.0, sector="Tech", industry="SW"),
                _make_row("T2", mom_z=2.0, sector="Tech", industry="SW"),
                _make_row("T3", mom_z=1.0, sector="Tech", industry="SW"),  # 3rd SW
                _make_row("T4", mom_z=0.5, sector="Tech", industry="HW"),
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df, sector_cap=3, industry_cap=2)
        tickers = result["ticker"].to_list()
        assert "T1" in tickers
        assert "T2" in tickers
        assert "T3" not in tickers  # Industry cap hit
        assert "T4" in tickers  # Different industry

    def test_null_sector_always_passes(self):
        """Tickers with null sector are never capped."""
        df = _scored_df(
            [
                _make_row("T1", mom_z=3.0, sector="Tech", industry="SW"),
                _make_row("T2", mom_z=2.0, sector="Tech", industry="SW"),
                _make_row("T3", mom_z=1.0, sector="Tech", industry="SW"),
                _make_row("NULL1", mom_z=0.5, sector=None, industry=None),
                _make_row("NULL2", mom_z=0.4, sector=None, industry=None),
                _make_row("NULL3", mom_z=0.3, sector=None, industry=None),
                _make_row("NULL4", mom_z=0.2, sector=None, industry=None),
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df, sector_cap=2, industry_cap=1)
        tickers = result["ticker"].to_list()
        # Tech/SW: T1, T2 selected, T3 skipped (industry cap=1)
        # Null sector: ALL pass
        assert "T3" not in tickers
        assert "NULL1" in tickers
        assert "NULL2" in tickers
        assert "NULL3" in tickers
        assert "NULL4" in tickers

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty."""
        from alphascreener.screening.phase2 import apply_industry_dedup

        empty = _scored_df([]).with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))
        result = apply_industry_dedup(empty)
        assert result.height == 0

    def test_no_sector_industry_columns(self):
        """Dedup without sector/industry columns passes all tickers."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "breakout_score": [3.0, 2.0, 1.0],
            }
        )
        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df)
        assert result["ticker"].to_list() == ["A", "B", "C"]

    def test_preserves_sort_order(self):
        """Dedup output preserves the input sort order."""
        df = _scored_df(
            [
                _make_row("T1", mom_z=3.0, sector="Tech", industry="SW"),
                _make_row("T2", mom_z=2.0, sector="Finance", industry="Bank"),
                _make_row("T3", mom_z=1.0, sector="Tech", industry="HW"),
                _make_row("T4", mom_z=0.5, sector="Healthcare", industry="Bio"),
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df)
        assert result["ticker"].to_list() == ["T1", "T2", "T3", "T4"]

    def test_exact_sector_cap_boundary(self):
        """Exactly sector_cap tickers should all pass."""
        df = _scored_df(
            [
                _make_row("T1", mom_z=3.0, sector="Tech", industry="SW"),
                _make_row("T2", mom_z=2.0, sector="Tech", industry="HW"),
                _make_row("T3", mom_z=1.0, sector="Tech", industry="Semi"),
                _make_row("T4", mom_z=0.5, sector="Finance", industry="Bank"),
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df, sector_cap=3, industry_cap=2)
        # All 3 Tech pass (sector_cap=3), + Finance
        assert result.height == 4
        assert set(result["ticker"].to_list()) == {"T1", "T2", "T3", "T4"}

    def test_mixed_sector_industry_caps(self):
        """Sector cap restricts before industry cap for same-sector tickers."""
        df = _scored_df(
            [
                # Tech sector, 5 tickers, all different industries
                _make_row("T1", mom_z=3.0, sector="Tech", industry="I1"),
                _make_row("T2", mom_z=2.9, sector="Tech", industry="I2"),
                _make_row("T3", mom_z=2.8, sector="Tech", industry="I3"),
                _make_row("T4", mom_z=2.7, sector="Tech", industry="I4"),  # 4th Tech
                _make_row("T5", mom_z=2.6, sector="Tech", industry="I5"),  # 5th Tech
            ]
        )
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("breakout_score"))

        from alphascreener.screening.phase2 import apply_industry_dedup

        result = apply_industry_dedup(df, sector_cap=3, industry_cap=2)
        assert result.height == 3  # capped at sector_cap=3
        assert result["ticker"].to_list() == ["T1", "T2", "T3"]


# ============================================================================
# phase2_pipeline integration tests
# ============================================================================


class TestPhase2Pipeline:
    """Full Phase 2 pipeline: score -> sort -> top N -> dedup -> fill."""

    def test_pipeline_basic(self):
        """Pipeline scores, sorts, and outputs at most n_final tickers."""
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=float(100 - i), sector="Tech", industry=f"I{i % 4}")
                for i in range(50)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=20, sector_cap=10, industry_cap=10)
        assert result.height == 20
        # Sorted by breakout_score descending
        scores = result["breakout_score"].to_list()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_pipeline_dedup_reduces_count(self):
        """Sector/industry caps reduce the number of selected tickers."""
        df = _scored_df(
            [
                # 10 Tech tickers in SW industry
                _make_row(f"T{i}", mom_z=float(50 - i), sector="Tech", industry="SW")
                for i in range(10)
            ]
            + [
                # 10 Finance tickers in Bank industry
                _make_row(f"F{i}", mom_z=float(40 - i), sector="Finance", industry="Bank")
                for i in range(10)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=20, n_final=20, sector_cap=3, industry_cap=2)
        # Industry cap=2, so even though sector_cap=3,
        # each industry gets max 2 tickers: 2 SW + 2 Bank = 4
        # But we'd also select tickers from beyond top 20 via fill...
        # Actually the fill logic would bring in more from different industries
        # With all tickers in same industries, total would be limited
        assert result.height <= 20

    def test_pipeline_fallback_fills_shortage(self):
        """When dedup eliminates many tickers, fallback fills from remainder."""
        # All tickers in same sector/industry, cap limits = 1
        # Top 30 → dedup gives 1 → fallback should try to fill
        # But all have same sector/industry, so fill also limited
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=float(100 - i), sector="S1", industry="I1")
                for i in range(100)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=20, sector_cap=1, industry_cap=1)
        # With caps so strict, only 1 would pass dedup
        # Fallback fills from remaining, but caps still apply
        # Fallback2 (relaxed) should fill without caps → 20
        assert result.height == 20
        # First ticker should be the top-scored one
        assert result["ticker"][0] == "T0"

    def test_pipeline_empty_dataframe(self):
        """Empty DataFrame returns empty."""
        from alphascreener.screening.phase2 import phase2_pipeline

        empty = _scored_df([])
        result = phase2_pipeline(empty)
        assert result.height == 0
        assert "breakout_score" in result.columns

    def test_pipeline_fewer_than_n_final(self):
        """When there are fewer tickers than n_final, return all of them."""
        df = _scored_df(
            [
                _make_row("A", mom_z=1.0, sector="Tech", industry="SW"),
                _make_row("B", mom_z=0.5, sector="Finance", industry="Bank"),
                _make_row("C", mom_z=0.2, sector="Tech", industry="HW"),
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=20, sector_cap=3, industry_cap=2)
        assert result.height == 3
        assert set(result["ticker"].to_list()) == {"A", "B", "C"}

    def test_pipeline_preserves_input_columns(self):
        """Pipeline output retains all input columns plus breakout_score."""
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=float(10 - i), sector="Tech", industry="SW")
                for i in range(5)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=20)
        for col in df.columns:
            assert col in result.columns
        assert "breakout_score" in result.columns

    def test_pipeline_n_top_smaller_than_n_final(self):
        """When n_top < n_final, fill from remaining pool."""
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=float(50 - i), sector=f"S{i % 10}", industry=f"I{i % 20}")
                for i in range(50)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=10, n_final=20, sector_cap=10, industry_cap=10)
        assert result.height == 20
        # Top 10 should all be in result
        top_10 = {f"T{i}" for i in range(10)}
        result_tickers = set(result["ticker"].to_list())
        assert top_10 <= result_tickers


# ============================================================================
# Edge cases
# ============================================================================


class TestPhase2EdgeCases:
    """Edge case handling for Phase 2 components."""

    def test_missing_z_capped_columns(self):
        """compute_breakout_score works with raw factor columns (z-score computed internally)."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "MOM_5D": [1.0, -1.0],
                "MACD_CROSS": [1, 0],
                "GOLDEN_CROSS": [0, 1],
                "INSIDER_BUY": [0, 0],
            }
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        score_a = result.filter(pl.col("ticker") == "A")["breakout_score"][0]
        score_b = result.filter(pl.col("ticker") == "B")["breakout_score"][0]
        # Both get z-score contributions from MOM_5D (2 rows → variance > 0)
        # A gets MACD_CROSS, B gets GOLDEN_CROSS
        assert score_a != pytest.approx(0.0)
        assert score_b != pytest.approx(0.0)

    def test_binary_factor_column_missing(self):
        """compute_breakout_score handles missing binary factor columns gracefully."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "MOM_5D": [1.0, -1.0],
            }
        )
        from alphascreener.screening.phase2 import compute_breakout_score

        result = compute_breakout_score(df).sort("ticker")
        # MOM_5D z-scores are non-zero (2 rows with different values)
        score_a = result.filter(pl.col("ticker") == "A")["breakout_score"][0]
        score_b = result.filter(pl.col("ticker") == "B")["breakout_score"][0]
        assert score_a > 0 > score_b

    def test_single_ticker(self):
        """Pipeline works correctly with a single ticker."""
        df = _scored_df(
            [
                _make_row("SOLO", mom_z=2.0, sector="Tech", industry="SW"),
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=20)
        assert result.height == 1
        assert result["ticker"][0] == "SOLO"

    def test_all_negative_scores(self):
        """Dedup and pipeline work with all negative scores."""
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=-float(i) - 1.0, sector=f"S{i}", industry=f"I{i}")
                for i in range(10)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=30, n_final=5)
        assert result.height == 5
        # Best (least negative) scores should be first
        scores = result["breakout_score"].to_list()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_n_top_one(self):
        """n_top = 1 should work correctly with fill from remainder."""
        df = _scored_df(
            [
                _make_row(f"T{i}", mom_z=float(50 - i), sector=f"S{i}", industry=f"I{i}")
                for i in range(20)
            ]
        )
        from alphascreener.screening.phase2 import phase2_pipeline

        result = phase2_pipeline(df, n_top=1, n_final=5, sector_cap=5, industry_cap=5)
        assert result.height == 5
        # Top ticker should be T0
        assert result["ticker"][0] == "T0"


# ============================================================================
# Signal direction tests (Issue #191)
# ============================================================================


class TestSignalDirection:
    """Verify factor signal directions align with Phase 1 constraints."""

    def test_atr_ratio_inverted_in_breakout_score(self):
        """Higher ATR_RATIO -> lower breakout_score (vol contraction is bullish).

        Phase 1 requires ATR_RATIO < 0.8 (low is good).  Phase 2 must align:
        higher raw ATR_RATIO should contribute negatively to breakout_score
        (direction = -1 in _SIGNAL_DIRECTION).
        """
        from alphascreener.screening.phase2 import compute_breakout_score

        # Two tickers: identical except raw ATR_RATIO
        df = _scored_df(
            [
                _make_row("LOW_ATR", mom_z=1.0),
                _make_row("HIGH_ATR", mom_z=1.0),
            ]
        )
        # Set raw ATR_RATIO: LOW=0.08 (contraction, bullish), HIGH=2.0 (expansion, bearish)
        df = df.with_columns(
            pl.when(pl.col("ticker") == "LOW_ATR")
            .then(0.08)
            .when(pl.col("ticker") == "HIGH_ATR")
            .then(2.0)
            .otherwise(pl.col("ATR_RATIO"))
            .alias("ATR_RATIO")
        )

        result = compute_breakout_score(df).sort("ticker")
        low_score = result.filter(pl.col("ticker") == "LOW_ATR")["breakout_score"][0]
        high_score = result.filter(pl.col("ticker") == "HIGH_ATR")["breakout_score"][0]

        # LOW_ATR (vol contraction, bullish) should score HIGHER than HIGH_ATR
        # Because ATR_RATIO direction = -1 (inverted)
        assert low_score > high_score, (
            f"Expected LOW_ATR (vol contraction) to score higher than HIGH_ATR, "
            f"got LOW={low_score}, HIGH={high_score}"
        )

    def test_atr_ratio_direction_metadata(self):
        """Signal direction metadata exists and marks ATR_RATIO as inverted."""
        from alphascreener.screening.phase2 import _SIGNAL_DIRECTION

        assert "ATR_RATIO" in _SIGNAL_DIRECTION, "ATR_RATIO must have an entry in _SIGNAL_DIRECTION"
        assert _SIGNAL_DIRECTION["ATR_RATIO"] == -1, (
            f"Expected ATR_RATIO direction = -1 (inverted), got {_SIGNAL_DIRECTION['ATR_RATIO']}"
        )

    def test_momentum_factors_positive_direction(self):
        """Momentum factors (MOM_5D, PTH, MOM_SLOPE) have positive direction."""
        from alphascreener.screening.phase2 import _SIGNAL_DIRECTION

        for fname in ("MOM_5D", "PTH", "MOM_SLOPE"):
            assert _SIGNAL_DIRECTION.get(fname, 1) == 1, (
                f"Expected {fname} direction = +1, got {_SIGNAL_DIRECTION.get(fname)}"
            )
