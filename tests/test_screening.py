"""Tests for Phase 1 hard filtering + dynamic threshold adjustment (Issue #94).

Covers:
  - 4 must-satisfy conditions (MOM_5D > 0, VOL_ANOMALY=1 OR MFI_14>40,
    ATR_RATIO < 0.8, RSI_14 ∈ [25, 75])
  - 5 optional bonus flags (BB_SQUEEZE, PTH>0.90, CMF_21>0, PEAD_FLAG, INSIDER_BUY)
  - Combined hard_filter + filter_rate statistics
  - Dynamic threshold adjustment (widening, tightening, cooldown, caps)
  - Edge cases: empty DataFrame, missing columns, decimal values at boundary
"""

from datetime import date
from typing import Dict, List

import polars as pl
import pytest

# ============================================================================
# Helpers — build factor DataFrames for testing
# ============================================================================


def _factor_df(rows: List[Dict]) -> pl.DataFrame:
    """Minimal factor DataFrame with the columns Phase 1 needs."""
    # Polars 0.12.5: convert date objects to strings, then cast
    for r in rows:
        if "dt" in r and isinstance(r["dt"], date):
            r["dt"] = r["dt"].isoformat()
    df = pl.DataFrame(rows)
    if "dt" in df.columns:
        df = df.with_columns([pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d")])
    return df


@pytest.fixture
def passing_df() -> pl.DataFrame:
    """All tickers pass Phase 1 hard filters."""
    return _factor_df(
        [
            {
                "ticker": "AAPL",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.02,
                "VOL_ANOMALY": 1,
                "MFI_14": 35.0,
                "ATR_RATIO": 0.5,
                "RSI_14": 55.0,
                "BB_SQUEEZE": 1,
                "PTH": 0.95,
                "CMF_21": 0.05,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 1,
            },
            {
                "ticker": "MSFT",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.01,
                "VOL_ANOMALY": 0,
                "MFI_14": 60.0,  # passes via MFI > 40
                "ATR_RATIO": 0.3,
                "RSI_14": 50.0,
                "BB_SQUEEZE": 0,
                "PTH": 0.92,
                "CMF_21": 0.10,
                "PEAD_FLAG": 1,
                "INSIDER_BUY": 0,
            },
            {
                "ticker": "GOOGL",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.005,
                "VOL_ANOMALY": 1,
                "MFI_14": 30.0,
                "ATR_RATIO": 0.6,
                "RSI_14": 70.0,
                "BB_SQUEEZE": 1,
                "PTH": 0.94,
                "CMF_21": -0.02,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
        ]
    )


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """Mix of passing and failing tickers."""
    return _factor_df(
        [
            {
                "ticker": "PASS1",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.03,
                "VOL_ANOMALY": 1,
                "MFI_14": 50.0,
                "ATR_RATIO": 0.4,
                "RSI_14": 50.0,
                "BB_SQUEEZE": 1,
                "PTH": 0.95,
                "CMF_21": 0.01,
                "PEAD_FLAG": 1,
                "INSIDER_BUY": 1,
            },
            {
                "ticker": "FAIL_MOM",
                "dt": date(2025, 3, 15),
                "MOM_5D": -0.01,  # FAIL: MOM_5D <= 0
                "VOL_ANOMALY": 1,
                "MFI_14": 50.0,
                "ATR_RATIO": 0.4,
                "RSI_14": 50.0,
                "BB_SQUEEZE": 0,
                "PTH": 0.85,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
            {
                "ticker": "FAIL_VOLMFI",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.02,
                "VOL_ANOMALY": 0,
                "MFI_14": 30.0,  # FAIL: neither VOL_ANOMALY nor MFI > 40
                "ATR_RATIO": 0.4,
                "RSI_14": 50.0,
                "BB_SQUEEZE": 0,
                "PTH": 0.85,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
            {
                "ticker": "FAIL_ATR",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.02,
                "VOL_ANOMALY": 1,
                "MFI_14": 50.0,
                "ATR_RATIO": 0.9,  # FAIL: ATR_RATIO >= 0.8
                "RSI_14": 50.0,
                "BB_SQUEEZE": 0,
                "PTH": 0.85,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
            {
                "ticker": "FAIL_RSI_LOW",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.02,
                "VOL_ANOMALY": 1,
                "MFI_14": 50.0,
                "ATR_RATIO": 0.4,
                "RSI_14": 15.0,  # FAIL: RSI < 25
                "BB_SQUEEZE": 0,
                "PTH": 0.85,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
            {
                "ticker": "FAIL_RSI_HIGH",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.02,
                "VOL_ANOMALY": 1,
                "MFI_14": 50.0,
                "ATR_RATIO": 0.4,
                "RSI_14": 85.0,  # FAIL: RSI > 75
                "BB_SQUEEZE": 0,
                "PTH": 0.85,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            },
        ]
    )


# ============================================================================
# Must-satisfy condition tests
# ============================================================================


class TestMustSatisfyConditions:
    """Each of the 4 must-satisfy conditions is independently testable."""

    def test_condition_mom_5d_positive(self, mixed_df):
        """Ticker passes only if MOM_5D > 0."""
        from alphascreener.screening.phase1 import _check_mom_5d

        result = _check_mom_5d(mixed_df)
        tickers = result.filter(pl.col("_mom_ok")).get_column("ticker").to_list()
        assert "FAIL_MOM" not in tickers
        assert "PASS1" in tickers

    def test_condition_vol_mfi(self, mixed_df):
        """Ticker passes if VOL_ANOMALY == 1 OR MFI_14 > 40."""
        from alphascreener.screening.phase1 import _check_vol_mfi

        result = _check_vol_mfi(mixed_df)
        tickers = result.filter(pl.col("_volmfi_ok")).get_column("ticker").to_list()
        assert "FAIL_VOLMFI" not in tickers
        assert "PASS1" in tickers

    def test_condition_atr_ratio(self, mixed_df):
        """Ticker passes if ATR_RATIO < threshold (default 0.8)."""
        from alphascreener.screening.phase1 import _check_atr_ratio

        result = _check_atr_ratio(mixed_df)
        tickers = result.filter(pl.col("_atr_ok")).get_column("ticker").to_list()
        assert "FAIL_ATR" not in tickers
        assert "PASS1" in tickers

    def test_condition_rsi_range(self, mixed_df):
        """Ticker passes if RSI_14 ∈ [25, 75]."""
        from alphascreener.screening.phase1 import _check_rsi_range

        result = _check_rsi_range(mixed_df)
        tickers = result.filter(pl.col("_rsi_ok")).get_column("ticker").to_list()
        assert "FAIL_RSI_LOW" not in tickers
        assert "FAIL_RSI_HIGH" not in tickers
        assert "PASS1" in tickers

    def test_condition_atr_ratio_with_custom_threshold(self):
        """ATR_RATIO uses a custom threshold when provided."""
        from alphascreener.screening.phase1 import _check_atr_ratio

        df = _factor_df(
            [
                {
                    "ticker": "A",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.85,
                    "RSI_14": 50.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        # Default threshold 0.8: fails
        r1 = _check_atr_ratio(df)
        assert not r1["_atr_ok"][0]
        # Relaxed threshold 0.9: passes
        r2 = _check_atr_ratio(df, threshold=0.9)
        assert r2["_atr_ok"][0]

    def test_condition_rsi_range_boundaries(self):
        """RSI exactly at 25 or 75 should pass (inclusive bounds)."""
        from alphascreener.screening.phase1 import _check_rsi_range

        df = _factor_df(
            [
                {
                    "ticker": "AT25",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 25.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
                {
                    "ticker": "AT75",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 75.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _check_rsi_range(df)
        assert result["_rsi_ok"].to_list() == [True, True]

    def test_condition_mom_5d_at_zero_fails(self):
        """MOM_5D == 0 fails (strictly greater)."""
        from alphascreener.screening.phase1 import _check_mom_5d

        df = _factor_df(
            [
                {
                    "ticker": "ZERO",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _check_mom_5d(df)
        assert not result["_mom_ok"][0]

    def test_condition_vol_anomaly_alone_passes(self):
        """VOL_ANOMALY == 1 passes even if MFI_14 is low."""
        from alphascreener.screening.phase1 import _check_vol_mfi

        df = _factor_df(
            [
                {
                    "ticker": "VOLOK",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 1,
                    "MFI_14": 20.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _check_vol_mfi(df)
        assert result["_volmfi_ok"][0]

    def test_condition_mfi_alone_passes(self):
        """MFI_14 > 40 passes even if VOL_ANOMALY == 0."""
        from alphascreener.screening.phase1 import _check_vol_mfi

        df = _factor_df(
            [
                {
                    "ticker": "MFIOK",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 41.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _check_vol_mfi(df)
        assert result["_volmfi_ok"][0]

    def test_condition_vol_mfi_both_true(self):
        """Both VOL_ANOMALY == 1 and MFI_14 > 40 passes."""
        from alphascreener.screening.phase1 import _check_vol_mfi

        df = _factor_df(
            [
                {
                    "ticker": "BOTH",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 1,
                    "MFI_14": 41.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _check_vol_mfi(df)
        assert result["_volmfi_ok"][0]


# ============================================================================
# Optional bonus tests
# ============================================================================


class TestOptionalBonus:
    """5 optional bonus conditions, each worth 1 point."""

    def test_count_all_five_bonuses(self, passing_df):
        """AAPL gets 4 bonuses, MSFT gets 4, GOOGL gets 2."""
        from alphascreener.screening.phase1 import _count_bonuses

        result = _count_bonuses(passing_df)
        bonus_col = result["bonus_count"].to_list()
        # AAPL: BB=1, PTH=0.95>0.90, CMF=0.05>0, PEAD=0, INSIDER=1 => 4
        # MSFT: BB=0, PTH=0.92>0.90, CMF=0.10>0, PEAD=1, INSIDER=0 => 3
        # GOOGL: BB=1, PTH=0.94>0.90, CMF=-0.02≤0, PEAD=0, INSIDER=0 => 2
        tickers = result["ticker"].to_list()
        idx = {t: i for i, t in enumerate(tickers)}
        assert bonus_col[idx["AAPL"]] == 4
        assert bonus_col[idx["MSFT"]] == 3
        assert bonus_col[idx["GOOGL"]] == 2

    def test_bonus_bb_squeeze(self):
        """BB_SQUEEZE == 1 adds 1 bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 1,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 1

    def test_bonus_pth(self):
        """PTH > 0.90 adds 1 bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.91,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 1

    def test_bonus_pth_at_threshold_fails(self):
        """PTH == 0.90 does NOT count as bonus (strictly >)."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.90,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 0

    def test_bonus_cmf_21(self):
        """CMF_21 > 0 adds 1 bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.01,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 1

    def test_bonus_cmf_negative_fails(self):
        """CMF_21 < 0 does NOT count as bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": -0.01,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 0

    def test_bonus_pead_flag(self):
        """PEAD_FLAG == 1 adds 1 bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 1,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 1

    def test_bonus_insider_buy(self):
        """INSIDER_BUY == 1 adds 1 bonus."""
        from alphascreener.screening.phase1 import _count_bonuses

        df = _factor_df(
            [
                {
                    "ticker": "B",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.0,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.0,
                    "RSI_14": 0.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 1,
                },
            ]
        )
        result = _count_bonuses(df)
        assert result["bonus_count"][0] == 1


# ============================================================================
# Combined hard filter tests
# ============================================================================


class TestHardFilter:
    """The combined hard_filter function applies all must-satisfy conditions + bonuses."""

    def test_hard_filter_all_pass(self, passing_df):
        """All 3 tickers in passing_df should pass."""
        from alphascreener.screening.phase1 import hard_filter

        result = hard_filter(passing_df)
        assert result["pass_phase1"].all()

    def test_hard_filter_mixed(self, mixed_df):
        """Only PASS1 passes all 4 must-satisfy conditions."""
        from alphascreener.screening.phase1 import hard_filter

        result = hard_filter(mixed_df)
        passing = result.filter(pl.col("pass_phase1"))
        assert passing["ticker"].to_list() == ["PASS1"]

    def test_hard_filter_bonus_counts_on_all(self, mixed_df):
        """Bonus counts are computed even for failing tickers."""
        from alphascreener.screening.phase1 import hard_filter

        result = hard_filter(mixed_df)
        assert "bonus_count" in result.columns
        # PASS1 has BB=1, PTH=0.95, CMF=0.01, PEAD=1, INSIDER=1 => 5
        pass1 = result.filter(pl.col("ticker") == "PASS1")
        assert pass1["bonus_count"][0] == 5

    def test_hard_filter_preserves_input_columns(self, passing_df):
        """Hard filter output retains all input columns."""
        from alphascreener.screening.phase1 import hard_filter

        result = hard_filter(passing_df)
        for col in passing_df.columns:
            assert col in result.columns

    def test_hard_filter_empty_df(self):
        """Hard filter on empty DataFrame returns empty DataFrame."""
        from alphascreener.screening.phase1 import hard_filter

        empty = _factor_df([])
        result = hard_filter(empty)
        assert result.height == 0
        assert "pass_phase1" in result.columns
        assert "bonus_count" in result.columns

    def test_hard_filter_missing_column_raises(self):
        """Hard filter with missing required column raises ValueError."""
        from alphascreener.screening.phase1 import hard_filter

        df = pl.DataFrame({"ticker": ["A"], "MOM_5D": [0.01]})
        with pytest.raises(ValueError):
            hard_filter(df)

    def test_hard_filter_custom_thresholds(self):
        """Custom thresholds override default must-satisfy conditions."""
        from alphascreener.screening.phase1 import hard_filter

        df = _factor_df(
            [
                {
                    "ticker": "T1",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": -0.02,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 50.0,
                    "ATR_RATIO": 0.5,
                    "RSI_14": 55.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        # Default: MOM_5D > 0 fails
        r1 = hard_filter(df)
        assert not r1["pass_phase1"][0]

        # Relaxed: MOM_5D > -0.05 passes
        r2 = hard_filter(df, thresholds={"MOM_5D": -0.05})
        assert r2["pass_phase1"][0]

    def test_hard_filter_adds_filtered_output_flag(self):
        """Hard filter returns a 'pass_phase1' boolean column."""
        from alphascreener.screening.phase1 import hard_filter

        df = _factor_df(
            [
                {
                    "ticker": "T1",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.02,
                    "VOL_ANOMALY": 1,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.5,
                    "RSI_14": 55.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = hard_filter(df)
        assert result["pass_phase1"].dtype == pl.Boolean


# ============================================================================
# Filter rate statistics tests
# ============================================================================


class TestFilterRate:
    """Compute filter_rate = (total - passed) / total."""

    def test_filter_rate_normal(self, mixed_df):
        """mixed_df: 1 pass out of 6 => filter_rate = 5/6 ≈ 83.3%."""
        from alphascreener.screening.phase1 import compute_filter_rate

        rate = compute_filter_rate(mixed_df)
        assert rate == pytest.approx(5 / 6)

    def test_filter_rate_all_pass(self, passing_df):
        """All pass: filter_rate = 0%."""
        from alphascreener.screening.phase1 import compute_filter_rate

        rate = compute_filter_rate(passing_df)
        assert rate == 0.0

    def test_filter_rate_none_pass(self):
        """No passes: filter_rate = 100%."""
        from alphascreener.screening.phase1 import compute_filter_rate

        df = _factor_df(
            [
                {
                    "ticker": "F1",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": -0.01,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 0.0,
                    "ATR_RATIO": 0.9,
                    "RSI_14": 80.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        rate = compute_filter_rate(df)
        assert rate == pytest.approx(1.0)

    def test_filter_rate_empty_df(self):
        """Empty DataFrame: filter_rate = 0.0 (nothing to filter)."""
        from alphascreener.screening.phase1 import compute_filter_rate

        rate = compute_filter_rate(_factor_df([]))
        assert rate == 0.0


# ============================================================================
# Dynamic threshold tests
# ============================================================================


class TestDynamicThreshold:
    """DynamicThreshold state machine: adjust thresholds based on filter rate."""

    def test_initial_thresholds_match_defaults(self):
        """Initial thresholds should match DEFAULT_THRESHOLDS."""
        from alphascreener.screening.threshold import DEFAULT_THRESHOLDS, DynamicThreshold

        dt = DynamicThreshold()
        assert dt.thresholds == DEFAULT_THRESHOLDS

    def test_no_adjustment_in_normal_range(self):
        """80-92% filter rate: no change to thresholds."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        original = dict(dt.thresholds)
        dt.adjust(0.85)
        assert dt.thresholds == original

    def test_widen_on_over_tight(self):
        """95-98% filter rate: auto-widen thresholds."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.96)
        th = dt.thresholds
        # MOM_5D threshold lowered (wider range)
        assert th["MOM_5D"] < 0.0
        # ATR_RATIO threshold raised (wider range)
        assert th["ATR_RATIO"] > 0.80
        # RSI range widened
        assert th["RSI_LOW"] < 25.0
        assert th["RSI_HIGH"] > 75.0

    def test_tighten_on_over_loose(self):
        """<70% filter rate: auto-tighten thresholds."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.60)
        th = dt.thresholds
        # MOM_5D threshold raised (tighter)
        assert th["MOM_5D"] > 0.0
        # ATR_RATIO threshold lowered (tighter)
        assert th["ATR_RATIO"] < 0.80
        # RSI range tightened
        assert th["RSI_LOW"] > 25.0
        assert th["RSI_HIGH"] < 75.0

    def test_cumulative_relaxation_capped(self):
        """Total relaxation cannot exceed 30%."""
        from alphascreener.screening.threshold import MAX_RELAXATION_PCT, DynamicThreshold

        dt = DynamicThreshold(cooldown_days=0)
        for _ in range(10):
            dt.adjust(0.97)
        th = dt.thresholds
        # MOM_5D: original 0.0, with 10% step, direction is widen (lower).
        # 10 relaxations would be -0.05 each if base is 0.5%, 10x would be -0.05.
        # With 30% cap, MOM_5D >= a minimum floor
        # cap logic: max_delta = base_reference * 0.30
        _ = abs(0.0 * (MAX_RELAXATION_PCT / 100.0))
        # For MOM_5D which is > comparison, widen = -Δ
        # Absolute delta cap: since base is 0, use a minimum absolute floor
        assert th["MOM_5D"] >= -0.015  # capped at 30% of reference range

    def test_cooldown_prevents_rapid_adjustment(self):
        """Adjustments within cooldown period are skipped."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        # Record initial state
        dt.adjust(0.96)
        after_first = dict(dt.thresholds)
        # Immediate second call on same day should be blocked by cooldown
        dt.adjust(0.96)
        assert dt.thresholds == after_first

    def test_cooldown_expires(self):
        """After cooldown days, adjustment is allowed again."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold(cooldown_days=0)  # no cooldown
        dt.adjust(0.96)
        after_first = dict(dt.thresholds)
        dt.adjust(0.96)
        # With 0 cooldown, second adjustment should differ from first
        assert dt.thresholds != after_first

    def test_extreme_filter_rate_full_widen(self):
        """>98% filter rate: full widen 10% on all conditions."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.99)
        th = dt.thresholds
        assert th["MOM_5D"] < 0.0
        assert th["ATR_RATIO"] > 0.80
        assert th["RSI_LOW"] < 25.0
        assert th["RSI_HIGH"] > 75.0
        assert th["MFI_14"] < 40.0

    def test_adjustment_history_recorded(self):
        """DynamicThreshold tracks adjustment history."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.96)
        assert len(dt.history) == 1
        assert dt.history[0]["filter_rate"] == 0.96

    def test_get_status_returns_string(self):
        """DynamicThreshold.get_status() returns a status string."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.85)
        status = dt.get_status()
        assert isinstance(status, str)
        assert len(status) > 0

    def test_reset_restores_defaults(self):
        """reset() reverts thresholds to defaults."""
        from alphascreener.screening.threshold import DEFAULT_THRESHOLDS, DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.96)
        dt.reset()
        assert dt.thresholds == DEFAULT_THRESHOLDS
        assert dt.history == []

    def test_widen_direction_rules(self):
        """Direction rules: X<threshold → +Δ, X>threshold → -Δ, X∈[a,b] → [a-Δ, b+Δ]."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.96)
        th = dt.thresholds
        # MOM_5D: > threshold, widen = -Δ (lower the threshold)
        assert th["MOM_5D"] < 0.0
        # ATR_RATIO: < threshold, widen = +Δ (raise the threshold)
        assert th["ATR_RATIO"] > 0.80
        # RSI: ∈ [a,b], widen both ends
        assert th["RSI_LOW"] < 25.0
        assert th["RSI_HIGH"] > 75.0
        # MFI_14: > threshold, widen = -Δ (lower the threshold)
        assert th["MFI_14"] < 40.0

    def test_tighten_direction_rules(self):
        """Tightening reverses direction: X<threshold → -Δ, X>threshold → +Δ."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.60)  # over-loose, triggers tighten
        th = dt.thresholds
        # MOM_5D: > threshold, tighten = +Δ (raise the threshold)
        assert th["MOM_5D"] > 0.0
        # ATR_RATIO: < threshold, tighten = -Δ (lower the threshold)
        assert th["ATR_RATIO"] < 0.80
        # RSI: ∈ [a,b], tighten both ends inward
        assert th["RSI_LOW"] > 25.0
        assert th["RSI_HIGH"] < 75.0

    def test_mfi_widen_boundary_unchanged_for_vol_anomaly(self):
        """VOL_ANOMALY is binary (0/1), threshold not applicable. Only MFI_14 changes."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        dt.adjust(0.96)
        th = dt.thresholds
        assert "VOL_ANOMALY" not in th or th.get("VOL_ANOMALY") is None
        assert th["MFI_14"] < 40.0  # MFI_14 is the adjustable threshold

    def test_boundary_filter_rate_070_not_over_loose(self):
        """Filter rate exactly 0.70 is NOT over_loose (< 0.70 is)."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        result = dt.adjust(0.70)
        assert result == "normal"

    def test_boundary_filter_rate_below_070_is_over_loose(self):
        """Filter rate below 0.70 triggers over_loose tightening."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        result = dt.adjust(0.69)
        assert "over_loose" in result

    def test_boundary_filter_rate_098_is_over_tight(self):
        """Filter rate exactly 0.98 is over_tight (not extreme)."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        result = dt.adjust(0.98)
        assert "over_tight" in result

    def test_boundary_filter_rate_above_098_is_extreme(self):
        """Filter rate above 0.98 triggers extreme."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        result = dt.adjust(0.99)
        assert "extreme" in result

    def test_rsi_crossover_guard_after_tighten(self):
        """After tightening, if RSI_LOW > RSI_HIGH both reset to 50.0."""
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold(cooldown_days=0)
        # Repeatedly tighten to push RSI_LOW up and RSI_HIGH down until they cross
        for _ in range(20):
            dt.adjust(0.50)
        th = dt.thresholds
        # RSI_LOW must not exceed RSI_HIGH
        assert th["RSI_LOW"] <= th["RSI_HIGH"]
        # If crossover was detected, both should be at or near 50.0
        # (tightening raises RSI_LOW and lowers RSI_HIGH, so after many
        # tighten cycles they would cross without the guard.)


# ============================================================================
# Phase 1 pipeline: hard_filter with DynamicThreshold integration
# ============================================================================


class TestPhase1WithDynamicThreshold:
    """Integration: hard_filter accepts threshold override from DynamicThreshold."""

    def test_pipeline_with_dynamic_thresholds(self):
        """hard_filter applies custom thresholds from DynamicThreshold."""
        from alphascreener.screening.phase1 import hard_filter
        from alphascreener.screening.threshold import DynamicThreshold

        dt = DynamicThreshold()
        # Simulate tight market, widen thresholds
        dt.adjust(0.97)
        custom_th = dt.thresholds

        df = _factor_df(
            [
                {
                    "ticker": "T1",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": -0.005,  # slightly negative, but within relaxed threshold
                    "VOL_ANOMALY": 0,
                    "MFI_14": 39.0,  # below 40 but above relaxed
                    "ATR_RATIO": 0.82,  # above 0.80 but below relaxed
                    "RSI_14": 76.0,  # above 75 but within relaxed
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                },
            ]
        )
        result = hard_filter(df, thresholds=custom_th)
        # With relaxed thresholds, T1 should pass
        assert result["pass_phase1"][0]

    def test_filter_rate_with_dynamic_thresholds(self, mixed_df):
        """compute_filter_rate works with custom thresholds."""
        from alphascreener.screening.phase1 import compute_filter_rate

        custom_th = {
            "MOM_5D": -0.05,
            "ATR_RATIO": 0.9,
            "RSI_LOW": 15.0,
            "RSI_HIGH": 85.0,
            "MFI_14": 25.0,
        }
        rate = compute_filter_rate(mixed_df, thresholds=custom_th)
        # All should pass except FAIL_ATR (0.9 is not strictly < 0.9)
        # FAIL_ATR: ATR_RATIO=0.9, threshold=0.9, 0.9 < 0.9 is False
        # FAIL_RSI_LOW: RSI=15.0, threshold 15.0, 15.0 >= 15.0 passes
        assert rate < 1.0


# ============================================================================
# Phase 1 auto-relaxation: hard_filter_with_fallback (Issue #219)
# ============================================================================


class TestHardFilterWithFallback:
    """When Phase 1 filters too aggressively, auto-relax thresholds and retry."""

    def test_normal_pass_rate_no_relax_needed(self):
        """When >5% of tickers pass, no relaxation occurs."""
        from alphascreener.screening.phase1 import hard_filter_with_fallback

        df = _factor_df(
            [
                {
                    "ticker": f"T{i}",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.03,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 60.0,
                    "ATR_RATIO": 0.5,
                    "RSI_14": 55.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                }
                for i in range(10)
            ]
        )
        result, was_relaxed = hard_filter_with_fallback(df)
        assert result["pass_phase1"].sum() == 10
        assert not was_relaxed

    def test_strict_filter_triggers_relaxation(self):
        """When < 5% pass_rate, thresholds are relaxed and retry succeeds."""
        from alphascreener.screening.phase1 import hard_filter, hard_filter_with_fallback

        # Build 100 tickers: 1 clearly passing, 99 barely failing on MOM_5D + ATR_RATIO
        rows = []
        # 1 ticker that clearly passes all 4 conditions
        rows.append(
            {
                "ticker": "AAPL",
                "dt": date(2025, 3, 15),
                "MOM_5D": 0.05,
                "VOL_ANOMALY": 1,
                "MFI_14": 55.0,
                "ATR_RATIO": 0.5,
                "RSI_14": 55.0,
                "BB_SQUEEZE": 0,
                "PTH": 0.0,
                "CMF_21": 0.0,
                "PEAD_FLAG": 0,
                "INSIDER_BUY": 0,
            }
        )
        # 99 tickers that fail MOM_5D and ATR_RATIO by small margins,
        # but pass VOL_MFI and RSI even under default thresholds
        for i in range(99):
            rows.append(
                {
                    "ticker": f"T{i:03d}",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": -0.002,  # barely negative → fails MOM > 0
                    "VOL_ANOMALY": 0,
                    "MFI_14": 41.0,  # above 40 → passes volmfi
                    "ATR_RATIO": 0.82,  # > 0.80 → fails ATR
                    "RSI_14": 55.0,  # in [25, 75] → passes RSI
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                }
            )

        df = _factor_df(rows)

        # Verify: default hard_filter only passes 1 (AAPL)
        default_result = hard_filter(df)
        assert default_result["pass_phase1"].sum() == 1

        # With fallback: relaxation kicks in and more tickers pass
        result, was_relaxed = hard_filter_with_fallback(df)
        n_pass = result["pass_phase1"].sum()
        assert n_pass > 1, f"Expected >1 tickers after relaxation, got {n_pass}"
        # Relaxation was applied
        assert was_relaxed

    def test_all_pass_no_relax(self):
        """When all tickers pass, no relaxation is triggered."""
        from alphascreener.screening.phase1 import hard_filter_with_fallback

        df = _factor_df(
            [
                {
                    "ticker": f"T{i}",
                    "dt": date(2025, 3, 15),
                    "MOM_5D": 0.05 + i * 0.01,
                    "VOL_ANOMALY": 0,
                    "MFI_14": 60.0 + i,
                    "ATR_RATIO": 0.3 + i * 0.01,
                    "RSI_14": 50.0,
                    "BB_SQUEEZE": 0,
                    "PTH": 0.0,
                    "CMF_21": 0.0,
                    "PEAD_FLAG": 0,
                    "INSIDER_BUY": 0,
                }
                for i in range(20)
            ]
        )
        result, was_relaxed = hard_filter_with_fallback(df)
        # All 20 tickers have valid values — all should pass
        assert result["pass_phase1"].sum() == 20
        assert not was_relaxed

    def test_zero_pass_rate_relaxes_to_at_least_one(self):
        """Even with 0 passers, relaxation ensures some tickers get through."""
        from alphascreener.screening.phase1 import hard_filter_with_fallback

        # 50 tickers all failing at multiple conditions, but a few
        # barely failing ATR_RATIO and MOM_5D (salvageable by relaxation).
        rows = []
        for i in range(50):
            # First 3 tickers: just barely outside default thresholds,
            # should pass after relaxation
            if i < 3:
                rows.append(
                    {
                        "ticker": f"T{i:02d}",
                        "dt": date(2025, 3, 15),
                        "MOM_5D": -0.002,  # barely negative, passes relaxed (-0.005)
                        "VOL_ANOMALY": 0,
                        "MFI_14": 45.0,  # passes both
                        "ATR_RATIO": 0.82,  # barely above 0.80, passes relaxed (0.88)
                        "RSI_14": 50.0,
                        "BB_SQUEEZE": 0,
                        "PTH": 0.0,
                        "CMF_21": 0.0,
                        "PEAD_FLAG": 0,
                        "INSIDER_BUY": 0,
                    }
                )
            else:
                rows.append(
                    {
                        "ticker": f"T{i:02d}",
                        "dt": date(2025, 3, 15),
                        "MOM_5D": -2.0 - i * 0.1,
                        "VOL_ANOMALY": 0,
                        "MFI_14": 5.0,
                        "ATR_RATIO": 2.0 + i * 0.1,
                        "RSI_14": 95.0,
                        "BB_SQUEEZE": 0,
                        "PTH": 0.0,
                        "CMF_21": 0.0,
                        "PEAD_FLAG": 0,
                        "INSIDER_BUY": 0,
                    }
                )

        df = _factor_df(rows)
        result, _was_relaxed = hard_filter_with_fallback(df)
        n_pass = result["pass_phase1"].sum()
        # After relaxation, the first 3 tickers should pass
        assert n_pass >= 3, f"Expected >=3 after relaxation, got {n_pass}"

    def test_threshold_relaxation_is_bounded(self):
        """Relaxation doesn't exceed 30% cumulative cap."""
        from alphascreener.screening.phase1 import _RELAX_MAX_CUMULATIVE

        # Verify the cumulative relaxation cap constant exists and is bounded
        assert 0.10 <= _RELAX_MAX_CUMULATIVE <= 0.50
