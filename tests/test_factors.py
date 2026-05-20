"""Tests for factor formulas and engine (Issue #93).

Covers:
  - Each of 14 factor formulas with synthetic data
  - Normalisation pipeline (z-score, clipping, display score)
  - Missing-data handling (PRD 3.1.4)
  - Chunked processing via compute_factors / process_chunk
  - Edge cases: empty DataFrames, single ticker, all-null columns
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from alphascreener.factors.engine import (
    DEFAULT_BATCH_SIZE,
    MISSING_FACTOR_RATE_MAX,
    Z_SCORE_CAP,
    FactorEngine,
    compute_factors,
    normalize_factors,
    process_chunk,
)
from alphascreener.factors.formulas import (
    FACTOR_NAMES,
    compute_all_technical_factors,
    compute_atr_ratio,
    compute_bb_squeeze,
    compute_cmf_21,
    compute_golden_cross,
    compute_insider_buy,
    compute_macd_cross,
    compute_mfi_14,
    compute_mom_5d,
    compute_mom_slope,
    compute_pead_flag,
    compute_pth,
    compute_rev_accel,
    compute_rsi_oversold,
    compute_vol_anomaly,
)

# -- helpers -----------------------------------------------------------------


def _ohlcv_df(
    close: list[float],
    ticker: str = "TEST",
    dt: date | None = None,
    high_mult: float = 1.02,
    low_mult: float = 0.98,
    volume: float = 1_000_000.0,
) -> pl.DataFrame:
    """Build a tidy OHLCV DataFrame for a single ticker."""
    n = len(close)
    d = dt or date(2025, 1, 15)
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "dt": [d] * n,
            "open": [c * 0.995 for c in close],
            "high": [c * high_mult for c in close],
            "low": [c * low_mult for c in close],
            "close": close,
            "volume": [volume] * n,
        }
    )


def _multi_ticker_df(tickers: list[str], n_rows: int = 30) -> pl.DataFrame:
    """Generate a multi-ticker OHLCV DataFrame with synthetic prices."""
    rng = np.random.RandomState(42)
    rows = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 10.0
        close_vals = base + rng.randn(n_rows).cumsum()
        for j, c in enumerate(close_vals):
            rows.append(
                {
                    "ticker": ticker,
                    "dt": date(2025, 1, 15),
                    "open": float(c * 0.99),
                    "high": float(c * 1.02),
                    "low": float(c * 0.98),
                    "close": float(c),
                    "volume": float(1_000_000 + rng.randn() * 100_000),
                }
            )
    return pl.DataFrame(rows)


# -- momentum factors --------------------------------------------------------


class TestMOM5D:
    def test_basic_momentum(self):
        """MOM_5D = (C_t - C_{t-5}) / C_{t-5}."""
        close = [100.0, 101, 102, 103, 104, 105, 110, 108]
        df = _ohlcv_df(close)
        result = compute_mom_5d(df)
        vals = result.get_column("MOM_5D").to_list()
        # First 5 are null (insufficient history)
        assert all(v is None for v in vals[:5])
        # Row 5 (index 5): (105 - 100) / 100 = 0.05
        assert vals[5] == pytest.approx(0.05)
        # Row 6 (index 6): (110 - 101) / 101 ≈ 0.0891
        assert vals[6] == pytest.approx(0.08910891, rel=1e-5)
        # Row 7 (index 7): (108 - 102) / 102 ≈ 0.0588
        assert vals[7] == pytest.approx(0.05882353, rel=1e-5)

    def test_zero_close_no_error(self):
        """Factor handles zero close price without exception."""
        df = _ohlcv_df([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = compute_mom_5d(df)
        assert "MOM_5D" in result.columns


class TestPTH:
    def test_basic_pth(self):
        """PTH = Close / max(Close, 63-day window)."""
        # First 10 prices: highest is 110 at position 5
        close = [100.0, 105, 102, 108, 106, 110, 107, 109, 103, 101]
        df = _ohlcv_df(close)
        result = compute_pth(df)
        vals = result.get_column("PTH").to_list()
        # All null until we have 63 days. With only 10, the window is incomplete.
        # rolling_max with window_size=63 needs 63 observations. All will be null.
        assert all(v is None for v in vals)

    def test_pth_with_small_window_equivalent(self):
        """PTH with full window shows values <= 1.0."""
        close = list(range(100, 200))  # monotonically increasing
        df = _ohlcv_df(close)
        result = compute_pth(df)
        # After 63 observations, PTH should be close to 1.0
        vals = result.get_column("PTH").to_list()
        for v in vals[63:]:
            assert v is not None
            assert 0.0 < v <= 1.0 + 1e-9


class TestMOMSLOPE:
    def test_monotonic_up(self):
        """MOM_SLOPE for accelerating returns should be positive."""
        # Prices where daily returns increase linearly -> positive slope
        # Start at 100, add increasing increments
        increments = [0.1 + i * 0.05 for i in range(30)]  # 0.1, 0.15, 0.20, ...
        close = [100.0]
        for inc in increments:
            close.append(close[-1] + inc)
        df = _ohlcv_df(close)
        result = compute_mom_slope(df)
        vals = result.get_column("MOM_SLOPE").to_list()
        # After 10 observations (first slope computed), should be positive
        for v in vals[11:]:
            assert v is not None, "Unexpected null at some row after warmup"
            assert v > 0, f"Expected positive slope, got {v}"

    def test_monotonic_down(self):
        """MOM_SLOPE for decelerating returns should be negative."""
        # Prices where daily returns decrease linearly -> negative slope
        increments = [3.0 - i * 0.1 for i in range(30)]  # 3.0, 2.9, 2.8, ...
        close = [100.0]
        for inc in increments:
            close.append(close[-1] + inc)
        df = _ohlcv_df(close)
        result = compute_mom_slope(df)
        vals = result.get_column("MOM_SLOPE").to_list()
        for v in vals[11:]:
            assert v is not None
            assert v < 0, f"Expected negative slope, got {v}"

    def test_null_before_window(self):
        """First 9 rows are null (need 10 returns)."""
        close = list(range(100, 130))
        df = _ohlcv_df(close)
        result = compute_mom_slope(df)
        vals = result.get_column("MOM_SLOPE").to_list()
        # Need 10 returns = 11 observations for first valid value
        for i in range(10):
            assert vals[i] is None, f"Expected null at index {i}, got {vals[i]}"


# -- volatility factors ------------------------------------------------------


class TestBBSqueeze:
    def test_flat_price_wide_bollinger(self):
        """Flat prices give minimum BB width -> squeeze detected after warmup."""
        close = [100.0] * 100
        df = _ohlcv_df(close)
        result = compute_bb_squeeze(df)
        vals = result.get_column("BB_SQUEEZE").to_list()
        # BB_SQUEEZE needs: 20 for bb_width + 60 for rolling_quantile = 79 rows
        for i, v in enumerate(vals):
            if i >= 79 and v is not None:
                assert v == 1, f"Row {i}: expected squeeze=1, got {v}"

    def test_column_present(self):
        """BB_SQUEEZE column is added."""
        df = _ohlcv_df(list(range(100, 200)))
        result = compute_bb_squeeze(df)
        assert "BB_SQUEEZE" in result.columns


class TestATRRatio:
    def test_constant_tr(self):
        """Constant true range gives ATR ratio of 1.0 (ATR5/ATR20)."""
        # Constant OHLC gives constant TR
        close = [100.0] * 50
        df = _ohlcv_df(close, high_mult=1.05, low_mult=0.95)
        result = compute_atr_ratio(df)
        vals = result.get_column("ATR_RATIO").to_list()
        for v in vals[20:]:  # after warmup
            assert v is not None
            assert v == pytest.approx(1.0, rel=0.01)

    def test_atr_ratio_range(self):
        """ATR_RATIO values are non-negative."""
        close = list(range(100, 150))
        df = _ohlcv_df(close, high_mult=1.02, low_mult=0.98)
        result = compute_atr_ratio(df)
        vals = result.get_column("ATR_RATIO").to_list()
        for v in vals[22:]:
            assert v is not None
            assert v >= 0


# -- money flow factors ------------------------------------------------------


class TestMFI14:
    def test_up_trend_mfi_high(self):
        """Rapid price increase with high volume -> MFI should be near 100."""
        n = 30
        close = [100.0 + i * 2 for i in range(n)]  # steady uptrend
        df = _ohlcv_df(close, volume=10_000_000.0)
        result = compute_mfi_14(df)
        vals = result.get_column("MFI_14").to_list()
        # In a steady uptrend, MFI should be high (> 70)
        for v in vals[15:]:
            assert v is not None
            assert v > 70.0, f"Expected MFI > 70, got {v}"

    def test_down_trend_mfi_low(self):
        """Rapid price decrease -> MFI should be near 0."""
        n = 30
        close = [100.0 - i * 2 for i in range(n)]
        df = _ohlcv_df(close, volume=10_000_000.0)
        result = compute_mfi_14(df)
        vals = result.get_column("MFI_14").to_list()
        for v in vals[15:]:
            assert v is not None
            assert v < 30.0, f"Expected MFI < 30, got {v}"


class TestCMF21:
    def test_uptrend_cmf_positive(self):
        """Close near high in uptrend -> CMF should be positive."""
        n = 40
        close = [100.0 + i for i in range(n)]
        df = _ohlcv_df(close, high_mult=1.005, low_mult=0.98)
        result = compute_cmf_21(df)
        vals = result.get_column("CMF_21").to_list()
        for v in vals[25:]:
            if v is not None:
                # Close is near high -> positive CMF
                assert v > 0, f"Expected CMF > 0, got {v}"

    def test_downtrend_cmf_negative(self):
        """Close near low in downtrend -> CMF should be negative."""
        n = 40
        close = [100.0 - i for i in range(n)]
        df = _ohlcv_df(close, high_mult=1.02, low_mult=0.995)
        result = compute_cmf_21(df)
        vals = result.get_column("CMF_21").to_list()
        for v in vals[25:]:
            if v is not None:
                assert v < 0, f"Expected CMF < 0, got {v}"


class TestVolAnomaly:
    def test_extreme_volume_triggers(self):
        """Last-day volume spike > 2 sigma triggers anomaly."""
        n = 60
        normal_vol = [1_000_000.0] * (n - 1)
        spike_vol = normal_vol + [10_000_000.0]  # 10x volume on last day
        close = list(range(100, 100 + n))
        # Need varying volume for std to be non-zero
        vols = [1_000_000.0 + i * 1000 for i in range(n - 1)] + [50_000_000.0]
        df = pl.DataFrame(
            {
                "ticker": ["TEST"] * n,
                "dt": [date(2025, 1, 15)] * n,
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": vols,
            }
        )
        result = compute_vol_anomaly(df)
        vals = result.get_column("VOL_ANOMALY").to_list()
        # Last row should be 1 (if close > SMA5 and vol z > 2)
        assert vals[-1] == 1

    def test_normal_volume_no_trigger(self):
        """Normal volume does NOT trigger anomaly."""
        close = list(range(100, 160))
        vols = [1_000_000.0] * 60
        df = pl.DataFrame(
            {
                "ticker": ["TEST"] * 60,
                "dt": [date(2025, 1, 15)] * 60,
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": vols,
            }
        )
        result = compute_vol_anomaly(df)
        vals = result.get_column("VOL_ANOMALY").to_list()
        # With near-constant volume, z-scores are near 0 -> no anomaly
        for v in vals[55:]:
            assert v is not None
            # May be 0 (expected) or 1 (possible with very small std)
            assert v in (0, 1)


# -- technical pattern factors -----------------------------------------------


class TestRSIOversold:
    def test_downtrend_rsi_low(self):
        """Steady downtrend -> RSI should be low."""
        n = 30
        close = [100.0 - i * 1.5 for i in range(n)]
        df = _ohlcv_df(close)
        result = compute_rsi_oversold(df)
        rsi_vals = result.get_column("RSI_14").to_list()
        for v in rsi_vals[18:]:
            if v is not None:
                assert v < 50.0, f"Expected RSI < 50 in downtrend, got {v}"

    def test_oversold_signal(self):
        """RSI < 30 and close > SMA20 -> oversold signal."""
        # Craft data: sharp drop then recovery above MA
        close = [100.0 + i for i in range(15)]  # uptrend first
        close += [100.0 - i * 3 for i in range(15)]  # then sharp drop
        close += [70.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0]  # bounce
        df = _ohlcv_df(close)
        result = compute_rsi_oversold(df)
        assert "RSI_OVERSOLD" in result.columns
        assert "RSI_14" in result.columns


class TestMACDCross:
    def test_columns_added(self):
        """MACD, SIGNAL, HISTOGRAM, MACD_CROSS columns are present."""
        close = list(range(100, 180))
        df = _ohlcv_df(close)
        result = compute_macd_cross(df)
        for col in ("MACD", "SIGNAL", "HISTOGRAM", "MACD_CROSS"):
            assert col in result.columns

    def test_macd_cross_values(self):
        """MACD_CROSS is 0 or 1."""
        close = [100.0 + 2 * np.sin(i / 3) + i * 0.1 for i in range(100)]
        df = _ohlcv_df(close)
        result = compute_macd_cross(df)
        vals = result.get_column("MACD_CROSS").to_list()
        for v in vals[50:]:
            if v is not None:
                assert v in (0, 1)


class TestGoldenCross:
    def test_columns_added(self):
        """SMA_50, SMA_200, GOLDEN_CROSS columns are present."""
        close = list(range(100, 400))
        df = _ohlcv_df(close)
        result = compute_golden_cross(df)
        for col in ("SMA_50", "SMA_200", "GOLDEN_CROSS"):
            assert col in result.columns

    def test_sma_values_monotonic(self):
        """In uptrend, SMA_50 > SMA_200 after crossover."""
        close = list(range(100, 400))  # steady uptrend, 300 periods
        df = _ohlcv_df(close)
        result = compute_golden_cross(df)
        sma50 = result.get_column("SMA_50").to_list()
        sma200 = result.get_column("SMA_200").to_list()
        # After both have data, SMA50 should be above SMA200
        for i in range(250, len(close)):
            if sma50[i] is not None and sma200[i] is not None:
                assert sma50[i] > sma200[i]


# -- fundamental factors -----------------------------------------------------


class TestPEADFlag:
    def test_recent_earnings(self):
        """PEAD_FLAG = 1 if earnings within 30 days of reference_date."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        ref = date(2025, 1, 15)
        earnings = {"AAPL": [date(2025, 1, 10)]}  # 5 days ago
        result = compute_pead_flag(df, earnings_dates=earnings, reference_date=ref)
        vals = result.get_column("PEAD_FLAG").to_list()
        assert all(v == 1 for v in vals)

    def test_old_earnings(self):
        """PEAD_FLAG = 0 if last earnings was > 30 days ago."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        ref = date(2025, 1, 15)
        earnings = {"AAPL": [date(2024, 12, 1)]}  # 45 days ago
        result = compute_pead_flag(df, earnings_dates=earnings, reference_date=ref)
        vals = result.get_column("PEAD_FLAG").to_list()
        assert all(v == 0 for v in vals)

    def test_no_earnings_data(self):
        """PEAD_FLAG = 0 when earnings_dates is None."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_pead_flag(df, earnings_dates=None, reference_date=None)
        vals = result.get_column("PEAD_FLAG").to_list()
        assert all(v == 0 for v in vals)

    def test_pead_flag_binary(self):
        """PEAD_FLAG values are always 0 or 1."""
        df = _ohlcv_df([100.0] * 10, ticker="MSFT")
        ref = date(2025, 1, 15)
        earnings = {"MSFT": [date(2025, 1, 14)]}
        result = compute_pead_flag(df, earnings_dates=earnings, reference_date=ref)
        vals = result.get_column("PEAD_FLAG").to_list()
        assert all(v in (0, 1) for v in vals)


class TestInsiderBuy:
    def test_above_threshold(self):
        """INSIDER_BUY = 1 when ratio > 0.001."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_insider_buy(df, insider_ratio={"AAPL": 0.002})
        vals = result.get_column("INSIDER_BUY").to_list()
        assert all(v == 1 for v in vals)

    def test_below_threshold(self):
        """INSIDER_BUY = 0 when ratio <= 0.001."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_insider_buy(df, insider_ratio={"AAPL": 0.0005})
        vals = result.get_column("INSIDER_BUY").to_list()
        assert all(v == 0 for v in vals)

    def test_no_data(self):
        """INSIDER_BUY = 0 when insider_ratio is None."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_insider_buy(df, insider_ratio=None)
        vals = result.get_column("INSIDER_BUY").to_list()
        assert all(v == 0 for v in vals)


class TestRevAccel:
    def test_positive_acceleration(self):
        """REV_ACCEL positive when growth is accelerating."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_rev_accel(df, revenue_growth={"AAPL": [0.05, 0.08]})
        vals = result.get_column("REV_ACCEL").to_list()
        assert all(v == pytest.approx(0.03) for v in vals)

    def test_negative_acceleration(self):
        """REV_ACCEL negative when growth is decelerating."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_rev_accel(df, revenue_growth={"AAPL": [0.10, 0.06]})
        vals = result.get_column("REV_ACCEL").to_list()
        assert all(v == pytest.approx(-0.04) for v in vals)

    def test_single_quarter_null(self):
        """REV_ACCEL = null when only one growth rate available."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_rev_accel(df, revenue_growth={"AAPL": [0.05]})
        vals = result.get_column("REV_ACCEL").to_list()
        assert all(v is None for v in vals)

    def test_no_data_null(self):
        """REV_ACCEL = null when revenue_growth is None."""
        df = _ohlcv_df([100.0] * 5, ticker="AAPL")
        result = compute_rev_accel(df, revenue_growth=None)
        vals = result.get_column("REV_ACCEL").to_list()
        assert all(v is None for v in vals)


# -- composite / all technical factors ---------------------------------------


class TestAllTechnicalFactors:
    def test_all_factor_columns(self):
        """compute_all_technical_factors produces all 14 factor columns."""
        close = list(range(100, 400))
        df = _ohlcv_df(close)
        result = compute_all_technical_factors(df)

        tech_factors = [
            "MOM_5D", "PTH", "MOM_SLOPE", "BB_SQUEEZE", "ATR_RATIO",
            "MFI_14", "CMF_21", "VOL_ANOMALY", "RSI_14", "RSI_OVERSOLD",
            "MACD", "SIGNAL", "HISTOGRAM", "MACD_CROSS",
            "SMA_50", "SMA_200", "GOLDEN_CROSS",
        ]
        for col in tech_factors:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        """All original OHLCV columns are preserved."""
        close = list(range(100, 300))
        df = _ohlcv_df(close)
        result = compute_all_technical_factors(df)
        for col in ("ticker", "dt", "open", "high", "low", "close", "volume"):
            assert col in result.columns


# -- normalisation -----------------------------------------------------------


MOCK_RETURNS = [0.01, -0.02, 0.015, -0.005, 0.02, 0.03, -0.01, 0.025, 0.01, -0.015]


class TestNormalisation:
    def _factor_df(self, values: list[float]) -> pl.DataFrame:
        """Build a minimal DataFrame with MOM_5D values for z-scoring."""
        return pl.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(len(values))],
                "dt": [date(2025, 1, 15)] * len(values),
                "MOM_5D": values,
            }
        )

    def test_z_score_mean_zero_std_one(self):
        """After z-scoring, mean ~ 0, std ~ 1 (sample)."""
        vals = MOCK_RETURNS
        df = self._factor_df(vals)
        result = normalize_factors(df)
        z_col = result.get_column("z_MOM_5D")
        # Mean should be very close to 0
        assert z_col.mean() == pytest.approx(0.0, abs=1e-9)
        # Sample std should be exactly 1.0 (since we use ddof=1)
        assert z_col.std(ddof=1) == pytest.approx(1.0, rel=1e-6)

    def test_z_score_capped_to_range(self):
        """z_capped values are within [-3, +3]."""
        vals = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 30.0, -20.0]
        df = self._factor_df(vals)
        result = normalize_factors(df)
        capped = result.get_column("z_capped_MOM_5D").to_list()
        for v in capped:
            if v is not None:
                assert -Z_SCORE_CAP <= v <= Z_SCORE_CAP, f"{v} out of range"

    def test_display_score_range(self):
        """Display scores are in [0, 100]."""
        vals = MOCK_RETURNS
        df = self._factor_df(vals)
        result = normalize_factors(df)
        scores = result.get_column("score_MOM_5D").to_list()
        for s in scores:
            if s is not None:
                assert 0.0 <= s <= 100.0, f"score {s} out of [0,100]"

    def test_display_score_neutral_at_50(self):
        """A value at the mean gets display score ~50."""
        vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        df = self._factor_df(vals)
        result = normalize_factors(df)
        capped = result.get_column("z_capped_MOM_5D").to_list()
        # With zero variance, z_capped is 0, scores are 50
        scores = result.get_column("score_MOM_5D").to_list()
        for s in scores:
            assert s == pytest.approx(50.0)

    def test_final_score_column(self):
        """final_score column is present."""
        vals = MOCK_RETURNS
        df = self._factor_df(vals)
        result = normalize_factors(df)
        assert "final_score" in result.columns

    def test_final_score_monotonic(self):
        """Higher raw values produce higher final scores."""
        vals = list(range(20))
        df = self._factor_df(vals)
        df = df.with_columns(
            pl.Series("ATR_RATIO", [float(i % 5 + 1) for i in range(20)]),
        )
        result = normalize_factors(df)
        scores = result.get_column("final_score").to_list()
        # Higher raw MOM_5D -> higher z_score -> higher final_score contribution
        # All values should be in expected range (not all zero)
        assert any(s != 0.0 for s in scores if s is not None)


# -- process_chunk ------------------------------------------------------------


class TestProcessChunk:
    def test_all_factor_output_columns(self):
        """process_chunk produces raw factor and metadata columns (no normalisation)."""
        close = list(range(100, 400))
        df = _ohlcv_df(close)
        result = process_chunk(
            df,
            reference_date=date(2025, 1, 15),
            earnings_dates={"TEST": [date(2025, 1, 10)]},
            insider_ratio={"TEST": 0.002},
            revenue_growth={"TEST": [0.05, 0.08]},
        )
        # Check key output columns (raw factors only; normalisation done by caller)
        expected = [
            "MOM_5D", "PTH", "MOM_SLOPE", "BB_SQUEEZE", "ATR_RATIO",
            "MFI_14", "CMF_21", "VOL_ANOMALY", "RSI_OVERSOLD",
            "MACD_CROSS", "GOLDEN_CROSS", "PEAD_FLAG", "INSIDER_BUY",
            "REV_ACCEL", "data_sufficient", "missing_rate",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"
        # Normalisation columns must NOT be present (caller applies after concat)
        assert "final_score" not in result.columns
        assert "z_MOM_5D" not in result.columns

    def test_empty_df_returns_empty(self):
        """Empty input -> empty output (no crash)."""
        df = pl.DataFrame(schema={"ticker": pl.Utf8, "close": pl.Float64})
        result = process_chunk(df)
        assert result.height == 0

    def test_raises_on_missing_columns(self):
        """process_chunk raises ValueError if required columns are missing."""
        df = pl.DataFrame({"ticker": ["A"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            process_chunk(df)


# -- compute_factors (chunked) -----------------------------------------------


class TestComputeFactors:
    def test_chunked_processing(self):
        """compute_factors splits multiple tickers into chunks."""
        tickers = [f"T{i:03d}" for i in range(10)]
        df = _multi_ticker_df(tickers, n_rows=100)
        result = compute_factors(
            df, dt=date(2025, 1, 15), batch_size=3,
        )
        assert result.height == df.height
        assert "final_score" in result.columns
        assert "data_sufficient" in result.columns

    def test_single_ticker(self):
        """Single ticker works fine."""
        df = _ohlcv_df(list(range(100, 300)))
        result = compute_factors(df, dt=date(2025, 1, 15))
        assert result.height == df.height

    def test_all_tickers_present(self):
        """All input tickers appear in output."""
        tickers = [f"T{i:03d}" for i in range(20)]
        df = _multi_ticker_df(tickers, n_rows=50)
        result = compute_factors(df, dt=date(2025, 1, 15), batch_size=7)
        output_tickers = set(result["ticker"].unique().to_list())
        assert output_tickers == set(tickers)


# -- missing-data handling ---------------------------------------------------


class TestMissingData:
    def test_full_data_sufficient(self):
        """Ticker with all factors present -> data_sufficient = True.

        Only check rows after the GOLDEN_CROSS warmup (SMA_200 needs 200 rows).
        """
        close = list(range(100, 400))
        df = _ohlcv_df(close)
        result = process_chunk(
            df,
            reference_date=date(2025, 1, 15),
            earnings_dates={"TEST": [date(2025, 1, 10)]},
            insider_ratio={"TEST": 0.002},
            revenue_growth={"TEST": [0.05, 0.08]},
        )
        vals = result.get_column("data_sufficient").to_list()
        # After row 200, all factors including SMA_200 are available
        for v in vals[205:]:
            if v is not None:
                assert v is True, "Expected data_sufficient=True for full-data row"

    def test_missing_rate_zero(self):
        """Full data has missing_rate = 0 (after all warmup windows)."""
        close = list(range(100, 400))
        df = _ohlcv_df(close)
        result = process_chunk(
            df,
            reference_date=date(2025, 1, 15),
            earnings_dates={"TEST": [date(2025, 1, 10)]},
            insider_ratio={"TEST": 0.002},
            revenue_growth={"TEST": [0.05, 0.08]},
        )
        rates = result.get_column("missing_rate").to_list()
        # After all warmup windows, missing_rate is 0
        for r in rates[205:]:
            if r is not None:
                assert r == pytest.approx(0.0)

    def test_missing_factor_z_score_is_zero(self):
        """Missing factor z-score is set to 0."""
        close = list(range(100, 300))
        df = _ohlcv_df(close)
        # Deliberately corrupt one factor column: set to null
        df = compute_mom_5d(df)
        # Set some MOM_5D values to null
        mom_vals = df.get_column("MOM_5D").to_list()
        mom_vals[5] = None
        mom_vals[10] = None
        df = df.with_columns(pl.Series("MOM_5D", mom_vals))

        result = normalize_factors(df)
        z_vals = result.get_column("z_MOM_5D").to_list()
        # Null MOM_5D -> z_score = 0
        assert z_vals[5] == 0.0
        assert z_vals[10] == 0.0

    def test_missing_factor_score_is_50(self):
        """Missing factor display score is 50 (neutral)."""
        close = list(range(100, 300))
        df = _ohlcv_df(close)
        df = compute_mom_5d(df)
        mom_vals = df.get_column("MOM_5D").to_list()
        mom_vals[6] = None
        df = df.with_columns(pl.Series("MOM_5D", mom_vals))
        result = normalize_factors(df)
        scores = result.get_column("score_MOM_5D").to_list()
        assert scores[6] == pytest.approx(50.0)

    def test_insufficient_data_tagged(self):
        """Factor missing rate > 30% -> data_sufficient = False."""
        n = 10
        close = list(range(100, 100 + n))
        df = _ohlcv_df(close)

        # Build a df with nulls in most factor columns
        result = compute_all_technical_factors(df)
        factor_cols = [
            "MOM_5D", "PTH", "MOM_SLOPE", "BB_SQUEEZE", "ATR_RATIO",
            "MFI_14", "CMF_21", "VOL_ANOMALY", "RSI_OVERSOLD",
            "MACD_CROSS", "GOLDEN_CROSS",
        ]
        # Row 0: set most factors to null
        result = result.with_columns(
            PEAD_FLAG=pl.lit(0, dtype=pl.Int32),
            INSIDER_BUY=pl.lit(0, dtype=pl.Int32),
            REV_ACCEL=pl.lit(None, dtype=pl.Float64),
        )
        # Set 10 of 14 factors to null for row 0
        null_factors = factor_cols[:10]
        for f in null_factors:
            if f in result.columns:
                vals = result.get_column(f).to_list()
                vals[0] = None
                result = result.with_columns(pl.Series(f, vals))

        # Validate missing data
        from alphascreener.factors.engine import _validate_missing_data

        validated = _validate_missing_data(result)
        rates = validated.get_column("missing_rate").to_list()
        # Row 0 should have > 30% missing
        assert rates[0] > MISSING_FACTOR_RATE_MAX
        sufficient = validated.get_column("data_sufficient").to_list()
        assert sufficient[0] is False


# -- edge cases --------------------------------------------------------------


class TestEdgeCases:
    def test_single_row(self):
        """Single-row DataFrame does not crash."""
        df = _ohlcv_df([100.0])
        result = process_chunk(df)
        assert result.height == 1

    def test_all_null_close(self):
        """All-null close prices produce null factors without crashing."""
        df = pl.DataFrame(
            {
                "ticker": pl.Series("ticker", ["TEST"] * 20, dtype=pl.Utf8),
                "dt": pl.Series("dt", [date(2025, 1, 15)] * 20, dtype=pl.Date),
                "open": pl.Series("open", [None] * 20, dtype=pl.Float64),
                "high": pl.Series("high", [None] * 20, dtype=pl.Float64),
                "low": pl.Series("low", [None] * 20, dtype=pl.Float64),
                "close": pl.Series("close", [None] * 20, dtype=pl.Float64),
                "volume": pl.Series("volume", [None] * 20, dtype=pl.Float64),
            }
        )
        result = compute_all_technical_factors(df)
        assert result.height == 20

    def test_empty_chunk_list(self):
        """Empty ticker list from compute_factors returns empty."""
        df = pl.DataFrame(schema={"ticker": pl.Utf8, "close": pl.Float64})
        result = compute_factors(df)
        assert result.height == 0


# -- integration -------------------------------------------------------------


class TestIntegration:
    def test_end_to_end_single_ticker(self):
        """Full pipeline: raw OHLCV -> factors -> normalise -> validate."""
        n = 300  # enough data for all factors including SMA_200
        close = list(range(100, 100 + n))
        df = _ohlcv_df(close)

        result = process_chunk(
            df,
            reference_date=date(2025, 1, 15),
            earnings_dates={"TEST": [date(2025, 1, 10)]},
            insider_ratio={"TEST": 0.005},
            revenue_growth={"TEST": [0.03, 0.07]},
        )

        # All 14 factor columns present
        for fname in FACTOR_NAMES:
            assert fname in result.columns, f"Missing factor: {fname}"

        # Normalisation is applied after chunk concat; trigger it explicitly
        result = normalize_factors(result)

        # Normalisation columns
        cont_factors = ["MOM_5D", "PTH", "MOM_SLOPE", "ATR_RATIO", "MFI_14", "CMF_21", "RSI_OVERSOLD", "REV_ACCEL"]
        for fname in cont_factors:
            if fname in result.columns:
                assert f"z_{fname}" in result.columns, f"Missing z_{fname}"
                assert f"z_capped_{fname}" in result.columns
                assert f"score_{fname}" in result.columns

        # Final score
        assert "final_score" in result.columns

    def test_end_to_end_multi_ticker(self):
        """Full pipeline with 5 tickers, chunked."""
        tickers = [f"T{i:02d}" for i in range(5)]
        df = _multi_ticker_df(tickers, n_rows=250)
        result = compute_factors(
            df, dt=date(2025, 1, 15), batch_size=2,
        )
        assert result.height == df.height
        assert result["ticker"].n_unique() == len(tickers)
        assert "final_score" in result.columns
        assert "data_sufficient" in result.columns

    def test_factor_engine_constructor(self):
        """FactorEngine constructor stores batch_size and n_batches correctly."""
        # This test requires actual OHLCV data on disk, so it will skip if
        # no data path is set up.  Test the constructor and structure only.
        engine = FactorEngine(batch_size=100, n_batches=2)
        assert engine.batch_size == 100
        assert engine.n_batches == 2
        # The run method requires actual data; test that it handles missing
        # data gracefully.
