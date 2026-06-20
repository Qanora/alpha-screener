"""Factor formula implementations (Issue #93).

14 factors across 5 categories, computed via pure polars window expressions:

  Momentum:   MOM_5D, PTH, MOM_SLOPE
  Volatility: BB_SQUEEZE, ATR_RATIO
  Money Flow: MFI_14, CMF_21, VOL_ANOMALY
  Technical:  RSI_OVERSOLD, MACD_CROSS, GOLDEN_CROSS
  Fundamentals: PEAD_FLAG, INSIDER_BUY, REV_ACCEL

Reference: PRD 3.1.2.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

_logger = logging.getLogger("alphascreener.factors.formulas")

# -- minimum-history helpers (Issue #225) --------------------------------------


def _get_min_ticker_rows(df: pl.DataFrame) -> int:
    """Return the minimum number of rows per ticker in the DataFrame."""
    if "ticker" not in df.columns:
        return df.height
    return df.group_by("ticker").len().get_column("len").min()


def _warn_insufficient_history(
    factor_name: str,
    required: int,
    available: int,
) -> None:
    """Log a WARNING when a factor requires more history than available."""
    _logger.warning(
        "%s requires %d rows per ticker but min available is %d. "
        "Factor will be null for affected tickers.",
        factor_name,
        required,
        available,
    )


def _check_min_history(
    df: pl.DataFrame,
    *requirements: tuple[str, int],
) -> None:
    """Warn for each factor whose required window exceeds available ticker history.

    All checks are performed in a single group-by pass, avoiding the per-function
    overhead of repeated ``_get_min_ticker_rows`` calls in the chained
    ``compute_all_technical_factors`` pipeline.

    Args:
        df: OHLCV DataFrame.
        *requirements: Pairs of ``(factor_name, min_rows_required)``.
    """
    min_rows = _get_min_ticker_rows(df)
    for name, required in requirements:
        if min_rows < required:
            _warn_insufficient_history(name, required, min_rows)

# -- factor name constants ---------------------------------------------------

MOMENTUM_FACTORS: tuple[str, ...] = ("MOM_5D", "PTH", "MOM_SLOPE")
VOLATILITY_FACTORS: tuple[str, ...] = ("BB_SQUEEZE", "ATR_RATIO")
MONEY_FLOW_FACTORS: tuple[str, ...] = ("MFI_14", "CMF_21", "VOL_ANOMALY")
TECHNICAL_FACTORS: tuple[str, ...] = (
    "RSI_OVERSOLD",
    "MACD_CROSS",
    "GOLDEN_CROSS",
)
FUNDAMENTAL_FACTORS: tuple[str, ...] = ("PEAD_FLAG", "INSIDER_BUY", "REV_ACCEL")

FACTOR_NAMES: tuple[str, ...] = (
    MOMENTUM_FACTORS
    + VOLATILITY_FACTORS
    + MONEY_FLOW_FACTORS
    + TECHNICAL_FACTORS
    + FUNDAMENTAL_FACTORS
)

# -- momentum factors --------------------------------------------------------


def compute_mom_5d(df: pl.DataFrame) -> pl.DataFrame:
    """MOM_5D = (Close_t - Close_{t-5}) / Close_{t-5}.

    Uses ``over("ticker")`` so that lagged values are per-ticker (Issue #225).

    Returns the input DataFrame with a ``MOM_5D`` column added.
    """
    prev_close5 = pl.col("close").shift(5).over("ticker")
    return df.with_columns(
        ((pl.col("close") - prev_close5) / prev_close5).alias("MOM_5D")
    )


def compute_pth(df: pl.DataFrame) -> pl.DataFrame:
    """PTH = Close_t / max(Close, past 63 trading days).

    Hard-filter threshold > 0.90 at screening time; the raw ratio is stored.

    Uses ``over("ticker")`` so that rolling_max is computed per-ticker (Issue #225).

    Returns the input DataFrame with a ``PTH`` column added.
    """
    return df.with_columns(
        (
            pl.col("close")
            / pl.col("close").rolling_max(window_size=63).over("ticker")
        ).alias("PTH")
    )


def compute_mom_slope(df: pl.DataFrame) -> pl.DataFrame:
    """MOM_SLOPE = linear-regression slope of daily returns over trailing 10 days.

    Computed efficiently via weighted-sum over 10 lagged return columns using the
    standard OLS formula: slope = cov(x, r) / var(x).

    For a fixed 10-day window with x = [0, 1, ..., 9], var(x) = 8.25 and the
    numerator is Σ((x_i - x̄) * r_i) where x̄ = 4.5.

    Uses ``over("ticker")`` so that lagged returns are per-ticker (Issue #225).

    Returns the input DataFrame with a ``MOM_SLOPE`` column added.
    """
    daily_ret = pl.col("close").pct_change().over("ticker")
    # Centred weights: x_i - x̄ for i=0..9
    w = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    numerator = daily_ret.shift(9).over("ticker") * w[0]
    for i in range(1, 10):
        numerator = numerator + daily_ret.shift(9 - i).over("ticker") * w[i]
    slope = numerator / 82.5  # var(x) for n=10
    return df.with_columns(slope.alias("MOM_SLOPE"))


# -- volatility factors ------------------------------------------------------


def compute_bb_squeeze(df: pl.DataFrame) -> pl.DataFrame:
    """BB_SQUEEZE = 1 if BB_Width < 20th-percentile of BB_Width over last 60 days.

    BB_Width = (upper_band - lower_band) / SMA_20 = 4 * std_20 / SMA_20.

    Uses ``over("ticker")`` for per-ticker rolling statistics (Issue #225).

    Returns the input DataFrame with a ``BB_SQUEEZE`` column added (i32, 0 or 1).
    """
    sma20 = pl.col("close").rolling_mean(window_size=20).over("ticker")
    std20 = pl.col("close").rolling_std(window_size=20).over("ticker")
    # Guard against div-by-zero when sma20 == 0
    bb_width = pl.when(sma20 > 1e-12).then(4.0 * std20 / sma20).otherwise(None)
    # 20th percentile of bb_width over trailing 60 days
    bb_pct20 = bb_width.rolling_quantile(quantile=0.20, window_size=60).over("ticker")
    squeeze = (
        pl.when(
            bb_width.is_not_null()
            & bb_pct20.is_not_null()
            & ((bb_width < bb_pct20) | (bb_width == 0.0))
        )
        .then(1)
        .otherwise(0)
    )
    return df.with_columns(squeeze.alias("BB_SQUEEZE"))


def compute_atr_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """ATR_RATIO = ATR(5) / ATR(20).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    ATR(n) = rolling_mean of True Range over n periods.

    Squeeze signal when ATR_RATIO < 0.8.

    Division-by-zero guard (Issue #225): when ATR(20) <= 1e-12 or null,
    ATR_RATIO is set to null.  Uses ``over("ticker")`` for per-ticker rolling means.

    Returns the input DataFrame with an ``ATR_RATIO`` column added.
    """
    prev_close = pl.col("close").shift(1).over("ticker")
    high_low = pl.col("high") - pl.col("low")
    high_prev = (pl.col("high") - prev_close).abs()
    low_prev = (pl.col("low") - prev_close).abs()
    true_range = pl.max_horizontal(high_low, high_prev, low_prev)

    atr5 = true_range.rolling_mean(window_size=5).over("ticker")
    atr20 = true_range.rolling_mean(window_size=20).over("ticker")

    atr_ratio = pl.when(atr20 > 1e-12).then(atr5 / atr20).otherwise(None)

    return df.with_columns(atr_ratio.alias("ATR_RATIO"))


# -- money flow factors ------------------------------------------------------


def compute_mfi_14(df: pl.DataFrame) -> pl.DataFrame:
    """MFI_14 = 100 - 100 / (1 + Money_Flow_Ratio).

    TP = (High + Low + Close) / 3.
    Raw MF = TP * Volume.
    Positive MF = Raw MF when TP > previous TP, else 0.
    Negative MF = Raw MF when TP < previous TP, else 0.
    MFR = sum(Positive_MF, 14) / sum(Negative_MF, 14).

    Uses Wilder-style smoothing (EMA with alpha = 1/14) on positive/negative MF
    for a standard RSI-like smoothing.

    Returns the input DataFrame with an ``MFI_14`` column added.
    """
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    prev_tp = tp.shift(1).over("ticker")
    raw_mf = tp * pl.col("volume")

    pos_mf = pl.when(tp > prev_tp).then(raw_mf).otherwise(0.0)
    neg_mf = pl.when(tp < prev_tp).then(raw_mf).otherwise(0.0)

    # Wilder smoothing: EMA with alpha = 1/14, per-ticker
    alpha = 1.0 / 14.0
    avg_pos = pos_mf.ewm_mean(alpha=alpha, adjust=False).over("ticker")
    avg_neg = neg_mf.ewm_mean(alpha=alpha, adjust=False).over("ticker")

    mfr = avg_pos / avg_neg
    mfi = 100.0 - 100.0 / (1.0 + mfr)

    # Handle edge: when avg_neg is 0, MFR → inf, MFI → 100
    mfi = pl.when(avg_neg == 0.0).then(100.0).otherwise(mfi)
    # When both are 0, MFI → 50 (neutral)
    mfi = pl.when((avg_pos == 0.0) & (avg_neg == 0.0)).then(50.0).otherwise(mfi)

    return df.with_columns(mfi.alias("MFI_14"))


def compute_cmf_21(df: pl.DataFrame) -> pl.DataFrame:
    """CMF_21 = sum(MF_Volume, 21) / sum(Volume, 21).

    MF_Multiplier = ((Close - Low) - (High - Close)) / (High - Low), 0 if H==L.
    MF_Volume = MF_Multiplier * Volume.

    Uses ``over("ticker")`` so that rolling sums are per-ticker (Issue #225).

    Returns the input DataFrame with a ``CMF_21`` column added.
    """
    hl_range = pl.col("high") - pl.col("low")
    mf_mult = (
        pl.when(hl_range == 0.0)
        .then(0.0)
        .otherwise(
            ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / hl_range
        )
    )
    mf_volume = mf_mult * pl.col("volume")

    cmf = (
        mf_volume.rolling_sum(window_size=21).over("ticker")
        / pl.col("volume").rolling_sum(window_size=21).over("ticker")
    )

    return df.with_columns(cmf.alias("CMF_21"))


def compute_vol_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """VOL_ANOMALY = 1 if Volume z-score(50d) > 2.0 AND Close > SMA(5), else 0.

    Volume z-score = (volume - SMA_volume_50) / Std_volume_50.

    Uses ``over("ticker")`` for per-ticker rolling statistics (Issue #225).
    A WARNING is emitted when any ticker has fewer than 50 rows (the z-score window).

    Returns the input DataFrame with a ``VOL_ANOMALY`` column added (i32, 0 or 1).
    """
    vol_sma50 = pl.col("volume").rolling_mean(window_size=50).over("ticker")
    vol_std50 = pl.col("volume").rolling_std(window_size=50).over("ticker")
    # Guard against div-by-zero in z-score
    vol_z = pl.when(vol_std50 > 1e-12).then(
        (pl.col("volume") - vol_sma50) / vol_std50
    ).otherwise(0.0)
    close_sma5 = pl.col("close").rolling_mean(window_size=5).over("ticker")

    anomaly = (
        pl.when(
            (vol_z > 2.0)
            & (pl.col("close") > close_sma5)
            & vol_z.is_not_null()
            & close_sma5.is_not_null()
        )
        .then(1)
        .otherwise(0)
    )

    return df.with_columns(anomaly.alias("VOL_ANOMALY"))


# -- technical pattern factors -----------------------------------------------


def compute_rsi_oversold(df: pl.DataFrame) -> pl.DataFrame:
    """RSI_OVERSOLD = 1 if RSI(14) < 30 AND Close > SMA(20), else 0.

    RSI(14) uses Wilder smoothing (EMA alpha=1/14).

    Uses ``over("ticker")`` for per-ticker differencing and rolling mean (Issue #225).

    Returns the input DataFrame with both ``RSI_OVERSOLD`` (i32) and ``RSI_14``
    (f64, intermediate) columns added.
    """
    delta = pl.col("close").diff().over("ticker")
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)

    alpha = 1.0 / 14.0
    avg_gain = gain.ewm_mean(alpha=alpha, adjust=False).over("ticker")
    avg_loss = loss.ewm_mean(alpha=alpha, adjust=False).over("ticker")

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Edge: when avg_loss = 0, RSI = 100
    rsi = pl.when(avg_loss == 0.0).then(100.0).otherwise(rsi)
    # Edge: when both = 0, RSI = 50
    rsi = pl.when((avg_gain == 0.0) & (avg_loss == 0.0)).then(50.0).otherwise(rsi)

    sma20 = pl.col("close").rolling_mean(window_size=20).over("ticker")
    oversold = (
        pl.when((rsi < 30.0) & (pl.col("close") > sma20) & rsi.is_not_null() & sma20.is_not_null())
        .then(1)
        .otherwise(0)
    )

    return df.with_columns(rsi.alias("RSI_14"), oversold.alias("RSI_OVERSOLD"))


def compute_macd_cross(df: pl.DataFrame) -> pl.DataFrame:
    """MACD_CROSS = 1 if MACD > Signal AND Histogram just crossed above 0.

    MACD = EMA(12) - EMA(26).
    Signal = EMA(MACD, 9).
    Histogram = MACD - Signal.

    Cross-up event: Histogram_{t} > 0 AND Histogram_{t-1} <= 0 AND MACD > Signal.

    Returns the input DataFrame with ``MACD``, ``SIGNAL``, ``HISTOGRAM``, and
    ``MACD_CROSS`` columns added.
    """
    ema12 = pl.col("close").ewm_mean(span=12).over("ticker")
    ema26 = pl.col("close").ewm_mean(span=26).over("ticker")
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm_mean(span=9).over("ticker")
    histogram = macd_line - signal_line

    prev_histogram = histogram.shift(1).over("ticker")
    crossed_up = (
        (histogram > 0.0)
        & (prev_histogram <= 0.0)
        & histogram.is_not_null()
        & prev_histogram.is_not_null()
    )

    return df.with_columns(
        macd_line.alias("MACD"),
        signal_line.alias("SIGNAL"),
        histogram.alias("HISTOGRAM"),
        pl.when(crossed_up).then(1).otherwise(0).alias("MACD_CROSS"),
    )


def compute_golden_cross(df: pl.DataFrame) -> pl.DataFrame:
    """GOLDEN_CROSS = 1 when SMA(50) crosses above SMA(200).

    Cross-up: SMA(50)_t > SMA(200)_t AND SMA(50)_{t-1} <= SMA(200)_{t-1}.

    Uses ``over("ticker")`` so that rolling means are per-ticker (Issue #225).
    A WARNING is emitted when any ticker has fewer rows than the required window.

    Returns the input DataFrame with ``SMA_50``, ``SMA_200``, and ``GOLDEN_CROSS``
    columns added.
    """
    sma50 = pl.col("close").rolling_mean(window_size=50).over("ticker")
    sma200 = pl.col("close").rolling_mean(window_size=200).over("ticker")
    prev_sma50 = sma50.shift(1).over("ticker")
    prev_sma200 = sma200.shift(1).over("ticker")

    crossed_up = (
        (sma50 > sma200)
        & (prev_sma50 <= prev_sma200)
        & sma50.is_not_null()
        & sma200.is_not_null()
        & prev_sma50.is_not_null()
        & prev_sma200.is_not_null()
    )

    return df.with_columns(
        sma50.alias("SMA_50"),
        sma200.alias("SMA_200"),
        pl.when(crossed_up).then(1).otherwise(0).alias("GOLDEN_CROSS"),
    )


# -- fundamental factors -----------------------------------------------------


def compute_pead_flag(
    df: pl.DataFrame,
    *,
    earnings_dates: dict[str, list[date]] | None = None,
    reference_date: date | None = None,
) -> pl.DataFrame:
    """PEAD_FLAG = 1 if an earnings release occurred within the last 30 calendar days.

    Coarse-screening dummy variable.  Requires a mapping of ``ticker -> [earnings_dates]``
    and the *reference_date* (typically the observation date).

    If *earnings_dates* is ``None`` or a ticker has no recorded earnings, the flag
    is set to 0 (not activated).

    Returns the input DataFrame with a ``PEAD_FLAG`` column added (i32, 0 or 1).
    """
    if earnings_dates is None or reference_date is None:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("PEAD_FLAG"))

    if "ticker" not in df.columns:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("PEAD_FLAG"))

    from datetime import timedelta

    cutoff = reference_date - timedelta(days=30)

    # Build a lookup DataFrame: ticker -> PEAD_FLAG
    lookup_rows: list[dict] = []
    for ticker, dates in earnings_dates.items():
        has_recent = any(cutoff <= d <= reference_date for d in dates)
        lookup_rows.append({"ticker": ticker, "PEAD_FLAG": 1 if has_recent else 0})

    if not lookup_rows:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("PEAD_FLAG"))

    lookup_df = pl.DataFrame(lookup_rows)
    return df.join(lookup_df, on="ticker", how="left").with_columns(
        pl.col("PEAD_FLAG").fill_null(0)
    )


def compute_insider_buy(
    df: pl.DataFrame,
    *,
    insider_ratio: dict[str, float] | None = None,
) -> pl.DataFrame:
    """INSIDER_BUY = 1 if sum(Buy_Amount, 60d) / Market_Cap > 0.001.

    Requires pre-computed *insider_ratio* mapping ``ticker -> ratio`` where
    ratio = total_buy_amount_60d / market_cap.

    If *insider_ratio* is ``None`` or a ticker is not present, the flag is 0.

    Returns the input DataFrame with an ``INSIDER_BUY`` column added (i32, 0 or 1).
    """
    if insider_ratio is None or "ticker" not in df.columns:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("INSIDER_BUY"))

    # Build a lookup DataFrame: ticker -> INSIDER_BUY
    lookup_rows: list[dict] = []
    for ticker, ratio in (insider_ratio or {}).items():
        lookup_rows.append({"ticker": ticker, "INSIDER_BUY": 1 if ratio > 0.001 else 0})

    if not lookup_rows:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("INSIDER_BUY"))

    lookup_df = pl.DataFrame(lookup_rows)
    return df.join(lookup_df, on="ticker", how="left").with_columns(
        pl.col("INSIDER_BUY").fill_null(0)
    )


def compute_rev_accel(
    df: pl.DataFrame,
    *,
    revenue_growth: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """REV_ACCEL = Rev_Growth_Q - Rev_Growth_{Q-1}.

    Rev_Growth_Q = (Revenue_Q - Revenue_{Q-1}) / abs(Revenue_{Q-1}).

    Requires pre-computed *revenue_growth* mapping ``ticker -> [growth rates in
    chronological order, most recent last]``.  The acceleration is the difference
    between the last two growth rates.

    If *revenue_growth* is ``None``, the value is set to null (missing data) and
    a WARNING is emitted (Issue #225).  REV_ACCEL does **not** depend on SMA_200
    or other technical factors — it uses only fundamental revenue data.

    Returns the input DataFrame with a ``REV_ACCEL`` column added.
    """
    if revenue_growth is None:
        _logger.warning(
            "REV_ACCEL: revenue_growth data is None, factor will be null for all tickers"
        )
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("REV_ACCEL"))

    if "ticker" not in df.columns:
        _logger.warning("REV_ACCEL: df has no ticker column, factor will be null")
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("REV_ACCEL"))

    # Build a lookup DataFrame: ticker -> REV_ACCEL
    lookup_rows: list[dict] = []
    for ticker, growths in (revenue_growth or {}).items():
        if len(growths) >= 2:
            lookup_rows.append({"ticker": ticker, "REV_ACCEL": growths[-1] - growths[-2]})
        else:
            lookup_rows.append({"ticker": ticker, "REV_ACCEL": None})

    if not lookup_rows:
        _logger.warning(
            "REV_ACCEL: revenue_growth dict is empty, factor will be null for all tickers"
        )
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("REV_ACCEL"))

    lookup_df = pl.DataFrame(lookup_rows)
    return df.join(lookup_df, on="ticker", how="left")


# -- composite ---------------------------------------------------------------


def compute_all_technical_factors(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all 11 technical factors in one pass.

    This function chains all OHLCV-only factor computations, returning the input
    DataFrame with all factor columns appended.

    Minimum-history checks (Issue #225) are performed once upfront, avoiding
    repeated ``group_by("ticker")`` calls across the chained pipeline.

    Returns a DataFrame with columns: MOM_5D, PTH, MOM_SLOPE, BB_SQUEEZE,
    ATR_RATIO, MFI_14, CMF_21, VOL_ANOMALY, RSI_14, RSI_OVERSOLD, MACD, SIGNAL,
    HISTOGRAM, MACD_CROSS, SMA_50, SMA_200, GOLDEN_CROSS.
    """
    _check_min_history(
        df,
        ("PTH", 63),
        ("BB_SQUEEZE", 60),
        ("ATR_RATIO", 20),
        ("CMF_21", 21),
        ("VOL_ANOMALY", 50),
        ("SMA_50", 50),
        ("SMA_200", 200),
    )
    return (
        df.pipe(compute_mom_5d)
        .pipe(compute_pth)
        .pipe(compute_mom_slope)
        .pipe(compute_bb_squeeze)
        .pipe(compute_atr_ratio)
        .pipe(compute_mfi_14)
        .pipe(compute_cmf_21)
        .pipe(compute_vol_anomaly)
        .pipe(compute_rsi_oversold)
        .pipe(compute_macd_cross)
        .pipe(compute_golden_cross)
    )
