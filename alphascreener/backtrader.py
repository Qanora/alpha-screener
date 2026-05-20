"""Backtrader integration for strategy backtesting with SPY benchmark.

Issue #100: backtrader integration.
Reference: PRD 5.1 / 5.2.

Provides:
  - SevenDayBreakoutStrategy: 7-day holding, 8% stop loss, friction costs.
  - Data feed conversion from polars OHLCV + signals DataFrames.
  - Performance metrics computation.
  - daily_backtest_incremental / monthly_full_backtest run functions.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import backtrader as bt
import pandas as pd
import polars as pl

from alphascreener.data.io import scan_parquet
from alphascreener.logging import get_logger

_logger = get_logger("backtesting")

# ============================================================================
# Constants (PRD 5.1 / 5.2)
# ============================================================================

DEFAULT_HOLDING_DAYS: int = 7
DEFAULT_STOP_LOSS_PCT: float = 0.92
DEFAULT_COMMISSION_PCT: float = 0.001  # 0.1%
DEFAULT_SLIPPAGE_PCT: float = 0.002  # 0.2%
DEFAULT_MAX_POSITIONS: int = 20
DEFAULT_SIGNAL_THRESHOLD: float = 1.0
TRADING_DAYS_PER_YEAR: int = 252


# ============================================================================
# 1. SevenDayBreakoutStrategy
# ============================================================================


class SevenDayBreakoutStrategy(bt.Strategy):
    """7-day breakout strategy with stop loss, friction costs, and SPY benchmark.

    Entry (PRD 5.1):
      - T+1 open market buy when a signal fires at T.
      - entry_price = Open_{T+1}.

    Exit (PRD 5.1):
      1. T+7 close market (holding period expires).
      2. Intraday Low <= entry * stop_loss_pct (0.92 default) -> sell at close.
      3. Suspension / delisting (detected via missing data).

    Costs (PRD 5.2):
      - Commission: 0.1% per trade.
      - Slippage: 0.2% on entry and exit.

    Position sizing:
      - Equal allocation: cash / max_positions per trade.
      - Max 20 concurrent positions (default).
    """

    params = (
        ("holding_days", DEFAULT_HOLDING_DAYS),
        ("stop_loss_pct", DEFAULT_STOP_LOSS_PCT),
        ("commission_pct", DEFAULT_COMMISSION_PCT),
        ("slippage_pct", DEFAULT_SLIPPAGE_PCT),
        ("max_positions", DEFAULT_MAX_POSITIONS),
        ("signal_threshold", DEFAULT_SIGNAL_THRESHOLD),
    )

    def __init__(self) -> None:
        """Initialise tracking dictionaries for open positions."""
        # Map data index -> entry details dict
        self._entries: dict[int, dict[str, Any]] = {}
        self._bar_index: int = 0  # current bar index counter

        # State tracking for tests
        self.n_trades: int = 0
        self.stopped_out: bool = False
        self.max_positions_held: int = 0
        self.first_trade_value: float | None = None

    def next(self) -> None:
        """Called for every bar of every data feed."""
        self._bar_index += 1

        # Process each data feed
        for i, d in enumerate(self.datas):
            ctx = self._entries.get(i)

            # --- Check stop loss on existing position ---
            if ctx is not None:
                entry_price = ctx["entry_price"]
                stop_price = entry_price * self.p.stop_loss_pct

                # Check stop loss: Low <= entry * stop_loss_pct
                if d.low[0] <= stop_price:
                    self._close_position(i, d, reason="stop_loss")
                    self.stopped_out = True
                    continue

                # Check holding period expiry
                bars_held = self._bar_index - ctx["bar_entered"]
                if bars_held >= self.p.holding_days:
                    self._close_position(i, d, reason="holding_expiry")
                    continue

                # Check suspension / delisting (price stuck at 0 or NaN)
                if d.close[0] <= 0 or math.isnan(d.close[0]):
                    self._close_position(i, d, reason="delisting")
                    continue

            # --- Check for new entry signal ---
            if ctx is None:
                # Signal fires when signal line > 0 at current bar
                signal_val = 0.0
                if hasattr(d.lines, "signal"):
                    try:
                        signal_val = d.lines.signal[0]
                    except (IndexError, AttributeError):
                        signal_val = 0.0

                if signal_val > 0:
                    # Check position limit
                    current_positions = len(self._entries)
                    if current_positions >= self.p.max_positions:
                        continue

                    # Enter at the NEXT bar's open (T+1)
                    # We use current bar's close to approximate next open
                    # because backtrader evaluates bar-by-bar
                    entry_price_t0 = d.close[0]

                    size = self._calculate_size(entry_price_t0)
                    if size > 0:
                        # Apply slippage to entry price
                        slip_entry = entry_price_t0 * (1.0 + self.p.slippage_pct)
                        self.buy(data=d, size=size, price=slip_entry)
                        self._entries[i] = {
                            "entry_price": entry_price_t0,
                            "bar_entered": self._bar_index,
                            "size": size,
                        }
                        self.n_trades += 1
                        if self.first_trade_value is None:
                            self.first_trade_value = size * entry_price_t0

    def _close_position(self, idx: int, data, reason: str) -> None:
        """Close an open position with slippage applied."""
        ctx = self._entries.pop(idx, None)
        if ctx is None:
            return

        close_price = data.close[0]
        # Apply slippage: sell at slightly worse price
        slip_close = close_price * (1.0 - self.p.slippage_pct)
        self.sell(data=data, size=ctx["size"], price=slip_close)
        _logger.debug("Closed position: reason=%s entry=%0.2f exit=%0.2f",
                       reason, ctx["entry_price"], close_price)

    def _calculate_size(self, price: float) -> int:
        """Calculate position size for equal allocation.

        size = (cash / max_positions) / price, floored to integer shares.
        """
        if price <= 0:
            return 0
        if not hasattr(self, "_initial_cash"):
            self._initial_cash = self.broker.getvalue()
        cash_per_position = self._initial_cash / self.p.max_positions
        return max(1, int(cash_per_position / price))

    def notify_trade(self, trade) -> None:
        """Track max concurrent positions for test assertions."""
        current = len(self._entries)
        if current > self.max_positions_held:
            self.max_positions_held = current


# ============================================================================
# 2. Data feed conversion
# ============================================================================


class _SignalPandasData(bt.feeds.PandasData):
    """Extended PandasData with a 'signal' line for trade entry signals."""

    lines = ("signal",)
    params = (
        ("signal", "signal"),
        ("datetime", "datetime"),
    )


_OBS_COLS: frozenset[str] = frozenset({"open", "high", "low", "close", "volume"})


def create_backtest_data_feed(
    df: pl.DataFrame,
    *,
    signals: pl.DataFrame | None = None,
) -> bt.feeds.PandasData:
    """Convert a polars OHLCV DataFrame to a backtrader data feed.

    The output DataFrame has columns: datetime, open, high, low, close,
    volume, signal.  The *signal* column is 1 on dates where a buy signal
    fires, 0 otherwise.

    Args:
        df: OHLCV DataFrame with columns ``ticker, dt, open, high, low,
            close, volume``.
        signals: Optional signals DataFrame with columns ``ticker, dt,
            refined_score``. Rows with ``refined_score >=
            DEFAULT_SIGNAL_THRESHOLD`` produce signal=1.

    Returns:
        A backtrader PandasData subclass instance ready for ``cerebro.adddata()``.

    Raises:
        ValueError: If *df* is empty or missing required OHLCV columns.
    """
    if df.height == 0:
        raise ValueError("DataFrame is empty; cannot create backtest feed")
    missing = _OBS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    # Build pandas DataFrame with datetime index and OHLCV columns
    pdf = df.select(["dt", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"dt": "datetime"})
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf = pdf.sort_values("datetime").reset_index(drop=True)

    # Add signal column
    pdf["signal"] = 0.0
    if signals is not None and signals.height > 0:
        signal_dates = _extract_signal_dates(signals, df["ticker"][0])
        pdf.loc[pdf["datetime"].isin(signal_dates), "signal"] = 1.0

    return _SignalPandasData(dataname=pdf)


def _extract_signal_dates(signals: pl.DataFrame, ticker: str) -> set[pd.Timestamp]:
    """Extract dates where the given ticker has a buy signal.

    A signal fires when refined_score >= DEFAULT_SIGNAL_THRESHOLD.
    """
    sig = signals.filter(
        (pl.col("ticker") == ticker) & (pl.col("refined_score") >= DEFAULT_SIGNAL_THRESHOLD)
    )
    if sig.height == 0:
        return set()
    dates: list[date] = sig.select("dt").to_series().to_list()
    return {pd.Timestamp(d) for d in dates}


def create_backtest_data_feeds(
    dfs: dict[str, pl.DataFrame],
    *,
    signals: pl.DataFrame | None = None,
) -> list[bt.feeds.PandasData]:
    """Convert multiple ticker OHLCV DataFrames to backtrader data feeds.

    Args:
        dfs: Mapping of ticker -> OHLCV DataFrame.
        signals: Optional signals DataFrame.

    Returns:
        List of backtrader data feeds (one per ticker).
    """
    feeds: list[bt.feeds.PandasData] = []
    for ticker, df in dfs.items():
        try:
            feed = create_backtest_data_feed(df, signals=signals)
            feeds.append(feed)
        except ValueError as e:
            _logger.warning("Skipping %s: %s", ticker, e)
    return feeds


# ============================================================================
# 3. Performance metrics
# ============================================================================


def compute_backtest_metrics(
    strategy_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute performance metrics from a daily returns series.

    Args:
        strategy_returns: Daily percentage returns (e.g. 0.01 = 1%).
        benchmark_returns: Optional benchmark daily returns for comparison.
        trading_days_per_year: Annualisation factor.

    Returns:
        Dict with keys: total_return, annualized_return, sharpe_ratio,
        max_drawdown, win_rate, volatility, and optionally excess_return,
        information_ratio, benchmark_total_return.
    """
    if len(strategy_returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
        }

    # Total return (compounded)
    total_return = (1.0 + strategy_returns).prod() - 1.0

    # Annualized return
    n_days = len(strategy_returns)
    annualized_return = (1.0 + total_return) ** (trading_days_per_year / max(n_days, 1)) - 1.0

    # Volatility (annualized)
    daily_vol = strategy_returns.std()
    volatility = daily_vol * math.sqrt(trading_days_per_year)

    # Sharpe ratio (assume 0 risk-free rate for simplicity)
    if daily_vol > 0 and n_days > 1:
        sharpe_ratio = (strategy_returns.mean() / daily_vol) * math.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = (1.0 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Win rate
    winning_days = (strategy_returns > 0).sum()
    win_rate = winning_days / max(n_days, 1)

    metrics: dict[str, float] = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "volatility": float(volatility),
    }

    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_total = (1.0 + benchmark_returns).prod() - 1.0
        excess_return = total_return - bench_total
        aligned = pd.DataFrame(
            {
                "strategy": strategy_returns,
                "benchmark": benchmark_returns,
            }
        ).dropna()
        if len(aligned) > 1:
            tracking = aligned["strategy"] - aligned["benchmark"]
            tracking_vol = tracking.std()
            info_ratio = (
                tracking.mean() / tracking_vol * math.sqrt(trading_days_per_year)
                if tracking_vol > 0
                else 0.0
            )
        else:
            info_ratio = 0.0

        metrics["excess_return"] = float(excess_return)
        metrics["information_ratio"] = float(info_ratio)
        metrics["benchmark_total_return"] = float(bench_total)

    return metrics


# ============================================================================
# 4. Full backtest run
# ============================================================================


def run_backtest(
    ticker_dfs: dict[str, pl.DataFrame],
    *,
    signals: pl.DataFrame | None = None,
    spy_data: pl.DataFrame | None = None,
    initial_cash: float = 1_000_000.0,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> dict[str, Any]:
    """Run a full backtrader backtest with the SevenDayBreakoutStrategy.

    Args:
        ticker_dfs: Mapping of ticker -> OHLCV DataFrame.
        signals: Signals DataFrame with buy signals.
        spy_data: Optional SPY OHLCV DataFrame for benchmark comparison.
        initial_cash: Starting capital.
        holding_days: Number of trading days to hold.
        stop_loss_pct: Stop loss percentage (0.92 = 8% loss).
        commission_pct: Commission per trade (0.001 = 0.1%).
        slippage_pct: Slippage per trade (0.002 = 0.2%).
        max_positions: Maximum concurrent positions.

    Returns:
        Dict with keys: metrics, n_trades, final_value, strategy_returns,
        benchmark_returns (if spy_data provided).
    """
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(
        SevenDayBreakoutStrategy,
        holding_days=holding_days,
        stop_loss_pct=stop_loss_pct,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        max_positions=max_positions,
    )

    # Add data feeds
    feeds = create_backtest_data_feeds(ticker_dfs, signals=signals)
    if not feeds:
        _logger.warning("No valid data feeds; returning empty result")
        return {
            "metrics": compute_backtest_metrics(pd.Series(dtype=float)),
            "n_trades": 0,
            "final_value": initial_cash,
            "strategy_returns": pd.Series(dtype=float),
        }

    for feed in feeds:
        cerebro.adddata(feed)

    # Set up broker
    cerebro.broker.setcash(initial_cash)
    # Commission as a percentage of trade value
    cerebro.broker.setcommission(commission=commission_pct)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns", timeframe=bt.TimeFrame.Days)

    # Run
    _logger.info("Starting backtest with %d feeds, cash=%0.0f", len(feeds), initial_cash)
    results = cerebro.run()
    strat = results[0]

    # Extract returns
    ret_analyzer = strat.analyzers.returns
    rets = ret_analyzer.get_analysis() if ret_analyzer else {}
    strategy_returns = pd.Series(rets) if rets else pd.Series(dtype=float)

    # Compute SPY benchmark returns
    benchmark_returns = None
    if spy_data is not None and spy_data.height > 0:
        try:
            spy_feed = create_backtest_data_feed(spy_data)
            spy_cerebro = bt.Cerebro()
            spy_cerebro.adddata(spy_feed)
            spy_cerebro.addanalyzer(
                bt.analyzers.TimeReturn, _name="returns", timeframe=bt.TimeFrame.Days
            )
            spy_cerebro.addstrategy(_BuyAndHoldStrategy)
            spy_cerebro.broker.setcash(initial_cash)
            spy_cerebro.broker.setcommission(commission=0.0)
            spy_results = spy_cerebro.run()
            spy_ret = spy_results[0].analyzers.returns
            spy_rets = spy_ret.get_analysis() if spy_ret else {}
            benchmark_returns = pd.Series(spy_rets) if spy_rets else pd.Series(dtype=float)
        except (ValueError, KeyError, IndexError) as e:
            _logger.warning("SPY benchmark computation failed: %s", e)

    metrics = compute_backtest_metrics(strategy_returns, benchmark_returns=benchmark_returns)

    final_value = cerebro.broker.getvalue()

    result: dict[str, Any] = {
        "metrics": metrics,
        "n_trades": strat.n_trades,
        "final_value": final_value,
        "strategy_returns": strategy_returns,
    }
    if benchmark_returns is not None:
        result["benchmark_returns"] = benchmark_returns

    _logger.info(
        "Backtest complete: total_return=%.2f%%, sharpe=%.2f, n_trades=%d",
        metrics["total_return"] * 100,
        metrics["sharpe_ratio"],
        strat.n_trades,
    )
    return result


class _BuyAndHoldStrategy(bt.Strategy):
    """Simple buy-and-hold strategy for benchmark computation."""

    def __init__(self) -> None:
        self._bought = False

    def next(self) -> None:
        if not self._bought:
            for d in self.datas:
                size = int(self.broker.getcash() / d.close[0])
                if size > 0:
                    self.buy(data=d, size=size)
            self._bought = True


# ============================================================================
# 5. Incremental and monthly backtest run functions
# ============================================================================


def _load_ohlcv_data(
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, pl.DataFrame]:
    """Load OHLCV data from the Parquet store, grouped by ticker.

    Args:
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        Mapping of ticker -> OHLCV DataFrame.
    """
    try:
        lf = scan_parquet("ohlcv")
    except FileNotFoundError:
        _logger.info("No OHLCV data found in store")
        return {}

    if start_date is not None:
        lf = lf.filter(pl.col("dt") >= start_date)
    if end_date is not None:
        lf = lf.filter(pl.col("dt") <= end_date)

    df = lf.collect()
    if df.height == 0:
        return {}

    ticker_map: dict[str, pl.DataFrame] = {}
    for ticker in df["ticker"].unique().to_list():
        ticker_map[ticker] = df.filter(pl.col("ticker") == ticker).sort("dt")
    return ticker_map


def _load_signals_data(
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame | None:
    """Load signals data from the Parquet store.

    Loads LLM-track signals (B-track) by default.
    """
    try:
        # Load all signals; filter for LLM track
        lf = scan_parquet("signals")
    except FileNotFoundError:
        _logger.info("No signals data found in store")
        return None

    if start_date is not None:
        lf = lf.filter(pl.col("dt") >= start_date)
    if end_date is not None:
        lf = lf.filter(pl.col("dt") <= end_date)

    df = lf.collect()
    if df.height == 0:
        return None

    # Use refined_score if available, otherwise fall back to any score column
    if "refined_score" not in df.columns and "refined_score_pure" in df.columns:
        df = df.rename({"refined_score_pure": "refined_score"})

    required = {"ticker", "dt", "refined_score"}
    if not required.issubset(set(df.columns)):
        _logger.warning("Signals data missing required columns: %s", required - set(df.columns))
        return None

    return df.select(list(required))


def daily_backtest_incremental(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run incremental daily backtest on recent data (target: <= 30 min).

    Loads OHLCV and signals data from the Parquet store and runs
    a full backtrader backtest with SevenDayBreakoutStrategy.

    Args:
        start_date: Optional start date for data range.
        end_date: Optional end date for data range.
        **kwargs: Passed through to run_backtest().

    Returns:
        Summary dict with keys: status, metrics, n_trades, etc.
    """
    _logger.info("Starting daily_backtest_incremental (%s .. %s)", start_date, end_date)

    try:
        ticker_dfs = _load_ohlcv_data(start_date=start_date, end_date=end_date)
        if not ticker_dfs:
            return {"status": "no_data", "message": "No OHLCV data available"}

        signals = _load_signals_data(start_date=start_date, end_date=end_date)

        # Load SPY for benchmark from already-loaded data
        spy_data = ticker_dfs.get("SPY")

        result = run_backtest(ticker_dfs, signals=signals, spy_data=spy_data, **kwargs)

        return {
            "status": "ok",
            "metrics": result["metrics"],
            "n_trades": result["n_trades"],
            "final_value": result["final_value"],
        }
    except Exception as e:
        _logger.exception("daily_backtest_incremental failed: %s", e)
        return {"status": "error", "message": str(e)}


def monthly_full_backtest(
    *,
    year: int | None = None,
    month: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run full monthly backtest on an entire month of data (target: <= 4 hours).

    Args:
        year: Year for the monthly backtest.
        month: Month (1-12) for the backtest.
        **kwargs: Passed through to run_backtest().

    Returns:
        Summary dict with keys: status, metrics, n_trades, etc.
    """
    _logger.info("Starting monthly_full_backtest (%s-%s)", year, month)

    try:
        # Determine date range for the given month
        if year is not None and month is not None:
            start_date = date(year, month, 1)
            # Last day of month
            if month == 12:
                end_date = date(year + 1, 1, 1)
            else:
                end_date = date(year, month + 1, 1)
            # end_date is exclusive in our filter, so set to first of next month
        else:
            # Default: use all available data
            start_date = None
            end_date = None

        # For monthly full backtest, also load data from 2 months before
        # to have enough history for signal generation context
        lookback_start = None
        if start_date is not None:
            lb_month = start_date.month - 2
            lb_year = start_date.year
            if lb_month <= 0:
                lb_month += 12
                lb_year -= 1
            lookback_start = date(lb_year, lb_month, 1)

        ticker_dfs = _load_ohlcv_data(start_date=lookback_start, end_date=end_date)
        if not ticker_dfs:
            return {"status": "no_data", "message": "No OHLCV data available"}

        signals = _load_signals_data(start_date=lookback_start, end_date=end_date)

        # Load SPY for benchmark
        spy_data = ticker_dfs.get("SPY")

        result = run_backtest(ticker_dfs, signals=signals, spy_data=spy_data, **kwargs)

        return {
            "status": "ok",
            "metrics": result["metrics"],
            "n_trades": result["n_trades"],
            "final_value": result["final_value"],
            "period": {
                "year": year,
                "month": month,
            },
        }
    except Exception as e:
        _logger.exception("monthly_full_backtest failed: %s", e)
        return {"status": "error", "message": str(e)}
