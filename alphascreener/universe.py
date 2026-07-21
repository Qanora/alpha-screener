"""Point-in-time, tradable US-equity universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from alphascreener.prediction_contract import INPUT_LOOKBACK_SESSIONS


@dataclass(frozen=True)
class UniverseRules:
    """Minimum requirements for a candidate that can be ranked."""

    min_close: float = 5.0
    min_average_dollar_volume: float = 5_000_000.0
    dollar_volume_sessions: int = 20
    required_sessions: int = INPUT_LOOKBACK_SESSIONS


def build_universe_snapshot(
    ohlcv: pl.DataFrame,
    *,
    cutoff_date: date | None = None,
    rules: UniverseRules = UniverseRules(),
) -> pl.DataFrame:
    """Classify every observed ticker as eligible or explain its exclusion.

    Only observations at or before ``cutoff_date`` are read.  An eligible
    ticker must have data on the cutoff, at least 60 trading sessions, a
    minimum closing price, and sufficient recent average dollar volume.
    """
    required = {"ticker", "dt", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    if ohlcv.is_empty():
        return _empty_snapshot()

    data = ohlcv.with_columns(pl.col("dt").cast(pl.Date))
    if cutoff_date is None:
        cutoff_date = data["dt"].max()
    data = data.filter(pl.col("dt") <= cutoff_date).sort(["ticker", "dt"])

    rows: list[dict[str, object]] = []
    for (ticker,), ticker_data in data.group_by("ticker", maintain_order=True):
        ticker_data = ticker_data.sort("dt")
        sessions = ticker_data.height
        last_date = ticker_data["dt"].max()
        last_close = float(ticker_data["close"].tail(1)[0])
        volume_window = ticker_data.tail(rules.dollar_volume_sessions)
        average_dollar_volume = float(
            (volume_window["close"] * volume_window["volume"]).mean()
        )
        lookback = ticker_data.tail(rules.required_sessions)
        invalid_observations = lookback.filter(
            pl.col("close").is_null()
            | ~pl.col("close").cast(pl.Float64).is_finite()
            | (pl.col("close") <= 0)
            | pl.col("volume").is_null()
            | ~pl.col("volume").cast(pl.Float64).is_finite()
            | (pl.col("volume") < 0)
        ).height

        exclusion_reason: str | None = None
        if sessions < rules.required_sessions:
            exclusion_reason = "insufficient_history"
        elif last_date != cutoff_date:
            exclusion_reason = "stale_data"
        elif invalid_observations:
            exclusion_reason = "invalid_data"
        elif last_close < rules.min_close:
            exclusion_reason = "low_price"
        elif average_dollar_volume < rules.min_average_dollar_volume:
            exclusion_reason = "low_dollar_volume"

        rows.append(
            {
                "ticker": ticker,
                "cutoff_date": cutoff_date,
                "history_sessions": sessions,
                "last_close": last_close,
                "average_dollar_volume": average_dollar_volume,
                "eligible": exclusion_reason is None,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pl.DataFrame(rows, schema=_snapshot_schema()).sort("ticker")


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=_snapshot_schema())


def _snapshot_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "cutoff_date": pl.Date,
        "history_sessions": pl.Int64,
        "last_close": pl.Float64,
        "average_dollar_volume": pl.Float64,
        "eligible": pl.Boolean,
        "exclusion_reason": pl.String,
    }
