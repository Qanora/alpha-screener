"""Ablation dual-track recording with 60-day rolling Delta-Lift@20 monitoring.

Issue #99: Ablation dual-track recording.
Reference: PRD 4.7.

Tracks:
  - A-track (pure factor): Refined_Score_pure = Coarse_Final_Score
    -> signals_refined_pure under the "signals" category.
  - B-track (LLM-corrected): Refined_Score = Coarse_Final_Score
    x score_correction x risk_filter
    -> signals_refined under the "signals" category.

Delta-Lift@20 is computed over a 60-trading-day rolling window from
outcomes keyed by (ticker, dt).  The ablation decision is derived from
the latest rolling Delta-Lift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import polars as pl

from alphascreener.config import Settings
from alphascreener.data.io import get_data_dir
from alphascreener.logging import get_logger

_logger = get_logger("screening")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PRD 4.7: 60-trading-day rolling window (~ 84 calendar days)
DEFAULT_WINDOW_TRADING_DAYS: int = 60
DEFAULT_WINDOW_CALENDAR_DAYS: int = 84

# Lift@K configuration
DEFAULT_K: int = 20

# Ablation decision thresholds (PRD 4.7 table)
ABLATION_PASS_THRESHOLD: float = 0.05  # Delta-Lift >= 0.05 -> PASS
# [0, 0.05) -> BORDERLINE; < 0 -> FAIL

# Filename prefixes for distinguishing tracks in the signals partition
FILENAME_PREFIX_PURE: str = "pure_"
FILENAME_PREFIX_LLM: str = "llm_"

# Every record written to the signals store MUST include at least these columns.
_SIGNALS_COMMON_COLS: tuple[str, ...] = ("ticker", "dt")


# ---------------------------------------------------------------------------
# Ablation decision enum
# ---------------------------------------------------------------------------


class AblationDecision(Enum):
    """Decision derived from 60-day rolling Delta-Lift@20 (PRD 4.7)."""

    PASS = "pass"  # Delta-Lift >= 0.05: LLM correction layer enabled
    BORDERLINE = "borderline"  # Delta-Lift in [0, 0.05): LLM for explainability only
    FAIL = "fail"  # Delta-Lift < 0: disable LLM correction, trigger PRD review


# ---------------------------------------------------------------------------
# Risk filter (PRD 4.4)
# ---------------------------------------------------------------------------

# Tags whose mere presence zeroes out the risk_filter multiplier.
HARD_BLOCK_TAGS: frozenset[str] = frozenset({"delisting_risk"})


def compute_risk_filter(
    risk_tags: list[str] | None,
    data_conflict_detected: bool = False,
) -> float:
    """Return the risk_filter multiplier per PRD 4.4.

    Returns 0.0 when any hard-block risk tag is present **or**
    ``data_conflict_detected`` is True.  Returns 1.0 otherwise.

    Args:
        risk_tags: Risk tags from the PM audit.
        data_conflict_detected: Whether contradictory signals were found.

    Returns:
        0.0 or 1.0.
    """
    if data_conflict_detected:
        return 0.0
    if risk_tags and any(t in HARD_BLOCK_TAGS for t in risk_tags):
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Refined scores (PRD 4.4)
# ---------------------------------------------------------------------------


def compute_refined_score_pure(coarse_final_score: float) -> float:
    """A-track: pure-factor score (no LLM correction)."""
    return coarse_final_score


def compute_refined_score_llm(
    coarse_final_score: float,
    score_correction: float,
    risk_filter: float,
) -> float:
    """B-track: LLM-corrected score = Coarse_Final_Score x score_correction x risk_filter."""
    return coarse_final_score * score_correction * risk_filter


# ---------------------------------------------------------------------------
# Ablation record (in-memory representation)
# ---------------------------------------------------------------------------


@dataclass
class AblationEntry:
    """A single observation for ablation dual-track recording.

    Holds both the pure (A-track) and LLM-corrected (B-track) scores
    for one symbol on one date.
    """

    ticker: str
    dt: date
    coarse_final_score: float
    score_correction: float = 1.0
    risk_filter: float = 1.0
    risk_tags: list[str] = field(default_factory=list)
    data_conflict_detected: bool = False
    phase1_pass: bool = True

    # Computed on init
    refined_score_pure: float = field(init=False)
    refined_score_llm: float = field(init=False)

    def __post_init__(self) -> None:
        self.refined_score_pure = compute_refined_score_pure(self.coarse_final_score)
        self.refined_score_llm = compute_refined_score_llm(
            self.coarse_final_score, self.score_correction, self.risk_filter
        )

    def to_dict_pure(self) -> dict[str, Any]:
        """Column set for A-track (signals_refined_pure)."""
        return {
            "ticker": self.ticker,
            "dt": self.dt,
            "coarse_final_score": self.coarse_final_score,
            "refined_score_pure": self.refined_score_pure,
            "phase1_pass": self.phase1_pass,
        }

    def to_dict_llm(self) -> dict[str, Any]:
        """Column set for B-track (signals_refined)."""
        return {
            "ticker": self.ticker,
            "dt": self.dt,
            "coarse_final_score": self.coarse_final_score,
            "score_correction": self.score_correction,
            "risk_filter": self.risk_filter,
            "refined_score": self.refined_score_llm,
            "risk_tags": self.risk_tags,
            "data_conflict_detected": self.data_conflict_detected,
            "phase1_pass": self.phase1_pass,
        }

    @classmethod
    def from_assessment(
        cls,
        ticker: str,
        dt: date,
        coarse_final_score: float,
        score_correction: float,
        risk_tags: list[str],
        data_conflict_detected: bool,
        phase1_pass: bool = True,
    ) -> AblationEntry:
        """Build an entry from pipeline outputs.

        Args:
            ticker: Symbol.
            dt: Observation date.
            coarse_final_score: Coarse_Final_Score from the factor engine.
            score_correction: PM score_correction from BreakoutAssessment.
            risk_tags: Risk tags from PM audit.
            data_conflict_detected: From BreakoutAssessment.
            phase1_pass: Whether Phase 1 hard filters were passed.
        """
        rf = compute_risk_filter(risk_tags, data_conflict_detected)
        return cls(
            ticker=ticker,
            dt=dt,
            coarse_final_score=coarse_final_score,
            score_correction=score_correction,
            risk_filter=rf,
            risk_tags=list(risk_tags or []),
            data_conflict_detected=data_conflict_detected,
            phase1_pass=phase1_pass,
        )


# ---------------------------------------------------------------------------
# Lift@K computation
# ---------------------------------------------------------------------------


def _top_k_scores(
    df: pl.DataFrame, score_col: str, k: int,
) -> pl.DataFrame:
    """Return the top-k rows by *score_col* (descending, nulls last)."""
    if df.height == 0:
        return df
    return (
        df.sort(score_col, descending=True, nulls_last=True)
        .head(k)
    )


def compute_precision_at_k(
    scores_df: pl.DataFrame,
    k: int,
    *,
    score_col: str,
    outcome_col: str,
) -> float:
    """Precision@K: fraction of top-K by score whose outcome is True / 1.

    Args:
        scores_df: DataFrame with at least *score_col* and *outcome_col*.
        k: Number of top entries to consider.
        score_col: Column name for ranking.
        outcome_col: Boolean (or 0/1) column indicating a hit.

    Returns:
        Precision in [0, 1].  Returns ``math.nan`` when there are fewer
        than *k* rows with a non-null score.
    """
    if scores_df.height < k:
        return math.nan

    valid = scores_df.filter(pl.col(score_col).is_not_null())
    if valid.height < k:
        return math.nan

    top = _top_k_scores(valid, score_col, k)
    hits = top.select(pl.col(outcome_col).sum()).item()
    return float(hits) / float(k)


def compute_base_rate(
    scores_df: pl.DataFrame,
    *,
    outcome_col: str,
) -> float:
    """Base rate: proportion of all tickers with a positive outcome.

    Args:
        scores_df: DataFrame with *outcome_col*.
        outcome_col: Boolean / 0-1 column.

    Returns:
        Base rate in [0, 1]. Returns ``math.nan`` when there are no rows.
    """
    n = scores_df.height
    if n == 0:
        return math.nan
    hits = scores_df.select(pl.col(outcome_col).sum()).item()
    return float(hits) / float(n)


def compute_lift_at_k(
    scores_df: pl.DataFrame,
    k: int,
    *,
    score_col: str,
    outcome_col: str,
) -> float:
    """Lift@K = Precision@K / base_rate.

    Args:
        scores_df: DataFrame with *score_col* and *outcome_col*.
        k: Top-K size.
        score_col: Column for ranking.
        outcome_col: Boolean / 0-1 outcome column.

    Returns:
        Lift@K. Returns ``math.nan`` when not computable
        (too few rows or base_rate == 0).
    """
    precision = compute_precision_at_k(scores_df, k, score_col=score_col, outcome_col=outcome_col)
    if math.isnan(precision):
        return math.nan
    base = compute_base_rate(scores_df, outcome_col=outcome_col)
    if math.isnan(base) or base == 0.0:
        return math.nan
    return precision / base


def compute_delta_lift(
    pure_df: pl.DataFrame,
    llm_df: pl.DataFrame,
    k: int,
    *,
    score_col_pure: str,
    score_col_llm: str,
    outcome_col: str,
) -> float:
    """Delta-Lift@K = Lift@K(B-track) - Lift@K(A-track).

    Both DataFrames are expected to cover the same universe and share
    the same *outcome_col*.

    Args:
        pure_df: DataFrame with A-track scores (must contain *outcome_col*).
        llm_df: DataFrame with B-track scores (must contain *outcome_col*).
        k: Top-K size.
        score_col_pure: Column with pure-factor scores.
        score_col_llm: Column with LLM-corrected scores.
        outcome_col: Shared outcome column.

    Returns:
        Delta-Lift.  Returns ``math.nan`` when either track's Lift
        cannot be computed.
    """
    lift_pure = compute_lift_at_k(pure_df, k, score_col=score_col_pure, outcome_col=outcome_col)
    if math.isnan(lift_pure):
        _logger.warning("A-track Lift@%d is NaN", k)
        return math.nan

    lift_llm = compute_lift_at_k(llm_df, k, score_col=score_col_llm, outcome_col=outcome_col)
    if math.isnan(lift_llm):
        _logger.warning("B-track Lift@%d is NaN", k)
        return math.nan

    return lift_llm - lift_pure


def compute_ablation_decision(delta_lift: float) -> AblationDecision:
    """Map Delta-Lift@20 to an AblationDecision per PRD 4.7.

    Args:
        delta_lift: Delta-Lift@20 value.

    Returns:
        PASS (>= 0.05), BORDERLINE ([0, 0.05)), or FAIL (< 0).
    """
    if delta_lift >= ABLATION_PASS_THRESHOLD:
        return AblationDecision.PASS
    if delta_lift >= 0.0:
        return AblationDecision.BORDERLINE
    return AblationDecision.FAIL


# ---------------------------------------------------------------------------
# AblationTracker
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Configuration for the AblationTracker."""

    window_trading_days: int = DEFAULT_WINDOW_TRADING_DAYS
    window_calendar_days: int = DEFAULT_WINDOW_CALENDAR_DAYS
    k: int = DEFAULT_K
    enabled: bool = True  # driven by LLM_ABLATION_ENABLED env var


class AblationTracker:
    """Dual-track recorder with 60-day rolling Delta-Lift@20 monitoring.

    Usage::

        tracker = AblationTracker()
        entry = AblationEntry.from_assessment(...)
        tracker.record(entry)
        tracker.flush()          # persist to parquet
        delta = tracker.delta_lift(outcomes_df)
        decision = tracker.decide(delta)
    """

    def __init__(self, config: AblationConfig | None = None) -> None:
        self._config = config or AblationConfig()
        # In-memory buffers keyed by (ticker, dt)
        self._pure_records: list[dict[str, Any]] = []
        self._llm_records: list[dict[str, Any]] = []
        self._ticker_date_keys: set[tuple[str, date]] = set()
        self._last_delta_lift: float | None = None

    # --- properties -------------------------------------------------------

    @property
    def n_records(self) -> int:
        return len(self._pure_records)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # --- record -----------------------------------------------------------

    def record(self, entry: AblationEntry) -> None:
        """Record one assessment into both tracks.

        If LLM_ABLATION_ENABLED is False the call is a no-op.

        Duplicate (ticker, dt) entries are silently skipped.
        """
        if not self._config.enabled:
            return

        key = (entry.ticker, entry.dt)
        if key in self._ticker_date_keys:
            _logger.debug("Skipping duplicate ablation entry: %s %s", entry.ticker, entry.dt)
            return

        self._pure_records.append(entry.to_dict_pure())
        self._llm_records.append(entry.to_dict_llm())
        self._ticker_date_keys.add(key)

    def record_batch(self, entries: list[AblationEntry]) -> None:
        """Record a batch of entries."""
        for entry in entries:
            self.record(entry)

    # --- to DataFrames ----------------------------------------------------

    def pure_df(self) -> pl.DataFrame:
        """Return the accumulated A-track records as a DataFrame."""
        if not self._pure_records:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8, "dt": pl.Date,
                    "coarse_final_score": pl.Float64,
                    "refined_score_pure": pl.Float64,
                    "phase1_pass": pl.Boolean,
                }
            )
        return pl.DataFrame(self._pure_records)

    def llm_df(self) -> pl.DataFrame:
        """Return the accumulated B-track records as a DataFrame."""
        if not self._llm_records:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8, "dt": pl.Date,
                    "coarse_final_score": pl.Float64,
                    "score_correction": pl.Float64,
                    "risk_filter": pl.Float64,
                    "refined_score": pl.Float64,
                    "risk_tags": pl.List(pl.Utf8),
                    "data_conflict_detected": pl.Boolean,
                    "phase1_pass": pl.Boolean,
                }
            )
        return pl.DataFrame(self._llm_records)

    # --- delta-lift -------------------------------------------------------

    def delta_lift(
        self,
        outcomes_df: pl.DataFrame,
        *,
        k: int | None = None,
    ) -> float:
        """Compute Delta-Lift@K from the tracker's current records.

        Args:
            outcomes_df: DataFrame with columns ``ticker, dt`` and an
                outcome column (``hit`` = 1 if T+7 return >= 10% else 0).
            k: Top-K (default: config.k).

        Returns:
            Delta-Lift.  ``math.nan`` when not computable.
        """
        k = k or self._config.k
        pure = self.pure_df()
        llm = self.llm_df()

        if pure.height == 0 or llm.height == 0:
            return math.nan

        # Merge outcomes into each track on (ticker, dt)
        if "hit" not in outcomes_df.columns:
            _logger.warning("outcomes_df missing 'hit' column")
            return math.nan

        pure_with_outcomes = pure.join(
            outcomes_df.select(["ticker", "dt", "hit"]),
            on=["ticker", "dt"], how="inner",
        )
        llm_with_outcomes = llm.join(
            outcomes_df.select(["ticker", "dt", "hit"]),
            on=["ticker", "dt"], how="inner",
        )

        delta = compute_delta_lift(
            pure_with_outcomes, llm_with_outcomes, k,
            score_col_pure="refined_score_pure",
            score_col_llm="refined_score",
            outcome_col="hit",
        )
        self._last_delta_lift = delta
        return delta

    def delta_lift_from_aligned(
        self,
        aligned_df: pl.DataFrame,
        *,
        k: int | None = None,
    ) -> float:
        """Compute Delta-Lift from a pre-joined DataFrame that already
        contains both score columns and the outcome column.

        Args:
            aligned_df: DataFrame with columns ``refined_score_pure``,
                ``refined_score``, and ``hit``.
            k: Top-K.

        Returns:
            Delta-Lift.
        """
        k = k or self._config.k
        delta = compute_delta_lift(
            aligned_df, aligned_df, k,
            score_col_pure="refined_score_pure",
            score_col_llm="refined_score",
            outcome_col="hit",
        )
        self._last_delta_lift = delta
        return delta

    # --- decide -----------------------------------------------------------

    def decide(
        self,
        delta_lift: float | None = None,
    ) -> AblationDecision | None:
        """Derive an ablation decision from a Delta-Lift value.

        Args:
            delta_lift: If None, uses the last cached delta_lift value.

        Returns:
            AblationDecision, or None if no delta_lift is available.
        """
        dl = delta_lift if delta_lift is not None else self._last_delta_lift
        if dl is None or math.isnan(dl):
            return None
        return compute_ablation_decision(dl)

    # --- persist ----------------------------------------------------------

    def flush(self) -> None:
        """Persist A-track and B-track records as Parquet under the signals
        category.  Each track is written into the same partition directory
        with a distinguishable filename prefix so that downstream readers
        can separate them.

        If there are no records, this is a no-op.
        """
        if not self._pure_records:
            return

        pure_df = self.pure_df()
        llm_df = self.llm_df()
        self._write_track(pure_df, prefix=FILENAME_PREFIX_PURE)
        self._write_track(llm_df, prefix=FILENAME_PREFIX_LLM)
        _logger.info(
            "Flushed %d ablation records (pure + llm) to signals store",
            len(self._pure_records),
        )

    @staticmethod
    def _write_track(df: pl.DataFrame, *, prefix: str) -> None:
        """Write a single track DataFrame to the signals partition store.

        Groups rows by date and writes one Parquet file per partition
        with *prefix* prepended to the filename so that pure and LLM
        tracks can coexist in the same ``signals`` partition directory.
        """
        import time as _time

        signals_dir = get_data_dir("signals")
        for (dt_val,) in df.select("dt").unique().sort("dt").iter_rows():
            part_dir = signals_dir / f"dt={dt_val.isoformat()}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_df = df.filter(pl.col("dt") == dt_val)
            fname = f"{prefix}data_{int(_time.time() * 1_000_000)}.parquet"
            part_df.write_parquet(part_dir / fname)

    def clear(self) -> None:
        """Clear in-memory records (does not delete persisted files)."""
        self._pure_records.clear()
        self._llm_records.clear()
        self._ticker_date_keys.clear()
        self._last_delta_lift = None


# ---------------------------------------------------------------------------
# Convenience: load outcomes from OHLCV (T+7 return)
# ---------------------------------------------------------------------------


def build_outcomes_from_ohlcv(
    ohlcv_df: pl.DataFrame,
    *,
    return_threshold: float = 0.10,
    holding_days: int = 7,
) -> pl.DataFrame:
    """Build an outcomes DataFrame from OHLCV data.

    For each (ticker, dt), compute the forward T+holding_days return
    using the close price.  An outcome ``hit`` is 1 if the forward
    return >= *return_threshold*.

    Args:
        ohlcv_df: DataFrame with columns ``ticker, dt, close``.
        return_threshold: Minimum return for a hit (default 10%).
        holding_days: Forward holding period in days.

    Returns:
        DataFrame with columns ``ticker, dt, fwd_return, hit``.
    """
    if ohlcv_df.height == 0:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8, "dt": pl.Date,
                "fwd_return": pl.Float64, "hit": pl.Int64,
            }
        )

    required = {"ticker", "dt", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise ValueError(f"ohlcv_df missing required columns: {sorted(missing)}")

    # Sort and compute forward close
    df = ohlcv_df.sort(["ticker", "dt"])
    df = df.with_columns(
        pl.col("close").shift(-holding_days).over("ticker").alias("close_fwd")
    )
    df = df.with_columns(
        ((pl.col("close_fwd") - pl.col("close")) / pl.col("close")).alias("fwd_return")
    )
    df = df.with_columns(
        (pl.col("fwd_return") >= return_threshold).cast(pl.Int64).alias("hit")
    )
    # Drop rows where forward return is null (end of window)
    df = df.filter(pl.col("fwd_return").is_not_null())
    return df.select(["ticker", "dt", "fwd_return", "hit"])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ablation_tracker(
    *,
    enabled: bool | None = None,
    window_trading_days: int = DEFAULT_WINDOW_TRADING_DAYS,
    k: int = DEFAULT_K,
) -> AblationTracker:
    """Create an AblationTracker with configuration from environment.

    Args:
        enabled: Override LLM_ABLATION_ENABLED.  If None, reads from settings.
        window_trading_days: Rolling window size in trading days.
        k: Top-K for Lift computation.

    Returns:
        Configured AblationTracker.
    """
    if enabled is None:
        try:
            settings = Settings()  # type: ignore[call-arg]
            enabled = settings.llm_ablation_enabled
        except Exception:
            enabled = True

    config = AblationConfig(
        window_trading_days=window_trading_days,
        window_calendar_days=int(window_trading_days * 1.4),
        k=k,
        enabled=enabled,
    )
    return AblationTracker(config)
