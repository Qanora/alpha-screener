"""Point-in-time SEC event features for ranking and research.

This module deliberately separates feature construction from EDGAR transport.
Callers can pass normalized Polars frames directly, or inject a provider that
returns the same frames.  Coverage is explicit: an empty event frame only
means "no events" when a complete coverage interval proves that the full
lookback was observed.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl

from alphascreener.market_calendar import market_dates_between, market_session_close

DEFAULT_LOOKBACK_SESSIONS = 60
CLUSTER_BUY_MIN_BUYERS = 2

FILING_COVERAGE_SOURCE = "filings"
INSIDER_COVERAGE_SOURCE = "insider_transactions"
_VALID_COVERAGE_STATUSES = frozenset({"complete", "partial", "missing"})
_SOURCE_ALIASES = {
    "filing": FILING_COVERAGE_SOURCE,
    "filings": FILING_COVERAGE_SOURCE,
    "insider": INSIDER_COVERAGE_SOURCE,
    "form4": INSIDER_COVERAGE_SOURCE,
    "insider_transaction": INSIDER_COVERAGE_SOURCE,
    "insider_transactions": INSIDER_COVERAGE_SOURCE,
}

# Item 9.01 is an exhibit index rather than a standalone material event.
_MATERIAL_8K_ITEMS = frozenset(
    {
        "1.01",
        "1.02",
        "1.03",
        "1.04",
        "2.01",
        "2.02",
        "2.03",
        "2.04",
        "2.05",
        "2.06",
        "3.01",
        "3.02",
        "3.03",
        "4.01",
        "4.02",
        "5.01",
        "5.02",
        "5.03",
        "5.04",
        "5.05",
        "5.06",
        "5.07",
        "5.08",
        "6.01",
        "6.02",
        "6.03",
        "6.04",
        "6.05",
        "7.01",
        "8.01",
    }
)
_OFFERING_FORMS = frozenset({"S-3", "424B5", "EFFECT"})
_LATE_FILING_FORMS = frozenset({"NT 10-Q", "NT 10-K"})
_ITEM_PATTERN = re.compile(r"(?<!\d)(\d+\.\d+)(?!\d)")
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SecSignalDataset:
    """Normalized SEC records and their independently auditable coverage."""

    filings: pl.DataFrame
    insider_transactions: pl.DataFrame
    coverage: pl.DataFrame


class SecSignalProvider(Protocol):
    """Injectable source interface for an EDGAR loader or a frozen snapshot."""

    def load(
        self,
        tickers: tuple[str, ...],
        start: date,
        end: date,
    ) -> SecSignalDataset:
        """Load records and coverage intersecting the requested session range."""


@dataclass(frozen=True)
class _Filing:
    ticker: str
    available_at: datetime
    effective_session: date
    form: str
    base_form: str
    items: frozenset[str]
    accession_number: str | None


@dataclass(frozen=True)
class _InsiderBuy:
    ticker: str
    available_at: datetime
    effective_session: date
    amount_usd: float
    insider_id: str
    is_10b5_1: bool | None
    transaction_id: str | None


@dataclass(frozen=True)
class _Coverage:
    ticker: str
    source: str
    start: date
    end: date
    status: str


@dataclass(frozen=True)
class _DecisionContext:
    decision_date: date
    cutoff: datetime
    sessions: tuple[date, ...]
    session_index: dict[date, int]

    @property
    def lookback_start(self) -> date:
        return self.sessions[0]


CoverageSignature = tuple[tuple[date, date, str], ...]


def build_sec_signal_features(
    decisions: pl.DataFrame,
    *,
    filings: pl.DataFrame | None = None,
    insider_transactions: pl.DataFrame | None = None,
    coverage: pl.DataFrame | None = None,
    provider: SecSignalProvider | None = None,
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
) -> pl.DataFrame:
    """Build deterministic, point-in-time SEC features.

    ``decisions`` must contain ``ticker`` and ``decision_date``.  Filing rows
    require ``ticker``, ``accepted_at``, ``form`` and ``items``; an optional
    ``available_at`` is honored in addition to ``accepted_at``.  Insider rows
    require ``ticker``, ``available_at``, ``transaction_code``, ``shares``,
    ``price`` and ``insider_id``.  Only Form 4 transaction code ``P`` enters
    the buy features, while the tri-state ``is_10b5_1`` value is retained in
    separate dollar buckets.

    Coverage rows require ``ticker``, ``source``, ``coverage_start``,
    ``coverage_end`` and ``status``.  A source is complete only when complete
    intervals cover all lookback sessions.  Features for incomplete sources
    are null, whereas complete coverage with no events produces zeros.
    """
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    normalized_decisions, decision_contexts = _normalize_decisions(
        decisions,
        lookback_sessions,
    )
    if normalized_decisions.is_empty():
        return _empty_feature_frame()

    supplied_frames = any(frame is not None for frame in (filings, insider_transactions, coverage))
    if provider is not None and supplied_frames:
        raise ValueError("pass either provider or normalized frames, not both")
    if provider is not None:
        requested_tickers = tuple(
            normalized_decisions["ticker"].unique(maintain_order=True).to_list()
        )
        dataset = provider.load(
            requested_tickers,
            min(context.lookback_start for context in decision_contexts.values()),
            max(decision_contexts),
        )
        filings = dataset.filings
        insider_transactions = dataset.insider_transactions
        coverage = dataset.coverage

    normalized_filings = _normalize_filings(filings)
    normalized_buys = _normalize_insider_transactions(insider_transactions)
    normalized_coverage = _normalize_coverage(coverage)

    filings_by_ticker: dict[str, list[_Filing]] = defaultdict(list)
    for filing in normalized_filings:
        filings_by_ticker[filing.ticker].append(filing)
    filing_sessions_by_ticker: dict[str, tuple[date, ...]] = {}
    for ticker, ticker_filings in filings_by_ticker.items():
        ticker_filings.sort(key=lambda filing: (filing.effective_session, filing.available_at))
        filing_sessions_by_ticker[ticker] = tuple(
            filing.effective_session for filing in ticker_filings
        )

    buys_by_ticker: dict[str, list[_InsiderBuy]] = defaultdict(list)
    for buy in normalized_buys:
        buys_by_ticker[buy.ticker].append(buy)
    buy_sessions_by_ticker: dict[str, tuple[date, ...]] = {}
    for ticker, ticker_buys in buys_by_ticker.items():
        ticker_buys.sort(key=lambda buy: (buy.effective_session, buy.available_at))
        buy_sessions_by_ticker[ticker] = tuple(buy.effective_session for buy in ticker_buys)

    coverage_by_key: dict[tuple[str, str], list[_Coverage]] = defaultdict(list)
    for interval in normalized_coverage:
        coverage_by_key[(interval.ticker, interval.source)].append(interval)
    coverage_signatures = {
        key: _coverage_signature(intervals) for key, intervals in coverage_by_key.items()
    }
    coverage_bounds_cache: dict[
        CoverageSignature,
        tuple[date | None, date | None],
    ] = {}
    coverage_status_cache: dict[tuple[CoverageSignature, date], str] = {}

    def coverage_bounds(
        signature: CoverageSignature,
    ) -> tuple[date | None, date | None]:
        cached = coverage_bounds_cache.get(signature)
        if cached is None:
            cached = _coverage_bounds(signature)
            coverage_bounds_cache[signature] = cached
        return cached

    def coverage_status(
        signature: CoverageSignature,
        context: _DecisionContext,
    ) -> str:
        key = (signature, context.decision_date)
        cached = coverage_status_cache.get(key)
        if cached is None:
            cached = _coverage_status(signature, context.sessions)
            coverage_status_cache[key] = cached
        return cached

    output_schema = {"_row_order": pl.UInt32, **_feature_schema()}
    output_chunks: list[pl.DataFrame] = []
    for key, decision_chunk in normalized_decisions.group_by(
        "decision_date",
        maintain_order=True,
    ):
        decision_date = key[0] if isinstance(key, tuple) else key
        context = decision_contexts[decision_date]
        output: list[dict[str, object]] = []
        for decision in decision_chunk.select("_row_order", "ticker").iter_rows(named=True):
            ticker = decision["ticker"]
            filing_signature = coverage_signatures.get(
                (ticker, FILING_COVERAGE_SOURCE),
                (),
            )
            insider_signature = coverage_signatures.get(
                (ticker, INSIDER_COVERAGE_SOURCE),
                (),
            )
            filing_status = coverage_status(filing_signature, context)
            insider_status = coverage_status(insider_signature, context)
            filing_start, filing_end = coverage_bounds(filing_signature)
            insider_start, insider_end = coverage_bounds(insider_signature)

            row: dict[str, object] = {
                "_row_order": decision["_row_order"],
                "ticker": ticker,
                "decision_date": decision_date,
                "decision_cutoff": context.cutoff,
                "sec_lookback_start": context.lookback_start,
                "lookback_sessions": len(context.sessions),
                "filings_coverage": filing_status,
                "filings_coverage_start": filing_start,
                "filings_coverage_end": filing_end,
                "insider_coverage": insider_status,
                "insider_coverage_start": insider_start,
                "insider_coverage_end": insider_end,
                "sec_coverage": _combined_coverage(
                    filing_status,
                    insider_status,
                ),
            }

            if filing_status == "complete":
                eligible_filings = _eligible_records(
                    filings_by_ticker.get(ticker, []),
                    filing_sessions_by_ticker.get(ticker, ()),
                    context.lookback_start,
                    decision_date,
                    context.cutoff,
                )
                row.update(
                    _filing_features(
                        eligible_filings,
                        session_index=context.session_index,
                        decision_date=decision_date,
                    )
                )
            else:
                row.update(_missing_filing_features())

            if insider_status == "complete":
                eligible_buys = _eligible_records(
                    buys_by_ticker.get(ticker, []),
                    buy_sessions_by_ticker.get(ticker, ()),
                    context.lookback_start,
                    decision_date,
                    context.cutoff,
                )
                row.update(_insider_features(eligible_buys))
            else:
                row.update(_missing_insider_features())
            output.append(row)

        output_chunks.append(
            pl.DataFrame(
                output,
                schema=output_schema,
                orient="row",
            )
        )

    return (
        pl.concat(output_chunks, how="vertical", rechunk=False)
        .sort("_row_order")
        .drop("_row_order")
    )


def _normalize_decisions(
    decisions: pl.DataFrame,
    lookback_sessions: int,
) -> tuple[pl.DataFrame, dict[date, _DecisionContext]]:
    _require_columns(decisions, {"ticker", "decision_date"}, "decisions")
    ticker_cache: dict[object, str] = {}
    date_cache: dict[object, date] = {}

    def normalize_ticker(value: object) -> str:
        try:
            return ticker_cache[value]
        except KeyError:
            normalized = _normalize_ticker(value)
            ticker_cache[value] = normalized
            return normalized
        except TypeError:
            return _normalize_ticker(value)

    def normalize_date(value: object) -> date:
        try:
            return date_cache[value]
        except KeyError:
            normalized = _coerce_date(value, "decision_date")
            date_cache[value] = normalized
            return normalized
        except TypeError:
            return _coerce_date(value, "decision_date")

    normalized = (
        decisions.select("ticker", "decision_date")
        .with_row_index("_row_order")
        .with_columns(
            pl.col("ticker").map_elements(
                normalize_ticker,
                return_dtype=pl.String,
                skip_nulls=False,
            ),
            pl.col("decision_date").map_elements(
                normalize_date,
                return_dtype=pl.Date,
                skip_nulls=False,
            ),
        )
    )
    if normalized.is_empty():
        return normalized, {}
    duplicates = normalized.group_by("ticker", "decision_date").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        example = duplicates.row(0, named=True)
        raise ValueError(
            "decisions contain duplicate normalized ticker/date rows: "
            f"{example['ticker']} {example['decision_date']}"
        )

    contexts: dict[date, _DecisionContext] = {}
    for decision_date in normalized["decision_date"].unique(maintain_order=True).to_list():
        sessions = tuple(_session_window(decision_date, lookback_sessions))
        contexts[decision_date] = _DecisionContext(
            decision_date=decision_date,
            cutoff=market_session_close(decision_date),
            sessions=sessions,
            session_index={session: index for index, session in enumerate(sessions)},
        )
    return normalized, contexts


def _normalize_filings(filings: pl.DataFrame | None) -> list[_Filing]:
    if filings is None:
        return []
    _require_columns(
        filings,
        {"ticker", "accepted_at", "form", "items"},
        "filings",
    )
    seen: set[tuple[object, ...]] = set()
    normalized: list[_Filing] = []
    for row in filings.iter_rows(named=True):
        ticker = _normalize_ticker(row["ticker"])
        accepted_at = _coerce_timestamp(row["accepted_at"], "accepted_at")
        available_value = row.get("available_at")
        available_at = (
            _coerce_timestamp(available_value, "available_at")
            if available_value is not None
            else accepted_at
        )
        known_at = max(accepted_at, available_at)
        form = _normalize_form(row["form"])
        items = _parse_items(row["items"])
        accession = _optional_text(row.get("accession_number"))
        dedupe_key = (
            ticker,
            accession or "",
            known_at,
            form,
            tuple(sorted(items)),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            _Filing(
                ticker=ticker,
                available_at=known_at,
                effective_session=_effective_session(known_at),
                form=form,
                base_form=_base_form(form),
                items=items,
                accession_number=accession,
            )
        )
    return normalized


def _normalize_insider_transactions(
    transactions: pl.DataFrame | None,
) -> list[_InsiderBuy]:
    if transactions is None:
        return []
    _require_columns(
        transactions,
        {
            "ticker",
            "available_at",
            "transaction_code",
            "shares",
            "price",
            "insider_id",
            "is_10b5_1",
        },
        "insider_transactions",
    )
    seen: set[tuple[object, ...]] = set()
    normalized: list[_InsiderBuy] = []
    for row in transactions.iter_rows(named=True):
        code = str(row["transaction_code"]).strip().upper()
        if code != "P":
            continue
        ticker = _normalize_ticker(row["ticker"])
        available_at = _coerce_timestamp(row["available_at"], "available_at")
        accepted_value = row.get("accepted_at")
        if accepted_value is not None:
            accepted_at = _coerce_timestamp(accepted_value, "accepted_at")
            available_at = max(available_at, accepted_at)
        shares = _nonnegative_number(row["shares"], "shares")
        price = _nonnegative_number(row["price"], "price")
        insider_id = _required_text(row["insider_id"], "insider_id")
        is_10b5_1 = _optional_bool(row["is_10b5_1"], "is_10b5_1")
        transaction_id = _optional_text(row.get("transaction_id"))
        dedupe_key = (
            ticker,
            transaction_id or "",
            available_at,
            insider_id,
            shares,
            price,
            is_10b5_1,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            _InsiderBuy(
                ticker=ticker,
                available_at=available_at,
                effective_session=_effective_session(available_at),
                amount_usd=shares * price,
                insider_id=insider_id,
                is_10b5_1=is_10b5_1,
                transaction_id=transaction_id,
            )
        )
    return normalized


def _normalize_coverage(coverage: pl.DataFrame | None) -> list[_Coverage]:
    if coverage is None:
        return []
    _require_columns(
        coverage,
        {"ticker", "source", "coverage_start", "coverage_end", "status"},
        "coverage",
    )
    normalized: list[_Coverage] = []
    for row in coverage.iter_rows(named=True):
        source_value = str(row["source"]).strip().lower()
        try:
            source = _SOURCE_ALIASES[source_value]
        except KeyError as exc:
            raise ValueError(f"unsupported SEC coverage source: {row['source']!r}") from exc
        status = str(row["status"]).strip().lower()
        if status not in _VALID_COVERAGE_STATUSES:
            raise ValueError(f"unsupported SEC coverage status: {row['status']!r}")
        start = _coerce_date(row["coverage_start"], "coverage_start")
        end = _coerce_date(row["coverage_end"], "coverage_end")
        if end < start:
            raise ValueError("coverage_end must not precede coverage_start")
        normalized.append(
            _Coverage(
                ticker=_normalize_ticker(row["ticker"]),
                source=source,
                start=start,
                end=end,
                status=status,
            )
        )
    return normalized


def _filing_features(
    filings: list[_Filing],
    *,
    session_index: Mapping[date, int],
    decision_date: date,
) -> dict[str, object]:
    earnings_sessions = [
        filing.effective_session
        for filing in filings
        if filing.base_form == "8-K" and "2.02" in filing.items
    ]
    latest_earnings = max(earnings_sessions, default=None)
    days_since_earnings = (
        session_index[decision_date] - session_index[latest_earnings]
        if latest_earnings is not None
        else None
    )
    material_count = sum(
        filing.base_form == "6-K"
        or (filing.base_form == "8-K" and bool(filing.items.intersection(_MATERIAL_8K_ITEMS)))
        for filing in filings
    )
    return {
        "days_since_8k_earnings": days_since_earnings,
        "current_report_count": sum(filing.base_form in {"8-K", "6-K"} for filing in filings),
        "material_event_count": material_count,
        "recent_offering_risk": any(filing.base_form in _OFFERING_FORMS for filing in filings),
        "late_filing_risk": any(filing.base_form in _LATE_FILING_FORMS for filing in filings),
        "new_13d": any(filing.form == "SC 13D" for filing in filings),
    }


def _insider_features(buys: list[_InsiderBuy]) -> dict[str, object]:
    # Joint Form 4 filings can associate one transaction with more than one
    # reporting owner.  Preserve all owners for the breadth signal, but count
    # the transaction dollars only once.
    amount_buys: list[_InsiderBuy] = []
    transactions: dict[str, tuple[float, bool | None]] = {}
    for buy in buys:
        if buy.transaction_id is None:
            amount_buys.append(buy)
            continue
        fingerprint = (buy.amount_usd, buy.is_10b5_1)
        previous = transactions.get(buy.transaction_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError(
                    "insider transaction_id has conflicting amounts or "
                    f"10b5-1 status: {buy.transaction_id}"
                )
            continue
        transactions[buy.transaction_id] = fingerprint
        amount_buys.append(buy)

    total = sum(buy.amount_usd for buy in amount_buys)
    plan = sum(buy.amount_usd for buy in amount_buys if buy.is_10b5_1 is True)
    non_plan = sum(buy.amount_usd for buy in amount_buys if buy.is_10b5_1 is False)
    unknown = sum(buy.amount_usd for buy in amount_buys if buy.is_10b5_1 is None)
    buyer_count = len({buy.insider_id for buy in buys})
    return {
        "form4_open_market_buy_usd": total,
        "form4_10b5_1_buy_usd": plan,
        "form4_non_10b5_1_buy_usd": non_plan,
        "form4_unknown_10b5_1_buy_usd": unknown,
        "distinct_insider_buyers": buyer_count,
        "cluster_buy": buyer_count >= CLUSTER_BUY_MIN_BUYERS,
    }


def _missing_filing_features() -> dict[str, object]:
    return {
        "days_since_8k_earnings": None,
        "current_report_count": None,
        "material_event_count": None,
        "recent_offering_risk": None,
        "late_filing_risk": None,
        "new_13d": None,
    }


def _missing_insider_features() -> dict[str, object]:
    return {
        "form4_open_market_buy_usd": None,
        "form4_10b5_1_buy_usd": None,
        "form4_non_10b5_1_buy_usd": None,
        "form4_unknown_10b5_1_buy_usd": None,
        "distinct_insider_buyers": None,
        "cluster_buy": None,
    }


def _eligible_records(
    records: list[_Filing] | list[_InsiderBuy],
    effective_sessions: tuple[date, ...],
    start: date,
    end: date,
    cutoff: datetime,
) -> list[_Filing] | list[_InsiderBuy]:
    first = bisect_left(effective_sessions, start)
    last = bisect_right(effective_sessions, end)
    return [record for record in records[first:last] if record.available_at <= cutoff]


def _coverage_signature(intervals: list[_Coverage]) -> CoverageSignature:
    return tuple(sorted((interval.start, interval.end, interval.status) for interval in intervals))


def _coverage_status(
    signature: CoverageSignature,
    sessions: tuple[date, ...],
) -> str:
    complete_sessions: set[date] = set()
    overlaps = False
    for start, end, status in signature:
        covered = {session for session in sessions if start <= session <= end}
        if covered and status != "missing":
            overlaps = True
        if status == "complete":
            complete_sessions.update(covered)
    if len(complete_sessions) == len(sessions):
        return "complete"
    if overlaps:
        return "partial"
    return "missing"


def _coverage_bounds(
    signature: CoverageSignature,
) -> tuple[date | None, date | None]:
    if not signature:
        return None, None
    return (
        min(start for start, _end, _status in signature),
        max(end for _start, end, _status in signature),
    )


def _combined_coverage(filings: str, insider: str) -> str:
    if filings == insider == "complete":
        return "complete"
    if filings == insider == "missing":
        return "missing"
    return "partial"


def _session_window(decision_date: date, count: int) -> list[date]:
    # Sixty sessions fit comfortably inside 120 calendar days, but expand
    # deterministically for unusually long exchange closures.
    calendar_days = max(120, count * 3)
    for _ in range(5):
        sessions = market_dates_between(
            decision_date - timedelta(days=calendar_days),
            decision_date,
        )
        if not sessions or sessions[-1] != decision_date:
            raise ValueError(f"{decision_date} is not an NYSE session")
        if len(sessions) >= count:
            return sessions[-count:]
        calendar_days *= 2
    raise ValueError(f"cannot resolve {count} NYSE sessions before {decision_date}")


def _effective_session(available_at: datetime) -> date:
    # A Friday after-close filing is first usable at Monday's close.
    local_date = available_at.astimezone(_NEW_YORK).date()
    sessions = market_dates_between(
        local_date - timedelta(days=7),
        local_date + timedelta(days=14),
    )
    for session in sessions:
        if market_session_close(session) >= available_at:
            return session
    raise ValueError(f"cannot map SEC availability timestamp to NYSE session: {available_at}")


def _base_form(form: str) -> str:
    return form[:-2] if form.endswith("/A") else form


def _normalize_form(value: object) -> str:
    form = " ".join(_required_text(value, "form").upper().split())
    return form


def _parse_items(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset(_ITEM_PATTERN.findall(value))
    if isinstance(value, Iterable):
        return frozenset(item for raw in value for item in _ITEM_PATTERN.findall(str(raw)))
    raise ValueError(f"items must be text or an iterable, got {type(value).__name__}")


def _normalize_ticker(value: object) -> str:
    ticker = _required_text(value, "ticker").upper().replace(".", "-")
    if any(character.isspace() for character in ticker):
        raise ValueError(f"invalid ticker: {value!r}")
    return ticker


def _coerce_timestamp(value: object, field: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} is not a valid ISO timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def _coerce_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} is not a valid ISO date") from exc
    raise ValueError(f"{field} must be a date")


def _nonnegative_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _required_text(value: object, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} must not be null")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} must be boolean or null")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    description: str,
) -> None:
    if missing := required - set(frame.columns):
        raise ValueError(f"{description} missing columns: {sorted(missing)}")


def _feature_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "decision_date": pl.Date,
        "decision_cutoff": pl.Datetime(time_zone="UTC"),
        "sec_lookback_start": pl.Date,
        "lookback_sessions": pl.Int64,
        "filings_coverage": pl.String,
        "filings_coverage_start": pl.Date,
        "filings_coverage_end": pl.Date,
        "insider_coverage": pl.String,
        "insider_coverage_start": pl.Date,
        "insider_coverage_end": pl.Date,
        "sec_coverage": pl.String,
        "days_since_8k_earnings": pl.Int64,
        "current_report_count": pl.Int64,
        "material_event_count": pl.Int64,
        "recent_offering_risk": pl.Boolean,
        "late_filing_risk": pl.Boolean,
        "new_13d": pl.Boolean,
        "form4_open_market_buy_usd": pl.Float64,
        "form4_10b5_1_buy_usd": pl.Float64,
        "form4_non_10b5_1_buy_usd": pl.Float64,
        "form4_unknown_10b5_1_buy_usd": pl.Float64,
        "distinct_insider_buyers": pl.Int64,
        "cluster_buy": pl.Boolean,
    }


def _empty_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_feature_schema())
