"""Point-in-time FINRA equity short-interest snapshots and features.

FINRA's free static files contain positions as of a *settlement* date, but the
information is not public until the corresponding publication date.  FINRA
does not publish an intraday release time, so this module conservatively makes
the value usable at the next NYSE decision close.  Revised rows are likewise
delayed until the first session after the revised file was observed.

Network access is deliberately optional and injectable.  The parser and
feature builder work directly with fixture bytes/data frames, while
``load_short_interest_file`` provides a small no-key production downloader
with an immutable, content-addressed raw cache.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import zipfile
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import Request, urlopen

import polars as pl

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_data_home
from alphascreener.market_calendar import future_market_date, market_dates_between

FINRA_SHORT_INTEREST_FILES_URL = (
    "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files"
)
FINRA_SHORT_INTEREST_FILE_URL = (
    "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{settlement:%Y%m%d}.csv"
)
DEFAULT_FINRA_USER_AGENT = "alpha-screener/0.2"

_MAX_ARCHIVE_MEMBERS = 64
_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_PUBLICATION_LAG_SESSIONS = 7
_DEFAULT_MAX_AGE_SESSIONS = 60
_FEATURE_CHUNK_SIZE = 50_000

_FEATURE_SCHEMA = {
    "ticker": pl.String,
    "decision_date": pl.Date,
    "short_interest_settlement_date": pl.Date,
    "short_interest_available_at": pl.Date,
    "short_interest_age_sessions": pl.Int64,
    "short_interest": pl.Int64,
    "short_interest_delta": pl.Float64,
    "days_to_cover": pl.Float64,
    "short_pct": pl.Float64,
    "short_interest_stock_split_flag": pl.Boolean,
    "short_interest_previous_stock_split_flag": pl.Boolean,
    "short_interest_delta_suppressed_by_split": pl.Boolean,
    "short_interest_revision_flag": pl.Boolean,
}
_FEATURE_COLUMNS = tuple(_FEATURE_SCHEMA)

_SYMBOL_FIELDS = (
    "symbolcode",
    "issuesymbolidentifier",
    "symbol",
    "issuesymbol",
)
_SETTLEMENT_FIELDS = ("settlementdate",)
_SHORT_INTEREST_FIELDS = (
    "currentshortpositionquantity",
    "currentshortsharenumber",
    "currentshort",
    "shortinterest",
)
_STOCK_SPLIT_FIELDS = ("stocksplitflag",)
_REVISION_FIELDS = ("revisionflag",)

Fetcher = Callable[[str], bytes]
PublicationDates = date | Mapping[date, date] | None


class FinraShortInterestDataError(RuntimeError):
    """Raised when a FINRA file cannot support deterministic point-in-time use."""


@dataclass(frozen=True)
class FinraShortInterestFile:
    """One parsed raw snapshot and its reproducibility metadata."""

    records: pl.DataFrame
    raw_path: Path
    sha256: str
    observed_at: date
    from_cache: bool


@dataclass(frozen=True)
class _Position:
    settlement_date: date
    available_at: date
    short_interest: int
    stock_split_flag: bool
    revision_flag: bool


@dataclass(frozen=True)
class _KnownPosition:
    current: _Position
    previous: _Position | None


def finra_publication_date(settlement_date: date) -> date:
    """Return FINRA's scheduled seventh-business-day publication date.

    Official publication dates may be supplied explicitly to
    :func:`parse_short_interest_file`.  This deterministic NYSE-session rule is
    the fallback for historical static files that only contain settlement
    dates.
    """
    try:
        return future_market_date(settlement_date, _PUBLICATION_LAG_SESSIONS)
    except (ValueError, IndexError) as exc:
        raise FinraShortInterestDataError(
            f"invalid FINRA settlement date: {settlement_date}"
        ) from exc


def finra_short_interest_file_url(settlement_date: date) -> str:
    """Return the official static-file URL for one reporting settlement date."""
    return FINRA_SHORT_INTEREST_FILE_URL.format(settlement=settlement_date)


def parse_short_interest_file(
    payload: bytes,
    *,
    publication_dates: PublicationDates = None,
    observed_at: date | None = None,
) -> pl.DataFrame:
    """Parse a FINRA pipe-delimited CSV or ZIP into a normalized frame.

    ``publication_dates`` may be one date for a single-cycle file or a mapping
    keyed by settlement date.  When omitted, FINRA's documented seventh
    business-day publication schedule is used.  ``observed_at`` is required if
    any row carries a revision flag; the revised value is unavailable before
    the exact file snapshot was observed.
    """
    if not payload:
        raise FinraShortInterestDataError("FINRA short-interest file is empty")

    normalized_rows: list[dict[str, Any]] = []
    for name, member in _payload_members(payload):
        normalized_rows.extend(
            _parse_delimited_member(
                member,
                name=name,
                publication_dates=publication_dates,
                observed_at=observed_at,
            )
        )
    if not normalized_rows:
        raise FinraShortInterestDataError("FINRA short-interest file contained no data rows")

    frame = pl.DataFrame(
        normalized_rows,
        schema={
            "symbol": pl.String,
            "settlement_date": pl.Date,
            "publication_date": pl.Date,
            "available_at": pl.Date,
            "short_interest": pl.Int64,
            "stock_split_flag": pl.Boolean,
            "revision_flag": pl.Boolean,
        },
    ).unique(maintain_order=True)
    _reject_conflicting_rows(frame)
    return frame.sort("symbol", "settlement_date", "available_at")


def load_short_interest_file(
    url: str,
    *,
    data_home: Path | None = None,
    fetcher: Fetcher | None = None,
    publication_dates: PublicationDates = None,
    observed_at: date | None = None,
    refresh: bool = False,
    user_agent: str = DEFAULT_FINRA_USER_AGENT,
) -> FinraShortInterestFile:
    """Load one official static file through an immutable local raw cache.

    A FINRA API key is not required.  Static URLs normally identify one
    reporting cycle, so a repeat reads the newest byte-for-byte snapshot for
    that URL without another request.  ``refresh=True`` checks for a corrected
    file while retaining every older raw snapshot.
    """
    if not url.lower().startswith("https://"):
        raise ValueError("FINRA short-interest URL must use HTTPS")

    cache_root = (data_home or get_data_home()) / "data" / "finra" / "short-interest" / "raw"
    source_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    source_dir = cache_root / source_key
    lock_path = cache_root / ".locks" / f"{source_key}.lock"

    with exclusive_file_lock(lock_path):
        cached = _latest_cached_snapshot(source_dir)
        if cached is not None and not refresh:
            raw_path, cached_observed_at = cached
            payload = _read_cached_payload(raw_path)
            frame = parse_short_interest_file(
                payload,
                publication_dates=publication_dates,
                observed_at=cached_observed_at,
            )
            return FinraShortInterestFile(
                records=frame,
                raw_path=raw_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                observed_at=cached_observed_at,
                from_cache=True,
            )

        acquisition_date = observed_at or date.today()
        network_fetcher = fetcher or _FinraHttpFetcher(user_agent=user_agent)
        try:
            payload = network_fetcher(url)
        except Exception as exc:
            raise FinraShortInterestDataError(
                f"could not download FINRA short-interest file: {exc}"
            ) from exc

        # Validate before persisting an HTTP error page or corrupt archive.
        frame = parse_short_interest_file(
            payload,
            publication_dates=publication_dates,
            observed_at=acquisition_date,
        )
        digest = hashlib.sha256(payload).hexdigest()
        suffix = ".zip" if zipfile.is_zipfile(io.BytesIO(payload)) else ".csv"
        raw_path = source_dir / f"{acquisition_date.isoformat()}-{digest}{suffix}"
        _create_immutable_file(raw_path, payload)
        return FinraShortInterestFile(
            records=frame,
            raw_path=raw_path,
            sha256=digest,
            observed_at=acquisition_date,
            from_cache=False,
        )


def short_interest_features(
    records: pl.DataFrame,
    *,
    tickers: Sequence[str] | Iterable[str],
    decision_dates: Sequence[date] | Iterable[date],
    average_daily_volume: Sequence[float | int | None] | Iterable[float | int | None],
    shares_outstanding: (Sequence[float | int | None] | Iterable[float | int | None] | None) = None,
    max_age_sessions: int = _DEFAULT_MAX_AGE_SESSIONS,
) -> pl.DataFrame:
    """Build point-in-time short-interest features for paired observations.

    Inputs are row-aligned (not a Cartesian product).  ``short_interest_delta``
    is the fractional change from the preceding *then-known* reporting cycle,
    except across a stock split reported for either cycle, where it remains
    null. ``days_to_cover`` uses the supplied point-in-time ADV, and
    ``short_pct`` remains null when point-in-time shares outstanding are
    unavailable.
    """
    if max_age_sessions < 0:
        raise ValueError("max_age_sessions must be non-negative")

    timelines = _build_known_position_timelines(records)
    frames: list[pl.DataFrame] = []
    columns = _empty_feature_columns()
    chunk_length = 0
    age_cache: dict[tuple[date, date], int] = {}
    for ticker, decision_date, adv, shares in _aligned_feature_inputs(
        tickers=tickers,
        decision_dates=decision_dates,
        average_daily_volume=average_daily_volume,
        shares_outstanding=shares_outstanding,
    ):
        normalized_ticker = _normalize_symbol(ticker)
        if isinstance(decision_date, datetime):
            decision_date = decision_date.date()
        elif not isinstance(decision_date, date):
            raise TypeError("decision_dates must contain date values")
        known = _known_as_of(timelines.get(normalized_ticker), decision_date)
        if known is not None:
            age_key = (known.current.available_at, decision_date)
            if age_key not in age_cache:
                age_cache[age_key] = _market_session_age(*age_key)
            age_sessions = age_cache[age_key]
            if age_sessions > max_age_sessions:
                known = None
        else:
            age_sessions = None

        _append_feature_values(
            columns,
            _feature_values(
                ticker=normalized_ticker,
                decision_date=decision_date,
                adv=adv,
                shares=shares,
                known=known,
                age_sessions=age_sessions if known is not None else None,
            ),
        )
        chunk_length += 1
        if chunk_length == _FEATURE_CHUNK_SIZE:
            frames.append(pl.DataFrame(columns, schema=_FEATURE_SCHEMA))
            columns = _empty_feature_columns()
            chunk_length = 0

    if chunk_length:
        frames.append(pl.DataFrame(columns, schema=_FEATURE_SCHEMA))
    if not frames:
        return pl.DataFrame(schema=_FEATURE_SCHEMA)
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical", rechunk=False)


def _aligned_feature_inputs(
    *,
    tickers: Sequence[str] | Iterable[str],
    decision_dates: Sequence[date] | Iterable[date],
    average_daily_volume: Sequence[float | int | None] | Iterable[float | int | None],
    shares_outstanding: Sequence[float | int | None] | Iterable[float | int | None] | None,
) -> Iterable[tuple[str, date, float | int | None, float | int | None]]:
    sentinel = object()
    inputs: tuple[Iterable[Any], ...]
    if shares_outstanding is None:
        inputs = (tickers, decision_dates, average_daily_volume)
    else:
        inputs = (tickers, decision_dates, average_daily_volume, shares_outstanding)

    for values in zip_longest(*inputs, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError(
                "tickers, decision_dates, average_daily_volume, and "
                "shares_outstanding must have equal lengths"
            )
        if shares_outstanding is None:
            ticker, decision_date, adv = values
            shares = None
        else:
            ticker, decision_date, adv, shares = values
        yield ticker, decision_date, adv, shares


def _empty_feature_columns() -> dict[str, list[Any]]:
    return {column: [] for column in _FEATURE_COLUMNS}


def _append_feature_values(
    columns: dict[str, list[Any]],
    values: tuple[Any, ...],
) -> None:
    for column, value in zip(_FEATURE_COLUMNS, values, strict=True):
        columns[column].append(value)


def _payload_members(payload: bytes) -> list[tuple[str, bytes]]:
    stream = io.BytesIO(payload)
    if not zipfile.is_zipfile(stream):
        return [("short-interest.csv", payload)]

    try:
        with zipfile.ZipFile(stream) as archive:
            members = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in {".csv", ".txt", ".psv"}
            ]
            if not members:
                raise FinraShortInterestDataError("FINRA ZIP contained no delimited data file")
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise FinraShortInterestDataError("FINRA ZIP contained too many data files")
            total_size = sum(info.file_size for info in members)
            if total_size > _MAX_UNCOMPRESSED_BYTES:
                raise FinraShortInterestDataError(
                    "FINRA ZIP uncompressed size exceeds safety limit"
                )
            if any(info.flag_bits & 0x1 for info in members):
                raise FinraShortInterestDataError("FINRA ZIP must not contain encrypted members")
            return [(info.filename, archive.read(info)) for info in members]
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise FinraShortInterestDataError(f"invalid FINRA ZIP: {exc}") from exc


def _parse_delimited_member(
    payload: bytes,
    *,
    name: str,
    publication_dates: PublicationDates,
    observed_at: date | None,
) -> list[dict[str, Any]]:
    text = _decode_payload(payload, name)
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return []
    delimiter = _delimiter(first_line)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise FinraShortInterestDataError(f"{name} has no header")
    canonical_fields = {
        _canonical_field(field): field for field in reader.fieldnames if field is not None
    }
    symbol_field = _required_field(canonical_fields, _SYMBOL_FIELDS, name)
    settlement_field = _required_field(canonical_fields, _SETTLEMENT_FIELDS, name)
    short_field = _required_field(canonical_fields, _SHORT_INTEREST_FIELDS, name)
    stock_split_field = _optional_field(canonical_fields, _STOCK_SPLIT_FIELDS)
    revision_field = _optional_field(canonical_fields, _REVISION_FIELDS)

    rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        if not any(value and value.strip() for value in row.values()):
            continue
        try:
            symbol = _normalize_symbol(row.get(symbol_field))
            settlement_date = _parse_date(row.get(settlement_field))
            short_interest = _parse_nonnegative_integer(row.get(short_field))
            stock_split_flag = _parse_flag(
                row.get(stock_split_field) if stock_split_field else None,
                label="stock split",
            )
            revision_flag = _parse_revision_flag(
                row.get(revision_field) if revision_field else None
            )
            publication_date = _publication_date_for(settlement_date, publication_dates)
        except (TypeError, ValueError, FinraShortInterestDataError) as exc:
            raise FinraShortInterestDataError(f"{name} row {row_number} is invalid: {exc}") from exc
        if publication_date < settlement_date:
            raise FinraShortInterestDataError(
                f"{name} row {row_number} publication date precedes settlement"
            )
        if revision_flag and observed_at is None:
            raise FinraShortInterestDataError(
                f"{name} row {row_number} is revised; observed_at is required"
            )
        release_date = max(
            publication_date,
            observed_at if revision_flag and observed_at is not None else publication_date,
        )
        available_at = _session_after_unknown_release(release_date)
        rows.append(
            {
                "symbol": symbol,
                "settlement_date": settlement_date,
                "publication_date": publication_date,
                "available_at": available_at,
                "short_interest": short_interest,
                "stock_split_flag": stock_split_flag,
                "revision_flag": revision_flag,
            }
        )
    return rows


def _decode_payload(payload: bytes, name: str) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise FinraShortInterestDataError(f"{name} is not decodable text")
    lowered = text[:500].lower()
    if "<html" in lowered or "<!doctype html" in lowered:
        raise FinraShortInterestDataError(f"{name} contained HTML, not FINRA data")
    return text


def _delimiter(first_line: str) -> str:
    counts = {delimiter: first_line.count(delimiter) for delimiter in ("|", ",", "\t")}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        raise FinraShortInterestDataError("FINRA file is not pipe-, comma-, or tab-delimited")
    return delimiter


def _canonical_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _required_field(
    fields: Mapping[str, str],
    aliases: Sequence[str],
    name: str,
) -> str:
    field = _optional_field(fields, aliases)
    if field is None:
        raise FinraShortInterestDataError(
            f"{name} missing required field; expected one of {list(aliases)}"
        )
    return field


def _optional_field(
    fields: Mapping[str, str],
    aliases: Sequence[str],
) -> str | None:
    return next((fields[alias] for alias in aliases if alias in fields), None)


def _normalize_symbol(value: Any) -> str:
    if value is None:
        raise ValueError("missing symbol")
    symbol = str(value).strip().upper().replace(".", "-")
    if not symbol or len(symbol) > 32:
        raise ValueError("invalid symbol")
    return symbol


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("missing date")
    raw = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"invalid date {raw!r}")


def _parse_nonnegative_integer(value: Any) -> int:
    if value is None:
        raise ValueError("missing short interest")
    raw = str(value).strip().replace(",", "")
    if not raw or not re.fullmatch(r"\d+", raw):
        raise ValueError(f"invalid short interest {value!r}")
    parsed = int(raw)
    if parsed > 2**63 - 1:
        raise ValueError("short interest exceeds Int64")
    return parsed


def _parse_revision_flag(value: Any) -> bool:
    return _parse_flag(value, label="revision")


def _parse_flag(value: Any, *, label: str) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().upper()
    if normalized in {"", "N", "NO", "FALSE", "0"}:
        return False
    truthy = {"Y", "YES", "TRUE", "1"}
    if label == "revision":
        truthy.update({"R", "REVISED"})
    elif label == "stock split":
        truthy.add("S")
    if normalized in truthy:
        return True
    raise ValueError(f"invalid {label} flag {value!r}")


def _publication_date_for(
    settlement_date: date,
    publication_dates: PublicationDates,
) -> date:
    if publication_dates is None:
        return finra_publication_date(settlement_date)
    if isinstance(publication_dates, date):
        return publication_dates
    try:
        value = publication_dates[settlement_date]
    except KeyError as exc:
        raise FinraShortInterestDataError(
            f"no publication date for settlement {settlement_date}"
        ) from exc
    if not isinstance(value, date):
        raise FinraShortInterestDataError("publication dates must be date values")
    return value


def _session_after_unknown_release(release_date: date) -> date:
    sessions = market_dates_between(
        release_date + timedelta(days=1),
        release_date + timedelta(days=14),
    )
    if not sessions:
        raise FinraShortInterestDataError(
            f"cannot resolve a session after FINRA release date {release_date}"
        )
    return sessions[0]


def _reject_conflicting_rows(frame: pl.DataFrame) -> None:
    conflicting = (
        frame.group_by("symbol", "settlement_date", "available_at")
        .agg(
            pl.col("short_interest").n_unique().alias("positions"),
            pl.col("stock_split_flag").n_unique().alias("stock_split_flags"),
            pl.col("revision_flag").n_unique().alias("revision_flags"),
        )
        .filter(
            (pl.col("positions") > 1)
            | (pl.col("stock_split_flags") > 1)
            | (pl.col("revision_flags") > 1)
        )
    )
    if not conflicting.is_empty():
        example = conflicting.row(0, named=True)
        raise FinraShortInterestDataError(
            "conflicting FINRA rows for "
            f"{example['symbol']} settlement {example['settlement_date']} "
            f"available {example['available_at']}"
        )


def _build_known_position_timelines(
    records: pl.DataFrame,
) -> dict[str, tuple[list[date], list[_KnownPosition]]]:
    required = {
        "symbol",
        "settlement_date",
        "available_at",
        "short_interest",
        "stock_split_flag",
        "revision_flag",
    }
    if missing := required - set(records.columns):
        raise ValueError(f"short-interest records missing columns: {sorted(missing)}")
    normalized = records.select(
        pl.col("symbol")
        .cast(pl.String, strict=True)
        .str.to_uppercase()
        .str.replace_all(r"\.", "-")
        .alias("symbol"),
        pl.col("settlement_date").cast(pl.Date, strict=True),
        pl.col("available_at").cast(pl.Date, strict=True),
        pl.col("short_interest").cast(pl.Int64, strict=True),
        pl.col("stock_split_flag").cast(pl.Boolean, strict=True),
        pl.col("revision_flag").cast(pl.Boolean, strict=True),
    ).sort("symbol", "available_at", "settlement_date")
    if normalized.filter(
        pl.any_horizontal(
            pl.col("symbol").is_null(),
            pl.col("settlement_date").is_null(),
            pl.col("available_at").is_null(),
            pl.col("short_interest").is_null(),
            pl.col("stock_split_flag").is_null(),
            pl.col("revision_flag").is_null(),
        )
        | (pl.col("available_at") < pl.col("settlement_date"))
        | (pl.col("short_interest") < 0)
    ).height:
        raise ValueError("short-interest records contain invalid point-in-time rows")
    _reject_conflicting_rows(normalized)

    timelines: dict[str, tuple[list[date], list[_KnownPosition]]] = {}
    for key, group in normalized.group_by("symbol", maintain_order=True):
        symbol = key[0] if isinstance(key, tuple) else key
        state: dict[date, _Position] = {}
        event_dates: list[date] = []
        snapshots: list[_KnownPosition] = []
        for event_key, event_group in group.group_by("available_at", maintain_order=True):
            event_date = event_key[0] if isinstance(event_key, tuple) else event_key
            for row in event_group.iter_rows(named=True):
                state[row["settlement_date"]] = _Position(
                    settlement_date=row["settlement_date"],
                    available_at=row["available_at"],
                    short_interest=row["short_interest"],
                    stock_split_flag=row["stock_split_flag"],
                    revision_flag=row["revision_flag"],
                )
            settlement_dates = sorted(state)
            current = state[settlement_dates[-1]]
            previous = state[settlement_dates[-2]] if len(settlement_dates) >= 2 else None
            event_dates.append(event_date)
            snapshots.append(
                _KnownPosition(
                    current=current,
                    previous=previous,
                )
            )
        timelines[str(symbol)] = (event_dates, snapshots)
    return timelines


def _known_as_of(
    timeline: tuple[list[date], list[_KnownPosition]] | None,
    decision_date: date,
) -> _KnownPosition | None:
    if timeline is None:
        return None
    event_dates, snapshots = timeline
    index = bisect_right(event_dates, decision_date) - 1
    return None if index < 0 else snapshots[index]


def _market_session_age(available_at: date, decision_date: date) -> int:
    if decision_date < available_at:
        raise ValueError("decision date precedes feature availability")
    # Same-day publication is age zero.  Every later NYSE session increments
    # age, even when publication happened on a weekend/holiday.
    return len(market_dates_between(available_at + timedelta(days=1), decision_date))


def _feature_values(
    *,
    ticker: str,
    decision_date: date,
    adv: float | int | None,
    shares: float | int | None,
    known: _KnownPosition | None,
    age_sessions: int | None,
) -> tuple[Any, ...]:
    if known is None:
        return (
            ticker,
            decision_date,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    current = known.current.short_interest
    previous = known.previous
    split_suppressed = previous is not None and (
        known.current.stock_split_flag or previous.stock_split_flag
    )
    delta = (
        None
        if previous is None or previous.short_interest <= 0 or split_suppressed
        else current / previous.short_interest - 1.0
    )
    safe_adv = _positive_float(adv)
    safe_shares = _positive_float(shares)
    return (
        ticker,
        decision_date,
        known.current.settlement_date,
        known.current.available_at,
        age_sessions,
        current,
        delta,
        None if safe_adv is None else current / safe_adv,
        None if safe_shares is None else current / safe_shares,
        known.current.stock_split_flag,
        None if previous is None else previous.stock_split_flag,
        split_suppressed,
        known.current.revision_flag,
    )


def _positive_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0 and math.isfinite(parsed) else None


def _latest_cached_snapshot(source_dir: Path) -> tuple[Path, date] | None:
    if not source_dir.is_dir():
        return None
    candidates: list[tuple[date, int, Path]] = []
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2})-([0-9a-f]{64})\.(?:csv|zip)",
            path.name,
        )
        if match is None:
            continue
        try:
            observed_at = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        candidates.append((observed_at, path.stat().st_mtime_ns, path))
    if not candidates:
        return None
    observed_at, _, path = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2].name),
    )
    return path, observed_at


def _read_cached_payload(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FinraShortInterestDataError(f"could not read cached FINRA file: {exc}") from exc
    match = re.fullmatch(
        r"\d{4}-\d{2}-\d{2}-([0-9a-f]{64})\.(?:csv|zip)",
        path.name,
    )
    if match is None or hashlib.sha256(payload).hexdigest() != match.group(1):
        raise FinraShortInterestDataError(f"cached FINRA file failed its content hash: {path}")
    return payload


def _create_immutable_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_cached_payload(path) != payload:
            raise FinraShortInterestDataError(f"FINRA content-addressed cache collision: {path}")
        return
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".tmp-",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if _read_cached_payload(path) != payload:
                raise FinraShortInterestDataError(
                    f"FINRA content-addressed cache collision: {path}"
                )
    except OSError as exc:
        raise FinraShortInterestDataError(
            f"could not cache FINRA short-interest file: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


class _FinraHttpFetcher:
    def __init__(self, *, user_agent: str) -> None:
        self.user_agent = user_agent

    def __call__(self, url: str) -> bytes:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/csv, application/zip, application/octet-stream",
            },
        )
        with urlopen(request, timeout=30) as response:
            return response.read()
