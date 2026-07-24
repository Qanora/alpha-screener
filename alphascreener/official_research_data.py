"""Official, point-in-time alternative data for offline ranking research.

The provider intentionally stays separate from the production sync path.  It
combines three no-key official sources:

* SEC quarterly ``master.zip`` filing indexes;
* SEC quarterly Insider Transactions data sets; and
* FINRA twice-monthly equity short-interest files.

Every raw response is retained in an atomic, content-addressed cache.  The SEC
quarterly files only contain filing dates, not acceptance times, so their rows
are conservatively unavailable until the close of the next XNYS session.
FINRA settlement and publication dates remain separate, and an unknown
intraday publication time is delayed to the following XNYS decision.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import polars as pl

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_data_home
from alphascreener.finra_short_interest import (
    DEFAULT_FINRA_USER_AGENT,
    finra_publication_date,
    finra_short_interest_file_url,
    load_short_interest_file,
    short_interest_features,
)
from alphascreener.market_calendar import market_dates_between, market_session_close
from alphascreener.sec_signals import (
    DEFAULT_LOOKBACK_SESSIONS,
    FILING_COVERAGE_SOURCE,
    INSIDER_COVERAGE_SOURCE,
    build_sec_signal_features,
)

SUPPORTED_START = date(2021, 6, 1)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_MASTER_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip"
SEC_INSIDER_DATASETS_URL = (
    "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
)
DEFAULT_SEC_USER_AGENT = "Qanora alpha-screener/0.2 lijinze0118@live.com"

_SNAPSHOT_VERSION = "official-research-data-v1"
_MAX_ZIP_MEMBERS = 32
_MAX_ZIP_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
_SEC_REQUEST_INTERVAL_SECONDS = 0.125
_SEC_HTTP_LOCK = threading.Lock()
_LAST_SEC_REQUEST = 0.0

_SEC_SIGNAL_FORMS = frozenset(
    {
        "6-K",
        "8-K",
        "EFFECT",
        "NT 10-K",
        "NT 10-Q",
        "S-3",
        "SC 13D",
        "424B5",
    }
)
_INSIDER_ARCHIVE_PATTERN = re.compile(
    r"(?P<year>20\d{2})q(?P<quarter>[1-4])_form345\.zip(?:[?#].*)?$",
    re.IGNORECASE,
)

Fetcher = Callable[[str], bytes]


class OfficialResearchDataError(RuntimeError):
    """Raised when an official source cannot support reproducible research."""


@dataclass(frozen=True)
class OfficialResearchFeatureSet:
    """Join-ready research features and the exact raw-source snapshot digest."""

    features: pl.DataFrame
    snapshot_digest: str


@dataclass(frozen=True)
class _CachedResource:
    payload: bytes
    sha256: str
    path: Path
    observed_at: date
    from_cache: bool


@dataclass(frozen=True)
class _Submission:
    tickers: tuple[str, ...]
    filing_date: date
    available_at: datetime
    is_10b5_1: bool | None


def load_official_research_features(
    decisions: pl.DataFrame,
    *,
    end: date | None = None,
    data_home: Path | None = None,
    fetcher: Fetcher | None = None,
    refresh: bool = False,
    as_of: date | None = None,
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
    average_daily_volume_column: str = "average_daily_volume",
    shares_outstanding_column: str | None = "shares_outstanding",
    sec_user_agent: str | None = None,
) -> OfficialResearchFeatureSet:
    """Load official data and build features for paired ticker decisions.

    ``decisions`` must contain ``ticker``, ``decision_date`` and the configured
    average-daily-volume column.  An optional point-in-time shares-outstanding
    column enables ``short_pct``.  The returned frame has one row per input
    pair and can be joined on ``ticker`` and ``decision_date``.

    Source history starts in June 2021, when FINRA's free files began covering
    exchange-listed securities.  Earlier decisions and lookbacks crossing that
    boundary remain explicitly missing or partial rather than being imputed.
    """
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    normalized = _normalize_decisions(
        decisions,
        average_daily_volume_column=average_daily_volume_column,
        shares_outstanding_column=shares_outstanding_column,
    )
    decision_dates = normalized["decision_date"].to_list()
    requested_end = end or max(decision_dates)
    if not isinstance(requested_end, date) or isinstance(requested_end, datetime):
        raise TypeError("end must be a date")
    if requested_end < max(decision_dates):
        raise ValueError("end must not precede a decision date")
    acquisition_date = as_of or date.today()
    if requested_end > acquisition_date:
        raise ValueError("end must not be after as_of")

    root = data_home or get_data_home()
    network_fetcher = fetcher or _OfficialHttpFetcher(
        sec_user_agent=(
            sec_user_agent
            or os.environ.get(
                "ALPHASCREENER_SEC_USER_AGENT",
                DEFAULT_SEC_USER_AGENT,
            )
        )
    )
    cache = _RawCache(root / "data" / "official-research" / "raw")
    resources: list[tuple[str, str, str | None]] = []
    quarters = _quarters_between(SUPPORTED_START, requested_end)
    has_open_quarter = any(
        _quarter_is_open(year, quarter, acquisition_date) for year, quarter in quarters
    )

    ticker_resource = cache.load(
        SEC_COMPANY_TICKERS_URL,
        category="sec-company-tickers",
        suffix=".json",
        fetcher=network_fetcher,
        refresh=refresh,
        validator=parse_company_tickers,
        observed_at=acquisition_date,
        max_age_days=0 if has_open_quarter else None,
    )
    ticker_to_cik = parse_company_tickers(ticker_resource.payload)
    tickers = tuple(normalized["ticker"].unique(maintain_order=True).to_list())
    if ticker_resource.from_cache and any(ticker not in ticker_to_cik for ticker in tickers):
        ticker_resource = cache.load(
            SEC_COMPANY_TICKERS_URL,
            category="sec-company-tickers",
            suffix=".json",
            fetcher=network_fetcher,
            refresh=True,
            validator=parse_company_tickers,
            observed_at=acquisition_date,
        )
        ticker_to_cik = parse_company_tickers(ticker_resource.payload)
    resources.append(
        (
            SEC_COMPANY_TICKERS_URL,
            ticker_resource.sha256,
            ticker_resource.observed_at.isoformat(),
        )
    )
    sec_tickers = tuple(ticker for ticker in tickers if ticker in ticker_to_cik)
    unmapped_tickers = tuple(ticker for ticker in tickers if ticker not in ticker_to_cik)
    cik_to_tickers = _invert_requested_tickers(sec_tickers, ticker_to_cik)

    filings: list[pl.DataFrame] = []
    filing_coverage: list[dict[str, object]] = []
    for year, quarter in quarters:
        url = sec_master_url(year, quarter)
        resource = cache.load(
            url,
            category="sec-master",
            suffix=".zip",
            fetcher=network_fetcher,
            refresh=refresh,
            validator=_validate_sec_master_archive,
            observed_at=acquisition_date,
            max_age_days=(0 if _quarter_is_open(year, quarter, acquisition_date) else None),
        )
        resources.append((url, resource.sha256, resource.observed_at.isoformat()))
        filings.append(
            parse_sec_master_archive(
                resource.payload,
                cik_to_tickers=cik_to_tickers,
            )
        )
        interval = _quarter_interval(year, quarter, requested_end)
        last_received = _sec_master_last_received(resource.payload)
        if last_received > acquisition_date:
            raise OfficialResearchDataError(
                "SEC master Last Data Received is after the resource "
                f"observation date: {last_received}"
            )
        effective_complete_end = min(
            interval[1],
            _next_session_close(last_received).date(),
        )
        if effective_complete_end >= interval[0]:
            filing_coverage.extend(
                _coverage_rows(
                    sec_tickers,
                    source=FILING_COVERAGE_SOURCE,
                    start=interval[0],
                    end=effective_complete_end,
                    status="complete",
                )
            )
        if effective_complete_end < interval[1]:
            filing_coverage.extend(
                _coverage_rows(
                    sec_tickers,
                    source=FILING_COVERAGE_SOURCE,
                    start=max(
                        interval[0],
                        effective_complete_end.fromordinal(effective_complete_end.toordinal() + 1),
                    ),
                    end=interval[1],
                    status="missing",
                )
            )
        filing_coverage.extend(
            _coverage_rows(
                unmapped_tickers,
                source=FILING_COVERAGE_SOURCE,
                start=interval[0],
                end=interval[1],
                status="missing",
            )
        )

    page_resource = cache.load(
        SEC_INSIDER_DATASETS_URL,
        category="sec-insider-index",
        suffix=".html",
        fetcher=network_fetcher,
        refresh=refresh,
        validator=_validate_insider_index,
        observed_at=acquisition_date,
        max_age_days=0 if has_open_quarter else None,
    )
    insider_urls = discover_insider_archives(
        page_resource.payload,
        page_url=SEC_INSIDER_DATASETS_URL,
    )
    if page_resource.from_cache and any(key not in insider_urls for key in quarters):
        page_resource = cache.load(
            SEC_INSIDER_DATASETS_URL,
            category="sec-insider-index",
            suffix=".html",
            fetcher=network_fetcher,
            refresh=True,
            validator=_validate_insider_index,
            observed_at=acquisition_date,
        )
        insider_urls = discover_insider_archives(
            page_resource.payload,
            page_url=SEC_INSIDER_DATASETS_URL,
        )
    resources.append(
        (
            SEC_INSIDER_DATASETS_URL,
            page_resource.sha256,
            page_resource.observed_at.isoformat(),
        )
    )
    insider_transactions: list[pl.DataFrame] = []
    insider_coverage: list[dict[str, object]] = []
    for year, quarter in quarters:
        interval = _quarter_interval(year, quarter, requested_end)
        archive_url = insider_urls.get((year, quarter))
        if archive_url is None:
            insider_coverage.extend(
                _coverage_rows(
                    tickers,
                    source=INSIDER_COVERAGE_SOURCE,
                    start=interval[0],
                    end=interval[1],
                    status="missing",
                )
            )
            continue
        resource = cache.load(
            archive_url,
            category="sec-insider",
            suffix=".zip",
            fetcher=network_fetcher,
            refresh=refresh,
            validator=_validate_insider_archive,
            observed_at=acquisition_date,
            max_age_days=(0 if _quarter_is_open(year, quarter, acquisition_date) else None),
        )
        resources.append((archive_url, resource.sha256, resource.observed_at.isoformat()))
        insider_transactions.append(
            parse_insider_transactions_archive(
                resource.payload,
                cik_to_tickers=cik_to_tickers,
            )
        )
        insider_coverage.extend(
            _coverage_rows(
                sec_tickers,
                source=INSIDER_COVERAGE_SOURCE,
                start=interval[0],
                end=interval[1],
                status=(
                    "partial" if _quarter_is_open(year, quarter, acquisition_date) else "complete"
                ),
            )
        )
        insider_coverage.extend(
            _coverage_rows(
                unmapped_tickers,
                source=INSIDER_COVERAGE_SOURCE,
                start=interval[0],
                end=interval[1],
                status="missing",
            )
        )

    filing_frame = _concat_frames(filings, _filing_schema())
    insider_frame = _concat_frames(
        insider_transactions,
        _insider_transaction_schema(),
    )
    coverage_frame = pl.DataFrame(
        filing_coverage + insider_coverage,
        schema=_coverage_schema(),
        orient="row",
    )
    sec_features = build_sec_signal_features(
        normalized.select("ticker", "decision_date"),
        filings=filing_frame,
        insider_transactions=insider_frame,
        coverage=coverage_frame,
        lookback_sessions=lookback_sessions,
    )

    finra_records: list[pl.DataFrame] = []
    ticker_filter = set(tickers)
    for settlement_date, publication_date in finra_reporting_cycles(
        SUPPORTED_START,
        requested_end,
    ):
        # A cycle published after the final research decision cannot influence
        # any requested row, and its static file may not exist yet.
        if publication_date > requested_end:
            continue
        url = finra_short_interest_file_url(settlement_date)
        loaded = load_short_interest_file(
            url,
            data_home=root,
            fetcher=network_fetcher,
            publication_dates=publication_date,
            observed_at=acquisition_date,
            refresh=refresh,
            user_agent=DEFAULT_FINRA_USER_AGENT,
        )
        resources.append((url, loaded.sha256, loaded.observed_at.isoformat()))
        finra_records.append(loaded.records.filter(pl.col("symbol").is_in(ticker_filter)))
    short_records = _concat_frames(finra_records, _short_interest_schema())
    short_features = short_interest_features(
        short_records,
        tickers=normalized["ticker"].to_list(),
        decision_dates=decision_dates,
        average_daily_volume=normalized["_average_daily_volume"].to_list(),
        shares_outstanding=normalized["_shares_outstanding"].to_list(),
    )

    ordered_keys = normalized.select("ticker", "decision_date", "_row_order")
    features = (
        ordered_keys.join(
            sec_features,
            on=["ticker", "decision_date"],
            how="left",
            validate="1:1",
        )
        .join(
            short_features,
            on=["ticker", "decision_date"],
            how="left",
            validate="1:1",
        )
        .sort("_row_order")
        .drop("_row_order")
    )
    return OfficialResearchFeatureSet(
        features=features,
        snapshot_digest=_snapshot_digest(
            resources,
            end=requested_end,
            lookback_sessions=lookback_sessions,
        ),
    )


def parse_company_tickers(payload: bytes) -> dict[str, str]:
    """Parse current SEC ticker/CIK associations."""
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialResearchDataError(f"SEC company ticker map is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise OfficialResearchDataError("SEC company ticker map is not an object")
    result: dict[str, str] = {}
    for raw_row in document.values():
        if not isinstance(raw_row, Mapping):
            raise OfficialResearchDataError("SEC company ticker map contains an invalid row")
        ticker = _normalize_ticker(raw_row.get("ticker"))
        cik = _normalize_cik(raw_row.get("cik_str"))
        prior = result.get(ticker)
        if prior is not None and prior != cik:
            raise OfficialResearchDataError(
                f"SEC ticker map contains conflicting CIKs for {ticker}"
            )
        result[ticker] = cik
    if not result:
        raise OfficialResearchDataError("SEC company ticker map is empty")
    return result


def sec_master_url(year: int, quarter: int) -> str:
    """Return one official SEC full-index archive URL."""
    if year < 1994:
        raise ValueError("SEC master index year is invalid")
    if quarter not in {1, 2, 3, 4}:
        raise ValueError("quarter must be between 1 and 4")
    return SEC_MASTER_URL.format(year=year, quarter=quarter)


def parse_sec_master_archive(
    payload: bytes,
    *,
    cik_to_tickers: Mapping[str, Iterable[str]],
) -> pl.DataFrame:
    """Parse relevant filing rows from an SEC quarterly ``master.zip``."""
    member = _zip_member(payload, {"master.idx"})
    text = _decode_text(member, "SEC master.idx", encodings=("utf-8-sig", "cp1252"))
    header_index = next(
        (
            index
            for index, line in enumerate(text.splitlines())
            if line.strip() == "CIK|Company Name|Form Type|Date Filed|Filename"
        ),
        None,
    )
    if header_index is None:
        raise OfficialResearchDataError("SEC master.idx has no expected header")

    requested = {
        _normalize_cik(cik): tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers))
        for cik, tickers in cik_to_tickers.items()
    }
    rows: list[dict[str, object]] = []
    data_rows = 0
    for line_number, line in enumerate(
        text.splitlines()[header_index + 1 :],
        start=header_index + 2,
    ):
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        fields = stripped.split("|", 4)
        if len(fields) != 5:
            raise OfficialResearchDataError(f"SEC master.idx row {line_number} is malformed")
        cik_raw, _company_name, form_raw, filed_raw, filing_path = fields
        cik = _normalize_cik(cik_raw)
        form = _normalize_form(form_raw)
        filed_date = _parse_date(filed_raw, "SEC filing date")
        _validate_edgar_path(filing_path)
        data_rows += 1
        tickers = requested.get(cik)
        if tickers is None or _base_form(form) not in _SEC_SIGNAL_FORMS:
            continue
        available_at = _next_session_close(filed_date)
        accession_number = PurePosixPath(filing_path).stem
        for ticker in tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "filed_date": filed_date,
                    "accepted_at": available_at,
                    "available_at": available_at,
                    "form": form,
                    "items": "",
                    "accession_number": accession_number,
                    "filing_path": filing_path,
                }
            )
    if data_rows == 0:
        raise OfficialResearchDataError("SEC master.idx contained no filing rows")
    return pl.DataFrame(rows, schema=_filing_schema(), orient="row")


def discover_insider_archives(
    payload: bytes,
    *,
    page_url: str = SEC_INSIDER_DATASETS_URL,
) -> dict[tuple[int, int], str]:
    """Discover quarterly SEC insider ZIP URLs from the official landing page."""
    text = _decode_text(payload, "SEC insider data page", encodings=("utf-8-sig",))
    parser = _HrefParser()
    try:
        parser.feed(text)
    except Exception as exc:
        raise OfficialResearchDataError(f"SEC insider data page is malformed: {exc}") from exc
    archives: dict[tuple[int, int], str] = {}
    for href in parser.hrefs:
        absolute = urljoin(page_url, href)
        match = _INSIDER_ARCHIVE_PATTERN.search(absolute)
        if match is None:
            continue
        parsed = urlparse(absolute)
        if parsed.scheme != "https" or not (
            parsed.hostname == "sec.gov" or (parsed.hostname or "").endswith(".sec.gov")
        ):
            raise OfficialResearchDataError(
                f"SEC insider archive has an unexpected URL: {absolute}"
            )
        key = (int(match.group("year")), int(match.group("quarter")))
        previous = archives.get(key)
        if previous is not None and previous != absolute:
            raise OfficialResearchDataError(
                f"SEC insider page has conflicting archives for {key[0]} Q{key[1]}"
            )
        archives[key] = absolute
    if not archives:
        raise OfficialResearchDataError("SEC insider page contained no quarterly archive links")
    return archives


def parse_insider_transactions_archive(
    payload: bytes,
    *,
    cik_to_tickers: Mapping[str, Iterable[str]],
) -> pl.DataFrame:
    """Parse measurable Form 4 code-P transactions from one quarterly ZIP."""
    members = _zip_members(payload)
    submission_payload = _named_member(members, "SUBMISSION")
    transaction_payload = _named_member(members, "NONDERIV_TRANS")
    owner_payload = _named_member(members, "REPORTINGOWNER")
    requested = {
        _normalize_cik(cik): tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers))
        for cik, tickers in cik_to_tickers.items()
    }

    submissions: dict[str, _Submission] = {}
    for row_number, row in _tab_rows(
        submission_payload,
        name="SUBMISSION",
    ):
        _require_fields(
            row,
            {
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "DOCUMENT_TYPE",
                "ISSUERCIK",
            },
            "SUBMISSION",
        )
        if _normalize_form(row["DOCUMENT_TYPE"]) != "4":
            continue
        cik = _normalize_cik(row["ISSUERCIK"])
        tickers = requested.get(cik)
        if tickers is None:
            continue
        accession = _required_text(
            row["ACCESSION_NUMBER"],
            f"SUBMISSION row {row_number} accession",
        )
        filing_date = _parse_date(
            row["FILING_DATE"],
            f"SUBMISSION row {row_number} filing date",
        )
        submission = _Submission(
            tickers=tickers,
            filing_date=filing_date,
            available_at=_next_session_close(filing_date),
            is_10b5_1=_optional_bool(row.get("AFF10B5ONE")),
        )
        previous = submissions.get(accession)
        if previous is not None and previous != submission:
            raise OfficialResearchDataError(
                f"SUBMISSION contains conflicting accession {accession}"
            )
        submissions[accession] = submission

    owners: dict[str, set[str]] = {}
    for row_number, row in _tab_rows(owner_payload, name="REPORTINGOWNER"):
        _require_fields(
            row,
            {"ACCESSION_NUMBER", "RPTOWNERCIK"},
            "REPORTINGOWNER",
        )
        accession = _required_text(
            row["ACCESSION_NUMBER"],
            f"REPORTINGOWNER row {row_number} accession",
        )
        if accession not in submissions:
            continue
        owner_cik = _normalize_cik(row["RPTOWNERCIK"])
        owners.setdefault(accession, set()).add(owner_cik)

    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row_number, row in _tab_rows(
        transaction_payload,
        name="NONDERIV_TRANS",
    ):
        _require_fields(
            row,
            {
                "ACCESSION_NUMBER",
                "NONDERIV_TRANS_SK",
                "TRANS_CODE",
                "TRANS_SHARES",
                "TRANS_PRICEPERSHARE",
            },
            "NONDERIV_TRANS",
        )
        accession = _required_text(
            row["ACCESSION_NUMBER"],
            f"NONDERIV_TRANS row {row_number} accession",
        )
        submission = submissions.get(accession)
        if submission is None or str(row["TRANS_CODE"]).strip().upper() != "P":
            continue
        acquired_disposed = str(row.get("TRANS_ACQUIRED_DISP_CD") or "").strip().upper()
        if acquired_disposed and acquired_disposed != "A":
            continue
        shares = _optional_nonnegative_float(row["TRANS_SHARES"])
        price = _optional_nonnegative_float(row["TRANS_PRICEPERSHARE"])
        if shares is None or price is None:
            # Dollar purchase features require both values.  The source remains
            # fully covered; this individual as-filed transaction is not
            # quantitatively usable.
            continue
        transaction_key = _required_text(
            row["NONDERIV_TRANS_SK"],
            f"NONDERIV_TRANS row {row_number} key",
        )
        transaction_id = f"{accession}:{transaction_key}"
        for owner_cik in sorted(owners.get(accession, ())):
            for ticker in submission.tickers:
                dedupe_key = (
                    ticker,
                    transaction_id,
                    owner_cik,
                    shares,
                    price,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(
                    {
                        "ticker": ticker,
                        "available_at": submission.available_at,
                        "accepted_at": submission.available_at,
                        "transaction_code": "P",
                        "shares": shares,
                        "price": price,
                        "insider_id": owner_cik,
                        "is_10b5_1": submission.is_10b5_1,
                        "transaction_id": transaction_id,
                        "accession_number": accession,
                    }
                )
    return pl.DataFrame(
        rows,
        schema=_insider_transaction_schema(),
        orient="row",
    )


def finra_reporting_cycles(start: date, end: date) -> list[tuple[date, date]]:
    """Return designated semi-monthly settlements and explicit publications."""
    if end < start:
        return []
    cycles: list[tuple[date, date]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        sessions = market_dates_between(date(year, month, 1), month_end)
        if sessions:
            midmonth = max(session for session in sessions if session.day <= 15)
            for settlement in dict.fromkeys((midmonth, sessions[-1])):
                if start <= settlement <= end:
                    cycles.append((settlement, finra_publication_date(settlement)))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return cycles


class _RawCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(
        self,
        url: str,
        *,
        category: str,
        suffix: str,
        fetcher: Fetcher,
        refresh: bool,
        validator: Callable[[bytes], object],
        observed_at: date,
        max_age_days: int | None = None,
    ) -> _CachedResource:
        if not url.lower().startswith("https://"):
            raise ValueError("official data URLs must use HTTPS")
        if max_age_days is not None and max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        source_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        source_dir = self.root / category / source_key
        lock_path = self.root / ".locks" / f"{category}-{source_key}.lock"
        with exclusive_file_lock(lock_path):
            if not refresh:
                cached = _latest_cached(
                    source_dir,
                    suffix,
                    known_at=observed_at,
                )
                if cached is not None:
                    cached_path, cached_observed_at = cached
                    age_days = (observed_at - cached_observed_at).days
                    if max_age_days is None or age_days <= max_age_days:
                        payload, digest = _read_cached(cached_path)
                        validator(payload)
                        return _CachedResource(
                            payload=payload,
                            sha256=digest,
                            path=cached_path,
                            observed_at=cached_observed_at,
                            from_cache=True,
                        )
            try:
                payload = fetcher(url)
            except Exception as exc:
                raise OfficialResearchDataError(
                    f"could not download official research resource {url}: {exc}"
                ) from exc
            if not payload:
                raise OfficialResearchDataError(f"official research resource is empty: {url}")
            validator(payload)
            digest = hashlib.sha256(payload).hexdigest()
            path = source_dir / f"observed_at={observed_at.isoformat()}" / f"{digest}{suffix}"
            _atomic_content_write(path, payload)
            return _CachedResource(
                payload=payload,
                sha256=digest,
                path=path,
                observed_at=observed_at,
                from_cache=False,
            )


class _OfficialHttpFetcher:
    def __init__(self, *, sec_user_agent: str, timeout: float = 45.0) -> None:
        if not sec_user_agent.strip():
            raise ValueError("SEC user agent must not be empty")
        self.sec_user_agent = sec_user_agent
        self.timeout = timeout

    def __call__(self, url: str) -> bytes:
        parsed = urlparse(url)
        is_sec = parsed.hostname == "sec.gov" or (parsed.hostname or "").endswith(".sec.gov")
        user_agent = self.sec_user_agent if is_sec else DEFAULT_FINRA_USER_AGENT
        last_error: Exception | None = None
        for attempt in range(3):
            if is_sec:
                _wait_for_sec_slot()
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/zip,application/json,text/html,text/csv,*/*",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
            except (TimeoutError, URLError, OSError) as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
        raise OfficialResearchDataError(f"request failed for {url}: {last_error}")


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


def _normalize_decisions(
    decisions: pl.DataFrame,
    *,
    average_daily_volume_column: str,
    shares_outstanding_column: str | None,
) -> pl.DataFrame:
    required = {"ticker", "decision_date", average_daily_volume_column}
    if missing := required - set(decisions.columns):
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    has_shares = (
        shares_outstanding_column is not None and shares_outstanding_column in decisions.columns
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, date]] = set()
    for row_order, row in enumerate(decisions.iter_rows(named=True)):
        ticker = _normalize_ticker(row["ticker"])
        decision_date = _coerce_date(row["decision_date"], "decision_date")
        key = (ticker, decision_date)
        if key in seen:
            raise ValueError(f"duplicate ticker/decision pair: {ticker} {decision_date}")
        seen.add(key)
        adv = _optional_finite_float(row[average_daily_volume_column])
        shares = (
            _optional_finite_float(row[shares_outstanding_column])
            if has_shares and shares_outstanding_column is not None
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "decision_date": decision_date,
                "_average_daily_volume": adv,
                "_shares_outstanding": shares,
                "_row_order": row_order,
            }
        )
    if not rows:
        raise ValueError("decisions must not be empty")
    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "decision_date": pl.Date,
            "_average_daily_volume": pl.Float64,
            "_shares_outstanding": pl.Float64,
            "_row_order": pl.Int64,
        },
        orient="row",
    )


def _invert_requested_tickers(
    tickers: Iterable[str],
    ticker_to_cik: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for ticker in tickers:
        result.setdefault(ticker_to_cik[ticker], []).append(ticker)
    return {cik: tuple(values) for cik, values in result.items()}


def _quarters_between(start: date, end: date) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    year = start.year
    quarter = (start.month - 1) // 3 + 1
    last = (end.year, (end.month - 1) // 3 + 1)
    while (year, quarter) <= last:
        result.append((year, quarter))
        if quarter == 4:
            year += 1
            quarter = 1
        else:
            quarter += 1
    return result


def _quarter_interval(
    year: int,
    quarter: int,
    end: date,
) -> tuple[date, date]:
    first_month = (quarter - 1) * 3 + 1
    start = max(SUPPORTED_START, date(year, first_month, 1))
    last_month = first_month + 2
    quarter_end = date(
        year,
        last_month,
        calendar.monthrange(year, last_month)[1],
    )
    return start, min(end, quarter_end)


def _quarter_is_open(year: int, quarter: int, as_of: date) -> bool:
    first_month = (quarter - 1) * 3 + 1
    last_month = first_month + 2
    quarter_end = date(
        year,
        last_month,
        calendar.monthrange(year, last_month)[1],
    )
    return as_of <= quarter_end


def _coverage_rows(
    tickers: Iterable[str],
    *,
    source: str,
    start: date,
    end: date,
    status: str,
) -> list[dict[str, object]]:
    return [
        {
            "ticker": ticker,
            "source": source,
            "coverage_start": start,
            "coverage_end": end,
            "status": status,
        }
        for ticker in tickers
    ]


def _validate_sec_master_archive(payload: bytes) -> None:
    parse_sec_master_archive(payload, cik_to_tickers={})
    _sec_master_last_received(payload)


def _validate_insider_index(payload: bytes) -> None:
    discover_insider_archives(payload)


def _validate_insider_archive(payload: bytes) -> None:
    members = _zip_members(payload)
    for name in ("SUBMISSION", "NONDERIV_TRANS", "REPORTINGOWNER"):
        member = _named_member(members, name)
        rows = _tab_rows(member, name=name)
        try:
            next(rows)
        except StopIteration as exc:
            raise OfficialResearchDataError(f"SEC insider {name} table is empty") from exc


def _sec_master_last_received(payload: bytes) -> date:
    member = _zip_member(payload, {"master.idx"})
    text = _decode_text(
        member,
        "SEC master.idx",
        encodings=("utf-8-sig", "cp1252"),
    )
    match = re.search(
        r"^Last Data Received:\s*(.+?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise OfficialResearchDataError("SEC master.idx has no Last Data Received header")
    return _parse_date(match.group(1), "SEC master last-data date")


def _zip_member(payload: bytes, basenames: set[str]) -> bytes:
    members = _zip_members(payload)
    lowered = {name.lower() for name in basenames}
    matches = [
        member for name, member in members.items() if PurePosixPath(name).name.lower() in lowered
    ]
    if len(matches) != 1:
        raise OfficialResearchDataError(f"ZIP must contain exactly one of {sorted(basenames)}")
    return matches[0]


def _zip_members(payload: bytes) -> dict[str, bytes]:
    stream = io.BytesIO(payload)
    if not zipfile.is_zipfile(stream):
        raise OfficialResearchDataError("official archive is not a ZIP file")
    try:
        with zipfile.ZipFile(stream) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not infos or len(infos) > _MAX_ZIP_MEMBERS:
                raise OfficialResearchDataError("official ZIP has an invalid member count")
            if any(info.flag_bits & 0x1 for info in infos):
                raise OfficialResearchDataError("official ZIP contains an encrypted member")
            if sum(info.file_size for info in infos) > _MAX_ZIP_UNCOMPRESSED_BYTES:
                raise OfficialResearchDataError("official ZIP exceeds the uncompressed size limit")
            result: dict[str, bytes] = {}
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise OfficialResearchDataError("official ZIP contains an unsafe member path")
                result[info.filename] = archive.read(info)
            return result
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise OfficialResearchDataError(f"invalid official ZIP: {exc}") from exc


def _named_member(members: Mapping[str, bytes], stem: str) -> bytes:
    matches = [
        payload for name, payload in members.items() if PurePosixPath(name).stem.upper() == stem
    ]
    if len(matches) != 1:
        raise OfficialResearchDataError(f"SEC insider ZIP must contain exactly one {stem} table")
    return matches[0]


def _tab_rows(
    payload: bytes,
    *,
    name: str,
):
    text = _decode_text(payload, name, encodings=("utf-8-sig", "cp1252"))
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames is None:
        raise OfficialResearchDataError(f"{name} has no header")
    canonical = {
        field: str(field).strip().upper() for field in reader.fieldnames if field is not None
    }
    for row_number, raw in enumerate(reader, start=2):
        if not any(value and value.strip() for value in raw.values()):
            continue
        yield (
            row_number,
            {
                canonical[key]: value.strip() if isinstance(value, str) else value
                for key, value in raw.items()
                if key in canonical
            },
        )


def _require_fields(
    row: Mapping[str, object],
    required: set[str],
    table: str,
) -> None:
    if missing := required - set(row):
        raise OfficialResearchDataError(f"{table} missing required fields: {sorted(missing)}")


def _decode_text(
    payload: bytes,
    description: str,
    *,
    encodings: tuple[str, ...],
) -> str:
    for encoding in encodings:
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise OfficialResearchDataError(f"{description} is not decodable text")
    if "<html" in text[:500].lower() and description != "SEC insider data page":
        raise OfficialResearchDataError(f"{description} unexpectedly contained HTML")
    return text


def _validate_edgar_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 3
        or path.parts[:2] != ("edgar", "data")
        or path.suffix.lower() != ".txt"
    ):
        raise OfficialResearchDataError(f"invalid EDGAR filing path: {value!r}")


def _next_session_close(filed_date: date) -> datetime:
    sessions = market_dates_between(
        filed_date.fromordinal(filed_date.toordinal() + 1),
        filed_date.fromordinal(filed_date.toordinal() + 14),
    )
    if not sessions:
        raise OfficialResearchDataError(f"cannot find an XNYS session after {filed_date}")
    return market_session_close(sessions[0])


def _base_form(form: str) -> str:
    return form[:-2] if form.endswith("/A") else form


def _normalize_form(value: object) -> str:
    return " ".join(_required_text(value, "form").upper().split())


def _normalize_ticker(value: object) -> str:
    ticker = _required_text(value, "ticker").upper().replace(".", "-")
    if any(character.isspace() for character in ticker) or len(ticker) > 32:
        raise OfficialResearchDataError(f"invalid ticker: {value!r}")
    return ticker


def _normalize_cik(value: object) -> str:
    raw = _required_text(value, "CIK")
    if not raw.isdigit() or len(raw) > 10:
        raise OfficialResearchDataError(f"invalid CIK: {value!r}")
    return str(int(raw))


def _required_text(value: object, field: str) -> str:
    if value is None:
        raise OfficialResearchDataError(f"{field} is missing")
    text = str(value).strip()
    if not text:
        raise OfficialResearchDataError(f"{field} is empty")
    return text


def _parse_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = _required_text(value, field)
    for pattern in (
        "%Y-%m-%d",
        "%Y%m%d",
        "%d-%b-%Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(raw, pattern).date()
        except ValueError:
            continue
    raise OfficialResearchDataError(f"{field} is invalid: {value!r}")


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
    raise TypeError(f"{field} must contain date values")


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized == "":
        return None
    if normalized in {"1", "TRUE", "Y", "YES"}:
        return True
    if normalized in {"0", "FALSE", "N", "NO"}:
        return False
    raise OfficialResearchDataError(f"invalid boolean value: {value!r}")


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    raw = str(value).strip().replace(",", "")
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError as exc:
        raise OfficialResearchDataError(f"invalid numeric value: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise OfficialResearchDataError(f"invalid non-negative value: {value!r}")
    return number


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input value is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"input value is not finite: {value!r}")
    return number


def _concat_frames(
    frames: list[pl.DataFrame],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema=schema)
    nonempty = [frame for frame in frames if frame.height]
    if not nonempty:
        return pl.DataFrame(schema=schema)
    return pl.concat(nonempty, how="vertical").unique(maintain_order=True)


def _filing_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "cik": pl.String,
        "filed_date": pl.Date,
        "accepted_at": pl.Datetime(time_zone="UTC"),
        "available_at": pl.Datetime(time_zone="UTC"),
        "form": pl.String,
        "items": pl.String,
        "accession_number": pl.String,
        "filing_path": pl.String,
    }


def _insider_transaction_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "available_at": pl.Datetime(time_zone="UTC"),
        "accepted_at": pl.Datetime(time_zone="UTC"),
        "transaction_code": pl.String,
        "shares": pl.Float64,
        "price": pl.Float64,
        "insider_id": pl.String,
        "is_10b5_1": pl.Boolean,
        "transaction_id": pl.String,
        "accession_number": pl.String,
    }


def _coverage_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "source": pl.String,
        "coverage_start": pl.Date,
        "coverage_end": pl.Date,
        "status": pl.String,
    }


def _short_interest_schema() -> dict[str, pl.DataType]:
    return {
        "symbol": pl.String,
        "settlement_date": pl.Date,
        "publication_date": pl.Date,
        "available_at": pl.Date,
        "short_interest": pl.Int64,
        "stock_split_flag": pl.Boolean,
        "revision_flag": pl.Boolean,
    }


def _snapshot_digest(
    resources: Iterable[tuple[str, str, str | None]],
    *,
    end: date,
    lookback_sessions: int,
) -> str:
    manifest = {
        "version": _SNAPSHOT_VERSION,
        "supported_start": SUPPORTED_START.isoformat(),
        "end": end.isoformat(),
        "lookback_sessions": lookback_sessions,
        "resources": [
            {
                "url": url,
                "sha256": digest,
                **({"observed_at": observed_at} if observed_at is not None else {}),
            }
            for url, digest, observed_at in sorted(
                set(resources),
                key=lambda resource: (
                    resource[0],
                    resource[1],
                    resource[2] or "",
                ),
            )
        ],
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _latest_cached(
    source_dir: Path,
    suffix: str,
    *,
    known_at: date,
) -> tuple[Path, date] | None:
    if not source_dir.is_dir():
        return None
    candidates: list[tuple[date, int, Path]] = []
    for path in source_dir.glob(f"observed_at=*/*{suffix}"):
        match = re.fullmatch(
            r"observed_at=(\d{4}-\d{2}-\d{2})",
            path.parent.name,
        )
        if not path.is_file() or match is None or re.fullmatch(r"[0-9a-f]{64}", path.stem) is None:
            continue
        try:
            observed_at = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if observed_at <= known_at:
            candidates.append((observed_at, path.stat().st_mtime_ns, path))
    if not candidates:
        return None
    observed_at, _, path = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2].name),
    )
    return path, observed_at


def _read_cached(path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise OfficialResearchDataError(
            f"could not read cached official resource {path}: {exc}"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if path.stem != digest:
        raise OfficialResearchDataError(f"cached official resource failed its content hash: {path}")
    return payload, digest


def _atomic_content_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing, _ = _read_cached(path)
        if existing != payload:
            raise OfficialResearchDataError(f"content-addressed cache collision: {path}")
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
            existing, _ = _read_cached(path)
            if existing != payload:
                raise OfficialResearchDataError(f"content-addressed cache collision: {path}")
    except OSError as exc:
        raise OfficialResearchDataError(
            f"could not cache official research resource: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _wait_for_sec_slot() -> None:
    global _LAST_SEC_REQUEST
    with _SEC_HTTP_LOCK:
        elapsed = time.monotonic() - _LAST_SEC_REQUEST
        if elapsed < _SEC_REQUEST_INTERVAL_SECONDS:
            time.sleep(_SEC_REQUEST_INTERVAL_SECONDS - elapsed)
        _LAST_SEC_REQUEST = time.monotonic()
