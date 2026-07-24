"""Point-in-time SEC corporate-action status for ranking exclusions.

The public entry points in this module intentionally accept an injectable
``fetcher``.  Production calls use a throttled stdlib HTTP client, while tests
and offline research can provide an exact SEC response snapshot.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from alphascreener.data.locking import exclusive_file_lock
from alphascreener.data.paths import get_data_home
from alphascreener.market_calendar import market_session_close

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKER_TEXT_URL = "https://www.sec.gov/include/ticker.txt"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSION_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# Pending public-company transactions normally produce another filing well
# inside this interval.  Limiting document inspection avoids downloading every
# 8-K in a long-lived issuer's history while still covering unusually slow
# regulatory reviews.
TRANSACTION_LOOKBACK_DAYS = 2 * 366
CACHE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_SEC_USER_AGENT = "Qanora alpha-screener/0.2 lijinze0118@live.com"

_NEW_YORK = ZoneInfo("America/New_York")
_HTTP_MIN_INTERVAL_SECONDS = 0.125
_HTTP_LOCK = threading.Lock()
_LAST_HTTP_REQUEST = 0.0

_TARGET_TRANSACTION_FORMS = {
    "DEFM14A",
    "PREM14A",
    "SC 13E3",
    "SC 14D9",
}
_IGNORED_TRANSACTION_FORMS = {
    "SC TO",
    "SC TO-I",
    "SC TO-T",
    "S-4",
}
_TEXT_FORMS = {"8-K", "6-K"}
_RELEVANT_FORMS = _TARGET_TRANSACTION_FORMS | _IGNORED_TRANSACTION_FORMS | _TEXT_FORMS
_RELEVANT_8K_ITEMS = {"1.01", "1.02", "2.01", "5.01", "8.01"}
_ACTIVATION_SEARCH_QUERIES = (
    (
        '"will be acquired by" OR "would be acquired by" OR '
        '"will acquire company" OR "will acquire the company" OR '
        '"would acquire company" OR "would acquire the company"'
    ),
    (
        '"merge with and into" OR "wholly owned subsidiary" OR '
        '"wholly-owned subsidiary" OR '
        '"converted into the right to receive" OR '
        '"converted into right to receive"'
    ),
)
_TERMINAL_SEARCH_QUERIES = (
    (
        '"mutually agreed to terminate" OR "merger agreement" OR '
        '"tender offer" OR "going private transaction" OR '
        '"going-private transaction" OR '
        '"termination of proposed transaction"'
    ),
    (
        '"completed the merger" OR "completed merger" OR '
        '"consummated the merger" OR "consummated merger" OR '
        '"merger was completed" OR "merger has been completed" OR '
        '"merger was consummated" OR "merger has been consummated" OR '
        '"merger became effective" OR '
        '"accepted for payment"'
    ),
)
_SEARCH_QUERIES = _ACTIVATION_SEARCH_QUERIES + _TERMINAL_SEARCH_QUERIES
_SEARCH_PAGE_SIZE = 100
_SEARCH_CIK_BATCH_SIZE = 30
_SEARCH_MAX_RESULTS_PER_QUERY = 10_000
# EFTS can lag EDGAR submissions.  Inspect recent text filings directly so an
# index delay can never turn a same-week signed deal into a false negative.
_SEARCH_INDEX_LAG_GUARD_DAYS = 7

_POSITIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:the\s+)?company\s+(?:will|would)\s+be\s+acquired\s+by\b",
        r"\bparent.{0,240}\b(?:will|would)\s+acquire\s+(?:the\s+)?company\b",
        r"\bmerger\s+sub(?:sidiary)?.{0,240}\bmerge\s+with\s+and\s+into\s+"
        r"(?:the\s+)?company\b",
        r"\b(?:the\s+)?company\s+(?:will|would)\s+become\s+(?:an?\s+)?"
        r"(?:direct\s+or\s+indirect\s+)?wholly[\s-]+owned\s+subsidiary\b",
        r"\beach\s+(?:outstanding\s+)?share.{0,320}\bconverted\s+into\s+"
        r"(?:the\s+)?right\s+to\s+receive\b",
    )
)
_TERMINATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bmutually\s+agreed\s+to\s+terminate.{0,160}"
        r"\b(?:merger|transaction|tender\s+offer|acquisition)\b",
        r"\b(?:merger\s+agreement|tender\s+offer|going[- ]private\s+transaction)"
        r"\s+(?:has|have|had|was|were|is|are)\s+(?:been\s+)?"
        r"(?:terminated|withdrawn)\b",
        r"\b(?:terminated|terminate|withdrew|withdrawn?)\s+(?:the\s+)?"
        r"(?:merger\s+agreement|tender\s+offer|going[- ]private\s+transaction)\b",
        r"\btermination\s+of\s+(?:the\s+)?"
        r"(?:merger\s+agreement|tender\s+offer|proposed\s+transaction)\b",
    )
)
_COMPLETION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:completed|consummated)\s+(?:the\s+)?merger\b",
        r"\bmerger\s+(?:was|has\s+been)\s+(?:completed|consummated)\b",
        r"\bmerger\s+became\s+effective\b",
        r"\btender\s+offer.{0,240}\bexpired\b.{0,240}\baccepted\s+for\s+payment\b",
        r"\baccepted\s+for\s+payment.{0,240}\btendered\s+shares\b",
    )
)


class CorporateActionDataError(RuntimeError):
    """Raised when SEC data cannot support a complete deterministic result."""


@dataclass(frozen=True)
class CorporateActionStatus:
    """Point-in-time definitive-transaction state for one ticker."""

    ticker: str
    cik: str
    decision_at: datetime
    state: str
    reason: str
    filing_form: str | None = None
    accession_number: str | None = None
    accepted_at: datetime | None = None
    filing_url: str | None = None

    @property
    def under_definitive_transaction(self) -> bool:
        """Whether ranking should treat this ticker as transaction-bound."""
        return self.state == "active"

    @property
    def exclude_from_ranking(self) -> bool:
        """Alias expressing the intended use by the candidate ranker."""
        return self.under_definitive_transaction


@dataclass(frozen=True)
class _Filing:
    accession_number: str
    filing_date: date
    accepted_at: datetime
    form: str
    is_amendment: bool
    primary_document: str
    items: str


Fetcher = Callable[[str], bytes]


def corporate_action_statuses(
    tickers: Iterable[str],
    decision_at: date | datetime,
    *,
    data_home: Path | None = None,
    fetcher: Fetcher | None = None,
    user_agent: str | None = None,
) -> dict[str, CorporateActionStatus]:
    """Return SEC definitive-transaction status for ``tickers``.

    A :class:`date` means the regular 16:00 New York close.  Callers that
    evaluate an early-close session or an intraday decision should pass an
    explicit timezone-aware :class:`datetime`.

    The function is deliberately fail-closed: a missing ticker mapping,
    malformed submissions response, missing acceptance timestamp, or failed
    primary-document fetch raises :class:`CorporateActionDataError` rather
    than silently treating the security as transaction-free.
    """
    normalized = tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers))
    if not normalized:
        return {}
    cutoff = _decision_cutoff(decision_at)
    cache_root = (data_home or get_data_home()) / "data" / "sec"
    use_search_prefilter = fetcher is None
    network_fetcher = fetcher or _SecHttpFetcher(user_agent=user_agent)
    try:
        mapping_payload = _load_json_resource(
            SEC_TICKER_URL,
            cache_root / "company_tickers.json",
            network_fetcher,
            max_age=CACHE_TTL_SECONDS,
            description="SEC ticker map",
        )
        ticker_to_cik = _ticker_cik_map(mapping_payload)
    except CorporateActionDataError as primary_error:
        try:
            fallback_payload = _load_resource(
                SEC_TICKER_TEXT_URL,
                cache_root / "ticker.txt",
                network_fetcher,
                max_age=CACHE_TTL_SECONDS,
                description="SEC ticker/CIK text map",
            )
            ticker_to_cik = _ticker_cik_text_map(fallback_payload)
        except CorporateActionDataError as fallback_error:
            raise CorporateActionDataError(
                "SEC ticker mapping is unavailable from both official sources: "
                f"JSON={primary_error}; text={fallback_error}"
            ) from fallback_error
    missing = sorted(set(normalized) - set(ticker_to_cik))
    if missing:
        raise CorporateActionDataError("SEC ticker map has no CIK for: " + ", ".join(missing))

    by_cik: dict[str, list[str]] = {}
    for ticker in normalized:
        by_cik.setdefault(ticker_to_cik[ticker], []).append(ticker)
    filings_by_cik = {
        cik: _load_filings(
            cik,
            cutoff,
            cache_root=cache_root,
            fetcher=network_fetcher,
        )
        for cik in by_cik
    }
    searchable_ciks = tuple(
        cik
        for cik, filings in filings_by_cik.items()
        if any(filing.form in _TEXT_FORMS for filing in filings)
    )
    activation_documents_by_cik = (
        _load_text_search_documents(
            searchable_ciks,
            cutoff,
            cache_root=cache_root,
            fetcher=network_fetcher,
            search_queries=_ACTIVATION_SEARCH_QUERIES,
        )
        if use_search_prefilter and searchable_ciks
        else {}
    )

    results: dict[str, CorporateActionStatus] = {}
    for cik, cik_tickers in by_cik.items():
        filings = filings_by_cik[cik]
        text_documents = (
            _with_search_lag_guard(
                filings,
                cutoff,
                activation_documents_by_cik.get(cik, {}),
            )
            if use_search_prefilter
            else None
        )
        for ticker in cik_tickers:
            results[ticker] = _classify_ticker(
                ticker,
                cik,
                cutoff,
                filings,
                text_documents=text_documents,
                cache_root=cache_root,
                fetcher=network_fetcher,
            )

    active_ciks = tuple(
        cik
        for cik, cik_tickers in by_cik.items()
        if any(results[ticker].under_definitive_transaction for ticker in cik_tickers)
    )
    if use_search_prefilter and active_ciks:
        terminal_documents_by_cik = _load_text_search_documents(
            active_ciks,
            cutoff,
            cache_root=cache_root,
            fetcher=network_fetcher,
            search_queries=_TERMINAL_SEARCH_QUERIES,
        )
        for cik in active_ciks:
            filings = filings_by_cik[cik]
            text_documents = _with_search_lag_guard(
                filings,
                cutoff,
                _merge_search_documents(
                    activation_documents_by_cik.get(cik, {}),
                    terminal_documents_by_cik.get(cik, {}),
                ),
            )
            for ticker in by_cik[cik]:
                results[ticker] = _classify_ticker(
                    ticker,
                    cik,
                    cutoff,
                    filings,
                    text_documents=text_documents,
                    cache_root=cache_root,
                    fetcher=network_fetcher,
                )
    return results


def filter_definitive_transactions(
    tickers: Iterable[str],
    decision_at: date | datetime,
    *,
    data_home: Path | None = None,
    fetcher: Fetcher | None = None,
    user_agent: str | None = None,
) -> tuple[tuple[str, ...], dict[str, CorporateActionStatus]]:
    """Return retained tickers and the complete auditable SEC status mapping."""
    normalized = tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers))
    statuses = corporate_action_statuses(
        normalized,
        decision_at,
        data_home=data_home,
        fetcher=fetcher,
        user_agent=user_agent,
    )
    retained = tuple(ticker for ticker in normalized if not statuses[ticker].exclude_from_ranking)
    return retained, statuses


class _SecHttpFetcher:
    """Small SEC-friendly stdlib HTTP client with process-wide throttling."""

    def __init__(self, *, user_agent: str | None = None, timeout: float = 30.0):
        configured = user_agent or os.environ.get(
            "ALPHASCREENER_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT
        )
        if not configured.strip():
            raise ValueError("SEC user agent must not be empty")
        self._user_agent = configured
        self._timeout = timeout

    def __call__(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(3):
            _wait_for_sec_request_slot()
            request = Request(
                url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    payload = response.read()
                if not payload:
                    raise CorporateActionDataError(f"SEC returned an empty response for {url}")
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 10.0))
            except (TimeoutError, URLError, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise CorporateActionDataError(f"could not fetch SEC resource {url}: {last_error}")


def _wait_for_sec_request_slot() -> None:
    global _LAST_HTTP_REQUEST
    with _HTTP_LOCK:
        throttle_root = get_data_home() / "data" / "sec"
        with exclusive_file_lock(throttle_root / ".http.lock"):
            timestamp_path = throttle_root / ".last_http_request"
            try:
                last_wall_time = float(timestamp_path.read_text())
            except (FileNotFoundError, OSError, ValueError):
                last_wall_time = 0.0
            now_wall_time = time.time()
            cross_process_delay = _HTTP_MIN_INTERVAL_SECONDS - (now_wall_time - last_wall_time)
            process_delay = _HTTP_MIN_INTERVAL_SECONDS - (time.monotonic() - _LAST_HTTP_REQUEST)
            delay = max(cross_process_delay, process_delay)
            if delay > 0:
                time.sleep(delay)
            _LAST_HTTP_REQUEST = time.monotonic()
            timestamp_path.write_text(str(time.time()))


def _load_filings(
    cik: str,
    cutoff: datetime,
    *,
    cache_root: Path,
    fetcher: Fetcher,
) -> tuple[_Filing, ...]:
    padded_cik = cik.zfill(10)
    current_new_york_date = datetime.now(_NEW_YORK).date()
    minimum_mtime = cutoff.timestamp() if cutoff.date() == current_new_york_date else None
    root = _load_json_resource(
        SEC_SUBMISSIONS_URL.format(cik=padded_cik),
        cache_root / "submissions" / f"CIK{padded_cik}.json",
        fetcher,
        max_age=CACHE_TTL_SECONDS,
        minimum_mtime=minimum_mtime,
        description=f"SEC submissions for CIK {padded_cik}",
    )
    if not isinstance(root, Mapping):
        raise CorporateActionDataError(f"SEC submissions for CIK {padded_cik} are not an object")
    filings_node = root.get("filings")
    if not isinstance(filings_node, Mapping):
        raise CorporateActionDataError(
            f"SEC submissions for CIK {padded_cik} have no filings object"
        )
    recent = filings_node.get("recent")
    if not isinstance(recent, Mapping):
        raise CorporateActionDataError(
            f"SEC submissions for CIK {padded_cik} have no recent filing table"
        )

    tables: list[tuple[str, Mapping[str, Any]]] = [("recent", recent)]
    lookback_start = cutoff.date() - timedelta(days=TRANSACTION_LOOKBACK_DAYS)
    shard_metadata = filings_node.get("files", [])
    if not isinstance(shard_metadata, list):
        raise CorporateActionDataError(
            f"SEC submissions for CIK {padded_cik} have invalid history metadata"
        )
    for entry in shard_metadata:
        if not isinstance(entry, Mapping):
            raise CorporateActionDataError(
                f"SEC submissions for CIK {padded_cik} have invalid history metadata"
            )
        name = entry.get("name")
        filing_from = _parse_date(entry.get("filingFrom"), f"{padded_cik} filingFrom")
        filing_to = _parse_date(entry.get("filingTo"), f"{padded_cik} filingTo")
        if not _is_safe_basename(name):
            raise CorporateActionDataError(
                f"SEC submissions for CIK {padded_cik} have an invalid history filename"
            )
        if filing_to < lookback_start or filing_from > cutoff.date():
            continue
        shard = _load_json_resource(
            SEC_SUBMISSION_FILE_URL.format(name=quote(name)),
            cache_root / "submissions" / name,
            fetcher,
            max_age=None,
            description=f"SEC submission history {name}",
        )
        if not isinstance(shard, Mapping):
            raise CorporateActionDataError(f"SEC submission history {name} is not an object")
        tables.append((name, shard))

    records: dict[str, _Filing] = {}
    for table_name, table in tables:
        for filing in _parse_filing_table(table, table_name, cutoff, lookback_start):
            existing = records.get(filing.accession_number)
            if existing is not None and existing != filing:
                raise CorporateActionDataError(
                    f"conflicting SEC metadata for accession {filing.accession_number}"
                )
            records[filing.accession_number] = filing
    return tuple(
        sorted(
            records.values(),
            key=lambda filing: (filing.accepted_at, filing.accession_number),
        )
    )


def _load_text_search_documents(
    ciks: Iterable[str],
    cutoff: datetime,
    *,
    cache_root: Path,
    fetcher: Fetcher,
    search_queries: tuple[str, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return validated EFTS candidate documents for a batch of issuer CIKs."""
    normalized_ciks = tuple(sorted({cik.zfill(10) for cik in ciks}))
    if not normalized_ciks:
        return {}
    query_version = _search_query_version(search_queries)
    search_start = _search_window_start(cutoff.date())
    cached: dict[str, dict[str, tuple[str, ...]]] = {}
    missing: list[str] = []
    for cik in normalized_ciks:
        path = _search_cache_path(
            cache_root,
            cik,
            cutoff.date(),
            query_version=query_version,
        )
        documents = _read_search_cache(
            path,
            required_start=search_start,
            required_end=cutoff.date(),
            query_version=query_version,
        )
        if documents is None:
            missing.append(cik)
        else:
            cached[cik] = documents

    if missing:
        fetched: dict[str, dict[str, tuple[str, ...]]] = {}
        for offset in range(0, len(missing), _SEARCH_CIK_BATCH_SIZE):
            batch = tuple(missing[offset : offset + _SEARCH_CIK_BATCH_SIZE])
            fetched.update(
                _fetch_text_search_documents(
                    batch,
                    search_start=search_start,
                    search_end=cutoff.date(),
                    fetcher=fetcher,
                    search_queries=search_queries,
                )
            )
        for cik in missing:
            documents = fetched[cik]
            _write_search_cache(
                _search_cache_path(
                    cache_root,
                    cik,
                    cutoff.date(),
                    query_version=query_version,
                ),
                search_start=search_start,
                search_end=cutoff.date(),
                documents=documents,
                query_version=query_version,
            )
            cached[cik] = documents
    return cached


def _fetch_text_search_documents(
    ciks: tuple[str, ...],
    *,
    search_start: date,
    search_end: date,
    fetcher: Fetcher,
    search_queries: tuple[str, ...] = _SEARCH_QUERIES,
) -> dict[str, dict[str, tuple[str, ...]]]:
    documents: dict[str, dict[str, set[str]]] = {cik: {} for cik in ciks}
    requested_ciks = set(ciks)
    for search_query in search_queries:
        offset = 0
        expected_total: int | None = None
        seen_hits: set[tuple[tuple[str, ...], str, str]] = set()
        while expected_total is None or offset < expected_total:
            query = urlencode(
                {
                    "q": search_query,
                    "dateRange": "custom",
                    "startdt": search_start.isoformat(),
                    "enddt": search_end.isoformat(),
                    "forms": "8-K,6-K",
                    "ciks": ",".join(ciks),
                    "from": offset,
                    "size": _SEARCH_PAGE_SIZE,
                }
            )
            description = "SEC full-text transaction search for CIKs " + ", ".join(ciks)
            payload = _fetch_json_payload(
                f"{SEC_SEARCH_URL}?{query}",
                fetcher,
                description=description,
            )
            total, hits = _parse_search_page(
                payload,
                requested_ciks=requested_ciks,
            )
            if expected_total is None:
                expected_total = total
                if expected_total > _SEARCH_MAX_RESULTS_PER_QUERY:
                    raise CorporateActionDataError(
                        "SEC full-text transaction search exceeded its complete-result limit"
                    )
            elif total != expected_total:
                raise CorporateActionDataError(
                    "SEC full-text transaction search total changed during pagination"
                )
            if not hits and offset < expected_total:
                raise CorporateActionDataError(
                    "SEC full-text transaction search returned an incomplete page"
                )
            for hit_ciks, accession, document in hits:
                hit_key = (tuple(sorted(hit_ciks)), accession, document)
                if hit_key in seen_hits:
                    raise CorporateActionDataError(
                        "SEC full-text transaction search returned duplicate hits"
                    )
                seen_hits.add(hit_key)
                for cik in hit_ciks:
                    documents[cik].setdefault(accession, set()).add(document)
            offset += len(hits)

    return {
        cik: {accession: tuple(sorted(names)) for accession, names in by_accession.items()}
        for cik, by_accession in documents.items()
    }


def _parse_search_page(
    payload: Any,
    *,
    requested_ciks: set[str],
) -> tuple[int, list[tuple[set[str], str, str]]]:
    if not isinstance(payload, Mapping):
        raise CorporateActionDataError("SEC full-text search response is not an object")
    if payload.get("timed_out") is not False:
        raise CorporateActionDataError("SEC full-text transaction search timed out")
    shards = payload.get("_shards")
    if not isinstance(shards, Mapping) or shards.get("failed") != 0:
        raise CorporateActionDataError("SEC full-text transaction search has failed shards")
    hits_node = payload.get("hits")
    if not isinstance(hits_node, Mapping):
        raise CorporateActionDataError("SEC full-text search response has no hits")
    total = hits_node.get("total")
    hits = hits_node.get("hits")
    if (
        not isinstance(total, Mapping)
        or not isinstance(total.get("value"), int)
        or total.get("value") < 0
        or total.get("relation") != "eq"
        or not isinstance(hits, list)
        or len(hits) > _SEARCH_PAGE_SIZE
    ):
        raise CorporateActionDataError("SEC full-text search response has incomplete hit metadata")

    parsed_hits: list[tuple[set[str], str, str]] = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            raise CorporateActionDataError(f"SEC full-text search hit {index} is not an object")
        source = hit.get("_source")
        identifier = hit.get("_id")
        if not isinstance(source, Mapping) or not isinstance(identifier, str):
            raise CorporateActionDataError(f"SEC full-text search hit {index} is incomplete")
        accession = source.get("adsh")
        source_ciks = source.get("ciks")
        hit_ciks = (
            set(source_ciks).intersection(requested_ciks)
            if isinstance(source_ciks, list)
            and all(isinstance(value, str) for value in source_ciks)
            else set()
        )
        if (
            not isinstance(accession, str)
            or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
            or not hit_ciks
        ):
            raise CorporateActionDataError(
                f"SEC full-text search hit {index} has invalid issuer metadata"
            )
        prefix = f"{accession}:"
        if not identifier.startswith(prefix):
            raise CorporateActionDataError(
                f"SEC full-text search hit {index} has an invalid document id"
            )
        document = identifier.removeprefix(prefix)
        if not _is_safe_basename(document):
            raise CorporateActionDataError(
                f"SEC full-text search hit {index} has an unsafe document name"
            )
        parsed_hits.append((hit_ciks, accession, document))
    return int(total["value"]), parsed_hits


def _search_window_start(cutoff: date) -> date:
    quarter_month = ((cutoff.month - 1) // 3) * 3 + 1
    quarter_start = date(cutoff.year, quarter_month, 1)
    return quarter_start - timedelta(days=TRANSACTION_LOOKBACK_DAYS)


def _search_query_version(search_queries: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(search_queries).encode()).hexdigest()[:12]


def _search_cache_path(
    cache_root: Path,
    cik: str,
    cutoff: date,
    *,
    query_version: str,
) -> Path:
    quarter = (cutoff.month - 1) // 3 + 1
    return (
        cache_root / "search" / query_version / f"CIK{cik.zfill(10)}-{cutoff.year}-Q{quarter}.json"
    )


def _read_search_cache(
    path: Path,
    *,
    required_start: date,
    required_end: date,
    query_version: str,
) -> dict[str, tuple[str, ...]] | None:
    if not path.is_file():
        return None
    if (datetime.now(_NEW_YORK).date() - required_end).days <= _SEARCH_INDEX_LAG_GUARD_DAYS and max(
        0.0, time.time() - path.stat().st_mtime
    ) > CACHE_TTL_SECONDS:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("query_version") != query_version
            or date.fromisoformat(payload["search_start"]) > required_start
            or date.fromisoformat(payload["search_end"]) < required_end
            or not isinstance(payload.get("documents"), Mapping)
        ):
            return None
        documents: dict[str, tuple[str, ...]] = {}
        for accession, names in payload["documents"].items():
            if (
                not isinstance(accession, str)
                or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                or not isinstance(names, list)
                or not all(_is_safe_basename(name) for name in names)
            ):
                return None
            documents[accession] = tuple(names)
        return documents
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_search_cache(
    path: Path,
    *,
    search_start: date,
    search_end: date,
    documents: Mapping[str, tuple[str, ...]],
    query_version: str,
) -> None:
    payload = json.dumps(
        {
            "query_version": query_version,
            "search_start": search_start.isoformat(),
            "search_end": search_end.isoformat(),
            "documents": documents,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    except OSError as exc:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise CorporateActionDataError(
            f"could not cache SEC full-text search results: {exc}"
        ) from exc


def _fetch_json_payload(
    url: str,
    fetcher: Fetcher,
    *,
    description: str,
) -> Any:
    try:
        payload = fetcher(url)
    except CorporateActionDataError:
        raise
    except Exception as exc:
        raise CorporateActionDataError(f"could not fetch {description}: {exc}") from exc
    if not isinstance(payload, bytes) or not payload:
        raise CorporateActionDataError(f"{description} fetch returned no bytes")
    _reject_sec_error_payload(payload, description)
    try:
        return json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorporateActionDataError(f"{description} is not valid JSON: {exc}") from exc


def _with_search_lag_guard(
    filings: tuple[_Filing, ...],
    cutoff: datetime,
    documents: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    guarded = dict(documents)
    for filing in filings:
        if (
            filing.form in _TEXT_FORMS
            and (cutoff.date() - filing.filing_date).days <= _SEARCH_INDEX_LAG_GUARD_DAYS
        ):
            guarded.setdefault(filing.accession_number, ())
    return guarded


def _merge_search_documents(
    *mappings: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = {}
    for mapping in mappings:
        for accession, documents in mapping.items():
            merged.setdefault(accession, set()).update(documents)
    return {accession: tuple(sorted(documents)) for accession, documents in merged.items()}


def _parse_filing_table(
    table: Mapping[str, Any],
    table_name: str,
    cutoff: datetime,
    lookback_start: date,
) -> tuple[_Filing, ...]:
    required_columns = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    )
    columns: dict[str, list[Any]] = {}
    for column in required_columns:
        values = table.get(column)
        if not isinstance(values, list):
            raise CorporateActionDataError(
                f"SEC filing table {table_name} has no valid {column} column"
            )
        columns[column] = values
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise CorporateActionDataError(f"SEC filing table {table_name} has misaligned columns")
    row_count = lengths.pop()
    raw_items = table.get("items")
    if raw_items is None:
        items = [""] * row_count
    elif not isinstance(raw_items, list) or len(raw_items) != row_count:
        raise CorporateActionDataError(
            f"SEC filing table {table_name} has a misaligned items column"
        )
    else:
        items = raw_items

    filings: list[_Filing] = []
    for index in range(row_count):
        form, is_amendment = _parse_form(columns["form"][index])
        if form not in _RELEVANT_FORMS:
            continue
        filing_date = _parse_date(
            columns["filingDate"][index],
            f"{table_name} filingDate row {index}",
        )
        if filing_date < lookback_start or filing_date > cutoff.date():
            continue
        accepted_at = _parse_acceptance_datetime(
            columns["acceptanceDateTime"][index],
            f"{table_name} acceptanceDateTime row {index}",
        )
        if accepted_at > cutoff:
            continue
        item_text = str(items[index] or "")
        if form == "8-K" and item_text:
            item_numbers = set(re.findall(r"\d+\.\d+", item_text))
            if item_numbers and not item_numbers.intersection(_RELEVANT_8K_ITEMS):
                continue
        accession = columns["accessionNumber"][index]
        document = columns["primaryDocument"][index]
        if not isinstance(accession, str) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise CorporateActionDataError(
                f"SEC filing table {table_name} row {index} has invalid accessionNumber"
            )
        if not isinstance(document, str) or not _is_safe_basename(document):
            raise CorporateActionDataError(
                f"SEC filing table {table_name} row {index} has invalid primaryDocument"
            )
        filings.append(
            _Filing(
                accession_number=accession,
                filing_date=filing_date,
                accepted_at=accepted_at,
                form=form,
                is_amendment=is_amendment,
                primary_document=document,
                items=item_text,
            )
        )
    return tuple(filings)


def _classify_ticker(
    ticker: str,
    cik: str,
    cutoff: datetime,
    filings: tuple[_Filing, ...],
    *,
    text_documents: Mapping[str, tuple[str, ...]] | None,
    cache_root: Path,
    fetcher: Fetcher,
) -> CorporateActionStatus:
    state = "none"
    reason = "no_definitive_transaction_filing"
    state_filing: _Filing | None = None
    state_url: str | None = None
    for filing in filings:
        if filing.form in _IGNORED_TRANSACTION_FORMS:
            continue
        if (
            filing.form in _TEXT_FORMS
            and text_documents is not None
            and filing.accession_number not in text_documents
        ):
            continue
        url = _filing_url(cik, filing)
        documents = [filing.primary_document]
        if state == "active" and filing.form in _TEXT_FORMS and text_documents is not None:
            documents.extend(text_documents.get(filing.accession_number, ()))
        filing_texts: list[str] = []
        for document in dict.fromkeys(documents):
            document_url = _filing_document_url(cik, filing, document)
            payload = _load_resource(
                document_url,
                cache_root
                / "filings"
                / cik.zfill(10)
                / filing.accession_number.replace("-", "")
                / document,
                fetcher,
                max_age=None,
                description=f"SEC filing {filing.accession_number} document {document}",
            )
            filing_texts.append(_document_text(payload, filing.accession_number))
        # An acquirer often attaches the negotiated merger agreement to its
        # own 8-K.  Inside that exhibit, "Company" means the target, not the
        # filing issuer.  Therefore only the issuer's primary narrative may
        # activate a text-form exclusion.  Supplemental EFTS hits can still
        # confirm completion or termination of an already-active target.
        event = _filing_event(filing, filing_texts[0], current_state=state)
        if event is None and state == "active":
            for supplemental_text in filing_texts[1:]:
                event = _terminal_event(supplemental_text)
                if event is not None:
                    break
        if event is None:
            continue
        state, reason = event
        state_filing = filing
        state_url = url

    if state == "unknown":
        raise CorporateActionDataError(
            f"SEC transaction state is incomplete for {ticker}: {reason}"
        )
    return CorporateActionStatus(
        ticker=ticker,
        cik=cik.zfill(10),
        decision_at=cutoff,
        state=state,
        reason=reason,
        filing_form=state_filing.form if state_filing else None,
        accession_number=state_filing.accession_number if state_filing else None,
        accepted_at=state_filing.accepted_at if state_filing else None,
        filing_url=state_url,
    )


def _filing_event(
    filing: _Filing,
    text: str,
    *,
    current_state: str,
) -> tuple[str, str] | None:
    if current_state == "active":
        terminal = _terminal_event(text)
        if terminal is not None:
            return terminal
    if filing.form == "SC 14D9":
        if filing.is_amendment:
            if current_state == "none":
                return "unknown", "transaction_amendment_without_base_filing"
            return None
        return "active", "target_tender_offer_form_SC_14D9"
    if any(pattern.search(text) for pattern in _POSITIVE_PATTERNS):
        return "active", "definitive_target_transaction_language"
    return None


def _terminal_event(text: str) -> tuple[str, str] | None:
    if any(pattern.search(text) for pattern in _COMPLETION_PATTERNS):
        return "completed", "transaction_completed"
    if any(pattern.search(text) for pattern in _TERMINATION_PATTERNS):
        return "terminated", "transaction_terminated_or_withdrawn"
    return None


def _filing_url(cik: str, filing: _Filing) -> str:
    return _filing_document_url(cik, filing, filing.primary_document)


def _filing_document_url(cik: str, filing: _Filing, document: str) -> str:
    return SEC_ARCHIVE_URL.format(
        cik=str(int(cik)),
        accession=filing.accession_number.replace("-", ""),
        document=quote(document),
    )


def _ticker_cik_map(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise CorporateActionDataError("SEC ticker map is not an object")
    mapping: dict[str, str] = {}
    for row in payload.values():
        if not isinstance(row, Mapping):
            raise CorporateActionDataError("SEC ticker map contains an invalid row")
        ticker = row.get("ticker")
        cik = row.get("cik_str")
        if not isinstance(ticker, str) or not isinstance(cik, (int, str)):
            raise CorporateActionDataError("SEC ticker map contains an incomplete row")
        cik_digits = str(cik)
        if not cik_digits.isdigit() or len(cik_digits) > 10:
            raise CorporateActionDataError("SEC ticker map contains an invalid CIK")
        normalized = _normalize_ticker(ticker)
        padded = cik_digits.zfill(10)
        previous = mapping.get(normalized)
        if previous is not None and previous != padded:
            raise CorporateActionDataError(
                f"SEC ticker map contains conflicting CIKs for {normalized}"
            )
        mapping[normalized] = padded
    if not mapping:
        raise CorporateActionDataError("SEC ticker map is empty")
    return mapping


def _ticker_cik_text_map(payload: bytes) -> dict[str, str]:
    try:
        contents = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CorporateActionDataError(f"SEC ticker/CIK text map is not UTF-8: {exc}") from exc
    mapping: dict[str, str] = {}
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise CorporateActionDataError(
                f"SEC ticker/CIK text map row {line_number} is malformed"
            )
        ticker, cik = parts
        if not cik.isdigit() or len(cik) > 10:
            raise CorporateActionDataError(
                f"SEC ticker/CIK text map row {line_number} has an invalid CIK"
            )
        normalized = _normalize_ticker(ticker)
        padded = cik.zfill(10)
        previous = mapping.get(normalized)
        if previous is not None and previous != padded:
            raise CorporateActionDataError(
                f"SEC ticker/CIK text map contains conflicting CIKs for {normalized}"
            )
        mapping[normalized] = padded
    if not mapping:
        raise CorporateActionDataError("SEC ticker/CIK text map is empty")
    return mapping


def _load_json_resource(
    url: str,
    path: Path,
    fetcher: Fetcher,
    *,
    max_age: float | None,
    minimum_mtime: float | None = None,
    description: str,
) -> Any:
    payload = _load_resource(
        url,
        path,
        fetcher,
        max_age=max_age,
        minimum_mtime=minimum_mtime,
        description=description,
    )
    try:
        return json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorporateActionDataError(f"{description} is not valid JSON: {exc}") from exc


def _load_resource(
    url: str,
    path: Path,
    fetcher: Fetcher,
    *,
    max_age: float | None,
    minimum_mtime: float | None = None,
    description: str,
) -> bytes:
    cached_mtime = path.stat().st_mtime if path.is_file() else None
    if (
        cached_mtime is not None
        and (max_age is None or max(0.0, time.time() - cached_mtime) <= max_age)
        and (minimum_mtime is None or cached_mtime >= minimum_mtime)
    ):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise CorporateActionDataError(f"could not read cached {description}: {exc}") from exc
        if payload:
            return payload
    try:
        payload = fetcher(url)
    except CorporateActionDataError:
        raise
    except Exception as exc:
        raise CorporateActionDataError(f"could not fetch {description}: {exc}") from exc
    if not isinstance(payload, bytes) or not payload:
        raise CorporateActionDataError(f"{description} fetch returned no bytes")
    _reject_sec_error_payload(payload, description)
    if minimum_mtime is not None and time.time() < minimum_mtime:
        raise CorporateActionDataError(
            f"{description} cannot be complete before the decision cutoff"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    except OSError as exc:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise CorporateActionDataError(f"could not cache {description}: {exc}") from exc
    return payload


def _reject_sec_error_payload(payload: bytes, description: str) -> None:
    lowered = payload[:20_000].lower()
    sec_error_markers = (
        b"your request originates from an undeclared automated tool",
        b"request rate threshold exceeded",
        b"your request has been identified as part of a network of automated tools",
        b"please declare your traffic",
        b"access denied",
    )
    if any(marker in lowered for marker in sec_error_markers):
        raise CorporateActionDataError(f"SEC blocked the request for {description}")


def _document_text(payload: bytes, accession: str) -> str:
    try:
        contents = payload.decode("utf-8")
    except UnicodeDecodeError:
        contents = payload.decode("latin-1")
    contents = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        " ",
        contents,
        flags=re.IGNORECASE | re.DOTALL,
    )
    contents = re.sub(r"<[^>]+>", " ", contents)
    contents = html.unescape(contents)
    normalized = " ".join(contents.split())
    if not normalized:
        raise CorporateActionDataError(
            f"SEC filing {accession} primary document contains no readable text"
        )
    return normalized


def _parse_form(value: Any) -> tuple[str, bool]:
    if not isinstance(value, str) or not value.strip():
        raise CorporateActionDataError("SEC filing table contains an invalid form")
    form = " ".join(value.upper().strip().split())
    is_amendment = form.endswith("/A")
    if is_amendment:
        form = form[:-2].rstrip()
    return form, is_amendment


def _normalize_ticker(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ticker must be a non-empty string")
    return value.strip().upper().replace(".", "-")


def _is_safe_basename(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
    )


def _decision_cutoff(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("decision_at datetime must be timezone-aware")
        cutoff = value.astimezone(_NEW_YORK)
    else:
        if not isinstance(value, date):
            raise TypeError("decision_at must be a date or datetime")
        cutoff = market_session_close(value).astimezone(_NEW_YORK)
    if cutoff > datetime.now(_NEW_YORK):
        raise ValueError("decision_at must not be in the future")
    return cutoff


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise CorporateActionDataError(f"SEC {field} is not a date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CorporateActionDataError(f"SEC {field} is invalid: {value!r}") from exc


def _parse_acceptance_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CorporateActionDataError(f"SEC {field} is missing")
    raw = value.strip()
    try:
        if re.fullmatch(r"\d{14}", raw):
            # Legacy EDGAR indexes expose a 14-digit Eastern wall clock.
            return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=_NEW_YORK)
        else:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("ISO acceptance timestamp has no timezone")
    except ValueError as exc:
        raise CorporateActionDataError(f"SEC {field} is invalid: {value!r}") from exc
    return parsed.astimezone(_NEW_YORK)
