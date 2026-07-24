from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest

import alphascreener.corporate_actions as corporate_actions
from alphascreener.corporate_actions import (
    SEC_ARCHIVE_URL,
    SEC_SEARCH_URL,
    SEC_SUBMISSION_FILE_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKER_TEXT_URL,
    SEC_TICKER_URL,
    CorporateActionDataError,
    corporate_action_statuses,
    filter_definitive_transactions,
)


class _FakeFetcher:
    def __init__(self, responses: dict[str, object]):
        self.responses = {
            url: value if isinstance(value, bytes) else json.dumps(value).encode()
            for url, value in responses.items()
        }
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        try:
            return self.responses[url]
        except KeyError as exc:
            raise OSError(f"unexpected URL: {url}") from exc


class _RoutingFetcher(_FakeFetcher):
    def __init__(self, responses: dict[str, object], search_response: object):
        super().__init__(responses)
        self.search_response = (
            search_response
            if isinstance(search_response, bytes)
            else json.dumps(search_response).encode()
        )

    def __call__(self, url: str) -> bytes:
        if url.startswith(f"{SEC_SEARCH_URL}?"):
            self.calls.append(url)
            payload = json.loads(self.search_response)
            hits_node = payload.get("hits")
            if isinstance(hits_node, dict) and isinstance(hits_node.get("hits"), list):
                requested = set(parse_qs(urlsplit(url).query)["ciks"][0].split(","))
                hits = [
                    hit
                    for hit in hits_node["hits"]
                    if requested.intersection(hit.get("_source", {}).get("ciks", []))
                ]
                hits_node["hits"] = hits
                hits_node["total"] = {"value": len(hits), "relation": "eq"}
            return json.dumps(payload).encode()
        return super().__call__(url)


def _ticker_map(**tickers: int) -> dict[str, object]:
    return {
        str(index): {"ticker": ticker, "cik_str": cik, "title": f"{ticker} Inc."}
        for index, (ticker, cik) in enumerate(tickers.items())
    }


def _filing(
    cik: int,
    sequence: int,
    *,
    filed: str,
    accepted: str,
    form: str = "8-K",
    document: str | None = None,
    items: str = "1.01,8.01",
) -> dict[str, str]:
    accession = f"{cik:010d}-{filed[2:4]}-{sequence:06d}"
    return {
        "accessionNumber": accession,
        "filingDate": filed,
        "acceptanceDateTime": accepted,
        "form": form,
        "primaryDocument": document or f"filing-{sequence}.htm",
        "items": items,
    }


def _submission(rows: list[dict[str, str]], *, files: list[dict[str, object]] | None = None):
    columns = {
        name: [row[name] for row in rows]
        for name in (
            "accessionNumber",
            "filingDate",
            "acceptanceDateTime",
            "form",
            "primaryDocument",
            "items",
        )
    }
    return {"filings": {"recent": columns, "files": files or []}}


def _shard(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        name: [row[name] for row in rows]
        for name in (
            "accessionNumber",
            "filingDate",
            "acceptanceDateTime",
            "form",
            "primaryDocument",
            "items",
        )
    }


def _submission_url(cik: int) -> str:
    return SEC_SUBMISSIONS_URL.format(cik=f"{cik:010d}")


def _document_url(cik: int, row: dict[str, str]) -> str:
    return SEC_ARCHIVE_URL.format(
        cik=cik,
        accession=row["accessionNumber"].replace("-", ""),
        document=row["primaryDocument"],
    )


def _archive_document_url(
    cik: int,
    row: dict[str, str],
    document: str,
) -> str:
    return SEC_ARCHIVE_URL.format(
        cik=cik,
        accession=row["accessionNumber"].replace("-", ""),
        document=document,
    )


def _search_response(
    cik: int,
    row: dict[str, str],
    *,
    document: str | None = None,
) -> dict[str, object]:
    hits = (
        []
        if document is None
        else [
            {
                "_id": f"{row['accessionNumber']}:{document}",
                "_source": {
                    "adsh": row["accessionNumber"],
                    "ciks": [f"{cik:010d}"],
                },
            }
        ]
    )
    return {
        "timed_out": False,
        "_shards": {"failed": 0},
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": hits,
        },
    }


def test_production_text_prefilter_reads_only_efts_matches(
    tmp_path,
    monkeypatch,
) -> None:
    filing = _filing(
        123,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T09:00:00.000Z",
    )
    exhibit = "transaction-exhibit.htm"
    fetcher = _RoutingFetcher(
        {
            SEC_TICKER_URL: _ticker_map(TARGET=123),
            _submission_url(123): _submission([filing]),
            _document_url(123, filing): b"The Company will be acquired by Parent.",
            _archive_document_url(123, filing, exhibit): (
                b"Merger Sub will merge with and into the Company."
            ),
        },
        _search_response(123, filing, document=exhibit),
    )
    monkeypatch.setattr(
        corporate_actions,
        "_SecHttpFetcher",
        lambda **_kwargs: fetcher,
    )

    status = corporate_action_statuses(
        ["TARGET"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )["TARGET"]

    assert status.state == "active"
    assert _document_url(123, filing) in fetcher.calls
    assert _archive_document_url(123, filing, exhibit) not in fetcher.calls
    search_calls = [url for url in fetcher.calls if url.startswith(f"{SEC_SEARCH_URL}?")]
    assert len(search_calls) == len(corporate_actions._SEARCH_QUERIES)
    assert {parse_qs(urlsplit(url).query)["q"][0] for url in search_calls} == set(
        corporate_actions._SEARCH_QUERIES
    )
    for url in search_calls:
        query = parse_qs(urlsplit(url).query)
        assert query["startdt"] == ["2022-12-31"]
        assert query["enddt"] == ["2025-02-03"]
        assert query["ciks"] == ["0000000123"]

    calls_before_repeat = len(fetcher.calls)
    repeated = corporate_action_statuses(
        ["TARGET"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )["TARGET"]
    assert repeated == status
    assert len(fetcher.calls) == calls_before_repeat


def test_acquirer_exhibit_cannot_activate_an_exclusion(
    tmp_path,
    monkeypatch,
) -> None:
    filing = _filing(
        126,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T09:00:00.000Z",
    )
    exhibit = "merger-agreement.htm"
    fetcher = _RoutingFetcher(
        {
            SEC_TICKER_URL: _ticker_map(BUYER=126),
            _submission_url(126): _submission([filing]),
            _document_url(126, filing): (
                b"The registrant entered into an agreement to acquire Target Corp."
            ),
            _archive_document_url(126, filing, exhibit): (
                b"The Company will be acquired by Buyer."
            ),
        },
        _search_response(126, filing, document=exhibit),
    )
    monkeypatch.setattr(
        corporate_actions,
        "_SecHttpFetcher",
        lambda **_kwargs: fetcher,
    )

    status = corporate_action_statuses(
        ["BUYER"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )["BUYER"]

    assert status.state == "none"
    assert _archive_document_url(126, filing, exhibit) not in fetcher.calls


def test_production_text_prefilter_skips_old_nonmatching_filing(
    tmp_path,
    monkeypatch,
) -> None:
    filing = _filing(
        124,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T09:00:00.000Z",
    )
    fetcher = _RoutingFetcher(
        {
            SEC_TICKER_URL: _ticker_map(SAFE=124),
            _submission_url(124): _submission([filing]),
        },
        _search_response(124, filing),
    )
    monkeypatch.setattr(
        corporate_actions,
        "_SecHttpFetcher",
        lambda **_kwargs: fetcher,
    )

    status = corporate_action_statuses(
        ["SAFE"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )["SAFE"]

    assert status.state == "none"
    assert _document_url(124, filing) not in fetcher.calls


def test_recent_text_filing_is_checked_during_search_index_lag(
    tmp_path,
    monkeypatch,
) -> None:
    filing = _filing(
        125,
        1,
        filed="2025-02-03",
        accepted="2025-02-03T09:00:00.000Z",
    )
    fetcher = _RoutingFetcher(
        {
            SEC_TICKER_URL: _ticker_map(RECENT=125),
            _submission_url(125): _submission([filing]),
            _document_url(125, filing): b"The Company will be acquired by Parent.",
        },
        _search_response(125, filing),
    )
    monkeypatch.setattr(
        corporate_actions,
        "_SecHttpFetcher",
        lambda **_kwargs: fetcher,
    )

    status = corporate_action_statuses(
        ["RECENT"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )["RECENT"]

    assert status.state == "active"
    assert _document_url(125, filing) in fetcher.calls


def test_efts_search_batches_multiple_issuers(
    tmp_path,
    monkeypatch,
) -> None:
    first = _filing(
        127,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T09:00:00.000Z",
    )
    second = _filing(
        128,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T09:30:00.000Z",
    )
    search_response = {
        "timed_out": False,
        "_shards": {"failed": 0},
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "hits": [
                {
                    "_id": (f"{first['accessionNumber']}:{first['primaryDocument']}"),
                    "_source": {
                        "adsh": first["accessionNumber"],
                        "ciks": ["0000000127"],
                    },
                },
                {
                    "_id": (f"{second['accessionNumber']}:{second['primaryDocument']}"),
                    "_source": {
                        "adsh": second["accessionNumber"],
                        "ciks": ["0000000128"],
                    },
                },
            ],
        },
    }
    fetcher = _RoutingFetcher(
        {
            SEC_TICKER_URL: _ticker_map(FIRST=127, SECOND=128),
            _submission_url(127): _submission([first]),
            _submission_url(128): _submission([second]),
            _document_url(127, first): b"The Company will be acquired by Parent.",
            _document_url(128, second): b"The Company declared a dividend.",
        },
        search_response,
    )
    monkeypatch.setattr(
        corporate_actions,
        "_SecHttpFetcher",
        lambda **_kwargs: fetcher,
    )

    statuses = corporate_action_statuses(
        ["FIRST", "SECOND"],
        date(2025, 2, 3),
        data_home=tmp_path,
    )

    assert statuses["FIRST"].state == "active"
    assert statuses["SECOND"].state == "none"
    search_calls = [url for url in fetcher.calls if url.startswith(f"{SEC_SEARCH_URL}?")]
    assert len(search_calls) == len(corporate_actions._SEARCH_QUERIES)
    assert [parse_qs(urlsplit(url).query)["ciks"][0] for url in search_calls] == [
        "0000000127,0000000128",
        "0000000127,0000000128",
        "0000000127",
        "0000000127",
    ]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "timed_out": True,
                "_shards": {"failed": 0},
                "hits": {
                    "total": {"value": 0, "relation": "eq"},
                    "hits": [],
                },
            },
            "timed out",
        ),
        (
            {
                "timed_out": False,
                "_shards": {"failed": 1},
                "hits": {
                    "total": {"value": 0, "relation": "eq"},
                    "hits": [],
                },
            },
            "failed shards",
        ),
        (
            {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "total": {"value": 1, "relation": "gte"},
                    "hits": [],
                },
            },
            "incomplete hit metadata",
        ),
        (
            {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "total": {"value": 1, "relation": "eq"},
                    "hits": [
                        {
                            "_id": "0000000129-25-000001:filing.htm",
                            "_source": {
                                "adsh": "0000000129-25-000001",
                                "ciks": ["0000000999"],
                            },
                        }
                    ],
                },
            },
            "invalid issuer metadata",
        ),
        (
            {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "total": {"value": 1, "relation": "eq"},
                    "hits": [
                        {
                            "_id": "0000000129-25-000001:..",
                            "_source": {
                                "adsh": "0000000129-25-000001",
                                "ciks": ["0000000129"],
                            },
                        }
                    ],
                },
            },
            "unsafe document name",
        ),
    ],
)
def test_efts_response_validation_fails_closed(payload, match) -> None:
    with pytest.raises(CorporateActionDataError, match=match):
        corporate_actions._parse_search_page(
            payload,
            requested_ciks={"0000000129"},
        )


def test_efts_search_paginates_until_the_reported_total() -> None:
    cik = "0000000130"
    accessions = [f"0000000130-25-{sequence:06d}" for sequence in range(1, 102)]
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        query = parse_qs(urlsplit(url).query)
        offset = int(query["from"][0])
        if query["q"][0] != corporate_actions._SEARCH_QUERIES[0]:
            page_accessions: list[str] = []
            total = 0
        else:
            page_accessions = accessions[offset : offset + 100]
            total = len(accessions)
        return json.dumps(
            {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "total": {"value": total, "relation": "eq"},
                    "hits": [
                        {
                            "_id": f"{accession}:filing.htm",
                            "_source": {"adsh": accession, "ciks": [cik]},
                        }
                        for accession in page_accessions
                    ],
                },
            }
        ).encode()

    documents = corporate_actions._fetch_text_search_documents(
        (cik,),
        search_start=date(2023, 1, 1),
        search_end=date(2025, 2, 3),
        fetcher=fetcher,
    )

    assert set(documents[cik]) == set(accessions)
    first_query_offsets = [
        int(parse_qs(urlsplit(url).query)["from"][0])
        for url in calls
        if parse_qs(urlsplit(url).query)["q"][0] == corporate_actions._SEARCH_QUERIES[0]
    ]
    assert first_query_offsets == [0, 100]


def test_search_window_covers_the_full_lookback_across_year_boundaries() -> None:
    cutoff = date(2026, 1, 2)

    assert corporate_actions._search_window_start(cutoff) == date(2023, 12, 31)
    assert corporate_actions._search_window_start(cutoff) <= (
        cutoff - timedelta(days=corporate_actions.TRANSACTION_LOOKBACK_DAYS)
    )


def test_acceptance_iso_timestamps_are_converted_from_utc_to_new_york() -> None:
    summer = corporate_actions._parse_acceptance_datetime(
        "2026-06-09T13:00:02.000Z",
        "acceptance",
    )
    winter = corporate_actions._parse_acceptance_datetime(
        "2026-01-09T14:00:02.000Z",
        "acceptance",
    )
    legacy = corporate_actions._parse_acceptance_datetime(
        "20260609090002",
        "acceptance",
    )

    assert (summer.hour, summer.utcoffset().total_seconds()) == (9, -4 * 3600)
    assert (winter.hour, winter.utcoffset().total_seconds()) == (9, -5 * 3600)
    assert legacy.hour == 9


def test_status_uses_only_filings_accepted_by_the_decision_time(tmp_path) -> None:
    positive = _filing(
        1234,
        1,
        filed="2025-01-02",
        accepted="2025-01-02T20:45:00.000Z",
    )
    after_close = _filing(
        1234,
        2,
        filed="2025-01-02",
        accepted="2025-01-02T21:30:00.000Z",
        items="1.02,8.01",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(TGT=1234),
            _submission_url(1234): _submission([after_close, positive]),
            _document_url(1234, positive): (
                b"<p>Merger Sub will merge with and into the Company.</p>"
            ),
            _document_url(1234, after_close): (
                b"<p>The Company and Buyer mutually agreed to terminate the merger transaction.</p>"
            ),
        }
    )

    status = corporate_action_statuses(
        ["tgt"],
        date(2025, 1, 2),
        data_home=tmp_path,
        fetcher=fetcher,
    )["TGT"]

    assert status.under_definitive_transaction
    assert status.accession_number == positive["accessionNumber"]
    assert status.decision_at.hour == 16
    assert _document_url(1234, after_close) not in fetcher.calls

    later = corporate_action_statuses(
        ["TGT"],
        datetime(2025, 1, 2, 17, 0, tzinfo=ZoneInfo("America/New_York")),
        data_home=tmp_path,
        fetcher=fetcher,
    )["TGT"]

    assert later.state == "terminated"
    assert not later.exclude_from_ranking
    assert later.accession_number == after_close["accessionNumber"]


def test_target_proxy_language_activates_and_filters_tickers(tmp_path) -> None:
    merger_proxy = _filing(
        111,
        1,
        filed="2025-02-03",
        accepted="2025-02-03T10:00:00.000Z",
        form="DEFM14A",
        items="",
    )
    unrelated = _filing(
        222,
        1,
        filed="2025-02-03",
        accepted="2025-02-03T11:00:00.000Z",
        items="8.01",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(DEAL=111, SAFE=222),
            _submission_url(111): _submission([merger_proxy]),
            _submission_url(222): _submission([unrelated]),
            _document_url(111, merger_proxy): (
                b"<p>The Company will become a wholly-owned subsidiary of Parent.</p>"
            ),
            _document_url(222, unrelated): b"<p>The company declared a dividend.</p>",
        }
    )

    retained, statuses = filter_definitive_transactions(
        ["deal", "safe"],
        date(2025, 2, 3),
        data_home=tmp_path,
        fetcher=fetcher,
    )

    assert retained == ("SAFE",)
    assert statuses["DEAL"].reason == "definitive_target_transaction_language"
    assert statuses["SAFE"].state == "none"


def test_completed_merger_clears_an_earlier_active_state(tmp_path) -> None:
    agreement = _filing(
        345,
        1,
        filed="2024-09-03",
        accepted="2024-09-03T08:00:00.000Z",
    )
    completion = _filing(
        345,
        2,
        filed="2025-03-04",
        accepted="2025-03-04T09:00:00.000Z",
        items="2.01,5.01",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(DONE=345),
            _submission_url(345): _submission([completion, agreement]),
            _document_url(345, agreement): (b"Merger Sub will merge with and into the Company."),
            _document_url(345, completion): b"The merger was completed and consummated.",
        }
    )

    status = corporate_action_statuses(
        ["DONE"],
        date(2025, 3, 4),
        data_home=tmp_path,
        fetcher=fetcher,
    )["DONE"]

    assert status.state == "completed"
    assert status.reason == "transaction_completed"
    assert not status.under_definitive_transaction


def test_historical_submission_shard_is_loaded_for_an_old_decision(tmp_path) -> None:
    old = _filing(
        456,
        1,
        filed="2022-06-01",
        accepted="2022-06-01T12:00:00.000Z",
        form="SC 14D9",
        items="",
    )
    shard_name = "CIK0000000456-submissions-001.json"
    metadata = [
        {
            "name": shard_name,
            "filingCount": 1,
            "filingFrom": "2022-01-01",
            "filingTo": "2022-12-31",
        }
    ]
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(OLD=456),
            _submission_url(456): _submission([], files=metadata),
            SEC_SUBMISSION_FILE_URL.format(name=shard_name): _shard([old]),
            _document_url(456, old): b"Recommendation statement for shareholders.",
        }
    )

    status = corporate_action_statuses(
        ["OLD"],
        date(2022, 6, 2),
        data_home=tmp_path,
        fetcher=fetcher,
    )["OLD"]

    assert status.under_definitive_transaction
    assert status.filing_form == "SC 14D9"
    assert SEC_SUBMISSION_FILE_URL.format(name=shard_name) in fetcher.calls


def test_cached_sec_resources_allow_a_repeat_without_network(tmp_path) -> None:
    filing = _filing(
        567,
        1,
        filed="2025-04-01",
        accepted="2025-04-01T09:00:00.000Z",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(CACHE=567),
            _submission_url(567): _submission([filing]),
            _document_url(567, filing): b"The Company will be acquired by Parent.",
        }
    )
    first = corporate_action_statuses(
        ["CACHE"],
        date(2025, 4, 1),
        data_home=tmp_path,
        fetcher=fetcher,
    )

    def no_network(url: str) -> bytes:
        raise AssertionError(f"unexpected network request: {url}")

    second = corporate_action_statuses(
        ["CACHE"],
        date(2025, 4, 1),
        data_home=tmp_path,
        fetcher=no_network,
    )

    assert first == second
    assert (tmp_path / "data" / "sec" / "company_tickers.json").is_file()


def test_official_text_ticker_map_is_used_when_json_endpoint_is_unavailable(
    tmp_path,
) -> None:
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_TEXT_URL: b"fallback\t906\n",
            _submission_url(906): _submission([]),
        }
    )

    status = corporate_action_statuses(
        ["FALLBACK"],
        date(2025, 4, 2),
        data_home=tmp_path,
        fetcher=fetcher,
    )["FALLBACK"]

    assert status.cik == "0000000906"
    assert status.state == "none"
    assert SEC_TICKER_URL in fetcher.calls
    assert SEC_TICKER_TEXT_URL in fetcher.calls


@pytest.mark.parametrize(
    ("responses", "match"),
    [
        ({SEC_TICKER_URL: _ticker_map(OTHER=678)}, "no CIK for: MISSING"),
        (
            {
                SEC_TICKER_URL: _ticker_map(BROKEN=678),
                _submission_url(678): {
                    "filings": {
                        "recent": {
                            "accessionNumber": [],
                            "filingDate": [],
                            "acceptanceDateTime": [],
                            "form": ["8-K"],
                            "primaryDocument": [],
                        },
                        "files": [],
                    }
                },
            },
            "misaligned columns",
        ),
    ],
)
def test_incomplete_sec_data_fails_clearly(tmp_path, responses, match) -> None:
    ticker = "MISSING" if "OTHER" in str(responses) else "BROKEN"

    with pytest.raises(CorporateActionDataError, match=match):
        corporate_action_statuses(
            [ticker],
            date(2025, 5, 1),
            data_home=tmp_path,
            fetcher=_FakeFetcher(responses),
        )


def test_missing_primary_document_fetch_does_not_silently_pass(tmp_path) -> None:
    filing = _filing(
        789,
        1,
        filed="2025-05-01",
        accepted="2025-05-01T09:00:00.000Z",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(BROKEN=789),
            _submission_url(789): _submission([filing]),
        }
    )

    with pytest.raises(CorporateActionDataError, match="could not fetch SEC filing"):
        corporate_action_statuses(
            ["BROKEN"],
            date(2025, 5, 1),
            data_home=tmp_path,
            fetcher=fetcher,
        )


def test_acquirer_8k_does_not_exclude_the_acquirer(tmp_path) -> None:
    filing = _filing(
        901,
        1,
        filed="2025-05-02",
        accepted="2025-05-02T09:00:00.000Z",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(BUYER=901),
            _submission_url(901): _submission([filing]),
            _document_url(901, filing): (
                b"The Company entered into an Agreement and Plan of Merger "
                b"and will acquire Target Corporation."
            ),
        }
    )

    status = corporate_action_statuses(
        ["BUYER"],
        date(2025, 5, 2),
        data_home=tmp_path,
        fetcher=fetcher,
    )["BUYER"]

    assert status.state == "none"
    assert not status.exclude_from_ranking


@pytest.mark.parametrize("form", ["SC TO-T", "SC TO-I", "SC TO", "S-4"])
def test_role_ambiguous_forms_do_not_exclude_the_filer(tmp_path, form) -> None:
    filing = _filing(
        902,
        1,
        filed="2025-05-05",
        accepted="2025-05-05T09:00:00.000Z",
        form=form,
        items="",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(FILER=902),
            _submission_url(902): _submission([filing]),
        }
    )

    status = corporate_action_statuses(
        ["FILER"],
        date(2025, 5, 5),
        data_home=tmp_path,
        fetcher=fetcher,
    )["FILER"]

    assert status.state == "none"
    assert _document_url(902, filing) not in fetcher.calls


def test_target_tender_amendment_can_terminate_active_offer(tmp_path) -> None:
    original = _filing(
        903,
        1,
        filed="2025-05-06",
        accepted="2025-05-06T09:00:00.000Z",
        form="SC 14D9",
        items="",
    )
    withdrawal = _filing(
        903,
        2,
        filed="2025-05-07",
        accepted="2025-05-07T09:00:00.000Z",
        form="SC 14D9/A",
        items="",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(TARGET=903),
            _submission_url(903): _submission([withdrawal, original]),
            _document_url(903, original): b"Schedule 14D-9 recommendation statement.",
            _document_url(903, withdrawal): b"The tender offer has been withdrawn.",
        }
    )

    status = corporate_action_statuses(
        ["TARGET"],
        date(2025, 5, 7),
        data_home=tmp_path,
        fetcher=fetcher,
    )["TARGET"]

    assert status.state == "terminated"
    assert not status.exclude_from_ranking


def test_early_close_uses_the_exchange_close_not_four_pm(tmp_path) -> None:
    before_close = _filing(
        904,
        1,
        filed="2025-11-28",
        accepted="2025-11-28T17:45:00.000Z",
    )
    after_close = _filing(
        904,
        2,
        filed="2025-11-28",
        accepted="2025-11-28T18:30:00.000Z",
        items="1.02",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(EARLY=904),
            _submission_url(904): _submission([after_close, before_close]),
            _document_url(904, before_close): b"The Company will be acquired by Parent.",
            _document_url(904, after_close): b"The merger agreement was terminated.",
        }
    )

    status = corporate_action_statuses(
        ["EARLY"],
        date(2025, 11, 28),
        data_home=tmp_path,
        fetcher=fetcher,
    )["EARLY"]

    assert status.state == "active"
    assert status.decision_at.hour == 13
    assert _document_url(904, after_close) not in fetcher.calls


def test_naive_intraday_cutoff_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        corporate_action_statuses(
            ["TGT"],
            datetime(2025, 5, 8, 12, 0),
            data_home=tmp_path,
            fetcher=_FakeFetcher({}),
        )


def test_future_cutoff_is_rejected_before_any_network_request(tmp_path) -> None:
    fetcher = _FakeFetcher({})
    future = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=1)

    with pytest.raises(ValueError, match="must not be in the future"):
        corporate_action_statuses(
            ["TGT"],
            future,
            data_home=tmp_path,
            fetcher=fetcher,
        )

    assert fetcher.calls == []


def test_sec_error_html_is_not_cached_as_a_safe_filing(tmp_path) -> None:
    filing = _filing(
        905,
        1,
        filed="2025-05-09",
        accepted="2025-05-09T09:00:00.000Z",
    )
    fetcher = _FakeFetcher(
        {
            SEC_TICKER_URL: _ticker_map(BLOCKED=905),
            _submission_url(905): _submission([filing]),
            _document_url(905, filing): (
                b"<html><title>SEC.gov | Request Rate Threshold Exceeded</title></html>"
            ),
        }
    )

    with pytest.raises(CorporateActionDataError, match="SEC blocked"):
        corporate_action_statuses(
            ["BLOCKED"],
            date(2025, 5, 9),
            data_home=tmp_path,
            fetcher=fetcher,
        )

    filing_cache = (
        tmp_path
        / "data"
        / "sec"
        / "filings"
        / "0000000905"
        / filing["accessionNumber"].replace("-", "")
        / filing["primaryDocument"]
    )
    assert not filing_cache.exists()
