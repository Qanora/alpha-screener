from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date

import polars as pl
import pytest

from alphascreener.finra_short_interest import (
    finra_publication_date,
    finra_short_interest_file_url,
)
from alphascreener.market_calendar import market_session_close
from alphascreener.official_research_data import (
    SEC_COMPANY_TICKERS_URL,
    SEC_INSIDER_DATASETS_URL,
    OfficialResearchDataError,
    discover_insider_archives,
    finra_reporting_cycles,
    load_official_research_features,
    parse_insider_transactions_archive,
    parse_sec_master_archive,
    sec_master_url,
)

CIK = "1001"
ACCESSION = "0000001001-21-000001"
INSIDER_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/2021q2_form345.zip"
)


def _zip(**members: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return output.getvalue()


def _ticker_map() -> bytes:
    return json.dumps(
        {
            "0": {
                "cik_str": int(CIK),
                "ticker": "abc",
                "title": "Example Inc.",
            }
        }
    ).encode()


def _master_zip(last_received: str = "June 30, 2021") -> bytes:
    contents = (
        "Description: Master Index of EDGAR Dissemination Feed\n"
        f"Last Data Received: {last_received}\n"
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "----------------------------------------------------------\n"
        f"{CIK}|Example Inc.|8-K|2021-06-14|"
        f"edgar/data/{CIK}/{ACCESSION}.txt\n"
        f"{CIK}|Example Inc.|S-1|2021-06-14|"
        f"edgar/data/{CIK}/0000001001-21-000002.txt\n"
        "9999|Other Inc.|8-K|2021-06-14|"
        "edgar/data/9999/0000009999-21-000001.txt\n"
    ).encode()
    return _zip(**{"master.idx": contents})


def _insider_zip() -> bytes:
    submission = (
        "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\t"
        "ISSUERCIK\tISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
        f"{ACCESSION}\t14-JUN-2021\t4\t{CIK}\tABC\t1\n"
        "0000001001-21-000003\t14-JUN-2021\t3\t1001\tABC\t0\n"
    ).encode()
    owners = (
        "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\n"
        f"{ACCESSION}\t9001\tBuyer One\n"
        "0000001001-21-000003\t9002\tIgnored Owner\n"
    ).encode()
    transactions = (
        "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_CODE\t"
        "TRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
        f"{ACCESSION}\t1\tP\t100\t10\tA\n"
        f"{ACCESSION}\t2\tS\t25\t12\tD\n"
        f"{ACCESSION}\t3\tP\t50\t\tA\n"
        "0000001001-21-000003\t4\tP\t500\t5\tA\n"
    ).encode()
    return _zip(
        **{
            "SUBMISSION.tsv": submission,
            "REPORTINGOWNER.tsv": owners,
            "NONDERIV_TRANS.tsv": transactions,
        }
    )


def _insider_page() -> bytes:
    return (
        b'<html><body><a href="/files/structureddata/data/'
        b'insider-transactions-data-sets/2021q2_form345.zip">'
        b"2021 Q2 345</a></body></html>"
    )


def _finra_file() -> bytes:
    header = (
        "accountingYearMonthNumber|symbolCode|issueName|"
        "issuerServicesGroupExchangeCode|marketClassCode|"
        "currentShortPositionQuantity|previousShortPositionQuantity|"
        "stockSplitFlag|averageDailyVolumeQuantity|daysToCoverQuantity|"
        "revisionFlag|changePercent|changePreviousNumber|settlementDate"
    )
    row = "20210615|ABC|Example Inc.|Q|NASDAQ|300|200||100|3||0|0|2021-06-15"
    return f"{header}\n{row}\n".encode()


def _responses() -> dict[str, bytes]:
    return {
        SEC_COMPANY_TICKERS_URL: _ticker_map(),
        sec_master_url(2021, 2): _master_zip(),
        SEC_INSIDER_DATASETS_URL: _insider_page(),
        INSIDER_URL: _insider_zip(),
        finra_short_interest_file_url(date(2021, 6, 15)): _finra_file(),
    }


def test_parses_sec_master_and_delays_unknown_acceptance_time() -> None:
    frame = parse_sec_master_archive(
        _master_zip(),
        cik_to_tickers={CIK: ("ABC",)},
    )

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["ticker"] == "ABC"
    assert row["form"] == "8-K"
    assert row["filed_date"] == date(2021, 6, 14)
    assert row["available_at"] == market_session_close(date(2021, 6, 15))
    assert row["accession_number"] == ACCESSION


def test_discovers_and_parses_only_measurable_form4_purchases() -> None:
    links = discover_insider_archives(_insider_page())
    assert links == {(2021, 2): INSIDER_URL}

    frame = parse_insider_transactions_archive(
        _insider_zip(),
        cik_to_tickers={CIK: ("ABC",)},
    )
    assert frame.to_dicts() == [
        {
            "ticker": "ABC",
            "available_at": market_session_close(date(2021, 6, 15)),
            "accepted_at": market_session_close(date(2021, 6, 15)),
            "transaction_code": "P",
            "shares": 100.0,
            "price": 10.0,
            "insider_id": "9001",
            "is_10b5_1": True,
            "transaction_id": f"{ACCESSION}:1",
            "accession_number": ACCESSION,
        }
    ]


def test_main_provider_is_point_in_time_join_ready_and_cached(tmp_path) -> None:
    responses = _responses()
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return responses[url]

    decisions = pl.DataFrame(
        {
            "ticker": ["abc", "ABC", "ABC", "ABC"],
            "decision_date": [
                date(2021, 6, 14),
                date(2021, 6, 15),
                date(2021, 6, 24),
                date(2021, 6, 25),
            ],
            "average_daily_volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    first = load_official_research_features(
        decisions,
        end=date(2021, 6, 25),
        data_home=tmp_path,
        fetcher=fetcher,
        as_of=date(2021, 7, 2),
        lookback_sessions=1,
    )

    before_filing, after_filing, before_finra, after_finra = first.features.to_dicts()
    assert before_filing["ticker"] == "ABC"
    assert before_filing["sec_coverage"] == "complete"
    assert before_filing["current_report_count"] == 0
    assert before_filing["form4_open_market_buy_usd"] == 0.0
    assert after_filing["current_report_count"] == 1
    assert after_filing["form4_open_market_buy_usd"] == 1_000.0

    assert before_finra["short_interest"] is None
    assert after_finra["short_interest_settlement_date"] == date(2021, 6, 15)
    assert after_finra["short_interest_available_at"] == date(2021, 6, 25)
    assert after_finra["short_interest"] == 300
    assert after_finra["days_to_cover"] == 3.0
    assert len(first.snapshot_digest) == 64
    assert set(calls) == set(responses)

    second = load_official_research_features(
        decisions,
        end=date(2021, 6, 25),
        data_home=tmp_path,
        fetcher=lambda url: pytest.fail(f"cache missed for {url}"),
        as_of=date(2021, 7, 2),
        lookback_sessions=1,
    )
    assert second.features.equals(first.features)
    assert second.snapshot_digest == first.snapshot_digest

    cached = list((tmp_path / "data" / "official-research" / "raw").rglob("*"))
    raw_files = [path for path in cached if path.is_file()]
    assert raw_files
    assert not any(path.name.startswith(".tmp-") for path in cached)
    assert all(
        path.stem == hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw_files
        if path.parent.name != ".locks"
    )


def test_pre_source_rows_and_unmapped_sec_tickers_are_explicitly_missing(
    tmp_path,
) -> None:
    responses = _responses()
    decisions = pl.DataFrame(
        {
            "ticker": ["ABC", "NOSEC"],
            "decision_date": [date(2021, 5, 28), date(2021, 6, 15)],
            "average_daily_volume": [100.0, 100.0],
        }
    )

    result = load_official_research_features(
        decisions,
        end=date(2021, 6, 15),
        data_home=tmp_path,
        fetcher=responses.__getitem__,
        as_of=date(2021, 7, 2),
        lookback_sessions=1,
    ).features

    before_source, unmapped = result.to_dicts()
    assert before_source["sec_coverage"] == "missing"
    assert before_source["current_report_count"] is None
    assert before_source["short_interest"] is None
    assert unmapped["sec_coverage"] == "missing"
    assert unmapped["current_report_count"] is None


def test_snapshot_digest_includes_feature_window(tmp_path) -> None:
    responses = _responses()
    decisions = pl.DataFrame(
        {
            "ticker": ["ABC"],
            "decision_date": [date(2021, 6, 15)],
            "average_daily_volume": [100.0],
        }
    )
    one_session = load_official_research_features(
        decisions,
        end=date(2021, 6, 15),
        data_home=tmp_path,
        fetcher=responses.__getitem__,
        as_of=date(2021, 7, 2),
        lookback_sessions=1,
    )
    two_sessions = load_official_research_features(
        decisions,
        end=date(2021, 6, 15),
        data_home=tmp_path,
        fetcher=lambda url: pytest.fail(f"cache missed for {url}"),
        as_of=date(2021, 7, 3),
        lookback_sessions=2,
    )

    assert one_session.snapshot_digest != two_sessions.snapshot_digest


def test_master_coverage_stops_after_last_received_date(tmp_path) -> None:
    responses = _responses()
    responses[sec_master_url(2021, 2)] = _master_zip("June 14, 2021")
    decisions = pl.DataFrame(
        {
            "ticker": ["ABC"],
            "decision_date": [date(2021, 6, 16)],
            "average_daily_volume": [100.0],
        }
    )

    result = load_official_research_features(
        decisions,
        end=date(2021, 6, 16),
        data_home=tmp_path,
        fetcher=responses.__getitem__,
        as_of=date(2021, 7, 2),
        lookback_sessions=2,
    ).features.row(0, named=True)

    assert result["filings_coverage"] == "partial"
    assert result["insider_coverage"] == "complete"
    assert result["sec_coverage"] == "partial"
    assert result["current_report_count"] is None
    assert result["form4_open_market_buy_usd"] == 1_000.0


def test_open_quarter_resources_refresh_on_a_later_observation_day(
    tmp_path,
) -> None:
    responses = _responses()
    responses[sec_master_url(2021, 2)] = _master_zip("June 23, 2021")
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return responses[url]

    decisions = pl.DataFrame(
        {
            "ticker": ["ABC"],
            "decision_date": [date(2021, 6, 24)],
            "average_daily_volume": [100.0],
        }
    )
    first = load_official_research_features(
        decisions,
        end=date(2021, 6, 24),
        data_home=tmp_path,
        fetcher=fetcher,
        as_of=date(2021, 6, 24),
        lookback_sessions=1,
    )
    second = load_official_research_features(
        decisions,
        end=date(2021, 6, 24),
        data_home=tmp_path,
        fetcher=fetcher,
        as_of=date(2021, 6, 25),
        lookback_sessions=1,
    )

    assert calls.count(SEC_COMPANY_TICKERS_URL) == 2
    assert calls.count(sec_master_url(2021, 2)) == 2
    assert calls.count(SEC_INSIDER_DATASETS_URL) == 2
    assert calls.count(INSIDER_URL) == 2
    assert calls.count(finra_short_interest_file_url(date(2021, 6, 15))) == 1
    assert first.snapshot_digest != second.snapshot_digest


def test_finra_cycles_use_explicit_seventh_session_publication() -> None:
    cycles = finra_reporting_cycles(
        date(2021, 6, 1),
        date(2021, 6, 30),
    )

    assert cycles[0] == (
        date(2021, 6, 15),
        finra_publication_date(date(2021, 6, 15)),
    )
    assert cycles[0][1] == date(2021, 6, 24)
    assert cycles[1][0] == date(2021, 6, 30)


def test_invalid_official_archives_fail_closed() -> None:
    with pytest.raises(OfficialResearchDataError, match="not a ZIP"):
        parse_sec_master_archive(
            b"<html>temporary error</html>",
            cik_to_tickers={CIK: ("ABC",)},
        )

    with pytest.raises(OfficialResearchDataError, match="no quarterly"):
        discover_insider_archives(b"<html><body>No files</body></html>")
