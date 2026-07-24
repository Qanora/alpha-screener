from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl
import pytest

import alphascreener.finra_short_interest as finra_short_interest
from alphascreener.finra_short_interest import (
    FinraShortInterestDataError,
    finra_publication_date,
    finra_short_interest_file_url,
    load_short_interest_file,
    parse_short_interest_file,
    short_interest_features,
)
from alphascreener.market_calendar import future_market_date


def _file(
    *rows: str,
    header: str = (
        "accountingYearMonthNumber|symbolCode|issueName|"
        "issuerServicesGroupExchangeCode|marketClassCode|"
        "currentShortPositionQuantity|previousShortPositionQuantity|"
        "stockSplitFlag|averageDailyVolumeQuantity|daysToCoverQuantity|"
        "revisionFlag|changePercent|changePreviousNumber|settlementDate"
    ),
) -> bytes:
    return ("\n".join((header, *rows)) + "\n").encode()


def _row(
    symbol: str,
    short_interest: int,
    settlement_date: str,
    *,
    stock_split_flag: str = "",
    revision_flag: str = "",
) -> str:
    compact_date = settlement_date.replace("-", "")
    return (
        f"{compact_date}|{symbol}|Example Inc.|Q|NASDAQ|{short_interest}|900|"
        f"{stock_split_flag}|100|10|{revision_flag}|0|0|{settlement_date}"
    )


def _zip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.writestr("short-interest.csv", payload)
    return output.getvalue()


def test_parse_official_pipe_file_normalizes_fields_and_symbol() -> None:
    frame = parse_short_interest_file(
        _file(_row("brk.b", 1_234_567, "2026-01-15")),
        publication_dates=date(2026, 1, 27),
    )

    assert frame.to_dicts() == [
        {
            "symbol": "BRK-B",
            "settlement_date": date(2026, 1, 15),
            "publication_date": date(2026, 1, 27),
            "available_at": date(2026, 1, 28),
            "short_interest": 1_234_567,
            "stock_split_flag": False,
            "revision_flag": False,
        }
    ]


def test_parse_zip_and_derive_documented_publication_date() -> None:
    settlement = date(2026, 1, 15)
    frame = parse_short_interest_file(_zip(_file(_row("ABC", 400, settlement.isoformat()))))

    assert finra_publication_date(settlement) == date(2026, 1, 27)
    assert frame["publication_date"].to_list() == [date(2026, 1, 27)]
    assert finra_short_interest_file_url(settlement).endswith("/shrt20260115.csv")


def test_parse_comma_delimited_api_field_aliases() -> None:
    payload = (
        b'"issueSymbolIdentifier","settlementDate","issueName",'
        b'"currentShortShareNumber","revisionFlag"\n'
        b'"ABC","2026-01-15","Example, Inc.","1,234",""\n'
    )

    frame = parse_short_interest_file(
        payload,
        publication_dates=date(2026, 1, 27),
    )

    assert frame.select("symbol", "short_interest").row(0) == ("ABC", 1_234)


def test_revised_value_is_not_available_before_file_was_observed() -> None:
    payload = _file(_row("ABC", 800, "2026-01-15", revision_flag="Y"))

    with pytest.raises(FinraShortInterestDataError, match="observed_at is required"):
        parse_short_interest_file(
            payload,
            publication_dates=date(2026, 1, 27),
        )

    frame = parse_short_interest_file(
        payload,
        publication_dates=date(2026, 1, 27),
        observed_at=date(2026, 2, 4),
    )
    assert frame.row(0, named=True)["available_at"] == date(2026, 2, 5)
    assert frame.row(0, named=True)["revision_flag"] is True


def test_feature_join_uses_availability_not_settlement_and_computes_ratios() -> None:
    first = parse_short_interest_file(
        _file(_row("ABC", 1_000, "2026-01-15")),
        publication_dates=date(2026, 1, 27),
    )
    second = parse_short_interest_file(
        _file(_row("ABC", 1_500, "2026-01-30")),
        publication_dates=date(2026, 2, 10),
    )
    records = pl.concat((first, second))

    features = short_interest_features(
        records,
        tickers=["ABC", "ABC", "ABC"],
        decision_dates=[
            date(2026, 1, 26),
            date(2026, 1, 28),
            date(2026, 2, 11),
        ],
        average_daily_volume=[100, 200, 300],
        shares_outstanding=[10_000, None, 12_000],
    )

    before, first_known, second_known = features.to_dicts()
    assert before["short_interest"] is None
    assert first_known["short_interest"] == 1_000
    assert first_known["short_interest_delta"] is None
    assert first_known["days_to_cover"] == 5.0
    assert first_known["short_pct"] is None
    assert first_known["short_interest_stock_split_flag"] is False
    assert first_known["short_interest_previous_stock_split_flag"] is None
    assert first_known["short_interest_delta_suppressed_by_split"] is False
    assert second_known["short_interest"] == 1_500
    assert second_known["short_interest_delta"] == pytest.approx(0.5)
    assert second_known["days_to_cover"] == 5.0
    assert second_known["short_pct"] == pytest.approx(0.125)


def test_revision_of_older_cycle_does_not_replace_newer_known_cycle() -> None:
    first = parse_short_interest_file(
        _file(_row("ABC", 1_000, "2026-01-15")),
        publication_dates=date(2026, 1, 27),
    )
    second = parse_short_interest_file(
        _file(_row("ABC", 1_500, "2026-01-30")),
        publication_dates=date(2026, 2, 10),
    )
    revised_first = parse_short_interest_file(
        _file(_row("ABC", 1_200, "2026-01-15", revision_flag="R")),
        publication_dates=date(2026, 1, 27),
        observed_at=date(2026, 2, 12),
    )

    feature = short_interest_features(
        pl.concat((first, second, revised_first)),
        tickers=["ABC"],
        decision_dates=[date(2026, 2, 13)],
        average_daily_volume=[300],
        shares_outstanding=[12_000],
    ).row(0, named=True)

    assert feature["short_interest_settlement_date"] == date(2026, 1, 30)
    assert feature["short_interest"] == 1_500
    assert feature["short_interest_delta"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("first_split", "second_split"),
    [("Y", ""), ("", "Y"), ("", "S")],
)
def test_stock_split_in_either_adjacent_cycle_suppresses_delta(
    first_split: str,
    second_split: str,
) -> None:
    first = parse_short_interest_file(
        _file(
            _row(
                "ABC",
                1_000,
                "2026-01-15",
                stock_split_flag=first_split,
            )
        ),
        publication_dates=date(2026, 1, 27),
    )
    second = parse_short_interest_file(
        _file(
            _row(
                "ABC",
                5_000,
                "2026-01-30",
                stock_split_flag=second_split,
            )
        ),
        publication_dates=date(2026, 2, 10),
    )

    feature = short_interest_features(
        pl.concat((first, second)),
        tickers=["ABC"],
        decision_dates=[date(2026, 2, 11)],
        average_daily_volume=[500],
    ).row(0, named=True)

    assert feature["short_interest_delta"] is None
    assert feature["short_interest_stock_split_flag"] is bool(second_split)
    assert feature["short_interest_previous_stock_split_flag"] is bool(first_split)
    assert feature["short_interest_delta_suppressed_by_split"] is True


def test_last_known_value_expires_after_sixty_market_sessions() -> None:
    publication = date(2025, 1, 27)
    records = parse_short_interest_file(
        _file(_row("ABC", 1_000, "2025-01-15")),
        publication_dates=publication,
    )
    available = records.item(0, "available_at")
    exactly_sixty_sessions_later = future_market_date(available, 60)
    sixty_one_sessions_later = future_market_date(available, 61)

    features = short_interest_features(
        records,
        tickers=["ABC", "ABC"],
        decision_dates=[exactly_sixty_sessions_later, sixty_one_sessions_later],
        average_daily_volume=[100, 100],
        shares_outstanding=[10_000, 10_000],
    )

    assert features["short_interest"].to_list() == [1_000, None]
    assert features["short_interest_age_sessions"].to_list() == [60, None]


def test_raw_cache_is_content_addressed_immutable_and_reusable(tmp_path) -> None:
    url = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260115.csv"
    first_payload = _file(_row("ABC", 1_000, "2026-01-15"))
    second_payload = _file(_row("ABC", 1_100, "2026-01-15", revision_flag="Y"))
    calls: list[str] = []
    payloads = iter((first_payload, second_payload))

    def fetcher(requested_url: str) -> bytes:
        calls.append(requested_url)
        return next(payloads)

    first = load_short_interest_file(
        url,
        data_home=tmp_path,
        fetcher=fetcher,
        observed_at=date(2026, 1, 27),
    )
    cached = load_short_interest_file(
        url,
        data_home=tmp_path,
        fetcher=lambda _url: pytest.fail("cache should avoid a request"),
    )
    refreshed = load_short_interest_file(
        url,
        data_home=tmp_path,
        fetcher=fetcher,
        observed_at=date(2026, 2, 3),
        refresh=True,
    )

    assert calls == [url, url]
    assert first.from_cache is False
    assert cached.from_cache is True
    assert refreshed.from_cache is False
    assert first.raw_path != refreshed.raw_path
    assert first.raw_path.read_bytes() == first_payload
    assert refreshed.raw_path.read_bytes() == second_payload
    assert len(list(first.raw_path.parent.iterdir())) == 2
    assert refreshed.records.row(0, named=True)["available_at"] == date(2026, 2, 4)


def test_invalid_response_is_not_cached(tmp_path) -> None:
    url = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260115.csv"

    with pytest.raises(FinraShortInterestDataError, match="HTML"):
        load_short_interest_file(
            url,
            data_home=tmp_path,
            fetcher=lambda _url: b"<html>temporarily unavailable</html>",
            observed_at=date(2026, 1, 27),
        )

    raw_root = tmp_path / "data" / "finra" / "short-interest" / "raw"
    assert not any(path.suffix in {".csv", ".zip"} for path in raw_root.rglob("*"))


def test_parser_rejects_missing_fields_and_conflicting_rows() -> None:
    with pytest.raises(FinraShortInterestDataError, match="missing required field"):
        parse_short_interest_file(b"symbolCode|settlementDate\nABC|2026-01-15\n")

    conflicting = _file(
        _row("ABC", 100, "2026-01-15"),
        _row("ABC", 200, "2026-01-15"),
    )
    with pytest.raises(FinraShortInterestDataError, match="conflicting"):
        parse_short_interest_file(
            conflicting,
            publication_dates=date(2026, 1, 27),
        )


def test_inputs_must_be_row_aligned() -> None:
    records = parse_short_interest_file(
        _file(_row("ABC", 1_000, "2026-01-15")),
        publication_dates=date(2026, 1, 27),
    )

    with pytest.raises(ValueError, match="equal lengths"):
        short_interest_features(
            records,
            tickers=["ABC"],
            decision_dates=[],
            average_daily_volume=[100],
        )


def test_large_feature_build_is_chunked_and_preserves_generator_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 200_003
    chunk_size = 50_000
    monkeypatch.setattr(finra_short_interest, "_FEATURE_CHUNK_SIZE", chunk_size)
    records = parse_short_interest_file(
        _file(_row("ABC", 1_000, "2026-01-15")),
        publication_dates=date(2026, 1, 27),
    ).head(0)

    features = short_interest_features(
        records,
        tickers=("ZZZ" if index % 2 == 0 else "YYY" for index in range(row_count)),
        decision_dates=(date(2026, 1, 28) for _ in range(row_count)),
        average_daily_volume=(100 for _ in range(row_count)),
    )

    assert features.height == row_count
    assert features.n_chunks() == 5
    assert features["ticker"].head(4).to_list() == ["ZZZ", "YYY", "ZZZ", "YYY"]
    assert features.item(row_count - 1, "ticker") == "ZZZ"
