from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

import alphascreener.sec_signals as sec_signals
from alphascreener.market_calendar import (
    market_dates_between,
    market_session_close,
)
from alphascreener.sec_signals import (
    SecSignalDataset,
    build_sec_signal_features,
)

DECISION_DATE = date(2025, 3, 14)


def _decisions(*tickers: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": list(tickers),
            "decision_date": [DECISION_DATE] * len(tickers),
        }
    )


def _coverage(
    *tickers: str,
    start: date | None = None,
    end: date = DECISION_DATE,
    status: str = "complete",
) -> pl.DataFrame:
    start = start or _window()[0]
    return pl.DataFrame(
        [
            {
                "ticker": ticker,
                "source": source,
                "coverage_start": start,
                "coverage_end": end,
                "status": status,
            }
            for ticker in tickers
            for source in ("filings", "insider_transactions")
        ]
    )


def _window() -> list[date]:
    return market_dates_between(DECISION_DATE - timedelta(days=120), DECISION_DATE)[-60:]


def _filing_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "ticker": pl.String,
        "accepted_at": pl.Datetime(time_zone="UTC"),
        "available_at": pl.Datetime(time_zone="UTC"),
        "form": pl.String,
        "items": pl.String,
        "accession_number": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def _insider_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "ticker": pl.String,
        "available_at": pl.Datetime(time_zone="UTC"),
        "accepted_at": pl.Datetime(time_zone="UTC"),
        "transaction_code": pl.String,
        "shares": pl.Float64,
        "price": pl.Float64,
        "insider_id": pl.String,
        "is_10b5_1": pl.Boolean,
        "transaction_id": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def test_complete_coverage_distinguishes_no_events_from_missing() -> None:
    result = build_sec_signal_features(
        _decisions("NONE", "MISS"),
        filings=_filing_frame([]),
        insider_transactions=_insider_frame([]),
        coverage=_coverage("NONE"),
    )

    no_events = result.row(0, named=True)
    assert no_events["sec_coverage"] == "complete"
    assert no_events["current_report_count"] == 0
    assert no_events["material_event_count"] == 0
    assert no_events["recent_offering_risk"] is False
    assert no_events["days_since_8k_earnings"] is None
    assert no_events["form4_open_market_buy_usd"] == 0.0
    assert no_events["distinct_insider_buyers"] == 0
    assert no_events["cluster_buy"] is False

    missing = result.row(1, named=True)
    assert missing["sec_coverage"] == "missing"
    assert missing["filings_coverage"] == "missing"
    assert missing["insider_coverage"] == "missing"
    assert missing["material_event_count"] is None
    assert missing["form4_open_market_buy_usd"] is None
    assert missing["cluster_buy"] is None


def test_filings_are_strictly_as_of_close_and_sixty_sessions() -> None:
    sessions = _window()
    prior_session = market_dates_between(
        sessions[0] - timedelta(days=14),
        sessions[0] - timedelta(days=1),
    )[-1]
    decision_close = market_session_close(DECISION_DATE)
    earnings_session = sessions[-4]
    earnings_before_close = market_session_close(earnings_session) - timedelta(minutes=1)
    rows = [
        {
            "ticker": "ABC",
            "accepted_at": earnings_before_close,
            "available_at": earnings_before_close,
            "form": "8-K",
            "items": "Item 2.02, Item 9.01",
            "accession_number": "earnings",
        },
        {
            "ticker": "ABC",
            "accepted_at": decision_close - timedelta(seconds=1),
            "available_at": decision_close - timedelta(seconds=1),
            "form": "6-K",
            "items": "",
            "accession_number": "before-close",
        },
        {
            "ticker": "ABC",
            "accepted_at": decision_close - timedelta(hours=2),
            "available_at": decision_close + timedelta(seconds=1),
            "form": "S-3",
            "items": "",
            "accession_number": "not-yet-available",
        },
        {
            "ticker": "ABC",
            "accepted_at": market_session_close(sessions[0]) - timedelta(minutes=1),
            "available_at": market_session_close(sessions[0]) - timedelta(minutes=1),
            "form": "424B5",
            "items": "",
            "accession_number": "window-edge",
        },
        {
            "ticker": "ABC",
            "accepted_at": market_session_close(prior_session),
            "available_at": market_session_close(prior_session),
            "form": "NT 10-Q",
            "items": "",
            "accession_number": "too-old",
        },
    ]

    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=_filing_frame(rows),
        insider_transactions=_insider_frame([]),
        coverage=_coverage("ABC"),
    ).row(0, named=True)

    assert result["days_since_8k_earnings"] == 3
    assert result["current_report_count"] == 2
    assert result["material_event_count"] == 2
    assert result["recent_offering_risk"] is True
    assert result["late_filing_risk"] is False


def test_filing_rules_normalize_amendments_and_exclude_13d_amendment() -> None:
    known_at = market_session_close(DECISION_DATE) - timedelta(hours=1)
    filings = _filing_frame(
        [
            {
                "ticker": "abc",
                "accepted_at": known_at,
                "available_at": known_at,
                "form": "S-3/A",
                "items": "",
                "accession_number": "shelf",
            },
            {
                "ticker": "abc",
                "accepted_at": known_at,
                "available_at": known_at,
                "form": "NT 10-K/A",
                "items": "",
                "accession_number": "late",
            },
            {
                "ticker": "abc",
                "accepted_at": known_at,
                "available_at": known_at,
                "form": "SC 13D/A",
                "items": "",
                "accession_number": "amended-13d",
            },
        ]
    )
    first = build_sec_signal_features(
        _decisions("ABC"),
        filings=filings,
        insider_transactions=_insider_frame([]),
        coverage=_coverage("ABC"),
    ).row(0, named=True)
    assert first["recent_offering_risk"] is True
    assert first["late_filing_risk"] is True
    assert first["new_13d"] is False

    initial_13d = filings.vstack(
        _filing_frame(
            [
                {
                    "ticker": "ABC",
                    "accepted_at": known_at,
                    "available_at": known_at,
                    "form": "SC 13D",
                    "items": "",
                    "accession_number": "initial-13d",
                }
            ]
        )
    )
    second = build_sec_signal_features(
        _decisions("ABC"),
        filings=initial_13d,
        insider_transactions=_insider_frame([]),
        coverage=_coverage("ABC"),
    ).row(0, named=True)
    assert second["new_13d"] is True


def test_form4_counts_only_purchases_and_preserves_10b5_1_buckets() -> None:
    known_at = market_session_close(DECISION_DATE) - timedelta(hours=1)
    common = {
        "ticker": "ABC",
        "available_at": known_at,
        "accepted_at": known_at,
    }
    transactions = _insider_frame(
        [
            {
                **common,
                "transaction_code": "P",
                "shares": 100.0,
                "price": 10.0,
                "insider_id": "one",
                "is_10b5_1": True,
                "transaction_id": "1",
            },
            {
                **common,
                "transaction_code": "p",
                "shares": 50.0,
                "price": 20.0,
                "insider_id": "two",
                "is_10b5_1": False,
                "transaction_id": "2",
            },
            {
                **common,
                "transaction_code": "P",
                "shares": 25.0,
                "price": 20.0,
                "insider_id": "two",
                "is_10b5_1": None,
                "transaction_id": "3",
            },
            {
                **common,
                "transaction_code": "A",
                "shares": 1_000.0,
                "price": 100.0,
                "insider_id": "three",
                "is_10b5_1": False,
                "transaction_id": "4",
            },
        ]
    )

    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=_filing_frame([]),
        insider_transactions=transactions,
        coverage=_coverage("ABC"),
    ).row(0, named=True)

    assert result["form4_open_market_buy_usd"] == 2_500.0
    assert result["form4_10b5_1_buy_usd"] == 1_000.0
    assert result["form4_non_10b5_1_buy_usd"] == 1_000.0
    assert result["form4_unknown_10b5_1_buy_usd"] == 500.0
    assert result["distinct_insider_buyers"] == 2
    assert result["cluster_buy"] is True


def test_jointly_reported_form4_counts_dollars_once_and_both_buyers() -> None:
    known_at = market_session_close(DECISION_DATE) - timedelta(hours=1)
    common = {
        "ticker": "ABC",
        "available_at": known_at,
        "accepted_at": known_at,
        "transaction_code": "P",
        "shares": 100.0,
        "price": 10.0,
        "is_10b5_1": False,
        "transaction_id": "joint-purchase",
    }
    transactions = _insider_frame(
        [
            {**common, "insider_id": "one"},
            {**common, "insider_id": "two"},
        ]
    )

    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=_filing_frame([]),
        insider_transactions=transactions,
        coverage=_coverage("ABC"),
    ).row(0, named=True)

    assert result["form4_open_market_buy_usd"] == 1_000.0
    assert result["form4_non_10b5_1_buy_usd"] == 1_000.0
    assert result["distinct_insider_buyers"] == 2
    assert result["cluster_buy"] is True


def test_after_close_form4_is_not_visible_until_next_session() -> None:
    after_close = market_session_close(DECISION_DATE) + timedelta(minutes=1)
    transactions = _insider_frame(
        [
            {
                "ticker": "ABC",
                "available_at": after_close,
                "accepted_at": after_close,
                "transaction_code": "P",
                "shares": 100.0,
                "price": 10.0,
                "insider_id": "one",
                "is_10b5_1": False,
                "transaction_id": "1",
            }
        ]
    )
    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=_filing_frame([]),
        insider_transactions=transactions,
        coverage=_coverage("ABC"),
    ).row(0, named=True)
    assert result["form4_open_market_buy_usd"] == 0.0


def test_partial_coverage_never_turns_unknown_into_zero() -> None:
    partial_start = _window()[1]
    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=_filing_frame([]),
        insider_transactions=_insider_frame([]),
        coverage=_coverage("ABC", start=partial_start),
    ).row(0, named=True)

    assert result["filings_coverage"] == "partial"
    assert result["insider_coverage"] == "partial"
    assert result["sec_coverage"] == "partial"
    assert result["material_event_count"] is None
    assert result["distinct_insider_buyers"] is None
    assert result["filings_coverage_start"] == partial_start


def test_provider_is_injected_with_requested_range() -> None:
    class Provider:
        def __init__(self) -> None:
            self.call: tuple[tuple[str, ...], date, date] | None = None

        def load(
            self,
            tickers: tuple[str, ...],
            start: date,
            end: date,
        ) -> SecSignalDataset:
            self.call = (tickers, start, end)
            return SecSignalDataset(
                filings=_filing_frame([]),
                insider_transactions=_insider_frame([]),
                coverage=_coverage(*tickers, start=start, end=end),
            )

    provider = Provider()
    result = build_sec_signal_features(
        _decisions("abc", "XYZ"),
        provider=provider,
    )

    assert provider.call == (("ABC", "XYZ"), _window()[0], DECISION_DATE)
    assert result["sec_coverage"].to_list() == ["complete"] * 2


def test_rejects_duplicate_normalized_decision_keys() -> None:
    with pytest.raises(ValueError, match="duplicate normalized ticker/date"):
        build_sec_signal_features(_decisions("abc", "ABC"))


def test_large_panel_reuses_date_and_coverage_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_dates = market_dates_between(
        DECISION_DATE - timedelta(days=180),
        DECISION_DATE,
    )[-100:]
    tickers = [f"T{index:03d}" for index in range(100)]
    decisions = pl.DataFrame(
        {
            "ticker": [ticker for ticker in tickers for _decision_date in reversed(decision_dates)],
            "decision_date": [
                decision_date for _ticker in tickers for decision_date in reversed(decision_dates)
            ],
        }
    )
    coverage_start = market_dates_between(
        decision_dates[0] - timedelta(days=120),
        decision_dates[0],
    )[-60]
    coverage = pl.DataFrame(
        [
            {
                "ticker": ticker,
                "source": source,
                "coverage_start": coverage_start,
                "coverage_end": decision_dates[-1],
                "status": "complete",
            }
            for ticker in tickers
            for source in ("filings", "insider_transactions")
        ]
    )

    session_window_calls: list[date] = []
    coverage_status_calls: list[date] = []
    original_session_window = sec_signals._session_window
    original_coverage_status = sec_signals._coverage_status

    def counted_session_window(decision_date: date, count: int) -> list[date]:
        session_window_calls.append(decision_date)
        return original_session_window(decision_date, count)

    def counted_coverage_status(
        signature: sec_signals.CoverageSignature,
        sessions: tuple[date, ...],
    ) -> str:
        coverage_status_calls.append(sessions[-1])
        return original_coverage_status(signature, sessions)

    monkeypatch.setattr(sec_signals, "_session_window", counted_session_window)
    monkeypatch.setattr(sec_signals, "_coverage_status", counted_coverage_status)

    result = build_sec_signal_features(
        decisions,
        filings=_filing_frame([]),
        insider_transactions=_insider_frame([]),
        coverage=coverage,
    )

    assert result.height == 10_000
    assert result.select("ticker", "decision_date").n_unique() == 10_000
    assert session_window_calls == list(reversed(decision_dates))
    assert coverage_status_calls == list(reversed(decision_dates))
    assert result.select("ticker", "decision_date").rows() == decisions.rows()


def test_rejects_naive_availability_timestamps() -> None:
    filings = pl.DataFrame(
        {
            "ticker": ["ABC"],
            "accepted_at": [datetime(2025, 3, 14, 12)],
            "form": ["8-K"],
            "items": ["2.02"],
        }
    )
    with pytest.raises(ValueError, match="accepted_at must be timezone-aware"):
        build_sec_signal_features(
            _decisions("ABC"),
            filings=filings,
            insider_transactions=_insider_frame([]),
            coverage=_coverage("ABC"),
        )


def test_rejects_non_session_decision_date() -> None:
    decisions = pl.DataFrame({"ticker": ["ABC"], "decision_date": [date(2025, 3, 15)]})
    with pytest.raises(ValueError, match="not an NYSE session"):
        build_sec_signal_features(decisions)


def test_utc_iso_timestamps_are_supported() -> None:
    known_at = market_session_close(DECISION_DATE) - timedelta(hours=1)
    filings = pl.DataFrame(
        {
            "ticker": ["ABC"],
            "accepted_at": [known_at.astimezone(UTC).isoformat()],
            "form": ["8-K"],
            "items": ["2.02"],
        }
    )
    result = build_sec_signal_features(
        _decisions("ABC"),
        filings=filings,
        insider_transactions=_insider_frame([]),
        coverage=_coverage("ABC"),
    ).row(0, named=True)
    assert result["days_since_8k_earnings"] == 0
