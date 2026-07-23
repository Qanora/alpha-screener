"""Tests for batched point-in-time ranking."""

from __future__ import annotations

from datetime import date

import polars as pl

from alphascreener.market_calendar import infer_market_dates, market_dates_between
from alphascreener.ranking import rank_candidate_dates, score_rank_v6


def _market_dates(count: int) -> list[date]:
    dates = market_dates_between(date(2025, 1, 2), date(2025, 12, 31))
    assert len(dates) >= count
    return dates[:count]


def _rows(sessions: int = 61) -> list[dict[str, object]]:
    market_dates = _market_dates(sessions)
    rows: list[dict[str, object]] = []
    for ticker, growth, volume in [
        ("SPY", 1.001, 3_000_000),
        ("HIGH", 1.01, 2_000_000),
        ("MID", 1.008, 1_000_000),
        ("LOW", 1.02, 50_000),
    ]:
        rows.extend(
            {
                "ticker": ticker,
                "dt": market_date,
                "close": 100.0 * growth**index,
                "volume": volume,
            }
            for index, market_date in enumerate(market_dates)
        )
    return rows


def test_future_rows_do_not_change_an_earlier_ranking() -> None:
    data = pl.DataFrame(_rows())
    decision_date = _market_dates(61)[-2]

    before = rank_candidate_dates(
        data.filter(pl.col("dt") <= decision_date),
        [decision_date],
    )
    after = rank_candidate_dates(data, [decision_date])

    assert before.equals(after)


def test_future_cross_section_growth_does_not_change_past_calendar_or_ranking() -> None:
    market_dates = _market_dates(61)
    decision_date = market_dates[-2]
    history = pl.DataFrame(_rows(60))
    future = pl.DataFrame([
        {
            "ticker": f"NEW{index:04d}",
            "dt": market_dates[-1],
            "close": 100.0,
            "volume": 5_000_000,
        }
        for index in range(500)
    ])
    expanded = pl.concat([history, future])

    before = rank_candidate_dates(history, [decision_date])
    after = rank_candidate_dates(expanded, [decision_date])

    assert infer_market_dates(history) == infer_market_dates(expanded)[:-1]
    assert before.equals(after)


def test_ranking_prefilter_keeps_only_the_most_liquid_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphascreener.ranking.MAX_CANDIDATES",
        2,
    )

    ranking = rank_candidate_dates(
        pl.DataFrame(_rows(60)),
        [_market_dates(60)[-1]],
    )

    assert ranking.height == 2
    assert set(ranking["ticker"].to_list()) == {"HIGH", "MID"}


def test_absolute_price_prefilter_uses_unadjusted_close() -> None:
    rows = _rows(60)
    for row in rows:
        row["raw_close"] = 1.0 if row["ticker"] == "HIGH" else row["close"]

    ranking = rank_candidate_dates(
        pl.DataFrame(rows),
        [_market_dates(60)[-1]],
    )

    assert "HIGH" not in ranking["ticker"].to_list()
    assert "MID" in ranking["ticker"].to_list()


def test_rank_v6_is_a_frozen_duplicate_weighted_baseline() -> None:
    decision_date = date(2025, 1, 2)
    candidates = pl.DataFrame({
        "ticker": ["A", "B", "C"],
        "dt": [decision_date] * 3,
        "return_5d": [3.0, 2.0, 1.0],
        "return_20d": [1.0, 3.0, 2.0],
        "distance_to_60d_high": [-0.1, 0.0, -0.2],
        "volume_zscore_20": [2.0, 1.0, 3.0],
        "relative_strength_20d": [1.0, 3.0, 2.0],
    })

    ranked = score_rank_v6(candidates)

    assert ranked["ticker"].to_list() == ["B", "A", "C"]
    assert ranked["score"].to_list() == [2.4, 1.8, 1.8]
    assert ranked["rank"].to_list() == [1, 2, 3]
