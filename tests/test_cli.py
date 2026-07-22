from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from click.testing import CliRunner

from alphascreener.cli import _ledger_outcome_requirements, cli
from alphascreener.data.sync import SyncResult
from alphascreener.market_calendar import latest_completed_market_date, market_dates_between
from alphascreener.prediction_contract import STRATEGY_VERSION
from alphascreener.ranking import rank_candidates

_DEFAULT_TICKERS = (
    ("SPY", 1.001),
    ("WIN", 1.01),
    *((f"T{index}", 1.002 + index / 10_000) for index in range(10)),
)


def _session_dates(sessions: int) -> list[date]:
    end = latest_completed_market_date()
    return market_dates_between(
        end - timedelta(days=sessions * 3),
        end,
    )[-sessions:]


def _backtest_records(
    days: int = 30,
    *,
    invalid_indexes: set[int] | None = None,
) -> pl.DataFrame:
    invalid_indexes = invalid_indexes or set()
    start = date(2025, 1, 1)
    rows = []
    for index in range(days):
        invalid = index in invalid_indexes
        rows.append({
            "strategy_version": STRATEGY_VERSION,
            "decision_date": start + timedelta(days=index),
            "result_date": start + timedelta(days=14 + index),
            "status": "INVALID" if invalid else "VALID",
            "invalid_reason": "missing_outcomes" if invalid else None,
            "universe_size": 100,
            "outcome_coverage": 0.5 if invalid else 1.0,
            "hits_at_10": None if invalid else 1,
            "precision_at_10": None if invalid else 0.1,
            "base_explosion_rate": None if invalid else 0.05,
            "passed": None if invalid else True,
            "universe_source": "current-directory",
        })
    return pl.DataFrame(
        rows,
        schema={
            "strategy_version": pl.String,
            "decision_date": pl.Date,
            "result_date": pl.Date,
            "status": pl.String,
            "invalid_reason": pl.String,
            "universe_size": pl.Int64,
            "outcome_coverage": pl.Float64,
            "hits_at_10": pl.Int64,
            "precision_at_10": pl.Float64,
            "base_explosion_rate": pl.Float64,
            "passed": pl.Boolean,
            "universe_source": pl.String,
        },
    )


def _market_data(
    tickers: tuple[tuple[str, float], ...] = _DEFAULT_TICKERS,
    *,
    sessions: int = 120,
) -> pl.DataFrame:
    rows = []
    dates = _session_dates(sessions)
    for ticker, growth in tickers:
        for index, session_date in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "dt": session_date,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })
    return pl.DataFrame(rows)


def _empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ticker": pl.String,
        "decision_date": pl.Date,
        "score": pl.Float64,
        "rank": pl.Int64,
        "strategy_version": pl.String,
        "universe_size": pl.Int64,
    })


def _stub_backtest(monkeypatch, records: pl.DataFrame | None = None) -> None:
    monkeypatch.setattr(
        "alphascreener.backtest.run_backtest",
        lambda data, *, days: records if records is not None else _backtest_records(days),
    )


def _stub_complete_sync(
    monkeypatch,
    ready_tickers: tuple[str, ...] = (),
) -> None:
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda **kwargs: SyncResult(0, 100, 100, (), ready_tickers),
    )
    monkeypatch.setattr("alphascreener.cli._ledger_outcome_requirements", lambda: ())


def _stub_screen_extras(monkeypatch, records: pl.DataFrame | None = None) -> None:
    _stub_backtest(monkeypatch, records)
    monkeypatch.setattr("alphascreener.cli._render_matured_evidence", lambda data: None)


def test_outcome_sync_tickers_are_driven_by_exact_result_rows(monkeypatch) -> None:
    decision_date = market_dates_between(date(2025, 1, 2), date(2025, 2, 28))[0]
    result_date = market_dates_between(decision_date, date(2025, 3, 31))[14]
    ledger = pl.DataFrame({
        "ticker": ["PRESENT", "MISSING"],
        "decision_date": [decision_date, decision_date],
        "score": [2.0, 1.0],
        "rank": [1, 2],
        "strategy_version": [STRATEGY_VERSION, STRATEGY_VERSION],
        "universe_size": [2, 2],
    })
    observations = pl.DataFrame({
        "ticker": ["PRESENT"],
        "dt": [result_date],
    })
    monkeypatch.setattr("alphascreener.evaluation.read_prediction_ledger", lambda: ledger)
    monkeypatch.setattr(
        "alphascreener.data.io.scan_ohlcv",
        lambda: observations.lazy(),
    )

    assert _ledger_outcome_requirements() == (
        ("MISSING", decision_date, result_date),
    )


def test_help_exposes_only_backtest_subcommand() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "backtest" in result.output
    assert "evaluate" not in result.output
    assert "optimize" not in result.output
    assert "sync" not in result.output


def test_rank_candidates_uses_the_60_session_window() -> None:
    rows = []
    dates = market_dates_between(date(2025, 1, 2), date(2025, 5, 1))[:60]
    for ticker, growth in [("SPY", 1.02), ("WIN", 1.01)]:
        for index, session_date in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "dt": session_date,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 * growth**index,
                "volume": 2_000_000,
            })

    ranked, cutoff = rank_candidates(pl.DataFrame(rows))

    assert cutoff == dates[-1]
    assert ranked.item(0, "ticker") == "WIN"


def test_rank_candidates_requires_a_current_spy_benchmark() -> None:
    rows = []
    dates = market_dates_between(date(2025, 1, 2), date(2025, 5, 1))[:60]
    for index, session_date in enumerate(dates):
        rows.append({
            "ticker": "WIN",
            "dt": session_date,
            "close": 100.0 * 1.01**index,
            "volume": 2_000_000,
        })

    with pytest.raises(ValueError, match="SPY benchmark unavailable"):
        rank_candidates(pl.DataFrame(rows))


def test_top_limits_display_but_ledger_receives_full_ranking_and_30_day_summary(
    monkeypatch,
) -> None:
    seen_backtest_days = []
    monkeypatch.setattr(
        "alphascreener.backtest.run_backtest",
        lambda data, *, days: seen_backtest_days.append(days) or _backtest_records(days),
    )
    monkeypatch.setattr("alphascreener.cli._render_matured_evidence", lambda data: None)
    current = (
        ("SPY", 1.001),
        ("WIN", 1.01),
        ("SECOND", 1.005),
        *((f"C{index}", 1.003 + index / 10_000) for index in range(8)),
    )
    _stub_complete_sync(monkeypatch, tuple(ticker for ticker, _ in current))
    data = _market_data((*current,
        ("STALE", 1.02),
    ))
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli, ["--top", "1"])

    assert result.exit_code == 0
    assert seen_backtest_days == [30]
    assert ledger_writes[0].height == 10
    assert ledger_writes[0]["universe_size"].unique().to_list() == [10]
    assert "STALE" not in ledger_writes[0]["ticker"].to_list()
    assert "Valid days: 30/30" in result.output
    assert "CURRENT_UNIVERSE_DIAGNOSTIC" in result.output


def test_incomplete_sync_does_not_record_a_ranking(monkeypatch) -> None:
    ledger_writes = []
    scan_calls = []
    monkeypatch.setattr("alphascreener.cli._ledger_outcome_requirements", lambda: ())
    monkeypatch.setattr(
        "alphascreener.data.io.scan_ohlcv",
        lambda: scan_calls.append(True),
    )
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda **kwargs: SyncResult(10, 100, 50, ("FAILED",)),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert ledger_writes == []
    assert scan_calls == []
    assert "Sync incomplete" in result.output


def test_stale_sync_does_not_read_cache_or_record_a_ranking(monkeypatch) -> None:
    scan_calls = []
    ledger_writes = []
    monkeypatch.setattr("alphascreener.cli._ledger_outcome_requirements", lambda: ())
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda **kwargs: SyncResult(
            0,
            100,
            100,
            (),
            (),
            (),
            date(2025, 1, 2),
            False,
        ),
    )
    monkeypatch.setattr(
        "alphascreener.data.io.scan_ohlcv",
        lambda: scan_calls.append(True),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert scan_calls == []
    assert ledger_writes == []
    assert "Market data is stale" in result.output


def test_same_day_rerun_displays_the_immutable_recorded_ranking(monkeypatch) -> None:
    _stub_screen_extras(monkeypatch)
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    decision_date = data["dt"].max()
    recorded = pl.DataFrame({
        "ticker": ["SAVED"],
        "decision_date": [decision_date],
        "score": [42.0],
        "rank": [1],
        "strategy_version": [STRATEGY_VERSION],
        "universe_size": [1],
    })
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr("alphascreener.evaluation.read_prediction_ledger", lambda: recorded)

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "SAVED" in result.output
    assert "WIN" not in result.output
    assert "immutable ranking already recorded" in result.output


def test_ledger_write_failure_is_clear_but_does_not_hide_candidates(monkeypatch) -> None:
    _stub_screen_extras(monkeypatch)
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "WIN" in result.output
    assert "Ranking was not recorded" in result.output
    assert "NOT RECORDED" in result.output
    assert "disk full" in result.output


def test_invalid_backtest_dates_do_not_block_current_ranking(monkeypatch) -> None:
    _stub_screen_extras(monkeypatch, _backtest_records(30, invalid_indexes={4}))
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "WIN" in result.output
    assert ledger_writes
    assert "Valid days: 29/30" in result.output
    assert "Invalid: 1" in result.output
    assert "Overall status: INCONCLUSIVE" in result.output
    assert "INVALID 2025-01-05: missing_outcomes" in result.output


def test_backtest_exception_does_not_block_current_ranking(monkeypatch) -> None:
    _stub_complete_sync(monkeypatch)
    monkeypatch.setattr("alphascreener.cli._render_matured_evidence", lambda data: None)
    data = _market_data()
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.backtest.run_backtest",
        lambda data, *, days: (_ for _ in ()).throw(ValueError("history unavailable")),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "WIN" in result.output
    assert ledger_writes
    assert "Historical backtest is unavailable" in result.output
    assert "history unavailable" in result.output


def test_cli_does_not_record_a_ranking_without_spy(monkeypatch) -> None:
    _stub_complete_sync(monkeypatch)
    data = _market_data((("WIN", 1.01),))
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "SPY benchmark unavailable" in result.output
    assert ledger_writes == []


def test_cli_does_not_record_fewer_than_top_10_candidates(monkeypatch) -> None:
    _stub_complete_sync(monkeypatch, ("SPY", "ONLY"))
    data = _market_data((("SPY", 1.001), ("ONLY", 1.01)))
    ledger_writes = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: ledger_writes.append(predictions),
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert ledger_writes == []
    assert "Need at least 10 candidates, found 1" in result.output


def test_no_saved_predictions_do_not_block_current_ranking(monkeypatch) -> None:
    _stub_backtest(monkeypatch)
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: None,
    )
    monkeypatch.setattr("alphascreener.evaluation.read_prediction_ledger", _empty_ledger)

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "WIN" in result.output
    assert "No saved predictions yet" in result.output


def test_asc_automatically_shows_matured_current_strategy_evidence(monkeypatch) -> None:
    _stub_backtest(monkeypatch)
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    market_dates = data.filter(pl.col("ticker") == "SPY")["dt"].sort().to_list()
    decision_dates = market_dates[-30:-25]
    ledger = pl.DataFrame([
        {
            "ticker": f"L{rank}",
            "decision_date": decision_date,
            "score": float(11 - rank),
            "rank": rank,
            "strategy_version": STRATEGY_VERSION,
            "universe_size": 10,
        }
        for decision_date in decision_dates
        for rank in range(1, 11)
    ])
    matured = ledger.with_columns(
        pl.lit(0.2).alias("forward_return"),
        pl.lit(True).alias("is_explosion"),
    )
    daily = pl.DataFrame({
        "strategy_version": [STRATEGY_VERSION] * 5,
        "decision_date": decision_dates,
        "universe_size": [10] * 5,
        "outcome_coverage": [1.0] * 5,
        "precision_at_k": [0.1] * 5,
        "base_explosion_rate": [0.05] * 5,
        "passed": [True] * 5,
    })
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: None,
    )
    monkeypatch.setattr("alphascreener.evaluation.read_prediction_ledger", lambda: ledger)
    monkeypatch.setattr(
        "alphascreener.evaluation.compute_forward_labels",
        lambda ohlcv: pl.DataFrame(),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.mature_predictions",
        lambda predictions, labels: matured,
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.evaluate_daily_rankings",
        lambda matured: daily,
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.longest_consecutive_passes",
        lambda metrics, market_dates, *, strategy_version: 5,
    )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "Matured prospective predictions" in result.output
    assert "Best consecutive passing days: 5/5" in result.output
    assert "Target reached: yes" in result.output


def test_matured_ledger_with_missing_original_top_rank_is_shown_as_invalid(
    monkeypatch,
) -> None:
    _stub_backtest(monkeypatch)
    _stub_complete_sync(monkeypatch)
    data = _market_data()
    market_dates = data.filter(pl.col("ticker") == "SPY")["dt"].sort().to_list()
    decision_date = market_dates[-20]
    result_date = market_dates[-6]
    data = data.filter(
        ~((pl.col("ticker") == "WIN") & (pl.col("dt") == result_date))
    )
    ledger_tickers = ["WIN", *(f"T{index}" for index in range(9))]
    ledger = pl.DataFrame({
        "ticker": ledger_tickers,
        "decision_date": [decision_date] * 10,
        "score": [float(10 - index) for index in range(10)],
        "rank": list(range(1, 11)),
        "strategy_version": [STRATEGY_VERSION] * 10,
        "universe_size": [10] * 10,
    })
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: None,
    )
    monkeypatch.setattr("alphascreener.evaluation.read_prediction_ledger", lambda: ledger)

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "Matured prospective predictions" in result.output
    assert "INVALID" in result.output
    assert "top_10_outcomes_incomplete" in result.output


@pytest.mark.parametrize("days", [0, 46])
def test_invalid_backtest_days_fail_before_sync(monkeypatch, days: int) -> None:
    sync_calls = []
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda **kwargs: sync_calls.append(True),
    )

    result = CliRunner().invoke(cli, ["backtest", "--days", str(days)])

    assert result.exit_code == 2
    assert sync_calls == []


@pytest.mark.parametrize(
    ("arguments", "expected_days"),
    [([], 30), (["--days", "1"], 1), (["--days", "45"], 45)],
)
def test_backtest_command_forwards_days_and_never_writes_ledger(
    monkeypatch,
    arguments: list[str],
    expected_days: int,
) -> None:
    _stub_complete_sync(monkeypatch)
    data = _market_data(sessions=1)
    seen_days = []
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.backtest.run_backtest",
        lambda ohlcv, *, days: seen_days.append(days) or _backtest_records(days),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: (_ for _ in ()).throw(AssertionError("ledger must not be written")),
    )

    result = CliRunner().invoke(cli, ["backtest", *arguments])

    assert result.exit_code == 0
    assert seen_days == [expected_days]
    assert "CURRENT_UNIVERSE_DIAGNOSTIC" in result.output
    assert f"strategy={STRATEGY_VERSION}" in result.output
    assert str(date(2025, 1, 1)) in result.output
    assert str(date(2025, 1, 1) + timedelta(days=expected_days - 1)) in result.output


def test_backtest_command_displays_invalid_reason_and_continues(monkeypatch) -> None:
    _stub_complete_sync(monkeypatch)
    data = _market_data(sessions=1)
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    monkeypatch.setattr(
        "alphascreener.backtest.run_backtest",
        lambda ohlcv, *, days: _backtest_records(days, invalid_indexes={0}),
    )

    result = CliRunner().invoke(cli, ["backtest", "--days", "2"])

    assert result.exit_code == 0
    assert "missing_outcomes" in result.output
    assert "2025-01-01" in result.output
    assert "2025-01-02" in result.output
    assert "Valid days: 1/2" in result.output
    assert "Overall status: INCONCLUSIVE" in result.output


def test_screen_keeps_current_directory_and_ledger_outcome_panels_separate(
    monkeypatch,
) -> None:
    data = _market_data((
        ("SPY", 1.001),
        ("CURRENT", 1.01),
        ("HISTORICAL", 1.005),
        ("DELISTED", 0.999),
    ))
    as_of_date = data["dt"].max()
    future_row = data.head(1).with_columns(
        pl.lit("FUTURE").alias("ticker"),
        pl.lit(as_of_date + timedelta(days=1)).cast(pl.Date).alias("dt"),
    )
    data = pl.concat([data, future_row])
    monkeypatch.setattr(
        "alphascreener.data.sync.sync_ohlcv",
        lambda **kwargs: SyncResult(
            0,
            100,
            99,
            ("HISTORICAL",),
            ("CURRENT", "SPY"),
            ("CURRENT", "HISTORICAL", "SPY"),
            as_of_date,
            True,
        ),
    )
    monkeypatch.setattr("alphascreener.data.io.scan_ohlcv", lambda: data.lazy())
    ranking = pl.DataFrame({
        "ticker": [f"R{rank}" for rank in range(1, 11)],
        "score": [float(11 - rank) for rank in range(1, 11)],
        "rank": list(range(1, 11)),
    })
    monkeypatch.setattr(
        "alphascreener.cli.rank_candidates",
        lambda ohlcv: (ranking, date.today()),
    )
    monkeypatch.setattr(
        "alphascreener.evaluation.write_prediction_ledger",
        lambda predictions: None,
    )
    seen: dict[str, set[str]] = {}

    def capture_backtest(ohlcv, *, days):
        seen["backtest"] = set(ohlcv["ticker"].unique().to_list())
        return _backtest_records(days)

    def capture_evidence(ohlcv):
        seen["evidence"] = set(ohlcv["ticker"].unique().to_list())

    monkeypatch.setattr("alphascreener.backtest.run_backtest", capture_backtest)
    monkeypatch.setattr("alphascreener.cli._render_matured_evidence", capture_evidence)

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert seen["backtest"] == {"SPY", "CURRENT", "HISTORICAL"}
    assert seen["evidence"] == {"SPY", "CURRENT", "HISTORICAL", "DELISTED"}
