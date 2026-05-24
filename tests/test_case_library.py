"""Tests for the breakout case library builder.

Issue #190: Breakout case library initialization.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from alphascreener.tradingagents.case_library import (
    CaseLibraryBuilder,
    rebuild_case_library,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_factor_df(
    n_rows: int = 10,
    *,
    with_breakout_score: bool = True,
    with_t7_return: bool = True,
    with_z_capped: bool = True,
) -> pl.DataFrame:
    """Build a synthetic factor DataFrame with all required columns."""
    import random
    from datetime import date as _date

    random.seed(42)

    data: dict = {
        "ticker": pl.Series("ticker", [f"TICKER_{i}" for i in range(n_rows)]),
        "dt": pl.Series("dt", [_date(2025, 6, i + 1) for i in range(n_rows)], dtype=pl.Date),
        "close": pl.Series("close", [100.0 + i * 5 for i in range(n_rows)], dtype=pl.Float64),
    }

    if with_breakout_score:
        data["breakout_score"] = pl.Series(
            "breakout_score", [random.uniform(-1.0, 3.0) for _ in range(n_rows)],
            dtype=pl.Float64,
        )

    if with_t7_return:
        data["t7_return"] = pl.Series(
            "t7_return", [random.uniform(-0.05, 0.25) for _ in range(n_rows)],
            dtype=pl.Float64,
        )

    if with_z_capped:
        for col in ["z_capped_MOM_5D", "z_capped_PTH", "z_capped_MOM_SLOPE",
                     "z_capped_BB_SQUEEZE", "z_capped_ATR_RATIO", "z_capped_MFI_14",
                     "z_capped_CMF_21", "z_capped_VOL_ANOMALY",
                     "z_capped_RSI_OVERSOLD", "z_capped_REV_ACCEL"]:
            data[col] = pl.Series(
                col, [random.uniform(-1.5, 1.5) for _ in range(n_rows)],
                dtype=pl.Float64,
            )

    return pl.DataFrame(data)


# ============================================================================
# Tests
# ============================================================================


class TestCaseLibraryBuilder:
    """Unit tests for CaseLibraryBuilder."""

    def test_rebuild_creates_parquet(self):
        """rebuild() writes cases.parquet with the correct schema."""
        df = _make_factor_df(n_rows=20)
        # Ensure at least some rows meet thresholds
        df = df.with_columns(
            pl.when(pl.col("ticker").str.contains("0|1|5"))
            .then(2.5).otherwise(pl.col("breakout_score")).alias("breakout_score"),
            pl.when(pl.col("ticker").str.contains("0|1|5"))
            .then(0.15).otherwise(pl.col("t7_return")).alias("t7_return"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"

            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )

            # Patch _load_all_factors to return our synthetic data
            builder._load_all_factors = lambda: df

            n = builder.rebuild()
            assert n > 0
            assert output.exists()

            # Verify schema
            result = pl.read_parquet(str(output))
            assert result.height == n
            assert "ticker" in result.columns
            assert "date" in result.columns
            assert "actual_pnl" in result.columns
            # Check f_* columns
            f_cols = [c for c in result.columns if c.startswith("f_")]
            assert len(f_cols) == 10

    def test_rebuild_empty_when_no_factors(self):
        """rebuild() returns 0 when no factor data is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(output_path=output)
            builder._load_all_factors = lambda: None

            n = builder.rebuild()
            assert n == 0
            assert output.exists()
            df = pl.read_parquet(str(output))
            assert df.height == 0

    def test_rebuild_empty_when_no_positives(self):
        """rebuild() returns 0 when no rows meet thresholds."""
        df = _make_factor_df(n_rows=5)
        df = df.with_columns(
            pl.lit(0.0).alias("breakout_score"),
            pl.lit(-0.1).alias("t7_return"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )
            builder._load_all_factors = lambda: df

            n = builder.rebuild()
            assert n == 0
            assert output.exists()
            df = pl.read_parquet(str(output))
            assert df.height == 0

    def test_select_positive_cases_filters_correctly(self):
        """_select_positive_cases enforces both score and return thresholds."""
        df = _make_factor_df(n_rows=10)
        # Manually set: first 3 rows get high score+return, rest get low values
        bscore = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t7ret = [0.15, 0.15, 0.15, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        df = df.with_columns([
            pl.Series("breakout_score", bscore),
            pl.Series("t7_return", t7ret),
        ])

        builder = CaseLibraryBuilder(breakout_score_pct=0.5, min_return=0.10)
        cases = builder._select_positive_cases(df)
        assert cases.height == 3

    def test_to_case_schema_maps_columns(self):
        """_to_case_schema correctly maps z_capped_* columns to f_* columns."""
        df = _make_factor_df(n_rows=3)

        builder = CaseLibraryBuilder()
        result = builder._to_case_schema(df)

        assert "ticker" in result.columns
        assert "date" in result.columns
        assert "actual_pnl" in result.columns
        assert "f_mom_5d" in result.columns
        assert "f_pth" in result.columns
        assert "f_mom_slope" in result.columns
        assert "f_bb_squeeze" in result.columns
        assert "f_atr_ratio" in result.columns
        assert "f_mfi_14" in result.columns
        assert "f_cmf_21" in result.columns
        assert "f_vol_anomaly" in result.columns
        assert "f_rsi_ovs" in result.columns
        assert "f_rev_accel" in result.columns

        # Verify date is string type
        assert result["date"].dtype == pl.String
        # Verify actual_pnl equals t7_return
        for row in df.iter_rows(named=True):
            tsv = row["t7_return"]
            matching = result.filter(pl.col("ticker") == row["ticker"])
            assert abs(matching["actual_pnl"][0] - tsv) < 1e-10

    def test_to_case_schema_handles_missing_t7_return(self):
        """_to_case_schema falls back to actual_pnl=0.0 when t7_return missing."""
        df = _make_factor_df(n_rows=3, with_t7_return=False)
        # t7_return column already absent (with_t7_return=False)

        builder = CaseLibraryBuilder()
        result = builder._to_case_schema(df)
        assert result["actual_pnl"].to_list() == [0.0, 0.0, 0.0]

    def test_status_on_empty_library(self):
        """status() reports exists=False when file is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nonexistent" / "cases.parquet"
            builder = CaseLibraryBuilder(output_path=output)
            info = builder.status()
            assert info["exists"] is False
            assert info["n_cases"] == 0

    def test_status_on_populated_library(self):
        """status() reports correct counts after rebuild."""
        df = _make_factor_df(n_rows=10)
        df = df.with_columns(
            pl.lit(2.0).alias("breakout_score"),
            pl.lit(0.15).alias("t7_return"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )
            builder._load_all_factors = lambda: df
            builder.rebuild()

            info = builder.status()
            assert info["exists"] is True
            assert info["n_cases"] == 10
            assert info["n_unique_tickers"] == 10
            assert info["date_range"] is not None

    def test_append_date_does_not_duplicate(self):
        """append_date() upserts: same ticker+date replaces, not duplicates."""
        df = _make_factor_df(n_rows=5)
        df = df.with_columns(
            pl.lit(2.0).alias("breakout_score"),
            pl.lit(0.15).alias("t7_return"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )
            builder._load_all_factors = lambda: df
            builder.rebuild()
            first_count = pl.read_parquet(str(output)).height

            # Exercise append_date with same ticker+date data
            target_dt = df["dt"][0]
            same_day = df.filter(pl.col("dt") == target_dt)
            builder.append_date(target_dt, df=same_day)
            second_count = pl.read_parquet(str(output)).height

            assert first_count == second_count

    def test_append_date_adds_new_cases(self):
        """append_date() adds new cases for a new date."""
        from datetime import date as _date

        df1 = _make_factor_df(n_rows=3)
        df1 = df1.with_columns(
            pl.lit(2.0).alias("breakout_score"),
            pl.lit(0.15).alias("t7_return"),
        )

        # Build df2 with distinct ticker+date
        new_dt = _date(2025, 7, 1)
        df2 = _make_factor_df(n_rows=3)
        df2 = df2.with_columns(
            pl.Series("ticker", ["TICKER_100", "TICKER_101", "TICKER_102"]),
            pl.Series("dt", [new_dt] * 3, dtype=pl.Date),
            pl.Series("breakout_score", [2.0, 2.0, 2.0], dtype=pl.Float64),
            pl.Series("t7_return", [0.15, 0.15, 0.15], dtype=pl.Float64),
            pl.Series("close", [105.0, 105.0, 105.0], dtype=pl.Float64),
        )
        # Re-assign z_capped cols
        for col in ["z_capped_MOM_5D", "z_capped_PTH", "z_capped_MOM_SLOPE",
                     "z_capped_BB_SQUEEZE", "z_capped_ATR_RATIO", "z_capped_MFI_14",
                     "z_capped_CMF_21", "z_capped_VOL_ANOMALY",
                     "z_capped_RSI_OVERSOLD", "z_capped_REV_ACCEL"]:
            df2 = df2.with_columns(pl.Series(col, [0.5, 0.5, 0.5], dtype=pl.Float64))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )

            # Write first batch
            builder._load_all_factors = lambda: df1
            builder.rebuild()
            first_count = pl.read_parquet(str(output)).height

            # Append second batch via append_date
            n_new = builder.append_date(new_dt, df=df2)
            assert n_new == 3

            merged = pl.read_parquet(str(output))
            # 3 original + 3 new = 6 (different tickers and dates)
            assert merged.height == first_count + n_new

    def test_select_positive_cases_handles_null_factor_vector(self):
        """Rows with null z_capped columns are excluded."""
        df = _make_factor_df(n_rows=5)
        df = df.with_columns(
            pl.lit(2.0).alias("breakout_score"),
            pl.lit(0.15).alias("t7_return"),
            # Set one z_capped column to null for all rows
            pl.lit(None, dtype=pl.Float64).alias("z_capped_MOM_5D"),
        )

        builder = CaseLibraryBuilder(breakout_score_pct=0.5, min_return=0.10)
        cases = builder._select_positive_cases(df)
        assert cases.height == 0  # All rows excluded due to null factor

    def test_rebuild_handles_missing_breakout_score_column(self):
        """rebuild() computes breakout_score if the column is missing."""
        df = _make_factor_df(n_rows=10, with_breakout_score=False)
        df = df.with_columns(pl.lit(0.15).alias("t7_return"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.0,
                min_return=0.10,
                output_path=output,
            )
            builder._load_all_factors = lambda: df
            n = builder.rebuild()
            # breakout_score_pct=0.0 ensures all rows pass the percentile filter;
            # with t7_return=0.15 > min_return=0.10, we expect positive cases
            assert n > 0
            assert output.exists()
            result = pl.read_parquet(str(output))
            assert result.height == n

    def test_rebuild_computes_forward_returns_when_t7_missing(self):
        """rebuild() calls _compute_forward_returns when t7_return is absent."""
        df = _make_factor_df(n_rows=6, with_t7_return=False)
        df = df.with_columns(pl.lit(2.0).alias("breakout_score"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cases.parquet"
            builder = CaseLibraryBuilder(
                breakout_score_pct=0.5,
                min_return=0.10,
                output_path=output,
            )
            builder._load_all_factors = lambda: df

            called = {"v": False}
            def _fake_forward(x: pl.DataFrame) -> pl.DataFrame:
                called["v"] = True
                return x.with_columns(pl.lit(0.15).alias("t7_return"))
            builder._compute_forward_returns = _fake_forward

            n = builder.rebuild()
            assert called["v"] is True
            assert n > 0


class TestRebuildConvenience:
    """Tests for the rebuild_case_library convenience function."""

    def test_returns_zero_when_no_data(self, monkeypatch):
        """rebuild_case_library returns 0 when scan_parquet raises FileNotFoundError."""
        # The convenience function calls CaseLibraryBuilder().rebuild()
        # which calls _load_all_factors -> scan_parquet("factors")
        # We patch scan_parquet to raise FileNotFoundError to test the
        # real exception handling path in _load_all_factors.
        import alphascreener.data.io as io

        def _raise(*args, **kwargs):
            raise FileNotFoundError("No data")

        monkeypatch.setattr(io, "scan_parquet", _raise)
        import alphascreener.tradingagents.case_library as cl
        n = cl.rebuild_case_library()
        assert n == 0
