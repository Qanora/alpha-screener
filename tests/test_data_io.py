"""Tests for Data I/O layer: Parquet read/write, paths, partitioning, archive.

Issue #86: Data I/O layer.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alphascreener.data import (
    DATA_CATEGORIES,
    archive_old_data,
    get_archive_dir,
    get_data_dir,
    get_data_home,
    get_partition_path,
    read_parquet,
    scan_parquet,
    write_parquet,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Small OHLCV-like DataFrame for IO round-trip testing."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
            "dt": [
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ],
            "open": [150.0, 151.0, 140.0, 141.5],
            "high": [152.0, 153.0, 142.0, 143.0],
            "low": [149.0, 150.0, 139.0, 140.5],
            "close": [151.5, 152.5, 141.0, 142.0],
            "volume": [100_000, 120_000, 200_000, 210_000],
        }
    )


@pytest.fixture
def multi_date_df() -> pl.DataFrame:
    """DataFrame spanning multiple dates for partition tests."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL"] * 3,
            "dt": [date(2025, 2, 1), date(2025, 3, 15), date(2025, 4, 28)],
            "value": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def multi_month_df() -> pl.DataFrame:
    """DataFrame spanning multiple months for monthly partition tests."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL"] * 3,
            "dt": [date(2025, 1, 15), date(2025, 2, 20), date(2025, 3, 10)],
            "value": [10.0, 20.0, 30.0],
        }
    )


# ============================================================================
# Path utilities
# ============================================================================


class TestGetDataHome:
    """~/.alphascreener path resolution."""

    def test_returns_expanded_path(self):
        result = get_data_home()
        assert isinstance(result, Path)
        assert result == Path.home() / ".alphascreener"


class TestGetDataDir:
    """Per-category data directory resolution."""

    @pytest.mark.parametrize("category", DATA_CATEGORIES)
    def test_returns_path_under_data_home(self, category):
        result = get_data_dir(category)
        assert isinstance(result, Path)
        assert result == get_data_home() / "data" / category

    def test_rejects_unknown_category(self):
        with pytest.raises(ValueError, match="Unknown data category"):
            get_data_dir("unknown_cat")


class TestGetPartitionPath:
    """Partition path construction (daily / monthly)."""

    def test_ohlcv_daily_partition(self):
        result = get_partition_path("ohlcv", date(2025, 3, 15))
        expected = get_data_home() / "data" / "ohlcv" / "dt=2025-03-15"
        assert result == expected

    def test_factors_daily_partition(self):
        result = get_partition_path("factors", date(2025, 6, 1))
        expected = get_data_home() / "data" / "factors" / "dt=2025-06-01"
        assert result == expected

    def test_signals_daily_partition(self):
        result = get_partition_path("signals", date(2025, 12, 31))
        expected = get_data_home() / "data" / "signals" / "dt=2025-12-31"
        assert result == expected

    def test_backtest_monthly_partition(self):
        result = get_partition_path("backtest", date(2025, 3, 15))
        expected = get_data_home() / "data" / "backtest" / "dt=2025-03"
        assert result == expected

    def test_ohlcv_partition_from_string(self):
        result = get_partition_path("ohlcv", "2025-03-15")
        expected = get_data_home() / "data" / "ohlcv" / "dt=2025-03-15"
        assert result == expected

    def test_backtest_partition_from_string(self):
        result = get_partition_path("backtest", "2025-03-15")
        expected = get_data_home() / "data" / "backtest" / "dt=2025-03"
        assert result == expected


class TestGetArchiveDir:
    """Archive directory path."""

    def test_returns_archive_path(self):
        result = get_archive_dir()
        assert result == get_data_home() / "archive"


# ============================================================================
# Write + Read round-trip (daily partition: ohlcv)
# ============================================================================


class TestParquetRoundTrip:
    """Write then read back: data integrity across round-trip."""

    def test_write_and_read_ohlcv(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")

        # Verify files created on disk
        ohlcv_dir = home / "data" / "ohlcv"
        assert ohlcv_dir.exists()
        partitions = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
        assert partitions == ["dt=2025-01-02", "dt=2025-01-03"]

        # Round-trip read
        result = read_parquet("ohlcv").collect()
        # Order may differ; sort for comparison
        result_sorted = result.sort(["ticker", "dt"])
        expected_sorted = sample_df.sort(["ticker", "dt"])
        assert result_sorted.to_dict(as_series=False) == expected_sorted.to_dict(as_series=False)

    def test_write_and_read_signals(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "signals")

        signals_dir = home / "data" / "signals"
        partitions = sorted(p.name for p in signals_dir.iterdir() if p.is_dir())
        assert partitions == ["dt=2025-01-02", "dt=2025-01-03"]

        result = read_parquet("signals").collect().sort(["ticker", "dt"])
        expected = sample_df.sort(["ticker", "dt"])
        assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)


# ============================================================================
# Monthly partition (backtest)
# ============================================================================


class TestMonthlyPartition:
    """Backtest data uses dt=YYYY-MM monthly partitioning."""

    def test_write_and_read_monthly(self, tmp_path, multi_month_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(multi_month_df, "backtest")

        backtest_dir = home / "data" / "backtest"
        partitions = sorted(p.name for p in backtest_dir.iterdir() if p.is_dir())
        assert partitions == ["dt=2025-01", "dt=2025-02", "dt=2025-03"]

        result = read_parquet("backtest").collect().sort(["ticker", "dt"])
        expected = multi_month_df.sort(["ticker", "dt"])
        assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)

    def test_monthly_partition_aggregates_dates(self, tmp_path, monkeypatch):
        """Multiple dates within same month land in same partition."""
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "dt": [date(2025, 3, 5), date(2025, 3, 25)],
                "value": [1.0, 2.0],
            }
        )
        write_parquet(df, "backtest")

        backtest_dir = home / "data" / "backtest"
        partitions = [p.name for p in backtest_dir.iterdir() if p.is_dir()]
        assert partitions == ["dt=2025-03"]

        result = read_parquet("backtest").collect().sort("dt")
        assert result["value"].to_list() == [1.0, 2.0]


# ============================================================================
# LazyFrame / scan_parquet
# ============================================================================


class TestScanParquet:
    """scan_parquet returns a LazyFrame with predicate pushdown support."""

    def test_scan_returns_lazyframe(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")

        lf = scan_parquet("ohlcv")
        assert isinstance(lf, pl.LazyFrame)

        result = lf.collect().sort(["ticker", "dt"])
        expected = sample_df.sort(["ticker", "dt"])
        assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)

    def test_scan_with_date_filter(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")

        # Read only data from 2025-01-02
        lf = scan_parquet("ohlcv", date_filter=date(2025, 1, 2))
        result = lf.collect()
        assert result.height == 2  # AAPL + GOOGL
        assert result["dt"].to_list() == [date(2025, 1, 2)] * 2

    def test_scan_with_polars_predicate(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")

        lf = scan_parquet("ohlcv")
        result = lf.filter(pl.col("ticker") == "AAPL").collect()
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL"] * 2


# ============================================================================
# read_parquet
# ============================================================================


class TestReadParquet:
    """read_parquet reads all partitions and returns a LazyFrame."""

    def test_returns_lazyframe(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")
        lf = read_parquet("ohlcv")
        assert isinstance(lf, pl.LazyFrame)

    def test_read_empty_raises_or_empty(self, tmp_path, monkeypatch):
        """Reading a category with no data should raise FileNotFoundError."""
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        with pytest.raises(FileNotFoundError):
            read_parquet("ohlcv")


# ============================================================================
# write_parquet
# ============================================================================


class TestWriteParquet:
    """write_parquet creates partitioned Parquet files."""

    def test_creates_parquet_files(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(sample_df, "ohlcv")

        partition_dir = home / "data" / "ohlcv" / "dt=2025-01-02"
        parquet_files = list(partition_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_append_mode(self, tmp_path, monkeypatch):
        """Appending data to an existing partition should not lose existing data."""
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        df1 = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "dt": [date(2025, 1, 2)],
                "value": [1.0],
            }
        )
        df2 = pl.DataFrame(
            {
                "ticker": ["GOOGL"],
                "dt": [date(2025, 1, 2)],
                "value": [2.0],
            }
        )

        write_parquet(df1, "ohlcv")
        write_parquet(df2, "ohlcv")

        result = read_parquet("ohlcv").collect().sort("ticker")
        assert result.height == 2
        assert result["ticker"].to_list() == ["AAPL", "GOOGL"]

    def test_rejects_unknown_category(self, tmp_path, sample_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        with pytest.raises(ValueError, match="Unknown data category"):
            write_parquet(sample_df, "random_cat")

    def test_rejects_missing_dt_column(self, tmp_path, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        df_no_dt = pl.DataFrame({"ticker": ["AAPL"], "value": [1.0]})
        with pytest.raises(ValueError, match="must contain a 'dt' column"):
            write_parquet(df_no_dt, "ohlcv")


# ============================================================================
# Archive
# ============================================================================


class TestArchive:
    """Archive old data to zstd Parquet in ~/.alphascreener/archive/."""

    def test_archive_moves_old_partitions(self, tmp_path, multi_date_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(multi_date_df, "ohlcv")

        # Archive records before 2025-03-01
        archive_old_data("ohlcv", before_date=date(2025, 3, 1))

        ohlcv_dir = home / "data" / "ohlcv"
        remaining = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
        # dt=2025-02-01 should be archived, dt=2025-03-15 and dt=2025-04-28 remain
        assert "dt=2025-02-01" not in remaining
        assert "dt=2025-03-15" in remaining
        assert "dt=2025-04-28" in remaining

        # Archive should exist
        archive_dir = home / "archive" / "ohlcv"
        assert archive_dir.exists()
        archive_files = list(archive_dir.glob("**/*.parquet"))
        assert len(archive_files) >= 1

    def test_archive_uses_zstd_compression(self, tmp_path, multi_date_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(multi_date_df, "ohlcv")
        archive_old_data("ohlcv", before_date=date(2025, 3, 1))

        archive_dir = home / "archive" / "ohlcv"
        parquet_files = list(archive_dir.glob("**/*.parquet"))
        assert len(parquet_files) >= 1
        # Read archived data to verify it's valid Parquet
        archived = pl.read_parquet(str(parquet_files[0]))
        assert archived.height > 0

    def test_archive_noop_when_nothing_to_archive(self, tmp_path, multi_date_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(multi_date_df, "ohlcv")

        # Archive records before year 2020 — nothing should match
        archive_old_data("ohlcv", before_date=date(2020, 1, 1))

        ohlcv_dir = home / "data" / "ohlcv"
        remaining = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
        assert len(remaining) == 3  # all three partitions remain

    def test_archive_backtest_monthly(self, tmp_path, multi_month_df, monkeypatch):
        home = tmp_path / ".alphascreener"
        monkeypatch.setattr("alphascreener.data.paths.get_data_home", lambda: home)

        write_parquet(multi_month_df, "backtest")
        archive_old_data("backtest", before_date=date(2025, 2, 1))

        backtest_dir = home / "data" / "backtest"
        remaining = sorted(p.name for p in backtest_dir.iterdir() if p.is_dir())
        assert "dt=2025-01" not in remaining  # January archived
        assert "dt=2025-02" in remaining
        assert "dt=2025-03" in remaining


# ============================================================================
# DATA_CATEGORIES constant
# ============================================================================


class TestDataCategories:
    """Verify the DATA_CATEGORIES set."""

    def test_contains_expected_categories(self):
        assert "ohlcv" in DATA_CATEGORIES
        assert "factors" in DATA_CATEGORIES
        assert "signals" in DATA_CATEGORIES
        assert "backtest" in DATA_CATEGORIES

    def test_daily_categories(self):
        """ohlcv, factors, signals use daily partitions."""
        for cat in ("ohlcv", "factors", "signals"):
            assert cat in DATA_CATEGORIES

    def test_monthly_category(self):
        """backtest uses monthly partitions."""
        assert "backtest" in DATA_CATEGORIES
