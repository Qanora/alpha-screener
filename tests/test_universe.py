"""Tests for Universe management: whitelist, pre-filters, monthly refresh, meta cache.

Issue #88: Universe management - SP500 + Russell 1000.
"""

from datetime import date

import polars as pl
import pytest

# ============================================================================
# Pre-filter tests (pure logic, no external deps)
# ============================================================================


@pytest.fixture
def candidate_df() -> pl.DataFrame:
    """Realistic candidate universe DataFrame for pre-filter testing."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "MEGA", "SMALL", "CHEAP", "NEWIPO", "HALTED", "DELISTED", "LOWVOL"],
            "avg_dollar_volume_20d": [
                80_000_000_000,  # AAPL - passes
                50_000_000,  # MEGA - passes
                25_000_000,  # SMALL - passes
                15_000_000,  # CHEAP - passes
                30_000_000,  # NEWIPO - passes volume but fails days_listed
                22_000_000,  # HALTED - passes volume but fails status
                21_000_000,  # DELISTED - passes volume but fails status
                18_000_000,  # LOWVOL - FAILS avg_dollar_volume
            ],
            "market_cap": [
                3_500_000_000_000,  # AAPL
                500_000_000,  # MEGA
                400_000_000,  # SMALL
                350_000_000,  # CHEAP
                500_000_000,  # NEWIPO
                450_000_000,  # HALTED
                350_000_000,  # DELISTED
                250_000_000,  # LOWVOL - FAILS market_cap
            ],
            "last_price": [
                190.0,  # AAPL
                80.0,  # MEGA
                25.0,  # SMALL
                3.5,  # CHEAP - FAILS price
                30.0,  # NEWIPO
                15.0,  # HALTED
                10.0,  # DELISTED
                8.0,  # LOWVOL
            ],
            "days_listed": [
                3600,  # AAPL
                2000,  # MEGA
                800,  # SMALL
                1500,  # CHEAP
                120,  # NEWIPO - FAILS (< 252)
                1000,  # HALTED
                3000,  # DELISTED
                500,  # LOWVOL
            ],
            "status": [
                "Active",  # AAPL
                "Active",  # MEGA
                "Active",  # SMALL
                "Active",  # CHEAP
                "Active",  # NEWIPO
                "Halted",  # HALTED - FAILS status
                "Delisted",  # DELISTED - FAILS status
                "Active",  # LOWVOL
            ],
        }
    )


class TestPreFilter:
    """Pre-filter conditions: avg_dollar_volume, market_cap, price, days_listed, status."""

    def test_filter_by_avg_dollar_volume(self, candidate_df):
        """avg_dollar_volume_20d > 20M."""
        from alphascreener.universe.filters import DEFAULT_MIN_DOLLAR_VOLUME, _filter_dollar_volume

        result = _filter_dollar_volume(candidate_df, DEFAULT_MIN_DOLLAR_VOLUME)
        tickers = result["ticker"].to_list()
        assert "LOWVOL" not in tickers  # 18M < 20M
        assert "AAPL" in tickers
        assert "HALTED" in tickers  # volume passes, status fails later

    def test_filter_by_market_cap(self, candidate_df):
        """market_cap > 300M."""
        from alphascreener.universe.filters import DEFAULT_MIN_MARKET_CAP, _filter_market_cap

        result = _filter_market_cap(candidate_df, DEFAULT_MIN_MARKET_CAP)
        tickers = result["ticker"].to_list()
        assert "LOWVOL" not in tickers  # 250M < 300M
        assert "AAPL" in tickers

    def test_filter_by_price(self, candidate_df):
        """last_price > 5."""
        from alphascreener.universe.filters import DEFAULT_MIN_PRICE, _filter_price

        result = _filter_price(candidate_df, DEFAULT_MIN_PRICE)
        tickers = result["ticker"].to_list()
        assert "CHEAP" not in tickers  # 3.5 < 5
        assert "AAPL" in tickers

    def test_filter_by_days_listed(self, candidate_df):
        """days_listed >= 252 trading days (~12 months)."""
        from alphascreener.universe.filters import DEFAULT_MIN_DAYS_LISTED, _filter_days_listed

        result = _filter_days_listed(candidate_df, DEFAULT_MIN_DAYS_LISTED)
        tickers = result["ticker"].to_list()
        assert "NEWIPO" not in tickers  # 120 < 252
        assert "AAPL" in tickers

    def test_filter_by_status_active(self, candidate_df):
        """status == 'Active' (not halted, not delisted)."""
        from alphascreener.universe.filters import _filter_status

        result = _filter_status(candidate_df)
        tickers = result["ticker"].to_list()
        assert "HALTED" not in tickers
        assert "DELISTED" not in tickers
        assert "AAPL" in tickers

    def test_pre_filter_combined(self, candidate_df):
        """Combined pre-filter returns only tickers passing ALL conditions."""
        from alphascreener.universe.filters import pre_filter

        result = pre_filter(candidate_df)
        tickers = result["ticker"].to_list()
        # Expected pass: AAPL, MEGA, SMALL
        assert set(tickers) == {"AAPL", "MEGA", "SMALL"}
        # Expected fail:
        assert "LOWVOL" not in tickers  # volume 18M < 20M
        assert "CHEAP" not in tickers  # price 3.5 < 5
        assert "NEWIPO" not in tickers  # days 120 < 252
        assert "HALTED" not in tickers  # status Halted
        assert "DELISTED" not in tickers  # status Delisted

    def test_pre_filter_empty_df(self):
        """Pre-filter on empty DataFrame returns empty DataFrame."""
        from alphascreener.universe.filters import pre_filter

        empty = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "avg_dollar_volume_20d": pl.Float64,
                "market_cap": pl.Float64,
                "last_price": pl.Float64,
                "days_listed": pl.Int64,
                "status": pl.Utf8,
            }
        )
        result = pre_filter(empty)
        assert result.height == 0

    def test_pre_filter_preserves_all_columns(self, candidate_df):
        """Pre-filter should not drop any input columns."""
        from alphascreener.universe.filters import pre_filter

        result = pre_filter(candidate_df)
        assert set(result.columns) == set(candidate_df.columns)

    def test_pre_filter_with_custom_thresholds(self, candidate_df):
        """Pre-filter accepts optional threshold overrides."""
        from alphascreener.universe.filters import pre_filter

        # Increase volume threshold to 25M — should filter out more
        result = pre_filter(candidate_df, min_dollar_volume=25_000_000)
        tickers = result["ticker"].to_list()
        # CHEAP had 15M volume, LOWVOL 18M, so both fail. SMALL at 25M is not > 25M, so also fails.
        assert "LOWVOL" not in tickers
        assert "CHEAP" not in tickers
        assert "SMALL" not in tickers  # 25M is not strictly > 25M


# ============================================================================
# Whitelist cache tests
# ============================================================================


class TestWhitelistCache:
    """Save/load whitelist to/from Parquet cache."""

    def test_save_and_load_whitelist(self, tmp_path, monkeypatch):
        """Round-trip: save tickers to cache, then load them back."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.whitelist import load_whitelist_cache, save_whitelist_cache

        tickers = {"AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"}
        month_key = "2025-03"
        save_whitelist_cache(tickers, month_key)

        loaded = load_whitelist_cache()
        assert loaded == tickers

    def test_load_whitelist_cache_empty(self, tmp_path, monkeypatch):
        """Loading cache from a directory with no cache file returns empty set."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.whitelist import load_whitelist_cache

        result = load_whitelist_cache()
        assert result == set()

    def test_save_whitelist_creates_parquet(self, tmp_path, monkeypatch):
        """Saving whitelist creates a .parquet file on disk."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.whitelist import save_whitelist_cache

        tickers = {"AAPL", "GOOGL"}
        save_whitelist_cache(tickers, "2025-04")

        cache_file = tmp_path / "universe" / "whitelist.parquet"
        assert cache_file.exists()

        df = pl.read_parquet(cache_file)
        assert "ticker" in df.columns
        assert "refresh_month" in df.columns

    def test_whitelist_is_deduplicated(self, tmp_path, monkeypatch):
        """Whitelist from SP500 + Russell 1000 is deduplicated (union set)."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.whitelist import build_whitelist

        # Mock the fetchers to return overlapping sets
        def mock_fetch_sp500():
            return {"AAPL", "MSFT", "GOOGL"}

        def mock_fetch_russell():
            return {"MSFT", "GOOGL", "NVDA"}  # MSFT, GOOGL overlap

        result = build_whitelist(
            fetcher_sp500=mock_fetch_sp500,
            fetcher_russell=mock_fetch_russell,
        )
        assert result == {"AAPL", "MSFT", "GOOGL", "NVDA"}


# ============================================================================
# Metadata cache tests (universe_meta.parquet)
# ============================================================================


class TestMetaCache:
    """Read/write universe_meta.parquet for sector/industry/market_cap."""

    @pytest.fixture
    def sample_meta_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Communication Services", "Technology"],
                "industry": ["Consumer Electronics", "Internet Content & Information", "Software"],
                "market_cap": [3.5e12, 2.1e12, 3.2e12],
                "index_source": [
                    "SP500,Russell1000",
                    "SP500,Russell1000",
                    "SP500,Russell1000",
                ],
                "refreshed_at": [
                    date(2025, 3, 1),
                    date(2025, 3, 1),
                    date(2025, 3, 1),
                ],
            }
        )

    def test_write_and_read_meta(self, tmp_path, monkeypatch, sample_meta_df):
        """Round-trip: write meta to parquet, read it back."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.meta import read_meta_cache, write_meta_cache

        write_meta_cache(sample_meta_df)
        lf = read_meta_cache()
        result = lf.collect()

        assert result.height == 3
        tickers = result["ticker"].to_list()
        assert set(tickers) == {"AAPL", "GOOGL", "MSFT"}

    def test_write_meta_creates_parquet(self, tmp_path, monkeypatch, sample_meta_df):
        """Writing meta creates universe_meta.parquet on disk."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.meta import write_meta_cache

        write_meta_cache(sample_meta_df)

        meta_file = tmp_path / "universe" / "universe_meta.parquet"
        assert meta_file.exists()

    def test_read_meta_when_cache_missing(self, tmp_path, monkeypatch):
        """Reading meta when no cache exists raises FileNotFoundError."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.meta import read_meta_cache

        with pytest.raises(FileNotFoundError):
            read_meta_cache()

    def test_meta_schema_validation(self, tmp_path, monkeypatch, sample_meta_df):
        """Meta cache must have required columns."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.meta import write_meta_cache

        # Missing 'sector' column
        bad_df = sample_meta_df.drop("sector")
        with pytest.raises(ValueError, match="sector"):
            write_meta_cache(bad_df)

    def test_meta_returns_lazyframe(self, tmp_path, monkeypatch, sample_meta_df):
        """read_meta_cache returns a LazyFrame for predicate pushdown."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.meta import read_meta_cache, write_meta_cache

        write_meta_cache(sample_meta_df)
        lf = read_meta_cache()
        assert isinstance(lf, pl.LazyFrame)

        # Filter using LazyFrame predicate pushdown
        tech = lf.filter(pl.col("sector") == "Technology").collect()
        assert tech.height == 2
        assert set(tech["ticker"].to_list()) == {"AAPL", "MSFT"}


# ============================================================================
# Monthly refresh task tests
# ============================================================================


class TestMonthlyUniverseRefresh:
    """monthly_universe_refresh orchestrates whitelist + filters + meta."""

    def test_refresh_is_target_month(self):
        """is_first_of_month returns True only on the 1st."""
        from alphascreener.universe.tasks import _is_target_refresh_day

        assert _is_target_refresh_day(date(2025, 3, 1)) is True
        assert _is_target_refresh_day(date(2025, 3, 2)) is False
        assert _is_target_refresh_day(date(2025, 3, 15)) is False
        assert _is_target_refresh_day(date(2025, 1, 1)) is True

    def test_refresh_month_key(self):
        """_refresh_month_key returns YYYY-MM for previous month."""
        from alphascreener.universe.tasks import _refresh_month_key

        # On 2025-03-01, the key should be "2025-02" (data for prior month)
        assert _refresh_month_key(date(2025, 3, 1)) == "2025-02"
        assert _refresh_month_key(date(2025, 1, 1)) == "2024-12"

    def test_monthly_refresh_pipeline(self, tmp_path, monkeypatch):
        """Full pipeline: fetch -> filter -> cache -> meta."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        from alphascreener.universe.tasks import monthly_universe_refresh

        # Mock the whitelist fetching functions
        def mock_fetch_sp500():
            return {"AAPL", "MSFT", "GOOGL"}

        def mock_fetch_russell():
            return {"MSFT", "GOOGL", "AMZN"}

        # Mock meta fetching function
        def mock_fetch_meta(tickers):
            return pl.DataFrame(
                {
                    "ticker": sorted(tickers),
                    "sector": ["Tech", "Tech", "Tech", "Tech"],
                    "industry": ["Hw", "Sw", "Net", "Cloud"],
                    "market_cap": [3e12, 2e12, 1e12, 1.5e12],
                    "index_source": [
                        "SP500",
                        "SP500,Russell1000",
                        "SP500,Russell1000",
                        "Russell1000",
                    ],
                    "refreshed_at": [date(2025, 3, 1)] * 4,
                }
            )

        result = monthly_universe_refresh(
            ref_date=date(2025, 3, 1),
            fetcher_sp500=mock_fetch_sp500,
            fetcher_russell=mock_fetch_russell,
            fetcher_meta=mock_fetch_meta,
        )

        # Verify result
        assert result["whitelist"] == {"AAPL", "MSFT", "GOOGL", "AMZN"}
        assert result["count"] == 4
        assert result["month_key"] == "2025-02"
        assert result["meta_refreshed"] is True

    def test_monthly_refresh_uses_cache_on_non_first_day(self, tmp_path, monkeypatch):
        """On non-first day of month, loads from cache instead of re-fetching."""
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )
        monkeypatch.setattr(
            "alphascreener.universe._paths._universe_dir",
            lambda: tmp_path / "universe",
        )

        # Pre-populate cache
        from alphascreener.universe.whitelist import save_whitelist_cache

        save_whitelist_cache({"AAPL", "MSFT"}, "2025-02")

        from alphascreener.universe.tasks import monthly_universe_refresh

        # Call on non-refresh day (not 1st of month) with fetchers that would
        # return different data
        def mock_fetch_sp500():
            return {"SHOULD_NOT_BE_CALLED"}  # pragma: no cover

        def mock_fetch_russell():
            return {"SHOULD_NOT_BE_CALLED"}  # pragma: no cover

        def mock_fetch_meta(tickers):
            return pl.DataFrame(  # pragma: no cover
                {
                    "ticker": list(tickers),
                    "sector": ["T"] * len(tickers),
                    "industry": ["I"] * len(tickers),
                    "market_cap": [0.0] * len(tickers),
                    "index_source": [""] * len(tickers),
                    "refreshed_at": [date(2025, 3, 1)] * len(tickers),
                }
            )

        result = monthly_universe_refresh(
            ref_date=date(2025, 3, 15),
            fetcher_sp500=mock_fetch_sp500,
            fetcher_russell=mock_fetch_russell,
            fetcher_meta=mock_fetch_meta,
        )

        # Should load from cache, not call fetchers
        assert result["whitelist"] == {"AAPL", "MSFT"}


# ============================================================================
# Whitelist SP500 Wikipedia parsing test
# ============================================================================


class TestSP500Fetch:
    """SP500 constituent fetching from Wikipedia."""

    def test_parse_wikipedia_table(self):
        """Parse mock Wikipedia HTML table to extract tickers."""
        from alphascreener.universe.whitelist import _parse_sp500_from_html

        html = """
        <html><body>
        <table id="constituents" class="wikitable">
        <tr><th>Symbol</th><th>Company</th></tr>
        <tr><td>AAPL</td><td>Apple Inc.</td></tr>
        <tr><td>MSFT</td><td>Microsoft Corp.</td></tr>
        <tr><td>GOOGL</td><td>Alphabet Inc.</td></tr>
        <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
        </table>
        </body></html>
        """
        tickers = _parse_sp500_from_html(html)
        assert tickers == {"AAPL", "MSFT", "GOOGL", "BRK.B"}

    def test_parse_wikipedia_table_with_newlines(self):
        """Handle <br> or whitespace in ticker cells."""
        from alphascreener.universe.whitelist import _parse_sp500_from_html

        html = """
        <html><body>
        <table id="constituents" class="wikitable">
        <tr><th>Symbol</th><th>Company</th></tr>
        <tr><td>\nAAPL\n</td><td>Apple</td></tr>
        <tr><td>  MSFT  </td><td>Microsoft</td></tr>
        </table>
        </body></html>
        """
        tickers = _parse_sp500_from_html(html)
        assert tickers == {"AAPL", "MSFT"}

    def test_parse_wikipedia_empty(self):
        """Empty HTML returns empty set."""
        from alphascreener.universe.whitelist import _parse_sp500_from_html

        assert _parse_sp500_from_html("<html></html>") == set()
