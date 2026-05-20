"""SP500 + Russell 1000 whitelist loading and caching (Issue #88).

Reference: PRD 1.4 / 3.3 - 指数白名单加载.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from alphascreener.universe import _paths

WHITELIST_FILENAME: str = "whitelist.parquet"


def _whitelist_path() -> Path:
    """Return the full path to whitelist.parquet."""
    return _paths._universe_dir() / WHITELIST_FILENAME


# ---------------------------------------------------------------------------
# Wikipedia HTML parsing (SP500)
# ---------------------------------------------------------------------------


def _parse_sp500_from_html(html: str) -> set[str]:
    """Parse SP500 constituent tickers from Wikipedia HTML.

    The SP500 page has a ``<table id="constituents">`` with a "Symbol" column.
    Handles BRK.B, BF.B, etc. (dots in ticker are preserved).

    Args:
        html: Raw HTML content of the Wikipedia page.

    Returns:
        Set of uppercase ticker symbols.
    """
    tickers: set[str] = set()

    # Find the constituents table
    table_match = re.search(
        r'<table[^>]*id="constituents"[^>]*>(.*?)</table>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not table_match:
        return tickers

    table_html = table_match.group(1)

    # Find all rows with <td> elements
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)

    for row_match in row_pattern.finditer(table_html):
        row_html = row_match.group(1)
        cells = cell_pattern.findall(row_html)
        if not cells:
            continue
        # First cell is the ticker symbol
        ticker = re.sub(r"<[^>]+>", "", cells[0]).strip()
        # Remove Wikipedia footnote links like [a] or superscript
        ticker = re.sub(r"\[.*?\]", "", ticker).strip()
        if ticker and re.match(r"^[A-Za-z.]+$", ticker):
            tickers.add(ticker.upper())

    return tickers


# ---------------------------------------------------------------------------
# Whitelist building
# ---------------------------------------------------------------------------


def build_whitelist(
    *,
    fetcher_sp500: callable | None = None,
    fetcher_russell: callable | None = None,
) -> set[str]:
    """Build the combined SP500 + Russell 1000 whitelist.

    Tickers are deduplicated (union set). Fetcher functions are passed as
    dependencies so tests can inject mocks.

    Args:
        fetcher_sp500: Callable returning ``set[str]`` of SP500 tickers.
        fetcher_russell: Callable returning ``set[str]`` of Russell 1000 tickers.

    Returns:
        Union set of all tickers from both indices.
    """
    if fetcher_sp500 is None and fetcher_russell is None:
        raise ValueError(
            "At least one fetcher must be provided "
            "(fetcher_sp500 or fetcher_russell)"
        )

    sp500: set[str] = set()
    russell: set[str] = set()

    if fetcher_sp500 is not None:
        sp500 = fetcher_sp500()
    if fetcher_russell is not None:
        russell = fetcher_russell()

    return sp500 | russell


# ---------------------------------------------------------------------------
# Cache read/write
# ---------------------------------------------------------------------------


def save_whitelist_cache(tickers: set[str], month_key: str) -> None:
    """Persist the whitelist to Parquet for fast reload.

    Args:
        tickers: Set of ticker strings.
        month_key: YYYY-MM string identifying the refresh month.
    """
    _paths._universe_dir().mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "ticker": sorted(tickers),
            "refresh_month": [month_key] * len(tickers),
        }
    )
    df.write_parquet(_whitelist_path())


def load_whitelist_cache() -> set[str]:
    """Load cached whitelist from Parquet.

    Returns:
        Set of ticker strings, or an empty set if no cache exists.
    """
    path = _whitelist_path()
    if not path.exists():
        return set()
    df = pl.read_parquet(path)
    if df.height == 0:
        return set()
    return set(df["ticker"].to_list())
