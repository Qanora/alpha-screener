"""Monthly universe refresh task (Issue #88).

Reference: PRD 7.7.1 - monthly_universe_refresh 任务逻辑:
  - 每月1日更新 SP500 + Russell 1000 白名单
  - 缓存 universe_meta.parquet（sector/industry/市值等）
"""

from __future__ import annotations

from datetime import date

from alphascreener.universe.meta import write_meta_cache
from alphascreener.universe.whitelist import (
    build_whitelist,
    load_whitelist_cache,
    save_whitelist_cache,
)

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _is_target_refresh_day(ref_date: date) -> bool:
    """Return True if ref_date is the 1st of any month (the target refresh day).

    Args:
        ref_date: Reference date to check.

    Returns:
        True if ref_date.day == 1.
    """
    return ref_date.day == 1


def _refresh_month_key(ref_date: date) -> str:
    """Return the YYYY-MM key for the *previous* month.

    Universe refresh on the 1st of month M uses data from month M-1.

    Args:
        ref_date: Reference date (typically the 1st of a month).

    Returns:
        A string like "2025-02" for ref_date 2025-03-01.
    """
    # Go back one month from the 1st
    if ref_date.month == 1:
        return f"{ref_date.year - 1}-12"
    return f"{ref_date.year}-{ref_date.month - 1:02d}"


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------


def monthly_universe_refresh(
    *,
    ref_date: date | None = None,
    fetcher_sp500: callable | None = None,
    fetcher_russell: callable | None = None,
    fetcher_meta: callable | None = None,
) -> dict:
    """Refresh the universe whitelist and metadata cache.

    On the 1st of the month, fetches SP500 + Russell 1000 constituents,
    caches the combined whitelist, and refreshes universe_meta.parquet.
    On all other days, loads from the existing cache.

    All fetcher arguments are callables so tests can inject mocks:

    - ``fetcher_sp500() -> set[str]``
    - ``fetcher_russell() -> set[str]``
    - ``fetcher_meta(tickers: set[str]) -> pl.DataFrame`` (with meta schema)

    Args:
        ref_date: Override the current date (for testing).
        fetcher_sp500: SP500 ticker fetcher.
        fetcher_russell: Russell 1000 ticker fetcher.
        fetcher_meta: Metadata fetcher (sector/industry/market_cap).

    Returns:
        Dict with keys: whitelist, count, month_key, meta_refreshed.
    """
    if ref_date is None:
        ref_date = date.today()

    if _is_target_refresh_day(ref_date):
        month_key = _refresh_month_key(ref_date)

        # Build whitelist
        whitelist = build_whitelist(
            fetcher_sp500=fetcher_sp500,
            fetcher_russell=fetcher_russell,
        )

        # Cache whitelist
        save_whitelist_cache(whitelist, month_key)

        # Refresh metadata cache
        meta_refreshed = False
        if fetcher_meta is not None and whitelist:
            meta_df = fetcher_meta(whitelist)
            if meta_df is not None and meta_df.height > 0:
                write_meta_cache(meta_df)
                meta_refreshed = True

        return {
            "whitelist": whitelist,
            "count": len(whitelist),
            "month_key": month_key,
            "meta_refreshed": meta_refreshed,
        }

    # Non-refresh day: load from cache
    whitelist = load_whitelist_cache()
    return {
        "whitelist": whitelist,
        "count": len(whitelist),
        "month_key": "",
        "meta_refreshed": False,
    }
