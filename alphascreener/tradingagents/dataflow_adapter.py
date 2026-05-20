"""Data-flow adapter — wraps TradingAgents data fetching layer.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Encapsulates the vendor-routed data-fetch tools (``get_stock_data``,
``get_indicators``, ``get_fundamentals``, etc.) and the vendor-routing
machinery (``interface`` module).
"""

from __future__ import annotations

import logging
from typing import Any

# ---------------------------------------------------------------------------
# Re-export vendor-routed tool functions (abstract methods used by agents)
# ---------------------------------------------------------------------------
from tradingagents.agents.utils.agent_utils import (  # noqa: F401, E402
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
)

# ---------------------------------------------------------------------------
# Re-export config accessors
# ---------------------------------------------------------------------------
from tradingagents.dataflows.config import get_config  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Re-export interface-level routing machinery
# ---------------------------------------------------------------------------
from tradingagents.dataflows.interface import (  # noqa: F401, E402
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
    get_category_for_method,
    get_vendor,
    route_to_vendor,
)

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Adapter-level conveniences
# ---------------------------------------------------------------------------

_logger: logging.Logger = get_logger("screening")


class DataFlowRouter:
    """Thin adapter over TradingAgents' :func:`route_to_vendor`.

    Provides a stable, documented API for dispatching data-fetch calls
    through the configured vendor chain (yfinance → alpha_vantage) with
    automatic fallback on rate-limit errors.
    """

    @staticmethod
    def fetch(method: str, *args: Any, **kwargs: Any) -> Any:
        """Route *method* through the configured vendor chain.

        Args:
            method: One of the keys in :data:`VENDOR_METHODS`
                (e.g. ``"get_stock_data"``).
            *args: Positional arguments forwarded to the vendor implementation.
            **kwargs: Keyword arguments forwarded to the vendor implementation.

        Returns:
            The vendor's return value (typically a DataFrame or dict).

        Raises:
            RuntimeError: If all vendors fail for *method*.
            ValueError: If *method* is not recognised.
        """
        _logger.debug("DataFlow dispatch: %s", method)
        return route_to_vendor(method, *args, **kwargs)

    @staticmethod
    def list_methods() -> list[str]:
        """Return the list of available data-flow method names."""
        return sorted(VENDOR_METHODS.keys())

    @staticmethod
    def list_categories() -> dict[str, list[str]]:
        """Return a mapping of category → tool names."""
        return {cat: info["tools"] for cat, info in TOOLS_CATEGORIES.items()}
