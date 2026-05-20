"""Analyst agent adapter — wraps TradingAgents analyst factory functions.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Encapsulates ``create_market_analyst``, ``create_fundamentals_analyst``,
``create_news_analyst``, and ``create_sentiment_analyst``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Re-export TradingAgents public API
# ---------------------------------------------------------------------------
from tradingagents.agents.analysts.fundamentals_analyst import (  # noqa: F401, E402
    create_fundamentals_analyst,
)
from tradingagents.agents.analysts.market_analyst import (  # noqa: F401, E402
    create_market_analyst,
)
from tradingagents.agents.analysts.news_analyst import (  # noqa: F401, E402
    create_news_analyst,
)
from tradingagents.agents.analysts.sentiment_analyst import (  # noqa: F401, E402, I100
    create_sentiment_analyst,
)

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

# Each analyst factory returns a callable that acts as a LangGraph node
# (taking a state dict and returning a dict).
AnalystNode = Callable[[dict[str, Any]], dict[str, Any]]

# ---------------------------------------------------------------------------
# Canonical list of analyst types
# ---------------------------------------------------------------------------

# Mirror of the analyst keys recognised by TradingAgentsGraph (selected_analysts).
ANALYST_TYPES: tuple[str, ...] = ("market", "social", "news", "fundamentals")

# Map of analyst-type string to the corresponding factory function.
ANALYST_FACTORIES: dict[str, Callable[..., AnalystNode]] = {
    "market": create_market_analyst,
    "social": create_sentiment_analyst,  # "social" maps to the renamed sentiment analyst
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
}

_logger = get_logger("screening")


def create_analyst(
    analyst_type: str,
    llm: Any,
) -> AnalystNode:
    """Create a single analyst node for the given type.

    Args:
        analyst_type: One of ``"market"``, ``"social"``, ``"news"``,
            ``"fundamentals"``.
        llm: An LLM instance (e.g. from ``create_llm_client(...).get_llm()``).

    Returns:
        A LangGraph-compatible analyst node function.

    Raises:
        ValueError: If *analyst_type* is not recognized.
    """
    factory = ANALYST_FACTORIES.get(analyst_type)
    if factory is None:
        _logger.warning("Unknown analyst type %r", analyst_type)
        raise ValueError(
            f"Unknown analyst type: {analyst_type!r}. "
            f"Valid types: {', '.join(sorted(ANALYST_FACTORIES))}"
        )
    _logger.debug("Creating %s analyst node", analyst_type)
    return factory(llm)
