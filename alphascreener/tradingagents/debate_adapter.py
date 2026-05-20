"""Debate agent adapter — wraps TradingAgents researcher/debater/manager factories.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Encapsulates Bull/Bear researchers, Research Manager, Portfolio Manager,
Trader, and the three risk debaters (aggressive / conservative / neutral).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Re-export TradingAgents public API
# ---------------------------------------------------------------------------
from tradingagents.agents.managers.portfolio_manager import (  # noqa: F401, E402
    create_portfolio_manager,
)
from tradingagents.agents.managers.research_manager import (  # noqa: F401, E402
    create_research_manager,
)
from tradingagents.agents.researchers.bear_researcher import (  # noqa: F401, E402
    create_bear_researcher,
)
from tradingagents.agents.researchers.bull_researcher import (  # noqa: F401, E402
    create_bull_researcher,
)
from tradingagents.agents.risk_mgmt.aggressive_debator import (  # noqa: F401, E402
    create_aggressive_debator,
)
from tradingagents.agents.risk_mgmt.conservative_debator import (  # noqa: F401, E402
    create_conservative_debator,
)
from tradingagents.agents.risk_mgmt.neutral_debator import (  # noqa: F401, E402
    create_neutral_debator,
)
from tradingagents.agents.trader.trader import (  # noqa: F401, E402
    create_trader,
)

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

DebateNode = Callable[[dict[str, Any]], dict[str, Any]]

# ---------------------------------------------------------------------------
# Convenience: create all debaters at once
# ---------------------------------------------------------------------------

_logger = get_logger("screening")


def create_debate_team(
    llm: Any,
) -> dict[str, DebateNode]:
    """Create the full debate team (bull, bear, research manager).

    Args:
        llm: An LLM instance (e.g. from ``create_llm_client(...).get_llm()``).

    Returns:
        Dict with keys ``"bull"``, ``"bear"``, ``"research_manager"``.
    """
    _logger.debug("Creating debate team (bull, bear, research manager)")
    return {
        "bull": create_bull_researcher(llm),
        "bear": create_bear_researcher(llm),
        "research_manager": create_research_manager(llm),
    }


def create_risk_team(
    llm: Any,
) -> dict[str, DebateNode]:
    """Create the full risk debate team (aggressive, conservative, neutral).

    Args:
        llm: An LLM instance (e.g. from ``create_llm_client(...).get_llm()``).

    Returns:
        Dict with keys ``"aggressive"``, ``"conservative"``, ``"neutral"``.
    """
    _logger.debug("Creating risk debate team (aggressive, conservative, neutral)")
    return {
        "aggressive": create_aggressive_debator(llm),
        "conservative": create_conservative_debator(llm),
        "neutral": create_neutral_debator(llm),
    }
