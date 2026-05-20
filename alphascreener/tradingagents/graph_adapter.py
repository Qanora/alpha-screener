"""Graph adapter — wraps the TradingAgents graph orchestration layer.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Encapsulates :class:`TradingAgentsGraph` and its supporting components
so the rest of the codebase never imports from ``tradingagents.graph``
directly.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Re-export TradingAgents graph components
# ---------------------------------------------------------------------------
from tradingagents.agents.utils.agent_states import (  # noqa: F401, E402
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.graph.conditional_logic import (  # noqa: F401, E402, I100
    ConditionalLogic,
)
from tradingagents.graph.propagation import Propagator  # noqa: F401, E402
from tradingagents.graph.reflection import Reflector  # noqa: F401, E402
from tradingagents.graph.setup import GraphSetup  # noqa: F401, E402
from tradingagents.graph.signal_processing import (  # noqa: F401, E402, I100
    SignalProcessor,
)
from tradingagents.graph.trading_graph import (  # noqa: F401, E402, I100
    TradingAgentsGraph,
)

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Adapter-level conveniences
# ---------------------------------------------------------------------------

_logger = get_logger("screening")


def create_graph(
    selected_analysts: list[str] | None = None,
    debug: bool = False,
    config: dict[str, Any] | None = None,
    callbacks: list | None = None,
) -> TradingAgentsGraph:
    """Create a configured :class:`TradingAgentsGraph` instance.

    This is the main entry point for running the multi-agent trading
    pipeline on a single ticker.

    Args:
        selected_analysts: Analyst types to include. Defaults to
            ``["market", "social", "news", "fundamentals"]``.
        debug: Enable verbose per-node output.
        config: Configuration dict (uses :data:`tradingagents.default_config.DEFAULT_CONFIG`
            when *None*).
        callbacks: Optional list of LangChain callback handlers.

    Returns:
        A ready-to-use :class:`TradingAgentsGraph`.
    """
    if selected_analysts is None:
        selected_analysts = ["market", "social", "news", "fundamentals"]

    _logger.info(
        "Creating TradingAgentsGraph with analysts=%s debug=%s",
        selected_analysts,
        debug,
    )
    return TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=debug,
        config=config,
        callbacks=callbacks,
    )
