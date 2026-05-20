"""TradingAgents framework adapters — isolate upstream API changes.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Five adapter modules wrap the TradingAgents public API so the rest of the
codebase never imports from ``tradingagents.*`` directly.
"""

from alphascreener.tradingagents.analyst_adapter import (
    create_analyst,
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from alphascreener.tradingagents.dataflow_adapter import (
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
    DataFlowRouter,
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
from alphascreener.tradingagents.debate_adapter import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_debate_team,
    create_neutral_debator,
    create_portfolio_manager,
    create_research_manager,
    create_risk_team,
    create_trader,
)
from alphascreener.tradingagents.graph_adapter import (
    ConditionalLogic,
    GraphSetup,
    Propagator,
    Reflector,
    SignalProcessor,
    TradingAgentsGraph,
    create_graph,
)
from alphascreener.tradingagents.llm_adapter import (
    BaseLLMClient,
    create_llm_client,
    create_llm_client_safe,
)

__all__ = [
    # LLM
    "BaseLLMClient",
    "create_llm_client",
    "create_llm_client_safe",
    # Analysts
    "create_analyst",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_news_analyst",
    "create_sentiment_analyst",
    # Debate
    "create_aggressive_debator",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_conservative_debator",
    "create_debate_team",
    "create_neutral_debator",
    "create_portfolio_manager",
    "create_research_manager",
    "create_risk_team",
    "create_trader",
    # DataFlow
    "DataFlowRouter",
    "TOOLS_CATEGORIES",
    "VENDOR_METHODS",
    "get_balance_sheet",
    "get_cashflow",
    "get_fundamentals",
    "get_global_news",
    "get_income_statement",
    "get_indicators",
    "get_insider_transactions",
    "get_news",
    "get_stock_data",
    # Graph
    "ConditionalLogic",
    "GraphSetup",
    "Propagator",
    "Reflector",
    "SignalProcessor",
    "TradingAgentsGraph",
    "create_graph",
]
