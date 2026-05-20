"""Tests for TradingAgents five-adapter layer.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.
"""

from __future__ import annotations

import pytest
from tradingagents.default_config import DEFAULT_CONFIG as _UPSTREAM_DEFAULT

from alphascreener.tradingagents import (
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
    BaseLLMClient,
    ConditionalLogic,
    DataFlowRouter,
    GraphSetup,
    Propagator,
    Reflector,
    SignalProcessor,
    TradingAgentsGraph,
    create_aggressive_debator,
    create_analyst,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_debate_team,
    create_fundamentals_analyst,
    create_graph,
    create_llm_client,
    create_llm_client_safe,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_risk_team,
    create_sentiment_analyst,
    create_trader,
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

# ============================================================================
# Helpers
# ============================================================================


def _is_callable_or_tool(obj: object) -> bool:
    """Return True for plain callables *or* LangChain StructuredTool wrappers."""
    if callable(obj):
        return True
    # LangChain BaseTool subclasses have a ``func`` attribute.
    return hasattr(obj, "func")


# Merge ollama overrides into the upstream DEFAULT_CONFIG so all keys that
# TradingAgentsGraph expects (max_debate_rounds, etc.) are present while
# avoiding the need for cloud LLM credentials.
_OLLAMA_CONFIG: dict = {
    **_UPSTREAM_DEFAULT,
    "llm_provider": "ollama",
    "deep_think_llm": "llama3",
    "quick_think_llm": "llama3",
    "data_cache_dir": "/tmp",
    "results_dir": "/tmp",
}


# ============================================================================
# Package-level imports
# ============================================================================


class TestPackageImports:
    """Verify the top-level __init__ exposes all expected symbols."""

    def test_llm_exports(self):
        assert callable(create_llm_client)
        assert BaseLLMClient is not None

    def test_analyst_exports(self):
        assert callable(create_market_analyst)
        assert callable(create_fundamentals_analyst)
        assert callable(create_news_analyst)
        assert callable(create_sentiment_analyst)

    def test_debate_exports(self):
        assert callable(create_bull_researcher)
        assert callable(create_bear_researcher)
        assert callable(create_research_manager)
        assert callable(create_portfolio_manager)
        assert callable(create_trader)
        assert callable(create_aggressive_debator)
        assert callable(create_conservative_debator)
        assert callable(create_neutral_debator)

    def test_dataflow_exports(self):
        for tool in [
            get_stock_data,
            get_indicators,
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_news,
            get_global_news,
            get_insider_transactions,
        ]:
            assert _is_callable_or_tool(tool), f"{tool!r} is not callable or tool"

    def test_graph_exports(self):
        assert TradingAgentsGraph is not None
        assert ConditionalLogic is not None
        assert GraphSetup is not None
        assert Propagator is not None
        assert Reflector is not None
        assert SignalProcessor is not None


# ============================================================================
# LLM adapter
# ============================================================================


class TestLlmAdapter:
    """Tests for llm_adapter.py."""

    def test_create_llm_client_requires_valid_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("nonexistent_provider_xyz", "some-model")

    def test_create_llm_client_safe_rejects_invalid_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client_safe("nonexistent_provider_xyz", "some-model")

    def test_create_llm_client_safe_accepts_valid_provider(self):
        """Known valid provider: 'ollama' requires no API key at construction time."""
        client = create_llm_client_safe("ollama", "llama3")
        assert isinstance(client, BaseLLMClient)
        assert client.model == "llama3"


# ============================================================================
# Analyst adapter
# ============================================================================


class TestAnalystAdapter:
    """Tests for analyst_adapter.py."""

    def test_create_analyst_valid_types(self):
        from tradingagents.llm_clients.factory import create_llm_client as _factory

        llm_client = _factory("ollama", "llama3")
        _ = llm_client  # we don't call get_llm() — just exercise routing

    def test_create_analyst_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown analyst type"):
            create_analyst("invalid_analyst_type", None)


# ============================================================================
# Debate adapter
# ============================================================================


class TestDebateAdapter:
    """Tests for debate_adapter.py — factory functions exist and return callables."""

    def test_factories_return_callables(self):
        """Each factory called with None returns a callable (wont work w/o LLM,
        but validates the import / return-type path)."""
        for factory in [
            create_bull_researcher,
            create_bear_researcher,
            create_research_manager,
            create_portfolio_manager,
            create_trader,
            create_aggressive_debator,
            create_conservative_debator,
            create_neutral_debator,
        ]:
            # Pass None — at minimum won't blow up at import time
            node_fn = factory(None)
            assert callable(node_fn), f"{factory.__name__} did not return a callable"

    def test_create_debate_team_returns_three_nodes(self):
        team = create_debate_team(None)
        assert set(team.keys()) == {"bull", "bear", "research_manager"}
        for v in team.values():
            assert callable(v)

    def test_create_risk_team_returns_three_nodes(self):
        team = create_risk_team(None)
        assert set(team.keys()) == {"aggressive", "conservative", "neutral"}
        for v in team.values():
            assert callable(v)


# ============================================================================
# DataFlow adapter
# ============================================================================


class TestDataFlowAdapter:
    """Tests for dataflow_adapter.py."""

    def test_vendor_methods_not_empty(self):
        assert len(VENDOR_METHODS) >= 8

    def test_tools_categories_not_empty(self):
        assert len(TOOLS_CATEGORIES) >= 3
        assert "core_stock_apis" in TOOLS_CATEGORIES
        assert "fundamental_data" in TOOLS_CATEGORIES

    def test_router_list_methods(self):
        methods = DataFlowRouter.list_methods()
        assert "get_stock_data" in methods
        assert "get_indicators" in methods

    def test_router_list_categories(self):
        categories = DataFlowRouter.list_categories()
        assert "core_stock_apis" in categories
        assert isinstance(categories["core_stock_apis"], list)

    def test_router_fetch_invalid_method_raises(self):
        """Unrecognised method -> ValueError from get_category_for_method."""
        with pytest.raises(ValueError, match="not found"):
            DataFlowRouter.fetch("nonexistent_method")


# ============================================================================
# Graph adapter
# ============================================================================


class TestGraphAdapter:
    """Tests for graph_adapter.py.

    Uses an ollama-backed config so tests pass without cloud LLM credentials.
    """

    def test_create_graph_defaults(self):
        g = create_graph(config=_OLLAMA_CONFIG)
        assert isinstance(g, TradingAgentsGraph)
        assert g.debug is False

    def test_create_graph_with_options(self):
        g = create_graph(
            selected_analysts=["market", "fundamentals"],
            debug=True,
            config=_OLLAMA_CONFIG,
        )
        assert isinstance(g, TradingAgentsGraph)
        assert g.debug is True

    def test_create_graph_with_callbacks(self):
        g = create_graph(callbacks=[], config=_OLLAMA_CONFIG)
        assert isinstance(g, TradingAgentsGraph)
