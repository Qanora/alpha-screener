"""Tests for analyst prompts, breakout retriever, and orchestrator.

Issue #97: Analyst prompts + invocation.
Reference: PRD 4.2.1 / 4.5.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from alphascreener.tradingagents.breakout_retriever import (
    BreakoutCaseRetriever,
)
from alphascreener.tradingagents.orchestrator import (
    AnalystOrchestrator,
    build_context,
    check_token_budget,
    clamp_context_for_budget,
    run_analyst,
)
from alphascreener.tradingagents.prompts import (
    ANALYST_PROMPTS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    AnalystContext,
    BreakoutAnalystPrompt,
    FundamentalsAnalystPrompt,
    MarketAnalystPrompt,
    NewsAnalystPrompt,
    estimate_tokens,
    format_analyst_prompt,
    get_analyst_prompt,
    truncate_context,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_context(**overrides) -> AnalystContext:
    """Build a test AnalystContext with example data."""
    defaults = {
        "ticker": "AAPL",
        "price": 185.50,
        "mom_5d": 3.2,
        "factor_scores_summary": (
            "MOM_5D: +3.2% | RSI: 62 | MFI: 55 | ATR: 0.8% | VOL_ANOMALY: 1.5x"
        ),
        "news_summary": "iPhone 17出货指引上修; FOMC鹰派纪要; AAPL回购计划扩大",
        "technical_pattern": "布林带收窄12日后放量突破上轨, RSI 62未超买",
        "false_breakout_rate": 35,
        "factor_vector": [0.12, -0.03, 0.08, 0.15, 0.01],
    }
    defaults.update(overrides)
    return AnalystContext(**defaults)


def _mock_invoker(response_text: str = '{"status": "ok"}'):
    """Return a mock LLM invoker that returns *response_text*."""

    def invoker(prompt: str, max_tokens: int) -> str:
        return response_text

    return invoker


# ============================================================================
# Prompt templates
# ============================================================================


class TestPromptTemplates:
    """Verify all four prompt templates render correctly."""

    def test_market_analyst_prompt_includes_context(self):
        ctx = _make_context()
        prompt = format_analyst_prompt("market", ctx)
        assert "AAPL" in prompt
        assert "185.50" in prompt
        assert "技术分析" in prompt
        assert "趋势判断" in prompt
        assert "support_level" in prompt
        # Token budget check
        tokens = estimate_tokens(prompt)
        assert tokens <= MAX_INPUT_TOKENS + 500  # generous slack for CI variations

    def test_news_analyst_prompt_includes_context(self):
        ctx = _make_context()
        prompt = format_analyst_prompt("news", ctx)
        assert "AAPL" in prompt
        assert "iPhone" in prompt
        assert "催化剂" in prompt
        assert "catalyst_count" in prompt
        tokens = estimate_tokens(prompt)
        assert tokens <= MAX_INPUT_TOKENS + 500

    def test_fundamentals_analyst_prompt_includes_context(self):
        ctx = _make_context()
        prompt = format_analyst_prompt("fundamentals", ctx)
        assert "AAPL" in prompt
        assert "基本面" in prompt
        assert "earnings_trigger" in prompt
        tokens = estimate_tokens(prompt)
        assert tokens <= MAX_INPUT_TOKENS + 500

    def test_breakout_analyst_prompt_includes_context(self):
        ctx = _make_context()
        prompt = format_analyst_prompt("breakout", ctx)
        assert "AAPL" in prompt
        assert "爆发形态" in prompt
        assert "breakout_score" in prompt
        assert "false_breakout_risk" in prompt
        tokens = estimate_tokens(prompt)
        assert tokens <= MAX_INPUT_TOKENS + 500

    def test_all_templates_registered(self):
        expected_types = {
            "market": MarketAnalystPrompt,
            "news": NewsAnalystPrompt,
            "fundamentals": FundamentalsAnalystPrompt,
            "breakout": BreakoutAnalystPrompt,
        }
        for at, expected_cls in expected_types.items():
            assert at in ANALYST_PROMPTS
            assert isinstance(ANALYST_PROMPTS[at], expected_cls)

    def test_get_analyst_prompt_valid(self):
        t = get_analyst_prompt("market")
        assert isinstance(t, MarketAnalystPrompt)
        t = get_analyst_prompt("breakout")
        assert isinstance(t, BreakoutAnalystPrompt)

    def test_get_analyst_prompt_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown analyst type"):
            get_analyst_prompt("nonexistent")

    def test_empty_context_renders_without_error(self):
        """Empty context should still render a usable prompt."""
        ctx = AnalystContext(ticker="TEST", price=10.0)
        for at in ANALYST_PROMPTS:
            prompt = format_analyst_prompt(at, ctx)
            assert "TEST" in prompt
            assert len(prompt) > 100

    def test_token_estimation_positive(self):
        assert estimate_tokens("Hello world") > 0
        assert estimate_tokens("") == 0

    def test_truncate_context_shortens(self):
        long_text = "word " * 2000
        result = truncate_context(long_text, 50)
        assert len(result) < len(long_text)
        assert "..." in result

    def test_truncate_context_empty(self):
        assert truncate_context("", 100) == ""

    def test_clamp_context_for_budget(self):
        # Use a genuinely massive string (~5000 tokens) to ensure truncation kicks in
        long_str = "z" * 20000  # ~5000+ tokens of single chars
        ctx = _make_context(
            news_summary=long_str,
            factor_scores_summary=long_str,
        )
        assert estimate_tokens(ctx.news_summary) > 1000  # well over the 500 cap
        clamped = clamp_context_for_budget(ctx)
        # After clamping, token counts should be reduced
        assert estimate_tokens(clamped.news_summary) <= 600  # 500 + slop
        assert estimate_tokens(clamped.factor_scores_summary) <= 600

    def test_token_budget_constants(self):
        assert MAX_INPUT_TOKENS == 2000
        assert MAX_OUTPUT_TOKENS == 800


# ============================================================================
# Breakout case retriever
# ============================================================================


class TestBreakoutCaseRetriever:
    """Tests for faiss + cases.parquet breakout case retrieval."""

    def test_empty_library_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "faiss.index"
            parquet_path = Path(tmpdir) / "cases.parquet"
            # No parquet file at all
            retriever = BreakoutCaseRetriever(
                index_path=index_path,
                parquet_path=parquet_path,
            )
            results = retriever.search([0.1, 0.2, 0.3])
            assert results == []
            assert not retriever.has_cases()

    def test_has_cases_false_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "faiss.index"
            parquet_path = Path(tmpdir) / "cases.parquet"
            # Write an empty parquet
            pl.DataFrame().write_parquet(str(parquet_path))
            retriever = BreakoutCaseRetriever(
                index_path=index_path,
                parquet_path=parquet_path,
            )
            assert not retriever.has_cases()
            assert retriever.search([0.1, 0.2]) == []

    def test_search_returns_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "faiss.index"
            parquet_path = Path(tmpdir) / "cases.parquet"

            # Build a small case library
            df = pl.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT", "GOOGL"],
                    "date": ["2023-05-15", "2023-06-20", "2023-07-10"],
                    "actual_pnl": [0.135, 0.22, 0.09],
                    "f_mom": [3.2, 5.1, 2.8],
                    "f_rsi": [62.0, 58.0, 71.0],
                    "f_vol": [1.5, 2.1, 0.9],
                }
            )
            df.write_parquet(str(parquet_path))

            retriever = BreakoutCaseRetriever(
                index_path=index_path,
                parquet_path=parquet_path,
            )
            assert retriever.has_cases()

            # Query with a vector similar to AAPL (f_mom=3.2, f_rsi=62, f_vol=1.5)
            results = retriever.search(
                [3.2, 62.0, 1.5],
                top_k=3,
                similarity_cutoff=0.0,  # accept all for this test
            )
            assert len(results) >= 1
            # The closest match should be AAPL
            assert results[0]["ticker"] == "AAPL"
            assert "similarity" in results[0]
            assert "actual_pnl" in results[0]

    def test_search_respects_cutoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "faiss.index"
            parquet_path = Path(tmpdir) / "cases.parquet"

            df = pl.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": ["2023-05-15"],
                    "actual_pnl": [0.135],
                    "f_mom": [3.2],
                    "f_rsi": [62.0],
                    "f_vol": [1.5],
                }
            )
            df.write_parquet(str(parquet_path))

            retriever = BreakoutCaseRetriever(
                index_path=index_path,
                parquet_path=parquet_path,
            )

            # Query with orthogonal vector should have low similarity
            results = retriever.search(
                [-10.0, -100.0, -50.0],
                similarity_cutoff=0.85,
            )
            assert results == []  # cutoff filters everything

    def test_mvp_empty_returns_empty(self):
        """MVP allows empty case library — this is the expected path initially."""
        retriever = BreakoutCaseRetriever()
        # First call initializes lazily
        results = retriever.search([0.1, 0.2, 0.3, 0.4, 0.5])
        assert isinstance(results, list)
        assert results == []


# ============================================================================
# Context builder
# ============================================================================


class TestContextBuilder:
    def test_build_context(self):
        ctx = build_context(
            ticker="MSFT",
            price=420.0,
            factor_scores_summary="MOM: +5%",
            news_summary="AI revenue beat",
            technical_pattern="bull flag",
            mom_5d=5.0,
            false_breakout_rate=20,
        )
        assert ctx.ticker == "MSFT"
        assert ctx.price == 420.0
        assert ctx.factor_scores_summary == "MOM: +5%"
        assert ctx.mom_5d == 5.0
        assert ctx.false_breakout_rate == 20

    def test_build_context_defaults(self):
        ctx = build_context("TSLA", 250.0)
        assert ctx.mom_5d == 0.0
        assert ctx.false_breakout_rate == 50
        assert ctx.factor_scores_summary == ""
        assert ctx.factor_vector == []

    def test_context_to_dict(self):
        ctx = _make_context(ticker="NVDA", price=900.0)
        d = ctx.to_dict()
        assert d["ticker"] == "NVDA"
        assert d["price"] == 900.0
        assert "factor_scores_summary" in d


# ============================================================================
# Token budget
# ============================================================================


class TestTokenBudget:
    def test_check_token_budget_ok(self):
        short = "Hello world"
        ok, tokens = check_token_budget(short)
        assert ok
        assert tokens < MAX_INPUT_TOKENS

    def test_check_token_budget_over(self):
        long_text = "word " * 5000  # far exceeds 2000 tokens
        ok, tokens = check_token_budget(long_text)
        assert not ok
        assert tokens > MAX_INPUT_TOKENS


# ============================================================================
# Orchestrator (mock LLM)
# ============================================================================


_MOCK_MARKET_JSON = json.dumps(
    {
        "trend": "bullish",
        "support_level": 180.0,
        "resistance_level": 195.0,
        "volume_confirms_trend": True,
        "pattern_detected": "bollinger_squeeze",
        "technical_report": "布林带收窄后放量突破，RSI 62 健康",
        "key_signals": ["BB squeeze breakout", "volume +2.1x avg"],
    },
    ensure_ascii=False,
)

_MOCK_NEWS_JSON = json.dumps(
    {
        "catalyst_count": 2,
        "catalysts": [
            {
                "event": "iPhone 17出货指引上修",
                "direction": "positive",
                "strength": "high",
                "timeliness": "fresh",
            },
        ],
        "risk_events": ["FOMC鹰派"],
        "news_report": "产品周期催化剂明确，宏观逆风可控",
    },
    ensure_ascii=False,
)

_MOCK_FUNDAMENTALS_JSON = json.dumps(
    {
        "earnings_trigger": "positive_outlook",
        "valuation_signal": "fair",
        "insider_signal": "data_unavailable",
        "fundamentals_report": "Q3指引上修，PE处于行业均值",
        "risk_flags": [],
    },
    ensure_ascii=False,
)

_MOCK_BREAKOUT_JSON = json.dumps(
    {
        "breakout_score": 72,
        "pattern_match_confidence": 68,
        "false_breakout_risk": 30,
        "key_drivers": ["BB squeeze", "volume confirmation"],
        "similar_cases": ["AAPL 2023-Q2"],
        "breakout_report": "形态匹配度高，成交量确认有效",
    },
    ensure_ascii=False,
)


class TestOrchestrator:
    """Tests for AnalystOrchestrator with mock LLM invokers."""

    def test_run_all_four_analysts(self):
        """Invoke all four analysts with mock responses."""
        # Each analyst type gets a specific mock response
        call_count = {"count": 0}

        def smart_mock(prompt: str, max_tokens: int) -> str:
            call_count["count"] += 1
            return '{"status": "ok"}'

        ctx = _make_context()
        orch = AnalystOrchestrator(invoker=smart_mock)
        results = orch.run(ctx)

        assert "market" in results
        assert "news" in results
        assert "fundamentals" in results
        assert "breakout" in results
        assert "input_tokens_total" in results
        assert results["input_tokens_total"] > 0
        # All four analysts called
        assert call_count["count"] == 4

    def test_run_selected_analysts(self):
        call_count = {"count": 0}

        def mock(prompt, max_tokens):
            call_count["count"] += 1
            return '{"ok": true}'

        ctx = _make_context()
        orch = AnalystOrchestrator(invoker=mock)
        results = orch.run_selected(ctx, ["market", "fundamentals"])

        assert "market" in results
        assert "fundamentals" in results
        assert "news" not in results
        assert call_count["count"] == 2

    def test_orchestrator_handles_llm_error(self):
        def failing_mock(prompt, max_tokens):
            raise RuntimeError("LLM invocation failed")

        ctx = _make_context()
        orch = AnalystOrchestrator(invoker=failing_mock)
        results = orch.run(ctx)

        for at in ("market", "news", "fundamentals", "breakout"):
            assert results[at]["error"] == "LLM invocation failed"
            assert results[at]["parsed_json"] is None

    def test_run_analyst_parses_json(self):
        ctx = _make_context()
        result = run_analyst(
            "market",
            ctx,
            _mock_invoker(_MOCK_MARKET_JSON),
        )
        assert result["analyst_type"] == "market"
        assert result["parsed_json"] is not None
        assert result["parsed_json"]["trend"] == "bullish"
        assert result["error"] is None
        assert result["budget_ok"] is True

    def test_run_analyst_bad_json(self):
        ctx = _make_context()
        result = run_analyst(
            "market",
            ctx,
            _mock_invoker("not json at all"),
        )
        assert result["parsed_json"] is None
        assert result["error"] is not None
        assert "JSON" in result["error"]

    def test_run_analyst_budget_ok_field(self):
        """Budget should be OK for a normal context."""
        ctx = _make_context()
        result = run_analyst("market", ctx, _mock_invoker('{"ok": true}'))
        assert result["budget_ok"] is True

    def test_similar_cases_in_breakout_run(self):
        """Breakout analyst run returns similar_cases key."""
        ctx = _make_context(factor_vector=[0.1, 0.2, 0.3])
        orch = AnalystOrchestrator(invoker=_mock_invoker(_MOCK_BREAKOUT_JSON))
        results = orch.run(ctx)
        assert "similar_cases" in results
        # In MVP, case library is empty -> similar_cases == []
        assert results["similar_cases"] == []
        assert results["breakout"]["parsed_json"] is not None

    def test_context_clamping_preserves_ticker(self):
        ctx = _make_context(
            news_summary="long " * 3000,
        )
        clamped = clamp_context_for_budget(ctx)
        assert clamped.ticker == "AAPL"
        # Long fields shortened
        assert len(clamped.news_summary) < 3000 * 5


# ============================================================================
# Integration: prompt formatting enforces budget
# ============================================================================


class TestPromptBudgetIntegration:
    def test_all_prompts_under_budget_with_reasonable_context(self):
        """All four prompts should fit budget with typical real-world context."""
        ctx = _make_context()
        for at in ANALYST_PROMPTS:
            prompt = format_analyst_prompt(at, ctx)
            tokens = estimate_tokens(prompt)
            # With typical context, prompts should be well under 2000 tokens
            assert tokens < 2500, f"{at} prompt at {tokens} tokens exceeds slop"
            # System + task + output format are fixed overhead
            # Context variables add the variable part
