"""Tests for Bull/Bear/PM pipeline and BreakoutAssessment schema.

Issue #98: Bull/Bear/PM pipeline + BreakoutAssessment.
Reference: PRD 4.2 / 4.3 / 4.4 / 4.5.1-4.5.5.
"""

from __future__ import annotations

import json
from unittest import mock as umock

import pytest

from alphascreener.tradingagents.bull_bear_pipeline import (
    ALLOWED_RISK_TAGS,
    SCORE_CORRECTION_MAX,
    SCORE_CORRECTION_MIN,
    SCORE_1_05_REQUIREMENTS,
    BULL_PROMPT,
    BEAR_PROMPT,
    PM_PROMPT,
    BatchConfig,
    BreakoutAssessment,
    BullBearContext,
    FinalRating,
    build_bull_bear_context,
    run_bull_bear_pm,
    run_pipeline_batch,
)


# ============================================================================
# Helpers
# ============================================================================


def _mock_invoker(response_text: str = '{"status": "ok"}',
                  input_tokens: int = 100, output_tokens: int = 50):
    """Factory for a mock LLM invoker that returns *response_text*."""
    def invoker(prompt: str, max_tokens: int) -> tuple[str, int, int]:
        return response_text, input_tokens, output_tokens
    return invoker


def _make_context(**overrides) -> BullBearContext:
    """Build a test BullBearContext with example data."""
    defaults = {
        "ticker": "AAPL",
        "price": 185.50,
        "mom_5d": 3.2,
        "factor_scores_summary": (
            "MOM_5D: +3.2% | RSI: 62 | MFI: 55 | ATR: 0.8% | VOL_ANOMALY: 1.5x"
        ),
        "news_summary": "iPhone 17出货指引上修; FOMC鹰派纪要; AAPL回购计划扩大",
        "technical_pattern": "布林带收窄12日后放量突破上轨, RSI 62未超买",
        "phase1_pass": True,
    }
    defaults.update(overrides)
    return BullBearContext(**defaults)


# ============================================================================
# 1. BreakoutAssessment Pydantic schema (PRD 4.3)
# ============================================================================


class TestBreakoutAssessmentSchema:
    """Validate the Pydantic model fields, clamping, and filtering."""

    def test_valid_minimal_construction(self):
        """Minimum required fields should produce a valid model."""
        ba = BreakoutAssessment(ticker="AAPL", final_rating=FinalRating.HOLD, breakout_probability=50.0)
        assert ba.ticker == "AAPL"
        assert ba.final_rating == FinalRating.HOLD
        assert ba.breakout_probability == 50.0
        # Defaults
        assert ba.bull_score == 50.0
        assert ba.bear_score == 50.0
        assert ba.score_correction == 1.00
        assert ba.data_conflict_detected is False
        assert ba.catalyst_consistency == 50.0
        assert ba.risk_tags == []
        assert ba.risk_flags == []
        assert ba.catalyst_events == []
        assert ba.phase1_pass is True

    def test_full_construction(self):
        """All fields populated."""
        ba = BreakoutAssessment(
            ticker="NVDA",
            final_rating=FinalRating.STRONG_BUY,
            breakout_probability=82.0,
            bull_score=78.0,
            bear_score=25.0,
            score_correction=1.03,
            data_conflict_detected=False,
            catalyst_consistency=72.0,
            risk_tags=["momentum_breakdown", "volume_divergence"],
            risk_flags=["watch for gap fill"],
            catalyst_events=["Earnings beat", "New product launch"],
            bull_thesis="Strong momentum with volume confirmation",
            bear_thesis="Overbought on daily RSI",
            pm_verdict="Bullish with manageable risk",
            phase1_pass=True,
        )
        assert ba.ticker == "NVDA"
        assert ba.final_rating == FinalRating.STRONG_BUY
        assert ba.breakout_probability == 82.0
        assert ba.risk_tags == ["momentum_breakdown", "volume_divergence"]

    def test_breakout_probability_clamped(self):
        """breakout_probability is clamped to [0, 100]."""
        ba_high = BreakoutAssessment(ticker="T1", final_rating=FinalRating.BUY, breakout_probability=150.0)
        assert ba_high.breakout_probability == 100.0

        ba_low = BreakoutAssessment(ticker="T2", final_rating=FinalRating.AVOID, breakout_probability=-20.0)
        assert ba_low.breakout_probability == 0.0

    def test_score_correction_clamped(self):
        """score_correction is clamped to [0.90, 1.05]."""
        ba_high = BreakoutAssessment(
            ticker="T1", final_rating=FinalRating.BUY, breakout_probability=60.0,
            score_correction=1.20,
        )
        assert ba_high.score_correction == SCORE_CORRECTION_MAX

        ba_low = BreakoutAssessment(
            ticker="T2", final_rating=FinalRating.AVOID, breakout_probability=10.0,
            score_correction=0.50,
        )
        assert ba_low.score_correction == SCORE_CORRECTION_MIN

    def test_risk_tags_filter_illegal(self):
        """Illegal risk tags are silently removed by the validator."""
        ba = BreakoutAssessment(
            ticker="T1",
            final_rating=FinalRating.HOLD,
            breakout_probability=50.0,
            risk_tags=["momentum_breakdown", "ILLEGAL_TAG", "volume_divergence", "BOGUS"],
        )
        assert sorted(ba.risk_tags) == sorted(["momentum_breakdown", "volume_divergence"])
        assert "ILLEGAL_TAG" not in ba.risk_tags
        assert "BOGUS" not in ba.risk_tags

    def test_all_allowed_risk_tags_valid(self):
        """All canonical risk tags pass through."""
        ba = BreakoutAssessment(
            ticker="T1",
            final_rating=FinalRating.HOLD,
            breakout_probability=50.0,
            risk_tags=list(ALLOWED_RISK_TAGS),
        )
        assert len(ba.risk_tags) == len(ALLOWED_RISK_TAGS)

    def test_final_rating_enum_values(self):
        """All four FinalRating enum values."""
        for rating in FinalRating:
            ba = BreakoutAssessment(
                ticker="T1",
                final_rating=rating,
                breakout_probability=50.0,
            )
            assert ba.final_rating == rating

    def test_is_score_correction_max_valid_triple_met(self):
        """score_correction=1.05 is valid when triple condition holds."""
        ba = BreakoutAssessment(
            ticker="AAPL",
            final_rating=FinalRating.STRONG_BUY,
            breakout_probability=85.0,
            bull_score=75.0,
            bear_score=30.0,
            score_correction=1.05,
            catalyst_consistency=70.0,
        )
        assert ba.is_score_correction_max_valid() is True

    def test_is_score_correction_max_valid_triple_fails_bull(self):
        """score_correction=1.05 invalid if bull_score < 70."""
        ba = BreakoutAssessment(
            ticker="AAPL",
            final_rating=FinalRating.STRONG_BUY,
            breakout_probability=85.0,
            bull_score=60.0,
            bear_score=30.0,
            score_correction=1.05,
            catalyst_consistency=70.0,
        )
        assert ba.is_score_correction_max_valid() is False

    def test_is_score_correction_max_valid_triple_fails_bear(self):
        """score_correction=1.05 invalid if bear_score > 40."""
        ba = BreakoutAssessment(
            ticker="AAPL",
            final_rating=FinalRating.STRONG_BUY,
            breakout_probability=85.0,
            bull_score=75.0,
            bear_score=50.0,
            score_correction=1.05,
            catalyst_consistency=70.0,
        )
        assert ba.is_score_correction_max_valid() is False

    def test_is_score_correction_max_valid_triple_fails_catalyst(self):
        """score_correction=1.05 invalid if catalyst_consistency < 60."""
        ba = BreakoutAssessment(
            ticker="AAPL",
            final_rating=FinalRating.STRONG_BUY,
            breakout_probability=85.0,
            bull_score=75.0,
            bear_score=30.0,
            score_correction=1.05,
            catalyst_consistency=50.0,
        )
        assert ba.is_score_correction_max_valid() is False

    def test_is_score_correction_max_valid_below_105(self):
        """If score_correction < 1.05, validation always returns True."""
        ba = BreakoutAssessment(
            ticker="AAPL",
            final_rating=FinalRating.BUY,
            breakout_probability=60.0,
            bull_score=30.0,
            bear_score=80.0,
            score_correction=1.04,
            catalyst_consistency=10.0,
        )
        assert ba.is_score_correction_max_valid() is True


# ============================================================================
# 2. BullBearContext
# ============================================================================


class TestBullBearContext:
    def test_default_construction(self):
        ctx = BullBearContext()
        assert ctx.ticker == ""
        assert ctx.price == 0.0
        assert ctx.phase1_pass is True

    def test_to_dict_includes_bull_bear_results(self):
        ctx = BullBearContext(
            ticker="MSFT",
            price=420.0,
            bull_result={"bull_score": 80},
            bear_result={"bear_score": 25},
        )
        d = ctx.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["bull_result"]  # serialised as JSON string
        assert d["bear_result"]
        # Verify JSON round-trip
        parsed_bull = json.loads(d["bull_result"])
        assert parsed_bull["bull_score"] == 80

    def test_to_dict_phase1_pass_string(self):
        ctx = BullBearContext(phase1_pass=False)
        d = ctx.to_dict()
        assert d["phase1_pass"] == "false"

        ctx = BullBearContext(phase1_pass=True)
        d = ctx.to_dict()
        assert d["phase1_pass"] == "true"


# ============================================================================
# 3. Prompt templates (PRD 4.5.1, 4.5.2, 4.5.3)
# ============================================================================


class TestResearcherPrompts:
    """Bull and Bear prompt templates."""

    def test_bull_prompt_includes_context(self):
        ctx = _make_context()
        prompt = BULL_PROMPT.format(ctx, side="bull")
        assert "AAPL" in prompt
        assert "185.50" in prompt
        assert "多头" in prompt
        assert "bull_score" in prompt
        assert "bull_thesis" in prompt

    def test_bear_prompt_includes_context(self):
        ctx = _make_context()
        prompt = BEAR_PROMPT.format(ctx, side="bear")
        assert "AAPL" in prompt
        assert "185.50" in prompt
        assert "空头" in prompt
        assert "bear_score" in prompt
        assert "bear_thesis" in prompt

    def test_bull_prompt_structure(self):
        """Bull prompt has all required structural elements."""
        ctx = _make_context()
        prompt = BULL_PROMPT.format(ctx, side="bull")
        assert "## Role" in prompt
        assert "## Context" in prompt
        assert "## Task" in prompt
        assert "## Output Format" in prompt

    def test_bear_prompt_structure(self):
        """Bear prompt has all required structural elements."""
        ctx = _make_context()
        prompt = BEAR_PROMPT.format(ctx, side="bear")
        assert "## Role" in prompt
        assert "## Context" in prompt
        assert "## Task" in prompt
        assert "## Output Format" in prompt

    def test_empty_context_renders(self):
        """Empty context still renders usable prompts."""
        ctx = BullBearContext(ticker="TEST", price=10.0)
        bull = BULL_PROMPT.format(ctx, side="bull")
        assert "TEST" in bull
        assert len(bull) > 100

        bear = BEAR_PROMPT.format(ctx, side="bear")
        assert "TEST" in bear
        assert len(bear) > 100


class TestPMPrompt:
    """Portfolio Manager prompt template."""

    def test_pm_prompt_includes_context(self):
        ctx = _make_context(
            bull_result={"bull_score": 80, "bull_thesis": "strong upside"},
            bear_result={"bear_score": 30, "bear_thesis": "limited risk"},
        )
        prompt = PM_PROMPT.format(ctx)
        assert "AAPL" in prompt
        assert "bull_score" in prompt
        assert "data_conflict_detected" in prompt
        assert "score_correction" in prompt
        assert "catalyst_consistency" in prompt

    def test_pm_prompt_includes_risk_tags(self):
        ctx = _make_context()
        prompt = PM_PROMPT.format(ctx)
        assert "momentum_breakdown" in prompt
        assert "volume_divergence" in prompt
        assert "false_breakout_risk" in prompt

    def test_pm_prompt_structure(self):
        ctx = _make_context()
        prompt = PM_PROMPT.format(ctx)
        assert "## Role" in prompt
        assert "## Context" in prompt
        assert "## Task" in prompt
        assert "## Output Format" in prompt


# ============================================================================
# 4. build_bull_bear_context convenience
# ============================================================================


class TestBuildContext:
    def test_build_with_all_fields(self):
        ctx = build_bull_bear_context(
            "TSLA",
            250.0,
            mom_5d=5.0,
            factor_scores_summary="RSI:58",
            news_summary="Delivery beat",
            technical_pattern="bull flag",
            phase1_pass=True,
            factor_vector=[0.1, 0.2, 0.3],
        )
        assert ctx.ticker == "TSLA"
        assert ctx.price == 250.0
        assert ctx.mom_5d == 5.0
        assert ctx.factor_scores_summary == "RSI:58"
        assert ctx.phase1_pass is True
        assert ctx.factor_vector == [0.1, 0.2, 0.3]

    def test_build_defaults(self):
        ctx = build_bull_bear_context("AAPL", 100.0)
        assert ctx.mom_5d == 0.0
        assert ctx.factor_scores_summary == ""
        assert ctx.phase1_pass is True
        assert ctx.factor_vector == []


# ============================================================================
# 5. run_bull_bear_pm — single symbol with mock LLM
# ============================================================================

_BULL_JSON = json.dumps(
    {
        "bull_score": 78,
        "momentum_signal": "strong_positive",
        "bull_thesis": "布林带突破+量能确认,目标T+7上探195",
        "bull_catalysts": ["iPhone 17出货上修", "回购计划扩大"],
        "key_evidence": ["MOM_5D: +3.2%", "volume +2.1x avg"],
        "risk_acknowledged": ["FOMC鹰派"],
    },
    ensure_ascii=False,
)

_BEAR_JSON = json.dumps(
    {
        "bear_score": 32,
        "bear_thesis": "RSI接近超买, 宏观逆风, 上行动能可能衰竭",
        "bear_catalysts": ["FOMC鹰派纪要"],
        "key_risks": ["RSI 62 接近超买区", "PE估值偏高"],
        "bull_rebuttals": ["放量突破有量能支撑"],
    },
    ensure_ascii=False,
)

_PM_JSON = json.dumps(
    {
        "final_rating": "Buy",
        "breakout_probability": 72,
        "score_correction": 1.02,
        "data_conflict_detected": False,
        "catalyst_consistency": 68,
        "risk_tags": ["volume_divergence", "macro_headwind"],
        "risk_flags": ["关注FOMC后续纪要"],
        "pm_verdict": "多头逻辑更强, 但需关注宏观逆风, 维持Buy评级",
    },
    ensure_ascii=False,
)


class TestRunBullBearPM:
    """End-to-end single-symbol pipeline with mock invokers."""

    def test_full_pipeline_all_valid(self):
        """Happy path: all three stages return valid JSON."""
        call_order: list[str] = []

        def mock_invoker(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            # Detect which stage we are in by unique prompt markers.
            # PM prompt includes "投资组合经理" (PM role) + "final_rating" output instruction.
            # Bull prompt includes "多头研究员" (Bull role).
            # Bear prompt includes "空头研究员" (Bear role).
            # The PM prompt also contains bull/bear result JSON, so we check
            # PM BEFORE Bull/Bear to avoid false matches.
            call_order.append("call")
            pl = prompt.lower()
            # PM prompt: contains PM role marker and "数据冲突"
            if "投资组合经理" in prompt or "数据冲突检测" in prompt:
                return _PM_JSON, 100, 50
            # Bull prompt: contains Bull-specific role
            if "多头研究员" in prompt:
                return _BULL_JSON, 100, 50
            # Bear prompt: contains Bear-specific role
            if "空头研究员" in prompt:
                return _BEAR_JSON, 100, 50
            # Retry prompts contain the original role text too, so above should match.
            # Ultimate fallback by call position:
            idx = len(call_order)  # last match position (1-based)
            if idx <= 1:
                return _BULL_JSON, 100, 50
            elif idx == 2:
                return _BEAR_JSON, 100, 50
            else:
                return _PM_JSON, 100, 50

        ctx = _make_context()
        result = run_bull_bear_pm(ctx, mock_invoker)

        assert isinstance(result, BreakoutAssessment)
        assert result.ticker == "AAPL"
        assert result.final_rating == FinalRating.BUY
        assert result.breakout_probability == 72.0
        assert result.bull_score == 78.0
        assert result.bear_score == 32.0
        assert result.score_correction == 1.02
        assert result.data_conflict_detected is False
        assert result.catalyst_consistency == 68.0
        assert "volume_divergence" in result.risk_tags
        assert result.bull_thesis != ""
        assert result.bear_thesis != ""
        assert "iPhone 17" in " ".join(result.catalyst_events)
        assert result.validation_errors == []

    def test_bull_invocation_failure_fallback(self):
        """Bull LLM call fails -> uses defaults, pipeline continues."""
        calls: list[str] = []

        def smarter_mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            calls.append("call")
            # Check PM first (most specific)
            if "投资组合经理" in prompt or "数据冲突检测" in prompt:
                return _PM_JSON, 100, 50
            # Bear
            if "空头研究员" in prompt:
                return _BEAR_JSON, 100, 50
            # Bull raises
            if "多头研究员" in prompt:
                raise RuntimeError("Bull API timeout")
            # Fallback by position
            if len(calls) == 1:
                raise RuntimeError("Bull API timeout")
            elif len(calls) == 2:
                return _BEAR_JSON, 100, 50
            else:
                return _PM_JSON, 100, 50

        ctx = _make_context()
        result = run_bull_bear_pm(ctx, smarter_mock)

        assert result.ticker == "AAPL"
        # Bull fell back to defaults
        assert result.bull_score == 50.0
        # Bear should have been called
        assert result.bear_score == 32.0
        # PM also called
        assert result.final_rating == FinalRating.BUY
        assert len(result.validation_errors) >= 1
        assert any("Bull" in e for e in result.validation_errors)

    def test_json_parse_failure_with_retry(self):
        """JSON parse failure triggers one retry (PRD 4.5.5)."""
        call_count: list[str] = []

        def mock_bad_then_good(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            call_count.append("call")
            # Retry prompts contain "上一轮输出不是合法 JSON"
            if "上一轮" in prompt:
                # This is a retry — use prompt role markers
                if "多头研究员" in prompt:
                    return _BULL_JSON, 100, 50
                elif "空头研究员" in prompt:
                    return _BEAR_JSON, 100, 50
                else:
                    return _PM_JSON, 100, 50
            # First call returns garbage for ALL stages
            return "not valid json at all {{{", 100, 50

        ctx = _make_context()
        result = run_bull_bear_pm(ctx, mock_bad_then_good)

        assert isinstance(result, BreakoutAssessment)
        # PM should have been reached (with defaults after failed retries
        # since the retry prompt also doesn't match our detection well)
        assert result.ticker == "AAPL"
        # Three stages * (initial + retry) = 6 invocations
        assert len(call_count) == 6

    def test_pm_json_parse_failure_fallback(self):
        """PM JSON parse fails -> uses validated defaults."""
        call_count = {"bull": 0, "bear": 0, "pm": 0}

        def mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            if "多头研究员" in prompt:
                call_count["bull"] += 1
                return _BULL_JSON, 100, 50
            elif "空头研究员" in prompt:
                call_count["bear"] += 1
                return _BEAR_JSON, 100, 50
            else:
                call_count["pm"] += 1
                # Both initial and retry return bad data
                return "garbage {{{ not json", 100, 50

        ctx = _make_context()
        result = run_bull_bear_pm(ctx, mock)

        assert result.ticker == "AAPL"
        assert result.bull_score == 78.0
        assert result.bear_score == 32.0
        # PM fell back to validated defaults
        assert result.final_rating == FinalRating.HOLD  # default
        assert result.breakout_probability == 50.0  # default

    def test_phase1_fail_does_not_override_in_schema(self):
        """Phase 1 pass status is tracked but does not force Avoid here
        (enforcement happens at PM prompt level, not Pydantic level)."""
        ctx = _make_context(phase1_pass=False)
        assert ctx.phase1_pass is False
        # The context correctly records the status

    def test_score_correction_105_rejected_by_pm_validator(self):
        """PM validator rejects score_correction=1.05 when conditions not met."""
        # Bull=50, Bear=50 -> triple condition fails
        from alphascreener.tradingagents.bull_bear_pipeline import _validate_pm_result

        pm_parsed = {
            "final_rating": "Buy",
            "breakout_probability": 80,
            "score_correction": 1.05,
            "data_conflict_detected": False,
            "catalyst_consistency": 40,  # below 60
            "risk_tags": [],
            "risk_flags": [],
            "pm_verdict": "test",
        }

        result = _validate_pm_result(pm_parsed, bull_score=50.0, bear_score=50.0)
        # Should be rejected down to 1.04
        assert result["score_correction"] == 1.04
        assert len(result["validation_errors"]) >= 1

    def test_score_correction_105_accepted_when_triple_met(self):
        """PM validator accepts score_correction=1.05 when triple condition met."""
        from alphascreener.tradingagents.bull_bear_pipeline import _validate_pm_result

        pm_parsed = {
            "final_rating": "Strong Buy",
            "breakout_probability": 85,
            "score_correction": 1.05,
            "data_conflict_detected": False,
            "catalyst_consistency": 70,  # >= 60
            "risk_tags": [],
            "risk_flags": [],
            "pm_verdict": "test",
        }

        result = _validate_pm_result(pm_parsed, bull_score=75.0, bear_score=30.0)
        assert result["score_correction"] == 1.05

    def test_illegal_risk_tags_filtered_in_pm_result(self):
        """PM output with illegal risk tags has them filtered."""
        from alphascreener.tradingagents.bull_bear_pipeline import _validate_pm_result

        pm_parsed = {
            "final_rating": "Buy",
            "breakout_probability": 70,
            "score_correction": 1.00,
            "data_conflict_detected": False,
            "catalyst_consistency": 60,
            "risk_tags": ["momentum_breakdown", "BAD_TAG", "BOGUS"],
            "risk_flags": [],
            "pm_verdict": "test",
        }

        result = _validate_pm_result(pm_parsed)
        assert "momentum_breakdown" in result["risk_tags"]
        assert "BAD_TAG" not in result["risk_tags"]
        assert "BOGUS" not in result["risk_tags"]

    def test_missing_fields_get_defaults(self):
        """Missing PM fields are filled with sensible defaults (PRD 4.5.5)."""
        from alphascreener.tradingagents.bull_bear_pipeline import _validate_pm_result

        result = _validate_pm_result({})
        assert result["final_rating"] == "Hold"
        assert result["breakout_probability"] == 50.0
        assert result["score_correction"] == 1.00
        assert result["data_conflict_detected"] is False
        assert result["catalyst_consistency"] == 50.0
        assert result["risk_tags"] == []
        assert result["risk_flags"] == []
        assert result["pm_verdict"] == ""

    def test_invalid_final_rating_defaults_to_hold(self):
        """Invalid final_rating string defaults to Hold."""
        from alphascreener.tradingagents.bull_bear_pipeline import _validate_pm_result

        result = _validate_pm_result({"final_rating": "SuperDuperBuy"})
        assert result["final_rating"] == "Hold"

    def test_retry_disabled_returns_defaults_on_json_failure(self):
        """When retry_on_json_failure=False, JSON failures return defaults."""
        def mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            return "not json", 100, 50

        ctx = _make_context()
        result = run_bull_bear_pm(ctx, mock, retry_on_json_failure=False)

        assert result.ticker == "AAPL"
        # All three stages failed JSON parse, all got defaults
        assert result.bull_score == 50.0
        assert result.bear_score == 50.0
        assert result.final_rating == FinalRating.HOLD


# ============================================================================
# 6. Batch processing
# ============================================================================


class TestPipelineBatch:
    """Tests for run_pipeline_batch multi-symbol orchestration."""

    def test_single_batch_single_symbol(self):
        """One symbol in one batch."""
        call_count = {"total": 0}

        def mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            call_count["total"] += 1
            if "多头研究员" in prompt:
                return _BULL_JSON, 100, 50
            elif "空头研究员" in prompt:
                return _BEAR_JSON, 100, 50
            else:
                return _PM_JSON, 100, 50

        ctx = _make_context()
        config = BatchConfig(batch_size=3, n_batches=7)
        results = run_pipeline_batch([ctx], mock, config)

        assert len(results) == 1
        assert isinstance(results[0], BreakoutAssessment)
        assert results[0].ticker == "AAPL"

    def test_multiple_symbols_multiple_batches(self):
        """3 symbols across 2 batches (batch_size=2)."""
        call_count = {"total": 0}

        def mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            call_count["total"] += 1
            # Simple rotation for deterministic testing
            mod = call_count["total"] % 3
            if mod == 1:
                return _BULL_JSON, 100, 50
            elif mod == 2:
                return _BEAR_JSON, 100, 50
            else:
                return _PM_JSON, 100, 50

        contexts = [
            _make_context(ticker="A"),
            _make_context(ticker="B"),
            _make_context(ticker="C"),
        ]
        config = BatchConfig(batch_size=2, n_batches=7)
        results = run_pipeline_batch(contexts, mock, config)

        assert len(results) == 3
        tickers = [r.ticker for r in results]
        assert tickers == ["A", "B", "C"]

    def test_batch_respects_n_batches_limit(self):
        """n_batches=1 with batch_size=2 should process only 2 of 5 symbols."""
        call_count = {"total": 0}

        def mock(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            call_count["total"] += 1
            mod = call_count["total"] % 3
            if mod == 1:
                return _BULL_JSON, 100, 50
            elif mod == 2:
                return _BEAR_JSON, 100, 50
            else:
                return _PM_JSON, 100, 50

        contexts = [_make_context(ticker=str(i)) for i in range(5)]
        config = BatchConfig(batch_size=2, n_batches=1)
        results = run_pipeline_batch(contexts, mock, config)

        # Only first batch of 2 processed
        assert len(results) == 2

    def test_empty_contexts(self):
        """Empty input produces empty output."""
        mock = _mock_invoker('{"status": "ok"}')
        results = run_pipeline_batch([], mock)
        assert results == []


# ============================================================================
# 7. Constants
# ============================================================================


class TestConstants:
    def test_score_correction_bounds(self):
        assert SCORE_CORRECTION_MIN == 0.90
        assert SCORE_CORRECTION_MAX == 1.05

    def test_105_requirements(self):
        reqs = SCORE_1_05_REQUIREMENTS
        assert "bull_breakout_score >= 70" in reqs
        assert "bear_risk_score <= 40" in reqs
        assert "catalyst_consistency >= 60" in reqs


# ============================================================================
# 8. Prompt format independence from context mutation
# ============================================================================


class TestPromptNonMutation:
    """Formatting prompts must not mutate the context."""

    def test_bull_prompt_does_not_mutate_context(self):
        ctx = _make_context(ticker="ORIGINAL")
        original_ticker = ctx.ticker
        BULL_PROMPT.format(ctx, side="bull")
        assert ctx.ticker == original_ticker

    def test_pm_prompt_does_not_mutate_context(self):
        ctx = _make_context(
            ticker="ORIGINAL",
            bull_result={"s": 80},
            bear_result={"s": 20},
        )
        original_ticker = ctx.ticker
        PM_PROMPT.format(ctx)
        assert ctx.ticker == original_ticker
