"""Bull/Bear adversarial analysis + PM risk audit + BreakoutAssessment schema.

Issue #98: Bull/Bear/PM pipeline + BreakoutAssessment.
Reference: PRD 4.2 / 4.3 / 4.4 / 4.5.1-4.5.5.

Pipeline per symbol:
  1. Bull Researcher (4.5.1) ─┐
  2. Bear Researcher (4.5.2) ─┤ parallel, then merge
  3. Portfolio Manager (4.5.3) ─┘ score correction + risk audit
  4. Output validation (4.5.5) → BreakoutAssessment Pydantic model

Execution model: batch-serial (3 symbols/batch, 7 batches),
within each batch (Bull || Bear) → PM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from alphascreener.logging import get_logger

if TYPE_CHECKING:
    from alphascreener.cost.tracker import CostTracker

_logger = get_logger("screening")

# ============================================================================
# 1. Pydantic BreakoutAssessment schema (PRD 4.3)
# ============================================================================


class FinalRating(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    AVOID = "Avoid"


# Canonical set of risk tags accepted by the system.
ALLOWED_RISK_TAGS: frozenset[str] = frozenset(
    {
        "momentum_breakdown",
        "volume_divergence",
        "false_breakout_risk",
        "catalyst_fade",
        "overbought",
        "oversold",
        "gap_risk",
        "liquidity_risk",
        "event_risk",
        "sector_headwind",
        "macro_headwind",
        "data_conflict",
        "insider_selling",
        "earnings_miss_risk",
        "valuation_stretch",
        "news_contradiction",
        "data_unavailable",
    }
)

# score_correction hard bounds (PRD 4.5.5)
SCORE_CORRECTION_MIN: float = 0.90
SCORE_CORRECTION_MAX: float = 1.05

# Conditions that must ALL hold for score_correction to be raised to 1.05
SCORE_1_05_REQUIREMENTS: tuple[str, ...] = (
    "bull_breakout_score >= 70",
    "bear_risk_score <= 40",
    "catalyst_consistency >= 60",
)


class BreakoutAssessment(BaseModel):
    """Complete assessment output for a single symbol after Bull/Bear/PM pipeline.

    Reference: PRD 4.3 — all required fields for downstream paper trading
    and alpha acceptance tracking.

    Field constraints (ge/le) are enforced via ``@field_validator`` instead of
    ``Field(ge=, le=)`` so that out-of-range inputs are clamped rather than
    rejected — matching the PRD 4.5.5 robustness contract.
    """

    ticker: str = Field(..., description="Symbol under assessment")
    final_rating: FinalRating = Field(..., description="Final buy/sell rating")
    breakout_probability: float = Field(
        ...,
        description="Final breakout probability (clamped to [0, 100])",
    )

    # Bull / Bear scores
    bull_score: float = Field(
        default=50.0, description="Bull researcher confidence (clamped to [0, 100])"
    )
    bear_score: float = Field(
        default=50.0, description="Bear researcher risk score (clamped to [0, 100])"
    )

    # PM correction (PRD 4.5.3)
    score_correction: float = Field(
        default=1.00,
        description="Portfolio Manager multiplicative correction factor (clamped to [0.90, 1.05])",
    )
    data_conflict_detected: bool = Field(
        default=False,
        description="PM detected contradictory signals across analysts",
    )
    catalyst_consistency: float = Field(
        default=50.0,
        description="Consistency between catalyst events and technical signals (clamped to [0, 100])",
    )

    # Risk audit
    risk_tags: list[str] = Field(
        default_factory=list,
        description="Risk tags from PM audit (canonical set only)",
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Free-text risk flags from PM for human review",
    )

    # Catalyst events (merged from Bull/Bear/News)
    catalyst_events: list[str] = Field(
        default_factory=list,
        description="Key catalyst events identified across researchers",
    )

    # Supporting detail
    bull_thesis: str = Field(default="", description="Bull researcher's core thesis")
    bear_thesis: str = Field(default="", description="Bear researcher's core thesis")
    pm_verdict: str = Field(
        default="", description="Portfolio Manager's summary verdict"
    )

    # Phase-1 filter pass-through
    phase1_pass: bool = Field(
        default=True, description="Whether the symbol passed Phase 1 hard filters"
    )

    # Metadata
    input_tokens_total: int = Field(
        default=0, description="Total input tokens consumed across all LLM calls"
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation issues encountered during output processing",
    )

    @field_validator("risk_tags")
    @classmethod
    def _filter_risk_tags(cls, v: list[str]) -> list[str]:
        """Filter out illegal risk tags, keeping only canonical values (PRD 4.5.5)."""
        if not v:
            return []
        filtered = [t for t in v if t in ALLOWED_RISK_TAGS]
        removed = len(v) - len(filtered)
        if removed:
            _logger.debug("Filtered %d illegal risk tags", removed)
        return filtered

    @field_validator("score_correction")
    @classmethod
    def _clamp_score_correction(cls, v: float) -> float:
        """Clamp score_correction to [0.90, 1.05] (PRD 4.5.5)."""
        return max(SCORE_CORRECTION_MIN, min(SCORE_CORRECTION_MAX, v))

    @field_validator("breakout_probability")
    @classmethod
    def _clamp_breakout_probability(cls, v: float) -> float:
        """Clamp breakout_probability to [0, 100]."""
        return max(0.0, min(100.0, v))

    @field_validator("bull_score")
    @classmethod
    def _clamp_bull_score(cls, v: float) -> float:
        """Clamp bull_score to [0, 100]."""
        return max(0.0, min(100.0, v))

    @field_validator("bear_score")
    @classmethod
    def _clamp_bear_score(cls, v: float) -> float:
        """Clamp bear_score to [0, 100]."""
        return max(0.0, min(100.0, v))

    @field_validator("catalyst_consistency")
    @classmethod
    def _clamp_catalyst_consistency(cls, v: float) -> float:
        """Clamp catalyst_consistency to [0, 100]."""
        return max(0.0, min(100.0, v))

    def is_score_correction_max_valid(self) -> bool:
        """Check whether score_correction == 1.05 satisfies the triple condition.

        Reference PRD 4.5.5: score_correction may only be raised to 1.05
        when all three of the following hold:
          - bull_score >= 70
          - bear_score <= 40
          - catalyst_consistency >= 60
        """
        if self.score_correction < 1.05:
            return True
        return (
            self.bull_score >= 70
            and self.bear_score <= 40
            and self.catalyst_consistency >= 60
        )


# ============================================================================
# 2. Bull Researcher prompt (PRD 4.5.1)
# ============================================================================

BULL_RESEARCHER_SYSTEM = """## Role
你是一位多头研究员（Bull Researcher），你的任务是构建标的的看涨逻辑。
你必须基于提供的技术指标、因子评分和新闻摘要，建立最强有力的看涨论证。

## 原则
1. 以数据驱动，每个结论必须有因子或指标支撑
2. 承认风险但聚焦于上行潜力
3. 用具体数字和阈值说话，避免模糊判断"""

BULL_RESEARCHER_CONTEXT_VARS: tuple[str, ...] = (
    "ticker",
    "price",
    "mom_5d",
    "factor_scores_summary",
    "news_summary",
    "technical_pattern",
)

BULL_RESEARCHER_TASK = """基于以上信息，完成多头论证:
1. **动量评估**: MOM_5D 方向和强度，是否有持续动能
2. **技术面论证**: 支撑位、突破形态、量价配合的看涨信号
3. **催化剂分析**: 从新闻中提取看涨催化剂，评估 T+7 影响
4. **风险对冲**: 识别主要风险但解释为何看涨逻辑更强
5. **核心论点**: 用 ≤100 字总结最强看涨理由

输出结构化的 Bull 分析报告。"""

BULL_RESEARCHER_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "bull_score": int (0-100, 多头置信度),
  "momentum_signal": "strong_positive" | "positive" | "neutral" | "negative",
  "bull_thesis": "<100字 核心看涨论点>",
  "bull_catalysts": ["催化剂1", "催化剂2", ...],
  "key_evidence": ["证据1: 具体指标", "证据2: ..."],
  "risk_acknowledged": ["风险1", "风险2"]
}"""


# ============================================================================
# 3. Bear Researcher prompt (PRD 4.5.2)
# ============================================================================

BEAR_RESEARCHER_SYSTEM = """## Role
你是一位空头研究员（Bear Researcher），你的任务是构建标的的看跌逻辑。
你必须基于提供的技术指标、因子评分和新闻摘要，建立最强有力的看跌论证。

## 原则
1. 以数据驱动，每个结论必须有因子或指标支撑
2. 承认利好但聚焦于下行风险
3. 用具体数字和阈值说话，避免模糊判断"""

BEAR_RESEARCHER_CONTEXT_VARS: tuple[str, ...] = (
    "ticker",
    "price",
    "mom_5d",
    "factor_scores_summary",
    "news_summary",
    "technical_pattern",
)

BEAR_RESEARCHER_TASK = """基于以上信息，完成空头论证:
1. **动量风险**: MOM_5D 是否可能反转，动能衰竭信号
2. **技术面风险**: 阻力位压制、假突破迹象、量价背离
3. **催化剂风险**: 从新闻中提取潜在利空，评估 T+7 负面影响
4. **反方论点**: 针对典型看涨逻辑的逐一反驳
5. **核心风险**: 用 ≤100 字总结最大下行风险

输出结构化的 Bear 分析报告。"""

BEAR_RESEARCHER_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "bear_score": int (0-100, 空头风险评分, 越高表示风险越大),
  "bear_thesis": "<100字 核心看跌论点>",
  "bear_catalysts": ["风险催化剂1", "风险催化剂2", ...],
  "key_risks": ["风险1: 具体指标", "风险2: ..."],
  "bull_rebuttals": ["看涨论点的反驳1", "反驳2"]
}"""


# ============================================================================
# 4. Portfolio Manager prompt (PRD 4.5.3)
# ============================================================================

PM_SYSTEM = """## Role
你是一位投资组合经理（Portfolio Manager），负责最终决策。
你需要审查 Bull/Bear 双方的分析，综合评估后给出:
- score_correction（因子评分修正系数）
- risk_tags（风险标签）
- 最终评级和爆发概率
- 数据冲突检测和催化剂一致性评估

## 硬约束
- score_correction ∈ [0.90, 1.05]
- 上调至 1.05 必须同时满足: bull_score ≥ 70, bear_score ≤ 40, catalyst_consistency ≥ 60
- risk_tags 只能从预定义集合中选择
- 输出必须为合法 JSON"""

PM_CONTEXT_VARS: tuple[str, ...] = (
    "ticker",
    "price",
    "bull_result",
    "bear_result",
    "factor_scores_summary",
    "phase1_pass",
)

PM_TASK = """基于以上 Bull/Bear 分析结果，完成最终决策:
1. **数据冲突检测**: Bull 和 Bear 在哪些指标上存在矛盾判断？
2. **催化剂一致性**: 催化剂事件与技术信号方向是否一致（0-100）？
3. **评分修正**: 根据矛盾程度和催化剂信心，给出 score_correction
4. **风险标签**: 从预定义集合中选择适用的风险标签（最多5个）
5. **最终评级**: Strong Buy | Buy | Hold | Avoid
6. **爆发概率**: 综合技术面、催化剂和风险，给出 0-100 爆发概率

注意:
- 若 phase1_pass = false，最终评级强制为 Avoid，爆发概率 ≤ 30
- risk_tags 必须从 canonical set 中选择
- score_correction < 1.00 表示向下修正, > 1.00 表示向上修正"""

PM_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "final_rating": "Strong Buy" | "Buy" | "Hold" | "Avoid",
  "breakout_probability": int (0-100),
  "score_correction": float (0.90-1.05),
  "data_conflict_detected": true | false,
  "catalyst_consistency": int (0-100),
  "risk_tags": ["momentum_breakdown", "volume_divergence", ...],
  "risk_flags": ["自由文本风险标记1", ...],
  "pm_verdict": "<200字符 最终裁决摘要>"
}

可用的 risk_tags (canonical set):
momentum_breakdown, volume_divergence, false_breakout_risk, catalyst_fade,
overbought, oversold, gap_risk, liquidity_risk, event_risk, sector_headwind,
macro_headwind, data_conflict, insider_selling, earnings_miss_risk,
valuation_stretch, news_contradiction, data_unavailable"""


# ============================================================================
# 5. Prompt templates (reusing AnalystPromptTemplate pattern from prompts.py)
# ============================================================================


@dataclass
class BullBearContext:
    """Context variables for Bull/Bear/PM pipeline.

    Extended from AnalystContext with bull/bear analyst results
    and Phase 1 filter status.
    """

    ticker: str = ""
    price: float = 0.0
    mom_5d: float = 0.0
    factor_scores_summary: str = ""
    news_summary: str = ""
    technical_pattern: str = ""
    bull_result: dict[str, Any] = field(default_factory=dict)
    bear_result: dict[str, Any] = field(default_factory=dict)
    phase1_pass: bool = True
    factor_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "mom_5d": self.mom_5d,
            "factor_scores_summary": self.factor_scores_summary,
            "news_summary": self.news_summary,
            "technical_pattern": self.technical_pattern,
            "bull_result": json.dumps(self.bull_result, ensure_ascii=False),
            "bear_result": json.dumps(self.bear_result, ensure_ascii=False),
            "phase1_pass": "true" if self.phase1_pass else "false",
        }


class ResearcherPrompt:
    """Prompt template for a single researcher (Bull or Bear)."""

    def __init__(
        self,
        analyst_name: str,
        system_message: str,
        context_vars: tuple[str, ...],
        task_description: str,
        output_format_desc: str,
    ) -> None:
        self.analyst_name = analyst_name
        self.system_message = system_message
        self.context_vars = context_vars
        self.task_description = task_description
        self.output_format_desc = output_format_desc

    def format(self, context: BullBearContext, side: str) -> str:
        """Render the full researcher prompt.

        Args:
            context: Run-time context.
            side: ``"bull"`` or ``"bear"`` (for logging / prompt differentiation).

        Returns:
            Complete prompt string.
        """
        ctx = context.to_dict()
        parts: list[str] = []

        parts.append(self.system_message)
        parts.append("\n## Context (系统注入)")
        for var in self.context_vars:
            value = ctx.get(var, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            parts.append(f"- {var}: {value}")

        parts.append("\n## Task")
        parts.append(self.task_description)

        parts.append("\n## Output Format")
        parts.append(self.output_format_desc)

        return "\n".join(parts)


class PortfolioManagerPrompt:
    """Prompt template for the Portfolio Manager risk audit step."""

    def __init__(self) -> None:
        pass

    def format(self, context: BullBearContext) -> str:
        """Render the PM prompt.

        Args:
            context: Context including Bull and Bear results.

        Returns:
            Complete PM prompt string.
        """
        ctx = context.to_dict()
        parts: list[str] = []

        parts.append(PM_SYSTEM)
        parts.append("\n## Context (系统注入)")
        for var in PM_CONTEXT_VARS:
            value = ctx.get(var, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            parts.append(f"- {var}: {value}")

        parts.append("\n## Task")
        parts.append(PM_TASK)

        parts.append("\n## Output Format")
        parts.append(PM_OUTPUT_FMT)

        return "\n".join(parts)


# Pre-built prompt instances
BULL_PROMPT = ResearcherPrompt(
    analyst_name="Bull Researcher",
    system_message=BULL_RESEARCHER_SYSTEM,
    context_vars=BULL_RESEARCHER_CONTEXT_VARS,
    task_description=BULL_RESEARCHER_TASK,
    output_format_desc=BULL_RESEARCHER_OUTPUT_FMT,
)

BEAR_PROMPT = ResearcherPrompt(
    analyst_name="Bear Researcher",
    system_message=BEAR_RESEARCHER_SYSTEM,
    context_vars=BEAR_RESEARCHER_CONTEXT_VARS,
    task_description=BEAR_RESEARCHER_TASK,
    output_format_desc=BEAR_RESEARCHER_OUTPUT_FMT,
)

PM_PROMPT = PortfolioManagerPrompt()


# ============================================================================
# 6. Invoker protocol
# ============================================================================

Invoker = Any  # callable (system_prompt: str, max_output_tokens: int) -> tuple[str, int, int]

# ============================================================================
# 7. JSON extraction & retry (PRD 4.5.5)
# ============================================================================

_JSON_PATTERN_BB = re.compile(r"\{[\s\S]*?\}", re.DOTALL)


def _extract_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text*.

    Raises:
        ValueError: If no JSON object is found.
        json.JSONDecodeError: If the extracted block is not valid JSON.
    """
    match = _JSON_PATTERN_BB.search(text)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group())


# ============================================================================
# 8. Output validation & defaults (PRD 4.5.5)
# ============================================================================


def _validate_bull_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and fill defaults for Bull researcher output."""
    defaults: dict[str, Any] = {
        "bull_score": 50,
        "momentum_signal": "neutral",
        "bull_thesis": "",
        "bull_catalysts": [],
        "key_evidence": [],
        "risk_acknowledged": [],
    }
    result: dict[str, Any] = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        if key == "bull_score":
            if not isinstance(val, (int, float)):
                val = default
            else:
                val = max(0, min(100, int(val)))
        elif isinstance(default, list) and not isinstance(val, list):
            val = default
        elif isinstance(default, str) and not isinstance(val, str):
            val = str(val) if val is not None else default
        result[key] = val
    return result


def _validate_bear_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and fill defaults for Bear researcher output."""
    defaults: dict[str, Any] = {
        "bear_score": 50,
        "bear_thesis": "",
        "bear_catalysts": [],
        "key_risks": [],
        "bull_rebuttals": [],
    }
    result: dict[str, Any] = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        if key == "bear_score":
            if not isinstance(val, (int, float)):
                val = default
            else:
                val = max(0, min(100, int(val)))
        elif isinstance(default, list) and not isinstance(val, list):
            val = default
        elif isinstance(default, str) and not isinstance(val, str):
            val = str(val) if val is not None else default
        result[key] = val
    return result


def _validate_pm_result(
    parsed: dict[str, Any],
    bull_score: float = 50.0,
    bear_score: float = 50.0,
    phase1_pass: bool = True,
) -> dict[str, Any]:
    """Validate and fill defaults for PM output (PRD 4.5.5).

    Args:
        parsed: Raw PM JSON output.
        bull_score: Bull researcher score (for triple-condition check).
        bear_score: Bear researcher score (for triple-condition check).
        phase1_pass: Whether Phase 1 hard filters passed (from context,
            *not* from PM JSON).
    """
    validation_errors: list[str] = []

    # score_correction: clamp to [0.90, 1.05]
    score_correction: float = 1.00
    raw_correction = parsed.get("score_correction", 1.00)
    if isinstance(raw_correction, (int, float)):
        score_correction = max(
            SCORE_CORRECTION_MIN, min(SCORE_CORRECTION_MAX, float(raw_correction))
        )
        if float(raw_correction) != score_correction:
            validation_errors.append(
                "score_correction {} clamped to {}".format(raw_correction, score_correction)
            )
    else:
        validation_errors.append(
            "score_correction missing or wrong type, defaulting to 1.00"
        )

    # breakout_probability: clamp to [0, 100]
    breakout_prob: float = 50.0
    raw_bp = parsed.get("breakout_probability", 50)
    if isinstance(raw_bp, (int, float)):
        breakout_prob = max(0.0, min(100.0, float(raw_bp)))
    else:
        validation_errors.append("breakout_probability invalid, defaulting to 50")

    # final_rating: validate enum
    raw_rating = parsed.get("final_rating", "Hold")
    valid_ratings = {r.value for r in FinalRating}
    final_rating: str = (
        raw_rating if raw_rating in valid_ratings else "Hold"
    )
    if raw_rating not in valid_ratings:
        validation_errors.append(
            "final_rating {!r} invalid, defaulting to Hold".format(raw_rating)
        )

    # catalyst_consistency: clamp to [0, 100]
    catalyst_consistency: float = 50.0
    raw_cc = parsed.get("catalyst_consistency", 50)
    if isinstance(raw_cc, (int, float)):
        catalyst_consistency = max(0.0, min(100.0, float(raw_cc)))
    else:
        validation_errors.append("catalyst_consistency invalid, defaulting to 50")

    # Check triple condition for 1.05 via unified BreakoutAssessment method.
    # Build a provisional assessment so is_score_correction_max_valid() is
    # the single source of truth for the triple-condition predicate.
    if score_correction >= 1.05:
        rating_for_provisional = (
            FinalRating(final_rating) if final_rating in valid_ratings
            else FinalRating.HOLD
        )
        provisional = BreakoutAssessment(
            ticker="_validate",
            final_rating=rating_for_provisional,
            breakout_probability=breakout_prob,
            bull_score=bull_score,
            bear_score=bear_score,
            score_correction=score_correction,
            catalyst_consistency=catalyst_consistency,
        )
        if not provisional.is_score_correction_max_valid():
            validation_errors.append(
                "score_correction=1.05 rejected: triple condition not met "
                "(bull={}, bear={}, catalyst={})".format(
                    bull_score, bear_score, catalyst_consistency
                )
            )
            score_correction = 1.04

    # Phase 1 override (explicit parameter, not from PM JSON)
    if not phase1_pass:
        final_rating = "Avoid"
        breakout_prob = min(breakout_prob, 30.0)

    # risk_tags: filter to canonical set
    raw_tags: list[str] = parsed.get("risk_tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    risk_tags: list[str] = [t for t in raw_tags if t in ALLOWED_RISK_TAGS]
    removed = len(raw_tags) - len(risk_tags)
    if removed:
        validation_errors.append(
            "Filtered {} illegal risk tags".format(removed)
        )

    # data_conflict_detected
    data_conflict: bool = bool(parsed.get("data_conflict_detected", False))

    return {
        "final_rating": final_rating,
        "breakout_probability": breakout_prob,
        "score_correction": score_correction,
        "data_conflict_detected": data_conflict,
        "catalyst_consistency": catalyst_consistency,
        "risk_tags": risk_tags,
        "risk_flags": parsed.get("risk_flags", [])
        if isinstance(parsed.get("risk_flags"), list)
        else [],
        "pm_verdict": parsed.get("pm_verdict", "")
        if isinstance(parsed.get("pm_verdict"), str)
        else "",
        "validation_errors": validation_errors,
    }


# ============================================================================
# 9. Single-symbol pipeline runner
# ============================================================================


def run_bull_bear_pm(
    context: BullBearContext,
    invoker: Invoker,
    max_output_tokens: int = 800,
    retry_on_json_failure: bool = True,
    *,
    cost_tracker: CostTracker | None = None,
) -> BreakoutAssessment:
    """Run the full Bull/Bear/PM pipeline for a single symbol.

    Pipeline:
      1. Run Bull and Bear researchers (conceptually parallel; sequential in MVP).
      2. Merge results into context, run Portfolio Manager.
      3. Validate and coercion all outputs into a :class:`BreakoutAssessment`.

    Args:
        context: Context with ticker, price, factor summaries, etc.
        invoker: LLM callable
            ``(system_prompt, max_output_tokens) -> tuple[str, int, int]``
            returning (response_text, input_tokens, output_tokens).
        max_output_tokens: Token budget for each individual LLM call.
        retry_on_json_failure: If True, retry once on JSON parse failure.
        cost_tracker: Optional :class:`~alphascreener.cost.tracker.CostTracker`
            for recording LLM call costs.

    Returns:
        A validated :class:`BreakoutAssessment` instance.
    """
    total_tokens = 0
    input_tokens_total = 0
    validation_errors: list[str] = []

    # --- Bull Researcher ---
    bull_raw: dict[str, Any] = {}
    try:
        bull_prompt = BULL_PROMPT.format(context, side="bull")
        bull_response, in_tok, out_tok = invoker(bull_prompt, max_output_tokens)
        total_tokens += in_tok + out_tok
        input_tokens_total += in_tok
        if cost_tracker is not None:
            try:
                cost_tracker.record_call("bull", in_tok, out_tok)
            except Exception:
                _logger.warning(
                    "Failed to record bull call cost for %s",
                    context.ticker, exc_info=True,
                )
    except Exception as exc:
        _logger.error("Bull researcher invocation failed for %s: %s", context.ticker, exc)
        bull_raw = _validate_bull_result({})
        validation_errors.append(f"Bull invocation error: {exc}")
    else:
        try:
            bull_raw = _extract_json_response(bull_response)
            bull_raw = _validate_bull_result(bull_raw)
        except (json.JSONDecodeError, ValueError) as exc:
            if retry_on_json_failure:
                _logger.warning(
                    "Bull JSON parse failed for %s, retrying once: %s",
                    context.ticker, exc,
                )
                try:
                    retry_prompt = (
                        f"{bull_prompt}\n\n上一轮输出不是合法 JSON。"
                        f"请严格输出合法 JSON，只输出 JSON，不要输出其他内容。"
                    )
                    bull_response, in_tok, out_tok = invoker(retry_prompt, max_output_tokens)
                    total_tokens += in_tok + out_tok
                    input_tokens_total += in_tok
                    if cost_tracker is not None:
                        try:
                            cost_tracker.record_call("bull", in_tok, out_tok)
                        except Exception:
                            _logger.warning(
                                "Failed to record bull retry call cost for %s",
                                context.ticker, exc_info=True,
                            )
                    bull_raw = _extract_json_response(bull_response)
                    bull_raw = _validate_bull_result(bull_raw)
                except Exception as exc2:
                    _logger.error(
                        "Bull JSON retry also failed for %s: %s",
                        context.ticker, exc2,
                    )
                    bull_raw = _validate_bull_result({})
                    validation_errors.append(f"Bull JSON parse failed: {exc}; retry: {exc2}")
            else:
                bull_raw = _validate_bull_result({})
                validation_errors.append(f"Bull JSON parse error: {exc}")

    # --- Bear Researcher ---
    bear_raw: dict[str, Any] = {}
    try:
        bear_prompt = BEAR_PROMPT.format(context, side="bear")
        bear_response, in_tok, out_tok = invoker(bear_prompt, max_output_tokens)
        total_tokens += in_tok + out_tok
        input_tokens_total += in_tok
        if cost_tracker is not None:
            try:
                cost_tracker.record_call("bear", in_tok, out_tok)
            except Exception:
                _logger.warning(
                    "Failed to record bear call cost for %s",
                    context.ticker, exc_info=True,
                )
    except Exception as exc:
        _logger.error("Bear researcher invocation failed for %s: %s", context.ticker, exc)
        bear_raw = _validate_bear_result({})
        validation_errors.append(f"Bear invocation error: {exc}")
    else:
        try:
            bear_raw = _extract_json_response(bear_response)
            bear_raw = _validate_bear_result(bear_raw)
        except (json.JSONDecodeError, ValueError) as exc:
            if retry_on_json_failure:
                _logger.warning(
                    "Bear JSON parse failed for %s, retrying once: %s",
                    context.ticker, exc,
                )
                try:
                    retry_prompt = (
                        f"{bear_prompt}\n\n上一轮输出不是合法 JSON。"
                        f"请严格输出合法 JSON，只输出 JSON，不要输出其他内容。"
                    )
                    bear_response, in_tok, out_tok = invoker(retry_prompt, max_output_tokens)
                    total_tokens += in_tok + out_tok
                    input_tokens_total += in_tok
                    if cost_tracker is not None:
                        try:
                            cost_tracker.record_call("bear", in_tok, out_tok)
                        except Exception:
                            _logger.warning(
                                "Failed to record bear retry call cost for %s",
                                context.ticker, exc_info=True,
                            )
                    bear_raw = _extract_json_response(bear_response)
                    bear_raw = _validate_bear_result(bear_raw)
                except Exception as exc2:
                    _logger.error(
                        "Bear JSON retry also failed for %s: %s",
                        context.ticker, exc2,
                    )
                    bear_raw = _validate_bear_result({})
                    validation_errors.append(f"Bear JSON parse failed: {exc}; retry: {exc2}")
            else:
                bear_raw = _validate_bear_result({})
                validation_errors.append(f"Bear JSON parse error: {exc}")

    # Update context with bull/bear results for PM
    context.bull_result = bull_raw
    context.bear_result = bear_raw

    # --- Portfolio Manager ---
    pm_raw: dict[str, Any] = {}
    try:
        pm_prompt = PM_PROMPT.format(context)
        pm_response, in_tok, out_tok = invoker(pm_prompt, max_output_tokens)
        total_tokens += in_tok + out_tok
        input_tokens_total += in_tok
        if cost_tracker is not None:
            try:
                cost_tracker.record_call("pm", in_tok, out_tok)
            except Exception:
                _logger.warning(
                    "Failed to record PM call cost for %s",
                    context.ticker, exc_info=True,
                )
    except Exception as exc:
        _logger.error("PM invocation failed for %s: %s", context.ticker, exc)
        pm_raw = _validate_pm_result(
            {},
            bull_score=bull_raw.get("bull_score", 50.0),
            bear_score=bear_raw.get("bear_score", 50.0),
            phase1_pass=context.phase1_pass,
        )
        validation_errors.append(f"PM invocation error: {exc}")
    else:
        try:
            pm_parsed = _extract_json_response(pm_response)
            pm_raw = _validate_pm_result(
                pm_parsed,
                bull_score=bull_raw.get("bull_score", 50.0),
                bear_score=bear_raw.get("bear_score", 50.0),
                phase1_pass=context.phase1_pass,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            if retry_on_json_failure:
                _logger.warning(
                    "PM JSON parse failed for %s, retrying once: %s",
                    context.ticker, exc,
                )
                try:
                    retry_prompt = (
                        f"{pm_prompt}\n\n上一轮输出不是合法 JSON。"
                        f"请严格输出合法 JSON，只输出 JSON，不要输出其他内容。"
                    )
                    pm_response, in_tok, out_tok = invoker(retry_prompt, max_output_tokens)
                    total_tokens += in_tok + out_tok
                    input_tokens_total += in_tok
                    if cost_tracker is not None:
                        try:
                            cost_tracker.record_call("pm", in_tok, out_tok)
                        except Exception:
                            _logger.warning(
                                "Failed to record PM retry call cost for %s",
                                context.ticker, exc_info=True,
                            )
                    pm_parsed = _extract_json_response(pm_response)
                    pm_raw = _validate_pm_result(
                        pm_parsed,
                        bull_score=bull_raw.get("bull_score", 50.0),
                        bear_score=bear_raw.get("bear_score", 50.0),
                        phase1_pass=context.phase1_pass,
                    )
                except Exception as exc2:
                    _logger.error(
                        "PM JSON retry also failed for %s: %s",
                        context.ticker, exc2,
                    )
                    pm_raw = _validate_pm_result(
                        {},
                        bull_score=bull_raw.get("bull_score", 50.0),
                        bear_score=bear_raw.get("bear_score", 50.0),
                        phase1_pass=context.phase1_pass,
                    )
                    validation_errors.append(f"PM JSON parse failed: {exc}; retry: {exc2}")
            else:
                pm_raw = _validate_pm_result(
                    {},
                    bull_score=bull_raw.get("bull_score", 50.0),
                    bear_score=bear_raw.get("bear_score", 50.0),
                    phase1_pass=context.phase1_pass,
                )
                validation_errors.append(f"PM JSON parse error: {exc}")

    # --- Merge validation errors ---
    validation_errors.extend(pm_raw.pop("validation_errors", []))

    # --- Catalyst events: merge from Bull + Bear ---
    catalyst_events: list[str] = []
    for cat in bull_raw.get("bull_catalysts", []):
        if isinstance(cat, str) and cat not in catalyst_events:
            catalyst_events.append(cat)
    for cat in bear_raw.get("bear_catalysts", []):
        if isinstance(cat, str) and cat not in catalyst_events:
            catalyst_events.append(cat)

    # --- Build BreakoutAssessment ---
    assessment = BreakoutAssessment(
        ticker=context.ticker,
        final_rating=FinalRating(pm_raw.get("final_rating", "Hold")),
        breakout_probability=pm_raw.get("breakout_probability", 50.0),
        bull_score=float(bull_raw.get("bull_score", 50)),
        bear_score=float(bear_raw.get("bear_score", 50)),
        score_correction=float(pm_raw.get("score_correction", 1.00)),
        data_conflict_detected=bool(pm_raw.get("data_conflict_detected", False)),
        catalyst_consistency=float(pm_raw.get("catalyst_consistency", 50.0)),
        risk_tags=pm_raw.get("risk_tags", []),
        risk_flags=pm_raw.get("risk_flags", []),
        catalyst_events=catalyst_events,
        bull_thesis=str(bull_raw.get("bull_thesis", "")),
        bear_thesis=str(bear_raw.get("bear_thesis", "")),
        pm_verdict=str(pm_raw.get("pm_verdict", "")),
        phase1_pass=context.phase1_pass,
        input_tokens_total=input_tokens_total,
        validation_errors=validation_errors,
    )

    return assessment


# ============================================================================
# 10. Batch orchestrator
# ============================================================================

DEFAULT_BATCH_SIZE: int = 3
DEFAULT_N_BATCHES: int | None = None


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = DEFAULT_BATCH_SIZE
    n_batches: int | None = DEFAULT_N_BATCHES
    max_output_tokens: int = 800
    retry_on_json_failure: bool = True
    cost_tracker: CostTracker | None = None
    """Optional cost tracker for recording LLM call costs per batch."""


def _chunk_contexts(
    contexts: list[BullBearContext], batch_size: int
) -> list[list[BullBearContext]]:
    """Split contexts into fixed-size batches."""
    batches: list[list[BullBearContext]] = []
    for i in range(0, len(contexts), batch_size):
        batches.append(contexts[i : i + batch_size])
    return batches


def run_pipeline_batch(
    contexts: list[BullBearContext],
    invoker: Invoker,
    config: BatchConfig | None = None,
) -> list[BreakoutAssessment]:
    """Run the Bull/Bear/PM pipeline over a batch of symbols.

    Execution model (PRD 4.5): batch-serial, within each batch:
      (Bull || Bear) conceptually parallel → PM sequential.

    In MVP, all calls are sequential (parallelism deferred to future
    optimisation).

    If ``config.cost_tracker`` is set, each LLM call is recorded and the
    circuit breaker is checked before every batch.  A tripped circuit
    (L4) will cause :class:`RuntimeError`; L2+ will skip fine screening
    for remaining symbols.

    Args:
        contexts: List of BullBearContext, one per symbol.
        invoker: LLM callable.
        config: Batch configuration.

    Returns:
        List of validated :class:`BreakoutAssessment` in the same order
        as the input contexts.

    Raises:
        RuntimeError: If the circuit breaker is at L4 (all LLM calls blocked).
    """
    cfg = config or BatchConfig()
    cost_tracker = cfg.cost_tracker

    # Check circuit breaker before processing
    fine_screening_paused = False
    if cost_tracker is not None:
        status = cost_tracker.check_circuit()
        _logger.info(
            "Circuit status before pipeline: level=%s daily=$%.4f monthly=$%.4f",
            status.label, status.daily_cost, status.monthly_cost,
        )
        if status.is_blocked():
            _logger.error("L4 BREAKER: all LLM calls stopped")
            raise RuntimeError(status.message)
        if not status.fine_screening_allowed():
            _logger.warning("L2+ DEGRADE: fine screening paused, coarse only")
            fine_screening_paused = True

    if fine_screening_paused:
        _logger.info("Fine screening skipped due to circuit breaker — returning empty results")
        return []

    # Limit to n_batches (None = no limit)
    batches = _chunk_contexts(contexts, cfg.batch_size)
    if cfg.n_batches is not None:
        batches = batches[: cfg.n_batches]

    results: list[BreakoutAssessment] = []
    total_symbols = 0

    for batch_idx, batch in enumerate(batches):
        # Re-check circuit before each batch (cost may have accumulated)
        if cost_tracker is not None:
            status = cost_tracker.check_circuit()
            if status.is_blocked():
                _logger.error("L4 BREAKER tripped during batch %d — stopping", batch_idx + 1)
                break
            if not status.fine_screening_allowed():
                _logger.warning(
                    "L2+ DEGRADE tripped during batch %d — stopping fine screening", batch_idx + 1
                )
                break

        _logger.info(
            "Processing batch %d/%d: %d symbols",
            batch_idx + 1,
            len(batches),
            len(batch),
        )

        for ctx in batch:
            assessment = run_bull_bear_pm(
                ctx,
                invoker,
                max_output_tokens=cfg.max_output_tokens,
                retry_on_json_failure=cfg.retry_on_json_failure,
                cost_tracker=cost_tracker,
            )
            results.append(assessment)
            total_symbols += 1

    _logger.info(
        "Pipeline complete: %d assessments for %d symbols",
        len(results),
        total_symbols,
    )
    return results


# ============================================================================
# 11. Convenience: build context from factor data
# ============================================================================


def build_bull_bear_context(
    ticker: str,
    price: float,
    *,
    mom_5d: float = 0.0,
    factor_scores_summary: str = "",
    news_summary: str = "",
    technical_pattern: str = "",
    phase1_pass: bool = True,
    factor_vector: list[float] | None = None,
) -> BullBearContext:
    """Build a :class:`BullBearContext` from factor and filter data.

    Args:
        ticker: Stock symbol.
        price: Current price.
        mom_5d: 5-day momentum.
        factor_scores_summary: Human-readable factor snapshot.
        news_summary: Aggregated news text.
        technical_pattern: Technical pattern description.
        phase1_pass: Whether Phase 1 hard filters were passed.
        factor_vector: Factor values as a numeric vector.

    Returns:
        A populated context ready for the pipeline.
    """
    return BullBearContext(
        ticker=ticker,
        price=price,
        mom_5d=mom_5d,
        factor_scores_summary=factor_scores_summary,
        news_summary=news_summary,
        technical_pattern=technical_pattern,
        phase1_pass=phase1_pass,
        factor_vector=factor_vector or [],
    )
