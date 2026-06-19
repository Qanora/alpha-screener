"""4 Analyst LLM prompt templates with context variable injection.

Issue #97: Analyst prompts + invocation.
Reference: PRD 4.2.1 / 4.5.

Each template accepts a context dict of variables (ticker, price,
factor_scores_summary, news_summary, technical_pattern, etc.) and
renders a complete prompt string suitable for an LLM chat completion.

Token budgets: ≤ 2000 input / ≤ 800 output per analyst (enforced
at the orchestrator level via tiktoken estimation).
"""

from __future__ import annotations

from typing import Any

import tiktoken

# ---------------------------------------------------------------------------
# Token budget constants
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS: int = 2000
MAX_OUTPUT_TOKENS: int = 800
_ENCODER = tiktoken.get_encoding("cl100k_base")


def truncate_context(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens* tokens.

    Keeps the first *max_tokens* tokens' worth of characters.
    """
    if not text:
        return text
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _ENCODER.decode(tokens[:max_tokens])
    return truncated + "..."


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using cl100k_base encoding."""
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


class AnalystContext:
    """Context variables injected into analyst prompt templates.

    Not all fields are used by every analyst; each template documents
    which subset it requires.
    """

    def __init__(
        self,
        ticker: str = "",
        price: float = 0.0,
        mom_5d: float = 0.0,
        factor_scores_summary: str = "",
        news_summary: str = "",
        technical_pattern: str = "",
        false_breakout_rate: int = 50,
        factor_vector: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.ticker = ticker
        self.price = price
        self.mom_5d = mom_5d
        self.factor_scores_summary = factor_scores_summary
        self.news_summary = news_summary
        self.technical_pattern = technical_pattern
        self.false_breakout_rate = false_breakout_rate
        self.factor_vector = factor_vector or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "mom_5d": self.mom_5d,
            "factor_scores_summary": self.factor_scores_summary,
            "news_summary": self.news_summary,
            "technical_pattern": self.technical_pattern,
            "false_breakout_rate": self.false_breakout_rate,
            "factor_vector": self.factor_vector,
        }


# ---------------------------------------------------------------------------
# Prompt template base
# ---------------------------------------------------------------------------


class AnalystPromptTemplate:
    """Base class for analyst prompt templates with context injection."""

    analyst_name: str = "Analyst"
    system_message: str = ""
    context_vars: tuple[str, ...] = ()
    task_description: str = ""
    output_format_desc: str = ""

    def format(self, context: AnalystContext) -> str:
        """Inject context variables and return the full system prompt.

        Args:
            context: Run-time context values (ticker, price, etc.).

        Returns:
            Complete prompt string ready for the LLM system message.
        """
        ctx = context.to_dict()
        parts: list[str] = []

        # System (role definition)
        parts.append(self.system_message)

        # Context section
        parts.append("\n## Context (系统注入)")
        for var in self.context_vars:
            value = ctx.get(var, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            parts.append(f"- {var}: {value}")

        # Task
        parts.append("\n## Task")
        parts.append(self.task_description)

        # Output format
        parts.append("\n## Output Format")
        parts.append(self.output_format_desc)

        prompt = "\n".join(parts)
        return prompt


# ---------------------------------------------------------------------------
# 1. Market Analyst — 技术面 + 量价分析 → 技术形态报告
# ---------------------------------------------------------------------------


MARKET_ANALYST_SYSTEM = """## Role
你是一位技术分析专家，专注于美股短期技术形态识别。
你的核心职责是基于技术指标和量价数据生成结构化的技术形态报告，
识别支撑/阻力、突破信号、量价背离等可操作的技术信号。"""

MARKET_ANALYST_CONTEXT_VARS = (
    "ticker",
    "price",
    "mom_5d",
    "technical_pattern",
    "factor_scores_summary",
)

MARKET_ANALYST_TASK = """基于以上信息，从以下维度完成技术分析:
1. **趋势判断**: 识别当前处于上升/下降/震荡趋势，引用均线和 MACD 证据
2. **关键价位**: 标注近 20 日支撑位和阻力位
3. **量价验证**: 判断成交量是否支撑当前价格方向（放量突破/缩量整理/量价背离）
4. **技术形态**: 识别是否出现双底/头肩/楔形/布林带收窄等经典形态
5. **风险提示**: 指出可能存在技术噪音或假信号的区域

输出一份简洁的技术形态报告，每个结论附一个支撑指标信号。"""

MARKET_ANALYST_OUTPUT_FMT = """请输出以下 JSON 结构（字段必须完整，report 用纯文本非 markdown）:
{
  "trend": "bullish" | "bearish" | "neutral",
  "support_level": float,
  "resistance_level": float,
  "volume_confirms_trend": true | false,
  "pattern_detected": (
    "double_bottom" | "head_and_shoulders" | "wedge"
    | "bollinger_squeeze" | "none"
  ),
  "technical_report": "<200字符的技术形态摘要>",
  "key_signals": ["信号1", "信号2", "信号3"]
}"""


class MarketAnalystPrompt(AnalystPromptTemplate):
    """Market Analyst: 技术面 + 量价分析 → 技术形态报告."""

    analyst_name = "Market Analyst"
    system_message = MARKET_ANALYST_SYSTEM
    context_vars = MARKET_ANALYST_CONTEXT_VARS
    task_description = MARKET_ANALYST_TASK
    output_format_desc = MARKET_ANALYST_OUTPUT_FMT


# ---------------------------------------------------------------------------
# 2. News Analyst — 近期新闻事件 → 事件催化剂列表
# ---------------------------------------------------------------------------


NEWS_ANALYST_SYSTEM = """## Role
你是一位专注美股市场的信息分析师，负责从近期新闻中提取与标的相关的
事件催化剂，评估其对短期（T+7）股价方向的影响强度和方向。"""

NEWS_ANALYST_CONTEXT_VARS = (
    "ticker",
    "price",
    "news_summary",
    "factor_scores_summary",
)

NEWS_ANALYST_TASK = """基于以上信息，完成事件催化剂分析:
1. **催化剂事件识别**: 从新闻中提取可能与标的相关的催化剂事件
2. **方向判断**: 每个事件对股价的方向影响（正/负/中性）
3. **强度评估**: 用 low/medium/high 三档评估事件对 T+7 窗口的冲击力
4. **时效性**: 判断事件在 T+7 窗口内是否有效（fresh/stale/passed）
5. **风险事件**: 如有潜在利空（监管/诉讼/竞争），单独列出

注意: 若 news_summary 为空，应声明"无可分析新闻"但仍输出合法 JSON。"""

NEWS_ANALYST_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "catalyst_count": int,
  "catalysts": [
    {
      "event": "<事件描述, ≤50字>",
      "direction": "positive" | "negative" | "neutral",
      "strength": "low" | "medium" | "high",
      "timeliness": "fresh" | "stale" | "passed"
    }
  ],
  "risk_events": ["风险事件1", ...],
  "news_report": "<250字符的新闻分析摘要>"
}"""


class NewsAnalystPrompt(AnalystPromptTemplate):
    """News Analyst: 近期新闻事件 → 事件催化剂列表."""

    analyst_name = "News Analyst"
    system_message = NEWS_ANALYST_SYSTEM
    context_vars = NEWS_ANALYST_CONTEXT_VARS
    task_description = NEWS_ANALYST_TASK
    output_format_desc = NEWS_ANALYST_OUTPUT_FMT


# ---------------------------------------------------------------------------
# 3. Fundamentals Analyst — 基本面变化 → 业绩/估值触发点
# ---------------------------------------------------------------------------


FUNDAMENTALS_ANALYST_SYSTEM = """## Role
你是一位专注基本面分析的研究员，评估标的近期基本面变化是否构成
短期（T+7）业绩驱动或估值重估触发点。"""

FUNDAMENTALS_ANALYST_CONTEXT_VARS = (
    "ticker",
    "price",
    "factor_scores_summary",
    "news_summary",
)

FUNDAMENTALS_ANALYST_TASK = """基于以上信息，完成基本面变化评估:
1. **业绩触发点**: 识别近期（过去 30 天 / 未来 7 天）的财报、业绩预告或指引变更
2. **估值信号**: PE/PS 与行业均值对比，当前估值是否有重估空间
3. **内部人信号**: 近 60 日内部人买入/卖出是否有异常
4. **基本面风险**: 财务造假迹象、负债率突变、大股东减持等

如缺少某些维度的数据，在对应字段标注 "data_unavailable"。"""

FUNDAMENTALS_ANALYST_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "earnings_trigger": "positive_outlook" | "negative_outlook" | "neutral" | "data_unavailable",
  "valuation_signal": "undervalued" | "fair" | "overvalued" | "data_unavailable",
  "insider_signal": "net_buy" | "net_sell" | "neutral" | "data_unavailable",
  "fundamentals_report": "<200字符的基本面摘要>",
  "risk_flags": ["风险标记1", ...]
}"""


class FundamentalsAnalystPrompt(AnalystPromptTemplate):
    """Fundamentals Analyst: 基本面变化 → 业绩/估值触发点."""

    analyst_name = "Fundamentals Analyst"
    system_message = FUNDAMENTALS_ANALYST_SYSTEM
    context_vars = FUNDAMENTALS_ANALYST_CONTEXT_VARS
    task_description = FUNDAMENTALS_ANALYST_TASK
    output_format_desc = FUNDAMENTALS_ANALYST_OUTPUT_FMT


# ---------------------------------------------------------------------------
# 4. Breakout Analyst — 爆发形态识别 → 历史相似案例检索
# ---------------------------------------------------------------------------


BREAKOUT_ANALYST_SYSTEM = """## Role
你是一位专注爆发形态识别的分析师。你的任务是判断给定标的的因子
向量是否匹配历史爆发案例的形态特征，检索历史相似标的，评估当前
形态与历史成功爆发案例的匹配度。"""

BREAKOUT_ANALYST_CONTEXT_VARS = (
    "ticker",
    "price",
    "mom_5d",
    "factor_scores_summary",
    "technical_pattern",
    "false_breakout_rate",
)

BREAKOUT_ANALYST_TASK = """基于以上信息，完成爆发形态分析:
1. **形态匹配**: 当前因子向量是否接近历史正样本的典型特征
2. **相似案例**: 引用 similar_cases 中的历史案例（由系统 faiss 检索提供）
3. **假突破风险**: 结合 false_breakout_rate 判断被假突破误导的概率
4. **综合爆发概率**: 综合技术形态、因子匹配度和历史案例，给出 0-100 爆发概率

注意: similar_cases 为空时说明案例库尚未积累足够数据（MVP 阶段正常现象），
此时仅依赖技术形态和因子评分做判断，并在 report 中注明"案例库空"。"""

BREAKOUT_ANALYST_OUTPUT_FMT = """请输出以下 JSON 结构:
{
  "breakout_score": int (0-100),
  "pattern_match_confidence": int (0-100),
  "false_breakout_risk": int (0-100),
  "key_drivers": ["驱动因子1", ...],
  "similar_cases": ["案例1", ...],
  "breakout_report": "<200字符的爆发分析摘要>"
}"""


class BreakoutAnalystPrompt(AnalystPromptTemplate):
    """Breakout Analyst: 爆发形态专项识别 → 历史相似案例检索."""

    analyst_name = "Breakout Analyst"
    system_message = BREAKOUT_ANALYST_SYSTEM
    context_vars = BREAKOUT_ANALYST_CONTEXT_VARS
    task_description = BREAKOUT_ANALYST_TASK
    output_format_desc = BREAKOUT_ANALYST_OUTPUT_FMT


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

ANALYST_PROMPTS: dict[str, AnalystPromptTemplate] = {
    "market": MarketAnalystPrompt(),
    "news": NewsAnalystPrompt(),
    "fundamentals": FundamentalsAnalystPrompt(),
    "breakout": BreakoutAnalystPrompt(),
}


def get_analyst_prompt(analyst_type: str) -> AnalystPromptTemplate:
    """Return the prompt template for *analyst_type*.

    Args:
        analyst_type: One of ``"market"``, ``"news"``, ``"fundamentals"``, ``"breakout"``.

    Returns:
        An :class:`AnalystPromptTemplate` instance.

    Raises:
        ValueError: If *analyst_type* is not recognized.
    """
    template = ANALYST_PROMPTS.get(analyst_type)
    if template is None:
        raise ValueError(
            f"Unknown analyst type: {analyst_type!r}. Valid: {sorted(ANALYST_PROMPTS)}"
        )
    return template


def format_analyst_prompt(analyst_type: str, context: AnalystContext) -> str:
    """Format a prompt for *analyst_type* with the given *context*.

    Convenience wrapper combining :func:`get_analyst_prompt` and
    :meth:`AnalystPromptTemplate.format`.
    """
    template = get_analyst_prompt(analyst_type)
    return template.format(context)
