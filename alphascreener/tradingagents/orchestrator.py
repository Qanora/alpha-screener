"""LLM invocation orchestrator for the 4-analyst pipeline.

Issue #97: Analyst prompts + invocation.
Reference: PRD 4.2.1 / 4.5.

Orchestrates the four analyst prompts (Market, News, Fundamentals, Breakout)
with context variable injection, token budget enforcement, and LLM invocation.

Token budget: ≤ 2000 input / ≤ 800 output per analyst.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from alphascreener.logging import get_logger
from alphascreener.tradingagents.breakout_retriever import (
    BreakoutCaseRetriever,
)
from alphascreener.tradingagents.prompts import (
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    AnalystContext,
    estimate_tokens,
    format_analyst_prompt,
    truncate_context,
)

if TYPE_CHECKING:
    from alphascreener.config import Settings

_logger = get_logger("screening")


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

# An invoker is a callable that takes a system prompt and returns the
# LLM text response.
Invoker = Callable[[str, int], str]
# Signature: (system_prompt: str, max_output_tokens: int) -> str


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_context(
    ticker: str,
    price: float,
    factor_scores_summary: str = "",
    news_summary: str = "",
    technical_pattern: str = "",
    mom_5d: float = 0.0,
    false_breakout_rate: int = 50,
    factor_vector: list[float] | None = None,
    **kwargs: Any,
) -> AnalystContext:
    """Build a fully-populated :class:`AnalystContext`.

    All fields default to reasonable empty/neutral values so callers
    can supply only the data they have available.
    """
    return AnalystContext(
        ticker=ticker,
        price=price,
        mom_5d=mom_5d,
        factor_scores_summary=factor_scores_summary,
        news_summary=news_summary,
        technical_pattern=technical_pattern,
        false_breakout_rate=false_breakout_rate,
        factor_vector=factor_vector,
    )


# ---------------------------------------------------------------------------
# Token budget checking
# ---------------------------------------------------------------------------


def check_token_budget(prompt: str) -> tuple[bool, int]:
    """Check whether *prompt* fits within the input token budget.

    Returns:
        ``(ok, token_count)`` — *ok* is ``True`` when token_count ≤ max.
    """
    tokens = estimate_tokens(prompt)
    return tokens <= MAX_INPUT_TOKENS, tokens




def clamp_context_for_budget(
    context: AnalystContext, max_extra_tokens: int = 500,
) -> AnalystContext:
    """Truncate long text fields in a *copy* of *context* to stay under token budget.

    This is a coarse prep step; the final budget check is done at prompt-
    format time.  The original *context* is not modified.
    """
    ctx = copy.copy(context)
    cap = max_extra_tokens
    ctx.factor_scores_summary = truncate_context(ctx.factor_scores_summary, cap)
    ctx.news_summary = truncate_context(ctx.news_summary, cap)
    ctx.technical_pattern = truncate_context(ctx.technical_pattern, cap // 2)
    return ctx


# ---------------------------------------------------------------------------
# Run single analyst
# ---------------------------------------------------------------------------


def run_analyst(
    analyst_type: str,
    context: AnalystContext,
    invoker: Invoker,
) -> dict[str, Any]:
    """Run a single analyst: inject context, call LLM, enforce budget.

    Args:
        analyst_type: ``"market"`` | ``"news"`` | ``"fundamentals"`` | ``"breakout"``.
        context: Context variables for prompt injection.
        invoker: A callable ``(system_prompt, max_output_tokens) -> str``.

    Returns:
        Dict with keys ``analyst_type``, ``prompt``, ``response``,
        ``input_tokens``, ``budget_ok``, ``parsed_json`` (or ``None``),
        ``error`` (or ``None``).
    """
    _logger.debug("Running %s analyst for %s", analyst_type, context.ticker)

    # Clamp context text fields
    ctx = clamp_context_for_budget(context)

    # Format prompt
    prompt = format_analyst_prompt(analyst_type, ctx)
    input_tokens = estimate_tokens(prompt)
    budget_ok = input_tokens <= MAX_INPUT_TOKENS

    if not budget_ok:
        _logger.warning(
            "%s prompt for %s exceeds budget: %d > %d tokens",
            analyst_type,
            ctx.ticker,
            input_tokens,
            MAX_INPUT_TOKENS,
        )
        return {
            "analyst_type": analyst_type,
            "prompt": prompt,
            "response": "",
            "input_tokens": input_tokens,
            "budget_ok": False,
            "parsed_json": None,
            "error": f"Token budget exceeded: {input_tokens} > {MAX_INPUT_TOKENS}",
        }

    # Invoke LLM with output token cap
    try:
        response = invoker(prompt, MAX_OUTPUT_TOKENS)
    except TimeoutError as exc:
        _logger.error(
            "%s analyst timed out for %s: %s", analyst_type, ctx.ticker, exc
        )
        return {
            "analyst_type": analyst_type,
            "prompt": prompt,
            "response": "",
            "input_tokens": input_tokens,
            "budget_ok": budget_ok,
            "parsed_json": None,
            "error": f"LLM invocation timed out: {exc}",
        }
    except Exception as exc:
        _logger.error("%s analyst invocation failed: %s", analyst_type, exc)
        return {
            "analyst_type": analyst_type,
            "prompt": prompt,
            "response": "",
            "input_tokens": input_tokens,
            "budget_ok": budget_ok,
            "parsed_json": None,
            "error": str(exc),
        }

    # Attempt to parse JSON from response
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = _extract_json(response)
    except json.JSONDecodeError as exc:
        snippet = response[:200].replace("\n", " ")
        _logger.warning(
            "%s analyst returned invalid JSON: %s (response preview: %s)",
            analyst_type,
            exc,
            snippet,
        )
        parse_error = f"Invalid JSON in LLM response: {exc}"
    except ValueError as exc:
        snippet = response[:200].replace("\n", " ")
        _logger.warning(
            "%s analyst returned no JSON object: %s (response preview: %s)",
            analyst_type,
            exc,
            snippet,
        )
        parse_error = f"No JSON object found in LLM response: {exc}"

    return {
        "analyst_type": analyst_type,
        "prompt": prompt,
        "response": response,
        "input_tokens": input_tokens,
        "budget_ok": budget_ok,
        "parsed_json": parsed,
        "error": parse_error,
    }


# ---------------------------------------------------------------------------
# LLM invoker factory
# ---------------------------------------------------------------------------


def build_llm_invoker(
    settings: Settings | None = None,
    provider: str = "openai",
) -> Invoker:
    """Build an LLM invoker callable from application settings.

    Reads ``openai_api_key``, ``openai_base_url``, and ``llm_model`` from
    the given :class:`~alphascreener.config.Settings` (or the default
    ``Settings()`` if none is provided).  The resulting invoker is suitable
    for use with :class:`AnalystOrchestrator`.

    Args:
        settings: Optional application settings.  When ``None``, a default
            ``Settings()`` instance is created.
        provider: LLM provider name forwarded to
            :func:`~alphascreener.tradingagents.llm_adapter.create_llm_client_safe`.

    Returns:
        A callable ``(prompt, max_output_tokens) -> str``.
    """
    if settings is None:
        from alphascreener.config import Settings as _Settings

        settings = _Settings()  # type: ignore[call-arg]

    from alphascreener.tradingagents.llm_adapter import create_llm_client_safe

    llm = create_llm_client_safe(
        provider,
        settings.llm_model,
        base_url=settings.openai_base_url or None,
        api_key=settings.openai_api_key or None,
    ).get_llm()

    def invoker(prompt: str, max_out_tok: int) -> str:
        from langchain_core.messages import SystemMessage

        msg = llm.invoke(
            [SystemMessage(content=prompt)],
            max_tokens=max_out_tok,
        )
        return msg.content

    return invoker


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AnalystOrchestrator:
    """Orchestrate the full 4-analyst pipeline with token budget enforcement.

    Usage::

        from alphascreener.config import Settings
        from alphascreener.tradingagents.orchestrator import build_llm_invoker

        settings = Settings()
        invoker = build_llm_invoker(settings=settings)

        orch = AnalystOrchestrator(invoker=invoker)
        ctx = build_context("AAPL", 185.50, factor_scores_summary="...")
        results = orch.run(ctx)
        # results["market"]["parsed_json"] -> dict
        # results["breakout"]["similar_cases"] -> list
    """

    def __init__(
        self,
        invoker: Invoker,
        retriever: BreakoutCaseRetriever | None = None,
    ) -> None:
        self._invoker = invoker
        self._retriever = retriever or BreakoutCaseRetriever()

    def run(self, context: AnalystContext) -> dict[str, Any]:
        """Run all four analysts and return aggregated results.

        Args:
            context: Context variables for prompt injection.

        Returns:
            Dict with keys ``"market"``, ``"news"``, ``"fundamentals"``,
            ``"breakout"``, ``"similar_cases"``, ``"input_tokens_total"``.
            Each analyst sub-dict contains the full :func:`run_analyst` output.
        """
        _logger.info("Orchestrator: starting 4-analyst run for %s", context.ticker)

        ctx = copy.copy(context)
        results: dict[str, Any] = {}
        total_tokens = 0

        # Run all four analysts sequentially (MVP budget-conscious pattern)
        for analyst_type in ("market", "news", "fundamentals", "breakout"):
            # For breakout analyst, pre-enrich context with similar cases
            if analyst_type == "breakout":
                similar = self._retriever.search(
                    factor_vector=ctx.factor_vector or [],
                )
                results["similar_cases"] = similar
                if similar:
                    # Append similar cases hint to factor_scores_summary
                    cases_hint = "相似案例: " + ", ".join(
                        f"{c['ticker']}({c['date']}, sim={c['similarity']})"
                        for c in similar[:5]
                    )
                    ctx.factor_scores_summary = (
                        f"{ctx.factor_scores_summary}\n{cases_hint}".strip()
                    )

            result = run_analyst(analyst_type, ctx, self._invoker)
            results[analyst_type] = result
            total_tokens += result.get("input_tokens", 0)

        results["input_tokens_total"] = total_tokens

        # Token budget summary
        overs = [
            a for a, r in results.items()
            if isinstance(r, dict) and not r.get("budget_ok", True)
        ]
        if overs:
            _logger.warning(
                "Token budget exceeded for analysts: %s", overs,
            )

        return results

    def run_selected(
        self, context: AnalystContext, analysts: list[str],
    ) -> dict[str, Any]:
        """Run a subset of analysts.

        Args:
            context: Context variables.
            analysts: Analyst types to run (e.g. ``["market", "breakout"]``).

        Returns:
            Same structure as :meth:`run`, but only for the selected types.
        """
        ctx = copy.copy(context)
        results: dict[str, Any] = {"input_tokens_total": 0}
        for at in analysts:
            if at == "breakout":
                similar = self._retriever.search(
                    factor_vector=ctx.factor_vector or [],
                )
                results["similar_cases"] = similar
                if similar:
                    cases_hint = "相似案例: " + ", ".join(
                        f"{c['ticker']}({c['date']}, sim={c['similarity']})"
                        for c in similar[:5]
                    )
                    ctx.factor_scores_summary = (
                        f"{ctx.factor_scores_summary}\n{cases_hint}".strip()
                    )
            result = run_analyst(at, ctx, self._invoker)
            results[at] = result
            results["input_tokens_total"] += result.get("input_tokens", 0)
        return results


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_PATTERN = re.compile(r"\{[\s\S]*?\}")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object block from *text* and parse it.

    Raises:
        ValueError: If no JSON object is found.
        json.JSONDecodeError: If the extracted block is not valid JSON.
    """
    match = _JSON_PATTERN.search(text)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group())
