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
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
# Error classification for retry logic
# ---------------------------------------------------------------------------


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient failure worth retrying.

    Retryable categories:
      - OpenAI rate-limit, server errors, connection/timeout errors
      - Python built-in TimeoutError, ConnectionError
      - Generic HTTP 429 / 5xx from any provider's HTTP stack

    Non-retryable (fail-fast):
      - Authentication / authorisation errors (401, 403)
      - BadRequest / validation errors (400, 404, 422)
      - ValueError from missing API key / misconfiguration
    """
    # ── OpenAI SDK errors ──────────────────────────────────────────────
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.InternalServerError):
            return True
        if isinstance(exc, openai.APIConnectionError):
            return True
        if isinstance(exc, openai.APITimeoutError):
            return True
        # Non-retryable OpenAI errors: AuthenticationError, BadRequestError,
        # PermissionDeniedError, NotFoundError, ConflictError,
        # UnprocessableEntityError — all descend from APIStatusError but
        # NOT from the retryable subtypes above.
    except ImportError:
        pass

    # ── Anthropic SDK errors ───────────────────────────────────────────
    try:
        import anthropic

        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.InternalServerError):
            return True
        if isinstance(exc, anthropic.APIConnectionError):
            return True
        if isinstance(exc, anthropic.APITimeoutError):
            return True
    except ImportError:
        pass

    # ── Google Generative AI SDK errors ────────────────────────────────
    try:
        import google.api_core.exceptions as google_exc

        if isinstance(exc, google_exc.ResourceExhausted):  # 429
            return True
        if isinstance(exc, google_exc.InternalServerError):  # 500
            return True
        if isinstance(exc, google_exc.ServiceUnavailable):  # 503
            return True
        if isinstance(exc, google_exc.DeadlineExceeded):
            return True
    except ImportError:
        pass

    # ── Generic Python / stdlib errors ─────────────────────────────────
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True

    # ── HTTP-level errors from any HTTP library ────────────────────────
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code", 0)
        if code == 429 or (500 <= code < 600):
            return True

    # ── LangGraph / LangChain retryable wrappers ───────────────────────
    exc_name = type(exc).__name__
    exc_msg = str(exc).lower()
    for keyword in ("rate limit", "too many requests", "server error",
                     "service unavailable", "internal server error",
                     "connection", "timeout", "timed out"):
        if keyword in exc_msg or keyword in exc_name.lower():
            return True

    return False


# ---------------------------------------------------------------------------
# Invocation stats tracking (Issue #188)
# ---------------------------------------------------------------------------


@dataclass
class InvocationStats:
    """Per-call-type invocation statistics.

    Thread-safe counters for tracking LLM invocation success/failure rates.
    """

    call_type: str = ""
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    total_retries: int = 0
    last_error: str = ""
    last_error_type: str = ""

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count

    @property
    def avg_retries_per_call(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_retries / self.call_count


@dataclass
class LLMInvocationTracker:
    """Collect per-call-type invocation stats and emit summary logs.

    Usage::

        tracker = LLMInvocationTracker()
        tracker.record_success("bull", retries=1)
        tracker.record_failure("bull", "RateLimitError")
        tracker.log_summary()  # emits a single log line with all stats
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stats: dict[str, InvocationStats] = field(default_factory=dict, init=False)
    _created_at: float = field(default_factory=time.monotonic, init=False)

    def _get_or_create(self, call_type: str) -> InvocationStats:
        with self._lock:
            if call_type not in self._stats:
                self._stats[call_type] = InvocationStats(call_type=call_type)
            return self._stats[call_type]

    def record_success(self, call_type: str, *, retries: int = 0) -> None:
        """Record a successful LLM invocation."""
        stats = self._get_or_create(call_type)
        with self._lock:
            stats.call_count += 1
            stats.success_count += 1
            if retries > 0:
                stats.retry_count += 1
                stats.total_retries += retries

    def record_failure(self, call_type: str, error: str,
                       error_type: str = "") -> None:
        """Record a failed LLM invocation (after all retries exhausted)."""
        stats = self._get_or_create(call_type)
        with self._lock:
            stats.call_count += 1
            stats.failure_count += 1
            stats.last_error = error
            stats.last_error_type = error_type

    def log_summary(self, *, logger: logging.Logger | None = None) -> None:
        """Emit a summary log line with per-type and aggregate stats."""
        _log = logger or _logger
        elapsed = time.monotonic() - self._created_at
        with self._lock:
            stats_list = list(self._stats.values())

        if not stats_list:
            return

        total_calls = sum(s.call_count for s in stats_list)
        total_success = sum(s.success_count for s in stats_list)
        total_failures = sum(s.failure_count for s in stats_list)
        total_retries = sum(s.total_retries for s in stats_list)
        overall_rate = total_success / total_calls if total_calls > 0 else 1.0

        per_type: list[str] = []
        for s in sorted(stats_list, key=lambda x: x.call_type):
            pct = int(s.success_rate * 100)
            per_type.append(
                f"{s.call_type}={s.success_count}/{s.call_count}({pct}pct)"
            )

        overall_pct = int(overall_rate * 100)
        _log.info(
            "LLM invocation stats (%.0fs): "
            "overall=%d/%d (%dpct) retries=%d | %s",
            elapsed,
            total_success,
            total_calls,
            overall_pct,
            total_retries,
            " ".join(per_type),
        )

        if total_failures > 0:
            failures_detail: list[str] = []
            for s in stats_list:
                if s.failure_count > 0:
                    failures_detail.append(
                        f"{s.call_type}: {s.failure_count} failures "
                        f"(last: {s.last_error_type} — {s.last_error[:120]})"
                    )
            _log.warning(
                "LLM invocation failures: %d total | %s",
                total_failures,
                " | ".join(failures_detail),
            )

    def snapshot(self) -> dict[str, InvocationStats]:
        """Return a thread-safe snapshot of current stats."""
        with self._lock:
            return dict(self._stats)


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

# An invoker is a callable that takes a system prompt and returns the
# LLM text response.
Invoker = Callable[[str, int], str]
# Signature: (system_prompt: str, max_output_tokens: int) -> str

# Providers whose API is OpenAI-compatible (chat completions).
# ``base_url`` and ``api_key`` are only forwarded for these providers;
# non-OpenAI providers (anthropic, google, azure) have their own auth
# mechanisms and should not receive OpenAI-specific configuration.
_OPENAI_COMPATIBLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "xai", "deepseek", "qwen", "qwen-cn", "glm", "glm-cn",
     "minimax", "minimax-cn", "ollama", "openrouter"}
)


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
    *,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    invocation_tracker: LLMInvocationTracker | None = None,
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
        max_retries: Maximum retry attempts for transient failures (default 3).
        retry_base_delay: Base delay in seconds for exponential backoff (default 1.0).
        invocation_tracker: Optional tracker for recording per-call stats.

    Returns:
        A callable ``(prompt, max_output_tokens) -> str``.
    """
    if settings is None:
        from alphascreener.config import Settings as _Settings

        settings = _Settings()  # type: ignore[call-arg]

    from alphascreener.tradingagents.llm_adapter import create_llm_client_safe

    # Only forward base_url and api_key for OpenAI-compatible providers.
    # Non-OpenAI providers (anthropic, google, azure) use their own auth
    # mechanisms and should not receive OpenAI-specific configuration.
    is_compat = provider.lower() in _OPENAI_COMPATIBLE_PROVIDERS
    llm = create_llm_client_safe(
        provider,
        settings.llm_model,
        base_url=settings.openai_base_url if is_compat and settings.openai_base_url else None,
        api_key=settings.openai_api_key if is_compat and settings.openai_api_key else None,
    ).get_llm()

    def invoker(prompt: str, max_out_tok: int) -> str:
        from langchain_core.messages import SystemMessage

        last_exc: BaseException | None = None
        retries_used = 0

        for attempt_num in range(max_retries + 1):
            try:
                msg = llm.invoke(
                    [SystemMessage(content=prompt)],
                    max_tokens=max_out_tok,
                )
                result = _normalize_content(msg.content)
                # Success — record and return
                if invocation_tracker is not None:
                    invocation_tracker.record_success(
                        "invoke", retries=retries_used,
                    )
                if retries_used > 0:
                    _logger.info(
                        "LLM invocation succeeded after %d retries", retries_used,
                    )
                return result
            except Exception as exc:
                last_exc = exc

                if not _is_retryable_error(exc):
                    # Non-retryable — fail immediately
                    _logger.error(
                        "LLM invocation failed with non-retryable error: %s: %s",
                        type(exc).__name__, exc,
                    )
                    if invocation_tracker is not None:
                        invocation_tracker.record_failure(
                            "invoke", str(exc), type(exc).__name__,
                        )
                    raise

                if attempt_num < max_retries:
                    delay = retry_base_delay * (2 ** attempt_num)
                    _logger.warning(
                        "LLM invocation attempt %d/%d failed: %s: %s — "
                        "retrying in %.1fs",
                        attempt_num + 1,
                        max_retries + 1,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    retries_used += 1
                else:
                    # All retries exhausted
                    _logger.error(
                        "LLM invocation failed after %d attempts: %s: %s",
                        max_retries + 1,
                        type(exc).__name__,
                        exc,
                    )
                    if invocation_tracker is not None:
                        invocation_tracker.record_failure(
                            "invoke", str(exc), type(exc).__name__,
                        )
                    raise

        # Unreachable — but placate type-checkers
        assert last_exc is not None
        raise last_exc

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


def _normalize_content(content: Any) -> str:
    """Normalise LLM response content to a plain string.

    LangChain ``BaseMessage.content`` is typed as ``str | list[str | dict]``.
    Some models (or tool-calling paths) return a list of content blocks
    instead of a plain string.  This helper converts any form to ``str``
    so the downstream JSON extraction and error-reporting code can safely
    assume a string.

    Args:
        content: The raw ``msg.content`` value.

    Returns:
        A plain ``str``.

    Raises:
        TypeError: If *content* is not a ``str`` or ``list``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Prefer "text" key; fall back to str(item)
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    raise TypeError(
        f"Unexpected LLM content type {type(content).__name__!r}; "
        f"expected str or list, got {content!r}"
    )


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
