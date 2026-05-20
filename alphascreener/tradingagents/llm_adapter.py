"""LLM client adapter — wraps ``tradingagents.llm_clients``.

Issue #96: TradingAgents adapters.
Reference: PRD 8.1 / 8.2.

Encapsulates ``create_llm_client`` and ``BaseLLMClient`` so that callers
never import from ``tradingagents.llm_clients`` directly.
"""

from __future__ import annotations

import logging
from typing import Any

# ---------------------------------------------------------------------------
# Re-export TradingAgents public API
# ---------------------------------------------------------------------------
from tradingagents.llm_clients.base_client import BaseLLMClient  # noqa: F401, E402
from tradingagents.llm_clients.factory import create_llm_client  # noqa: F401, E402, I100

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Adapter-level conveniences
# ---------------------------------------------------------------------------

_logger: logging.Logger = get_logger("screening")

# Canonical list of providers supported by the underlying factory.
# Mirrors the _OPENAI_COMPATIBLE tuple + explicit providers in factory.py.
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "xai",
    "deepseek",
    "qwen",
    "qwen-cn",
    "glm",
    "glm-cn",
    "minimax",
    "minimax-cn",
    "ollama",
    "openrouter",
    "anthropic",
    "google",
    "azure",
)


def create_llm_client_safe(
    provider: str,
    model: str,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Create an LLM client with graceful fallback for unsupported providers.

    Calls the upstream ``create_llm_client``.  When *provider* is not
    recognised, logs a warning and raises ``ValueError`` so the caller can
    degrade instead of crashing.

    Args:
        provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``).
        model: Model identifier.
        base_url: Optional base URL override.
        **kwargs: Provider-specific arguments forwarded to the client.

    Returns:
        A configured :class:`BaseLLMClient` instance.

    Raises:
        ValueError: If *provider* is not in :data:`SUPPORTED_PROVIDERS`.
    """
    if provider.lower() not in {p.lower() for p in SUPPORTED_PROVIDERS}:
        _logger.warning(
            "Unsupported LLM provider %r — falling back to ValueError",
            provider,
        )
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    return create_llm_client(provider, model, base_url=base_url, **kwargs)
