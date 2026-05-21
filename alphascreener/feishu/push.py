"""Feishu daily card push orchestrator.

Issue #104: Feishu daily card push.
Reference: PRD 7.5.2 / 7.5.4 — API call flow + failure handling.

Key behaviours:
  - Calls get_tenant_access_token() for authentication.
  - Sends interactive card via POST /im/v1/messages.
  - Tenacity retry: 3 attempts with wait 5s / 15s / 60s.
  - 401 response -> auto-refresh token + retry once.
  - 400 response -> fall back to plain-text message.
  - Consecutive failure counter (3+ days -> ERROR log, non-blocking).
  - Respects FEISHU_PUSH_ENABLED flag.
"""

from __future__ import annotations

import json
import logging
from enum import Enum

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from alphascreener.config import Settings
from alphascreener.feishu.card import CardData, build_card_json, build_fallback_text
from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token
from alphascreener.logging import get_logger

_logger: logging.Logger = get_logger("screening")

# Feishu message send endpoint
_MESSAGE_URL: str = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"

# Retry config (PRD 7.5.4)
_RETRY_ATTEMPTS: int = 3
_RETRY_WAIT_MULTIPLIER: float = 5.0  # base multiplier for exponential wait
_RETRY_MAX_WAIT: float = 60.0  # cap at 60s

# In-process consecutive failure counter
_consecutive_failures: int = 0
_CONSECUTIVE_THRESHOLD: int = 3


# ============================================================================
# Push result enum
# ============================================================================


class PushResult(Enum):
    """Outcome of a push_daily_report() call."""

    OK = "ok"
    OK_DEGRADED = "ok_degraded"  # card failed, plain text succeeded
    DISABLED = "disabled"  # FEISHU_PUSH_ENABLED=false
    SKIPPED = "skipped"  # missing credentials
    FAILED = "failed"


# ============================================================================
# Consecutive failure tracking
# ============================================================================


def _reset_consecutive_failures() -> None:
    """Reset the consecutive failure counter (for testing)."""
    global _consecutive_failures
    _consecutive_failures = 0


def _record_failure() -> None:
    """Increment failure counter and emit ERROR log at threshold."""
    global _consecutive_failures
    _consecutive_failures += 1
    if _consecutive_failures >= _CONSECUTIVE_THRESHOLD:
        _logger.error(
            "Feishu push has failed %d consecutive days — check credentials and network",
            _consecutive_failures,
        )


def _record_success() -> None:
    """Reset the consecutive failure counter on success."""
    global _consecutive_failures
    _consecutive_failures = 0


# ============================================================================
# HTTP helpers
# ============================================================================


def _send_message(
    token: str, target_openid: str, content: str, msg_type: str = "interactive"
) -> requests.Response:
    """Send a message to the Feishu IM API.

    Args:
        token: Valid tenant_access_token.
        target_openid: Target user's Open ID.
        content: JSON string for interactive card, or plain text.
        msg_type: ``"interactive"`` for card, ``"text"`` for plain text.

    Returns:
        ``requests.Response`` object.
    """
    if msg_type == "text":
        content = json_dumps_text(content)
    body: dict[str, str] = {
        "receive_id": target_openid,
        "msg_type": msg_type,
        "content": content,
    }

    resp = requests.post(
        _MESSAGE_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=30,
    )
    return resp


def json_dumps_text(text: str) -> str:
    """Wrap plain text in the Feishu text message JSON envelope."""
    return json.dumps({"text": text}, ensure_ascii=False)


# ============================================================================
# Main push function
# ============================================================================


@retry(
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=_RETRY_WAIT_MULTIPLIER, max=_RETRY_MAX_WAIT),
    reraise=True,
)
def _send_card_with_retry(token: str, target_openid: str, card_json: str) -> requests.Response:
    """Send the interactive card with tenacity retry on transient failures.

    This inner function is decorated with @retry so that only the actual
    HTTP call is retried, not the entire push_daily_report() flow.
    """
    resp = _send_message(token, target_openid, card_json, msg_type="interactive")
    if resp.status_code == 200:
        return resp
    # 401 is handled by the caller (token refresh), not retried here
    if resp.status_code == 401:
        return resp
    # 400 is handled by the caller (degradation), not retried here
    if resp.status_code == 400:
        return resp
    # All other errors (5xx, network) are retried
    raise RuntimeError(f"Feishu message send failed: HTTP {resp.status_code} {resp.text[:300]}")


def push_daily_report(data: CardData) -> PushResult:
    """Push the daily screening report as a Feishu interactive card.

    Complete flow:
    1. Check enabled + credentials.
    2. Get tenant_access_token (cached or fresh).
    3. Build interactive card JSON.
    4. Send with tenacity retry.
    5. 401 -> clear cache, refresh token, retry once.
    6. 400 -> fall back to plain text.
    7. Track consecutive failures.

    Args:
        data: Populated :class:`CardData`.

    Returns:
        :class:`PushResult` indicating the outcome.
    """
    settings = Settings()

    if not settings.feishu_push_enabled:
        _logger.debug("Feishu push disabled, skipping")
        return PushResult.DISABLED

    app_id = settings.feishu_app_id
    app_secret = settings.feishu_app_secret
    target_openid = settings.feishu_target_openid

    if not app_id or not app_secret or not target_openid:
        _logger.warning(
            "Feishu credentials incomplete (app_id=%s, secret=%s, openid=%s), skipping",
            "set" if app_id else "missing",
            "set" if app_secret else "missing",
            "set" if target_openid else "missing",
        )
        return PushResult.SKIPPED

    try:
        # Step 1: Get token
        token = get_tenant_access_token(app_id, app_secret)

        # Step 2: Build card JSON
        card_json = build_card_json(data)

        # Step 3: Send with retry
        resp = _send_card_with_retry(token, target_openid, card_json)

        # Step 4: Handle 401 -> refresh token + retry once
        if resp.status_code == 401:
            _logger.warning("Got 401, refreshing token and retrying once")
            clear_token_cache()
            token = get_tenant_access_token(app_id, app_secret)
            resp = _send_message(token, target_openid, card_json, msg_type="interactive")

        # Step 5: Handle 400 -> fallback to plain text
        if resp.status_code == 400:
            _logger.warning("Card rendering failed (400), falling back to plain text")
            fallback_text = build_fallback_text(
                report_date=data.report_date,
                top_five=data.top_five,
                p20_pure=data.p20_pure,
                p20_llm=data.p20_llm,
                lift_pure=data.lift_pure,
                lift_llm=data.lift_llm,
                base_rate=data.base_rate,
                alerts_summary=data.alerts_summary,
            )
            resp = _send_message(token, target_openid, fallback_text, msg_type="text")
            if resp.status_code == 200:
                _record_success()
                return PushResult.OK_DEGRADED

        # Step 6: Final status check
        if resp.status_code == 200:
            resp_json = resp.json()
            if resp_json.get("code", -1) != 0:
                _logger.error(
                    "Feishu API returned error: code=%s msg=%s",
                    resp_json.get("code"),
                    resp_json.get("msg", ""),
                )
                _record_failure()
                return PushResult.FAILED
            _record_success()
            return PushResult.OK

        # Any other non-200 status
        _logger.error(
            "Feishu push failed: HTTP %d %s",
            resp.status_code,
            resp.text[:300],
        )
        _record_failure()
        return PushResult.FAILED

    except Exception:
        _logger.exception("Feishu push raised unhandled exception")
        _record_failure()
        return PushResult.FAILED
