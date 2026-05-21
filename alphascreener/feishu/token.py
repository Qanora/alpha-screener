"""Tenant access token acquisition and caching.

Issue #104: Feishu daily card push.
Reference: PRD 7.5.2.

Token lifecycle:
- Fetch from Feishu API: POST /auth/v3/tenant_access_token/internal
- Cache in-memory with 2-hour TTL (7200 seconds)
- Refresh proactively when within 5 minutes of expiry
- Thread-safe via module-level variables (single-process scheduler)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

from alphascreener.logging import get_logger

_logger: logging.Logger = get_logger("screening")

_TOKEN_URL: str = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
_REFRESH_WINDOW_MINUTES: int = 5

# Module-level in-memory cache (single-process scheduler, no thread contention)
_cached_token: str | None = None
_cached_token_expiry: datetime | None = None


def clear_token_cache() -> None:
    """Reset the in-memory token cache (for testing)."""
    global _cached_token, _cached_token_expiry
    _cached_token = None
    _cached_token_expiry = None


def _is_token_valid() -> bool:
    """Check if the cached token exists and is still within its validity window."""
    if _cached_token is None or _cached_token_expiry is None:
        return False
    # Refresh 5 minutes before the hard expiry
    refresh_at = _cached_token_expiry - timedelta(minutes=_REFRESH_WINDOW_MINUTES)
    return datetime.now() < refresh_at


def _fetch_token(app_id: str, app_secret: str) -> tuple[str, int]:
    """Call the Feishu tenant_access_token API.

    Returns:
        Tuple of (token_string, expires_in_seconds).

    Raises:
        RuntimeError: If the API call fails.
    """
    resp = requests.post(
        _TOKEN_URL,
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=30,
    )
    if resp.status_code != 200:
        _logger.error(
            "Failed to fetch tenant_access_token: status=%d body=%s",
            resp.status_code,
            resp.text[:500],
        )
        raise RuntimeError(f"Failed to fetch tenant_access_token: HTTP {resp.status_code}")
    data = resp.json()
    token = data.get("tenant_access_token", "")
    expire = data.get("expire", 7200)
    if not token:
        raise RuntimeError("Feishu token response missing 'tenant_access_token'")
    return token, expire


def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    """Get a valid tenant_access_token, refreshing from API if needed.

    Args:
        app_id: Feishu app ID (FEISHU_APP_ID).
        app_secret: Feishu app secret (FEISHU_APP_SECRET).

    Returns:
        A valid tenant_access_token string.

    Raises:
        ValueError: If app_id or app_secret is empty.
        RuntimeError: If the API call fails.
    """
    if not app_id or not app_secret:
        raise ValueError("FEISHU_APP_ID and FEISHU_APP_SECRET must both be set")

    global _cached_token, _cached_token_expiry

    # Return cached token if still valid (with 5-min early refresh window)
    if _is_token_valid():
        _logger.debug("Using cached tenant_access_token")
        return _cached_token  # type: ignore[return-value]

    _logger.info("Fetching new tenant_access_token from Feishu API")
    token, expire_seconds = _fetch_token(app_id, app_secret)
    _cached_token = token
    _cached_token_expiry = datetime.now() + timedelta(seconds=expire_seconds)
    return token
