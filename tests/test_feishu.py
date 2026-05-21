"""Tests for Feishu daily card push module.

Issue #104: Feishu daily card push.
Reference: PRD 7.5.1-7.5.5.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest import mock

import pytest

# ============================================================================
# Shared mock fixtures
# ============================================================================


@pytest.fixture
def mock_token_api():
    """Mock requests.post for the Feishu token endpoint."""
    with mock.patch("alphascreener.feishu.token.requests.post") as mp:
        mp.return_value = _token_ok_response()
        yield mp


@pytest.fixture
def mock_settings_enabled():
    """Mock Settings with Feishu push enabled and credentials set."""
    with mock.patch("alphascreener.feishu.push.Settings") as ms:
        ms.return_value.feishu_push_enabled = True
        ms.return_value.feishu_app_id = "cli_test"
        ms.return_value.feishu_app_secret = "sec_test"
        ms.return_value.feishu_target_openid = "ou_test"
        yield ms


def _token_ok_response(token="t-test-token-123"):
    m = mock.MagicMock()
    m.status_code = 200
    m.json.return_value = {"tenant_access_token": token, "expire": 7200}
    return m


def _msg_ok_response():
    m = mock.MagicMock()
    m.status_code = 200
    m.json.return_value = {"code": 0, "msg": "ok"}
    return m


def _msg_401_response():
    m = mock.MagicMock()
    m.status_code = 401
    m.text = "Unauthorized"
    return m


def _msg_400_response():
    m = mock.MagicMock()
    m.status_code = 400
    m.text = "Card rendering failed"
    return m


def _msg_500_response():
    m = mock.MagicMock()
    m.status_code = 500
    m.text = "Server Error"
    return m


# ============================================================================
# Token tests
# ============================================================================


class TestTenantAccessToken:
    """Tests for tenant_access_token acquisition and caching."""

    def test_fetch_new_token_on_first_call(self, mock_token_api):
        """First call should fetch a fresh token from API."""
        from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token

        clear_token_cache()
        token = get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        assert token == "t-test-token-123"
        assert mock_token_api.called

    def test_cache_token_within_validity(self, mock_token_api):
        """Second call within validity window returns cached token."""
        from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token

        clear_token_cache()
        token1 = get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        call_count = mock_token_api.call_count
        token2 = get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        assert mock_token_api.call_count == call_count
        assert token1 == token2 == "t-test-token-123"

    def test_refresh_token_when_expired(self, mock_token_api):
        """When cached token is expired, fetch a new one."""
        import alphascreener.feishu.token as token_mod
        from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token

        clear_token_cache()
        get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        # Expire the cache
        token_mod._cached_token_expiry = datetime.now() - timedelta(minutes=10)
        mock_token_api.reset_mock()
        mock_token_api.return_value = _token_ok_response("t-new-token")
        token = get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        assert mock_token_api.called
        assert token == "t-new-token"

    def test_refresh_5min_before_expiry(self, mock_token_api):
        """Token should be refreshed when within 5 minutes of expiry."""
        import alphascreener.feishu.token as token_mod
        from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token

        clear_token_cache()
        token_mod._cached_token = "t-nearly-expired"
        token_mod._cached_token_expiry = datetime.now() + timedelta(minutes=4)
        mock_token_api.reset_mock()
        mock_token_api.return_value = _token_ok_response("t-fresh-token")
        token = get_tenant_access_token(app_id="cli_test", app_secret="sec_test")
        assert mock_token_api.called
        assert token == "t-fresh-token"

    def test_api_error_raises(self, mock_token_api):
        """API errors should propagate as RuntimeError."""
        from alphascreener.feishu.token import clear_token_cache, get_tenant_access_token

        clear_token_cache()
        mock_token_api.return_value = mock.MagicMock(status_code=500, text="Internal Server Error")
        with pytest.raises(RuntimeError, match="token"):
            get_tenant_access_token(app_id="cli_test", app_secret="sec_test")

    def test_missing_credentials_raises(self):
        """Missing app_id or app_secret should raise ValueError."""
        from alphascreener.feishu.token import get_tenant_access_token

        with pytest.raises(ValueError, match="APP_ID.*APP_SECRET"):
            get_tenant_access_token(app_id="", app_secret="")
        with pytest.raises(ValueError, match="APP_ID.*APP_SECRET"):
            get_tenant_access_token(app_id="cli_test", app_secret="")


# ============================================================================
# Card tests
# ============================================================================


class TestBuildCardJson:
    """Tests for interactive card JSON building."""

    @pytest.fixture
    def sample_card_data(self):
        from alphascreener.feishu.card import CardData

        return CardData(
            report_date="2026-05-22",
            total_symbols=2000,
            coarse_pass=150,
            refine_count=10,
            top_five=[
                {
                    "ticker": "AAPL",
                    "rating": "Strong Buy",
                    "confidence": 92.5,
                    "catalyst": "AI iPhone super cycle",
                },
                {
                    "ticker": "NVDA",
                    "rating": "Strong Buy",
                    "confidence": 89.0,
                    "catalyst": "B200 ramp",
                },
                {
                    "ticker": "MSFT",
                    "rating": "Buy",
                    "confidence": 85.0,
                    "catalyst": "Copilot enterprise adoption",
                },
                {
                    "ticker": "GOOGL",
                    "rating": "Buy",
                    "confidence": 82.3,
                    "catalyst": "Cloud re-acceleration",
                },
                {
                    "ticker": "META",
                    "rating": "Buy",
                    "confidence": 80.1,
                    "catalyst": "Reels monetization",
                },
            ],
            p20_pure=35.0,
            p20_llm=45.0,
            lift_pure=1.5,
            lift_llm=2.1,
            base_rate=10.0,
            win_rate=62.5,
            sharpe=1.32,
            avg_return=3.8,
            daily_cost=0.45,
            monthly_cost=12.80,
            alerts_summary="ok",
        )

    def test_builds_valid_json(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        parsed = json.loads(result)
        assert parsed["msg_type"] == "interactive"
        assert "card" in parsed

    def test_header_contains_date(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        parsed = json.loads(result)
        title = parsed["card"]["header"]["title"]
        assert "2026-05-22" in title

    def test_contains_scan_overview(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        assert "2000" in content
        assert "150" in content
        assert "10" in content

    def test_contains_top_five_tickers(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        for t in ("AAPL", "NVDA", "MSFT", "GOOGL", "META"):
            assert t in content

    def test_contains_alpha_ablation_metrics(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        for v in ("35.0", "45.0", "1.5", "2.1", "10.0"):
            assert v in content

    def test_contains_backtest_performance(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        for v in ("62.5", "1.32", "3.8"):
            assert v in content

    def test_contains_cost_tracking(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        assert "0.45" in content
        assert "12.80" in content

    def test_contains_alerts_section(self, sample_card_data):
        from alphascreener.feishu.card import build_card_json

        result = build_card_json(sample_card_data)
        content = json.dumps(result)
        assert "ok" in content

    def test_fewer_than_five_tickers(self):
        from alphascreener.feishu.card import CardData, build_card_json

        data = CardData(
            report_date="2026-05-22",
            total_symbols=1000,
            coarse_pass=50,
            refine_count=2,
            top_five=[
                {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"},
                {"ticker": "NVDA", "rating": "Buy", "confidence": 88.0, "catalyst": "T"},
            ],
            p20_pure=30.0,
            p20_llm=40.0,
            lift_pure=1.2,
            lift_llm=1.8,
            base_rate=8.0,
            win_rate=60.0,
            sharpe=1.1,
            avg_return=3.0,
            daily_cost=0.20,
            monthly_cost=5.0,
            alerts_summary="ok",
        )
        result = build_card_json(data)
        content = json.dumps(result)
        assert "AAPL" in content
        assert "NVDA" in content


class TestBuildFallbackText:
    """Tests for fallback plain text message (400 degradation)."""

    def test_fallback_contains_top_tickers(self):
        from alphascreener.feishu.card import build_fallback_text

        text = build_fallback_text(
            report_date="2026-05-22",
            top_five=[
                {"ticker": "AAPL", "rating": "Strong Buy", "confidence": 92.5, "catalyst": "AI"},
                {"ticker": "NVDA", "rating": "Strong Buy", "confidence": 89.0, "catalyst": "GPU"},
            ],
            p20_pure=35.0,
            p20_llm=45.0,
            lift_pure=1.5,
            lift_llm=2.1,
            base_rate=10.0,
            alerts_summary="ok",
        )
        assert "AAPL" in text
        assert "NVDA" in text
        assert "2026-05-22" in text
        assert "Strong Buy" in text

    def test_fallback_no_tickers(self):
        from alphascreener.feishu.card import build_fallback_text

        text = build_fallback_text(
            report_date="2026-05-22",
            top_five=[],
            p20_pure=0.0,
            p20_llm=0.0,
            lift_pure=0.0,
            lift_llm=0.0,
            base_rate=0.0,
            alerts_summary="ok",
        )
        assert "2026-05-22" in text
        assert len(text) > 0


# ============================================================================
# Push tests
# ============================================================================


class TestPushOrchestration:
    """Tests for push_daily_report with retry, degradation, and failure tracking."""

    def test_push_disabled_skips(self):
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import PushResult, push_daily_report

        with mock.patch("alphascreener.feishu.push.Settings") as ms:
            ms.return_value.feishu_push_enabled = False
            result = push_daily_report(CardData())
            assert result == PushResult.DISABLED

    def test_missing_credentials_skips(self):
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import PushResult, push_daily_report

        with mock.patch("alphascreener.feishu.push.Settings") as ms:
            ms.return_value.feishu_push_enabled = True
            ms.return_value.feishu_app_id = ""
            ms.return_value.feishu_app_secret = ""
            ms.return_value.feishu_target_openid = ""
            result = push_daily_report(CardData())
            assert result == PushResult.SKIPPED

    def test_successful_push(self, mock_settings_enabled):
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import (
            PushResult,
            _reset_consecutive_failures,
            push_daily_report,
        )

        _reset_consecutive_failures()
        with (
            mock.patch("alphascreener.feishu.push.get_tenant_access_token") as token_m,
            mock.patch("alphascreener.feishu.push._send_card_with_retry") as send_m,
        ):
            token_m.return_value = "t-test-token"
            send_m.return_value = _msg_ok_response()
            result = push_daily_report(
                CardData(
                    report_date="2026-05-22",
                    top_five=[
                        {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"}
                    ],
                )
            )
            assert result == PushResult.OK

    def test_401_auto_refresh_and_retry(self, mock_settings_enabled):
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import (
            PushResult,
            _reset_consecutive_failures,
            push_daily_report,
        )

        _reset_consecutive_failures()
        with (
            mock.patch("alphascreener.feishu.push.get_tenant_access_token") as token_m,
            mock.patch("alphascreener.feishu.push._send_card_with_retry") as send_m,
            mock.patch("alphascreener.feishu.push._send_message") as msg_m,
        ):
            # First call to _send_card_with_retry returns 401
            send_m.return_value = _msg_401_response()
            # Token refresh returns new token
            token_m.return_value = "t-refreshed"
            # Retry _send_message succeeds
            msg_m.return_value = _msg_ok_response()
            result = push_daily_report(
                CardData(
                    report_date="2026-05-22",
                    top_five=[
                        {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"}
                    ],
                )
            )
            assert result == PushResult.OK
            assert token_m.call_count == 2  # initial + refresh
            assert msg_m.called

    def test_400_degradation_to_plain_text(self, mock_settings_enabled):
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import (
            PushResult,
            _reset_consecutive_failures,
            push_daily_report,
        )

        _reset_consecutive_failures()
        with (
            mock.patch("alphascreener.feishu.push.get_tenant_access_token") as token_m,
            mock.patch("alphascreener.feishu.push._send_card_with_retry") as send_m,
            mock.patch("alphascreener.feishu.push._send_message") as msg_m,
        ):
            token_m.return_value = "t-test-token"
            # Card send fails with 400
            send_m.return_value = _msg_400_response()
            # Fallback text send succeeds
            msg_m.return_value = _msg_ok_response()
            result = push_daily_report(
                CardData(
                    report_date="2026-05-22",
                    top_five=[
                        {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"}
                    ],
                )
            )
            assert result == PushResult.OK_DEGRADED
            assert msg_m.called

    def test_consecutive_failure_tracking(self, mock_settings_enabled):
        import alphascreener.feishu.push as push_mod
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import (
            PushResult,
            _reset_consecutive_failures,
            push_daily_report,
        )

        _reset_consecutive_failures()
        with (
            mock.patch("alphascreener.feishu.push.get_tenant_access_token") as token_m,
            mock.patch("alphascreener.feishu.push._send_card_with_retry") as send_m,
        ):
            token_m.return_value = "t-test-token"
            # Simulate exhausted retries by raising exception
            send_m.side_effect = RuntimeError("All retries exhausted")
            result = push_daily_report(
                CardData(
                    report_date="2026-05-22",
                    top_five=[
                        {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"}
                    ],
                )
            )
            assert result == PushResult.FAILED
            assert push_mod._consecutive_failures >= 1

    def test_success_resets_consecutive_failures(self, mock_settings_enabled):
        import alphascreener.feishu.push as push_mod
        from alphascreener.feishu.card import CardData
        from alphascreener.feishu.push import (
            PushResult,
            _reset_consecutive_failures,
            push_daily_report,
        )

        _reset_consecutive_failures()
        push_mod._consecutive_failures = 2
        with (
            mock.patch("alphascreener.feishu.push.get_tenant_access_token") as token_m,
            mock.patch("alphascreener.feishu.push._send_card_with_retry") as send_m,
        ):
            token_m.return_value = "t-test-token"
            send_m.return_value = _msg_ok_response()
            result = push_daily_report(
                CardData(
                    report_date="2026-05-22",
                    top_five=[
                        {"ticker": "AAPL", "rating": "Buy", "confidence": 90.0, "catalyst": "T"}
                    ],
                )
            )
            assert result == PushResult.OK
            assert push_mod._consecutive_failures == 0


# ============================================================================
# Integration: scheduler task registration
# ============================================================================


class TestTaskRegistration:
    """Verify the Feishu push task is registered in the scheduler task map."""

    def test_daily_feishu_push_in_task_cron(self):
        from alphascreener.scheduler.tasks import TASK_CRON, TASK_FUNCS

        assert "daily_feishu_push" in TASK_CRON
        assert "daily_feishu_push" in TASK_FUNCS
        assert TASK_CRON["daily_feishu_push"] == "5 23 * * 1-5"

    def test_daily_feishu_push_callable(self):
        from alphascreener.scheduler.tasks import TASK_FUNCS

        func = TASK_FUNCS["daily_feishu_push"]
        assert callable(func)
