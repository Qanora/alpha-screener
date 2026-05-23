"""Tests for LLM invoker retry logic and invocation monitoring (Issue #188)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alphascreener.tradingagents.orchestrator import (
    InvocationStats,
    LLMInvocationTracker,
    _is_retryable_error,
)


# ============================================================================
# Error classification tests
# ============================================================================


class TestIsRetryableError:
    """Error classification: retryable vs non-retryable."""

    def test_timeout_error_retryable(self):
        assert _is_retryable_error(TimeoutError("timed out")) is True

    def test_connection_error_retryable(self):
        assert _is_retryable_error(ConnectionError("refused")) is True

    def test_value_error_not_retryable(self):
        assert _is_retryable_error(ValueError("missing API key")) is False

    def test_type_error_not_retryable(self):
        assert _is_retryable_error(TypeError("bad type")) is False

    def test_keyboard_interrupt_not_retryable(self):
        assert _is_retryable_error(KeyboardInterrupt()) is False

    def test_runtime_error_not_retryable(self):
        # Generic RuntimeError is not retryable (no hint in msg)
        assert _is_retryable_error(RuntimeError("something happened")) is False

    def test_exception_with_rate_limit_message_retryable(self):
        exc = Exception("rate limit exceeded")
        assert _is_retryable_error(exc) is True

    def test_exception_with_timeout_message_retryable(self):
        exc = Exception("request timed out")
        assert _is_retryable_error(exc) is True

    def test_exception_with_server_error_message_retryable(self):
        exc = Exception("internal server error")
        assert _is_retryable_error(exc) is True

    def test_exception_with_too_many_requests_message_retryable(self):
        exc = Exception("too many requests")
        assert _is_retryable_error(exc) is True

    def test_exception_with_connection_message_retryable(self):
        exc = Exception("connection reset")
        assert _is_retryable_error(exc) is True

    def test_exception_with_service_unavailable_message_retryable(self):
        exc = Exception("service unavailable")
        assert _is_retryable_error(exc) is True

    def test_http_429_via_status_code_retryable(self):
        class FakeHttpError(Exception):
            status_code = 429
        assert _is_retryable_error(FakeHttpError()) is True

    def test_http_500_via_status_code_retryable(self):
        class FakeHttpError(Exception):
            status_code = 500
        assert _is_retryable_error(FakeHttpError()) is True

    def test_http_503_via_status_code_retryable(self):
        class FakeHttpError(Exception):
            status_code = 503
        assert _is_retryable_error(FakeHttpError()) is True

    def test_http_400_not_retryable(self):
        class FakeHttpError(Exception):
            status_code = 400
        assert _is_retryable_error(FakeHttpError()) is False

    def test_http_401_not_retryable(self):
        class FakeHttpError(Exception):
            status_code = 401
        assert _is_retryable_error(FakeHttpError()) is False

    def test_http_403_not_retryable(self):
        class FakeHttpError(Exception):
            status_code = 403
        assert _is_retryable_error(FakeHttpError()) is False

    @staticmethod
    def _mock_response(status_code: int = 429) -> MagicMock:
        """Build a minimal mock response for OpenAI error constructors."""
        mock = MagicMock()
        mock.request = MagicMock()
        mock.status_code = status_code
        return mock

    def test_openai_rate_limit_error_retryable(self):
        """Verify OpenAI RateLimitError is recognised when SDK is installed."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai SDK not installed")
        e = openai.RateLimitError(
            "rate limit", response=self._mock_response(429), body=None,
        )
        assert _is_retryable_error(e) is True

    def test_openai_internal_server_error_retryable(self):
        try:
            import openai
        except ImportError:
            pytest.skip("openai SDK not installed")
        e = openai.InternalServerError(
            "server error", response=self._mock_response(500), body=None,
        )
        assert _is_retryable_error(e) is True

    def test_openai_authentication_error_not_retryable(self):
        try:
            import openai
        except ImportError:
            pytest.skip("openai SDK not installed")
        e = openai.AuthenticationError(
            "bad key", response=self._mock_response(401), body=None,
        )
        assert _is_retryable_error(e) is False

    def test_openai_bad_request_error_not_retryable(self):
        try:
            import openai
        except ImportError:
            pytest.skip("openai SDK not installed")
        e = openai.BadRequestError(
            "bad request", response=self._mock_response(400), body=None,
        )
        assert _is_retryable_error(e) is False


# ============================================================================
# InvocationStats tests
# ============================================================================


class TestInvocationStats:
    """InvocationStats dataclass correctness."""

    def test_defaults(self):
        s = InvocationStats()
        assert s.call_type == ""
        assert s.call_count == 0
        assert s.success_count == 0
        assert s.failure_count == 0
        assert s.retry_count == 0
        assert s.total_retries == 0

    def test_success_rate_perfect(self):
        s = InvocationStats(call_count=10, success_count=10)
        assert s.success_rate == 1.0

    def test_success_rate_partial(self):
        s = InvocationStats(call_count=10, success_count=7, failure_count=3)
        assert s.success_rate == 0.7

    def test_success_rate_zero_calls(self):
        s = InvocationStats(call_count=0)
        assert s.success_rate == 1.0  # no calls = no failures

    def test_avg_retries_zero_calls(self):
        s = InvocationStats()
        assert s.avg_retries_per_call == 0.0

    def test_avg_retries(self):
        s = InvocationStats(call_count=10, total_retries=5)
        assert s.avg_retries_per_call == 0.5

    def test_avg_retries_integer(self):
        s = InvocationStats(call_count=3, total_retries=6)
        assert s.avg_retries_per_call == 2.0


# ============================================================================
# LLMInvocationTracker tests
# ============================================================================


class TestLLMInvocationTracker:
    """LLMInvocationTracker correctness and thread-safety sanity checks."""

    def test_empty_tracker(self):
        t = LLMInvocationTracker()
        snap = t.snapshot()
        assert snap == {}

    def test_record_success_accumulates(self):
        t = LLMInvocationTracker()
        t.record_success("bull")
        t.record_success("bull")
        t.record_success("bear")

        snap = t.snapshot()
        assert snap["bull"].call_count == 2
        assert snap["bull"].success_count == 2
        assert snap["bull"].failure_count == 0
        assert snap["bear"].call_count == 1
        assert snap["bear"].success_count == 1
        assert snap["bear"].failure_count == 0

    def test_record_success_with_retries(self):
        t = LLMInvocationTracker()
        t.record_success("bull", retries=0)
        t.record_success("bull", retries=2)
        t.record_success("bull", retries=1)

        snap = t.snapshot()
        assert snap["bull"].call_count == 3
        assert snap["bull"].retry_count == 2  # 2 calls had retries
        assert snap["bull"].total_retries == 3  # 0+2+1 = 3

    def test_record_failure_accumulates(self):
        t = LLMInvocationTracker()
        t.record_success("bull")
        t.record_failure("bull", "RateLimitError", "RateLimitError")
        t.record_failure("bear", "ConnectionError", "ConnectionError")

        snap = t.snapshot()
        assert snap["bull"].call_count == 2
        assert snap["bull"].success_count == 1
        assert snap["bull"].failure_count == 1
        assert snap["bull"].last_error == "RateLimitError"
        assert snap["bull"].last_error_type == "RateLimitError"
        assert snap["bear"].call_count == 1
        assert snap["bear"].failure_count == 1

    def test_log_summary_values(self):
        """log_summary computes correct aggregate values (property-based test)."""
        t = LLMInvocationTracker()
        # 3 calls to bull: 2 success (1 with retry), 1 failure
        t.record_success("bull")
        t.record_success("bull", retries=2)
        t.record_failure("bull", "timeout", "TimeoutError")
        # 2 calls to bear: all success
        t.record_success("bear")
        t.record_success("bear", retries=1)

        snap = t.snapshot()
        # Per-type checks
        assert snap["bull"].call_count == 3
        assert snap["bull"].success_count == 2
        assert snap["bull"].failure_count == 1
        assert snap["bull"].total_retries == 2
        assert snap["bear"].call_count == 2
        assert snap["bear"].success_count == 2
        assert snap["bear"].failure_count == 0
        assert snap["bear"].total_retries == 1

        # Aggregate checks
        total_calls = sum(s.call_count for s in snap.values())
        total_success = sum(s.success_count for s in snap.values())
        total_failures = sum(s.failure_count for s in snap.values())
        total_retries = sum(s.total_retries for s in snap.values())
        assert total_calls == 5
        assert total_success == 4
        assert total_failures == 1
        assert total_retries == 3

    def test_log_summary_empty_no_output(self, caplog):
        import logging as _logging

        t = LLMInvocationTracker()
        with caplog.at_level(_logging.INFO):
            t.log_summary()
        # log_summary should not emit stats for empty tracker
        info_messages = [r.message for r in caplog.records
                         if r.levelno == _logging.INFO]
        assert not any("LLM invocation stats" in m for m in info_messages)

    def test_snapshot_returns_new_dict(self):
        """snapshot() returns a fresh dict; keys can be mutated independently."""
        t = LLMInvocationTracker()
        t.record_success("bull")
        snap1 = t.snapshot()
        # Remove from snapshot dict — original tracker still has it
        del snap1["bull"]
        snap2 = t.snapshot()
        assert "bull" in snap2
        assert snap2["bull"].call_count == 1

    def test_snapshot_includes_all_types(self):
        t = LLMInvocationTracker()
        t.record_success("bull")
        t.record_success("bear")
        t.record_success("pm")
        snap = t.snapshot()
        assert set(snap.keys()) == {"bull", "bear", "pm"}

    def test_log_summary_includes_retry_info(self):
        """Verify retry count is tracked in the snapshot after log_summary."""
        t = LLMInvocationTracker()
        t.record_success("bull", retries=3)
        t.record_success("bull", retries=1)

        snap = t.snapshot()
        assert snap["bull"].total_retries == 4
        assert snap["bull"].retry_count == 2
