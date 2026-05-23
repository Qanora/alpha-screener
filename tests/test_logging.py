"""Tests for structured JSON logging system.

Issue #87: Structured JSON logging.
Reference: PRD 9.1.
"""

import io
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4

import pytest

from alphascreener.logging import VALID_MODULES, JsonFormatter, get_logger

# ============================================================================
# Helpers
# ============================================================================


def _capture_one(level: int, msg: str, **extra) -> dict:
    """Emit a single log record and return the parsed JSON dict.

    Uses a UUID-based logger name so every call gets a fresh logger, avoiding
    test-order pollution from modules that pre-configure well-known loggers.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger_name = f"test_{uuid4().hex}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    method = {
        logging.DEBUG: logger.debug,
        logging.INFO: logger.info,
        logging.WARNING: logger.warning,  # levelname==WARNING -> "WARN" in JSON
        logging.ERROR: logger.error,
    }[level]
    method(msg, extra=extra if extra else None)
    handler.flush()

    stream.seek(0)
    return json.loads(stream.readline())


# ============================================================================
# JsonFormatter
# ============================================================================


class TestJsonFormatter:
    """JSON format output: {timestamp, level, module, event, data, cost_usd}."""

    def test_outputs_valid_json(self):
        record = _capture_one(logging.INFO, "test_message")
        assert isinstance(record, dict)

    def test_contains_required_fields(self):
        record = _capture_one(logging.INFO, "test_message")
        for field in ("timestamp", "level", "module", "event", "data", "cost_usd"):
            assert field in record, f"Missing required field: {field}"

    def test_timestamp_is_iso8601_utc(self):
        record = _capture_one(logging.INFO, "test_message")
        ts = record["timestamp"]
        assert "T" in ts
        # ISO8601 UTC ends with +00:00 or Z
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_module_field_from_logger_name(self):
        """Module field reflects the logger name (a UUID-based unique name)."""
        record = _capture_one(logging.INFO, "scan_completed")
        module = record["module"]
        # Logger name is "test_<32-hex-chars>"
        assert module.startswith("test_"), f"Expected module to start with 'test_', got {module!r}"
        assert len(module) == 37, f"Expected 37-char module name, got {len(module)}: {module!r}"

    def test_event_field_from_message(self):
        record = _capture_one(logging.INFO, "alpha_analysis_completed")
        assert record["event"] == "alpha_analysis_completed"

    def test_handles_non_serializable_objects_in_data(self):
        """Non-serializable objects in data are converted via default=str."""
        from datetime import UTC, datetime

        dt = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)
        record = _capture_one(
            logging.INFO,
            "scan_completed",
            data={"tickers": ["AAPL"], "scanned_at": dt},
        )
        assert record["data"] == {"tickers": ["AAPL"], "scanned_at": str(dt)}

    def test_data_field_defaults_to_empty_object(self):
        record = _capture_one(logging.INFO, "test")
        assert record["data"] == {}

    def test_cost_usd_defaults_to_zero(self):
        record = _capture_one(logging.INFO, "test")
        assert record["cost_usd"] == 0.0


# ============================================================================
# Log levels
# ============================================================================


class TestLogLevels:
    """ERROR, WARN, INFO, DEBUG levels map correctly in JSON output."""

    def test_info_level(self):
        record = _capture_one(logging.INFO, "msg")
        assert record["level"] == "INFO"

    def test_warning_level_maps_to_warn(self):
        record = _capture_one(logging.WARNING, "msg")
        assert record["level"] == "WARN"

    def test_error_level(self):
        record = _capture_one(logging.ERROR, "msg")
        assert record["level"] == "ERROR"

    def test_debug_level(self):
        record = _capture_one(logging.DEBUG, "msg")
        assert record["level"] == "DEBUG"


# ============================================================================
# Extra fields: data and cost_usd
# ============================================================================


class TestExtraFields:
    """data and cost_usd passed via extra dict appear in JSON output."""

    def test_data_extra_field(self):
        record = _capture_one(
            logging.INFO,
            "scan_completed",
            data={"tickers": 5, "passed": 3},
        )
        assert record["data"] == {"tickers": 5, "passed": 3}

    def test_cost_usd_extra_field(self):
        record = _capture_one(
            logging.INFO,
            "llm_call_completed",
            cost_usd=0.003,
        )
        assert record["cost_usd"] == 0.003

    def test_both_extra_fields_together(self):
        record = _capture_one(
            logging.INFO,
            "alpha_scored",
            data={"ticker": "AAPL", "score": 0.87},
            cost_usd=0.012,
        )
        assert record["data"] == {"ticker": "AAPL", "score": 0.87}
        assert record["cost_usd"] == 0.012


# ============================================================================
# Logger factory: get_logger
# ============================================================================


class TestGetLogger:
    """Module-level logger factory for screening/refining/backtesting/evolution."""

    @pytest.mark.parametrize("module", ["screening", "refining", "backtesting", "evolution"])
    def test_valid_modules_return_logger(self, module, tmp_path):
        logger = _fresh_get_logger(module, log_dir=str(tmp_path))
        assert isinstance(logger, logging.Logger)
        assert logger.name == module

    def test_invalid_module_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown module"):
            get_logger("invalid_module")

    def test_logger_has_json_formatter(self, tmp_path):
        logger = _fresh_get_logger("screening", log_dir=str(tmp_path))
        assert len(logger.handlers) >= 1
        formatters = {type(h.formatter) for h in logger.handlers if h.formatter}
        assert JsonFormatter in formatters

    def test_same_module_returns_same_logger(self, tmp_path):
        logger1 = _fresh_get_logger("screening", log_dir=str(tmp_path))
        logger2 = _fresh_get_logger("screening", log_dir=str(tmp_path))
        assert logger1 is logger2

    def test_different_modules_return_different_loggers(self, tmp_path):
        logger1 = _fresh_get_logger("screening", log_dir=str(tmp_path))
        logger2 = _fresh_get_logger("refining", log_dir=str(tmp_path))
        assert logger1 is not logger2


# ============================================================================
# File rotation
# ============================================================================


def _fresh_get_logger(module: str, log_dir: str | None = None) -> logging.Logger:
    """Return a get_logger result with handlers forcibly refreshed
    (needed when earlier tests have already created singleton loggers)."""
    logger = logging.getLogger(module)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    if hasattr(logger, "_alphascreener_configured"):
        delattr(logger, "_alphascreener_configured")
    return get_logger(module, log_dir=log_dir)


class TestLogRotation:
    """30-day log file rotation via TimedRotatingFileHandler."""

    def test_file_handler_has_30_day_rotation(self, tmp_path):
        """File handler is configured with D rotation and backupCount=30."""
        log_dir = tmp_path / "logs"
        logger = _fresh_get_logger("backtesting", log_dir=str(log_dir))

        file_handlers = [h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1, "Expected one TimedRotatingFileHandler"

        handler = file_handlers[0]
        assert handler.when == "D"
        assert handler.backupCount == 30
        # TimedRotatingFileHandler stores interval in seconds when when='D'
        assert handler.interval == 86400

    def test_log_file_is_created(self, tmp_path):
        """Writing a log message creates the log file on disk."""
        log_dir = tmp_path / "logs"
        logger = _fresh_get_logger("evolution", log_dir=str(log_dir))

        logger.info("system_started", extra={"data": {"version": "0.1.0"}})

        for h in logger.handlers:
            if isinstance(h, TimedRotatingFileHandler):
                h.flush()

        log_file = log_dir / "evolution.log"
        assert log_file.exists(), f"Log file not found: {log_file}"

        content = log_file.read_text()
        record = json.loads(content.strip())
        assert record["module"] == "evolution"
        assert record["event"] == "system_started"
        assert record["data"] == {"version": "0.1.0"}

    def test_default_log_dir_derived_from_settings(self, tmp_path, monkeypatch):
        """When log_dir is not provided, it derives from Settings().alphascreener_home."""
        monkeypatch.setenv("ALPHASCREENER_HOME", str(tmp_path / "custom_home"))

        _fresh_get_logger("screening").info("test")

        log_file = tmp_path / "custom_home" / "logs" / "screening.log"
        assert log_file.exists()

    def test_rotation_handler_writes_to_correct_path(self, tmp_path):
        """File handler baseFilename is inside the provided log_dir."""
        log_dir = tmp_path / "mylogs"
        logger = _fresh_get_logger("refining", log_dir=str(log_dir))

        file_handlers = [h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename.startswith(str(log_dir))


# ============================================================================
# VALID_MODULES constant
# ============================================================================


class TestValidModules:
    """Verify the VALID_MODULES frozenset."""

    def test_contains_expected_modules(self):
        assert "screening" in VALID_MODULES
        assert "refining" in VALID_MODULES
        assert "backtesting" in VALID_MODULES
        assert "evolution" in VALID_MODULES
        assert "monitoring" in VALID_MODULES

    def test_exactly_five_modules(self):
        assert len(VALID_MODULES) == 5
