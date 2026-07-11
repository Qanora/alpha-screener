"""Structured JSON logging with module-level logger factory.

Issue #87: Structured JSON logging.
Reference: PRD 9.1.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from alphascreener.config import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODULES = frozenset({"screening", "refining", "backtesting", "evolution", "monitoring"})


def _ensure_log_directory(log_dir: str) -> Path:
    """Return a writable log directory, with fallback on permission errors."""
    candidates = [Path(log_dir), Path(tempfile.gettempdir()) / "alphascreener" / "logs"]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / ".write_test").unlink(missing_ok=True)
            test_path = path / ".write_test"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
            return path
        except OSError:
            continue

    # Last resort: keep current directory logging for in-memory tests.
    fallback = Path.cwd() / "logs" / "alphascreener"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Logging formatter that outputs one JSON object per line.

    Output schema:
        {"timestamp": "<ISO8601>", "level": "<LEVEL>", "module": "<name>",
         "event": "<message>", "data": {...}, "cost_usd": 0.0}
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self._now_iso(),
            "level": "WARN" if record.levelname == "WARNING" else record.levelname,
            "module": record.name,
            "event": record.getMessage(),
            "data": getattr(record, "data", {}),
            "cost_usd": getattr(record, "cost_usd", 0.0),
        }
        return json.dumps(entry, ensure_ascii=False, default=str)

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO8601 string."""
        return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(module: str, log_dir: str | None = None) -> logging.Logger:
    """Create or retrieve a module-level logger with JSON formatting.

    Args:
        module: One of ``screening`` | ``refining`` | ``backtesting`` | ``evolution``.
        log_dir: Directory for rotated log files.  When *None*, the path is
            derived from ``Settings().alphascreener_home / "logs"``.

    Returns:
        A configured :class:`logging.Logger` with a JSON :class:`StreamHandler`
        and a :class:`~logging.handlers.TimedRotatingFileHandler` (30-day retention).
    """
    if module not in VALID_MODULES:
        raise ValueError(f"Unknown module: {module!r}. Valid modules: {sorted(VALID_MODULES)}")

    logger = logging.getLogger(module)

    # Guard: handlers are only attached once per logger.
    if getattr(logger, "_alphascreener_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = JsonFormatter()

    # File handler with 30-day rotation (CLI mode: no stdout — all output via rich)
    if log_dir is None:
        settings = Settings()
        log_dir = str(settings.alphascreener_home / "logs")
    log_path = _ensure_log_directory(log_dir)

    file_handler = TimedRotatingFileHandler(
        filename=str(log_path / f"{module}.log"),
        when="D",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._alphascreener_configured = True
    return logger
