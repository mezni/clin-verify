"""Structured JSON logging for the ClinVerify service.

Emits one JSON object per log record so request telemetry can be correlated by
``request_id`` in log aggregators. See ``docs/sdd.md`` §6.1.
"""

from __future__ import annotations

import json
import logging
import sys

REQUEST_LOGGER = "clinverify.request"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON handler on the ``clinverify`` logger (idempotent)."""

    logger = logging.getLogger("clinverify")
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and isinstance(
            handler.formatter, JsonFormatter
        ):
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def log_event(logger: logging.Logger, message: str, **fields: object) -> None:
    """Log ``message`` with structured ``fields`` merged into the JSON payload."""

    logger.info(message, extra={"extra_fields": fields})
