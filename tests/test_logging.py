"""Unit tests for structured JSON logging."""

from __future__ import annotations

import json
import logging

from clinverify.logging import JsonFormatter, configure_logging


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="clinverify.request",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )


def test_json_formatter_merges_extra_fields() -> None:
    record = _record()
    record.extra_fields = {"request_id": "abc", "status_code": 200}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "request"
    assert payload["logger"] == "clinverify.request"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "abc"
    assert payload["status_code"] == 200


def test_json_formatter_handles_missing_extra() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["message"] == "request"
    assert "request_id" not in payload


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()

    logger = logging.getLogger("clinverify")
    handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and isinstance(handler.formatter, JsonFormatter)
    ]
    assert len(handlers) == 1
