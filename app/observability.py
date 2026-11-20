"""Structured JSON logs, request IDs and the API's single error format.

Every log line is one JSON object: timestamp, level, logger, message, request_id and,
where relevant, model_version (plus method, path, status_code, duration_ms for the
request log). Every response carries an X-Request-ID header: the incoming one if it is
a sane value, otherwise a new one.

Every 4xx/5xx response body has the same shape:
    {"code": "...", "message": "...", "request_id": "...", ...optional extra fields}
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Extra attributes copied into the JSON line when a log call passes them (extra={...})
EXTRA_FIELDS = (
    "model_version",
    "previous_version",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "event",
    "reason",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key in EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Send the app's and uvicorn's logs to stdout as JSON lines.

    uvicorn's own access log is switched off: the request middleware writes one JSON
    line per request instead (with request_id and model_version).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers, access.propagate, access.disabled = [], False, True


def resolve_request_id(incoming: str | None) -> str:
    """The client's X-Request-ID if it is a safe token, otherwise a new UUID (hex)."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


def error_body(code: str, message: str, request_id: str | None, **extra) -> dict:
    """The one JSON error format of the API."""
    return {"code": code, "message": message, "request_id": request_id, **extra}
