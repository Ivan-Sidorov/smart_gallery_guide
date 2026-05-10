"""JSON-formatted structured logging with request-id propagation."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the request-id bound to the current context, if any."""
    return _request_id_ctx.get()


def bind_request_id(request_id: str) -> object:
    """Bind a request-id to the current context. Returns the reset token."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: object) -> None:
    """Reset request-id binding using the token returned by `bind_request_id`."""
    _request_id_ctx.reset(token)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter that includes `request_id` from contextvars."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            }:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "info") -> None:
    """Configure root logger to pass structured JSON to stdout."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        if getattr(handler, "_smart_guide_api", False):
            return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler._smart_guide_api = True
    root.addHandler(handler)
