import logging
import sys
from contextvars import ContextVar

import structlog

from app.core.config import settings

# Context variable для сквозной трассировки (пробрасывается через Celery/async)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
channel_var: ContextVar[str | None] = ContextVar("channel", default=None)


def _add_context(logger, method_name, event_dict):
    """Добавление контекстных переменных в каждый лог."""
    if trace_id := trace_id_var.get():
        event_dict["trace_id"] = trace_id
    if channel := channel_var.get():
        event_dict["channel"] = channel
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """Инициализация структурированного логирования."""
    log_level = logging.DEBUG if settings.app_debug else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True, sort_keys=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("linguabridge")
