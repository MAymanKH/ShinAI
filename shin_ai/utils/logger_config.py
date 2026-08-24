"""Human-readable, correlated application logging."""

from __future__ import annotations

import contextvars
import logging
import multiprocessing
import sys
import warnings
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from shin_ai.config import DEBUG, LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES

warnings.filterwarnings("ignore", message=".*automatic function calling.*")

_log_context: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "shinai_log_context",
    default=None,
)


@contextmanager
def bind_log_context(**fields):
    """Attach stable identifiers to all logs emitted in the current task."""
    current = _log_context.get() or {}
    merged = {**current, **{key: str(value) for key, value in fields.items() if value is not None}}
    token = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)


class ApplicationLogFilter(logging.Filter):
    """Add safe formatter fields and suppress unhelpful SDK warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if (
            "automatic function calling" in message
            or "AFC is disabled" in message
            or "AFC will be disabled" in message
        ):
            return False

        context = _log_context.get() or {}
        record.event_name = getattr(record, "event_name", "-")
        ordered = (
            ("rid", context.get("interaction_id")),
            ("platform", context.get("platform")),
            ("chat", context.get("chat_id")),
            ("msg", context.get("message_id")),
            ("user", context.get("user_id")),
        )
        record.log_context = " ".join(
            f"{key}={value}" for key, value in ordered if value not in (None, "")
        )
        if record.log_context:
            record.log_context += " | "
        return True


class ApplicationLogger(logging.Logger):
    """Preserve tracebacks for errors logged from an active exception block."""

    def error(self, msg, *args, **kwargs) -> None:
        if "exc_info" not in kwargs and sys.exc_info()[0] is not None:
            kwargs["exc_info"] = True
        super().error(msg, *args, **kwargs)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if record.levelno >= logging.WARNING:
            rendered += f" | source={record.module}:{record.funcName}:{record.lineno}"
        return rendered


_CONSOLE_FORMATTER = ConsoleFormatter(
    "%(asctime)s | %(levelname)-7s | %(event_name)-22s | %(log_context)s%(message)s",
    datefmt="%H:%M:%S",
)
_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(event_name)-22s | "
    "%(module)s:%(funcName)s:%(lineno)d | %(log_context)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    name: str = "shin_ai",
    *,
    log_file: Path | str | None = LOG_FILE,
    debug: bool = DEBUG,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Create one console handler and an optional rotating file handler."""
    previous_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(ApplicationLogger)
    try:
        application_logger = logging.getLogger(name)
    finally:
        logging.setLoggerClass(previous_logger_class)
    application_logger.propagate = False
    application_logger.handlers.clear()
    level = logging.DEBUG if debug else logging.INFO
    application_logger.setLevel(level)

    log_filter = ApplicationLogFilter()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_CONSOLE_FORMATTER)
    console.addFilter(log_filter)
    application_logger.addHandler(console)

    # A spawned Whisper worker must not rotate the same file as its parent.
    is_main_process = multiprocessing.current_process().name == "MainProcess"
    if log_file is not None and is_main_process:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_FILE_FORMATTER)
        file_handler.addFilter(log_filter)
        application_logger.addHandler(file_handler)

    return application_logger


_THIRD_PARTY_LOGGERS = (
    "httpx", "httpcore", "hpack",
    "pyrogram", "pyrogram.connection.connection", "pyrogram.session.session", "pyrogram.dispatcher",
    "discord", "discord.client", "discord.gateway", "discord.http",
    "whatsmeow", "whatsmeow.Client",
    "google_genai", "google_genai.models", "google.genai", "google.genai.models",
    "google.ai.generativelanguage", "google.api_core",
    "sentence_transformers",
)


def reconfigure_logger(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    third_party_level = logging.INFO if debug else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    if debug:
        logger.debug(
            "Debug logging enabled",
            extra={"event_name": "lifecycle.logging"},
        )


logger = setup_logger()

for _name in _THIRD_PARTY_LOGGERS:
    logging.getLogger(_name).setLevel(logging.INFO if DEBUG else logging.WARNING)

_root_filter = ApplicationLogFilter()
logging.getLogger().addFilter(_root_filter)
for _name in ("google_genai", "google_genai.models", "google.genai", "google.genai.models"):
    logging.getLogger(_name).addFilter(_root_filter)
