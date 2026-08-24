"""Human-readable, correlated application logging."""

from __future__ import annotations

import contextvars
import logging
import multiprocessing
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

from concurrent_log_handler import ConcurrentRotatingFileHandler

from shin_ai.config import DEBUG, LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

_SUPPRESSED_SDK_MESSAGES = (
    "automatic function calling",
    "AFC is disabled",
    "AFC will be disabled",
    "unauthenticated requests to the HF Hub",
)

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

    def __init__(self, *, compact_context: bool = False) -> None:
        super().__init__()
        self.compact_context = compact_context

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if any(fragment in message for fragment in _SUPPRESSED_SDK_MESSAGES):
            return False

        context = _log_context.get() or {}
        record.event_name = getattr(record, "event_name", None) or (
            f"{record.module}.{record.levelname.lower()}"
        )
        if self.compact_context:
            ordered = (
                ("rid", context.get("interaction_id")),
                ("platform", context.get("platform")),
            )
        else:
            ordered = (
                ("rid", context.get("interaction_id")),
                ("platform", context.get("platform")),
                ("chat", context.get("chat_id")),
                ("msg", context.get("message_id")),
                ("user", context.get("user_id")),
            )
        record.log_context = " ".join(f"{key}={value}" for key, value in ordered if value not in (None, ""))
        if record.log_context:
            record.log_context += " | "
        return True


class ThirdPartyNoiseFilter(logging.Filter):
    """Drop repeated SDK notices that do not help operate or debug the bot."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in _SUPPRESSED_SDK_MESSAGES)


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

    console_filter = ApplicationLogFilter(compact_context=not debug)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_CONSOLE_FORMATTER)
    console.addFilter(console_filter)
    application_logger.addHandler(console)

    # A spawned Whisper worker must not rotate the same file as its parent.
    is_main_process = multiprocessing.current_process().name == "MainProcess"
    if log_file is not None and is_main_process:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = ConcurrentRotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_FILE_FORMATTER)
        file_handler.addFilter(ApplicationLogFilter())
        application_logger.addHandler(file_handler)

    return application_logger


_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "hpack",
    "pyrogram",
    "pyrogram.connection.connection",
    "pyrogram.session.session",
    "pyrogram.dispatcher",
    "discord",
    "discord.client",
    "discord.gateway",
    "discord.http",
    "whatsmeow",
    "whatsmeow.Client",
    "google_genai",
    "google_genai.models",
    "google.genai",
    "google.genai.models",
    "google.ai.generativelanguage",
    "google.api_core",
    "sentence_transformers",
    "huggingface_hub",
    "huggingface_hub.utils._http",
)

_sdk_noise_filter = ThirdPartyNoiseFilter()


def _configure_third_party_loggers(debug: bool) -> None:
    level = logging.INFO if debug else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        sdk_logger = logging.getLogger(name)
        sdk_logger.setLevel(level)
        if _sdk_noise_filter not in sdk_logger.filters:
            sdk_logger.addFilter(_sdk_noise_filter)


def reconfigure_logger(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    _configure_third_party_loggers(debug)

    if debug:
        logger.debug(
            "Debug logging enabled",
            extra={"event_name": "lifecycle.logging"},
        )


logger = setup_logger()

_configure_third_party_loggers(DEBUG)

_root_filter = ApplicationLogFilter()
logging.getLogger().addFilter(_root_filter)
for _name in ("google_genai", "google_genai.models", "google.genai", "google.genai.models"):
    logging.getLogger(_name).addFilter(_root_filter)
