import logging
import sys
import warnings

# Suppress google_genai SDK warnings about automatic function calling (AFC) compatibility
warnings.filterwarnings("ignore", message=".*automatic function calling.*")


class AFCWarningFilter(logging.Filter):
    """Filters out annoying Gemini SDK automatic function calling (AFC) warnings."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "automatic function calling" in msg or "AFC is disabled" in msg or "AFC will be disabled" in msg:
            return False
        return True


def setup_logger(name: str = "ShinAI", log_file: str = "shinai_bot.log", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent propagation to the root logger.
    # neonize (the WhatsApp Go library) bridges Python's root logger to its own
    # Go stderr logger, which causes every line to appear twice in PM2 logs.
    logger.propagate = False

    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file, encoding="utf-8")

        # Console: concise — HH:MM:SS [LEVEL   ] message
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(message)s",
            datefmt="%H:%M:%S",
        )
        # File: full context for post-mortem analysis
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        c_handler.setFormatter(console_fmt)
        f_handler.setFormatter(file_fmt)

        c_handler.setLevel(level)
        f_handler.setLevel(level)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    # Attach filter to the ShinAI logger
    logger.addFilter(AFCWarningFilter())

    return logger


# Third-party loggers that are too verbose at INFO.
# Silenced to WARNING in production; restored to NOTSET (library default) in debug mode.
_THIRD_PARTY_LOGGERS = (
    # HTTP clients
    "httpx", "httpcore", "hpack",
    # Pyrogram (Telegram client internals)
    "pyrogram",
    "pyrogram.connection.connection",
    "pyrogram.session.session",
    "pyrogram.dispatcher",
    # Discord.py client internals
    "discord",
    "discord.client",
    "discord.gateway",
    "discord.http",
    # WhatsApp Go bridge (neonize pipes whatsmeow Go logs into Python logging)
    "whatsmeow",
    "whatsmeow.Client",
    # Google Gemini SDK (both underscore and dot package variants)
    "google_genai",
    "google_genai.models",
    "google.genai",
    "google.genai.models",
    "google.ai.generativelanguage",
    "google.api_core",
    # sentence-transformers: older versions show tqdm Batches bars when their
    # logger is at INFO level — raising to WARNING suppresses them per-request.
    "sentence_transformers",
)


def reconfigure_logger(debug: bool = False) -> None:
    """
    Apply the DEBUG flag from config.yaml to the root ShinAI logger.

    Call this once from main.py after the config has been loaded:

        from shin_ai.utils.logger_config import reconfigure_logger
        from shin_ai.config import DEBUG
        reconfigure_logger(DEBUG)

    When debug=True:
      - ShinAI logger drops to DEBUG level (all logger.debug() calls visible)
      - Third-party loggers (pyrogram, discord, whatsmeow, httpx) are restored
        to their natural level so you can see connection/session details.

    When debug=False (production):
      - Third-party loggers are silenced to WARNING — only errors/warnings show.
    """
    level = logging.DEBUG if debug else logging.INFO
    log = logging.getLogger("ShinAI")
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)

    # Toggle third-party verbosity based on debug mode
    third_party_level = logging.NOTSET if debug else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    if debug:
        log.debug("Logger reconfigured to DEBUG level (debug: true in config.yaml)")


# Module-level singleton — defaults to INFO until reconfigure_logger() is called.
logger = setup_logger()

# Apply production silence immediately on import (before reconfigure_logger is called).
for _noisy in _THIRD_PARTY_LOGGERS:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Register the AFC warning filter globally to root and Google loggers
_afc_filter = AFCWarningFilter()
logging.getLogger().addFilter(_afc_filter)
for _name in ("google_genai", "google_genai.models", "google.genai", "google.genai.models"):
    logging.getLogger(_name).addFilter(_afc_filter)
