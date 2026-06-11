import logging
import sys


def setup_logger(name: str = "ShinAI", log_file: str = "shinai_bot.log", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
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

    return logger


def reconfigure_logger(debug: bool = False) -> None:
    """
    Apply the DEBUG flag from config.yaml to the root ShinAI logger.

    Call this once from main.py after the config has been loaded:

        from shin_ai.utils.logger_config import reconfigure_logger
        from shin_ai.config import DEBUG
        reconfigure_logger(DEBUG)

    When debug=True the logger drops to DEBUG level, exposing all
    logger.debug() call sites (time detection decisions, per-message
    routing, memory skip reasons, etc.).
    """
    level = logging.DEBUG if debug else logging.INFO
    log = logging.getLogger("ShinAI")
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)
    if debug:
        log.debug("Logger reconfigured to DEBUG level (debug: true in config.yaml)")


# Module-level singleton — defaults to INFO until reconfigure_logger() is called.
logger = setup_logger()

# Silence chatty third-party loggers that spam at INFO level.
# httpx logs every single HTTP request; pyrogram internals are noise at INFO.
for _noisy in ("httpx", "httpcore", "hpack"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
