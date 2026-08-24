import importlib

from shin_ai.core.client import app
from shin_ai.utils.logger_config import logger

_HANDLER_MODULES = (
    "shin_ai.handlers.stats",
    "shin_ai.handlers.analytics",
    "shin_ai.handlers.telegram_chat",
    "shin_ai.handlers.discord_chat",
    "shin_ai.handlers.whatsapp_chat",
)


def load_handlers() -> None:
    """Import handler modules explicitly so their decorators are registered."""
    try:
        for module_name in _HANDLER_MODULES:
            importlib.import_module(module_name)
    except Exception as e:
        logger.error("Failed to load handlers: %s", e)
        return

    logger.info("Handlers loaded successfully.")


__all__ = ["app", "load_handlers"]
