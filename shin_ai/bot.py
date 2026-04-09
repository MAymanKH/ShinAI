from shin_ai.core.client import app
from shin_ai.utils.logger_config import logger

# Import handlers to register them
try:
    import shin_ai.handlers.stats
    import shin_ai.handlers.analytics
    import shin_ai.handlers.telegram_chat
    import shin_ai.handlers.discord_chat
    import shin_ai.handlers.whatsapp_chat
    logger.info("Handlers loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load handlers: {e}")

__all__ = ["app"]
