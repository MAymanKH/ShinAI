from shin_ai.core.client import app
from shin_ai.utils.logger_config import logger

# Import handlers to register them
for module_name in (
    "shin_ai.handlers.stats",
    "shin_ai.handlers.analytics",
    "shin_ai.handlers.chat",
):
    try:
        __import__(module_name)
        logger.info(f"Loaded handler module: {module_name}")
    except Exception as e:
        logger.error(f"Failed to load handler module {module_name}: {e}")

__all__ = ["app"]
