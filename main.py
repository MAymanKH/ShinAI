from shin_ai.bot import app
from shin_ai.utils.logger_config import logger
from shin_ai.services.social import index_social_context

if __name__ == "__main__":
    # Initialize the social context database
    try: index_social_context()
    except Exception as e: logger.error(f"Failed to index social context: {e}")

    logger.info("ShinAI Started Successfully.")
    app.run()
