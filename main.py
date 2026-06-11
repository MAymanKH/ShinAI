import asyncio
from pyrogram import idle
import shin_ai.bot
from shin_ai.utils.logger_config import logger, reconfigure_logger
from shin_ai.config import DEBUG
from shin_ai.services.social import index_social_context
from shin_ai.handlers.telegram_chat import telegram_platform
from shin_ai.handlers.discord_chat import discord_platform
from shin_ai.handlers.whatsapp_chat import whatsapp_platform

# Apply debug: true/false from config.yaml to the logger level
reconfigure_logger(DEBUG)

async def main():
    # Initialize the social context database
    try: 
        index_social_context()
    except Exception as e: 
        logger.error(f"Failed to index social context: {e}")

    active_platforms = []
    configured_platforms = [
        ("Telegram", telegram_platform),
        ("Discord", discord_platform),
        ("WhatsApp", whatsapp_platform),
    ]

    for platform_label, platform in configured_platforms:
        if platform is None:
            logger.info(f"{platform_label} platform is disabled or unavailable.")
            continue

        logger.info(f"Starting {platform_label} Platform...")
        try:
            await platform.start()
            active_platforms.append((platform_label, platform))
        except Exception as e:
            logger.error(f"Failed to start {platform_label} platform: {e}")

    logger.info("ShinAI Started Successfully. Listening for messages...")
    if not active_platforms:
        logger.warning("No chat platforms are active. Configure TELEGRAM_ENABLED, DISCORD_ENABLED, WHATSAPP_ENABLED and credentials.")
    
    # Wait until interrupted
    await idle()
    
    logger.info("Stopping platforms...")
    for platform_label, platform in reversed(active_platforms):
        try:
            await platform.stop()
        except Exception as e:
            logger.error(f"Failed to stop {platform_label} platform cleanly: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
