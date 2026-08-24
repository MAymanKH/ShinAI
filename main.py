import asyncio
from pyrogram import idle
import shin_ai.bot
from shin_ai.utils.logger_config import logger, reconfigure_logger
from shin_ai.config import DEBUG
from shin_ai.services.social import index_social_context
from shin_ai.services.embeddings import close_embedding_service
from shin_ai.services.audio_transcriber import close_audio_transcriber
from shin_ai.coordination.runtime import close_coordination_store
from shin_ai.core.handler import shutdown_interaction_scheduler
from shin_ai.handlers.telegram_chat import telegram_platform
from shin_ai.handlers.discord_chat import discord_platform
from shin_ai.handlers.whatsapp_chat import whatsapp_platform

# Apply debug: true/false from config.yaml to the logger level
reconfigure_logger(DEBUG)

async def main():
    # Initialize the social context database
    try: 
        await index_social_context()
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
    
    try:
        await idle()
    finally:
        logger.info("Draining interactions...")
        await shutdown_interaction_scheduler()

        logger.info("Stopping platforms...")
        for platform_label, platform in reversed(active_platforms):
            try:
                await platform.stop()
            except Exception as e:
                logger.error(f"Failed to stop {platform_label} platform cleanly: {e}")
        await close_audio_transcriber()
        await close_embedding_service()
        await close_coordination_store()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
