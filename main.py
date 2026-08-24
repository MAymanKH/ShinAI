import asyncio

from pyrogram import idle

from shin_ai.bot import load_handlers
from shin_ai.config import DEBUG
from shin_ai.core.lifecycle import shutdown_application
from shin_ai.handlers.discord_chat import discord_platform
from shin_ai.handlers.telegram_chat import telegram_platform
from shin_ai.handlers.whatsapp_chat import whatsapp_platform
from shin_ai.services.social import index_social_context
from shin_ai.utils.logger_config import logger, reconfigure_logger

# Apply debug: true/false from config.yaml to the logger level
reconfigure_logger(DEBUG)
load_handlers()


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
            logger.info(
                "%s platform is disabled or unavailable.",
                platform_label,
                extra={"event_name": "platform.disabled"},
            )
            continue

        logger.info(
            "Starting %s platform...",
            platform_label,
            extra={"event_name": "platform.starting"},
        )
        try:
            await platform.start()
            active_platforms.append((platform_label, platform))
        except Exception as e:
            logger.error(f"Failed to start {platform_label} platform: {e}")

    logger.info(
        "ShinAI started successfully; listening for messages",
        extra={"event_name": "lifecycle.ready"},
    )
    if not active_platforms:
        logger.warning(
            "No chat platforms are active. Configure TELEGRAM_ENABLED, "
            "DISCORD_ENABLED, WHATSAPP_ENABLED and credentials."
        )

    try:
        await idle()
    finally:
        await shutdown_application(active_platforms)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
