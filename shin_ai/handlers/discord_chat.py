import discord

from shin_ai.core.handler import process_message
from shin_ai.handlers.common import should_record_context, should_respond_to_message
from shin_ai.platforms.discord import DiscordPlatform
from shin_ai.settings import get_settings
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger


def register() -> DiscordPlatform | None:
    """Build the Discord platform and attach its handlers.

    Returns None when Discord is disabled or has no bot token.
    """
    platform_settings = get_settings().platform
    if not platform_settings.discord_enabled:
        logger.info("Discord handler is disabled by configuration.")
        return None
    if not platform_settings.discord_configured:
        logger.warning("Discord is enabled but its bot token is missing; Discord handler is disabled.")
        return None

    discord_platform = DiscordPlatform(platform_settings.discord_bot_token)
    debug = get_settings().debug

    @discord_platform.client.event
    async def on_message(message: discord.Message):
        if message.author == discord_platform.client.user:
            return

        unified_msg = discord_platform.to_unified_message(message)

        if should_record_context(unified_msg):
            add_message_to_context(unified_msg)

        def _discord_debug(reason: str, text: str) -> None:
            if not debug:
                return
            logger.debug(
                "[DiscordFilter] chat=%s user=%s reason=%s text='%s'",
                unified_msg.chat.id,
                unified_msg.from_user.id if unified_msg.from_user else "unknown",
                reason,
                (text or "<no text>").replace(chr(10), " ")[:80],
            )

        try:
            should_respond = await should_respond_to_message(
                unified_msg,
                coordination_scope=discord_platform.coordination_scope,
                debug_hook=_discord_debug,
            )
        except Exception as e:
            logger.error("Discord filter evaluation failed: %s", e, exc_info=True)
            return

        if not should_respond:
            return

        try:
            await process_message(discord_platform, unified_msg)
        except Exception as e:
            logger.error("Error processing Discord message: %s", e, exc_info=True)

    logger.info("Discord handlers registered.")
    return discord_platform
