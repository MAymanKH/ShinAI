import discord
from shin_ai.platforms.discord import DiscordPlatform
from shin_ai.core.handler import process_message
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger
from shin_ai.handlers.common import should_record_context, should_respond_to_message
from shin_ai.core import state
from shin_ai.config import DEBUG, DISCORD_BOT_TOKEN, DISCORD_CONFIGURED, DISCORD_ENABLED

# Initialize Discord Platform
discord_platform = None
if DISCORD_ENABLED and DISCORD_CONFIGURED:
    discord_platform = DiscordPlatform(DISCORD_BOT_TOKEN)

    @discord_platform.client.event
    async def on_message(message: discord.Message):
        if message.author == discord_platform.client.user:
            return

        unified_msg = discord_platform.to_unified_message(message)

        if should_record_context(unified_msg):
            add_message_to_context(unified_msg)

        def _discord_debug(reason: str, text: str) -> None:
            if not DEBUG:
                return
            logger.info(
                f"[DiscordFilter] chat={unified_msg.chat.id} user={unified_msg.from_user.id if unified_msg.from_user else 'unknown'} "
                f"reason={reason} text='{(text or '<no text>').replace(chr(10), ' ')[:80]}'"
            )

        try:
            should_respond = await should_respond_to_message(unified_msg, debug_hook=_discord_debug)
        except Exception as e:
            logger.error(f"Discord filter evaluation failed: {e}")
            return

        if not should_respond or state.IS_CHECKING_KEYS:
            return

        try:
            await process_message(discord_platform, unified_msg)
        except Exception as e:
            logger.error(f"Error processing Discord message: {e}")
elif DISCORD_ENABLED and not DISCORD_CONFIGURED:
    logger.warning("Discord is enabled but DISCORD_BOT_TOKEN is missing; Discord handler is disabled.")
else:
    logger.info("Discord handler is disabled by configuration.")

