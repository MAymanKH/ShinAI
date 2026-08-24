from pyrogram import Client, StopPropagation, filters
from pyrogram.types import Message

from shin_ai.handlers.common import COMMAND_HANDLER_GROUP
from shin_ai.providers.gemini import get_gemini_stats_message
from shin_ai.settings import get_settings
from shin_ai.utils.rate_limit import check_gstats_rate_limit


async def _reply_with_stats(msg: Message, *, detailed: bool) -> None:
    status_msg = await msg.reply_text("Reading shared Gemini health...")
    stats_msg = await get_gemini_stats_message(detailed=detailed)
    await status_msg.edit_text(stats_msg)


def register(client) -> None:
    """Attach the Gemini health commands to the Telegram client."""

    @client.on_message(filters.command("gstats"), group=COMMAND_HANDLER_GROUP)
    async def stats_command(client: Client, msg: Message):
        """Display Gemini API key statistics."""
        if msg.from_user:
            # Rate limit check (admin is exempt)
            wait_time = check_gstats_rate_limit(msg.from_user.id)
            if wait_time > 0:
                await msg.reply_text(
                    f"⏳ Please wait {wait_time // 60}m {wait_time % 60}s before checking stats again."
                )
            else:
                await _reply_with_stats(msg, detailed=False)
        # A command is never conversation, so it stops here even when ignored.
        raise StopPropagation

    @client.on_message(filters.command("gstats_details"), group=COMMAND_HANDLER_GROUP)
    async def stats_details_command(client: Client, msg: Message):
        """Display detailed Gemini API key statistics (admin only)."""
        if msg.from_user and msg.from_user.id == get_settings().admin_user_id:
            await _reply_with_stats(msg, detailed=True)
        raise StopPropagation
