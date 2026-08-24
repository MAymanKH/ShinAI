from pyrogram import Client, filters
from pyrogram.types import Message

from shin_ai.providers.gemini import get_gemini_stats_message
from shin_ai.settings import get_settings
from shin_ai.utils.rate_limit import check_gstats_rate_limit


def register(client) -> None:
    """Attach the Gemini health commands to the Telegram client."""

    @client.on_message(filters.command("gstats"))
    async def stats_command(client: Client, msg: Message):
        """Display Gemini API key statistics."""
        if not msg.from_user:
            return
        # Rate limit check (admin is exempt)
        wait_time = check_gstats_rate_limit(msg.from_user.id)
        if wait_time > 0:
            return await msg.reply_text(
                f"⏳ Please wait {wait_time // 60}m {wait_time % 60}s before checking stats again."
            )

        status_msg = await msg.reply_text("Reading shared Gemini health...")
        stats_msg = await get_gemini_stats_message(detailed=False)
        await status_msg.edit_text(stats_msg)

    @client.on_message(filters.command("gstats_details"))
    async def stats_details_command(client: Client, msg: Message):
        """Display detailed Gemini API key statistics (admin only)."""
        if not msg.from_user or msg.from_user.id != get_settings().admin_user_id:
            return

        status_msg = await msg.reply_text("Reading shared Gemini health...")
        stats_msg = await get_gemini_stats_message(detailed=True)
        await status_msg.edit_text(stats_msg)
