from pyrogram import Client, filters
from pyrogram.types import Message
from shin_ai.config import ADMIN_USER_ID
from shin_ai.providers.gemini import get_gemini_stats_message
from shin_ai.utils.rate_limit import check_gstats_rate_limit
from shin_ai.core.client import app
from shin_ai.core import state


@app.on_message(filters.command("gstats"))
async def stats_command(client: Client, msg: Message):
    """Display Gemini API key statistics."""
    # Rate limit check (admin is exempt)
    wait_time = check_gstats_rate_limit(msg.from_user.id)
    if wait_time > 0:
        return await msg.reply_text(f"⏳ Please wait {wait_time // 60}m {wait_time % 60}s before checking stats again.")
    
    state.IS_CHECKING_KEYS = True
    try:
        status_msg = await msg.reply_text("Checking keys... this may take a moment.")
        stats_msg = await get_gemini_stats_message(detailed=False)
        await status_msg.edit_text(stats_msg)
    finally:
        state.IS_CHECKING_KEYS = False


@app.on_message(filters.command("gstats_details"))
async def stats_details_command(client: Client, msg: Message):
    """Display detailed Gemini API key statistics (admin only)."""
    if msg.from_user.id != ADMIN_USER_ID:
        return
    
    state.IS_CHECKING_KEYS = True
    try:
        status_msg = await msg.reply_text("Checking keys... this may take a moment.")
        stats_msg = await get_gemini_stats_message(detailed=True)
        await status_msg.edit_text(stats_msg)
    finally:
        state.IS_CHECKING_KEYS = False
