import random
import discord
from shin_ai.platforms.discord import DiscordPlatform
from shin_ai.core.handler import process_message
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger
from shin_ai.services.replies import check_reply_chain
from shin_ai.core import state
from shin_ai.config import DISCORD_TOKEN

# Initialize Discord Platform
discord_platform = None
if DISCORD_TOKEN:
    discord_platform = DiscordPlatform(DISCORD_TOKEN)

    @discord_platform.client.event
    async def on_message(message: discord.Message):
        if message.author == discord_platform.client.user:
            return

        unified_msg = discord_platform.to_unified_message(message)

        # Context recording
        add_message_to_context(unified_msg)

        # Filter logic (similar to Telegram)
        should_respond = False
        text = message.content or ""
        
        # DMs
        if isinstance(message.channel, discord.DMChannel):
            if not text.startswith('/'):
                should_respond = True
        
        # Media catch in reply chains
        if not text and message.attachments:
            if await check_reply_chain(unified_msg):
                should_respond = True
                
        # Direct Mention by text
        if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
            should_respond = True
            
        # Mention ping
        if discord_platform.client.user in message.mentions:
            should_respond = True
            
        # Reply chain check
        if not should_respond and await check_reply_chain(unified_msg):
            should_respond = True
            
        # 5% chance
        if not should_respond and random.random() < 0.05:
            should_respond = True
            
        if should_respond and not state.IS_CHECKING_KEYS:
            try:
                await process_message(discord_platform, unified_msg)
            except Exception as e:
                logger.error(f"Error processing Discord message: {e}")

