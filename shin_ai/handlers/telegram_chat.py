import random
from pyrogram import Client, filters, enums
from pyrogram.types import Message

from shin_ai.core.client import app
from shin_ai.platforms.telegram import TelegramPlatform
from shin_ai.core.handler import process_message
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger
from shin_ai.services.replies import check_reply_chain
from shin_ai.core import state

# Single instance definition for the platform wrapper
telegram_platform = TelegramPlatform(app)

@app.on_message(filters.group | filters.private, group=-1)
async def context_recorder(client: Client, msg: Message):
    """
    Records messages in the short-term rolling buffer.
    Runs in group -1 to execute before the main handler.
    """
    try:
        unified_msg = telegram_platform.to_unified_message(msg)
        add_message_to_context(unified_msg)
    except Exception as e:
        logger.error(f"Context recorder failed: {e}")

async def yalbot_filter_func(_, client: Client, msg: Message) -> bool:
    text = msg.text or msg.caption
    
    if msg.from_user and msg.from_user.is_self:
        return False
    
    if msg.chat.type == enums.ChatType.PRIVATE:
        if text and text.startswith('/'):
            return False
        return True
    
    if not text:
        if not (msg.photo or msg.sticker):
            return False
        unified_msg = telegram_platform.to_unified_message(msg)
        if await check_reply_chain(unified_msg):
            return True
        return False
    
    if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
        return True
    
    if getattr(msg, "mentioned", False):
        return True
    
    unified_msg = telegram_platform.to_unified_message(msg)
    if await check_reply_chain(unified_msg):
        return True
    
    # 5% chance
    if random.random() < 0.05:
        return True
    
    return False

yalbot_filter = filters.create(yalbot_filter_func)

@app.on_message(yalbot_filter)
async def yalbot(client: Client, msg: Message):
    """Main message handler translating Pyrogram out to unified layer."""
    if state.IS_CHECKING_KEYS:
        return
    unified_msg = telegram_platform.to_unified_message(msg)
    await process_message(telegram_platform, unified_msg)
