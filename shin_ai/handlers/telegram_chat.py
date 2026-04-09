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
from shin_ai.config import DEBUG, TELEGRAM_CONFIGURED, TELEGRAM_ENABLED

# Single instance definition for the platform wrapper
telegram_platform = None

if TELEGRAM_ENABLED and TELEGRAM_CONFIGURED:
    telegram_platform = TelegramPlatform(app)
    logger.info("Telegram handlers registered.")

    def _is_supported_chat(msg: Message) -> bool:
        chat_type = str(getattr(msg.chat, "type", "")).lower()
        return any(kind in chat_type for kind in ("private", "group", "supergroup"))

    @app.on_message(filters.incoming, group=-1)
    async def context_recorder(client: Client, msg: Message):
        """
        Records messages in the short-term rolling buffer.
        Runs in group -1 to execute before the main handler.
        """
        if not _is_supported_chat(msg):
            return

        try:
            unified_msg = telegram_platform.to_unified_message(msg)
            add_message_to_context(unified_msg)
        except Exception as e:
            logger.error(f"Context recorder failed: {e}")

    async def yalbot_filter_func(_, client: Client, msg: Message) -> bool:
        text = msg.text or msg.caption

        def _debug(reason: str) -> None:
            if DEBUG:
                chat_id = getattr(msg.chat, "id", "unknown")
                user_id = getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown"
                text_preview = (text or "<no text>").replace("\n", " ")[:80]
                logger.info(
                    f"[TelegramFilter] chat={chat_id} user={user_id} reason={reason} text='{text_preview}'"
                )

        if msg.from_user and msg.from_user.is_self:
            _debug("skip:self")
            return False

        chat_type = str(getattr(msg.chat, "type", "")).lower()

        if "private" in chat_type:
            if text and text.startswith('/'):
                _debug("skip:private_command")
                return False
            _debug("pass:private")
            return True

        if not text:
            if not (msg.photo or msg.sticker):
                _debug("skip:no_text_no_media")
                return False
            unified_msg = telegram_platform.to_unified_message(msg)
            if await check_reply_chain(unified_msg):
                _debug("pass:reply_chain_media")
                return True
            _debug("skip:media_without_reply_chain")
            return False

        if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
            _debug("pass:keyword")
            return True

        if getattr(msg, "mentioned", False):
            _debug("pass:mentioned")
            return True

        unified_msg = telegram_platform.to_unified_message(msg)
        if await check_reply_chain(unified_msg):
            _debug("pass:reply_chain")
            return True

        # 5% chance
        if random.random() < 0.05:
            _debug("pass:random")
            return True

        _debug("skip:no_trigger")
        return False

    @app.on_message(filters.incoming)
    async def yalbot(client: Client, msg: Message):
        """Main message handler translating Pyrogram out to unified layer."""
        if not _is_supported_chat(msg):
            return

        if DEBUG:
            chat_type = str(getattr(msg.chat, "type", "")).lower()
            chat_id = getattr(msg.chat, "id", "unknown")
            user_id = getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown"
            logger.info(f"[TelegramRecv] chat={chat_id} type={chat_type} user={user_id}")

        try:
            should_respond = await yalbot_filter_func(None, client, msg)
        except Exception as e:
            logger.error(f"Telegram filter evaluation failed: {e}")
            return

        if not should_respond:
            return

        if state.IS_CHECKING_KEYS:
            if DEBUG:
                logger.info("[TelegramHandler] skip because IS_CHECKING_KEYS is true")
            return
        unified_msg = telegram_platform.to_unified_message(msg)
        try:
            await process_message(telegram_platform, unified_msg)
        except Exception as e:
            logger.error(f"Telegram process_message failed: {e}")
elif TELEGRAM_ENABLED and not TELEGRAM_CONFIGURED:
    logger.warning(
        "Telegram handlers were not registered because Telegram credentials are incomplete."
    )
else:
    logger.info("Telegram handlers are disabled by configuration.")
