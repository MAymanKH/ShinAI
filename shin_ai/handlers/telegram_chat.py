from pyrogram import Client, filters
from pyrogram.types import Message

from shin_ai.core.client import app
from shin_ai.platforms.telegram import TelegramPlatform
from shin_ai.core.handler import process_message
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger
from shin_ai.handlers.common import (
    is_supported_chat,
    should_record_context,
    should_respond_to_message,
)
from shin_ai.core import state
from shin_ai.config import DEBUG, TELEGRAM_CONFIGURED, TELEGRAM_ENABLED

# Single instance definition for the platform wrapper
telegram_platform = None

if TELEGRAM_ENABLED and TELEGRAM_CONFIGURED:
    telegram_platform = TelegramPlatform(app)
    logger.info("Telegram handlers registered.")

    @app.on_message(filters.incoming, group=-1)
    async def context_recorder(client: Client, msg: Message):
        """
        Records messages in the short-term rolling buffer.
        Runs in group -1 to execute before the main handler.
        """
        try:
            unified_msg = telegram_platform.to_unified_message(msg)
            if should_record_context(unified_msg):
                add_message_to_context(unified_msg)
        except Exception as e:
            logger.error(f"Context recorder failed: {e}")

    def _telegram_debug(reason: str, text: str, msg: Message) -> None:
        def _debug(reason: str) -> None:
            if DEBUG:
                chat_id = getattr(msg.chat, "id", "unknown")
                user_id = getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown"
                text_preview = (text or "<no text>").replace("\n", " ")[:80]
                logger.debug(
                    "[TelegramFilter] chat=%s user=%s reason=%s text='%s'",
                    chat_id, user_id, reason, text_preview,
                )

        _debug(reason)

    @app.on_message(filters.incoming)
    async def yalbot(client: Client, msg: Message):
        """Main message handler translating Pyrogram out to unified layer."""
        try:
            unified_msg = telegram_platform.to_unified_message(msg)

            if not is_supported_chat(unified_msg):
                return

            if DEBUG:
                chat_id = getattr(msg.chat, "id", "unknown")
                user_id = getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown"
                logger.debug(
                    "[TelegramRecv] chat=%s type=%s user=%s",
                    chat_id, str(unified_msg.chat.type).lower(), user_id,
                )

            should_respond = await should_respond_to_message(
                unified_msg,
                debug_hook=lambda reason, text: _telegram_debug(reason, text, msg),
            )
        except Exception as e:
            logger.error("Telegram filter evaluation failed: %s", e, exc_info=True)
            return

        if not should_respond:
            return

        if state.IS_CHECKING_KEYS:
            logger.debug("[TelegramHandler] Skipping message — IS_CHECKING_KEYS is active")
            return

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
