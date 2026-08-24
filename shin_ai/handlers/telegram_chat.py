from pyrogram import Client, filters
from pyrogram.types import Message

from shin_ai.core.handler import process_message
from shin_ai.handlers.common import (
    is_supported_chat,
    should_record_context,
    should_respond_to_message,
)
from shin_ai.platforms.telegram import TelegramPlatform
from shin_ai.settings import get_settings
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger


def register(client) -> TelegramPlatform | None:
    """Attach Telegram handlers to ``client`` and return its platform adapter.

    Returns None when Telegram is disabled or its credentials are incomplete,
    which is how the caller learns there is no platform to start.
    """
    platform_settings = get_settings().platform
    if not platform_settings.telegram_enabled:
        logger.info("Telegram handlers are disabled by configuration.")
        return None
    if not platform_settings.telegram_configured:
        logger.warning("Telegram handlers were not registered because Telegram credentials are incomplete.")
        return None

    telegram_platform = TelegramPlatform(client)
    debug = get_settings().debug

    @client.on_message(filters.incoming, group=-1)
    async def context_recorder(_client: Client, msg: Message):
        """Record messages in the short-term buffer.

        Runs in group -1 so it executes before the main handler.
        """
        try:
            unified_msg = telegram_platform.to_unified_message(msg)
            if should_record_context(unified_msg):
                add_message_to_context(unified_msg)
        except Exception as e:
            logger.error("Context recorder failed: %s", e)

    def _telegram_debug(reason: str, text: str, msg: Message) -> None:
        if not debug:
            return
        logger.debug(
            "[TelegramFilter] chat=%s user=%s reason=%s text='%s'",
            getattr(msg.chat, "id", "unknown"),
            getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown",
            reason,
            (text or "<no text>").replace("\n", " ")[:80],
        )

    @client.on_message(filters.incoming)
    async def yalbot(_client: Client, msg: Message):
        """Main message handler translating Pyrogram out to the unified layer."""
        try:
            unified_msg = telegram_platform.to_unified_message(msg)

            if not is_supported_chat(unified_msg):
                return

            if debug:
                logger.debug(
                    "[TelegramRecv] chat=%s type=%s user=%s",
                    getattr(msg.chat, "id", "unknown"),
                    str(unified_msg.chat.type).lower(),
                    getattr(msg.from_user, "id", "unknown") if msg.from_user else "unknown",
                )

            should_respond = await should_respond_to_message(
                unified_msg,
                coordination_scope=telegram_platform.coordination_scope,
                debug_hook=lambda reason, text: _telegram_debug(reason, text, msg),
            )
        except Exception as e:
            logger.error("Telegram filter evaluation failed: %s", e, exc_info=True)
            return

        if not should_respond:
            return

        try:
            await process_message(telegram_platform, unified_msg)
        except Exception as e:
            logger.error("Telegram process_message failed: %s", e)

    logger.info("Telegram handlers registered.")
    return telegram_platform
