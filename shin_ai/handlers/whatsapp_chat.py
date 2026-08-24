import asyncio
from typing import Any

from shin_ai.core.handler import process_message
from shin_ai.handlers.common import should_record_context, should_respond_to_message
from shin_ai.settings import get_settings
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger


def register():
    """Build the WhatsApp platform and attach its handlers.

    Returns None when WhatsApp is disabled or the native client cannot be
    constructed; neonize failures should not stop the other platforms.
    """
    if not get_settings().platform.whatsapp_enabled:
        logger.info("WhatsApp handler is disabled by configuration.")
        return None

    try:
        from shin_ai.platforms.whatsapp import MessageEventType, WhatsAppPlatform

        whatsapp_platform = WhatsAppPlatform("shin_ai_whatsapp")
    except Exception as e:
        logger.error("Failed to initialize WhatsApp handler: %s", e)
        return None

    debug = get_settings().debug

    async def _handle_whatsapp_message(event_msg: MessageEventType) -> None:
        unified_msg = await whatsapp_platform.ingest_event_message(event_msg)

        if should_record_context(unified_msg):
            add_message_to_context(unified_msg)

        def _whatsapp_debug(reason: str, text: str) -> None:
            if not debug:
                return
            logger.debug(
                "[WhatsAppFilter] chat=%s user=%s reason=%s text='%s'",
                unified_msg.chat.id,
                unified_msg.from_user.id if unified_msg.from_user else "unknown",
                reason,
                (text or "<no text>").replace("\n", " ")[:80],
            )

        should_respond = await should_respond_to_message(
            unified_msg,
            coordination_scope=whatsapp_platform.coordination_scope,
            debug_hook=_whatsapp_debug,
        )

        if should_respond:
            await process_message(whatsapp_platform, unified_msg)

    @whatsapp_platform.client.event(MessageEventType)
    def on_whatsapp_message(_, event_msg: MessageEventType) -> None:
        # neonize dispatches from its own Go-backed thread, so work has to be
        # handed back to the application loop rather than awaited here.
        loop = whatsapp_platform.event_loop
        if loop is None:
            logger.warning("Skipping WhatsApp message because platform loop is not ready yet.")
            return

        future = asyncio.run_coroutine_threadsafe(_handle_whatsapp_message(event_msg), loop)

        def _log_failure(done_future: Any) -> None:
            try:
                done_future.result()
            except Exception as e:
                logger.error("Error processing WhatsApp message: %s", e, exc_info=True)

        future.add_done_callback(_log_failure)

    logger.info("WhatsApp handlers registered.")
    return whatsapp_platform
