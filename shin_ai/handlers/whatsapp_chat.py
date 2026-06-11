import asyncio
from typing import Any

from shin_ai.core import state
from shin_ai.config import DEBUG, WHATSAPP_ENABLED
from shin_ai.core.handler import process_message
from shin_ai.handlers.common import should_record_context, should_respond_to_message
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger

whatsapp_platform = None

if WHATSAPP_ENABLED:
    try:
        from shin_ai.platforms.whatsapp import MessageEventType, WhatsAppPlatform

        whatsapp_platform = WhatsAppPlatform("shin_ai_whatsapp")

        async def _handle_whatsapp_message(event_msg: MessageEventType) -> None:
            unified_msg = await whatsapp_platform.ingest_event_message(event_msg)

            if should_record_context(unified_msg):
                add_message_to_context(unified_msg)

            def _whatsapp_debug(reason: str, text: str) -> None:
                if not DEBUG:
                    return
                logger.info(
                    f"[WhatsAppFilter] chat={unified_msg.chat.id} user={unified_msg.from_user.id if unified_msg.from_user else 'unknown'} "
                    f"reason={reason} text='{(text or '<no text>').replace(chr(10), ' ')[:80]}'"
                )

            should_respond = await should_respond_to_message(unified_msg, debug_hook=_whatsapp_debug)

            if should_respond and not state.IS_CHECKING_KEYS:
                await process_message(whatsapp_platform, unified_msg)

        @whatsapp_platform.client.event(MessageEventType)
        def on_whatsapp_message(_, event_msg: MessageEventType) -> None:
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

    except Exception as e:
        logger.error(f"Failed to initialize WhatsApp handler: {e}")
        whatsapp_platform = None
else:
    logger.info("WhatsApp handler is disabled by configuration.")
