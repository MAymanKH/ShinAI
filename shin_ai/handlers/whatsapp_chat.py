import asyncio
import random
from typing import Any

from shin_ai.core import state
from shin_ai.config import WHATSAPP_ENABLED, WHATSAPP_SESSION_NAME
from shin_ai.core.handler import process_message
from shin_ai.services.replies import check_reply_chain
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger

whatsapp_platform = None

if WHATSAPP_ENABLED:
    try:
        from shin_ai.platforms.whatsapp import MessageEventType, WhatsAppPlatform

        whatsapp_platform = WhatsAppPlatform(WHATSAPP_SESSION_NAME)

        async def _handle_whatsapp_message(event_msg: MessageEventType) -> None:
            unified_msg = whatsapp_platform.ingest_event_message(event_msg)

            if not unified_msg.from_user or unified_msg.from_user.is_self:
                return

            # WhatsApp system/status messages should not be fed into the bot pipeline.
            if str(unified_msg.chat.id) == "status@broadcast":
                return

            add_message_to_context(unified_msg)

            should_respond = False
            text = (unified_msg.text or unified_msg.caption or "").strip()

            if unified_msg.chat.type == "PRIVATE":
                if not text.startswith("/"):
                    should_respond = True

            has_media = any([
                unified_msg.photo,
                unified_msg.sticker,
                unified_msg.video,
                unified_msg.document,
                unified_msg.audio,
                unified_msg.voice,
            ])

            if not text and has_media:
                if await check_reply_chain(unified_msg):
                    should_respond = True

            if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
                should_respond = True

            if unified_msg.mentioned:
                should_respond = True

            if not should_respond and await check_reply_chain(unified_msg):
                should_respond = True

            if not should_respond and random.random() < 0.05:
                should_respond = True

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
                    logger.error(f"Error processing WhatsApp message: {e}")

            future.add_done_callback(_log_failure)

    except Exception as e:
        logger.error(f"Failed to initialize WhatsApp handler: {e}")
        whatsapp_platform = None
