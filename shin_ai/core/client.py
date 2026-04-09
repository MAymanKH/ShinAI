from pyrogram import Client

from shin_ai.config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CONFIGURED,
    TELEGRAM_ENABLED,
)
from shin_ai.utils.logger_config import logger


class DisabledTelegramClient:
    """No-op client used when Telegram is intentionally disabled."""

    def on_message(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def on_callback_query(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


if TELEGRAM_ENABLED and TELEGRAM_CONFIGURED:
    app = Client(
        "shin_ai_bot",
        api_id=TELEGRAM_API_ID,
        api_hash=TELEGRAM_API_HASH,
        bot_token=TELEGRAM_BOT_TOKEN,
        workdir=".",  # Keep session files in project root.
    )
elif TELEGRAM_ENABLED and not TELEGRAM_CONFIGURED:
    logger.warning(
        "Telegram is enabled but credentials are incomplete; Telegram platform will be skipped."
    )
    app = DisabledTelegramClient()
else:
    logger.info("Telegram platform is disabled by configuration.")
    app = DisabledTelegramClient()
