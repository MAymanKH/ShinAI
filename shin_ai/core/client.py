"""Telegram client construction.

Pyrogram captures the running event loop when a Client is constructed, so the
client must be built after asyncio.run() has created the application loop --
never at import time.
"""

from __future__ import annotations

from shin_ai.settings import PlatformSettings
from shin_ai.utils.logger_config import logger


class DisabledTelegramClient:
    """No-op stand-in used when Telegram is disabled or not configured."""

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


def create_telegram_client(platform: PlatformSettings):
    """Return a live Pyrogram client, or a no-op stand-in when unavailable."""
    if not platform.telegram_enabled:
        logger.info("Telegram platform is disabled by configuration.")
        return DisabledTelegramClient()

    if not platform.telegram_configured:
        logger.warning(
            "Telegram is enabled but credentials are incomplete; Telegram platform will be skipped."
        )
        return DisabledTelegramClient()

    from pyrogram import Client

    return Client(
        "shin_ai_bot",
        api_id=platform.telegram_api_id,
        api_hash=platform.telegram_api_hash,
        bot_token=platform.telegram_bot_token,
        workdir=".",  # Keep session files in project root.
    )
