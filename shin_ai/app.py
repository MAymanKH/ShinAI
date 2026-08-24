"""Application composition root.

Everything the bot needs is built here, in order, from settings that were
loaded once at startup. Modules below this one take what they need as
arguments or read settings at the point of use; none of them do work at
import time, which is what keeps them importable (and testable) without a
config file, a network, or a running event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shin_ai.settings import ShinAISettings, get_settings
from shin_ai.utils.logger_config import logger

PlatformEntry = tuple[str, Any]


@dataclass(slots=True)
class Application:
    """Owns the configured platforms and their lifecycle."""

    settings: ShinAISettings
    telegram_client: Any = None
    platforms: list[PlatformEntry] = field(default_factory=list)
    started: list[PlatformEntry] = field(default_factory=list)

    @classmethod
    def build(cls, settings: ShinAISettings | None = None) -> Application:
        """Construct clients and register handlers without starting anything.

        Pyrogram binds the running event loop when its Client is constructed,
        so this must be called from inside the application's loop.
        """
        application = cls(settings=settings or get_settings())
        application._register_handlers()
        return application

    def _register_handlers(self) -> None:
        from shin_ai.core.client import create_telegram_client
        from shin_ai.handlers import analytics, discord_chat, stats, telegram_chat, whatsapp_chat

        self.telegram_client = create_telegram_client(self.settings.platform)

        telegram_platform = telegram_chat.register(self.telegram_client)
        if telegram_platform is not None:
            # Admin commands are Telegram-only and share its client.
            stats.register(self.telegram_client)
            analytics.register(self.telegram_client)
            self.platforms.append(("Telegram", telegram_platform))

        for label, module in (("Discord", discord_chat), ("WhatsApp", whatsapp_chat)):
            try:
                platform = module.register()
            except Exception as error:
                logger.error("Failed to register the %s platform: %s", label, error)
                continue
            if platform is not None:
                self.platforms.append((label, platform))

        logger.info("Handlers loaded successfully.", extra={"event_name": "lifecycle.handlers"})

    async def start(self) -> list[PlatformEntry]:
        """Start every registered platform, tolerating individual failures."""
        for label, platform in self.platforms:
            logger.info("Starting %s platform...", label, extra={"event_name": "platform.starting"})
            try:
                await platform.start()
                self.started.append((label, platform))
            except Exception as error:
                logger.error("Failed to start %s platform: %s", label, error)

        if not self.started:
            logger.warning(
                "No chat platforms are active. Enable one under `platform:` in "
                "config.yaml and provide its credentials."
            )
        else:
            logger.info(
                "ShinAI started successfully; listening for messages",
                extra={"event_name": "lifecycle.ready"},
            )
        return self.started

    async def shutdown(self) -> None:
        from shin_ai.core.lifecycle import shutdown_application

        await shutdown_application(self.started)
