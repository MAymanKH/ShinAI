"""Provider ordering built on the application's typed settings."""

from __future__ import annotations

import itertools
import threading

from shin_ai.settings import (
    AISettings,
    FirecrawlSettings,
    PlatformSettings,
    ProviderSettings,
    ShinAISettings,
    WhisperSettings,
    get_settings,
    parse_settings,
    reload_settings,
)

# Compatibility aliases while callers move to the settings terminology.
ProviderConfig = ProviderSettings
AIConfig = AISettings
PlatformConfig = PlatformSettings
WhisperConfig = WhisperSettings
FirecrawlConfig = FirecrawlSettings
ShinAIConfig = ShinAISettings
_parse_config = parse_settings

_round_robin_counter = itertools.count()
_round_robin_lock = threading.Lock()


def get_config() -> ShinAISettings:
    return get_settings()


def reload_config() -> ShinAISettings:
    return reload_settings()


def get_primary() -> ProviderSettings:
    settings = get_settings()
    return settings.ai.providers[settings.ai.primary]


def get_provider_chain(
    platform: str | None = None, chat_id: int | str | None = None
) -> list[ProviderSettings]:
    settings = get_settings()
    ai = settings.ai
    assigned = [
        provider
        for provider in ai.providers.values()
        if provider.chats and provider.is_available_for(platform, chat_id)
    ]
    names = [name for name in dict.fromkeys([ai.primary, *ai.fallbacks]) if not ai.providers[name].chats]
    if ai.rotation == "round_robin" and names:
        with _round_robin_lock:
            index = next(_round_robin_counter) % len(names)
        names = names[index:] + names[:index]
    return [*assigned, *(ai.providers[name] for name in names)]


def get_first_gemini_provider(
    platform: str | None = None, chat_id: int | str | None = None
) -> ProviderSettings | None:
    eligible = [
        provider
        for provider in get_settings().ai.providers.values()
        if provider.type == "gemini" and provider.is_available_for(platform, chat_id)
    ]
    return next(
        (provider for provider in eligible if provider.chats),
        eligible[0] if eligible else None,
    )
