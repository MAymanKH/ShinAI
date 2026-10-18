"""Provider ordering built on the application's typed settings."""

from __future__ import annotations

import itertools
import threading

from shin_ai.core.request_context import paid_route
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
    route = paid_route.get()
    if route is not None:
        return [
            ai.providers[name]
            for name in route.providers
            if ai.providers[name].is_available_for(platform, chat_id)
        ]
    assigned = [
        provider
        for provider in ai.providers.values()
        if provider.name not in settings.billing.providers
        and provider.chats
        and provider.is_available_for(platform, chat_id)
    ]
    names = [
        name
        for name in dict.fromkeys([ai.primary, *ai.fallbacks])
        if name not in settings.billing.providers and not ai.providers[name].chats
    ]
    if ai.rotation == "round_robin" and names:
        with _round_robin_lock:
            index = next(_round_robin_counter) % len(names)
        names = names[index:] + names[:index]
    return [*assigned, *(ai.providers[name] for name in names)]


def get_first_gemini_provider(
    platform: str | None = None, chat_id: int | str | None = None
) -> ProviderSettings | None:
    settings = get_settings()
    route = paid_route.get()
    candidates = (
        [settings.ai.providers[name] for name in route.providers]
        if route is not None
        else [
            provider
            for provider in settings.ai.providers.values()
            if provider.name not in settings.billing.providers
        ]
    )
    eligible = [
        provider
        for provider in candidates
        if provider.type == "gemini" and provider.is_available_for(platform, chat_id)
    ]
    return next(
        (provider for provider in eligible if provider.chats),
        eligible[0] if eligible else None,
    )
