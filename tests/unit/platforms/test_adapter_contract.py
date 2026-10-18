"""Every adapter must answer the capability questions the executor asks."""

import inspect

import pytest

from shin_ai.platforms.base import PlatformAdapter

CAPABILITIES = (
    "platform_name",
    "supports_stickers",
    "supports_member_restrictions",
    "uses_integer_message_ids",
    "sticker_id_prefix",
    "prefers_native_reply",
    "coordination_scope",
)


def _adapters():
    from shin_ai.platforms.discord import DiscordPlatform
    from shin_ai.platforms.telegram import TelegramPlatform
    from shin_ai.platforms.whatsapp import WhatsAppPlatform

    return [TelegramPlatform, DiscordPlatform, WhatsAppPlatform]


@pytest.mark.parametrize("adapter", _adapters())
def test_adapter_implements_the_full_abstract_surface(adapter) -> None:
    missing = {
        name
        for name, value in vars(PlatformAdapter).items()
        if getattr(value, "__isabstractmethod__", False)
        and getattr(adapter, name, None) is getattr(PlatformAdapter, name, None)
    }
    assert not missing, f"{adapter.__name__} leaves abstract: {sorted(missing)}"


@pytest.mark.parametrize("adapter", _adapters())
@pytest.mark.parametrize("capability", CAPABILITIES)
def test_capability_is_declared_as_a_property(adapter, capability) -> None:
    """Read as attributes on instances, so they must not be plain methods."""
    attribute = inspect.getattr_static(adapter, capability)
    assert isinstance(attribute, property), f"{adapter.__name__}.{capability} is not a property"


@pytest.mark.parametrize("adapter", _adapters())
def test_to_unified_message_is_part_of_the_contract(adapter) -> None:
    assert callable(getattr(adapter, "to_unified_message", None))


class TestDeclaredCapabilities:
    """Pin the values action_executor branches on."""

    def test_whatsapp_uses_opaque_string_ids(self) -> None:
        from shin_ai.platforms.whatsapp import WhatsAppPlatform

        assert WhatsAppPlatform.uses_integer_message_ids.fget(None) is False
        assert WhatsAppPlatform.sticker_id_prefix.fget(None) == "wa:"
        assert WhatsAppPlatform.prefers_native_reply.fget(None) is True
        assert WhatsAppPlatform.supports_member_restrictions.fget(None) is False

    def test_telegram_uses_numeric_ids_and_no_sticker_prefix(self) -> None:
        from shin_ai.platforms.telegram import TelegramPlatform

        assert TelegramPlatform.uses_integer_message_ids.fget(None) is True
        assert TelegramPlatform.sticker_id_prefix.fget(None) == ""
        assert TelegramPlatform.prefers_native_reply.fget(None) is False

    def test_discord_declines_stickers(self) -> None:
        from shin_ai.platforms.discord import DiscordPlatform

        assert DiscordPlatform.supports_stickers.fget(None) is False
        assert DiscordPlatform.uses_integer_message_ids.fget(None) is True


def test_discord_reaction_propagates_delivery_failure() -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from shin_ai.platforms.discord import DiscordPlatform

    message = SimpleNamespace(add_reaction=AsyncMock(side_effect=RuntimeError("rejected")))
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    platform = SimpleNamespace(client=SimpleNamespace(get_channel=Mock(return_value=channel)))
    with pytest.raises(RuntimeError, match="rejected"):
        asyncio.run(DiscordPlatform.react(platform, 1, 2, "👍"))


def test_telegram_reaction_missing_fallback_target_raises() -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from shin_ai.platforms.telegram import TelegramPlatform

    platform = SimpleNamespace(
        client=SimpleNamespace(
            send_reaction=AsyncMock(side_effect=AttributeError("unavailable")),
            get_messages=AsyncMock(return_value=None),
        )
    )
    with pytest.raises(ValueError, match="target message was not found"):
        asyncio.run(TelegramPlatform.react(platform, 1, 2, "👍"))


def test_whatsapp_reaction_missing_target_raises(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from shin_ai.platforms import whatsapp

    monkeypatch.setattr(whatsapp, "Jid2String", str)
    platform = SimpleNamespace(
        _chat_id_to_jid=Mock(return_value="chat"),
        get_cached_raw_message=Mock(return_value=None),
        get_message=AsyncMock(return_value=None),
    )
    with pytest.raises(ValueError, match="target message was not found"):
        asyncio.run(whatsapp.WhatsAppPlatform.react(platform, "chat", "msg", "👍"))


@pytest.mark.parametrize("method", ["send_message", "send_sticker"])
@pytest.mark.parametrize("cancel_cache", [False, True])
def test_whatsapp_send_records_delivery_before_cache_failure(method, cancel_cache) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    from shin_ai.core.request_context import delivery_callback
    from shin_ai.platforms.whatsapp import WhatsAppPlatform

    async def scenario():
        events = []

        async def send(*_args, **_kwargs):
            events.append("sent")
            return SimpleNamespace(ID="sent-id")

        async def cache(*_args):
            events.append("cache")
            if cancel_cache:
                raise asyncio.CancelledError
            raise RuntimeError("cache failed")

        async def delivered():
            events.append("charged")

        platform = SimpleNamespace(
            _chat_id_to_jid=Mock(return_value="chat"),
            get_cached_raw_message=Mock(return_value=None),
            _resolve_sticker_source=Mock(return_value="https://example.com/sticker.webp"),
            _run_sync=send,
            _cache_outgoing_message=cache,
            client=SimpleNamespace(send_message=Mock(), send_sticker=Mock()),
        )
        token = delivery_callback.set(delivered)
        try:
            with pytest.raises(asyncio.CancelledError if cancel_cache else RuntimeError):
                await getattr(WhatsAppPlatform, method)(platform, "chat", "content")
            assert events == ["sent", "charged", "cache"]
        finally:
            delivery_callback.reset(token)

    asyncio.run(scenario())


def test_telegram_kick_records_completed_ban_before_unban_failure() -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from shin_ai.core.request_context import delivery_callback
    from shin_ai.platforms.telegram import TelegramPlatform

    async def scenario():
        deliveries = []

        async def delivered():
            deliveries.append(True)

        platform = SimpleNamespace(
            client=SimpleNamespace(
                ban_chat_member=AsyncMock(),
                unban_chat_member=AsyncMock(side_effect=RuntimeError("unban failed")),
            )
        )
        token = delivery_callback.set(delivered)
        try:
            with pytest.raises(RuntimeError, match="unban failed"):
                await TelegramPlatform.kick_chat_member(platform, 1, 2)
            assert deliveries == [True]
        finally:
            delivery_callback.reset(token)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method", ["ban_chat_member", "kick_chat_member", "unban_chat_member", "restrict_chat_member"]
)
def test_discord_moderation_missing_channel_raises(method) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    from shin_ai.platforms.discord import DiscordPlatform

    platform = SimpleNamespace(client=SimpleNamespace(get_channel=Mock(return_value=None)))
    args = (platform, 1, 2, False) if method == "restrict_chat_member" else (platform, 1, 2)
    with pytest.raises(ValueError, match="available guild channel"):
        asyncio.run(getattr(DiscordPlatform, method)(*args))
