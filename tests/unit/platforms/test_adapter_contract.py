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
