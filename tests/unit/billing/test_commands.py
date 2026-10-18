from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shin_ai.billing import commands
from shin_ai.billing.store import Balance, Identity, Scope, SponsorConflict
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage, UnifiedUser


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(
        commands, "get_settings", lambda: SimpleNamespace(billing=SimpleNamespace(enabled=True))
    )
    store = SimpleNamespace(balance=AsyncMock(return_value=Balance(25, 2)), configure=AsyncMock())
    monkeypatch.setattr(commands, "get_credit_store", lambda: store)
    platform = SimpleNamespace(
        reply_to_message=AsyncMock(),
        get_bot_user=AsyncMock(return_value=UnifiedUser(99, "ShinBot", "Bot", True)),
    )
    msg = UnifiedMessage(
        "discord", 10, UnifiedChat(20, "Friends", "GROUP", 30), UnifiedUser(1, "u", "U", False)
    )
    return platform, msg, store


@pytest.mark.asyncio
async def test_disabled_does_not_access_storage(setup, monkeypatch):
    platform, msg, store = setup
    msg.text = "/balance"
    monkeypatch.setattr(
        commands, "get_settings", lambda: SimpleNamespace(billing=SimpleNamespace(enabled=False))
    )
    assert not await commands.handle_usage_command(platform, msg)
    store.balance.assert_not_called()
    platform.reply_to_message.assert_not_called()


@pytest.mark.asyncio
async def test_balance_is_private(setup):
    platform, msg, store = setup
    msg.text = "/balance"
    assert await commands.handle_usage_command(platform, msg)
    store.balance.assert_not_called()
    assert "private chat" in platform.reply_to_message.call_args.args[1]
    msg.chat.type = "PRIVATE"
    assert await commands.handle_usage_command(platform, msg)
    store.balance.assert_awaited_once_with(Identity("discord", "1"))
    assert "25" in platform.reply_to_message.call_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,handled", [("/balance@OtherBot", False), ("/balance@shinbot", True), ("/unknown", False)]
)
async def test_command_recipient(setup, text, handled):
    platform, msg, _store = setup
    msg.text = text
    assert await commands.handle_usage_command(platform, msg) is handled


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "only_me", "everyone"])
@pytest.mark.parametrize("server", [False, True])
async def test_permission_scope(setup, mode, server):
    platform, msg, store = setup
    msg.text = f"/paid_usage {mode}{' server' if server else ''}"
    assert await commands.handle_usage_command(platform, msg)
    store.configure.assert_awaited_once_with(
        Identity("discord", "1"),
        Scope("discord", "guild" if server else "chat", "30" if server else "20"),
        mode,
    )
    if mode == "everyone":
        assert "Other members can spend your balance" in platform.reply_to_message.call_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["/paid_usage", "/paid_usage on", "/paid_usage everyone wrong", "/paid_usage off server extra"]
)
async def test_invalid_command_never_changes_settings(setup, text):
    platform, msg, store = setup
    msg.text = text
    assert await commands.handle_usage_command(platform, msg)
    store.configure.assert_not_called()


@pytest.mark.asyncio
async def test_server_requires_discord_guild(setup):
    platform, msg, store = setup
    msg.chat.guild_id = None
    msg.text = "/paid_usage everyone server"
    assert await commands.handle_usage_command(platform, msg)
    store.configure.assert_not_called()


@pytest.mark.asyncio
async def test_existing_shared_sponsor_is_explained(setup):
    platform, msg, store = setup
    store.configure.side_effect = SponsorConflict()
    msg.text = "/paid_usage everyone"
    assert await commands.handle_usage_command(platform, msg)
    assert "already has a shared sponsor" in platform.reply_to_message.call_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_identifiers_are_canonical(setup):
    platform, msg, store = setup
    msg.platform = "whatsapp"
    msg.from_user.id = "123:4@s.whatsapp.net"
    msg.chat.id = "456:3@g.us"
    msg.text = "/paid_usage only_me"
    assert await commands.handle_usage_command(platform, msg)
    store.configure.assert_awaited_once_with(
        Identity("whatsapp", "123@s.whatsapp.net"), Scope("whatsapp", "chat", "456@g.us"), "only_me"
    )


@pytest.mark.parametrize("guild_id", [None, 900])
def test_discord_conversion_preserves_server_scope(guild_id):
    from datetime import UTC, datetime

    from shin_ai.platforms.discord import DiscordPlatform

    native = SimpleNamespace(
        channel=SimpleNamespace(id=100, name="general"),
        guild=SimpleNamespace(id=guild_id) if guild_id is not None else None,
        author=None,
        id=200,
        content="hello",
        created_at=datetime.now(UTC),
        reference=None,
        attachments=[],
        mentions=[],
    )
    adapter = object.__new__(DiscordPlatform)
    adapter.client = SimpleNamespace(user=None)
    converted = adapter.to_unified_message(native)
    assert converted.chat.guild_id == guild_id
    assert converted.chat.id == 100
