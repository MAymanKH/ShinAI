"""Command handlers must outrank the catch-all chat handler.

Pyrogram runs at most one matching handler per group, so a command registered
after the chat handler in the same group never fires -- ``/gstats`` was
answered by the chat model instead of the stats reader.
"""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pyrogram import StopPropagation
from pyrogram.enums import ChatType

from shin_ai.handlers import analytics, stats, telegram_chat
from shin_ai.handlers.common import COMMAND_HANDLER_GROUP

CHAT_HANDLERS = {"context_recorder", "yalbot"}
COMMAND_HANDLERS = {"stats_command", "stats_details_command", "show_analytics"}


@dataclass
class _Registration:
    callback: Any
    filters: Any
    group: int


class _RecordingClient:
    """Stands in for a Pyrogram client and remembers what was attached."""

    def __init__(self) -> None:
        self.messages: dict[str, _Registration] = {}
        self.callbacks: dict[str, _Registration] = {}

    def _record(self, target, filters, group):
        def decorator(func):
            target[func.__name__] = _Registration(func, filters, group)
            return func

        return decorator

    def on_message(self, filters=None, group: int = 0):
        return self._record(self.messages, filters, group)

    def on_callback_query(self, filters=None, group: int = 0):
        return self._record(self.callbacks, filters, group)


class _Message:
    """The little of a Pyrogram message these handlers touch before declining."""

    def __init__(self, from_user=None) -> None:
        self.from_user = from_user
        self.replies: list[str] = []

    async def reply_text(self, text, *_args, **_kwargs):
        self.replies.append(text)
        return self

    async def reply(self, text, *_args, **_kwargs):
        self.replies.append(text)
        return self


@pytest.fixture
def registered(override_settings):
    """Register the real Telegram handlers onto a recording client."""
    override_settings(telegram_chat, "platform", telegram_enabled=True)
    client = _RecordingClient()
    assert telegram_chat.register(client) is not None
    stats.register(client)
    analytics.register(client)
    return client


class TestHandlerGroups:
    def test_every_expected_handler_is_registered(self, registered) -> None:
        assert set(registered.messages) == CHAT_HANDLERS | COMMAND_HANDLERS

    def test_commands_are_dispatched_before_the_chat_handler(self, registered) -> None:
        chat_groups = [registered.messages[name].group for name in CHAT_HANDLERS]
        command_groups = [registered.messages[name].group for name in COMMAND_HANDLERS]

        assert max(command_groups) < min(chat_groups)

    def test_every_command_handler_shares_one_group(self, registered) -> None:
        groups = {registered.messages[name].group for name in COMMAND_HANDLERS}
        assert groups == {COMMAND_HANDLER_GROUP}


class TestPropagation:
    @pytest.mark.parametrize("name", sorted(COMMAND_HANDLERS))
    def test_a_declined_command_never_reaches_the_chat_handler(self, registered, name) -> None:
        """An anonymous sender fails every command's guard; none may fall through."""
        msg = _Message(from_user=None)

        with pytest.raises(StopPropagation):
            asyncio.run(registered.messages[name].callback(object(), msg))

        assert msg.replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "chat_type", "user_id", "delivery"),
    [
        ("/shinai_analytics short", ChatType.PRIVATE, 42, "short"),
        ("/shinai_analytics@ShinAIBot", ChatType.PRIVATE, 42, "attachment"),
        ("/shinai_analytics", ChatType.SUPERGROUP, 42, "private attachment"),
        ("/shinai_analytics short extra", ChatType.PRIVATE, 42, "usage"),
        ("/shinai_analytics", ChatType.PRIVATE, 99, "ignored"),
    ],
)
async def test_analytics_command_modes(registered, monkeypatch, command, chat_type, user_id, delivery):
    monkeypatch.setattr(analytics, "get_settings", lambda: SimpleNamespace(admin_user_id=42))
    monkeypatch.setattr(analytics, "_analytics_cache", None)
    collection = SimpleNamespace(get=lambda **_: {"ids": ["one"], "metadatas": [{"username": "<name>"}]})
    monkeypatch.setattr(analytics, "_get_memory_collection", lambda: collection)
    files = []

    async def capture_document(document, *, force_document):
        assert force_document
        files.append((document.name, document.read().decode("utf-8")))

    msg = SimpleNamespace(
        text=command,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(type=chat_type),
        reply=AsyncMock(),
        reply_document=AsyncMock(side_effect=capture_document),
    )

    async def send_document(chat_id, document, **kwargs):
        assert chat_id == 42
        await capture_document(document, **kwargs)

    registered.me = SimpleNamespace(username="ShinAIBot", usernames=[])
    registered.send_document = AsyncMock(side_effect=send_document)
    handler = registered.messages["show_analytics"]
    assert await handler.filters(registered, msg)
    with pytest.raises(StopPropagation):
        await handler.callback(registered, msg)

    if delivery == "short":
        assert "Bot Analytics" in msg.reply.call_args.args[0]
        assert msg.reply.call_args.kwargs["reply_markup"]
    elif delivery == "usage":
        assert "/shinai_analytics short" in msg.reply.call_args.args[0]
    elif delivery == "ignored":
        msg.reply.assert_not_called()
    else:
        [(name, html)] = files
        assert name.endswith(".html")
        assert "<title>ShinAI usage report</title>" in html
        if delivery == "attachment":
            msg.reply_document.assert_awaited_once()
            msg.reply.assert_not_called()
        else:
            registered.send_document.assert_awaited_once()
            msg.reply_document.assert_not_called()
        return
    assert not files


def test_usage_commands_register_before_context_only_when_enabled(override_settings, monkeypatch):
    override_settings(telegram_chat, "platform", telegram_enabled=True)
    base = telegram_chat.get_settings()
    from dataclasses import replace

    settings = replace(base, billing=replace(base.billing, enabled=True))
    monkeypatch.setattr(telegram_chat, "get_settings", lambda: settings)
    client = _RecordingClient()
    platform = telegram_chat.register(client)
    assert client.messages["usage_command"].group == COMMAND_HANDLER_GROUP
    assert client.messages["usage_command"].group < client.messages["context_recorder"].group
    message = _Message(from_user=object())
    message.stop_propagation = lambda: (_ for _ in ()).throw(StopPropagation)
    monkeypatch.setattr(platform, "to_unified_message", lambda _msg: object())
    handler = AsyncMock(return_value=True)
    monkeypatch.setattr(telegram_chat, "handle_usage_command", handler)
    with pytest.raises(StopPropagation):
        asyncio.run(client.messages["usage_command"].callback(client, message))
    handler.assert_awaited_once()
