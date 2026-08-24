"""Command handlers must outrank the catch-all chat handler.

Pyrogram runs at most one matching handler per group, so a command registered
after the chat handler in the same group never fires -- ``/gstats`` was
answered by the chat model instead of the stats reader.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from pyrogram import StopPropagation

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
