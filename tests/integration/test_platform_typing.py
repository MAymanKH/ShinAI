import asyncio
from collections import OrderedDict
from threading import RLock

import pytest

from shin_ai.platforms.discord import DiscordPlatform
from shin_ai.platforms.telegram import TelegramPlatform
from shin_ai.platforms.whatsapp import WhatsAppPlatform


@pytest.mark.integration
def test_discord_typing_context_is_entered_once_and_exited() -> None:
    class TypingContext:
        def __init__(self) -> None:
            self.actions: list[str] = []

        async def __aenter__(self):
            self.actions.append("enter")
            return self

        async def __aexit__(self, *_args):
            self.actions.append("exit")

    class Channel:
        def __init__(self) -> None:
            self.context = TypingContext()

        def typing(self):
            return self.context

    class Client:
        def __init__(self) -> None:
            self.channel = Channel()

        def get_channel(self, _chat_id):
            return self.channel

    async def scenario() -> None:
        platform = DiscordPlatform.__new__(DiscordPlatform)
        platform.client = Client()
        platform._typing_contexts = {}

        await platform.send_chat_action(1, "typing")
        await platform.send_chat_action(1, "typing")
        await platform.send_chat_action(1, "cancel")

        assert platform.client.channel.context.actions == ["enter", "exit"]
        assert platform._typing_contexts == {}

    asyncio.run(scenario())


@pytest.mark.integration
def test_telegram_forwards_typing_and_cancel_actions() -> None:
    class Client:
        def __init__(self) -> None:
            self.actions = []

        async def send_chat_action(self, chat_id, action) -> None:
            self.actions.append((chat_id, str(action)))

    async def scenario() -> None:
        client = Client()
        platform = TelegramPlatform(client)

        await platform.send_chat_action("123", "typing")
        await platform.send_chat_action("123", "cancel")

        assert [chat_id for chat_id, _action in client.actions] == [123, 123]
        assert "TYPING" in client.actions[0][1]
        assert "CANCEL" in client.actions[1][1]

    asyncio.run(scenario())


@pytest.mark.integration
def test_whatsapp_forwards_composing_then_paused_presence() -> None:
    class Client:
        def __init__(self) -> None:
            self.actions = []

        def send_chat_presence(self, chat_id, presence, media) -> None:
            self.actions.append((chat_id, presence, media))

    async def scenario() -> None:
        platform = WhatsAppPlatform.__new__(WhatsAppPlatform)
        platform.client = Client()
        platform._chat_id_to_jid = lambda chat_id: f"jid:{chat_id}"

        async def run_sync(function, *args, **kwargs):
            return function(*args, **kwargs)

        platform._run_sync = run_sync
        await platform.send_chat_action("123", "typing")
        await platform.send_chat_action("123", "cancel")

        assert [entry[0] for entry in platform.client.actions] == ["jid:123", "jid:123"]
        assert platform.client.actions[0][1] != platform.client.actions[1][1]

    asyncio.run(scenario())


@pytest.mark.integration
def test_whatsapp_stop_terminates_and_joins_connection_task() -> None:
    async def scenario() -> None:
        stopped = asyncio.Event()
        events = []

        class Client:
            def stop(self) -> None:
                events.append("stop")
                stopped.set()

            def disconnect(self) -> None:
                raise AssertionError("disconnect() does not terminate connect()")

        async def connection() -> None:
            await stopped.wait()
            events.append("connection-ended")

        platform = WhatsAppPlatform.__new__(WhatsAppPlatform)
        platform.client = Client()
        platform._loop = asyncio.get_running_loop()
        platform._connect_task = asyncio.create_task(connection())
        platform._cache_lock = RLock()
        platform._raw_message_cache = OrderedDict()
        platform._unified_message_cache = OrderedDict()
        platform._group_title_cache = OrderedDict()
        platform._bot_user_cache = object()

        async def run_sync(function, *args, **kwargs):
            return function(*args, **kwargs)

        platform._run_sync = run_sync

        await platform.stop()

        assert events == ["stop", "connection-ended"]
        assert platform._connect_task is None
        assert platform._loop is None

    asyncio.run(scenario())
