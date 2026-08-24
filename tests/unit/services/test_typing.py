import asyncio

from shin_ai.services.typing import active_typing_count, start_typing, stop_typing


def run(coro):
    return asyncio.run(coro)


class _Platform:
    platform_name = "test"

    def __init__(self) -> None:
        self.actions: list[str] = []

    async def send_chat_action(self, _chat_id, action: str) -> None:
        self.actions.append(action)


def test_stop_prevents_future_typing_refreshes() -> None:
    async def scenario() -> None:
        platform = _Platform()
        session = await start_typing(platform, "chat", refresh_seconds=0.01)
        await asyncio.sleep(0)
        await stop_typing(session)
        stopped_actions = list(platform.actions)
        await asyncio.sleep(0.03)

        assert stopped_actions == ["typing", "cancel"]
        assert platform.actions == stopped_actions
        assert active_typing_count() == 0

    run(scenario())


def test_stop_waits_for_inflight_typing_before_cancel() -> None:
    class SlowPlatform(_Platform):
        def __init__(self) -> None:
            super().__init__()
            self.typing_started = asyncio.Event()
            self.release_typing = asyncio.Event()

        async def send_chat_action(self, _chat_id, action: str) -> None:
            self.actions.append(f"{action}:start")
            if action == "typing":
                self.typing_started.set()
                await self.release_typing.wait()
            self.actions.append(f"{action}:end")

    async def scenario() -> None:
        platform = SlowPlatform()
        session = await start_typing(platform, "chat")
        await platform.typing_started.wait()

        stopping = asyncio.create_task(stop_typing(session))
        await asyncio.sleep(0)
        assert platform.actions == ["typing:start"]

        platform.release_typing.set()
        await stopping

        assert platform.actions == [
            "typing:start",
            "typing:end",
            "cancel:start",
            "cancel:end",
        ]
        assert active_typing_count() == 0

    run(scenario())


def test_safety_timeout_cancels_indicator_and_cleans_registry() -> None:
    async def scenario() -> None:
        platform = _Platform()
        await start_typing(
            platform,
            "chat",
            refresh_seconds=1,
            max_duration_seconds=0.01,
        )
        await asyncio.sleep(0.03)

        assert platform.actions == ["typing", "cancel"]
        assert active_typing_count() == 0

    run(scenario())
