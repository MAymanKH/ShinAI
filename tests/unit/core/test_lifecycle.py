import asyncio

from shin_ai.core.lifecycle import shutdown_application


def run(coro):
    return asyncio.run(coro)


class _Platform:
    def __init__(self, label: str, events: list[str], *, fail: bool = False) -> None:
        self.label = label
        self.events = events
        self.fail = fail

    async def stop(self) -> None:
        self.events.append(f"stop:{self.label}")
        if self.fail:
            raise RuntimeError("stop failed")


def test_shutdown_uses_safe_order_and_continues_after_failures() -> None:
    async def scenario() -> None:
        events: list[str] = []

        async def drain() -> None:
            events.append("drain")

        async def close_first() -> None:
            events.append("close:first")
            raise RuntimeError("close failed")

        async def close_second() -> None:
            events.append("close:second")

        platforms = [
            ("first", _Platform("first", events)),
            ("second", _Platform("second", events, fail=True)),
        ]
        await shutdown_application(
            platforms,
            interaction_closer=drain,
            resource_closers=(
                ("first resource", close_first),
                ("second resource", close_second),
            ),
        )

        assert events == [
            "drain",
            "stop:second",
            "stop:first",
            "close:first",
            "close:second",
        ]

    run(scenario())
