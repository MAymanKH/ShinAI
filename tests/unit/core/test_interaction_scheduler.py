import asyncio

from shin_ai.core.interaction_scheduler import InteractionScheduler


def run(coro):
    return asyncio.run(coro)


def test_scheduler_preserves_order_within_each_chat() -> None:
    async def scenario() -> None:
        completed: list[tuple[str, int]] = []

        async def handle(payload) -> None:
            await asyncio.sleep(0)
            completed.append(payload)

        scheduler = InteractionScheduler(
            handle,
            max_concurrent=4,
            max_pending=20,
            per_chat_limit=10,
            job_ttl_seconds=60,
        )
        for index in range(5):
            await scheduler.submit("chat-a", ("a", index))
            await scheduler.submit("chat-b", ("b", index))
        await asyncio.wait_for(scheduler.wait_idle(), timeout=1)
        await scheduler.close()

        assert [item[1] for item in completed if item[0] == "a"] == list(range(5))
        assert [item[1] for item in completed if item[0] == "b"] == list(range(5))

    run(scenario())


def test_scheduler_never_exceeds_global_concurrency() -> None:
    async def scenario() -> None:
        active = 0
        peak = 0
        gate = asyncio.Event()

        async def handle(_payload) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await gate.wait()
            active -= 1

        scheduler = InteractionScheduler(
            handle,
            max_concurrent=3,
            max_pending=20,
            per_chat_limit=5,
            job_ttl_seconds=60,
        )
        for index in range(10):
            await scheduler.submit(f"chat-{index}", index)
        await asyncio.sleep(0.02)
        assert scheduler.active_count == 3
        assert peak == 3
        gate.set()
        await asyncio.wait_for(scheduler.wait_idle(), timeout=1)
        await scheduler.close()

    run(scenario())


def test_scheduler_rejects_global_overflow() -> None:
    async def scenario() -> None:
        async def handle(_payload) -> None:
            await asyncio.sleep(1)

        scheduler = InteractionScheduler(
            handle,
            max_concurrent=1,
            max_pending=2,
            per_chat_limit=2,
            job_ttl_seconds=60,
        )
        assert (await scheduler.submit("a", 1, delay_seconds=10)).accepted
        assert (await scheduler.submit("b", 2, delay_seconds=10)).accepted
        rejected = await scheduler.submit("c", 3)
        assert rejected.accepted is False
        assert rejected.reason == "global_queue_full"
        await scheduler.close(grace_seconds=0.01)

    run(scenario())


def test_scheduler_drops_oldest_pending_job_for_busy_chat() -> None:
    async def scenario() -> None:
        handled: list[int] = []
        dropped: list[tuple[int, str]] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def handle(payload: int) -> None:
            if payload == 0:
                first_started.set()
                await release_first.wait()
            handled.append(payload)

        scheduler = InteractionScheduler(
            handle,
            max_concurrent=1,
            max_pending=10,
            per_chat_limit=2,
            job_ttl_seconds=60,
            on_drop=lambda payload, reason: dropped.append((payload, reason)),
        )
        await scheduler.submit("chat", 0)
        await first_started.wait()
        await scheduler.submit("chat", 1)
        await scheduler.submit("chat", 2)
        result = await scheduler.submit("chat", 3)
        assert result.dropped == 1
        release_first.set()
        await asyncio.wait_for(scheduler.wait_idle(), timeout=1)
        await scheduler.close()

        assert handled == [0, 2, 3]
        assert dropped == [(1, "per_chat_queue_full")]

    run(scenario())


def test_scheduler_only_applies_delay_to_start_of_chat_burst() -> None:
    async def scenario() -> None:
        scheduler = InteractionScheduler(
            lambda _payload: asyncio.sleep(0),
            max_concurrent=1,
            max_pending=10,
            per_chat_limit=10,
            job_ttl_seconds=60,
        )

        first = await scheduler.submit("chat", 1, delay_seconds=10)
        second = await scheduler.submit("chat", 2, delay_seconds=10)

        assert first.delay_applied == 10
        assert second.delay_applied == 0
        await scheduler.close(grace_seconds=0.01)

    run(scenario())
