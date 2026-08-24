import asyncio

import pytest

from shin_ai.services.native_work import NativeWorkLimiter


def test_cancellation_removes_work_that_has_not_entered_native_code() -> None:
    async def scenario() -> None:
        limiter = NativeWorkLimiter(1, task_name="test-native-work")
        first_release = asyncio.Event()
        second_ran = False

        async def first(commit):
            commit()
            await first_release.wait()

        async def second(commit):
            nonlocal second_ran
            second_ran = True
            commit()

        first_task = asyncio.create_task(limiter.run(first))
        await asyncio.sleep(0)
        second_task = asyncio.create_task(limiter.run(second))
        await asyncio.sleep(0)
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task

        first_release.set()
        await first_task
        await limiter.close()
        assert not second_ran
        assert limiter.pending_count == 0

    asyncio.run(scenario())
