import asyncio

from shin_ai.coordination.store import InMemoryCoordinationStore
from shin_ai.utils.rate_limit import (
    _group_max_responses,
    check_group_rate_limit_shared,
    check_rate_limit_shared,
)


def run(coro):
    return asyncio.run(coro)


def test_user_rate_limit_is_shared_but_credential_scoped() -> None:
    async def scenario() -> None:
        store = InMemoryCoordinationStore()
        assert await check_rate_limit_shared("telegram", 1, coordination_scope="bot-a", store=store)
        assert not await check_rate_limit_shared("telegram", 1, coordination_scope="bot-a", store=store)
        assert await check_rate_limit_shared("telegram", 1, coordination_scope="bot-b", store=store)

    run(scenario())


def test_group_rate_limit_is_atomic_across_callers() -> None:
    async def scenario() -> None:
        store = InMemoryCoordinationStore()
        allowed = _group_max_responses()
        results = await asyncio.gather(
            *(
                check_group_rate_limit_shared(
                    "discord",
                    "chat",
                    coordination_scope="bot",
                    store=store,
                    now=100.0,
                )
                for _ in range(allowed + 2)
            )
        )

        assert sum(results) == allowed

    run(scenario())
