import asyncio

from shin_ai.services import replies


def run(coro):
    return asyncio.run(coro)


def test_shutdown_flushes_and_releases_process_caches(monkeypatch) -> None:
    async def scenario() -> None:
        flushed: list[bool] = []

        async def fake_flush() -> None:
            flushed.append(True)

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(replies, "_flush_to_disk", fake_flush)
        replies._replies_cache = {"telegram_1": ["2"]}
        replies._replies_dirty = True
        replies._replies_revision = 1
        replies._next_message_watch["telegram_1"] = True
        replies._flush_task = asyncio.create_task(wait_forever())

        await replies.shutdown_replies_service()

        assert flushed == [True]
        assert replies._flush_task is None
        assert replies._replies_cache is None
        assert replies._replies_dirty is False
        assert replies._replies_revision == 0
        assert replies._next_message_watch == {}

    run(scenario())
