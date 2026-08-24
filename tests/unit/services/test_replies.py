import asyncio

from shin_ai.coordination import SQLiteCoordinationStore
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage, UnifiedUser
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


def test_reply_state_is_shared_only_with_matching_credentials(
    monkeypatch,
    tmp_path,
) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "coordination.sqlite3"
        sending_instance = SQLiteCoordinationStore(database_path, "shared")
        receiving_instance = SQLiteCoordinationStore(database_path, "shared")
        monkeypatch.setattr(replies, "_ensure_flush_task", lambda: None)

        try:
            await replies.save_reply(
                chat_id=20,
                message_id=99,
                platform="telegram",
                coordination_scope="telegram:same-credential",
                store=sending_instance,
            )

            incoming = UnifiedMessage(
                platform="telegram",
                id=100,
                chat=UnifiedChat(id=20, title="Test", type="GROUP"),
                from_user=UnifiedUser(
                    id=30,
                    username="user",
                    first_name="User",
                    is_self=False,
                ),
                text="reply",
                reply_to_message_id=99,
            )
            assert not await replies.check_reply_chain(
                incoming,
                coordination_scope="telegram:different-credential",
                store=receiving_instance,
            )

            # Simulate another process: its fast local caches do not contain the send.
            replies._replies_cache = {}
            replies._next_message_watch.clear()
            assert await replies.check_reply_chain(
                incoming,
                coordination_scope="telegram:same-credential",
                store=receiving_instance,
            )
            assert not await replies.check_reply_chain(
                incoming,
                coordination_scope="telegram:different-credential",
                store=receiving_instance,
            )
            assert await replies.check_and_clear_next_message_watch(
                "telegram",
                20,
                coordination_scope="telegram:same-credential",
                store=receiving_instance,
            )
            assert not await replies.check_and_clear_next_message_watch(
                "telegram",
                20,
                coordination_scope="telegram:same-credential",
                store=sending_instance,
            )
        finally:
            await sending_instance.close()
            await receiving_instance.close()

        replies._replies_cache = None
        replies._replies_dirty = False
        replies._replies_revision = 0

    run(scenario())
