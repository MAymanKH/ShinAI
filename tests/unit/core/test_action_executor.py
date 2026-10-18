import asyncio

from shin_ai.core.action_executor import execute_pending_actions, save_interaction_memory
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage, UnifiedUser


def run(coro):
    return asyncio.run(coro)


def _message() -> UnifiedMessage:
    return UnifiedMessage(
        platform="telegram",
        id=10,
        chat=UnifiedChat(id=20, title="Test", type="GROUP"),
        from_user=UnifiedUser(id=30, username="user", first_name="User", is_self=False),
        text="hello",
    )


class _Platform:
    platform_name = "telegram"
    supports_stickers = True

    def __init__(self, *, fail_reaction: bool = False) -> None:
        self.fail_reaction = fail_reaction

    async def react(self, _chat_id, _message_id, _emoji) -> None:
        if self.fail_reaction:
            raise RuntimeError("reaction failed")


def test_action_result_contains_only_completed_actions() -> None:
    async def scenario() -> None:
        action = {"type": "reaction", "emoji": "👍"}
        succeeded = await execute_pending_actions(_Platform(), _message(), [action], 10)
        failed = await execute_pending_actions(
            _Platform(fail_reaction=True),
            _message(),
            [action],
            10,
        )

        assert succeeded.completed_actions == [action]
        assert failed.completed_actions == []

    run(scenario())


def test_interaction_memory_combines_text_and_actions_once() -> None:
    async def scenario() -> None:
        saved: list[dict] = []

        async def save(**kwargs) -> None:
            saved.append(kwargs)

        await save_interaction_memory(
            platform="telegram",
            msg=_message(),
            messages=["sent text"],
            completed_actions=[{"type": "reaction", "emoji": "👍"}],
            original_prompt="hello",
            reply_text="",
            memory_saver=save,
            trigger="mention",
        )

        assert len(saved) == 1
        assert saved[0]["response"] == "sent text [Reacted: 👍]"
        assert saved[0]["trigger"] == "mention"
        assert saved[0]["deliveries"] == "reaction,text"

    run(scenario())


def test_interaction_memory_skips_when_nothing_was_delivered() -> None:
    async def scenario() -> None:
        calls = 0

        async def save(**_kwargs) -> None:
            nonlocal calls
            calls += 1

        await save_interaction_memory(
            platform="telegram",
            msg=_message(),
            messages=[],
            completed_actions=[],
            original_prompt="hello",
            reply_text="",
            memory_saver=save,
        )
        assert calls == 0

    run(scenario())


def test_sticker_delivery_is_recorded_before_cache_failure_or_cancellation(monkeypatch) -> None:
    from shin_ai.core import action_executor
    from shin_ai.core.request_context import delivery_callback

    async def scenario(error: BaseException) -> None:
        events = []

        class Platform(_Platform):
            sticker_id_prefix = ""
            uses_integer_message_ids = True
            coordination_scope = "test"

            async def send_sticker(self, *_args):
                events.append("sent")
                return 42

        async def delivered():
            events.append("charged")

        async def save(*_args, **_kwargs):
            events.append("cache")
            raise error

        monkeypatch.setattr(action_executor, "save_reply", save)
        token = delivery_callback.set(delivered)
        try:
            try:
                await execute_pending_actions(
                    Platform(),
                    _message(),
                    [{"type": "sticker", "sticker_id": "sticker"}],
                    10,
                )
            except asyncio.CancelledError:
                assert isinstance(error, asyncio.CancelledError)
            assert events == ["sent", "charged", "cache"]
        finally:
            delivery_callback.reset(token)

    run(scenario(RuntimeError("cache failed")))
    run(scenario(asyncio.CancelledError()))


def test_failed_reaction_does_not_record_delivery() -> None:
    from shin_ai.core.request_context import delivery_callback

    async def scenario():
        deliveries = []

        async def delivered():
            deliveries.append(True)

        token = delivery_callback.set(delivered)
        try:
            result = await execute_pending_actions(
                _Platform(fail_reaction=True),
                _message(),
                [{"type": "reaction", "emoji": "👍"}],
                10,
            )
            assert result.completed_actions == []
            assert deliveries == []
        finally:
            delivery_callback.reset(token)

    run(scenario())


def test_undelivered_moderation_invite_does_not_record_delivery(monkeypatch) -> None:
    from shin_ai.core import action_executor
    from shin_ai.core.request_context import delivery_callback

    async def scenario():
        deliveries = []

        class Platform(_Platform):
            async def create_chat_invite_link(self, *_args):
                return "https://example.com/invite"

            async def send_message(self, *_args):
                return None

        async def target(*_args):
            return _message().from_user

        async def delivered():
            deliveries.append(True)

        monkeypatch.setattr(action_executor, "_resolve_mod_target", target)
        token = delivery_callback.set(delivered)
        try:
            result = await execute_pending_actions(
                Platform(),
                _message(),
                [{"type": "moderation", "action": "add"}],
                10,
            )
            assert result.completed_actions == []
            assert result.errors == ["ADD FAILED: The invitation was not delivered."]
            assert deliveries == []
        finally:
            delivery_callback.reset(token)

    run(scenario())
