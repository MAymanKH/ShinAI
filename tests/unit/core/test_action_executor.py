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
        )

        assert len(saved) == 1
        assert saved[0]["response"] == "sent text [Reacted: 👍]"

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
