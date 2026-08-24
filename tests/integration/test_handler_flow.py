import asyncio

import pytest

from shin_ai.core import handler
from shin_ai.core.action_executor import ActionExecutionResult
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage, UnifiedUser


class _Platform:
    platform_name = "telegram"
    coordination_scope = "telegram:test"


def _message() -> UnifiedMessage:
    return UnifiedMessage(
        platform="telegram",
        id=10,
        chat=UnifiedChat(id=20, title="Test", type="GROUP"),
        from_user=UnifiedUser(
            id=30,
            username="user",
            first_name="User",
            is_self=False,
        ),
        text="hello",
    )


@pytest.mark.integration
def test_visible_response_stops_typing_before_memory_persistence(monkeypatch) -> None:
    async def scenario() -> None:
        events: list[str] = []
        session = object()

        async def start_typing(*_args, **_kwargs):
            events.append("typing:start")
            return session

        async def stop_typing(received) -> None:
            assert received is session
            events.append("typing:stop")

        async def call_provider(**_kwargs):
            return "first---second", []

        async def execute_actions(**_kwargs):
            return ActionExecutionResult(errors=[], completed_actions=[])

        async def execute_messages(**_kwargs):
            events.append("response:sent")
            return ["first", "second"]

        async def save_memory(**kwargs) -> None:
            assert kwargs["messages"] == ["first", "second"]
            events.append("memory:saved")

        monkeypatch.setattr(handler, "is_trivial_message", lambda _msg: False)
        monkeypatch.setattr(handler, "start_typing", start_typing)
        monkeypatch.setattr(handler, "stop_typing", stop_typing)
        monkeypatch.setattr(handler, "call_ai_provider", call_provider)
        monkeypatch.setattr(handler, "execute_pending_actions", execute_actions)
        monkeypatch.setattr(handler, "execute_text_messages", execute_messages)
        monkeypatch.setattr(handler, "save_interaction_memory", save_memory)

        await handler._execute_frozen_message(
            platform=_Platform(),
            msg=_message(),
            prompt="hello",
            media_list=[],
            reply_text="",
            style_examples="",
            social_context_section="",
            memory_section="",
            runtime_context="",
            target_instructions="",
        )

        assert events == [
            "typing:start",
            "response:sent",
            "typing:stop",
            "memory:saved",
        ]

    asyncio.run(scenario())


@pytest.mark.integration
def test_provider_failure_still_stops_typing(monkeypatch) -> None:
    async def scenario() -> None:
        events: list[str] = []
        session = object()

        async def start_typing(*_args, **_kwargs):
            events.append("typing:start")
            return session

        async def stop_typing(received) -> None:
            assert received is session
            events.append("typing:stop")

        async def call_provider(**_kwargs):
            raise RuntimeError("provider failed")

        monkeypatch.setattr(handler, "is_trivial_message", lambda _msg: False)
        monkeypatch.setattr(handler, "start_typing", start_typing)
        monkeypatch.setattr(handler, "stop_typing", stop_typing)
        monkeypatch.setattr(handler, "call_ai_provider", call_provider)

        with pytest.raises(RuntimeError, match="provider failed"):
            await handler._execute_frozen_message(
                platform=_Platform(),
                msg=_message(),
                prompt="hello",
                media_list=[],
                reply_text="",
                style_examples="",
                social_context_section="",
                memory_section="",
                runtime_context="",
                target_instructions="",
            )

        assert events == ["typing:start", "typing:stop"]

    asyncio.run(scenario())
