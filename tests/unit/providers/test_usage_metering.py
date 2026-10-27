from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shin_ai.billing.pricing import usage_meter
from shin_ai.providers import gemini, tool_loop

pytestmark = pytest.mark.asyncio


class MeterSpy:
    def __init__(self, events):
        self.events = events
        self.responses = []
        self.bounds = []
        self.failed = []

    async def before_call(self, provider, model, inputs, outputs):
        self.events.append("before")
        self.bounds.append((provider, model, inputs, outputs))
        return object()

    async def record_response(self, ticket, response):
        self.events.append("record")
        self.responses.append(response)

    def failed_call(self, ticket):
        self.failed.append(ticket)


def openai_response(*, tool=False):
    call = SimpleNamespace(id="tool-1", function=SimpleNamespace(name="lookup", arguments="{}"))
    message = SimpleNamespace(
        content=None if tool else "final answer",
        tool_calls=[call] if tool else [],
        model_dump=lambda **_kwargs: {"role": "assistant", "content": None},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage={"prompt_tokens": 10})


async def test_openai_records_each_tool_turn_before_dispatch(monkeypatch):
    events = []
    spy = MeterSpy(events)
    responses = [openai_response(tool=True), openai_response()]

    async def create_completion(**_kwargs):
        events.append("generate")
        return responses.pop(0)

    async def dispatch(*_args):
        events.append("tool")
        return "tool output", None

    monkeypatch.setattr(tool_loop, "_execute_tool_call", dispatch)
    token = usage_meter.set(spy)
    try:
        answer, actions = await tool_loop.run_tool_calling_chat(
            provider_name="paid-provider",
            create_completion=create_completion,
            system_prompt="system",
            prompt="hello",
            model="model",
            max_tokens=16,
        )
    finally:
        usage_meter.reset(token)
    assert answer == "final answer" and actions == []
    assert events == ["before", "generate", "record", "tool", "before", "generate", "record"]
    assert len(spy.responses) == 2
    assert all(bound[0:2] == ("paid-provider", "model") and bound[3] == 16 for bound in spy.bounds)
    assert spy.bounds[1][2] > spy.bounds[0][2]


async def test_openai_free_path_does_not_require_usage_metadata():
    token = usage_meter.set(None)
    try:
        response = openai_response()
        del response.usage
        result = await tool_loop.run_tool_calling_chat(
            provider_name="free-provider",
            create_completion=AsyncMock(return_value=response),
            system_prompt="system",
            prompt="hello",
            model="model",
        )
    finally:
        usage_meter.reset(token)
    assert result == ("final answer", [])


async def test_gemini_records_each_turn_after_exact_input_count(monkeypatch):
    events = []
    spy = MeterSpy(events)
    first = SimpleNamespace(
        function_calls=[SimpleNamespace(name="lookup", args={})],
        candidates=[SimpleNamespace(content=gemini.genai.types.Content(role="model", parts=[]))],
    )
    final = SimpleNamespace(function_calls=[])
    responses = [first, final]
    counts = []

    async def count_tokens(**kwargs):
        events.append("count")
        counts.append(len(kwargs["contents"]))
        return SimpleNamespace(total_tokens=20 + len(kwargs["contents"]))

    async def generate_content(**_kwargs):
        events.append("generate")
        return responses.pop(0)

    async def dispatch(*_args):
        events.append("tool")
        return "tool output", None

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=count_tokens,
                generate_content=generate_content,
            )
        )
    )
    config = SimpleNamespace(system_instruction="system", tools=[], max_output_tokens=16)
    monkeypatch.setattr(gemini, "_dispatch_gemini_tool", dispatch)
    token = usage_meter.set(spy)
    try:
        result, actions = await gemini._run_gemini_generation_loop(
            client,
            "model",
            [],
            config,
            provider_name="paid-gemini",
        )
    finally:
        usage_meter.reset(token)
    assert result is final and actions == []
    assert counts == [0, 2]
    assert events == [
        "count",
        "before",
        "generate",
        "record",
        "tool",
        "count",
        "before",
        "generate",
        "record",
    ]
    assert spy.responses == [first, final]
    assert spy.bounds == [("paid-gemini", "model", 20, 16), ("paid-gemini", "model", 22, 16)]


async def test_gemini_free_path_does_not_count_or_meter():
    final = SimpleNamespace(function_calls=[])
    models = SimpleNamespace(generate_content=AsyncMock(return_value=final))
    token = usage_meter.set(None)
    try:
        result, actions = await gemini._run_gemini_generation_loop(
            SimpleNamespace(aio=SimpleNamespace(models=models)),
            "model",
            [],
            object(),
        )
    finally:
        usage_meter.reset(token)
    assert result is final and actions == []
    models.generate_content.assert_awaited_once()


async def test_sdk_failure_marks_unknown_call_without_dispatch(monkeypatch):
    spy = MeterSpy([])
    dispatch = AsyncMock()
    monkeypatch.setattr(tool_loop, "_execute_tool_call", dispatch)
    token = usage_meter.set(spy)
    try:
        with pytest.raises(TimeoutError):
            await tool_loop.run_tool_calling_chat(
                provider_name="paid-provider",
                create_completion=AsyncMock(side_effect=TimeoutError()),
                system_prompt="system",
                prompt="hello",
                model="model",
                max_tokens=16,
            )
    finally:
        usage_meter.reset(token)
    assert len(spy.failed) == 1
    assert spy.responses == []
    dispatch.assert_not_awaited()


async def test_paid_gemini_does_not_fold_failed_rotation_into_a_later_success(monkeypatch):
    from shin_ai.coordination import InMemoryCoordinationStore
    from shin_ai.providers.gemini_scheduler import GeminiScheduler

    scheduler = GeminiScheduler(
        {"one": "test-key-one", "two": "test-key-two"},
        ("model", "another-model"),
        InMemoryCoordinationStore("metered-rotation"),
    )
    generation = AsyncMock(side_effect=RuntimeError("first generation failed"))
    monkeypatch.setattr(gemini, "_get_genai_client", lambda _key: object())
    monkeypatch.setattr(gemini, "_build_gemini_config", lambda *_args: None)
    monkeypatch.setattr(gemini, "_run_gemini_generation_loop", generation)
    token = usage_meter.set(MeterSpy([]))
    try:
        with pytest.raises(RuntimeError, match="first generation failed"):
            await gemini.gemini_api("system", "hello", scheduler=scheduler)
    finally:
        usage_meter.reset(token)
    generation.assert_awaited_once()
