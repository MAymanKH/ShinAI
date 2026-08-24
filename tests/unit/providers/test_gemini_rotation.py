"""The key/model rotation must not be charged to the generation timeout."""

import asyncio
from types import SimpleNamespace

import pytest

from shin_ai.coordination import InMemoryCoordinationStore
from shin_ai.providers import gemini as gemini_module
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.gemini_scheduler import GeminiScheduler


class FakeMonotonic:
    """A clock that only moves when a model is actually generating."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _scheduler(keys: int = 3, models: tuple[str, ...] = ("model-a", "model-b")) -> GeminiScheduler:
    return GeminiScheduler(
        {f"key-{index}": f"secret-{index}" for index in range(1, keys + 1)},
        models,
        InMemoryCoordinationStore("test"),
    )


@pytest.fixture
def stub_gemini(monkeypatch):
    """Replace the SDK-facing helpers so only the rotation logic is exercised."""
    attempts: list[tuple[str, str]] = []

    def _install(generation_loop):
        monkeypatch.setattr(gemini_module, "_get_genai_client", lambda api_key: api_key)
        monkeypatch.setattr(gemini_module, "_build_gemini_config", lambda *args, **kwargs: None)

        async def loop(genai_client, model, _contents, _config, _tool_context, _media):
            attempts.append((model, genai_client))
            return await generation_loop(model)

        monkeypatch.setattr(gemini_module, "_run_gemini_generation_loop", loop)
        return attempts

    return _install


def test_rejected_keys_do_not_consume_the_generation_budget(stub_gemini, monkeypatch) -> None:
    """A 429 answered instantly costs no budget, so every pair still gets a turn."""
    clock = FakeMonotonic()
    monkeypatch.setattr(gemini_module, "time", SimpleNamespace(monotonic=clock))

    async def rejected(_model):
        raise RuntimeError("429 rate limit")

    attempts = stub_gemini(rejected)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="429"):
            await gemini_api(
                "system",
                "prompt",
                scheduler=_scheduler(),
                attempt_timeout_seconds=60.0,
                rotation_budget_seconds=1.0,
            )

    asyncio.run(scenario())

    assert len(attempts) == 6
    assert {model for model, _ in attempts} == {"model-a", "model-b"}


def test_generation_time_is_what_exhausts_the_budget(stub_gemini, monkeypatch) -> None:
    clock = FakeMonotonic()
    monkeypatch.setattr(gemini_module, "time", SimpleNamespace(monotonic=clock))

    async def slow_failure(_model):
        clock.advance(0.5)
        raise RuntimeError("model unavailable")

    attempts = stub_gemini(slow_failure)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="model unavailable"):
            await gemini_api(
                "system",
                "prompt",
                scheduler=_scheduler(),
                attempt_timeout_seconds=60.0,
                rotation_budget_seconds=1.0,
            )

    asyncio.run(scenario())

    assert len(attempts) == 2


def test_a_hung_pair_is_cut_off_without_ending_the_rotation(stub_gemini) -> None:
    """The timeout bounds one pair; the remaining models still get tried."""

    async def hang_until_model_b(model):
        if model == "model-a":
            await asyncio.sleep(3600)
        return SimpleNamespace(text="answered", usage_metadata=None), []

    attempts = stub_gemini(hang_until_model_b)

    async def scenario() -> None:
        answer, actions = await gemini_api(
            "system",
            "prompt",
            scheduler=_scheduler(),
            attempt_timeout_seconds=0.02,
            rotation_budget_seconds=600.0,
        )
        assert answer == "answered"
        assert actions == []

    asyncio.run(scenario())

    assert [model for model, _ in attempts] == [
        "model-a",
        "model-a",
        "model-a",
        "model-b",
    ]
