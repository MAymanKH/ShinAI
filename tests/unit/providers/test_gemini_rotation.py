"""The key/model rotation must not be charged to the generation timeout."""

import asyncio
from types import SimpleNamespace

import pytest

from shin_ai.coordination import InMemoryCoordinationStore
from shin_ai.providers import gemini as gemini_module
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.gemini_scheduler import GeminiScheduler


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
