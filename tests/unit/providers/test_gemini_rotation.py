"""The key/model rotation must not be charged to the generation timeout."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shin_ai.coordination import InMemoryCoordinationStore
from shin_ai.providers import gemini as gemini_module
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.gemini_scheduler import GeminiScheduler
from shin_ai.settings import ProviderSettings, get_settings


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


@pytest.fixture
def scoped_gemini(monkeypatch):
    from shin_ai.providers import registry

    providers = {
        "shared": ProviderSettings(
            name="shared", type="gemini", models=("shared-model",), api_keys={"shared-key": "shared-secret"}
        ),
        "dedicated": ProviderSettings(
            name="dedicated",
            type="gemini",
            models=("dedicated-model",),
            api_keys={"dedicated-key": "dedicated-secret"},
            chats={"telegram": ("10",)},
        ),
    }
    base = get_settings()
    settings = replace(
        base, ai=replace(base.ai, providers=providers, primary="shared", fallbacks=("dedicated",))
    )
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_module, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_module, "_gemini_schedulers", {})
    monkeypatch.setattr(
        gemini_module,
        "get_api_keys",
        lambda: {
            "shared-key": "shared-secret",
            "dedicated-key": "dedicated-secret",
        },
    )
    store = InMemoryCoordinationStore("scoped-gemini-test")
    monkeypatch.setattr(gemini_module, "get_coordination_store", lambda: store)
    return providers


def _message(platform="telegram", chat_id=10):
    return SimpleNamespace(platform=platform, chat=SimpleNamespace(id=chat_id))


def test_router_uses_only_the_selected_gemini_keys_and_models(scoped_gemini, stub_gemini):
    from shin_ai.core.provider_router import call_ai_provider

    async def answer(_model):
        return SimpleNamespace(text="answered", usage_metadata=None), []

    attempts = stub_gemini(answer)

    async def scenario():
        for msg in (_message(), _message(chat_id=99), _message(platform="discord")):
            response, _ = await call_ai_provider(
                msg=msg, system_prompt="system", prompt="hello", media_list=[]
            )
            assert response == "answered"

    asyncio.run(scenario())
    assert attempts == [
        ("dedicated-model", "dedicated-secret"),
        ("shared-model", "shared-secret"),
        ("shared-model", "shared-secret"),
    ]


def test_media_helpers_use_only_chat_eligible_gemini_providers(scoped_gemini, stub_gemini):
    from shin_ai.core.provider_router import describe_media_with_gemini
    from shin_ai.providers.tool_loop import ask_gemini_about_image

    async def answer(_model):
        return SimpleNamespace(text="an image", usage_metadata=None), []

    attempts = stub_gemini(answer)

    async def scenario():
        for msg in (_message(), _message(chat_id=99)):
            media = [
                {
                    "bytes": b"image",
                    "mime_type": "image/png",
                    "sender": "User",
                    "position": "current message",
                    "media_type": "photo",
                }
            ]
            assert await describe_media_with_gemini("describe", media, msg=msg) == "an image"
            assert await ask_gemini_about_image("question", media, (None, msg)) == "an image"

    asyncio.run(scenario())
    assert attempts == [("dedicated-model", "dedicated-secret")] * 2 + [("shared-model", "shared-secret")] * 2


def test_direct_gemini_execution_rejects_other_chats(scoped_gemini, stub_gemini):
    from shin_ai.core.provider_router import execute_provider_once

    async def answer(_model):
        raise AssertionError("Restricted provider must not execute")

    attempts = stub_gemini(answer)

    async def scenario():
        provider = scoped_gemini["dedicated"]
        with pytest.raises(ValueError, match="not assigned"):
            await execute_provider_once(provider, _message(chat_id=99), "system", "prompt", [])
        with pytest.raises(ValueError, match="No eligible"):
            await gemini_api("system", "prompt", provider=provider, msg=_message(chat_id=99))
        with pytest.raises(ValueError, match="No eligible"):
            await gemini_api("system", "prompt", provider=provider)

    asyncio.run(scenario())
    assert attempts == []


def test_gemini_health_reports_separate_provider_pools(scoped_gemini, monkeypatch):
    from shin_ai.providers import gemini_keys

    monkeypatch.setattr(gemini_keys, "get_settings", gemini_module.get_settings)
    report = asyncio.run(gemini_keys.get_gemini_stats_message())
    shared, dedicated = report.split("**Provider: dedicated**")
    assert "**Provider: shared**" in shared
    assert "shared-model" in shared and "dedicated-model" not in shared
    assert "dedicated-model" in dedicated and "shared-model" not in dedicated
    assert report.count("Eligible keys: 1/1") == 2


def test_image_helpers_do_not_call_gemini_without_an_eligible_provider(scoped_gemini, stub_gemini):
    from shin_ai.core.provider_router import describe_media_with_gemini
    from shin_ai.providers.tool_loop import ask_gemini_about_image

    scoped_gemini.pop("shared")

    async def answer(_model):
        raise AssertionError("No eligible provider must mean no API calls")

    attempts = stub_gemini(answer)

    async def scenario():
        msg = _message(chat_id=99)
        media = [{"bytes": b"image"}]
        assert await describe_media_with_gemini("describe", media, msg=msg) == ""
        assert "No eligible Gemini provider" in await ask_gemini_about_image("question", media, (None, msg))

    asyncio.run(scenario())
    assert attempts == []
