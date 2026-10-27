import asyncio
import dataclasses
import time
from unittest.mock import AsyncMock

import pytest

from shin_ai import settings
from shin_ai.core import provider_router
from shin_ai.core.request_context import PaidRoute, paid_route
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage
from shin_ai.providers import registry


def configured(monkeypatch, tmp_path):
    raw = {
        "ai": {
            "primary": "free",
            "providers": [
                {"name": "free", "type": "gemini", "models": ["free-model"], "api_keys": {"f": "free-key"}},
                {
                    "name": "paid",
                    "type": "gemini",
                    "models": ["paid-model"],
                    "api_keys": {"p": "paid-key"},
                    "chats": {"discord": ["10"]},
                },
            ],
        },
        "billing": {"providers": ["paid"], "database_path": str(tmp_path / "credits.sqlite3")},
    }
    config = settings.parse_settings(raw)
    for module in (registry, provider_router):
        monkeypatch.setattr(module, "get_settings", lambda: config)
    return config


def route(seconds=1):
    return PaidRoute(("paid",), time.monotonic() + seconds, seconds, 1)


def test_paid_assigned_provider_and_gemini_are_excluded_even_when_disabled(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    assert [p.name for p in registry.get_provider_chain("discord", 10)] == ["free"]
    assert registry.get_first_gemini_provider("discord", 10).name == "free"
    token = paid_route.set(route())
    try:
        assert [p.name for p in registry.get_provider_chain("discord", 10)] == ["paid"]
        assert registry.get_first_gemini_provider("discord", 10).name == "paid"
        assert registry.get_provider_chain("discord", 99) == []
        assert registry.get_first_gemini_provider("discord", 99) is None
    finally:
        paid_route.reset(token)


def test_paid_timeout_bounds_entire_executor_and_cannot_use_explicit_free_chain(monkeypatch, tmp_path):
    config = configured(monkeypatch, tmp_path)

    async def scenario():
        called = []

        async def blocked(provider, *_args, **_kwargs):
            called.append(provider.name)
            await asyncio.sleep(5)

        token = paid_route.set(route(0.03))
        try:
            started = time.monotonic()
            result = await provider_router.call_ai_provider(
                msg=UnifiedMessage("discord", 1, UnifiedChat(10, "chat", "GROUP"), None),
                system_prompt="system",
                prompt="hello",
                media_list=[],
                executor=blocked,
                provider_chain=[config.ai.providers["free"]],
            )
            assert result == (None, [])
            assert called == ["paid"]
            assert time.monotonic() - started < 1
        finally:
            paid_route.reset(token)

    asyncio.run(scenario())


def test_paid_media_description_uses_paid_gemini(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from shin_ai.providers import gemini

    api = AsyncMock(return_value=("a cat", []))
    monkeypatch.setattr(gemini, "gemini_api", api)

    async def scenario():
        token = paid_route.set(route())
        try:
            description = await provider_router.describe_media_with_gemini(
                "look",
                [{}],
                msg=UnifiedMessage("discord", 1, UnifiedChat(10, "chat", "GROUP"), None),
            )
            assert description == "a cat"
            assert api.call_args.kwargs["provider"].name == "paid"
        finally:
            paid_route.reset(token)

    asyncio.run(scenario())


def test_paid_input_limit_rejects_before_any_provider_call(monkeypatch, tmp_path):
    config = configured(monkeypatch, tmp_path)
    config = dataclasses.replace(config, billing=dataclasses.replace(config.billing, max_input_chars=5))
    monkeypatch.setattr(provider_router, "get_settings", lambda: config)
    execute = AsyncMock()

    async def scenario():
        token = paid_route.set(route())
        try:
            assert await provider_router.call_ai_provider(
                msg=UnifiedMessage("discord", 1, UnifiedChat(10, "chat", "GROUP"), None),
                system_prompt="system",
                prompt="hello",
                media_list=[],
                executor=execute,
            ) == (None, [])
            execute.assert_not_awaited()
        finally:
            paid_route.reset(token)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "billing,match",
    [
        ({"enabled": True}, "at least one"),
        ({"enabled": "false"}, "boolean"),
        ({"providers": ["missing"]}, "unknown provider"),
        ({"providers": ["free"]}, "cannot appear"),
        ({"response_cost": 0}, "replaced"),
        ({"response_cost": 1.5}, "replaced"),
        ({"global_timeout_seconds": float("inf")}, "finite"),
    ],
)
def test_invalid_billing_settings_are_rejected(billing, match):
    raw = {
        "ai": {
            "primary": "free",
            "providers": [
                {"name": "free", "type": "gemini", "models": ["free-model"], "api_keys": {"f": "key"}},
            ],
        },
        "billing": billing,
    }
    with pytest.raises(ValueError, match=match):
        settings.parse_settings(raw)


@pytest.mark.parametrize("paid", [False, True])
def test_openai_provider_limits_paid_output_only(monkeypatch, tmp_path, paid):
    from types import SimpleNamespace

    from shin_ai.providers import openai_compatible

    config = configured(monkeypatch, tmp_path)
    monkeypatch.setattr(openai_compatible, "get_settings", lambda: config)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=object())))
    monkeypatch.setattr(openai_compatible, "_get_client", AsyncMock(return_value=client))
    monkeypatch.setattr(openai_compatible, "_get_semaphore", AsyncMock(return_value=None))
    completion = AsyncMock(return_value=("hello", []))
    monkeypatch.setattr(openai_compatible, "run_tool_calling_chat", completion)

    async def scenario():
        token = paid_route.set(route() if paid else None)
        try:
            provider = settings.ProviderSettings(
                name="paid-openai",
                type="openai",
                base_url="https://provider.invalid/v1",
                api_key="test-key",
                model="test-model",
            )
            await openai_compatible.openai_provider(provider, "system", "hello")
        finally:
            paid_route.reset(token)

    asyncio.run(scenario())
    if paid:
        assert completion.call_args.kwargs["max_tokens"] == config.billing.max_output_tokens
    else:
        assert "max_tokens" not in completion.call_args.kwargs


def test_native_gemini_paid_output_limit(monkeypatch, tmp_path):
    from shin_ai.providers import gemini

    config = configured(monkeypatch, tmp_path)
    monkeypatch.setattr(gemini, "get_settings", lambda: config)
    token = paid_route.set(route())
    try:
        assert gemini._build_gemini_config("system", "gemini-test").max_output_tokens == 1024
    finally:
        paid_route.reset(token)
