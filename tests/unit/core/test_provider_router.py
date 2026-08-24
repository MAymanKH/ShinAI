import asyncio
from types import SimpleNamespace

from shin_ai.core.provider_router import call_ai_provider


def run(coro):
    return asyncio.run(coro)


def _provider(name: str, provider_type: str = "openai"):
    return SimpleNamespace(name=name, type=provider_type)


def test_router_retries_then_falls_back() -> None:
    async def scenario() -> None:
        primary = _provider("primary", "gemini")
        fallback = _provider("fallback", "gemini")
        calls: list[tuple[str, str]] = []

        async def execute(provider, _msg, _system, prompt, _media, _platform, **_kwargs):
            calls.append((provider.name, prompt))
            if provider is primary:
                raise RuntimeError("temporary failure")
            return "ok", []

        answer, actions = await call_ai_provider(
            msg=None,
            system_prompt="system",
            prompt="hello",
            media_list=[],
            provider_chain=[primary, fallback],
            executor=execute,
            max_retries=2,
            attempt_timeout_seconds=1,
            global_timeout_seconds=10,
        )

        assert answer == "ok"
        assert actions == []
        assert [name for name, _ in calls] == ["primary", "primary", "fallback"]
        assert "Previous attempt failed with RuntimeError" in calls[1][1]
        assert calls[2][1] == "hello"

    run(scenario())


def test_router_describes_media_only_once_across_fallbacks() -> None:
    async def scenario() -> None:
        first = _provider("first")
        second = _provider("second")
        descriptions = 0
        prompts: list[str] = []

        async def describe(_prompt, _media):
            nonlocal descriptions
            descriptions += 1
            return "a blue image"

        async def execute(provider, _msg, _system, prompt, media, _platform, **_kwargs):
            prompts.append(prompt)
            assert media == [{"bytes": b"image"}]
            if provider is first:
                raise RuntimeError("provider unavailable")
            return "done", []

        answer, _ = await call_ai_provider(
            msg=None,
            system_prompt="system",
            prompt="what is this?",
            media_list=[{"bytes": b"image"}],
            provider_chain=[first, second],
            executor=execute,
            media_describer=describe,
            max_retries=1,
            attempt_timeout_seconds=1,
            global_timeout_seconds=10,
        )

        assert answer == "done"
        assert descriptions == 1
        assert all("a blue image" in prompt for prompt in prompts)

    run(scenario())


def test_router_accepts_tool_only_response() -> None:
    async def execute(*_args, **_kwargs):
        return "", [{"type": "reaction", "emoji": "👍"}]

    answer, actions = run(
        call_ai_provider(
            msg=None,
            system_prompt="system",
            prompt="hello",
            media_list=[],
            provider_chain=[_provider("gemini", "gemini")],
            executor=execute,
        )
    )

    assert answer == ""
    assert actions == [{"type": "reaction", "emoji": "👍"}]


def test_router_lets_a_slow_provider_finish_its_own_rotation() -> None:
    """The router must not cut a provider off: rotation is the provider's job.

    The executor takes longer than the attempt timeout, which previously killed
    it mid-rotation and left the remaining models untried.
    """

    async def scenario() -> None:
        async def execute(_provider, _msg, _system, _prompt, _media, _platform, **kwargs):
            assert kwargs["attempt_timeout_seconds"] == 0.05
            await asyncio.sleep(0.2)
            return "finished", []

        answer, _ = await call_ai_provider(
            msg=None,
            system_prompt="system",
            prompt="hello",
            media_list=[],
            provider_chain=[_provider("gemini", "gemini")],
            executor=execute,
            max_retries=1,
            attempt_timeout_seconds=0.05,
            global_timeout_seconds=10,
        )

        assert answer == "finished"

    run(scenario())
