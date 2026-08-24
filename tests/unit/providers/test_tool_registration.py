"""The system prompt advertises a fixed tool set; every provider must honour it."""

import asyncio

import pytest

from shin_ai.providers.tool_loop import (
    ASK_GEMINI_ABOUT_IMAGE_TOOL_SCHEMA,
    TOOLS,
    ask_gemini_about_image,
)
from shin_ai.utils.action_tools import ACTION_TOOL_HANDLERS, POST_ACTION_TOOL_REMINDER

# Tools named in the static system prompt under "TOOLS & CAPABILITIES".
ADVERTISED_TOOLS = {
    "search_web_tool",
    "memory_lookup_tool",
    "ask_gemini_about_image",
    "transcribe_audio",
    "send_reaction",
    "send_sticker",
    "moderate_user",
}


def _tool_names(schemas) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def test_every_advertised_tool_has_a_schema() -> None:
    available = _tool_names(TOOLS) | _tool_names([ASK_GEMINI_ABOUT_IMAGE_TOOL_SCHEMA])
    assert ADVERTISED_TOOLS <= available


def test_action_tools_all_have_handlers() -> None:
    assert set(ACTION_TOOL_HANDLERS) == {"send_reaction", "send_sticker", "moderate_user"}


def test_post_action_reminder_has_one_definition() -> None:
    from shin_ai.providers import gemini, tool_loop

    assert gemini.POST_ACTION_TOOL_REMINDER is POST_ACTION_TOOL_REMINDER
    assert tool_loop.POST_ACTION_TOOL_REMINDER is POST_ACTION_TOOL_REMINDER


class TestAskGeminiAboutImage:
    def test_reports_when_no_media_is_attached(self) -> None:
        result = asyncio.run(ask_gemini_about_image("what is this?", None))
        assert "No image" in result

    def test_does_not_redeclare_itself_on_the_nested_call(self, monkeypatch) -> None:
        """Guards against the tool being offered to the call it makes itself."""
        from shin_ai.providers import gemini

        captured = {}

        async def fake_gemini_api(system_prompt, prompt, *, media_list=None, **kwargs):
            captured.update(kwargs)
            return "a cat", []

        monkeypatch.setattr(gemini, "gemini_api", fake_gemini_api)
        result = asyncio.run(ask_gemini_about_image("what is this?", [{"bytes": b"x"}]))

        assert result == "a cat"
        assert captured["allow_image_tool"] is False

    def test_returns_an_error_string_instead_of_raising(self, monkeypatch) -> None:
        from shin_ai.providers import gemini

        async def boom(*_args, **_kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(gemini, "gemini_api", boom)
        result = asyncio.run(ask_gemini_about_image("what is this?", [{"bytes": b"x"}]))
        assert "provider exploded" in result


@pytest.mark.parametrize("schema", [*TOOLS, ASK_GEMINI_ABOUT_IMAGE_TOOL_SCHEMA])
def test_schemas_are_well_formed(schema) -> None:
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] and function["description"]
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    for required in parameters.get("required", []):
        assert required in parameters["properties"]
