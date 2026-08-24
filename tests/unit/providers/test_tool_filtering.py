"""A tool the adapter would drop must never be offered.

The tool result tells the model the side-effect "has been performed" and the
system prompt then invites it to reply with [SKIP]. Offering a tool the adapter
silently drops therefore produces total silence, not a degraded reply.
"""

import pytest

from shin_ai.providers.tool_loop import TOOLS, tools_for_platform
from shin_ai.utils.action_tools import MODERATE_USER_TOOL_SCHEMA


class _Adapter:
    def __init__(self, *, stickers=True, restrictions=True, actions=None):
        self.supports_stickers = stickers
        self.supports_member_restrictions = restrictions
        self.supported_moderation_actions = frozenset(
            actions if actions is not None else {"kick", "ban", "unban", "mute", "unmute", "add"}
        )


def _names(tools):
    return {tool["function"]["name"] for tool in tools}


def _actions(tools):
    tool = next(t for t in tools if t["function"]["name"] == "moderate_user")
    return set(tool["function"]["parameters"]["properties"]["action"]["enum"])


class TestStickerFiltering:
    def test_sticker_tool_is_withheld_when_unsupported(self) -> None:
        assert "send_sticker" not in _names(tools_for_platform(TOOLS, _Adapter(stickers=False)))

    def test_sticker_tool_is_offered_when_supported(self) -> None:
        assert "send_sticker" in _names(tools_for_platform(TOOLS, _Adapter(stickers=True)))

    def test_other_tools_are_untouched(self) -> None:
        filtered = _names(tools_for_platform(TOOLS, _Adapter(stickers=False)))
        assert {"search_web_tool", "memory_lookup_tool", "transcribe_audio", "send_reaction"} <= filtered


class TestModerationNarrowing:
    def test_unsupported_actions_are_removed_from_the_enum(self) -> None:
        filtered = tools_for_platform(TOOLS, _Adapter(actions={"kick"}))
        assert _actions(filtered) == {"kick"}

    def test_mute_requires_member_restrictions(self) -> None:
        filtered = tools_for_platform(TOOLS, _Adapter(restrictions=False))
        assert {"mute", "unmute"}.isdisjoint(_actions(filtered))
        assert "kick" in _actions(filtered)

    def test_tool_is_dropped_entirely_when_nothing_is_supported(self) -> None:
        filtered = tools_for_platform(TOOLS, _Adapter(actions=set()))
        assert "moderate_user" not in _names(filtered)

    def test_the_shared_schema_is_never_mutated(self) -> None:
        """Narrowing must copy: TOOLS is module state shared by every request."""
        before = list(MODERATE_USER_TOOL_SCHEMA["function"]["parameters"]["properties"]["action"]["enum"])
        tools_for_platform(TOOLS, _Adapter(actions={"kick"}))
        after = MODERATE_USER_TOOL_SCHEMA["function"]["parameters"]["properties"]["action"]["enum"]
        assert after == before

    def test_full_support_passes_the_schema_through_unchanged(self) -> None:
        filtered = tools_for_platform(TOOLS, _Adapter())
        assert _names(filtered) == _names(TOOLS)
        assert _actions(filtered) == _actions(TOOLS)


class TestNoPlatform:
    def test_every_tool_is_offered_when_there_is_no_platform(self) -> None:
        """Media-description and pre-flight calls run without a bound chat."""
        assert _names(tools_for_platform(TOOLS, None)) == _names(TOOLS)


class TestRealAdapters:
    @pytest.mark.parametrize(
        ("adapter_path", "expected_sticker", "expected_actions"),
        [
            ("shin_ai.platforms.discord:DiscordPlatform", False, {"kick", "ban", "unban", "mute", "unmute"}),
            ("shin_ai.platforms.whatsapp:WhatsAppPlatform", True, {"kick"}),
        ],
    )
    def test_declared_capabilities_drive_the_filter(
        self, adapter_path, expected_sticker, expected_actions
    ) -> None:
        import importlib

        module_name, class_name = adapter_path.split(":")
        adapter = getattr(importlib.import_module(module_name), class_name)

        class _Frozen:
            supports_stickers = adapter.supports_stickers.fget(None)
            supports_member_restrictions = adapter.supports_member_restrictions.fget(None)
            supported_moderation_actions = adapter.supported_moderation_actions.fget(None)

        filtered = tools_for_platform(TOOLS, _Frozen())
        assert ("send_sticker" in _names(filtered)) is expected_sticker
        if expected_actions:
            assert _actions(filtered) == expected_actions

    def test_discord_never_sees_the_sticker_tool(self) -> None:
        """The exact path that produced silence: sticker offered, then dropped."""
        from shin_ai.platforms.discord import DiscordPlatform

        class _Frozen:
            supports_stickers = DiscordPlatform.supports_stickers.fget(None)
            supports_member_restrictions = True
            supported_moderation_actions = DiscordPlatform.supported_moderation_actions.fget(None)

        assert "send_sticker" not in _names(tools_for_platform(TOOLS, _Frozen()))
