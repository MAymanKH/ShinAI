"""The trigger matrix: whether the bot speaks at all.

Every decision here is reached before any provider call, so a mistake either
makes the bot silent or makes it talk over people. These were previously
untestable -- the module could not be imported without a live config.
"""

import asyncio

import pytest

from shin_ai.handlers import common
from shin_ai.handlers.common import (
    is_supported_chat,
    is_system_broadcast,
    should_record_context,
    should_respond_to_message,
)
from shin_ai.platforms.models import UnifiedChat, UnifiedMedia, UnifiedMessage, UnifiedUser


def _user(user_id=7, username="ahmad", is_self=False) -> UnifiedUser:
    return UnifiedUser(id=user_id, username=username, first_name="Ahmad", is_self=is_self)


def _message(
    *,
    platform="telegram",
    text=None,
    caption=None,
    chat_type="GROUP",
    chat_id=-100,
    from_user=...,
    mentioned=False,
    reply_to_message_id=None,
    **media,
) -> UnifiedMessage:
    return UnifiedMessage(
        platform=platform,
        id=1,
        chat=UnifiedChat(id=chat_id, title="Group", type=chat_type),
        from_user=_user() if from_user is ... else from_user,
        text=text,
        caption=caption,
        mentioned=mentioned,
        reply_to_message_id=reply_to_message_id,
        **media,
    )


@pytest.fixture
def triggers(monkeypatch):
    """Drive should_respond_to_message with the two async lookups stubbed."""

    state = {"is_next": False, "is_bot_reply": False, "random": 1.0, "reasons": []}

    async def fake_watch(*_args, **_kwargs):
        return state["is_next"]

    async def fake_chain(*_args, **_kwargs):
        return state["is_bot_reply"]

    monkeypatch.setattr(common, "check_and_clear_next_message_watch", fake_watch)
    monkeypatch.setattr(common, "check_reply_chain", fake_chain)
    monkeypatch.setattr(common.random, "random", lambda: state["random"])

    def _run(msg, **overrides):
        state.update(overrides)
        state["reasons"] = []
        result = asyncio.run(
            should_respond_to_message(
                msg,
                coordination_scope="scope",
                debug_hook=lambda reason, _text: state["reasons"].append(reason),
            )
        )
        return result, state["reasons"][-1] if state["reasons"] else None

    return _run


class TestChatEligibility:
    @pytest.mark.parametrize("chat_type", ["PRIVATE", "GROUP", "SUPERGROUP"])
    def test_supported_chat_types(self, chat_type) -> None:
        assert is_supported_chat(_message(chat_type=chat_type)) is True

    @pytest.mark.parametrize("chat_type", ["CHANNEL", "BOT", "unknown"])
    def test_unsupported_chat_types(self, chat_type) -> None:
        assert is_supported_chat(_message(chat_type=chat_type)) is False

    def test_chat_type_comparison_is_case_insensitive(self) -> None:
        assert is_supported_chat(_message(chat_type="supergroup")) is True

    def test_whatsapp_status_broadcast_is_a_system_chat(self) -> None:
        msg = _message(platform="whatsapp", chat_id="status@broadcast")
        assert is_system_broadcast(msg) is True

    def test_same_chat_id_on_another_platform_is_not_a_broadcast(self) -> None:
        msg = _message(platform="telegram", chat_id="status@broadcast")
        assert is_system_broadcast(msg) is False


class TestContextRecording:
    def test_records_ordinary_messages(self) -> None:
        assert should_record_context(_message(text="hi")) is True

    def test_never_records_the_bots_own_messages(self) -> None:
        assert should_record_context(_message(text="hi", from_user=_user(is_self=True))) is False

    def test_skips_messages_with_no_sender(self) -> None:
        assert should_record_context(_message(text="hi", from_user=None)) is False

    def test_skips_whatsapp_status_broadcasts(self) -> None:
        msg = _message(platform="whatsapp", chat_id="status@broadcast", text="hi")
        assert should_record_context(msg) is False


class TestPrivateChats:
    def test_always_responds(self, triggers) -> None:
        result, reason = triggers(_message(chat_type="PRIVATE", text="anything"))
        assert result is True
        assert reason == "pass:private"

    def test_ignores_slash_commands(self, triggers) -> None:
        result, reason = triggers(_message(chat_type="PRIVATE", text="/start"))
        assert result is False
        assert reason == "skip:private_command"

    def test_responds_to_media_with_no_text(self, triggers) -> None:
        msg = _message(chat_type="PRIVATE", photo=UnifiedMedia(type="PHOTO", id="p"))
        assert triggers(msg)[0] is True


class TestSenderGuards:
    def test_ignores_its_own_messages(self, triggers) -> None:
        result, reason = triggers(_message(text="hi", from_user=_user(is_self=True)))
        assert result is False
        assert reason == "skip:self_or_missing_sender"

    def test_ignores_messages_with_no_sender(self, triggers) -> None:
        result, _reason = triggers(_message(text="hi", from_user=None))
        assert result is False

    def test_ignores_unsupported_chats(self, triggers) -> None:
        result, reason = triggers(_message(text="hi", chat_type="CHANNEL"))
        assert result is False
        assert reason == "skip:unsupported_chat"


class TestGroupTriggers:
    def test_responds_when_mentioned(self, triggers) -> None:
        result, reason = triggers(_message(text="hey bot", mentioned=True))
        assert result is True
        assert reason == "pass:mentioned"

    def test_responds_to_the_arabic_keyword(self, triggers) -> None:
        result, reason = triggers(_message(text="يالبوت رد عليا"))
        assert result is True
        assert reason == "pass:keyword"

    def test_feminine_form_alone_does_not_trigger(self, triggers) -> None:
        """ "يالبوتة" contains the keyword as a prefix but addresses someone else."""
        result, reason = triggers(_message(text="يالبوتة"))
        assert result is False
        assert reason == "skip:no_trigger"

    def test_keyword_still_wins_when_both_forms_appear(self, triggers) -> None:
        assert triggers(_message(text="يالبوت مش يالبوتة"))[0] is True

    def test_responds_inside_a_reply_chain(self, triggers) -> None:
        result, reason = triggers(_message(text="and then?"), is_bot_reply=True)
        assert result is True
        assert reason == "pass:reply_chain"

    def test_stays_quiet_for_unrelated_chatter(self, triggers) -> None:
        result, reason = triggers(_message(text="lunch anyone"))
        assert result is False
        assert reason == "skip:no_trigger"

    def test_random_interjection_fires_below_the_probability(self, triggers) -> None:
        result, reason = triggers(_message(text="lunch anyone"), random=0.0)
        assert result is True
        assert reason == "pass:random"


class TestSpeculativeReplies:
    def test_first_message_after_the_bot_speaks_is_speculative(self, triggers) -> None:
        msg = _message(text="ok cool")
        result, reason = triggers(msg, is_next=True)
        assert result is True
        assert reason == "pass:speculative_next_message"
        assert msg.is_speculative_reply is True

    def test_a_reply_to_someone_else_is_not_speculative(self, triggers) -> None:
        """An explicit reply target means the message is aimed at that message."""
        msg = _message(text="ok cool", reply_to_message_id=99)
        result, _ = triggers(msg, is_next=True)
        assert msg.is_speculative_reply is False
        assert result is False

    def test_speculative_flag_defaults_off(self) -> None:
        assert _message(text="hi").is_speculative_reply is False


class TestMediaWithoutText:
    @pytest.mark.parametrize(
        "media",
        [
            {"photo": UnifiedMedia(type="PHOTO", id="p")},
            {"sticker": UnifiedMedia(type="STICKER", id="s")},
            {"voice": UnifiedMedia(type="VOICE", id="v")},
            {"audio": UnifiedMedia(type="AUDIO", id="a")},
            {"video": UnifiedMedia(type="VIDEO", id="vi")},
            {"animation": UnifiedMedia(type="ANIMATION", id="g")},
            {"document": UnifiedMedia(type="DOCUMENT", id="d")},
        ],
    )
    def test_media_alone_is_ignored_without_a_reason_to_reply(self, triggers, media) -> None:
        result, reason = triggers(_message(**media))
        assert result is False
        assert reason == "skip:media_without_reply_chain"

    def test_media_in_a_reply_chain_is_answered(self, triggers) -> None:
        msg = _message(photo=UnifiedMedia(type="PHOTO", id="p"))
        result, reason = triggers(msg, is_bot_reply=True)
        assert result is True
        assert reason == "pass:reply_chain_media"

    def test_media_with_a_mention_is_answered(self, triggers) -> None:
        msg = _message(photo=UnifiedMedia(type="PHOTO", id="p"), mentioned=True)
        result, reason = triggers(msg, is_bot_reply=False)
        assert result is True
        assert reason == "pass:mentioned_media"

    def test_empty_message_with_no_media_is_ignored(self, triggers) -> None:
        result, reason = triggers(_message(text="   "))
        assert result is False
        assert reason == "skip:no_text_no_supported_media"

    def test_a_caption_counts_as_text(self, triggers) -> None:
        msg = _message(caption="يالبوت", photo=UnifiedMedia(type="PHOTO", id="p"))
        assert triggers(msg)[0] is True


class TestWhatsAppBroadcasts:
    def test_status_broadcasts_never_get_a_reply(self, triggers) -> None:
        msg = _message(platform="whatsapp", chat_id="status@broadcast", text="يالبوت")
        result, reason = triggers(msg)
        assert result is False
        assert reason == "skip:system_broadcast"
