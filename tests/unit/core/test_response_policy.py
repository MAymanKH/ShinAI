from shin_ai.core.response_policy import (
    is_trivial_message,
    parse_model_response,
    split_reply_messages,
)
from shin_ai.platforms.models import (
    UnifiedChat,
    UnifiedMedia,
    UnifiedMessage,
    UnifiedUser,
)


def _message(text: str | None = None, *, sticker: bool = False) -> UnifiedMessage:
    return UnifiedMessage(
        platform="telegram",
        id=1,
        chat=UnifiedChat(id=2, title="Test", type="GROUP"),
        from_user=UnifiedUser(id=3, username="user", first_name="User", is_self=False),
        text=text,
        sticker=UnifiedMedia(type="STICKER", id="sticker") if sticker else None,
    )


def test_pure_skip_variants_never_become_visible_text() -> None:
    for answer in ("[SKIP]", "[skip].", "`[skip]`.", "SKIP"):
        decision = parse_model_response(answer, has_actions=False)
        assert decision.skip_token_found is True
        assert decision.skips_all_text is True
        assert decision.messages == ()


def test_skip_token_with_real_text_preserves_the_text() -> None:
    leading = parse_model_response("[SKIP] Still send this", has_actions=True)
    trailing = parse_model_response("Still send this [SKIP]", has_actions=True)

    assert leading.messages == (("Still send this", None),)
    assert trailing.messages == (("Still send this", None),)
    assert not leading.skips_all_text
    assert not trailing.skips_all_text


def test_natural_use_of_skip_is_not_treated_as_a_control_token() -> None:
    decision = parse_model_response("Please don't skip", has_actions=False)

    assert decision.skip_token_found is False
    assert decision.messages == (("Please don't skip", None),)


def test_action_meta_commentary_is_filtered_without_dropping_real_text() -> None:
    decision = parse_model_response(
        "(No further action needed as the sticker was sent).---Hope you like it",
        has_actions=True,
    )

    assert decision.messages == (("Hope you like it", None),)
    assert decision.filtered_meta_messages == 1


def test_meta_commentary_is_not_filtered_without_an_action() -> None:
    decision = parse_model_response(
        "No further action needed",
        has_actions=False,
    )

    assert decision.messages == (("No further action needed", None),)
    assert decision.filtered_meta_messages == 0


def test_reply_targets_are_parsed_per_message() -> None:
    assert split_reply_messages("[REPLY_TO:123] First---[reply_to:ABC_9] Second---Third") == [
        ("First", "123"),
        ("Second", "ABC_9"),
        ("Third", None),
    ]


def test_trivial_input_detection_handles_stickers_laughter_and_emoji() -> None:
    assert is_trivial_message(_message(sticker=True))
    assert is_trivial_message(_message("هههههه"))
    assert is_trivial_message(_message("😂😂"))
    assert not is_trivial_message(_message("😂 that was good"))
    assert not is_trivial_message(_message("Can you help?"))
