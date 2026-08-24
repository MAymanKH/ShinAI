import asyncio

from shin_ai.platforms.models import UnifiedChat, UnifiedMedia, UnifiedMessage, UnifiedUser
from shin_ai.services.media import (
    download_message_media,
    extract_prompt,
    find_audio_in_reply_chain,
    prepare_prompt_and_media,
)


def run(coro):
    return asyncio.run(coro)


def _message(message_id: int, **kwargs) -> UnifiedMessage:
    return UnifiedMessage(
        platform="telegram",
        id=message_id,
        chat=UnifiedChat(id=1, title="chat", type="GROUP"),
        from_user=UnifiedUser(id=2, username="user", first_name="User", is_self=False),
        **kwargs,
    )


class _Platform:
    platform_name = "telegram"

    async def download_media(self, media) -> bytes:
        return f"content-{media.id}".encode()


def test_extract_prompt_uses_text_then_media_placeholder() -> None:
    assert extract_prompt(_message(1, text="hello")) == "hello"
    assert extract_prompt(
        _message(2, photo=UnifiedMedia(type="PHOTO", id="photo"))
    ) == "[User sent a photo]"


def test_download_message_media_preserves_reply_positions() -> None:
    reply = _message(
        1,
        sticker=UnifiedMedia(type="STICKER", id="sticker", emoji="🙂"),
    )
    current = _message(
        2,
        photo=UnifiedMedia(type="PHOTO", id="photo", mime_type="image/png"),
        reply_to_message=reply,
    )

    media = run(download_message_media(_Platform(), current))

    assert [entry["bytes"] for entry in media] == [b"content-photo", b"content-sticker"]
    assert [entry["position"] for entry in media] == ["Current message", "1 messages back"]
    assert media[0]["mime_type"] == "image/png"


def test_prepare_prompt_transcribes_audio_inside_injected_service() -> None:
    voice = UnifiedMedia(type="VOICE", id="voice", mime_type="audio/ogg")
    message = _message(1, text="answer this", voice=voice)

    async def transcriber(loader, mime_type: str) -> str:
        assert mime_type == "audio/ogg"
        assert await loader() == b"content-voice"
        return "heard words"

    prompt, media = run(
        prepare_prompt_and_media(_Platform(), message, transcriber=transcriber)
    )

    assert "heard words" in prompt
    assert prompt.endswith("answer this")
    assert media == []


def test_find_audio_walks_reply_chain() -> None:
    audio = _message(1, audio=UnifiedMedia(type="AUDIO", id="audio"))
    middle = _message(2, reply_to_message=audio)
    current = _message(3, reply_to_message=middle)

    assert find_audio_in_reply_chain(current) is audio
