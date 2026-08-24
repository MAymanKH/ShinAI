"""Media discovery, download, and audio-to-prompt preparation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.services.audio_transcriber import transcribe_audio_source
from shin_ai.utils.context_manager import get_recent_media_messages
from shin_ai.utils.logger_config import logger

Transcriber = Callable[[Callable[[], Awaitable[bytes]], str], Awaitable[str]]


async def prepare_prompt_and_media(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    *,
    transcriber: Transcriber = transcribe_audio_source,
) -> tuple[str, list[dict]]:
    prompt = extract_prompt(msg)
    media = await download_message_media(platform, msg)
    prompt = await attach_audio_transcription(
        platform,
        msg,
        prompt,
        transcriber=transcriber,
    )
    if not media:
        media.extend(await download_mentioned_recent_media(platform, msg, prompt))
    return prompt, media


def extract_prompt(msg: UnifiedMessage) -> str:
    prompt = msg.text or msg.caption
    if prompt:
        return prompt
    if msg.sticker:
        return f"[User sent a sticker {msg.sticker.emoji or ''}]"
    if msg.photo:
        return "[User sent a photo]"
    if msg.animation:
        return "[User sent a GIF/Animation]"
    if msg.video:
        return "[User sent a Video]"
    if msg.voice:
        return "[User sent a Voice Message]"
    if msg.audio:
        return "[User sent an Audio file]"
    if msg.document:
        return "[User sent a Document]"
    return " "


async def attach_audio_transcription(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
    *,
    transcriber: Transcriber = transcribe_audio_source,
) -> str:
    audio_msg = msg if msg.voice or msg.audio else find_audio_in_reply_chain(msg)
    if audio_msg is None:
        return prompt

    transcription = await transcribe_audio_message(
        platform,
        audio_msg,
        transcriber=transcriber,
    )
    if not transcription:
        return prompt

    sender_name = audio_msg.from_user.first_name if audio_msg.from_user else "Unknown"
    media_type = "Voice message" if audio_msg.voice else "Audio file"
    source = "from user" if audio_msg is msg else f"from {sender_name} (replied-to message)"
    disclaimer = (
        f"[{media_type} {source} - Transcription]: \"{transcription}\"\n"
        "[TRANSCRIPTION NOTE: The above was transcribed from audio. It may contain "
        "phonetic spelling errors, hallucinated artifacts, or illogical words due to "
        "dialect variations (especially Egyptian Arabic). Before responding, intelligently "
        "interpret any illogical words based on context to find the nearest logical meaning.]"
    )
    return f"{disclaimer}\n\n{prompt}" if prompt.strip() else disclaimer


def find_audio_in_reply_chain(msg: UnifiedMessage) -> UnifiedMessage | None:
    current = msg
    for _ in range(10):
        reply = current.reply_to_message
        if reply is None:
            break
        if reply.voice or reply.audio:
            return reply
        current = reply
    return None


async def download_message_media(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
) -> list[dict]:
    async def download(target: UnifiedMessage):
        sender = (
            target.from_user.username or target.from_user.first_name
            if target.from_user
            else "Unknown"
        )
        if target.photo:
            return (
                await platform.download_media(target.photo),
                target.photo.mime_type or "image/jpeg",
                "photo",
                sender,
            )
        if target.sticker and not target.sticker.is_animated and not target.sticker.is_video:
            return (
                await platform.download_media(target.sticker),
                target.sticker.mime_type or "image/webp",
                f"sticker {target.sticker.emoji or ''}".strip(),
                sender,
            )
        return None

    media = []
    current: UnifiedMessage | None = msg
    depth = 0
    while current is not None and depth <= 10:
        downloaded = await download(current)
        if downloaded is not None and downloaded[0]:
            content, mime_type, media_type, sender = downloaded
            media.append(
                {
                    "bytes": content,
                    "mime_type": mime_type,
                    "sender": sender,
                    "position": "Current message" if depth == 0 else f"{depth} messages back",
                    "media_type": media_type,
                }
            )
        current = current.reply_to_message
        depth += 1
    return media


async def transcribe_audio_message(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    *,
    transcriber: Transcriber = transcribe_audio_source,
) -> str:
    media_handle = msg.voice or msg.audio
    if not media_handle:
        return ""
    mime_type = media_handle.mime_type or "audio/ogg"
    try:
        transcription = await transcriber(
            lambda: platform.download_media(media_handle),
            mime_type,
        )
        if transcription:
            logger.info(
                "Audio transcribed — mime=%s chars=%d preview=\"%s%s\"",
                mime_type,
                len(transcription),
                transcription[:80],
                "..." if len(transcription) > 80 else "",
            )
        else:
            logger.warning("Whisper returned empty transcription — mime=%s", mime_type)
        return transcription
    except Exception as error:
        logger.error("Audio transcription failed: %s", error)
        return ""


async def download_mentioned_recent_media(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
) -> list[dict]:
    keywords = ("image", "photo", "picture", "pic", "sticker", "صورة", "الصورة", "صوره")
    if not any(keyword in prompt.lower() for keyword in keywords):
        return []

    logger.debug("Media mentioned without a reply; checking recent context")
    recent = get_recent_media_messages(platform.platform_name, msg.chat.id, max_count=10)
    return await download_context_media(
        platform,
        msg.chat.id,
        [entry["msg_id"] for entry in recent[:5]],
    )


async def download_context_media(
    platform: PlatformAdapter,
    chat_id: int | str,
    message_ids: list[int | str],
) -> list[dict]:
    media = []
    for index, message_id in enumerate(message_ids):
        message = await platform.get_message(chat_id, message_id)
        if not message:
            continue
        sender = (
            message.from_user.username or message.from_user.first_name
            if message.from_user
            else "Unknown"
        )
        if message.photo:
            content = await platform.download_media(message.photo)
            if content:
                media.append(
                    {
                        "bytes": content,
                        "mime_type": message.photo.mime_type or "image/jpeg",
                        "sender": sender,
                        "position": f"From context msg {index + 1}",
                        "media_type": "photo",
                    }
                )
        elif message.sticker and not message.sticker.is_animated and not message.sticker.is_video:
            content = await platform.download_media(message.sticker)
            if content:
                media.append(
                    {
                        "bytes": content,
                        "mime_type": message.sticker.mime_type or "image/webp",
                        "sender": sender,
                        "position": f"From context msg {index + 1}",
                        "media_type": f"sticker {message.sticker.emoji or ''}",
                    }
                )
    return media
