from __future__ import annotations

from typing import Optional

from shin_ai.platforms.models import UnifiedMedia, UnifiedMessage, UnifiedMessageEntity, UnifiedUser
from shin_ai.platforms.whatsapp_helpers import extract_local_user_id, media_id, normalize_jid_identity
from shin_ai.platforms.whatsapp_runtime import ContextInfoType, WaMessageType


def unwrap_message(message: WaMessageType) -> WaMessageType:
    current = message

    while True:
        advanced = False

        if current.deviceSentMessage.ListFields() and current.deviceSentMessage.message.ListFields():
            current = current.deviceSentMessage.message
            advanced = True

        for wrapper_name in (
            "ephemeralMessage",
            "viewOnceMessage",
            "viewOnceMessageV2",
            "viewOnceMessageV2Extension",
            "editedMessage",
        ):
            wrapper = getattr(current, wrapper_name, None)
            if wrapper and wrapper.ListFields() and getattr(wrapper, "message", None):
                inner = wrapper.message
                if inner and inner.ListFields():
                    current = inner
                    advanced = True
                    break

        if not advanced:
            break

    return current


def extract_context_info(message: WaMessageType) -> Optional[ContextInfoType]:
    # Scan every populated sub-message field for a contextInfo child.
    # This is intentionally exhaustive: instead of hardcoding a list of
    # known message types, we iterate over ALL fields the protobuf reports
    # as set, and check whether they contain a contextInfo with data.
    for field_descriptor, value in message.ListFields():
        if not hasattr(value, "contextInfo"):
            continue
        try:
            ctx = value.contextInfo
            if ctx and ctx.ListFields():
                return ctx
        except Exception:
            continue

    # Also check top-level contextInfo (some message types place it there).
    top_ctx = getattr(message, "contextInfo", None)
    if top_ctx:
        try:
            if top_ctx.ListFields():
                return top_ctx
        except Exception:
            pass

    return None


def extract_text_and_caption(message: WaMessageType) -> tuple[Optional[str], Optional[str]]:
    text: Optional[str] = None
    caption: Optional[str] = None

    if message.conversation:
        text = message.conversation
    elif message.extendedTextMessage.ListFields():
        text = message.extendedTextMessage.text or None

    if message.imageMessage.ListFields():
        caption = message.imageMessage.caption or None
    elif message.videoMessage.ListFields():
        caption = message.videoMessage.caption or None
    elif message.documentMessage.ListFields():
        caption = message.documentMessage.caption or None

    return text, caption


def apply_media(
    unified_msg: UnifiedMessage,
    message: WaMessageType,
    download_message: WaMessageType = None,
    message_id: int | str | None = None,
) -> None:
    # Use the original (non-unwrapped) message for downloads so that
    # Neonize's download_any() can locate the media URL and encryption
    # keys that may live in outer wrapper layers (ephemeral, view-once, etc.).
    native_payload = {"wa_message": download_message or message}

    media_msg_id = message_id or unified_msg.id

    if message.imageMessage.ListFields():
        mime = message.imageMessage.mimetype or "image/jpeg"
        unified_msg.photo = UnifiedMedia(
            type="PHOTO",
            id=media_id(media_msg_id, "photo"),
            mime_type=mime,
            native_obj=native_payload,
        )
    if message.stickerMessage.ListFields():
        mime = message.stickerMessage.mimetype or "image/webp"
        unified_msg.sticker = UnifiedMedia(
            type="STICKER",
            id=media_id(media_msg_id, "sticker"),
            is_animated=bool(message.stickerMessage.isAnimated),
            mime_type=mime,
            native_obj=native_payload,
        )
    if message.videoMessage.ListFields():
        mime = message.videoMessage.mimetype or "video/mp4"
        unified_msg.video = UnifiedMedia(
            type="VIDEO",
            id=media_id(media_msg_id, "video"),
            mime_type=mime,
            native_obj=native_payload,
        )
    if message.audioMessage.ListFields():
        mime = message.audioMessage.mimetype or "audio/ogg"
        if bool(message.audioMessage.PTT):
            unified_msg.voice = UnifiedMedia(
                type="VOICE",
                id=media_id(media_msg_id, "voice"),
                mime_type=mime,
                native_obj=native_payload,
            )
        else:
            unified_msg.audio = UnifiedMedia(
                type="AUDIO",
                id=media_id(media_msg_id, "audio"),
                mime_type=mime,
                native_obj=native_payload,
            )
    if message.documentMessage.ListFields():
        mime = message.documentMessage.mimetype or "application/octet-stream"
        unified_msg.document = UnifiedMedia(
            type="DOCUMENT",
            id=media_id(media_msg_id, "document"),
            mime_type=mime,
            native_obj=native_payload,
        )


def build_entities_from_context(
    context_info: Optional[ContextInfoType],
    source_text: str,
) -> list[UnifiedMessageEntity]:
    entities: list[UnifiedMessageEntity] = []
    if not context_info:
        return entities

    for mentioned_jid in context_info.mentionedJID:
        normalized_id = normalize_jid_identity(mentioned_jid)
        user_name = extract_local_user_id(mentioned_jid)
        mention_token = user_name or normalized_id
        token = f"@{mention_token}" if mention_token else ""
        offset = source_text.find(token)
        length = len(token) if offset >= 0 else 0
        entities.append(
            UnifiedMessageEntity(
                type="MENTION",
                offset=max(offset, 0),
                length=length,
                user=UnifiedUser(
                    id=normalized_id or user_name or mentioned_jid,
                    username=user_name or None,
                    first_name=user_name or normalized_id or mentioned_jid,
                    is_self=False,
                ),
            )
        )
    return entities


def build_text_caption_entities(
    context_info: Optional[ContextInfoType],
    text: Optional[str],
    caption: Optional[str],
) -> tuple[list[UnifiedMessageEntity], list[UnifiedMessageEntity]]:
    text_entities = build_entities_from_context(context_info, text or "")
    caption_entities = build_entities_from_context(context_info, caption or "")

    # Keep Telegram-like separation: entities map to text, caption_entities map to captions.
    if text:
        caption_entities = []
    elif caption:
        text_entities = []

    return text_entities, caption_entities
