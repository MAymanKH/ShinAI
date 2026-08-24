"""Bounded, process-local short-term conversation context."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from shin_ai.platforms.models import UnifiedMessage, UnifiedUser
from shin_ai.settings import get_settings
from shin_ai.utils.chat_identity import chat_scope_key


@dataclass(slots=True)
class _ChatContext:
    messages: deque[dict]
    last_access: float


class ContextBuffer:
    """LRU/TTL bounded message history keyed by platform and chat."""

    def __init__(
        self,
        *,
        max_chats: int,
        messages_per_chat: int,
        ttl_seconds: float,
        max_text_chars: int = 4_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_chats = max_chats
        self.messages_per_chat = messages_per_chat
        self.ttl_seconds = ttl_seconds
        self.max_text_chars = max_text_chars
        self._clock = clock
        self._chats: OrderedDict[str, _ChatContext] = OrderedDict()

    def __len__(self) -> int:
        return len(self._chats)

    def _prune(self, now: float) -> None:
        while self._chats:
            first_key = next(iter(self._chats))
            if now - self._chats[first_key].last_access <= self.ttl_seconds:
                break
            self._chats.popitem(last=False)
        while len(self._chats) > self.max_chats:
            self._chats.popitem(last=False)

    def append(self, chat_key: str, entry: dict) -> None:
        now = self._clock()
        self._prune(now)
        context = self._chats.get(chat_key)
        if context is None:
            context = _ChatContext(deque(maxlen=self.messages_per_chat), now)
            self._chats[chat_key] = context
        stored_entry = dict(entry)
        text = stored_entry.get("text")
        if isinstance(text, str) and len(text) > self.max_text_chars:
            stored_entry["text"] = text[: self.max_text_chars]
        context.messages.append(stored_entry)
        context.last_access = now
        self._chats.move_to_end(chat_key)
        self._prune(now)

    def snapshot(self, chat_key: str) -> list[dict]:
        now = self._clock()
        self._prune(now)
        context = self._chats.get(chat_key)
        if context is None:
            return []
        context.last_access = now
        self._chats.move_to_end(chat_key)
        return list(context.messages)


_context_buffer: ContextBuffer | None = None


def _get_buffer() -> ContextBuffer:
    """Build the buffer on first use so importing this module reads no config."""
    global _context_buffer
    if _context_buffer is None:
        runtime = get_settings().runtime
        _context_buffer = ContextBuffer(
            max_chats=runtime.context_max_chats,
            messages_per_chat=runtime.context_messages_per_chat,
            ttl_seconds=runtime.context_ttl_seconds,
            max_text_chars=runtime.context_message_chars,
        )
    return _context_buffer


def reset_context_buffer() -> None:
    """Drop the buffer; used by tests and by settings reloads."""
    global _context_buffer
    _context_buffer = None


def _get_chat_key(platform: str, chat_id: int | str) -> str:
    return chat_scope_key(platform, platform, chat_id)


def add_message_to_context(msg: UnifiedMessage) -> None:
    if not msg.chat or not msg.from_user:
        return

    user_name = msg.from_user.first_name
    if msg.from_user.username:
        user_name += f" (@{msg.from_user.username})"

    replied_to_id = msg.reply_to_message.id if msg.reply_to_message else None
    replied_to_user = (
        msg.reply_to_message.from_user.first_name
        if msg.reply_to_message and msg.reply_to_message.from_user
        else None
    )

    media_type = None
    if msg.photo:
        media_type, text_content = "photo", msg.caption or "[Photo]"
    elif msg.sticker:
        emoji = msg.sticker.emoji or ""
        media_type, text_content = f"sticker {emoji}".strip(), f"[Sticker {emoji}]"
    elif msg.video:
        media_type, text_content = "video", msg.caption or "[Video]"
    elif msg.animation:
        media_type, text_content = "animation", msg.caption or "[GIF/Animation]"
    elif msg.voice:
        media_type, text_content = "voice", "[Voice Message]"
    elif msg.audio:
        media_type, text_content = "audio", "[Audio]"
    else:
        text_content = msg.text or msg.caption or "[Other Media]"

    _get_buffer().append(
        _get_chat_key(msg.platform, msg.chat.id),
        {
            "platform": msg.platform,
            "msg_id": msg.id,
            "user_id": msg.from_user.id,
            "user_name": user_name,
            "text": text_content,
            "media_type": media_type,
            "reply_to_id": replied_to_id,
            "reply_to_user": replied_to_user,
            "timestamp": msg.date or time.time(),
        },
    )


def add_bot_message_to_context(
    *,
    platform: str,
    chat_id: int | str,
    msg_id: int | str,
    text: str | None,
    bot_user: UnifiedUser,
    reply_to_id: int | str | None = None,
    reply_to_user: str | None = None,
    media_type: str | None = None,
    timestamp: float | None = None,
) -> None:
    if not bot_user:
        return

    user_name = bot_user.first_name or "Bot"
    if bot_user.username:
        user_name += f" (@{bot_user.username})"
    if media_type and not text:
        text_content = (
            "[Sticker]"
            if media_type.startswith("sticker")
            else ("[Photo]" if media_type == "photo" else "[Media]")
        )
    else:
        text_content = text or ""

    _get_buffer().append(
        _get_chat_key(platform, chat_id),
        {
            "platform": platform,
            "msg_id": msg_id,
            "user_id": bot_user.id,
            "user_name": user_name,
            "text": text_content,
            "media_type": media_type,
            "reply_to_id": reply_to_id,
            "reply_to_user": reply_to_user,
            "timestamp": timestamp or time.time(),
        },
    )


def get_recent_context_string(
    platform: str,
    chat_id: int | str,
    current_msg_id: int | str | None = None,
) -> str:
    messages = _get_buffer().snapshot(_get_chat_key(platform, chat_id))
    lines = []
    for message in messages:
        if current_msg_id is not None and str(message["msg_id"]) == str(current_msg_id):
            continue
        try:
            time_string = datetime.fromtimestamp(message["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, TypeError, ValueError):
            time_string = "Unknown Time"
        user_label = message["user_name"]
        if message["reply_to_user"]:
            user_label += f" (replying to {message['reply_to_user']})"
        lines.append(f"[{time_string}] [{user_label}] (id:{message['msg_id']}): {message['text']}")
    return "\n".join(lines)


def _recent_media(
    platform: str,
    chat_id: int | str,
    *,
    allowed: Callable[[str | None], bool],
    max_count: int,
) -> list[dict]:
    result = []
    for message in reversed(_get_buffer().snapshot(_get_chat_key(platform, chat_id))):
        if allowed(message.get("media_type")):
            result.append(
                {
                    "msg_id": message["msg_id"],
                    "user_name": message["user_name"],
                    "media_type": message["media_type"],
                    "timestamp": message["timestamp"],
                }
            )
            if len(result) >= max_count:
                break
    return result


def get_recent_media_messages(platform: str, chat_id: int | str, max_count: int = 10) -> list[dict]:
    return _recent_media(
        platform,
        chat_id,
        allowed=lambda media_type: (
            media_type == "photo" or bool(media_type and media_type.startswith("sticker"))
        ),
        max_count=max_count,
    )


def get_recent_audio_messages(platform: str, chat_id: int | str, max_count: int = 10) -> list[dict]:
    return _recent_media(
        platform,
        chat_id,
        allowed=lambda media_type: media_type in {"voice", "audio"},
        max_count=max_count,
    )
