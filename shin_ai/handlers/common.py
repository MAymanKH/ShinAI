import random
from collections.abc import Callable

from shin_ai.config import RANDOM_TRIGGER_PROBABILITY
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.services.replies import check_reply_chain, check_and_clear_next_message_watch


SUPPORTED_CHAT_TYPES = {"PRIVATE", "GROUP", "SUPERGROUP"}


def _message_text(msg: UnifiedMessage) -> str:
    return (msg.text or msg.caption or "").strip()


def _has_supported_media(msg: UnifiedMessage) -> bool:
    return bool(
        msg.photo
        or msg.sticker
        or msg.voice
        or msg.audio
        or msg.video
        or msg.animation
        or msg.document
    )


def is_supported_chat(msg: UnifiedMessage) -> bool:
    if not msg.chat:
        return False
    return str(msg.chat.type).upper() in SUPPORTED_CHAT_TYPES


def is_system_broadcast(msg: UnifiedMessage) -> bool:
    if msg.platform != "whatsapp":
        return False
    return str(msg.chat.id).lower() == "status@broadcast"


def should_record_context(msg: UnifiedMessage) -> bool:
    if not msg.from_user or msg.from_user.is_self:
        return False
    if not is_supported_chat(msg):
        return False
    if is_system_broadcast(msg):
        return False
    return True


async def should_respond_to_message(
    msg: UnifiedMessage,
    debug_hook: Callable[[str, str], None] | None = None,
) -> bool:
    def _debug(reason: str) -> None:
        if debug_hook:
            debug_hook(reason, _message_text(msg))

    if not msg.from_user or msg.from_user.is_self:
        _debug("skip:self_or_missing_sender")
        return False

    if not is_supported_chat(msg):
        _debug("skip:unsupported_chat")
        return False

    if is_system_broadcast(msg):
        _debug("skip:system_broadcast")
        return False

    is_next = check_and_clear_next_message_watch(msg.platform, msg.chat.id)

    text = _message_text(msg)

    if str(msg.chat.type).upper() == "PRIVATE":
        if text.startswith("/"):
            _debug("skip:private_command")
            return False
        _debug("pass:private")
        return True

    is_bot_reply = await check_reply_chain(msg)

    if is_next and not msg.reply_to_message_id:
        msg.is_speculative_reply = True
        _debug("pass:speculative_next_message")
        return True

    if not text:
        if not _has_supported_media(msg):
            _debug("skip:no_text_no_supported_media")
            return False
        if is_bot_reply:
            _debug("pass:reply_chain_media")
            return True
        if msg.mentioned:
            _debug("pass:mentioned_media")
            return True
        _debug("skip:media_without_reply_chain")
        return False

    if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
        _debug("pass:keyword")
        return True

    if msg.mentioned:
        _debug("pass:mentioned")
        return True

    if is_bot_reply:
        _debug("pass:reply_chain")
        return True

    if random.random() < RANDOM_TRIGGER_PROBABILITY:
        _debug("pass:random")
        return True

    _debug("skip:no_trigger")
    return False
