from collections import deque, defaultdict
import time
from datetime import datetime
from shin_ai.platforms.models import UnifiedMessage, UnifiedMedia, UnifiedUser

# Store last 50 messages per chat
# Map f"{platform}_{chat_id}" -> deque of message dicts
_context_buffer = defaultdict(lambda: deque(maxlen=50))


def _normalize_chat_id(platform: str, chat_id: int | str) -> str:
    raw_chat_id = str(chat_id).strip()
    if platform != "whatsapp":
        return raw_chat_id

    lowered = raw_chat_id.lower()
    if "@" in lowered:
        user, server = lowered.split("@", 1)
        user = user.split(":", 1)[0]
        return f"{user}@{server}"

    return lowered.split(":", 1)[0]


def _get_chat_key(platform: str, chat_id: int | str) -> str:
    normalized_chat_id = _normalize_chat_id(platform, chat_id)
    return f"{platform}_{normalized_chat_id}"

def add_message_to_context(msg: UnifiedMessage):
    """
    Adds a message to the short-term context buffer.
    """
    if not msg.chat or not msg.from_user:
        return

    user_name = msg.from_user.first_name
    if msg.from_user.username:
        user_name += f" (@{msg.from_user.username})"
    
    replied_to_id = None
    replied_to_user = None
    
    if msg.reply_to_message:
        replied_to_id = msg.reply_to_message.id
        if msg.reply_to_message.from_user:
            replied_to_user = msg.reply_to_message.from_user.first_name

    # Determine media type
    media_type = None
    if msg.photo:
        media_type = "photo"
        text_content = msg.caption or "[Photo]"
    elif msg.sticker:
        emoji = msg.sticker.emoji or ""
        media_type = f"sticker {emoji}".strip()
        text_content = f"[Sticker {emoji}]"
    elif msg.video:
        media_type = "video"
        text_content = msg.caption or "[Video]"
    elif msg.animation:
        media_type = "animation"
        text_content = msg.caption or "[GIF/Animation]"
    else:
        text_content = msg.text or msg.caption or "[Other Media]"

    entry = {
        "platform": msg.platform,
        "msg_id": msg.id,
        "user_id": msg.from_user.id,
        "user_name": user_name,
        "text": text_content,
        "media_type": media_type,
        "reply_to_id": replied_to_id,
        "reply_to_user": replied_to_user,
        "timestamp": msg.date or time.time()
    }
    
    chat_key = _get_chat_key(msg.platform, msg.chat.id)
    _context_buffer[chat_key].append(entry)


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
    """
    Adds an outgoing bot message to the short-term context buffer.
    """
    if not bot_user:
        return

    user_name = bot_user.first_name or "Bot"
    if bot_user.username:
        user_name += f" (@{bot_user.username})"

    if media_type and not text:
        if media_type.startswith("sticker"):
            text_content = "[Sticker]"
        elif media_type == "photo":
            text_content = "[Photo]"
        else:
            text_content = "[Media]"
    else:
        text_content = text or ""

    entry = {
        "platform": platform,
        "msg_id": msg_id,
        "user_id": bot_user.id,
        "user_name": user_name,
        "text": text_content,
        "media_type": media_type,
        "reply_to_id": reply_to_id,
        "reply_to_user": reply_to_user,
        "timestamp": timestamp or time.time(),
    }

    chat_key = _get_chat_key(platform, chat_id)
    _context_buffer[chat_key].append(entry)

def get_recent_context_string(platform: str, chat_id: int | str, current_msg_id: int | str = None) -> str:
    """
    Returns a formatted string of the recent conversation history.
    Excludes the current message if provided (to avoid duplication in prompt).
    Each message is tagged with its actual message ID (id:XXXXX)
    so the AI can directly reference them for targeting.
    """
    chat_key = _get_chat_key(platform, chat_id)
    if chat_key not in _context_buffer:
        return ""

    lines = []
    msgs = list(_context_buffer[chat_key])
    
    for m in msgs:
        if current_msg_id and str(m["msg_id"]) == str(current_msg_id):
            continue
            
        # Format Timestamp
        try:
            ts = m["timestamp"]
            dt_obj = datetime.fromtimestamp(ts)
            time_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = "Unknown Time"

        # Format: [Time] [User Name] (id:XXXXX): Message
        prefix = f"[{time_str}] [{m['user_name']}]"
        if m['reply_to_user']:
            prefix = f"[{time_str}] [{m['user_name']} (replying to {m['reply_to_user']})]"
        
        # Embed the actual message ID
        prefix += f" (id:{m['msg_id']})"
            
        lines.append(f"{prefix}: {m['text']}")
    
    return "\n".join(lines)

def get_recent_media_messages(platform: str, chat_id: int | str, max_count: int = 10) -> list[dict]:
    """
    Returns a list of recent messages that contain photos or stickers.
    Limited to max_count most recent media messages.
    
    Returns list of dicts with: msg_id, user_name, media_type, timestamp
    """
    chat_key = _get_chat_key(platform, chat_id)
    if chat_key not in _context_buffer:
        return []
    
    media_messages = []
    # Iterate in reverse to get most recent first
    for m in reversed(list(_context_buffer[chat_key])):
        if m.get("media_type") and m["media_type"] in ["photo"] or (m.get("media_type") and m["media_type"].startswith("sticker")):
            media_messages.append({
                "msg_id": m["msg_id"],
                "user_name": m["user_name"],
                "media_type": m["media_type"],
                "timestamp": m["timestamp"]
            })
            
            if len(media_messages) >= max_count:
                break
    
    return media_messages
