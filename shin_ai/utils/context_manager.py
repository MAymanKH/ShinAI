from collections import deque
from collections import defaultdict
import time
from datetime import datetime
from pyrogram.types import Message

# Store last 50 messages per chat
# Map chat_id -> deque of message dicts
_context_buffer = defaultdict(lambda: deque(maxlen=50))

def add_message_to_context(msg: Message):
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
        "msg_id": msg.id,
        "user_id": msg.from_user.id,
        "user_name": user_name,
        "text": text_content,
        "media_type": media_type,
        "reply_to_id": replied_to_id,
        "reply_to_user": replied_to_user,
        "timestamp": getattr(msg, "date", time.time())
    }
    
    _context_buffer[msg.chat.id].append(entry)

def get_recent_context_string(chat_id: int, current_msg_id: int = None) -> str:
    """
    Returns a formatted string of the recent conversation history.
    Excludes the current message if provided (to avoid duplication in prompt).
    """
    context_str, _ = get_recent_context_with_targets(chat_id, current_msg_id, max_targets=0)
    return context_str


def get_recent_context_with_targets(chat_id: int, current_msg_id: int = None, max_targets: int = 10) -> tuple[str, list[dict]]:
    """
    Returns both formatted context string AND list of messages for targeting in a single pass.
    More efficient than calling separate functions.
    
    Args:
        chat_id: The chat ID to get context from
        current_msg_id: Message ID to exclude (typically the current message)
        max_targets: Maximum number of recent messages to return for targeting (0 to disable)
        
    Returns:
        Tuple of (formatted_context_string, list_of_message_dicts_for_targeting)
    """
    if chat_id not in _context_buffer:
        return "", []

    lines = []
    target_messages = []
    msgs = list(_context_buffer[chat_id])
    
    # First pass: collect target messages (most recent ones)
    msg_to_target = {}  # Map msg_id to target label
    if max_targets > 0:
        target_idx = 1
        for m in reversed(msgs):
            if current_msg_id and m["msg_id"] == current_msg_id:
                continue
            
            msg_to_target[m["msg_id"]] = f"target:msg{target_idx}"
            
            # Truncate text for display in target list
            text = m["text"]
            if len(text) > 50:
                text = text[:47] + "..."
            
            target_messages.append({
                "msg_id": m["msg_id"],
                "user_id": m["user_id"],
                "user_name": m["user_name"],
                "text": text,
                "timestamp": m["timestamp"]
            })
            
            target_idx += 1
            if len(target_messages) >= max_targets:
                break
    
    # Second pass: format context with embedded target tags
    for m in msgs:
        if current_msg_id and m["msg_id"] == current_msg_id:
            continue
            
        # Format Timestamp
        try:
            ts = m["timestamp"]
            dt_obj = datetime.fromtimestamp(ts)
            time_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = "Unknown Time"

        # Format: [Time] [User Name] [target:msgX]: Message
        prefix = f"[{time_str}] [{m['user_name']}]"
        if m['reply_to_user']:
            prefix = f"[{time_str}] [{m['user_name']} (replying to {m['reply_to_user']})]"
        
        # Add target tag if this message is in the recent targets
        if m["msg_id"] in msg_to_target:
            prefix += f" [{msg_to_target[m['msg_id']]}]"
            
        lines.append(f"{prefix}: {m['text']}")
    
    return "\n".join(lines), target_messages

def get_recent_media_messages(chat_id: int, max_count: int = 10) -> list[dict]:
    """
    Returns a list of recent messages that contain photos or stickers.
    Limited to max_count most recent media messages.
    
    Returns list of dicts with: msg_id, user_name, media_type, timestamp
    """
    if chat_id not in _context_buffer:
        return []
    
    media_messages = []
    # Iterate in reverse to get most recent first
    for m in reversed(list(_context_buffer[chat_id])):
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


