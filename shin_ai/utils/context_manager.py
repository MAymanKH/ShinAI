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

    text_content = msg.text or msg.caption or "[Media/Sticker]"

    entry = {
        "msg_id": msg.id,
        "user_id": msg.from_user.id,
        "user_name": user_name,
        "text": text_content,
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
    if chat_id not in _context_buffer:
        return ""

    lines = []
    # Sort by timestamp/id just in case, though deque should be ordered
    msgs = list(_context_buffer[chat_id])
    
    for m in msgs:
        if current_msg_id and m["msg_id"] == current_msg_id:
            continue
            
        # Format Timestamp
        try:
            ts = m["timestamp"]
            # Convert to local time string including timezone if possible, or just standard format
            dt_obj = datetime.fromtimestamp(ts)
            time_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = "Unknown Time"

        # Format: [Time] [User Name]: Message
        # Add reply indicator if applicable
        prefix = f"[{time_str}] [{m['user_name']}]"
        if m['reply_to_user']:
            prefix = f"[{time_str}] [{m['user_name']} (replying to {m['reply_to_user']})]"
            
        lines.append(f"{prefix}: {m['text']}")
    
    return "\n".join(lines)
