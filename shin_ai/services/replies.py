"""
Replies Service

Tracks bot replies for reply chain detection.
"""
import json
from shin_ai.config import DATA_DIR
from shin_ai.utils.logger_config import logger
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.platforms.base import PlatformAdapter

REPLIES_FILE = DATA_DIR / "bot_replies.json"


def load_replies() -> dict:
    """Load saved bot replies from file."""
    if not REPLIES_FILE.exists():
        return {}
    try:
        with open(REPLIES_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_reply(chat_id: int | str, message_id: int | str) -> None:
    """Save a bot reply for future chain detection."""
    replies = load_replies()
    chat_id_str = str(chat_id)
    
    if chat_id_str not in replies:
        replies[chat_id_str] = []
    
    replies[chat_id_str].append(str(message_id))
    
    # Keep only last 100 replies per chat
    if len(replies[chat_id_str]) > 100:
        replies[chat_id_str] = replies[chat_id_str][-100:]
    
    # Ensure directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(REPLIES_FILE, "w") as f:
        json.dump(replies, f)


async def check_reply_chain(msg: UnifiedMessage):
    if msg.reply_to_message_id:
        replies = load_replies()
        chat_id_str = str(msg.chat.id)
        if chat_id_str in replies:
            if str(msg.reply_to_message_id) in replies[chat_id_str]:
                return True
    return False

async def get_reply_chain(msg: UnifiedMessage, platform: PlatformAdapter = None):
    # This walks the reply chain. If platform is provided, it tries to fetch
    # missing messages from the API to get deeper context.
    chain = []
    current_msg = msg
    depth = 0
    max_depth = 10
    
    while depth < max_depth:
        # Move up the chain
        parent = current_msg.reply_to_message
        parent_id = current_msg.reply_to_message_id
        
        # If we have neither, the chain ends
        if not parent and not parent_id:
            break
            
        # If we have the ID but not the full message, try fetching it
        if not parent and parent_id and platform:
            try:
                parent = await platform.get_message(msg.chat.id, parent_id)
            except Exception as e:
                logger.warning(f"Failed to fetch parent message {parent_id} for reply chain: {e}")
                break
                
        # If we still don't have the parent message, we can't go deeper
        if not parent:
            break
            
        sender_name = "Unknown"
        if parent.from_user:
            sender_name = f"{parent.from_user.username or 'NoUser'}/{parent.from_user.first_name}"
        
        text = parent.text or parent.caption or "[Media]"
        chain.append(f"Message from {sender_name}: {text}")

        current_msg = parent
        depth += 1
    
    return chain
