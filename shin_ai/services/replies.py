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
_next_message_watch: dict[str, bool] = {}


def set_next_message_watch(platform: str, chat_id: int | str):
    _next_message_watch[_reply_key(platform, chat_id)] = True


def check_and_clear_next_message_watch(platform: str, chat_id: int | str) -> bool:
    key = _reply_key(platform, chat_id)
    if _next_message_watch.get(key):
        _next_message_watch[key] = False
        return True
    return False


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


def _reply_key(platform: str, chat_id: int | str) -> str:
    return f"{platform}_{_normalize_chat_id(platform, chat_id)}"


def load_replies() -> dict:
    """Load saved bot replies from file."""
    if not REPLIES_FILE.exists():
        return {}
    try:
        with open(REPLIES_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_reply(chat_id: int | str, message_id: int | str, platform: str | None = None) -> None:
    """Save a bot reply for future chain detection."""
    replies = load_replies()
    chat_id_str = str(chat_id)
    scoped_chat_id = _reply_key(platform, chat_id) if platform else chat_id_str
    
    if scoped_chat_id not in replies:
        replies[scoped_chat_id] = []
    
    replies[scoped_chat_id].append(str(message_id))
    if platform:
        set_next_message_watch(platform, chat_id)
    
    # Keep only last 100 replies per chat
    if len(replies[scoped_chat_id]) > 100:
        replies[scoped_chat_id] = replies[scoped_chat_id][-100:]
    
    # Ensure directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(REPLIES_FILE, "w") as f:
        json.dump(replies, f)


async def check_reply_chain(msg: UnifiedMessage):
    if msg.reply_to_message_id:
        replies = load_replies()
        scoped_chat_id = _reply_key(msg.platform, msg.chat.id)
        legacy_chat_id = str(msg.chat.id)

        for key in (scoped_chat_id, legacy_chat_id):
            if key in replies and str(msg.reply_to_message_id) in replies[key]:
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
