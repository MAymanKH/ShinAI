"""
Replies Service

Tracks bot replies for reply chain detection.
"""
import json
from pyrogram import Client
from pyrogram.types import Message
from shin_ai.config import DATA_DIR
from shin_ai.utils.logger_config import logger

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


def save_reply(chat_id: int, message_id: int) -> None:
    """Save a bot reply for future chain detection."""
    replies = load_replies()
    chat_id_str = str(chat_id)
    
    if chat_id_str not in replies:
        replies[chat_id_str] = []
    
    replies[chat_id_str].append(message_id)
    
    # Keep only last 100 replies per chat
    if len(replies[chat_id_str]) > 100:
        replies[chat_id_str] = replies[chat_id_str][-100:]
    
    # Ensure directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(REPLIES_FILE, "w") as f:
        json.dump(replies, f)


async def check_reply_chain(msg: Message):
    if msg.reply_to_message:
        replies = load_replies()
        chat_id_str = str(msg.chat.id)
        if chat_id_str in replies:
            if msg.reply_to_message.id in replies[chat_id_str]:
                return True
    return False

async def get_reply_chain(client: Client, msg: Message):
    chain = []
    current_msg = msg
    depth = 0
    max_depth = 10
    
    while current_msg.reply_to_message and depth < max_depth:
        reply = current_msg.reply_to_message
        sender_name = "Unknown"
        if reply.from_user:
            sender_name = f"{reply.from_user.username}/{reply.from_user.full_name}"
        
        text = reply.text or reply.caption or "[Media]"
        chain.append(f"Message from {sender_name}: {text}")

        if reply.reply_to_message:
            current_msg = reply
        elif reply.reply_to_message_id:
            try:
                current_msg = await client.get_messages(msg.chat.id, reply.id)
            except Exception as e:
                logger.error(f"Error fetching reply chain: {e}")
                break
        else:
            break
            
        depth += 1
    
    return chain
