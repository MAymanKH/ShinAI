"""
Replies Service

Tracks bot replies for reply chain detection.
"""
import asyncio
import json

from shin_ai.config import DATA_DIR
from shin_ai.utils.logger_config import logger
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.platforms.base import PlatformAdapter

REPLIES_FILE = DATA_DIR / "bot_replies.json"
_next_message_watch: dict[str, bool] = {}

# In-memory cache of the replies file. Loaded lazily on first access.
_replies_cache: dict[str, list[str]] | None = None
_replies_dirty: bool = False

_FLUSH_INTERVAL_SECONDS = 30.0
_flush_task: asyncio.Task | None = None


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


def _load_cache_from_disk() -> dict[str, list[str]]:
    """Load saved bot replies from file. Called once at first access."""
    global _replies_cache
    if _replies_cache is not None:
        return _replies_cache

    if not REPLIES_FILE.exists():
        _replies_cache = {}
        return _replies_cache

    try:
        with open(REPLIES_FILE, "r") as f:
            _replies_cache = json.load(f)
    except Exception:
        _replies_cache = {}

    return _replies_cache


async def _flush_to_disk() -> None:
    """Write the in-memory replies cache to disk asynchronously."""
    global _replies_dirty
    if not _replies_dirty or _replies_cache is None:
        return

    def _sync_write():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPLIES_FILE, "w") as f:
            json.dump(_replies_cache, f)

    await asyncio.to_thread(_sync_write)
    _replies_dirty = False


async def _flush_periodically() -> None:
    """Background task: flush to disk every _FLUSH_INTERVAL_SECONDS."""
    while True:
        await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
        try:
            await _flush_to_disk()
        except Exception as e:
            logger.error("Failed to flush replies cache to disk: %s", e)


def _ensure_flush_task() -> None:
    """Start the periodic flush task if not already running."""
    global _flush_task
    if _flush_task is None or _flush_task.done():
        _flush_task = asyncio.create_task(_flush_periodically())


async def save_reply(chat_id: int | str, message_id: int | str, platform: str | None = None) -> None:
    """Save a bot reply for future chain detection.

    Writes to an in-memory cache and marks it dirty; a background task
    flushes to disk periodically to avoid blocking the event loop.
    """
    global _replies_dirty
    _load_cache_from_disk()
    assert _replies_cache is not None

    scoped_chat_id = _reply_key(platform, chat_id) if platform else str(chat_id)

    if scoped_chat_id not in _replies_cache:
        _replies_cache[scoped_chat_id] = []

    _replies_cache[scoped_chat_id].append(str(message_id))
    if platform:
        set_next_message_watch(platform, chat_id)

    # Keep only last 100 replies per chat
    if len(_replies_cache[scoped_chat_id]) > 100:
        _replies_cache[scoped_chat_id] = _replies_cache[scoped_chat_id][-100:]

    _replies_dirty = True
    _ensure_flush_task()


async def check_reply_chain(msg: UnifiedMessage):
    if msg.reply_to_message_id:
        _load_cache_from_disk()
        assert _replies_cache is not None

        scoped_chat_id = _reply_key(msg.platform, msg.chat.id)
        legacy_chat_id = str(msg.chat.id)

        for key in (scoped_chat_id, legacy_chat_id):
            if key in _replies_cache and str(msg.reply_to_message_id) in _replies_cache[key]:
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
