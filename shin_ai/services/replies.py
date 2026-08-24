"""
Replies Service

Tracks bot replies for reply chain detection.
"""

import asyncio
import hashlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING

from shin_ai.config import CONTEXT_MAX_CHATS, DATA_DIR, REPLY_STATE_TTL_SECONDS
from shin_ai.coordination.runtime import get_coordination_store
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.utils.chat_identity import chat_scope_key
from shin_ai.utils.logger_config import logger

if TYPE_CHECKING:
    from shin_ai.coordination.store import CoordinationStore

REPLIES_FILE = DATA_DIR / "bot_replies.json"
_next_message_watch: OrderedDict[str, bool] = OrderedDict()

# In-memory cache of the replies file. Loaded lazily on first access.
_replies_cache: dict[str, list[str]] | None = None
_replies_dirty: bool = False
_replies_revision: int = 0

_FLUSH_INTERVAL_SECONDS = 30.0
_flush_task: asyncio.Task | None = None
_flush_file_lock = threading.Lock()


def set_next_message_watch(
    platform: str,
    chat_id: int | str,
    coordination_scope: str | None = None,
) -> None:
    key = _reply_key(platform, chat_id, coordination_scope)
    if key not in _next_message_watch:
        while len(_next_message_watch) >= CONTEXT_MAX_CHATS:
            _next_message_watch.popitem(last=False)
    _next_message_watch[key] = True
    # Re-arming an existing watch must also refresh its recency, otherwise an
    # active chat can be evicted ahead of a dormant one.
    _next_message_watch.move_to_end(key)


async def check_and_clear_next_message_watch(
    platform: str,
    chat_id: int | str,
    *,
    coordination_scope: str | None = None,
    store: "CoordinationStore | None" = None,
) -> bool:
    key = _reply_key(platform, chat_id, coordination_scope)
    local_watch = _next_message_watch.pop(key, False)
    if not coordination_scope:
        return local_watch

    try:
        shared_watch = await (store or get_coordination_store()).delete(
            _shared_state_key("next", coordination_scope, chat_id)
        )
        return shared_watch or local_watch
    except Exception:
        logger.exception("Shared next-message marker lookup failed")
        return local_watch


def _reply_key(
    platform: str,
    chat_id: int | str,
    coordination_scope: str | None = None,
) -> str:
    return chat_scope_key(coordination_scope or platform, platform, chat_id)


def _shared_state_key(
    kind: str,
    coordination_scope: str,
    chat_id: int | str,
    message_id: int | str | None = None,
) -> str:
    identity = f"{coordination_scope}|{chat_id}"
    if message_id is not None:
        identity += f"|{message_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"reply-state:{kind}:{digest}"


def _load_cache_from_disk() -> dict[str, list[str]]:
    """Load saved bot replies from file. Called once at first access."""
    global _replies_cache
    if _replies_cache is not None:
        return _replies_cache

    if not REPLIES_FILE.exists():
        _replies_cache = {}
        return _replies_cache

    try:
        with open(REPLIES_FILE) as f:
            _replies_cache = json.load(f)
    except Exception:
        _replies_cache = {}

    if not isinstance(_replies_cache, dict):
        _replies_cache = {}
    while len(_replies_cache) > CONTEXT_MAX_CHATS:
        oldest_chat = next(iter(_replies_cache))
        _replies_cache.pop(oldest_chat, None)
    for chat_id, reply_ids in tuple(_replies_cache.items()):
        if not isinstance(reply_ids, list):
            _replies_cache.pop(chat_id, None)
        elif len(reply_ids) > 100:
            _replies_cache[chat_id] = reply_ids[-100:]

    return _replies_cache


async def _flush_to_disk() -> None:
    """Write the in-memory replies cache to disk asynchronously."""
    global _replies_dirty
    if not _replies_dirty or _replies_cache is None:
        return

    revision = _replies_revision
    snapshot = {chat_id: list(reply_ids) for chat_id, reply_ids in _replies_cache.items()}

    def _sync_write() -> None:
        with _flush_file_lock:
            if revision != _replies_revision:
                return
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temporary_file = REPLIES_FILE.with_name(
                f".{REPLIES_FILE.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with open(temporary_file, "w", encoding="utf-8") as file:
                    json.dump(snapshot, file)
                os.replace(temporary_file, REPLIES_FILE)
            finally:
                temporary_file.unlink(missing_ok=True)

    await asyncio.to_thread(_sync_write)
    if revision == _replies_revision:
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


async def save_reply(
    chat_id: int | str,
    message_id: int | str,
    platform: str | None = None,
    *,
    coordination_scope: str | None = None,
    store: "CoordinationStore | None" = None,
) -> None:
    """Save a bot reply for future chain detection.

    Writes to an in-memory cache and marks it dirty; a background task
    flushes to disk periodically to avoid blocking the event loop.
    """
    global _replies_dirty, _replies_revision
    _load_cache_from_disk()
    assert _replies_cache is not None

    scoped_chat_id = _reply_key(platform, chat_id, coordination_scope) if platform else str(chat_id)

    if scoped_chat_id not in _replies_cache:
        while len(_replies_cache) >= CONTEXT_MAX_CHATS:
            oldest_chat = next(iter(_replies_cache))
            _replies_cache.pop(oldest_chat, None)
        _replies_cache[scoped_chat_id] = []

    _replies_cache[scoped_chat_id].append(str(message_id))
    if platform:
        set_next_message_watch(platform, chat_id, coordination_scope)

    # Keep only last 100 replies per chat
    if len(_replies_cache[scoped_chat_id]) > 100:
        _replies_cache[scoped_chat_id] = _replies_cache[scoped_chat_id][-100:]

    _replies_dirty = True
    _replies_revision += 1
    _ensure_flush_task()

    if coordination_scope:
        try:
            shared_store = store or get_coordination_store()
            await shared_store.set(
                _shared_state_key("message", coordination_scope, chat_id, message_id),
                "1",
                ttl_seconds=REPLY_STATE_TTL_SECONDS,
            )
            await shared_store.set(
                _shared_state_key("next", coordination_scope, chat_id),
                "1",
                ttl_seconds=REPLY_STATE_TTL_SECONDS,
            )
        except Exception:
            logger.exception("Shared bot-reply marker update failed")


async def shutdown_replies_service() -> None:
    """Stop background work, persist pending replies, and release caches."""
    global _flush_task, _replies_cache, _replies_dirty, _replies_revision

    flush_task = _flush_task
    _flush_task = None
    if flush_task is not None:
        flush_task.cancel()
        await asyncio.gather(flush_task, return_exceptions=True)

    try:
        await _flush_to_disk()
    except Exception:
        logger.exception("Failed to flush replies cache during shutdown")
    finally:
        _replies_cache = None
        _replies_dirty = False
        _replies_revision = 0
        _next_message_watch.clear()


async def check_reply_chain(
    msg: UnifiedMessage,
    *,
    coordination_scope: str | None = None,
    store: "CoordinationStore | None" = None,
) -> bool:
    if msg.reply_to_message and msg.reply_to_message.from_user and msg.reply_to_message.from_user.is_self:
        return True

    if msg.reply_to_message_id:
        _load_cache_from_disk()
        assert _replies_cache is not None

        scoped_chat_id = _reply_key(
            msg.platform,
            msg.chat.id,
            coordination_scope,
        )
        candidate_keys = (scoped_chat_id,)
        if not coordination_scope:
            candidate_keys += (str(msg.chat.id),)

        for key in candidate_keys:
            if key in _replies_cache and str(msg.reply_to_message_id) in _replies_cache[key]:
                return True
        if coordination_scope:
            try:
                marker = await (store or get_coordination_store()).get(
                    _shared_state_key(
                        "message",
                        coordination_scope,
                        msg.chat.id,
                        msg.reply_to_message_id,
                    )
                )
                return marker is not None
            except Exception:
                logger.exception("Shared bot-reply marker lookup failed")
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
