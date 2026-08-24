"""
Rate limiting utilities for ShinAI.
"""

import time

from shin_ai.config import (
    ADMIN_USER_ID,
    GROUP_MAX_RESPONSES_PER_WINDOW,
    GROUP_RATE_LIMIT_WINDOW_SECONDS,
)
from shin_ai.coordination.runtime import get_coordination_store
from shin_ai.coordination.store import CoordinationStore

# user_id -> last_request_time
_last_used: dict[int | str, float] = {}

COOLDOWN_SECONDS = 4
_MAX_ENTRIES = 1000

# Group-level rate limit: max N AI responses per chat per sliding window
_group_timestamps: dict[tuple, list[float]] = {}
GROUP_WINDOW_SECONDS = GROUP_RATE_LIMIT_WINDOW_SECONDS
GROUP_MAX_RESPONSES = GROUP_MAX_RESPONSES_PER_WINDOW

# Periodic cleanup tracking
_last_cleanup: float = time.time()
_CLEANUP_INTERVAL = 300.0  # 5 minutes
_ENTRY_TTL = 3600.0  # Remove entries older than 1 hour


def _cleanup_expired(now: float) -> None:
    """Remove entries older than _ENTRY_TTL from tracking dicts."""
    global _last_cleanup
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return

    _last_cleanup = now

    # Clean up per-user rate limit entries
    expired_users = [uid for uid, ts in _last_used.items() if now - ts > _ENTRY_TTL]
    for uid in expired_users:
        _last_used.pop(uid, None)

    # Clean up group rate limit entries — trim old timestamps from each list
    expired_groups = []
    for key, timestamps in _group_timestamps.items():
        _group_timestamps[key] = [ts for ts in timestamps if now - ts < GROUP_WINDOW_SECONDS * 3]
        if not _group_timestamps[key]:
            expired_groups.append(key)
    for key in expired_groups:
        del _group_timestamps[key]


def check_rate_limit(user_id: int | str) -> bool:
    """Check if user can make a request based on cooldown."""
    now = time.time()
    _cleanup_expired(now)

    last = _last_used.get(user_id, 0)

    if now - last < COOLDOWN_SECONDS:
        return False

    # Enforce maximum dict size
    if len(_last_used) >= _MAX_ENTRIES:
        # Evict the oldest entry
        oldest_key = min(_last_used, key=lambda k: _last_used[k])
        _last_used.pop(oldest_key, None)

    _last_used[user_id] = now
    return True


def check_group_rate_limit(platform_name: str, chat_id: int | str) -> bool:
    """Check if the bot has sent too many AI responses in a group recently.

    Limits AI-callable responses to GROUP_MAX_RESPONSES per GROUP_WINDOW_SECONDS
    per chat. This prevents a single busy group from burning through API quota.
    """
    now = time.time()
    _cleanup_expired(now)

    key = (platform_name, str(chat_id))

    # Prune timestamps outside the sliding window
    if key in _group_timestamps:
        _group_timestamps[key] = [ts for ts in _group_timestamps[key] if now - ts < GROUP_WINDOW_SECONDS]

    timestamps = _group_timestamps.setdefault(key, [])
    if len(timestamps) >= GROUP_MAX_RESPONSES:
        return False

    timestamps.append(now)
    return True


async def check_rate_limit_shared(
    platform_name: str,
    user_id: int | str,
    *,
    coordination_scope: str,
    store: CoordinationStore | None = None,
) -> bool:
    """Atomically enforce the user cooldown across cooperating instances."""
    key = f"rate:user:{coordination_scope}:{platform_name}:{user_id}"
    try:
        backend = store or get_coordination_store()
        return await backend.claim(key, "used", ttl_seconds=COOLDOWN_SECONDS)
    except Exception:
        # Coordination failure should reduce protection, not take the bot down.
        return check_rate_limit(f"{coordination_scope}:{platform_name}:{user_id}")


async def check_group_rate_limit_shared(
    platform_name: str,
    chat_id: int | str,
    *,
    coordination_scope: str,
    store: CoordinationStore | None = None,
    now: float | None = None,
) -> bool:
    """Enforce a compact fixed-window group quota across bot instances."""
    current_time = time.time() if now is None else now
    bucket = int(current_time / GROUP_WINDOW_SECONDS)
    key = f"rate:group:{coordination_scope}:{platform_name}:{chat_id}:{bucket}"
    try:
        backend = store or get_coordination_store()
        count = await backend.increment(key, ttl_seconds=GROUP_WINDOW_SECONDS * 2)
        return count <= GROUP_MAX_RESPONSES
    except Exception:
        return check_group_rate_limit(platform_name, chat_id)


_last_gstats_time = 0.0
GSTATS_COOLDOWN = 1200  # 20 minutes


def check_gstats_rate_limit(user_id: int | str) -> int:
    """
    Returns the number of seconds the user needs to wait.
    Returns 0 if the request is allowed.
    Updates the global timer if allowed.
    Admin users bypass the rate limit.
    """
    global _last_gstats_time
    now = time.time()

    if user_id != ADMIN_USER_ID:
        elapsed = now - _last_gstats_time
        if elapsed < GSTATS_COOLDOWN:
            return int(GSTATS_COOLDOWN - elapsed)

    _last_gstats_time = now
    return 0
