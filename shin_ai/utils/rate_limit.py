"""
Rate limiting utilities for ShinAI.
"""

import time
from collections import OrderedDict

from shin_ai.coordination.runtime import get_coordination_store
from shin_ai.coordination.store import CoordinationStore
from shin_ai.settings import get_settings

# user_id -> last_request_time, ordered least-recently-used first
_last_used: OrderedDict[int | str, float] = OrderedDict()

COOLDOWN_SECONDS = 4
_MAX_ENTRIES = 1000

# Group-level rate limit: max N AI responses per chat per sliding window
_group_timestamps: dict[tuple, list[float]] = {}


def _group_window_seconds() -> float:
    return get_settings().group_rate_limit_window_seconds


def _group_max_responses() -> int:
    return get_settings().group_rate_limit_max_responses


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
        _group_timestamps[key] = [ts for ts in timestamps if now - ts < _group_window_seconds() * 3]
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

    # Enforce maximum dict size. popitem(last=False) drops the least recently
    # touched entry in O(1); the previous min() scanned all 1000 entries on
    # every call once the cache was full.
    if user_id not in _last_used:
        while len(_last_used) >= _MAX_ENTRIES:
            _last_used.popitem(last=False)

    _last_used[user_id] = now
    _last_used.move_to_end(user_id)
    return True


def check_group_rate_limit(platform_name: str, chat_id: int | str) -> bool:
    """Check if the bot has sent too many AI responses in a group recently.

    Limits AI-callable responses to _group_max_responses() per _group_window_seconds()
    per chat. This prevents a single busy group from burning through API quota.
    """
    now = time.time()
    _cleanup_expired(now)

    key = (platform_name, str(chat_id))

    # Prune timestamps outside the sliding window
    if key in _group_timestamps:
        _group_timestamps[key] = [ts for ts in _group_timestamps[key] if now - ts < _group_window_seconds()]

    timestamps = _group_timestamps.setdefault(key, [])
    if len(timestamps) >= _group_max_responses():
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
    bucket = int(current_time / _group_window_seconds())
    key = f"rate:group:{coordination_scope}:{platform_name}:{chat_id}:{bucket}"
    try:
        backend = store or get_coordination_store()
        count = await backend.increment(key, ttl_seconds=_group_window_seconds() * 2)
        return count <= _group_max_responses()
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

    if user_id != get_settings().admin_user_id:
        elapsed = now - _last_gstats_time
        if elapsed < GSTATS_COOLDOWN:
            return int(GSTATS_COOLDOWN - elapsed)

    _last_gstats_time = now
    return 0
