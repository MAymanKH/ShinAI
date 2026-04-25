"""
Rate limiting utilities for ShinAI.
"""
import time
from shin_ai.config import ADMIN_USER_ID

# user_id -> last_request_time
_last_used: dict[int, float] = {}

COOLDOWN_SECONDS = 4


def check_rate_limit(user_id: int | str) -> bool:
    """Check if user can make a request based on cooldown."""
    now = time.time()
    last = _last_used.get(user_id, 0)

    if now - last < COOLDOWN_SECONDS:
        return False

    _last_used[user_id] = now
    return True


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
