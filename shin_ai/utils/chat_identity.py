"""One canonical form for a chat id across platforms.

WhatsApp addresses the same conversation several ways: a bare number, a
number with a device suffix (``:12``), and a full JID (``user@s.whatsapp.net``).
Short-term context and bot-reply tracking both key on chat id, so they have to
agree on which of those forms is canonical or the same chat ends up split
across two buckets.
"""

from __future__ import annotations


def normalize_chat_id(platform: str, chat_id: int | str) -> str:
    """Return a stable chat identity, stripping WhatsApp device suffixes."""
    raw_chat_id = str(chat_id).strip()
    if platform != "whatsapp":
        return raw_chat_id

    lowered = raw_chat_id.lower()
    if "@" in lowered:
        user, server = lowered.split("@", 1)
        user = user.split(":", 1)[0]
        return f"{user}@{server}"
    return lowered.split(":", 1)[0]


def chat_scope_key(scope: str, platform: str, chat_id: int | str) -> str:
    """Build a namespaced key for per-chat state within a coordination scope."""
    return f"{scope}_{normalize_chat_id(platform, chat_id)}"
