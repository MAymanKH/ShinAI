from __future__ import annotations

from collections import OrderedDict
from typing import Any

from shin_ai.platforms.whatsapp_runtime import JIDType, Jid2String


def normalize_jid_identity(jid_value: str) -> str:
    raw = (jid_value or "").strip().lower()
    if not raw:
        return ""

    # Normalize device-scoped IDs like 201234567890:12@s.whatsapp.net to
    # the stable user identity 201234567890@s.whatsapp.net.
    if "@" in raw:
        user, server = raw.split("@", 1)
        user = user.split(":", 1)[0]
        return f"{user}@{server}"

    return raw.split(":", 1)[0]


def extract_local_user_id(jid_value: str) -> str:
    normalized = normalize_jid_identity(jid_value)
    if not normalized:
        return ""
    if "@" in normalized:
        return normalized.split("@", 1)[0]
    return normalized


def jid_to_user_id(jid: JIDType) -> str:
    raw_jid = Jid2String(jid)
    normalized = normalize_jid_identity(raw_jid)
    if normalized:
        return normalized
    if jid.User:
        return str(jid.User).split(":", 1)[0]
    return raw_jid


def jid_to_username(jid: JIDType) -> str:
    raw_jid = Jid2String(jid)
    local_user = extract_local_user_id(raw_jid)
    if local_user:
        return local_user
    if jid.User:
        return str(jid.User).split(":", 1)[0]
    return raw_jid


def normalize_message_timestamp(raw_timestamp: int | float | None) -> float:
    if not raw_timestamp:
        return 0.0

    ts = float(raw_timestamp)
    # Neonize timestamps can be emitted in milliseconds depending on source wrappers.
    if ts > 1e12:
        ts = ts / 1000.0
    return ts


def media_id(message_id: int | str, media_type: str) -> str:
    return f"{message_id}:{media_type}"


def collect_mentioned_identity_tokens(mentioned_jid_list: list[str]) -> set[str]:
    """Build a set of identity strings from a mentionedJID list."""
    tokens: set[str] = set()
    for jid_str in mentioned_jid_list:
        raw = (jid_str or "").strip()
        if not raw:
            continue
        tokens.add(raw.lower())

        normalized = normalize_jid_identity(raw)
        if normalized:
            tokens.add(normalized.lower())

        local = extract_local_user_id(raw)
        if local:
            tokens.add(local.lower())

        user_part = raw.split("@", 1)[0] if "@" in raw else raw
        user_part = user_part.split(":", 1)[0]
        if user_part:
            tokens.add(user_part.lower())

    return tokens


def is_same_chat_identity(first_chat_id: str, second_chat_id: str) -> bool:
    if first_chat_id == second_chat_id:
        return True

    normalized_first = normalize_jid_identity(first_chat_id)
    normalized_second = normalize_jid_identity(second_chat_id)
    if normalized_first and normalized_first == normalized_second:
        return True

    first_local = extract_local_user_id(first_chat_id)
    second_local = extract_local_user_id(second_chat_id)
    return bool(first_local and first_local == second_local)


def find_cache_key(
    cache_map: OrderedDict[tuple[str, str], Any],
    chat_id: int | str,
    message_id: int | str,
) -> tuple[str, str] | None:
    requested_chat = str(chat_id)
    requested_msg = str(message_id)
    exact_key = (requested_chat, requested_msg)

    if exact_key in cache_map:
        return exact_key

    # WhatsApp may emit the same chat identity in slightly different JID
    # forms (e.g., device-scoped IDs). Match by normalized identity too.
    for candidate_key in reversed(cache_map):
        candidate_chat, candidate_msg = candidate_key
        if candidate_msg != requested_msg:
            continue
        if is_same_chat_identity(candidate_chat, requested_chat):
            return candidate_key

    return None
