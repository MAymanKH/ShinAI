"""Pure parsing and filtering rules for model responses and trivial inputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from shin_ai.platforms.models import UnifiedMessage


ResponseMessage = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class ResponseDecision:
    skip_token_found: bool
    messages: tuple[ResponseMessage, ...]
    filtered_meta_messages: int = 0

    @property
    def skips_all_text(self) -> bool:
        return self.skip_token_found and not self.messages


_REPLY_TO_TAG_PATTERN = re.compile(
    r"^\s*`*\[?\s*REPLY_TO\s*[:=]\s*([A-Za-z0-9_\-]+)\s*\]?,?`*\s*[:.\-]*[ \t]*\n?",
    re.IGNORECASE,
)
_BRACKETED_SKIP_AFTER = re.compile(
    r"`?\[\s*SKIP\s*\]`?,?[.,!?\s]*$",
    re.IGNORECASE,
)
_BRACKETED_SKIP_BEFORE = re.compile(
    r"^[.,!?\s]*`?\[\s*SKIP\s*\]`?,?",
    re.IGNORECASE,
)
_BARE_SKIP = re.compile(r"^[.,!?\s]*SKIP[.,!?\s]*$", re.IGNORECASE)
_TRIVIAL_LAUGH_PATTERN = re.compile(
    r"^[هح\s]+$"
    r"|^h[ha]+$"
    r"|^lo+l+$"
    r"|^lma+o+$"
    r"|^x+d+$"
    r"|^😂+$|^🤣+$|^😭+$"
    r"|^ك+$",
    re.IGNORECASE,
)
_ACTION_META_COMMENTARY_PATTERN = re.compile(
    r"^\s*\(?("
    r"no\s+(further|additional)\s+(action|response|message|reply)"
    r"|sticker\s+(was|has been)\s+sent"
    r"|reaction\s+(was|has been)\s+(sent|added|applied)"
    r"|already\s+(sent|reacted|responded)"
    r"|nothing\s+(else|more)\s+(to|needed)"
    r"|action\s+(completed|done|taken)"
    r"|that'?s?\s+(all|it)\b"
    r")\b",
    re.IGNORECASE,
)


def parse_model_response(answer: str, *, has_actions: bool) -> ResponseDecision:
    """Remove control syntax and return only user-visible response messages."""
    skip_found, remaining = extract_skip_token(answer)
    messages = split_reply_messages(remaining)

    filtered_count = 0
    if has_actions and messages:
        visible_messages = tuple(
            message
            for message in messages
            if not _ACTION_META_COMMENTARY_PATTERN.search(message[0])
        )
        filtered_count = len(messages) - len(visible_messages)
        messages = list(visible_messages)

    return ResponseDecision(skip_found, tuple(messages), filtered_count)


def extract_skip_token(answer: str) -> tuple[bool, str]:
    """Strip a standalone leading/trailing `[SKIP]` control token."""
    text = (answer or "").strip()
    if not text:
        return False, ""

    match = _BRACKETED_SKIP_AFTER.search(text)
    if match:
        remainder = text[: match.start()].strip()
    else:
        match = _BRACKETED_SKIP_BEFORE.match(text)
        if not match:
            bare_skip = _BARE_SKIP.fullmatch(text) is not None
            return bare_skip, "" if bare_skip else text
        remainder = text[match.end():].strip()

    if not remainder or all(character in "-–—`[].,!? \n" for character in remainder):
        return True, ""
    return True, remainder


def split_reply_messages(answer: str) -> list[ResponseMessage]:
    """Split multi-message output and extract optional reply targets."""
    messages: list[ResponseMessage] = []
    for part in (answer or "").split("---"):
        part = part.strip()
        if not part:
            continue
        target, clean_text = _parse_reply_to_tag(part)
        messages.append((clean_text, target))
    return messages


def _parse_reply_to_tag(text: str) -> tuple[str | None, str]:
    match = _REPLY_TO_TAG_PATTERN.match(text or "")
    if not match:
        return None, text

    target = match.group(1)
    remainder = text[match.end():].strip()
    if not remainder:
        return None, text
    return target, remainder


def is_trivial_message(msg: UnifiedMessage) -> bool:
    """Return whether an input is too low-content to justify model work."""
    text = (msg.text or msg.caption or "").strip()
    if msg.sticker and not text:
        return True
    if not text:
        return False

    if all(
        unicodedata.category(character).startswith(("So", "Sk", "Sm"))
        or unicodedata.category(character) == "Zs"
        or character in "\ufe0f\u200d"
        for character in text
    ):
        return True

    cleaned = re.sub(r"[\s.,!?]+", "", text)
    return bool(cleaned and _TRIVIAL_LAUGH_PATTERN.fullmatch(cleaned))
