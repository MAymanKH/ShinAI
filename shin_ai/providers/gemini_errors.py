"""Stable Gemini failure classification independent of SDK message wording."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class GeminiFailureKind(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GeminiFailure:
    kind: GeminiFailureKind
    status_code: int | None
    retry_after_seconds: float | None
    message: str


def _status_code(error: BaseException) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def _retry_after(error: BaseException, message: str) -> float | None:
    response = getattr(error, "response", None)
    headers: Any = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass

    for attribute in ("retry_after", "retry_delay"):
        raw = getattr(error, attribute, None)
        if hasattr(raw, "total_seconds"):
            return max(0.0, float(raw.total_seconds()))
        try:
            if raw is not None:
                return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass

    match = re.search(r"retry(?:Delay| after)?[^0-9]{0,8}(\d+(?:\.\d+)?)\s*s", message, re.I)
    return float(match.group(1)) if match else None


def classify_gemini_error(error: BaseException) -> GeminiFailure:
    message = str(error).strip()[:500]
    lowered = message.lower()
    status = _status_code(error)

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        kind = GeminiFailureKind.TIMEOUT
    elif status in {401, 403}:
        kind = GeminiFailureKind.AUTHENTICATION
    elif status == 429:
        kind = GeminiFailureKind.RATE_LIMIT
    elif status in {400, 404, 405, 422}:
        kind = GeminiFailureKind.INVALID_REQUEST
    elif status in {408, 409, 425, 500, 502, 503, 504}:
        kind = GeminiFailureKind.TRANSIENT
    elif "quota" in lowered or "rate limit" in lowered or re.search(r"\b429\b", lowered):
        kind = GeminiFailureKind.RATE_LIMIT
    elif "unauthenticated" in lowered or "invalid api key" in lowered or re.search(r"\b(401|403)\b", lowered):
        kind = GeminiFailureKind.AUTHENTICATION
    elif "timed out" in lowered or "timeout" in lowered:
        kind = GeminiFailureKind.TIMEOUT
    elif re.search(r"\b(500|502|503|504)\b", lowered) or "temporarily unavailable" in lowered:
        kind = GeminiFailureKind.TRANSIENT
    else:
        kind = GeminiFailureKind.UNKNOWN

    return GeminiFailure(
        kind=kind,
        status_code=status,
        retry_after_seconds=_retry_after(error, message),
        message=message or type(error).__name__,
    )
