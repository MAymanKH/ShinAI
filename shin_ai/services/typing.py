"""Cooperative, race-free typing indicator lifecycle management."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from shin_ai.utils.logger_config import logger


class TypingPlatform(Protocol):
    platform_name: str

    async def send_chat_action(self, chat_id: int | str, action: str) -> None: ...


@dataclass(slots=True)
class TypingSession:
    platform: TypingPlatform
    chat_id: int | str
    key: tuple[int, str]
    stop_event: asyncio.Event
    task: asyncio.Task[None] | None = None


_active_sessions: dict[tuple[int, str], TypingSession] = {}


async def start_typing(
    platform: TypingPlatform,
    chat_id: int | str,
    *,
    refresh_seconds: float = 4.0,
    max_duration_seconds: float = 120.0,
) -> TypingSession:
    """Start one typing session per platform adapter and chat."""
    key = (id(platform), str(chat_id))
    existing = _active_sessions.get(key)
    if existing is not None:
        await stop_typing(existing)

    session = TypingSession(
        platform=platform,
        chat_id=chat_id,
        key=key,
        stop_event=asyncio.Event(),
    )
    session.task = asyncio.create_task(
        _run_typing_session(
            session,
            refresh_seconds=max(0.01, refresh_seconds),
            max_duration_seconds=max(0.01, max_duration_seconds),
        ),
        name=f"shinai-typing-{platform.platform_name}-{chat_id}",
    )
    _active_sessions[key] = session
    return session


async def stop_typing(session: TypingSession) -> None:
    """Stop after any in-flight typing call, then send the final cancel action."""
    session.stop_event.set()
    if session.task is not None:
        await asyncio.gather(session.task, return_exceptions=True)


async def _run_typing_session(
    session: TypingSession,
    *,
    refresh_seconds: float,
    max_duration_seconds: float,
) -> None:
    deadline = time.monotonic() + max_duration_seconds
    timed_out = False
    try:
        while not session.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            await session.platform.send_chat_action(session.chat_id, "typing")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            try:
                await asyncio.wait_for(
                    session.stop_event.wait(),
                    timeout=min(refresh_seconds, remaining),
                )
            except TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.debug(
            "Typing loop ended due to platform error: %s",
            error,
            extra={"event_name": "response.typing_error"},
        )
    finally:
        if timed_out:
            logger.debug(
                "Typing indicator reached its %.0fs safety limit",
                max_duration_seconds,
                extra={"event_name": "response.typing_timeout"},
            )
        try:
            await session.platform.send_chat_action(session.chat_id, "cancel")
        except Exception as error:
            logger.debug(
                "Failed to cancel typing indicator: %s",
                error,
                extra={"event_name": "response.typing_error"},
            )
        if _active_sessions.get(session.key) is session:
            del _active_sessions[session.key]


def active_typing_count() -> int:
    """Return active session count for health checks and tests."""
    return len(_active_sessions)
