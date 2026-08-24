"""Cooperative, race-free typing indicator lifecycle management."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from shin_ai.settings import get_settings
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
    action_timeout_seconds: float
    task: asyncio.Task[None] | None = None


_active_sessions: dict[tuple[int, str], TypingSession] = {}
_detached_actions: set[asyncio.Task] = set()


def _consume_detached_action(task: asyncio.Task) -> None:
    _detached_actions.discard(task)
    if task.cancelled():
        return
    task.exception()


def _track_detached_action(task: asyncio.Task) -> None:
    _detached_actions.add(task)
    task.add_done_callback(_consume_detached_action)


async def _send_action_with_timeout(
    session: TypingSession,
    action: str,
    *,
    recover_late_typing: bool = True,
) -> bool:
    task = asyncio.create_task(
        session.platform.send_chat_action(session.chat_id, action),
        name=f"shinai-typing-action-{session.platform.platform_name}-{session.chat_id}",
    )
    done, _ = await asyncio.wait({task}, timeout=session.action_timeout_seconds)
    if task in done:
        await task
        return True

    task.cancel()
    _track_detached_action(task)
    logger.warning(
        "Typing platform action timed out — platform=%s chat=%s action=%s timeout=%.1fs",
        session.platform.platform_name,
        session.chat_id,
        action,
        session.action_timeout_seconds,
        extra={"event_name": "response.typing_timeout"},
    )

    if action == "typing" and recover_late_typing:

        def cancel_after_late_typing(_task: asyncio.Task) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            recovery = loop.create_task(
                _send_action_with_timeout(
                    session,
                    "cancel",
                    recover_late_typing=False,
                ),
                name=(f"shinai-typing-recovery-{session.platform.platform_name}-{session.chat_id}"),
            )
            _track_detached_action(recovery)

        task.add_done_callback(cancel_after_late_typing)
    return False


async def start_typing(
    platform: TypingPlatform,
    chat_id: int | str,
    *,
    refresh_seconds: float = 4.0,
    max_duration_seconds: float = 120.0,
    action_timeout_seconds: float | None = None,
) -> TypingSession:
    """Start one typing session per platform adapter and chat."""
    if action_timeout_seconds is None:
        action_timeout_seconds = get_settings().runtime.typing_action_timeout_seconds
    key = (id(platform), str(chat_id))
    existing = _active_sessions.get(key)
    if existing is not None:
        await stop_typing(existing)

    session = TypingSession(
        platform=platform,
        chat_id=chat_id,
        key=key,
        stop_event=asyncio.Event(),
        action_timeout_seconds=max(0.01, action_timeout_seconds),
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
        stop_timeout = session.action_timeout_seconds * 2 + 0.1
        done, _ = await asyncio.wait({session.task}, timeout=stop_timeout)
        if session.task in done:
            await asyncio.gather(session.task, return_exceptions=True)
        else:
            session.task.cancel()
            _track_detached_action(session.task)
            logger.warning(
                "Typing session shutdown timed out — platform=%s chat=%s timeout=%.1fs",
                session.platform.platform_name,
                session.chat_id,
                stop_timeout,
                extra={"event_name": "response.typing_timeout"},
            )
    if _active_sessions.get(session.key) is session:
        del _active_sessions[session.key]


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

            if not await _send_action_with_timeout(session, "typing"):
                break
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
            await _send_action_with_timeout(
                session,
                "cancel",
                recover_late_typing=False,
            )
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
