"""Application lifecycle orchestration for orderly, leak-free shutdown."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable, Sequence
from importlib import import_module
from typing import Any

from shin_ai.utils.logger_config import logger

AsyncCloser = Callable[[], Awaitable[None]]
PlatformEntry = tuple[str, Any]
ResourceCloser = tuple[str, AsyncCloser]


async def wait_for_shutdown() -> None:
    """Wait for SIGINT/SIGTERM without letting a signal handler swallow Ctrl+C."""
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    registered_signals: list[signal.Signals] = []

    def request_shutdown(received_signal: signal.Signals) -> None:
        if stop_requested.is_set():
            return

        logger.info(
            "Stop signal received (%s); shutting down",
            received_signal.name,
            extra={"event_name": "lifecycle.signal"},
        )
        stop_requested.set()

        # The first signal requests graceful shutdown. Restore the normal OS
        # handlers immediately so a second Ctrl+C can still force termination.
        for registered_signal in registered_signals:
            loop.remove_signal_handler(registered_signal)

    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                shutdown_signal,
                request_shutdown,
                shutdown_signal,
            )
        except (NotImplementedError, RuntimeError):
            continue
        registered_signals.append(shutdown_signal)

    try:
        await stop_requested.wait()
    finally:
        for registered_signal in registered_signals:
            loop.remove_signal_handler(registered_signal)


def _default_interaction_closer() -> AsyncCloser:
    from shin_ai.core.handler import shutdown_interaction_scheduler

    return shutdown_interaction_scheduler


def _lazy_closer(module_name: str, closer_name: str) -> AsyncCloser:
    async def close() -> None:
        module = import_module(module_name)
        await getattr(module, closer_name)()

    return close


def _default_resource_closers() -> tuple[ResourceCloser, ...]:
    return (
        (
            "reply cache",
            _lazy_closer("shin_ai.services.replies", "shutdown_replies_service"),
        ),
        (
            "audio transcriber",
            _lazy_closer("shin_ai.services.audio_transcriber", "close_audio_transcriber"),
        ),
        (
            "embedding service",
            _lazy_closer("shin_ai.services.embeddings", "close_embedding_service"),
        ),
        (
            "OpenAI-compatible clients",
            _lazy_closer("shin_ai.providers.openai_compatible", "close_openai_clients"),
        ),
        (
            "Gemini clients",
            _lazy_closer("shin_ai.providers.gemini", "close_gemini_clients"),
        ),
        (
            "Chroma client",
            _lazy_closer("shin_ai.utils.db", "close_chroma_client"),
        ),
        (
            "coordination store",
            _lazy_closer("shin_ai.coordination.runtime", "close_coordination_store"),
        ),
    )


async def _close_component(label: str, closer: AsyncCloser) -> None:
    try:
        await closer()
    except Exception:
        logger.exception("Failed to close %s cleanly", label)


async def shutdown_application(
    active_platforms: Sequence[PlatformEntry],
    *,
    interaction_closer: AsyncCloser | None = None,
    resource_closers: Sequence[ResourceCloser] | None = None,
) -> None:
    """Drain work, stop ingress, then close every owned process resource."""
    logger.info("Draining interactions", extra={"event_name": "lifecycle.shutdown"})
    try:
        await _close_component(
            "interaction scheduler",
            interaction_closer or _default_interaction_closer(),
        )
    finally:
        # Platform and service cleanup must still run if a second Ctrl+C cancels
        # the graceful interaction drain.
        logger.info("Stopping platforms", extra={"event_name": "lifecycle.shutdown"})
        try:
            for platform_label, platform in reversed(active_platforms):
                await _close_component(f"{platform_label} platform", platform.stop)
        finally:
            logger.info("Closing services", extra={"event_name": "lifecycle.shutdown"})
            try:
                closers = resource_closers
                if closers is None:
                    closers = _default_resource_closers()
                for label, closer in closers:
                    await _close_component(label, closer)
            finally:
                logger.info("Shutdown complete", extra={"event_name": "lifecycle.stopped"})
