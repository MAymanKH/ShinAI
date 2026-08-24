"""Lazy process-local handle to the configured coordination backend."""

from __future__ import annotations

import asyncio
import threading

from shin_ai.coordination.store import CoordinationStore, create_coordination_store
from shin_ai.settings import get_settings
from shin_ai.utils.logger_config import logger


_store: CoordinationStore | None = None
_maintenance_task: asyncio.Task | None = None
_lock = threading.Lock()


def get_coordination_store() -> CoordinationStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = create_coordination_store(get_settings().coordination)
    _ensure_maintenance_task()
    return _store


def _ensure_maintenance_task() -> None:
    global _maintenance_task
    if _maintenance_task is not None and not _maintenance_task.done():
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    _maintenance_task = asyncio.create_task(
        _run_maintenance(),
        name="shinai-coordination-maintenance",
    )


async def _run_maintenance() -> None:
    interval = get_settings().coordination.cleanup_interval_seconds
    while True:
        await asyncio.sleep(interval)
        current = _store
        if current is not None:
            try:
                await current.cleanup()
            except Exception:
                logger.exception("Coordination maintenance failed; retrying later")


async def close_coordination_store() -> None:
    global _maintenance_task, _store
    maintenance = _maintenance_task
    _maintenance_task = None
    if maintenance is not None:
        maintenance.cancel()
        await asyncio.gather(maintenance, return_exceptions=True)
    with _lock:
        current = _store
        _store = None
    if current is not None:
        await current.close()
