"""Lazy process-local handle to the configured coordination backend."""

from __future__ import annotations

import threading

from shin_ai.coordination.store import CoordinationStore, create_coordination_store
from shin_ai.settings import get_settings


_store: CoordinationStore | None = None
_lock = threading.Lock()


def get_coordination_store() -> CoordinationStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = create_coordination_store(get_settings().coordination)
    return _store


async def close_coordination_store() -> None:
    global _store
    with _lock:
        current = _store
        _store = None
    if current is not None:
        await current.close()

