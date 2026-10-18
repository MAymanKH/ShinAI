"""Optional billing ownership and crash recovery for one bot process per database."""

from __future__ import annotations

from typing import BinaryIO

from shin_ai.billing.store import CreditStore
from shin_ai.settings import get_settings

_store: CreditStore | None = None
_ownership_file: BinaryIO | None = None
_initialized = False


def get_credit_store() -> CreditStore | None:
    global _store, _ownership_file
    settings = get_settings().billing
    if not settings.enabled:
        return None
    if _store is None:
        # Recovery refunds ambiguous holds, so another bot must not still own them.
        import fcntl

        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        ownership_file = open(f"{settings.database_path}.owner.lock", "a+b")
        try:
            fcntl.flock(ownership_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _store = CreditStore(settings.database_path)
        except BaseException:
            ownership_file.close()
            raise
        _ownership_file = ownership_file
    return _store


async def initialize_billing() -> None:
    global _initialized
    if _initialized:
        return
    store = get_credit_store()
    if store is not None:
        await store.recover()
        _initialized = True


async def close_billing() -> None:
    global _store, _ownership_file, _initialized
    _initialized = False
    _store = None
    if _ownership_file is not None:
        _ownership_file.close()
        _ownership_file = None
