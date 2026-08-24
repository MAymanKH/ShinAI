"""Namespaced coordination backends.

The API intentionally stays small. It covers expiring claims, leases, compact
state values and counters without making application code depend on SQLite.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from shin_ai.settings import CoordinationSettings


Clock = Callable[[], float]


class CoordinationStore(ABC):
    """Atomic namespaced operations shared by one or more bot instances."""

    namespace: str

    @abstractmethod
    async def get(self, key: str) -> str | None:
        pass

    @abstractmethod
    async def get_many(self, keys: list[str] | tuple[str, ...]) -> dict[str, str]:
        """Return live values for the requested keys in one store round trip."""

    @abstractmethod
    async def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        pass

    @abstractmethod
    async def claim(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        """Set an expiring value only when the key is absent or expired."""

    @abstractmethod
    async def refresh(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        """Extend an unexpired claim when its current value matches ``value``."""

    @abstractmethod
    async def delete(self, key: str, *, expected_value: str | None = None) -> bool:
        pass

    @abstractmethod
    async def increment(self, key: str, *, ttl_seconds: float | None = None) -> int:
        pass

    @abstractmethod
    async def cleanup(self) -> int:
        """Delete expired records and return the number removed."""

    @abstractmethod
    async def close(self) -> None:
        pass


class InMemoryCoordinationStore(CoordinationStore):
    """Process-local backend for tests and deliberately isolated instances."""

    def __init__(self, namespace: str = "shinai", *, clock: Clock = time.time) -> None:
        self.namespace = namespace
        self._clock = clock
        self._entries: dict[str, tuple[str, float | None]] = {}
        self._counters: dict[str, tuple[int, float | None]] = {}
        self._lock = asyncio.Lock()

    def _live_entry(self, key: str) -> tuple[str, float | None] | None:
        entry = self._entries.get(key)
        if entry is not None and entry[1] is not None and entry[1] <= self._clock():
            self._entries.pop(key, None)
            return None
        return entry

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._live_entry(key)
            return entry[0] if entry else None

    async def get_many(self, keys: list[str] | tuple[str, ...]) -> dict[str, str]:
        async with self._lock:
            values: dict[str, str] = {}
            for key in keys:
                entry = self._live_entry(key)
                if entry is not None:
                    values[key] = entry[0]
            return values

    async def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        async with self._lock:
            expires_at = self._clock() + ttl_seconds if ttl_seconds is not None else None
            self._entries[key] = (value, expires_at)

    async def claim(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        async with self._lock:
            if self._live_entry(key) is not None:
                return False
            self._entries[key] = (value, self._clock() + ttl_seconds)
            return True

    async def refresh(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        async with self._lock:
            entry = self._live_entry(key)
            if entry is None or entry[0] != value:
                return False
            self._entries[key] = (value, self._clock() + ttl_seconds)
            return True

    async def delete(self, key: str, *, expected_value: str | None = None) -> bool:
        async with self._lock:
            entry = self._live_entry(key)
            if entry is None or (expected_value is not None and entry[0] != expected_value):
                return False
            del self._entries[key]
            return True

    async def increment(self, key: str, *, ttl_seconds: float | None = None) -> int:
        async with self._lock:
            now = self._clock()
            current, expires_at = self._counters.get(key, (0, None))
            if expires_at is not None and expires_at <= now:
                current = 0
            current += 1
            new_expiry = now + ttl_seconds if ttl_seconds is not None else expires_at
            self._counters[key] = (current, new_expiry)
            return current

    async def cleanup(self) -> int:
        async with self._lock:
            now = self._clock()
            entry_keys = [key for key, (_, expiry) in self._entries.items() if expiry and expiry <= now]
            counter_keys = [key for key, (_, expiry) in self._counters.items() if expiry and expiry <= now]
            for key in entry_keys:
                del self._entries[key]
            for key in counter_keys:
                del self._counters[key]
            return len(entry_keys) + len(counter_keys)

    async def close(self) -> None:
        return None


class SQLiteCoordinationStore(CoordinationStore):
    """SQLite/WAL backend suitable for processes sharing one local disk."""

    def __init__(
        self,
        database_path: Path | str,
        namespace: str = "shinai",
        *,
        clock: Clock = time.time,
    ) -> None:
        self.namespace = namespace
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS coordination_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                expires_at REAL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS coordination_counters (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value INTEGER NOT NULL,
                expires_at REAL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._closed = False

    async def _run(self, function, *args):
        # Each operation is one indexed SQLite statement or a tiny transaction.
        # Keeping it inline avoids executor hand-off overhead on the hottest
        # coordination path. WAL mode and the busy timeout handle the second
        # local process; longer application work never runs inside a transaction.
        return function(*args)

    def _get_sync(self, key: str) -> str | None:
        now = self._clock()
        with self._lock:
            row = self._connection.execute(
                """
                SELECT value FROM coordination_entries
                WHERE namespace = ? AND key = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (self.namespace, key, now),
            ).fetchone()
            return str(row[0]) if row else None

    async def get(self, key: str) -> str | None:
        return await self._run(self._get_sync, key)

    def _get_many_sync(self, keys: list[str] | tuple[str, ...]) -> dict[str, str]:
        if not keys:
            return {}

        now = self._clock()
        values: dict[str, str] = {}
        with self._lock:
            # Stay below SQLite's variable limit even for unusually large pools.
            for offset in range(0, len(keys), 400):
                chunk = keys[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"""
                    SELECT key, value FROM coordination_entries
                    WHERE namespace = ? AND key IN ({placeholders})
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (self.namespace, *chunk, now),
                ).fetchall()
                values.update((str(key), str(value)) for key, value in rows)
        return values

    async def get_many(self, keys: list[str] | tuple[str, ...]) -> dict[str, str]:
        return await self._run(self._get_many_sync, keys)

    def _set_sync(self, key: str, value: str, ttl_seconds: float | None) -> None:
        now = self._clock()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO coordination_entries(namespace, key, value, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (self.namespace, key, value, expires_at, now),
            )

    async def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        await self._run(self._set_sync, key, value, ttl_seconds)

    def _claim_sync(self, key: str, value: str, ttl_seconds: float) -> bool:
        now = self._clock()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO coordination_entries(namespace, key, value, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE coordination_entries.expires_at IS NOT NULL
                  AND coordination_entries.expires_at <= excluded.updated_at
                """,
                (self.namespace, key, value, now + ttl_seconds, now),
            )
            return cursor.rowcount == 1

    async def claim(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        return await self._run(self._claim_sync, key, value, ttl_seconds)

    def _refresh_sync(self, key: str, value: str, ttl_seconds: float) -> bool:
        now = self._clock()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE coordination_entries
                SET expires_at = ?, updated_at = ?
                WHERE namespace = ? AND key = ? AND value = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (now + ttl_seconds, now, self.namespace, key, value, now),
            )
            return cursor.rowcount == 1

    async def refresh(self, key: str, value: str, *, ttl_seconds: float) -> bool:
        return await self._run(self._refresh_sync, key, value, ttl_seconds)

    def _delete_sync(self, key: str, expected_value: str | None) -> bool:
        with self._lock:
            if expected_value is None:
                cursor = self._connection.execute(
                    "DELETE FROM coordination_entries WHERE namespace = ? AND key = ?",
                    (self.namespace, key),
                )
            else:
                cursor = self._connection.execute(
                    """
                    DELETE FROM coordination_entries
                    WHERE namespace = ? AND key = ? AND value = ?
                    """,
                    (self.namespace, key, expected_value),
                )
            return cursor.rowcount == 1

    async def delete(self, key: str, *, expected_value: str | None = None) -> bool:
        return await self._run(self._delete_sync, key, expected_value)

    def _increment_sync(self, key: str, ttl_seconds: float | None) -> int:
        now = self._clock()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT value, expires_at FROM coordination_counters
                    WHERE namespace = ? AND key = ?
                    """,
                    (self.namespace, key),
                ).fetchone()
                if row is None or (row[1] is not None and float(row[1]) <= now):
                    value = 1
                    expires_at = now + ttl_seconds if ttl_seconds is not None else None
                else:
                    value = int(row[0]) + 1
                    expires_at = now + ttl_seconds if ttl_seconds is not None else row[1]
                self._connection.execute(
                    """
                    INSERT INTO coordination_counters(namespace, key, value, expires_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        value = excluded.value,
                        expires_at = excluded.expires_at,
                        updated_at = excluded.updated_at
                    """,
                    (self.namespace, key, value, expires_at, now),
                )
                self._connection.execute("COMMIT")
                return value
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    async def increment(self, key: str, *, ttl_seconds: float | None = None) -> int:
        return await self._run(self._increment_sync, key, ttl_seconds)

    def _cleanup_sync(self) -> int:
        now = self._clock()
        with self._lock:
            first = self._connection.execute(
                """
                DELETE FROM coordination_entries
                WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (self.namespace, now),
            ).rowcount
            second = self._connection.execute(
                """
                DELETE FROM coordination_counters
                WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (self.namespace, now),
            ).rowcount
            return first + second

    async def cleanup(self) -> int:
        return await self._run(self._cleanup_sync)

    def _close_sync(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    async def close(self) -> None:
        await self._run(self._close_sync)


def create_coordination_store(
    settings: CoordinationSettings,
    *,
    clock: Clock = time.time,
) -> CoordinationStore:
    if settings.backend == "memory":
        return InMemoryCoordinationStore(settings.namespace, clock=clock)
    return SQLiteCoordinationStore(
        settings.database_path,
        settings.namespace,
        clock=clock,
    )
