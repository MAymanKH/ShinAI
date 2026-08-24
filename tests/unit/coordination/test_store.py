import asyncio
import sqlite3
import time

from shin_ai.coordination import InMemoryCoordinationStore, SQLiteCoordinationStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def run(coro):
    return asyncio.run(coro)


def test_memory_claim_refresh_and_expiry() -> None:
    clock = FakeClock()
    store = InMemoryCoordinationStore("test", clock=clock)

    assert run(store.claim("lease", "owner-a", ttl_seconds=10)) is True
    assert run(store.claim("lease", "owner-b", ttl_seconds=10)) is False
    assert run(store.refresh("lease", "owner-b", ttl_seconds=10)) is False
    assert run(store.refresh("lease", "owner-a", ttl_seconds=10)) is True

    clock.advance(11)
    assert run(store.claim("lease", "owner-b", ttl_seconds=10)) is True
    assert run(store.get("lease")) == "owner-b"


def test_sqlite_instances_share_atomic_claims_and_counters(tmp_path) -> None:
    path = tmp_path / "coordination.sqlite3"
    first = SQLiteCoordinationStore(path, "shared")
    second = SQLiteCoordinationStore(path, "shared")
    try:
        assert run(first.claim("event:1", "first", ttl_seconds=60)) is True
        assert run(second.claim("event:1", "second", ttl_seconds=60)) is False
        assert run(second.get("event:1")) == "first"

        assert run(first.increment("cursor")) == 1
        assert run(second.increment("cursor")) == 2
        assert run(first.increment("cursor")) == 3
    finally:
        run(first.close())
        run(second.close())


def test_sqlite_namespaces_are_independent(tmp_path) -> None:
    path = tmp_path / "coordination.sqlite3"
    first = SQLiteCoordinationStore(path, "bot-a")
    second = SQLiteCoordinationStore(path, "bot-b")
    try:
        assert run(first.claim("same-event", "a", ttl_seconds=60)) is True
        assert run(second.claim("same-event", "b", ttl_seconds=60)) is True
        assert run(first.get("same-event")) == "a"
        assert run(second.get("same-event")) == "b"
    finally:
        run(first.close())
        run(second.close())


def test_sqlite_expired_claim_can_be_replaced_and_cleaned(tmp_path) -> None:
    clock = FakeClock()
    store = SQLiteCoordinationStore(tmp_path / "coordination.sqlite3", "test", clock=clock)
    try:
        assert run(store.claim("lease", "old", ttl_seconds=5)) is True
        clock.advance(6)
        assert run(store.claim("lease", "new", ttl_seconds=5)) is True
        clock.advance(6)
        assert run(store.cleanup()) == 1
        assert run(store.get("lease")) is None
    finally:
        run(store.close())


def test_delete_can_require_ownership() -> None:
    store = InMemoryCoordinationStore("test")
    run(store.set("lease", "owner"))

    assert run(store.delete("lease", expected_value="other")) is False
    assert run(store.delete("lease", expected_value="owner")) is True


def test_get_many_returns_only_live_requested_values() -> None:
    clock = FakeClock()
    store = InMemoryCoordinationStore("test", clock=clock)
    run(store.set("live", "one", ttl_seconds=20))
    run(store.set("expired", "two", ttl_seconds=5))
    run(store.set("unrequested", "three"))
    clock.advance(10)

    assert run(store.get_many(["live", "expired", "missing"])) == {"live": "one"}


def test_sqlite_get_many_handles_large_key_sets(tmp_path) -> None:
    store = SQLiteCoordinationStore(tmp_path / "coordination.sqlite3", "test")
    try:
        for index in range(450):
            run(store.set(f"key-{index}", str(index)))

        values = run(store.get_many([f"key-{index}" for index in range(450)]))
        assert len(values) == 450
        assert values["key-0"] == "0"
        assert values["key-449"] == "449"
    finally:
        run(store.close())


def test_sqlite_lock_contention_does_not_block_event_loop(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "coordination.sqlite3"
        store = SQLiteCoordinationStore(path, "test")
        blocker = sqlite3.connect(path, isolation_level=None)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            started_at = time.monotonic()
            write = asyncio.create_task(store.set("key", "value"))

            await asyncio.sleep(0.05)

            assert time.monotonic() - started_at < 0.5
            assert not write.done()
            blocker.execute("ROLLBACK")
            await asyncio.wait_for(write, timeout=1)
            assert await store.get("key") == "value"
        finally:
            if blocker.in_transaction:
                blocker.execute("ROLLBACK")
            blocker.close()
            await store.close()

    run(scenario())
