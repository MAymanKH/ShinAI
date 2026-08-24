import asyncio

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
