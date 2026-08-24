import asyncio

from shin_ai.coordination import InMemoryCoordinationStore, SQLiteCoordinationStore
from shin_ai.providers.gemini_errors import GeminiFailure, GeminiFailureKind
from shin_ai.providers.gemini_scheduler import GeminiScheduler


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def failure(kind: GeminiFailureKind, retry_after: float | None = None) -> GeminiFailure:
    return GeminiFailure(kind, None, retry_after, kind.value)


def run(coro):
    return asyncio.run(coro)


def test_failed_key_does_not_cool_healthy_keys_or_model() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        store = InMemoryCoordinationStore("test", clock=clock)
        scheduler = GeminiScheduler(
            {"key-a": "secret-a", "key-b": "secret-b"},
            ["model"],
            store,
            clock=clock,
        )

        first = await scheduler.reserve("model")
        assert first is not None
        await first.failed(failure(GeminiFailureKind.RATE_LIMIT, retry_after=60))

        second = await scheduler.reserve("model", excluded_keys={first.key_name})
        assert second is not None
        assert second.key_name != first.key_name
        snapshot = await scheduler.health_snapshot()
        assert snapshot["models"]["model"]["available"] is True
        assert snapshot["models"]["model"]["eligible_keys"] == 1
        await second.release()

    run(scenario())


def test_model_becomes_unavailable_only_after_every_pair_fails() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        scheduler = GeminiScheduler(
            {"a": "one", "b": "two", "c": "three"},
            ["model"],
            InMemoryCoordinationStore("test", clock=clock),
            clock=clock,
        )
        tried: set[str] = set()
        for remaining in (2, 1, 0):
            reservation = await scheduler.reserve("model", excluded_keys=tried)
            assert reservation is not None
            tried.add(reservation.key_name)
            await reservation.failed(failure(GeminiFailureKind.TRANSIENT))
            snapshot = await scheduler.health_snapshot()
            assert snapshot["models"]["model"]["eligible_keys"] == remaining

        assert await scheduler.reserve("model") is None
        assert (await scheduler.health_snapshot())["models"]["model"]["available"] is False

    run(scenario())


def test_pair_is_eligible_again_after_cooldown() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        scheduler = GeminiScheduler(
            {"a": "one"},
            ["model"],
            InMemoryCoordinationStore("test", clock=clock),
            clock=clock,
        )
        reservation = await scheduler.reserve("model")
        assert reservation is not None
        await reservation.failed(failure(GeminiFailureKind.RATE_LIMIT, retry_after=10))
        assert await scheduler.reserve("model") is None

        clock.advance(11)
        retry = await scheduler.reserve("model")
        assert retry is not None
        await retry.succeeded()
        assert (await scheduler.pair_health("model", "a")).status == "healthy"

    run(scenario())


def test_concurrent_reservations_do_not_share_pairs() -> None:
    async def scenario() -> None:
        keys = {f"key-{index}": f"secret-{index}" for index in range(10)}
        scheduler = GeminiScheduler(
            keys,
            ["model"],
            InMemoryCoordinationStore("test"),
        )
        reservations = await asyncio.gather(*(scheduler.reserve("model") for _ in keys))
        names = [reservation.key_name for reservation in reservations if reservation]
        assert len(names) == len(keys)
        assert len(set(names)) == len(keys)
        await asyncio.gather(*(reservation.release() for reservation in reservations if reservation))

    run(scenario())


def test_auth_failure_temporarily_disables_key_for_every_model() -> None:
    async def scenario() -> None:
        scheduler = GeminiScheduler(
            {"bad": "secret"},
            ["model-a", "model-b"],
            InMemoryCoordinationStore("test"),
        )
        reservation = await scheduler.reserve("model-a")
        assert reservation is not None
        await reservation.failed(failure(GeminiFailureKind.AUTHENTICATION))

        assert await scheduler.reserve("model-b") is None

    run(scenario())


def test_two_process_stores_reserve_different_pairs(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "coordination.sqlite3"
        first_store = SQLiteCoordinationStore(path, "shared")
        second_store = SQLiteCoordinationStore(path, "shared")
        keys = {"a": "one", "b": "two"}
        first = GeminiScheduler(keys, ["model"], first_store, owner_id="process-1")
        second = GeminiScheduler(keys, ["model"], second_store, owner_id="process-2")
        try:
            first_reservation, second_reservation = await asyncio.gather(
                first.reserve("model"), second.reserve("model")
            )
            assert first_reservation is not None
            assert second_reservation is not None
            assert first_reservation.key_name != second_reservation.key_name
            await first_reservation.release()
            await second_reservation.release()
        finally:
            await first_store.close()
            await second_store.close()

    run(scenario())


def test_different_credentials_with_the_same_label_do_not_interfere(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "coordination.sqlite3"
        first_store = SQLiteCoordinationStore(path, "shared")
        second_store = SQLiteCoordinationStore(path, "shared")
        first = GeminiScheduler(
            {"key-1": "first-secret"},
            ["model"],
            first_store,
            owner_id="process-1",
        )
        second = GeminiScheduler(
            {"key-1": "second-secret"},
            ["model"],
            second_store,
            owner_id="process-2",
        )
        try:
            first_reservation, second_reservation = await asyncio.gather(
                first.reserve("model"),
                second.reserve("model"),
            )
            assert first_reservation is not None
            assert second_reservation is not None
            await first_reservation.release()
            await second_reservation.release()
        finally:
            await first_store.close()
            await second_store.close()

    run(scenario())


def test_same_credential_with_different_labels_shares_one_lease(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "coordination.sqlite3"
        first_store = SQLiteCoordinationStore(path, "shared")
        second_store = SQLiteCoordinationStore(path, "shared")
        first = GeminiScheduler(
            {"primary": "shared-secret"},
            ["model"],
            first_store,
            owner_id="process-1",
        )
        second = GeminiScheduler(
            {"renamed": "shared-secret"},
            ["model"],
            second_store,
            owner_id="process-2",
        )
        try:
            reservations = await asyncio.gather(
                first.reserve("model"),
                second.reserve("model"),
            )
            assert sum(reservation is not None for reservation in reservations) == 1
            for reservation in reservations:
                if reservation is not None:
                    await reservation.release()
        finally:
            await first_store.close()
            await second_store.close()

    run(scenario())


def test_duplicate_credentials_in_one_file_are_deduplicated() -> None:
    scheduler = GeminiScheduler(
        {"first": "same-secret", "duplicate": "same-secret"},
        ["model"],
        InMemoryCoordinationStore("test"),
    )

    assert scheduler.keys == {"first": "same-secret"}


def test_partially_overlapping_pools_coordinate_only_the_shared_key(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "coordination.sqlite3"
        first_store = SQLiteCoordinationStore(path, "shared")
        second_store = SQLiteCoordinationStore(path, "shared")
        first = GeminiScheduler(
            {"shared-a": "shared-secret", "only-first": "first-secret"},
            ["model"],
            first_store,
            owner_id="process-1",
        )
        second = GeminiScheduler(
            {"shared-b": "shared-secret", "only-second": "second-secret"},
            ["model"],
            second_store,
            owner_id="process-2",
        )
        try:
            shared_reservation = await first.reserve(
                "model",
                excluded_keys={"only-first"},
            )
            assert shared_reservation is not None
            assert (
                await second.reserve(
                    "model",
                    excluded_keys={"only-second"},
                )
                is None
            )

            independent_reservation = await second.reserve(
                "model",
                excluded_keys={"shared-b"},
            )
            assert independent_reservation is not None
            await shared_reservation.release()
            await independent_reservation.release()
        finally:
            await first_store.close()
            await second_store.close()

    run(scenario())
