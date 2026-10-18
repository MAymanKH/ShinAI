import asyncio
import sqlite3

import pytest

from shin_ai.billing.store import (
    Balance,
    CreditStore,
    DuplicateRequest,
    Identity,
    Reservation,
    Scope,
    SponsorConflict,
)

ALICE = Identity("discord", "alice")
BOB = Identity("discord", "bob")
CAROL = Identity("discord", "carol")
CHAT = Scope("discord", "chat", "channel")
GUILD = Scope("discord", "guild", "server")


def test_grants_are_durable_and_idempotent(tmp_path):
    async def scenario():
        path = tmp_path / "billing.sqlite3"
        store = CreditStore(path)
        assert await store.balance(ALICE) == Balance(0, 0)
        assert await store.grant(ALICE, 5, "payment") == 5
        assert await CreditStore(path).grant(ALICE, 5, "payment") == 5
        for owner, amount in ((ALICE, 6), (BOB, 5)):
            with pytest.raises(ValueError, match="different grant"):
                await store.grant(owner, amount, "payment")
        assert await store.balance(ALICE) == Balance(5, 0)
        assert await store.balance(BOB) == Balance(0, 0)
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM ledger").fetchone() == (1,)
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute("DELETE FROM ledger")

    asyncio.run(scenario())


@pytest.mark.parametrize("credits", [0, -1, True, 1.5, 2**63])
def test_invalid_grants_do_not_create_database(tmp_path, credits):
    path = tmp_path / "billing.sqlite3"
    with pytest.raises(ValueError):
        asyncio.run(CreditStore(path).grant(ALICE, credits, "grant"))
    assert not path.exists()


def test_concurrent_connections_cannot_spend_last_credit_twice(tmp_path):
    async def scenario():
        path = tmp_path / "billing.sqlite3"
        store = CreditStore(path)
        await store.grant(ALICE, 1, "grant")
        await store.configure(ALICE, CHAT, "everyone")
        reservations = await asyncio.gather(
            *(CreditStore(path).reserve(f"request-{index}", BOB, (CHAT,), 1) for index in range(12))
        )
        assert len([result for result in reservations if result is not None]) == 1
        assert await store.balance(ALICE) == Balance(0, 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["settle", "release"])
def test_terminal_requests_are_idempotent_and_cannot_replay(tmp_path, terminal):
    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        await store.grant(ALICE, 2, "grant")
        await store.configure(ALICE, CHAT, "only_me")
        held = await store.reserve("request", ALICE, (CHAT,), 1)
        assert held == Reservation("request", ALICE, 1)
        assert await store.balance(ALICE) == Balance(1, 1)
        await getattr(store, terminal)(held)
        await store.release(held)
        await store.settle(held)
        assert await store.balance(ALICE) == Balance(1 if terminal == "settle" else 2, 0)
        with pytest.raises(DuplicateRequest):
            await store.reserve("request", ALICE, (CHAT,), 1)
        with pytest.raises(ValueError, match="details"):
            await store.release(Reservation("request", BOB, 1))

    asyncio.run(scenario())


def test_recovery_is_explicit_and_preserves_terminal_requests(tmp_path):
    async def scenario():
        path = tmp_path / "billing.sqlite3"
        store = CreditStore(path)
        await store.grant(ALICE, 3, "grant")
        await store.configure(ALICE, CHAT, "only_me")
        settled = await store.reserve("delivered", ALICE, (CHAT,), 1)
        await store.settle(settled)
        await store.reserve("interrupted", ALICE, (CHAT,), 1)
        reopened = CreditStore(path)
        assert await reopened.balance(ALICE) == Balance(1, 1)
        assert await reopened.recover() == 1
        assert await reopened.recover() == 0
        assert await reopened.balance(ALICE) == Balance(2, 0)
        with pytest.raises(DuplicateRequest):
            await reopened.reserve("interrupted", ALICE, (CHAT,), 1)

    asyncio.run(scenario())


def test_own_balance_then_chat_sponsor_then_guild_sponsor(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        for owner in (ALICE, BOB, CAROL):
            await store.grant(owner, 1, owner.user_id)
        await store.configure(ALICE, GUILD, "everyone")
        await store.configure(BOB, CHAT, "everyone")
        await store.configure(CAROL, GUILD, "only_me")
        for reference, expected in (("own", CAROL), ("chat", BOB), ("guild", ALICE)):
            reservation = await store.reserve(reference, CAROL, (GUILD, CHAT), 1)
            assert reservation.owner == expected
            await store.settle(reservation)
        assert await store.reserve("empty", CAROL, (CHAT, GUILD), 1) is None

    asyncio.run(scenario())


def test_only_me_and_off_cannot_modify_another_sponsor(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        await store.grant(ALICE, 1, "grant")
        await store.configure(ALICE, CHAT, "only_me")
        assert await store.reserve("not-eligible", BOB, (CHAT,), 1) is None
        await store.configure(ALICE, CHAT, "everyone")
        with pytest.raises(SponsorConflict):
            await store.configure(BOB, CHAT, "everyone")
        await store.configure(BOB, CHAT, "off")
        assert (await store.reserve("shared", BOB, (CHAT,), 1)).owner == ALICE
        await store.configure(ALICE, CHAT, "off")
        await store.configure(BOB, CHAT, "everyone")

    asyncio.run(scenario())


def test_platform_identity_is_namespaced(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        owner = Identity("telegram", "alice")
        await store.grant(ALICE, 1, "grant")
        assert await store.balance(owner) == Balance(0, 0)
        with pytest.raises(ValueError, match="same platform"):
            await store.configure(owner, CHAT, "everyone")
        with pytest.raises(ValueError, match="same platform"):
            await store.reserve("cross-platform", owner, (CHAT,), 1)

    asyncio.run(scenario())


def test_duplicate_concurrent_request_reserves_once(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        await store.grant(ALICE, 2, "grant")
        await store.configure(ALICE, CHAT, "only_me")
        results = await asyncio.gather(
            store.reserve("same", ALICE, (CHAT,), 1),
            store.reserve("same", ALICE, (CHAT,), 1),
            return_exceptions=True,
        )
        assert sum(isinstance(result, Reservation) for result in results) == 1
        assert sum(isinstance(result, DuplicateRequest) for result in results) == 1
        assert await store.balance(ALICE) == Balance(1, 1)

    asyncio.run(scenario())


def test_cancellation_during_reservation_refunds_completed_transaction(tmp_path, monkeypatch):
    import threading

    async def scenario():
        store = CreditStore(tmp_path / "billing.sqlite3")
        await store.grant(ALICE, 1, "grant")
        await store.configure(ALICE, CHAT, "only_me")
        started = threading.Event()
        finish = threading.Event()
        original_record = store._record

        def slow_record(connection, reference, owner, available, reserved, kind):
            if kind == "reserve":
                started.set()
                assert finish.wait(timeout=5)
            original_record(connection, reference, owner, available, reserved, kind)

        monkeypatch.setattr(store, "_record", slow_record)
        request = asyncio.create_task(store.reserve("cancelled", ALICE, (CHAT,), 1))
        assert await asyncio.to_thread(started.wait, 5)
        request.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert await store.balance(ALICE) == Balance(1, 0)
        with pytest.raises(DuplicateRequest):
            await store.reserve("cancelled", ALICE, (CHAT,), 1)

    asyncio.run(scenario())
