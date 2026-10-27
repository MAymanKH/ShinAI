import asyncio
import sqlite3

import pytest

from shin_ai.billing.store import Balance, CreditStore, DuplicateRequest, Identity, LegacyBalanceError, Scope
from shin_ai.billing.types import UsageCall

OWNER = Identity("discord", "payer")
SCOPE = Scope("discord", "chat", "friends")


async def funded(tmp_path, amount=100):
    store = CreditStore(tmp_path / "money.sqlite3")
    await store.grant(OWNER, amount, "seed")
    await store.configure(OWNER, SCOPE, "only_me")
    return store


def call(reference="call", cost=1500, attempt="attempt", usage="{}"):
    return UsageCall(reference, attempt, "deepseek", "model", cost, usage, '{"version":"v1"}')


def test_variable_settlement_releases_unused_hold(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        reservation = await store.reserve("request", OWNER, (SCOPE,), 50)
        await store.settle(reservation, 7)
        assert await store.balance(OWNER) == Balance(93, 0)
        await store.settle(reservation, 20)
        await store.release(reservation)
        assert await store.balance(OWNER) == Balance(93, 0)
        with pytest.raises(ValueError, match="exceeds"):
            await store.settle(reservation, 51)
        with pytest.raises(ValueError):
            await store.settle(reservation, -1)

    asyncio.run(scenario())


def test_pending_calls_complete_once_and_round_total_once(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        reservation = await store.reserve("request", OWNER, (SCOPE,), 50)
        await store.configure_reservation(reservation, 2500)
        await store.record_call(reservation, call(cost=None))
        with pytest.raises(ValueError, match="incomplete"):
            await store.accept_attempt(reservation, "attempt")
        await store.record_call(reservation, call(cost=1500))
        await store.record_call(reservation, call(cost=1500))
        await store.record_call(reservation, call("call-2", cost=1500))
        await store.accept_attempt(reservation, "attempt")
        assert await store.final_cost(reservation, 2500) == 0
        await store.mark_delivered(reservation)
        assert await store.final_cost(reservation, 2500) == 4
        with pytest.raises(ValueError, match="Conflicting"):
            await store.record_call(reservation, call(cost=1501))
        with pytest.raises(ValueError, match="cannot change"):
            await store.configure_reservation(reservation, 5000)
        await store.mark_delivered(reservation)
        assert await store.recover() == 1
        assert await store.balance(OWNER) == Balance(96, 0)
        history = await store.history(OWNER)
        assert history["usage"][0]["customer_charge"] == 4
        assert history["usage"][0]["cost_nano_usd"] == 3000
        assert history["usage"][0]["provider"] == "deepseek"
        assert history["usage"][0]["model"] == "model"

    asyncio.run(scenario())


def test_failed_attempt_cost_is_internal_and_delivered_cost_capped(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        reservation = await store.reserve("request", OWNER, (SCOPE,), 10)
        await store.record_call(reservation, call("failed", cost=90000, attempt="failed"))
        await store.record_call(reservation, call("success", cost=15000))
        await store.accept_attempt(reservation, "attempt")
        await store.mark_delivered(reservation)
        assert await store.final_cost(reservation, 0) == 10
        await store.mark_delivered(reservation)
        assert await store.recover() == 1
        assert await store.balance(OWNER) == Balance(90, 0)

    asyncio.run(scenario())


def test_undelivered_known_usage_is_refunded_on_recovery(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        reservation = await store.reserve("request", OWNER, (SCOPE,), 50)
        await store.record_call(reservation, call(cost=9000))
        await store.accept_attempt(reservation, "attempt")
        assert await store.recover() == 1
        assert await store.balance(OWNER) == Balance(100, 0)
        with pytest.raises(DuplicateRequest):
            await store.reserve("request", OWNER, (SCOPE,), 50)

    asyncio.run(scenario())


def make_legacy(path, *, held=False):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE wallets(platform TEXT,user_id TEXT,available INTEGER,reserved INTEGER,PRIMARY KEY(platform,user_id));
            CREATE TABLE permissions(owner_platform TEXT,owner_id TEXT,platform TEXT,kind TEXT,scope_id TEXT,mode TEXT);
            CREATE TABLE grants(reference TEXT,owner_platform TEXT,owner_id TEXT,credits INTEGER);
            CREATE TABLE reservations(reference TEXT,owner_platform TEXT,owner_id TEXT,credits INTEGER,status TEXT);
            CREATE TABLE ledger(event_reference TEXT,owner_platform TEXT,owner_id TEXT,available_delta INTEGER,reserved_delta INTEGER,kind TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO wallets VALUES ('discord','payer',5,0);
            INSERT INTO permissions VALUES ('discord','payer','discord','chat','friends','only_me');
            INSERT INTO grants VALUES ('old','discord','payer',5);
            INSERT INTO ledger(event_reference,owner_platform,owner_id,available_delta,reserved_delta,kind) VALUES ('grant:old','discord','payer',5,0,'grant');
        """)
        if held:
            connection.execute("UPDATE wallets SET reserved=1,available=4")
            connection.execute("INSERT INTO reservations VALUES ('pending','discord','payer',1,'held')")


def test_legacy_currency_is_never_silently_reinterpreted(tmp_path):
    async def scenario():
        path = tmp_path / "legacy.sqlite3"
        make_legacy(path)
        store = CreditStore(path)
        with pytest.raises(LegacyBalanceError, match="explicit conversion"):
            await store.balance(OWNER)
        assert await store.migrate_legacy(1000) == 1
        assert await store.balance(OWNER) == Balance(5000, 0)
        assert await store.migrate_legacy(1000) == 0
        with pytest.raises(LegacyBalanceError, match="already uses"):
            await store.migrate_legacy(2000)
        reservation = await store.reserve("new", OWNER, (SCOPE,), 2000)
        assert reservation is not None
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT available FROM legacy_wallets").fetchone()[0] == 5
            assert connection.execute("SELECT COUNT(*) FROM legacy_ledger").fetchone()[0] == 1
            assert (
                connection.execute("SELECT kind FROM ledger ORDER BY rowid LIMIT 1").fetchone()[0]
                == "legacy_conversion"
            )

    asyncio.run(scenario())


def test_migration_refuses_outstanding_holds(tmp_path):
    async def scenario():
        path = tmp_path / "legacy.sqlite3"
        make_legacy(path, held=True)
        with pytest.raises(LegacyBalanceError, match="reservations"):
            await CreditStore(path).migrate_legacy(1000)
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT available,reserved FROM wallets").fetchone() == (4, 1)

    asyncio.run(scenario())


def test_unknown_schema_version_fails_closed(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        with sqlite3.connect(store.path) as connection:
            connection.execute("UPDATE billing_meta SET value='99' WHERE key='version'")
        with pytest.raises(LegacyBalanceError, match="schema version"):
            await store.balance(OWNER)

    asyncio.run(scenario())


def test_empty_attempt_cannot_be_accepted_as_free_usage(tmp_path):
    async def scenario():
        store = await funded(tmp_path)
        held = await store.reserve("request", OWNER, (SCOPE,), 50)
        with pytest.raises(ValueError, match="without recorded calls"):
            await store.accept_attempt(held, "empty")
        await store.configure_reservation(held, 1000)
        with pytest.raises(ValueError, match="pricing snapshot"):
            await store.final_cost(held, 2000)

    asyncio.run(scenario())
