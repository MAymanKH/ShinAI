import asyncio
import dataclasses
import sqlite3

import pytest

from shin_ai.billing.store import Balance, CreditStore, Identity, Scope
from shin_ai.billing.types import Checkout, PaymentState

ALICE = Identity("discord", "alice")
BOB = Identity("discord", "bob")
CHAT = Scope("discord", "chat", "friends")
QUOTE = {"integration_ids": [123], "merchant_id": 45, "is_live": False, "fx": "0.02"}


def state(**changes):
    result = PaymentState(
        "order", "transaction", 10000, "EGP", True, False, False, 0, 123, 45, False, "purchase"
    )
    return dataclasses.replace(result, **changes)


async def purchase(tmp_path, *, units=100):
    store = CreditStore(tmp_path / "account.sqlite3")
    result = await store.create_purchase(ALICE, "purchase", 10000, "EGP", units, QUOTE)
    assert result.status == "created"
    assert await store.claim_checkout("purchase") is True
    assert await store.claim_checkout("purchase") is False
    await store.bind_checkout("purchase", Checkout("order", "intention", "https://checkout.example"))
    return store


def test_login_is_single_use_expiring_and_only_hashes_are_stored(tmp_path):
    async def scenario():
        now = [1000.0]
        store = CreditStore(tmp_path / "account.sqlite3", clock=lambda: now[0])
        token = await store.issue_login(ALICE, 60)
        result = await store.redeem_login(token, 100)
        session_token, session = result
        assert session.owner == ALICE
        assert len(session.csrf) > 30
        assert await store.redeem_login(token) is None
        assert await store.get_session(session_token) == session
        with sqlite3.connect(store.path) as connection:
            assert token not in str(connection.execute("SELECT * FROM login_tokens").fetchall())
            assert session_token not in str(connection.execute("SELECT * FROM sessions").fetchall())
        now[0] += 101
        assert await store.get_session(session_token) is None
        old = await store.issue_login(ALICE, 10)
        now[0] += 11
        assert await store.redeem_login(old) is None

    asyncio.run(scenario())


def test_login_redemption_race_has_one_winner_and_logout_revokes(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "account.sqlite3")
        token = await store.issue_login(ALICE)
        results = await asyncio.gather(store.redeem_login(token), store.redeem_login(token))
        winners = [result for result in results if result is not None]
        assert len(winners) == 1
        session_token, session = winners[0]
        assert session.owner == ALICE
        await store.revoke_session(session_token)
        assert await store.get_session(session_token) is None

    asyncio.run(scenario())


def test_payment_credits_once_and_is_owner_isolated(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        same = await store.create_purchase(ALICE, "purchase", 10000, "EGP", 100, QUOTE)
        assert same.order_id == "order"
        with pytest.raises(ValueError, match="different purchase"):
            await store.create_purchase(BOB, "purchase", 10000, "EGP", 100, QUOTE)
        assert await store.get_purchase("purchase", BOB) is None
        assert await store.get_purchase_by_order("order") == same
        for _ in range(3):
            assert (await store.apply_payment(state())).status == "paid"
        assert await store.balance(ALICE) == Balance(100, 0)
        assert await store.balance(BOB) == Balance(0, 0)
        history = await store.history(ALICE)
        assert len(history["purchases"]) == 1
        assert len(history["ledger"]) == 1
        assert await store.history(BOB) == {"purchases": [], "usage": [], "ledger": []}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"order_id": "other"},
        {"amount_cents": 9999},
        {"currency": "USD"},
        {"integration_id": 999},
        {"merchant_id": 999},
        {"is_live": True},
        {"special_reference": "other"},
        {"refunded_cents": 10001},
    ],
)
def test_payment_mismatch_never_changes_balance(tmp_path, changes):
    async def scenario():
        store = await purchase(tmp_path)
        with pytest.raises(ValueError):
            await store.apply_payment(state(**changes))
        assert await store.balance(ALICE) == Balance(0, 0)

    asyncio.run(scenario())


def test_pending_and_failed_events_cannot_downgrade_paid_or_regrant(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        assert (await store.apply_payment(state(success=False, pending=True))).status == "pending"
        assert (await store.apply_payment(state(success=False))).status == "failed"
        await store.apply_payment(state())
        assert (await store.apply_payment(state(success=False))).status == "paid"
        assert (await store.apply_payment(state(pending=True))).status == "paid"
        assert await store.balance(ALICE) == Balance(100, 0)

    asyncio.run(scenario())


def test_partial_full_refunds_are_monotonic_and_never_regrant(tmp_path):
    async def scenario():
        store = await purchase(tmp_path, units=101)
        await store.apply_payment(state())
        result = await store.apply_payment(state(refunded_cents=2500))
        assert result.status == "partially_refunded"
        assert await store.balance(ALICE) == Balance(75, 0)
        await store.apply_payment(state(refunded_cents=1000))
        assert await store.balance(ALICE) == Balance(75, 0)
        await store.apply_payment(state())
        assert await store.balance(ALICE) == Balance(75, 0)
        await store.apply_payment(state(refunded_cents=10000))
        await store.apply_payment(state())
        assert await store.balance(ALICE) == Balance(0, 0)

    asyncio.run(scenario())


def test_refund_seen_before_success_does_not_mint_full_credit(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        await store.apply_payment(state(success=False, refunded_cents=4000))
        assert await store.balance(ALICE) == Balance(0, 0)
        await store.apply_payment(state())
        assert await store.balance(ALICE) == Balance(60, 0)
        await store.apply_payment(state())
        assert await store.balance(ALICE) == Balance(60, 0)

    asyncio.run(scenario())


def test_spent_refund_becomes_debt_and_future_grants_repay_first(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        await store.apply_payment(state())
        await store.configure(ALICE, CHAT, "everyone")
        reservation = await store.reserve("used", BOB, (CHAT,), 80)
        await store.settle(reservation, 80)
        await store.apply_payment(state(refunded_cents=10000))
        assert await store.balance(ALICE) == Balance(0, 0, 80)
        assert await store.reserve("debt", BOB, (CHAT,), 1) is None
        await store.grant(ALICE, 50, "repay-1")
        assert await store.balance(ALICE) == Balance(0, 0, 30)
        await store.grant(ALICE, 40, "repay-2")
        assert await store.balance(ALICE) == Balance(10, 0, 0)

    asyncio.run(scenario())


def test_refund_during_hold_and_release_repay_debt_without_restoring_reversed_funds(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        await store.apply_payment(state())
        await store.configure(ALICE, CHAT, "only_me")
        reservation = await store.reserve("inflight", ALICE, (CHAT,), 100)
        await store.apply_payment(state(refunded_cents=10000))
        assert await store.balance(ALICE) == Balance(0, 100, 100)
        await store.release(reservation)
        assert await store.balance(ALICE) == Balance(0, 0, 0)

    asyncio.run(scenario())


def test_freeze_blocks_spending_until_verified_payment_refresh(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        await store.apply_payment(state())
        await store.configure(ALICE, CHAT, "only_me")
        await store.freeze_purchase("purchase", 5000)
        assert (await store.get_purchase("purchase")).frozen
        with pytest.raises(ValueError, match="already pending"):
            await store.freeze_purchase("purchase", 5000)
        assert await store.reserve("frozen", ALICE, (CHAT,), 1) is None
        await store.apply_payment(state())
        assert (await store.get_purchase("purchase")).frozen
        assert await store.reserve("stale-refund", ALICE, (CHAT,), 1) is None
        await store.apply_payment(state(refunded_cents=5000))
        assert not (await store.get_purchase("purchase")).frozen
        assert await store.reserve("thawed", ALICE, (CHAT,), 1) is not None

    asyncio.run(scenario())


def test_checkout_binding_cannot_change_gateway_identity(tmp_path):
    async def scenario():
        store = await purchase(tmp_path)
        await store.bind_checkout("purchase", Checkout("order", "intention", "https://checkout.example"))
        with pytest.raises(ValueError, match="cannot change"):
            await store.bind_checkout("purchase", Checkout("other", "intention", "https://checkout.example"))
        assert len(await store.list_purchases("pending")) == 1
        await store.apply_payment(state())
        with pytest.raises(ValueError, match="transaction"):
            await store.apply_payment(state(transaction_id="other"))
        assert len(await store.list_purchases("paid")) == 1

    asyncio.run(scenario())


def test_unknown_checkout_can_be_restored_only_from_matching_canonical_payment(tmp_path):
    async def scenario():
        store = CreditStore(tmp_path / "restore.sqlite3")
        await store.create_purchase(ALICE, "purchase", 10000, "EGP", 100, QUOTE)
        await store.claim_checkout("purchase")
        for changed in (
            state(special_reference=None),
            state(special_reference="other"),
            state(amount_cents=5000),
        ):
            with pytest.raises(ValueError):
                await store.restore_order("purchase", changed)
        await store.restore_order("purchase", state())
        assert await store.balance(ALICE) == Balance(0, 0)
        await store.apply_payment(state())
        await store.bind_checkout("purchase", Checkout("order", "intention", "https://checkout.example"))
        assert (await store.get_purchase("purchase")).status == "paid"
        assert await store.balance(ALICE) == Balance(100, 0)

    asyncio.run(scenario())
