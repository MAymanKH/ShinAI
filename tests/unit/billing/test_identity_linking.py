import asyncio
import dataclasses
import sqlite3

import pytest

from shin_ai.billing.store import Balance, CreditStore, Identity, Scope
from shin_ai.billing.types import Checkout, PaymentState

DISCORD = Identity("discord", "person")
TELEGRAM = Identity("telegram", "person")
WHATSAPP = Identity("whatsapp", "person")
DISCORD_CHAT = Scope("discord", "chat", "friends")
TELEGRAM_CHAT = Scope("telegram", "chat", "family")
QUOTE = {"integration_ids": [123], "merchant_id": 45, "is_live": False}
PAYMENT = PaymentState(
    "order", "transaction", 10000, "EGP", True, False, False, 0, 123, 45, False, "purchase"
)


async def seed(tmp_path):
    store = CreditStore(tmp_path / "links.sqlite3")
    await store.grant(DISCORD, 20, "discord-seed")
    await store.grant(TELEGRAM, 30, "telegram-seed")
    return store


def test_linking_requires_unexpired_unused_private_proof(tmp_path):
    async def scenario():
        now = [100.0]
        store = CreditStore(tmp_path / "links.sqlite3", clock=lambda: now[0])
        token = await store.issue_login(TELEGRAM, 10)
        now[0] = 111
        with pytest.raises(ValueError, match="invalid or expired"):
            await store.link_identity(DISCORD, token)
        assert await store.balance(DISCORD) == Balance(0, 0)
        valid = await store.issue_login(TELEGRAM)
        assert await store.link_identity(DISCORD, valid) == DISCORD
        assert await store.redeem_login(valid) is None
        with pytest.raises(ValueError):
            await store.link_identity(DISCORD, valid)

    asyncio.run(scenario())


def test_linked_wallet_combines_funds_preserves_history_and_spending_permissions(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        await store.configure(TELEGRAM, TELEGRAM_CHAT, "only_me")
        token = await store.issue_login(TELEGRAM)
        assert await store.link_identity(DISCORD, token) == DISCORD
        assert await store.balance(DISCORD) == Balance(50, 0)
        assert await store.balance(TELEGRAM) == Balance(50, 0)
        reservation = await store.reserve("telegram-request", TELEGRAM, (TELEGRAM_CHAT,), 10)
        assert reservation.owner == DISCORD
        await store.settle(reservation, 4)
        assert await store.balance(TELEGRAM) == Balance(46, 0)
        await store.grant(TELEGRAM, 30, "telegram-seed")
        assert await store.balance(DISCORD) == Balance(46, 0)
        await store.grant(TELEGRAM, 5, "later-grant")
        assert await store.balance(DISCORD) == Balance(51, 0)
        history = await store.history(TELEGRAM)
        assert len(history["ledger"]) == 7
        assert history == await store.history(DISCORD)
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT available FROM wallets WHERE platform='telegram'"
            ).fetchone() == (0,)
        with pytest.raises(ValueError, match="same platform"):
            await store.configure(TELEGRAM, DISCORD_CHAT, "everyone")

    asyncio.run(scenario())


def test_link_moves_whole_alias_group_and_preserves_session_origin(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        token = await store.issue_login(TELEGRAM)
        login, before = await store.redeem_login(token)
        assert before.login_platform == "telegram"
        proof = await store.issue_login(TELEGRAM)
        await store.link_identity(DISCORD, proof)
        after = await store.get_session(login)
        assert after.owner == DISCORD
        assert after.login_platform == "telegram"
        await store.grant(WHATSAPP, 7, "whatsapp-seed")
        proof = await store.issue_login(DISCORD)
        await store.link_identity(WHATSAPP, proof)
        assert await store.balance(TELEGRAM) == Balance(57, 0)
        assert await store.balance(DISCORD) == Balance(57, 0)
        assert await store.balance(WHATSAPP) == Balance(57, 0)
        assert (await store.get_session(login)).owner == WHATSAPP
        assert (await store.get_session(login)).login_platform == "telegram"

    asyncio.run(scenario())


def test_linking_rejects_either_active_wallet_and_keeps_proof_unused(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        await store.configure(TELEGRAM, TELEGRAM_CHAT, "only_me")
        reservation = await store.reserve("active", TELEGRAM, (TELEGRAM_CHAT,), 10)
        proof = await store.issue_login(TELEGRAM)
        with pytest.raises(ValueError, match="active requests"):
            await store.link_identity(DISCORD, proof)
        assert await store.balance(DISCORD) == Balance(20, 0)
        assert await store.balance(TELEGRAM) == Balance(20, 10)
        await store.release(reservation)
        await store.link_identity(DISCORD, proof)
        assert await store.balance(DISCORD) == Balance(50, 0)

    asyncio.run(scenario())


def test_old_payment_reversals_follow_linked_wallet_and_debt_is_combined(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        await store.create_purchase(TELEGRAM, "purchase", 10000, "EGP", 100, QUOTE)
        await store.bind_checkout("purchase", Checkout("order", "intent", "https://checkout.example"))
        await store.apply_payment(PAYMENT)
        await store.configure(TELEGRAM, TELEGRAM_CHAT, "only_me")
        held = await store.reserve("spend", TELEGRAM, (TELEGRAM_CHAT,), 120)
        await store.settle(held, 120)
        proof = await store.issue_login(TELEGRAM)
        await store.link_identity(DISCORD, proof)
        assert await store.balance(DISCORD) == Balance(30, 0)
        await store.apply_payment(dataclasses.replace(PAYMENT, refunded_cents=10000))
        assert await store.balance(TELEGRAM) == Balance(0, 0, 70)
        assert await store.get_purchase("purchase", DISCORD) is not None
        assert len((await store.history(DISCORD))["purchases"]) == 1
        await store.grant(DISCORD, 80, "repay")
        assert await store.balance(TELEGRAM) == Balance(10, 0)

    asyncio.run(scenario())


def test_frozen_refund_cannot_be_evaded_by_linking(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        await store.create_purchase(TELEGRAM, "purchase", 10000, "EGP", 100, QUOTE)
        await store.bind_checkout("purchase", Checkout("order", "intent", "https://checkout.example"))
        await store.apply_payment(PAYMENT)
        await store.freeze_purchase("purchase", 10000)
        proof = await store.issue_login(TELEGRAM)
        with pytest.raises(ValueError, match="pending refunds"):
            await store.link_identity(DISCORD, proof)

    asyncio.run(scenario())


def test_merge_offsets_existing_debt_with_other_accounts_funds(tmp_path):
    async def scenario():
        store = await seed(tmp_path)
        await store.create_purchase(TELEGRAM, "purchase", 10000, "EGP", 100, QUOTE)
        await store.bind_checkout("purchase", Checkout("order", "intent", "https://checkout.example"))
        await store.apply_payment(PAYMENT)
        await store.configure(TELEGRAM, TELEGRAM_CHAT, "only_me")
        reservation = await store.reserve("spent", TELEGRAM, (TELEGRAM_CHAT,), 120)
        await store.settle(reservation, 120)
        await store.apply_payment(dataclasses.replace(PAYMENT, refunded_cents=10000))
        assert await store.balance(TELEGRAM) == Balance(0, 0, 90)
        proof = await store.issue_login(TELEGRAM)
        await store.link_identity(DISCORD, proof)
        assert await store.balance(TELEGRAM) == Balance(0, 0, 70)
        assert await store.balance(DISCORD) == Balance(0, 0, 70)
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT available,reserved,debt FROM wallets WHERE platform='telegram'"
            ).fetchone() == (0, 0, 0)

    asyncio.run(scenario())
