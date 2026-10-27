from types import SimpleNamespace

import pytest

from shin_ai.billing import __main__ as cli
from shin_ai.billing.store import CreditStore, Identity


@pytest.mark.asyncio
async def test_grant_replay_and_balance(tmp_path, monkeypatch, capsys):
    database = tmp_path / "credits.sqlite3"
    settings = SimpleNamespace(billing=SimpleNamespace(enabled=True, database_path=database))
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(
        [
            "grant",
            "--platform",
            "whatsapp",
            "--user",
            "123:4@s.whatsapp.net",
            "--amount-usd",
            "0.000020",
            "--reference",
            "manual-001",
        ]
    )
    await cli.run(args)
    await cli.run(args)
    assert (await CreditStore(database).balance(Identity("whatsapp", "123@s.whatsapp.net"))).available == 20
    assert "Available balance: $0.00002" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_disabled_cli_creates_no_database(tmp_path, monkeypatch):
    database = tmp_path / "credits.sqlite3"
    settings = SimpleNamespace(billing=SimpleNamespace(enabled=False, database_path=database))
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(["balance", "--platform", "discord", "--user", "1"])
    with pytest.raises(ValueError, match="Enable billing"):
        await cli.run(args)
    assert not database.exists()


@pytest.mark.asyncio
async def test_cli_rejects_negative_grant(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        billing=SimpleNamespace(enabled=True, database_path=tmp_path / "credits.sqlite3")
    )
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(
        [
            "grant",
            "--platform",
            "telegram",
            "--user",
            "1",
            "--amount-usd",
            "-1",
            "--reference",
            "bad",
        ]
    )
    with pytest.raises(ValueError):
        await cli.run(args)


def test_grant_requires_reference():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["grant", "--platform", "telegram", "--user", "1", "--amount-usd", "5"])


@pytest.mark.asyncio
async def test_reconcile_bound_order_and_repeat_partial_refund(tmp_path, monkeypatch):
    from decimal import Decimal

    from shin_ai.billing import paymob
    from shin_ai.billing.money import purchase_quote
    from shin_ai.billing.types import Checkout, PaymentState
    from shin_ai.settings import PaymobSettings

    payment_settings = PaymobSettings(
        enabled=True, merchant_id=42, integration_ids=(17,), egp_per_usd=Decimal("50"), packages=(10_000,)
    )
    billing = SimpleNamespace(
        enabled=True,
        database_path=tmp_path / "billing.sqlite3",
        paymob=payment_settings,
        web=SimpleNamespace(public_url="https://example.test"),
    )
    monkeypatch.setattr(cli, "load_settings", lambda _: SimpleNamespace(billing=billing))
    store = CreditStore(billing.database_path)
    owner = Identity("discord", "1")
    units, quote = purchase_quote(payment_settings, 10_000)
    await store.create_purchase(owner, "purchase-1", 10_000, "EGP", units, quote)
    await store.bind_checkout("purchase-1", Checkout("101", "intention-1", "https://accept.paymob.com/test"))
    total_refunded = 0

    def current():
        return PaymentState(
            "101", "9001", 10_000, "EGP", True, False, False, total_refunded, 17, 42, False, "purchase-1"
        )

    class Gateway:
        def __init__(self, *_):
            pass

        async def lookup_purchase(self, reference=None, *, order_id=None):
            assert reference is None and order_id == "101"
            return current()

        async def refund(self, transaction_id, amount_cents):
            nonlocal total_refunded
            assert transaction_id == "9001"
            total_refunded += amount_cents
            return current()

        async def close(self):
            pass

    monkeypatch.setattr(paymob, "PaymobClient", Gateway)
    await cli.run(cli.parser().parse_args(["reconcile", "--reference", "purchase-1"]))
    for _ in range(2):
        await cli.run(cli.parser().parse_args(["refund", "--reference", "purchase-1", "--amount-egp", "25"]))
    purchase = await store.get_purchase("purchase-1")
    assert purchase.refunded_cents == 5000 and not purchase.frozen
    assert (await store.balance(owner)).available == units // 2
