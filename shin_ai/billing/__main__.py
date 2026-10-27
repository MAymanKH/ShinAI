"""Local administration for monetary prepaid balances and payment reconciliation."""

import argparse
import asyncio
from pathlib import Path

from shin_ai.billing.money import format_usd, usd_to_units
from shin_ai.billing.store import CreditStore, Identity
from shin_ai.settings import load_settings
from shin_ai.utils.chat_identity import normalize_chat_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=None)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("grant", "balance"):
        command = commands.add_parser(name)
        command.add_argument("--platform", choices=("telegram", "discord", "whatsapp"), required=True)
        command.add_argument("--user", required=True)
        if name == "grant":
            command.add_argument("--amount-usd", required=True)
            command.add_argument("--reference", required=True)
    migrate = commands.add_parser(
        "migrate-legacy", help="Explicitly convert old response credits to microUSD"
    )
    migrate.add_argument("--micro-usd-per-credit", type=int, required=True)
    reconcile = commands.add_parser("reconcile", help="Verify a purchase directly with Paymob")
    reconcile.add_argument("--reference", required=True)
    refund = commands.add_parser("refund", help="Request a refund, then reconcile its cumulative total")
    refund.add_argument("--reference", required=True)
    refund.add_argument("--amount-egp", required=True)
    return result


async def run(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    if not settings.billing.enabled:
        raise ValueError("Enable billing in the selected configuration before managing balances.")
    store = CreditStore(settings.billing.database_path)
    if args.command == "migrate-legacy":
        import fcntl

        settings.billing.database_path.parent.mkdir(parents=True, exist_ok=True)
        with await asyncio.to_thread(open, f"{settings.billing.database_path}.owner.lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            count = await store.migrate_legacy(args.micro_usd_per_credit)
        print(f"Converted {count} accounts. Historical credit records were archived.")
        return
    if args.command in {"reconcile", "refund"}:
        from shin_ai.billing.money import decimal_amount
        from shin_ai.billing.paymob import PaymobClient

        purchase = await store.get_purchase(args.reference)
        if purchase is None:
            raise ValueError("Purchase not found")
        gateway = PaymobClient(settings.billing.paymob, settings.billing.web.public_url)
        try:
            if args.command == "refund":
                if purchase.frozen:
                    raise ValueError(
                        "A refund is already pending. Reconcile this purchase before any further refund."
                    )
                if not purchase.transaction_id or purchase.status not in {"paid", "partially_refunded"}:
                    raise ValueError("Only confirmed paid purchases can be refunded")
                scaled = decimal_amount(args.amount_egp, "Refund amount") * 100
                if scaled != scaled.to_integral_value():
                    raise ValueError("Refund amount requires at most two decimal places")
                cents = int(scaled)
                if purchase.refunded_cents + cents > purchase.amount_cents:
                    raise ValueError("Refund exceeds the remaining payment amount")
                await store.freeze_purchase(purchase.reference, purchase.refunded_cents + cents)
                state = await gateway.refund(purchase.transaction_id, cents)
            elif purchase.transaction_id:
                state = await gateway.fetch_transaction(purchase.transaction_id)
            else:
                state = await gateway.lookup_purchase(
                    None if purchase.order_id else purchase.reference, order_id=purchase.order_id
                )
            if state is None:
                print("No confirmed gateway transaction yet. Purchase remains pending.")
                return
            if purchase.order_id is None:
                await store.restore_order(purchase.reference, state)
            updated = await store.apply_payment(state)
            print(
                f"Purchase {updated.reference}: {updated.status}; refunded EGP {updated.refunded_cents / 100:.2f}"
            )
        finally:
            await gateway.close()
        return
    owner = Identity(args.platform, normalize_chat_id(args.platform, args.user))
    if args.command == "grant":
        await store.grant(owner, usd_to_units(args.amount_usd), args.reference)
    balance = await store.balance(owner)
    print(f"Available balance: {format_usd(balance.available)}\nReserved: {format_usd(balance.reserved)}")
    if balance.debt:
        print(f"Balance adjustment due: {format_usd(balance.debt)}")


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ValueError, OSError, RuntimeError) as error:
        argument_parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
