"""Local administrator commands for prepaid test credits."""

import argparse
import asyncio
from pathlib import Path

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
            command.add_argument("--credits", type=int, required=True)
            command.add_argument("--reference", required=True)
    return result


async def run(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    if not settings.billing.enabled:
        raise ValueError("Enable billing in the selected configuration before managing credits.")
    store = CreditStore(settings.billing.database_path)
    owner = Identity(args.platform, normalize_chat_id(args.platform, args.user))
    if args.command == "grant":
        await store.grant(owner, args.credits, args.reference)
    balance = await store.balance(owner)
    print(f"Available credits: {balance.available}\nReserved credits: {balance.reserved}")


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ValueError, OSError) as error:
        argument_parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
