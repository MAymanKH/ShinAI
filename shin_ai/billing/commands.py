"""Optional prepaid usage commands shared by chat platforms."""

from shin_ai.billing.runtime import get_credit_store
from shin_ai.billing.store import Identity, Scope, SponsorConflict
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.settings import get_settings
from shin_ai.utils.chat_identity import normalize_chat_id
from shin_ai.utils.logger_config import logger


def is_usage_command(text: str | None) -> bool:
    parts = (text or "").split()
    return bool(parts and parts[0].split("@", 1)[0].lower() in {"/balance", "/paid_usage"})


async def handle_usage_command(platform: PlatformAdapter, msg: UnifiedMessage) -> bool:
    """Return whether this message belongs to the optional usage commands."""
    if not get_settings().billing.enabled or not is_usage_command(msg.text):
        return False
    parts = (msg.text or "").split()
    command, separator, recipient = parts[0].partition("@")
    if separator:
        bot = await platform.get_bot_user()
        if not bot.username or recipient.lower() != bot.username.lstrip("@").lower():
            return False
    if not msg.from_user or msg.from_user.is_self:
        return True

    async def reply(text: str) -> None:
        await platform.reply_to_message(msg, text)

    if command.lower() == "/balance" and msg.chat.type != "PRIVATE":
        await reply("Use /balance in a private chat with me to see your credits.")
        return True

    try:
        store = get_credit_store()
        if store is None:
            return False
        owner = Identity(msg.platform, normalize_chat_id(msg.platform, msg.from_user.id))
        if command.lower() == "/balance":
            if len(parts) != 1:
                await reply("Usage: /balance")
                return True
            balance = await store.balance(owner)
            await reply(f"Available credits: {balance.available}\nReserved credits: {balance.reserved}")
            return True

        if (
            len(parts) not in {2, 3}
            or parts[1] not in {"off", "only_me", "everyone"}
            or (len(parts) == 3 and parts[2] != "server")
        ):
            await reply("Usage: /paid_usage off|only_me|everyone [server]")
            return True
        server = len(parts) == 3
        if server and (msg.platform != "discord" or msg.chat.guild_id is None):
            await reply("The server option is available only inside a Discord server.")
            return True
        scope = Scope(
            msg.platform,
            "guild" if server else "chat",
            normalize_chat_id(msg.platform, msg.chat.guild_id if server else msg.chat.id),
        )
        mode = parts[1]
        await store.configure(owner, scope, mode)
        location = "this server" if server else "this chat"
        if mode == "off":
            text = f"Your paid usage is off for {location}."
        elif mode == "only_me":
            text = f"Your credits will cover only your own direct requests in {location}."
        else:
            text = (
                f"Your credits will cover everyone's direct requests in {location}. "
                "Other members can spend your balance. Use /paid_usage off"
                f"{' server' if server else ''} to stop."
            )
        await reply(text)
    except SponsorConflict:
        await reply("This chat or server already has a shared sponsor. Your setting was not changed.")
    except Exception:
        logger.exception("Usage command failed")
        await reply("I couldn't update or read your credits. Please try again later.")
    return True
