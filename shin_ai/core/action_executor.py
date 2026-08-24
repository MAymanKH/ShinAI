"""
Action Executor Module

Executes AI-triggered actions (reactions, stickers, text messages, moderation).
Actions arrive as:
  - A plain text string (split on '---' in handler.py for multi-message support)
  - A list of pending_action dicts queued by the tool-calling loop
    (send_reaction / send_sticker / moderate_user tool calls)
"""

import asyncio
import random
from dataclasses import dataclass

from shin_ai.data.loader import (
    MEMBERS,
    TELEGRAM_STICKER_TO_DESCRIPTION,
    WHATSAPP_STICKER_TO_DESCRIPTION,
)
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.services.replies import save_reply
from shin_ai.settings import get_settings
from shin_ai.utils.context_manager import add_bot_message_to_context
from shin_ai.utils.logger_config import logger

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionExecutionResult:
    errors: list[str]
    completed_actions: list[dict]


async def execute_text_messages(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    messages: list,
    default_reply_to_id: int | str,
) -> list[str]:
    """Send the plain-text messages produced by the AI (split from '---').

    `messages` is a list of (text, tag_target) pairs as returned by
    `response_policy.split_reply_messages` (older callers may still pass bare
    strings — those are treated as having no tag).

    Per-message reply targeting: any message may have been decorated by the
    model with a leading `[REPLY_TO:message_id]` tag to reply to ANY message
    from the chat history — including the very first message of the response.

    Resolution precedence per message:
      1. An explicit tag on that message (if valid for the platform).
      2. Index 0 with no tag -> the triggering message (default_reply_to_id).
      3. Index > 0 with no tag -> no explicit reply target (natural
         follow-on message).

    WhatsApp's untagged first output uses the freshly ingested native event;
    explicit targets are resolved through the adapter's raw-message cache.
    """
    # Normalize to (text, tag_target) pairs; bare strings from older call
    # sites are treated as (text, None).
    pairs: list[tuple[str, str | None]] = [(m, None) if isinstance(m, str) else m for m in messages]

    sent_messages: list[str] = []

    for idx, (text, tag_target) in enumerate(pairs):
        if not text:
            continue

        if idx > 0:
            delay = _human_inter_message_delay(text)
            logger.debug(
                "Inter-message delay: %.2fs",
                delay,
                extra={"event_name": "response.delay"},
            )
            try:
                await platform.send_chat_action(msg.chat.id, "typing")
            except Exception as error:
                # Cosmetic only; never let a presence hiccup drop the message.
                logger.debug("Could not refresh typing indicator: %s", error)
            await asyncio.sleep(delay)

        reply_to_id = None
        if tag_target is not None:
            candidate = tag_target
            if platform.uses_integer_message_ids and isinstance(candidate, str) and candidate.isdigit():
                candidate = int(candidate)
            reply_to_id = _normalize_reply_target(platform, candidate)
            if reply_to_id is None:
                logger.warning(
                    "[%s] Ignoring invalid [REPLY_TO] target %r for chat=%s",
                    platform.platform_name,
                    tag_target,
                    msg.chat.id,
                )
        elif idx == 0:
            reply_to_id = _normalize_reply_target(platform, default_reply_to_id)

        try:
            if platform.prefers_native_reply and idx == 0 and tag_target is None:
                sent_id = await platform.reply_to_message(msg, text)
            else:
                sent_id = await platform.send_message(msg.chat.id, text, reply_to_id)
            if sent_id:
                sent_messages.append(text)
                preview = text.replace("\n", " ")[: get_settings().logging.content_preview_chars]
                logger.info(
                    'Responded — part=%d/%d text="%s%s"',
                    idx + 1,
                    len(pairs),
                    preview if get_settings().logging.content_preview_chars else "<hidden>",
                    "..."
                    if get_settings().logging.content_preview_chars
                    and len(text) > get_settings().logging.content_preview_chars
                    else "",
                    extra={"event_name": "response.sent"},
                )
                logger.debug(
                    "Delivery details — sent_id=%s reply_to=%s",
                    sent_id,
                    reply_to_id or "none",
                    extra={"event_name": "response.delivery"},
                )
                if tag_target is not None and reply_to_id is not None:
                    logger.info(
                        "[%s] Reply-overrode target — chat=%s sent=%s -> reply_to=%s (model-chosen)",
                        platform.platform_name,
                        msg.chat.id,
                        sent_id,
                        reply_to_id,
                    )
                await save_reply(
                    msg.chat.id,
                    sent_id,
                    platform.platform_name,
                    coordination_scope=platform.coordination_scope,
                )
                await _record_outgoing_context(
                    platform=platform,
                    msg=msg,
                    sent_id=sent_id,
                    text_content=text,
                    reply_to_id=reply_to_id,
                )
        except Exception as e:
            logger.error("Text reply failed: %s", e)
    return sent_messages


async def execute_pending_actions(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    pending_actions: list[dict],
    default_reply_to_id: int | str,
) -> ActionExecutionResult:
    """Execute action dicts queued during the AI's tool-calling loop.

    Each dict has a 'type' key: 'reaction', 'sticker', or 'moderation'.
    Reports both failed moderation actions and actions that actually completed.
    """
    if not pending_actions:
        return ActionExecutionResult([], [])

    mod_errors: list[str] = []
    completed_actions: list[dict] = []

    for action in pending_actions:
        action_type = action.get("type")

        if action_type == "reaction":
            if await _execute_reaction(platform, msg, action):
                completed_actions.append(action)

        elif action_type == "sticker":
            if await _execute_sticker(platform, msg, action, default_reply_to_id):
                completed_actions.append(action)

        elif action_type == "moderation":
            error = await _execute_mod_action(platform, msg, action)
            if error:
                mod_errors.append(error)
            else:
                completed_actions.append(action)

        else:
            logger.warning("Unknown pending action type: %r", action_type)

    return ActionExecutionResult(mod_errors, completed_actions)


# ---------------------------------------------------------------------------
# Internal helpers — reactions
# ---------------------------------------------------------------------------


async def _execute_reaction(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    action: dict,
) -> bool:
    emoji = action.get("emoji", "")
    if not emoji:
        return False

    # Resolve the target message ID: use the tool-specified one if present,
    # otherwise fall back to the triggering message.
    raw_id = action.get("message_id")
    if raw_id is not None:
        message_id = int(raw_id) if str(raw_id).isdigit() else raw_id
    else:
        message_id = msg.id

    try:
        await platform.react(msg.chat.id, message_id, emoji)
        logger.info(
            "Reaction sent — target=%s emoji=%s",
            message_id,
            emoji,
            extra={"event_name": "action.reaction"},
        )
        return True
    except Exception as e:
        logger.error("Reaction failed on %s: %s", platform.platform_name, e)
        return False


# ---------------------------------------------------------------------------
# Internal helpers — stickers
# ---------------------------------------------------------------------------


async def _execute_sticker(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    action: dict,
    default_reply_to_id: int | str,
) -> bool:
    sticker_id = action.get("sticker_id", "")
    if not sticker_id:
        return False

    if not platform.supports_stickers:
        logger.info("Platform %s doesn't support stickers. Dropping.", platform.platform_name)
        return False

    # Some adapters expect a prefixed identifier; normalise if the model omitted it.
    prefix = platform.sticker_id_prefix
    if prefix and not sticker_id.lower().startswith(prefix.lower()):
        sticker_id = f"{prefix}{sticker_id}"

    raw_reply = action.get("reply_to_message_id")
    reply_to_id = _normalize_reply_target(
        platform,
        (int(raw_reply) if str(raw_reply).isdigit() else raw_reply) if raw_reply else default_reply_to_id,
    )

    try:
        sent_id = await platform.send_sticker(msg.chat.id, sticker_id, reply_to_id)
        if sent_id:
            logger.info(
                "Sticker sent — sent_id=%s reply_to=%s",
                sent_id,
                reply_to_id or "none",
                extra={"event_name": "action.sticker"},
            )
            await save_reply(
                msg.chat.id,
                sent_id,
                platform.platform_name,
                coordination_scope=platform.coordination_scope,
            )
            await _record_outgoing_context(
                platform=platform,
                msg=msg,
                sent_id=sent_id,
                text_content=None,
                reply_to_id=reply_to_id,
                media_type="sticker",
            )
            return True
    except Exception as e:
        logger.error("Sticker failed: %s", e)
    return False


# ---------------------------------------------------------------------------
# Internal helpers — moderation
# ---------------------------------------------------------------------------


async def _execute_mod_action(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    action: dict,
) -> str | None:
    mod_action = action.get("action", "")
    if not mod_action:
        return "MODERATION FAILED: No action was specified."
    if mod_action not in {"unban", "add", "kick", "ban", "mute", "unmute"}:
        return f"MODERATION FAILED: Unsupported action '{mod_action}'."

    target_username = action.get("target_username")
    target_message_id = action.get("target_message_id")

    if mod_action in ("unban", "add"):
        target = await _resolve_mod_target(platform, msg, target_username, None)
        if not target:
            return f"{mod_action.upper()} FAILED: Could not find the user."

        try:
            if mod_action == "unban":
                await platform.unban_chat_member(msg.chat.id, target.id)
            else:  # add
                link = await platform.create_chat_invite_link(msg.chat.id)
                if not link:
                    return "ADD FAILED: The platform did not create an invite link."
                await platform.send_message(target.id, f"You've been invited: {link}")
        except Exception as e:
            return f"{mod_action.upper()} FAILED: {e}"

        logger.info(
            "Moderation action completed — action=%s target=%s",
            mod_action,
            target.id,
            extra={"event_name": "action.moderation"},
        )
        return None

    target = await _resolve_mod_target(platform, msg, target_username, target_message_id)
    if not target:
        return f"{mod_action.upper()} FAILED: Could not determine who to {mod_action}."

    try:
        status = await platform.get_chat_member_status(msg.chat.id, target.id)
        if status in ("ADMINISTRATOR", "OWNER"):
            return f"{mod_action.upper()} FAILED: Target is an admin/owner."

        if mod_action == "kick":
            await platform.kick_chat_member(msg.chat.id, target.id)
        elif mod_action == "ban":
            await platform.ban_chat_member(msg.chat.id, target.id)
        elif mod_action == "mute":
            if not platform.supports_member_restrictions:
                return f"MUTE FAILED: Platform {platform.platform_name} does not support per-user mute."
            await platform.restrict_chat_member(msg.chat.id, target.id, False)
        elif mod_action == "unmute":
            if not platform.supports_member_restrictions:
                return f"UNMUTE FAILED: Platform {platform.platform_name} does not support per-user unmute."
            await platform.restrict_chat_member(msg.chat.id, target.id, True)

        logger.info(
            "Moderation action completed — action=%s target=%s",
            mod_action,
            target.id,
            extra={"event_name": "action.moderation"},
        )
        return None
    except Exception as e:
        return f"{mod_action.upper()} FAILED: {e}"


async def _resolve_mod_target(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    ai_specified_username: str | None,
    target_message_id: str | None,
):
    if ai_specified_username:
        clean = ai_specified_username.replace("@", "")
        user = await platform.get_user_by_username(clean)
        if user:
            return user

        resolved_username = _resolve_name_to_username(clean, platform.platform_name)
        if resolved_username:
            user = await platform.get_user_by_username(resolved_username)
            if user:
                return user

    if target_message_id:
        try:
            t_msg = await platform.get_message(msg.chat.id, target_message_id)
            if t_msg and t_msg.from_user and not t_msg.from_user.is_self:
                return t_msg.from_user
        except Exception as error:
            logger.debug(
                "Could not resolve moderation target from message %s: %s",
                target_message_id,
                error,
            )

    for ent in msg.entities + msg.caption_entities:
        if ent.type in ("MENTION", "TEXT_MENTION"):
            if ent.user and not ent.user.is_self:
                return ent.user

    if msg.reply_to_message and msg.reply_to_message.from_user:
        if not msg.reply_to_message.from_user.is_self:
            return msg.reply_to_message.from_user

    if msg.from_user:
        return msg.from_user
    return None


def _resolve_name_to_username(name: str, platform_name: str = "") -> str | None:
    """Resolve a display name / preferred name to the correct platform username."""
    from shin_ai.services.social import get_platform_username_for_member, resolve_username_to_key

    name_clean = name.lower().strip().replace("@", "")

    member_key = resolve_username_to_key(name_clean, platform_name)
    if member_key:
        platform_uname = get_platform_username_for_member(member_key, platform_name)
        if platform_uname:
            return platform_uname
        if not member_key.startswith("!") and not member_key[0].isdigit():
            return member_key

    for key, data in MEMBERS.items():
        for pname in data.get("preferred_name", "").split(","):
            if pname.strip().lower() == name_clean:
                platform_uname = get_platform_username_for_member(key, platform_name)
                if platform_uname:
                    return platform_uname
                if not key.startswith("!") and not key[0].isdigit():
                    return key
    return None


# ---------------------------------------------------------------------------
# Internal helpers — context / memory
# ---------------------------------------------------------------------------


async def _record_outgoing_context(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    sent_id: int | str,
    text_content: str | None,
    reply_to_id: int | str | None,
    media_type: str | None = None,
) -> None:
    try:
        bot_user = await platform.get_bot_user()
        reply_to_user = None
        if reply_to_id and msg.from_user and str(reply_to_id) == str(msg.id):
            reply_to_user = msg.from_user.first_name
        add_bot_message_to_context(
            platform=platform.platform_name,
            chat_id=msg.chat.id,
            msg_id=sent_id,
            text=text_content,
            bot_user=bot_user,
            reply_to_id=reply_to_id,
            reply_to_user=reply_to_user,
            media_type=media_type,
        )
    except Exception as e:
        logger.debug("Failed to record outgoing context: %s", e)


async def save_interaction_memory(
    platform: str,
    msg: UnifiedMessage,
    messages: list[str],
    completed_actions: list[dict],
    original_prompt: str,
    reply_text: str,
    memory_saver=None,
) -> None:
    if not messages and not completed_actions:
        return

    # Anonymous / channel-posted senders have no from_user. Reading through it
    # raised AttributeError inside the broad handler below, so the interaction
    # was silently dropped and reported as a save failure.
    if msg.from_user is None:
        logger.debug(
            "Not saving memory: message has no sender (chat=%s msg=%s)",
            msg.chat.id,
            msg.id,
        )
        return

    try:
        if memory_saver is None:
            from shin_ai.utils.memory import save_memory as memory_saver

        short_context = ""
        if reply_text:
            if "reply to a conversation chain" in reply_text:
                short_context = (
                    reply_text.split(":\n", 1)[-1].replace("\n- ", " > ").replace("\n", " ").strip()
                )
            else:
                short_context = reply_text.strip()

        mem_parts = list(messages)  # text messages first

        for action in completed_actions:
            action_type = action.get("type")
            if action_type == "reaction":
                mem_parts.append(f"[Reacted: {action.get('emoji', '')}]")
            elif action_type == "sticker":
                sid = action.get("sticker_id", "")
                if platform == "whatsapp":
                    desc = WHATSAPP_STICKER_TO_DESCRIPTION.get(sid, "Unknown Sticker")
                else:
                    desc = TELEGRAM_STICKER_TO_DESCRIPTION.get(sid, "Unknown Sticker")
                mem_parts.append(f"[Sent Sticker: {desc}]")
            elif action_type == "moderation":
                target = action.get("target_username") or "the reply target"
                mem_parts.append(f"[Action: {action.get('action', '')} on {target}]")

        final_memory = " ".join(part for part in mem_parts if part)

        await memory_saver(
            platform=platform,
            user_id=msg.from_user.id,
            username=msg.from_user.username,
            prompt=original_prompt,
            response=final_memory,
            context=short_context,
            chat_id=msg.chat.id,
            chat_title=msg.chat.title or "Private Chat",
        )
    except Exception as e:
        logger.error("Failed to save long-term memory: %s", e)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _normalize_reply_target(
    platform: PlatformAdapter,
    reply_to_id: int | str | None,
) -> int | str | None:
    if reply_to_id is None:
        return None

    if platform.uses_integer_message_ids:
        if isinstance(reply_to_id, int):
            return reply_to_id
        if isinstance(reply_to_id, str) and reply_to_id.isdigit():
            return int(reply_to_id)
        logger.warning(
            "Ignoring non-numeric target id '%s' for platform %s",
            reply_to_id,
            platform.platform_name,
        )
        return None

    return reply_to_id


def _human_inter_message_delay(text: str) -> float:
    """Return a realistic inter-message delay based on text length.

    ~8–9 chars/sec for phone typing, with Gaussian jitter.
    """
    chars = len(text)
    if chars == 0:
        return random.uniform(0.5, 1.0)
    base = chars / random.uniform(8, 9)
    jitter = random.gauss(0, 0.3)
    return max(0.5, min(base + jitter, 25.0))
