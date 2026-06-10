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

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.data.loader import (
    MEMBERS,
    TELEGRAM_STICKER_TO_DESCRIPTION,
    WHATSAPP_STICKER_TO_DESCRIPTION,
)
from shin_ai.services.replies import save_reply
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory import save_memory
from shin_ai.utils.context_manager import add_bot_message_to_context


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def execute_text_messages(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    messages: list[str],
    default_reply_to_id: int | str,
    original_prompt: str,
    raw_answer: str,
    reply_text: str = "",
) -> None:
    """Send the plain-text messages produced by the AI (split from '---').

    The first message replies to default_reply_to_id; subsequent messages
    are sent without an explicit reply target (natural follow-on messages).
    """
    await _save_interaction_memory(
        platform=platform.platform_name,
        msg=msg,
        messages=messages,
        pending_actions=[],
        original_prompt=original_prompt,
        raw_answer=raw_answer,
        reply_text=reply_text,
    )

    for idx, text in enumerate(messages):
        if not text:
            continue

        if idx > 0:
            delay = _human_inter_message_delay(text)
            logger.info("Inter-message delay: %.2fs", delay)
            try:
                await platform.send_chat_action(msg.chat.id, "typing")
            except Exception:
                pass
            await asyncio.sleep(delay)

        reply_to_id = _normalize_reply_target(platform, default_reply_to_id if idx == 0 else None)

        try:
            sent_id = await platform.send_message(msg.chat.id, text, reply_to_id)
            if sent_id:
                save_reply(msg.chat.id, sent_id, platform.platform_name)
                await _record_outgoing_context(
                    platform=platform,
                    msg=msg,
                    sent_id=sent_id,
                    text_content=text,
                    reply_to_id=reply_to_id,
                )
        except Exception as e:
            logger.error(f"Text reply failed: {e}")


async def execute_pending_actions(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    pending_actions: list[dict],
    default_reply_to_id: int | str,
    original_prompt: str,
    raw_answer: str,
    reply_text: str = "",
) -> list[str]:
    """Execute action dicts queued during the AI's tool-calling loop.

    Each dict has a 'type' key: 'reaction', 'sticker', or 'moderation'.
    Returns a list of error strings for any failed moderation actions so the
    caller can pass them back to the AI for a natural error response.
    """
    if not pending_actions:
        return []

    await _save_interaction_memory(
        platform=platform.platform_name,
        msg=msg,
        messages=[],
        pending_actions=pending_actions,
        original_prompt=original_prompt,
        raw_answer=raw_answer,
        reply_text=reply_text,
    )

    mod_errors: list[str] = []

    for action in pending_actions:
        action_type = action.get("type")

        if action_type == "reaction":
            await _execute_reaction(platform, msg, action)

        elif action_type == "sticker":
            await _execute_sticker(platform, msg, action, default_reply_to_id)

        elif action_type == "moderation":
            error = await _execute_mod_action(platform, msg, action)
            if error:
                mod_errors.append(error)

        else:
            logger.warning("Unknown pending action type: %r", action_type)

    return mod_errors


# ---------------------------------------------------------------------------
# Internal helpers — reactions
# ---------------------------------------------------------------------------

async def _execute_reaction(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    action: dict,
) -> None:
    emoji = action.get("emoji", "")
    if not emoji:
        return

    # Resolve the target message ID: use the tool-specified one if present,
    # otherwise fall back to the triggering message.
    raw_id = action.get("message_id")
    if raw_id is not None:
        message_id = int(raw_id) if str(raw_id).isdigit() else raw_id
    else:
        message_id = msg.id

    try:
        await platform.react(msg.chat.id, message_id, emoji)
    except Exception as e:
        logger.error(f"Reaction failed on {platform.platform_name}: {e}")


# ---------------------------------------------------------------------------
# Internal helpers — stickers
# ---------------------------------------------------------------------------

async def _execute_sticker(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    action: dict,
    default_reply_to_id: int | str,
) -> None:
    sticker_id = action.get("sticker_id", "")
    if not sticker_id:
        return

    if not platform.supports_stickers:
        logger.info(f"Platform {platform.platform_name} doesn't support stickers. Dropping.")
        return

    # WhatsApp expects 'wa:<filename>' prefix; normalise if the model omitted it.
    if platform.platform_name == "whatsapp" and not sticker_id.lower().startswith("wa:"):
        sticker_id = f"wa:{sticker_id}"

    raw_reply = action.get("reply_to_message_id")
    reply_to_id = _normalize_reply_target(
        platform,
        (int(raw_reply) if str(raw_reply).isdigit() else raw_reply) if raw_reply else default_reply_to_id,
    )

    try:
        sent_id = await platform.send_sticker(msg.chat.id, sticker_id, reply_to_id)
        if sent_id:
            save_reply(msg.chat.id, sent_id, platform.platform_name)
            await _record_outgoing_context(
                platform=platform,
                msg=msg,
                sent_id=sent_id,
                text_content=None,
                reply_to_id=reply_to_id,
                media_type="sticker",
            )
    except Exception as e:
        logger.error(f"Sticker failed: {e}")


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
        return None

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
                if link:
                    await platform.send_message(target.id, f"You've been invited: {link}")
        except Exception as e:
            return f"{mod_action.upper()} FAILED: {e}"

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
            if not getattr(platform, "supports_member_restrictions", True):
                return f"MUTE FAILED: Platform {platform.platform_name} does not support per-user mute."
            await platform.restrict_chat_member(msg.chat.id, target.id, False)
        elif mod_action == "unmute":
            if not getattr(platform, "supports_member_restrictions", True):
                return f"UNMUTE FAILED: Platform {platform.platform_name} does not support per-user unmute."
            await platform.restrict_chat_member(msg.chat.id, target.id, True)

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
        except Exception:
            pass

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
    from shin_ai.services.social import resolve_username_to_key, get_platform_username_for_member

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
        logger.debug(f"Failed to record outgoing context: {e}")


async def _save_interaction_memory(
    platform: str,
    msg: UnifiedMessage,
    messages: list[str],
    pending_actions: list[dict],
    original_prompt: str,
    raw_answer: str,
    reply_text: str,
) -> None:
    if not raw_answer:
        return

    try:
        short_context = ""
        if reply_text:
            if "reply to a conversation chain" in reply_text:
                short_context = reply_text.split(":\n", 1)[-1].replace("\n- ", " > ").replace("\n", " ").strip()
            else:
                short_context = reply_text.strip()

        mem_parts = list(messages)  # text messages first

        for action in pending_actions:
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

        final_memory = " ".join(p for p in mem_parts if p) if mem_parts else raw_answer

        await save_memory(
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
        logger.error(f"Failed to save long-term memory: {e}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalize_reply_target(
    platform: PlatformAdapter,
    reply_to_id: int | str | None,
) -> int | str | None:
    if reply_to_id is None:
        return None

    if platform.platform_name in {"telegram", "discord"}:
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
