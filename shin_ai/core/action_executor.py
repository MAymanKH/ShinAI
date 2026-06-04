"""
Action Executor Module

Executes parsed AI response actions (reactions, stickers, text, moderation).
"""
import asyncio
import random

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.core.response_parser import ParsedResponse
from shin_ai.data.loader import (
    MEMBERS,
    TELEGRAM_STICKER_TO_DESCRIPTION,
    WHATSAPP_STICKER_TO_DESCRIPTION,
)
from shin_ai.services.replies import save_reply
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory import save_memory
from shin_ai.utils.context_manager import add_bot_message_to_context


def _normalize_reply_target_for_platform(
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


async def execute_response(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    parsed: ParsedResponse | list[ParsedResponse],
    default_target_id: int | str,
    original_prompt: str,
    raw_answer: str,
    reply_text: str = "",
) -> list[str]:
    # Normalize to list
    parsed_list = [parsed] if isinstance(parsed, ParsedResponse) else parsed
    
    # Save interaction to memory first (using all messages to capture actions)
    await _save_interaction_memory(
        platform=platform.platform_name,
        msg=msg,
        parsed_list=parsed_list,
        original_prompt=original_prompt,
        raw_answer=raw_answer,
        reply_text=reply_text,
    )
    
    # Execute each message in sequence
    mod_errors = []
    for idx, single_parsed in enumerate(parsed_list):
        # Add a human-like delay between messages (not before the first one)
        if idx > 0:
            delay = _human_inter_message_delay(single_parsed)
            logger.info("Inter-message delay: %.2fs", delay)
            # Send a typing indicator so the recipient sees "typing..."
            # throughout the gap between messages
            try:
                await platform.send_chat_action(msg.chat.id, "typing")
            except Exception:
                pass
            await asyncio.sleep(delay)
        
        # Resolve target: use AI-specified ID if present, otherwise default to sender
        if single_parsed.target_id:
            reply_to_id = single_parsed.target_id
        elif idx == 0:
            # First message defaults to replying to the sender
            reply_to_id = default_target_id
        else:
            # Subsequent messages without explicit target are sent without reply
            reply_to_id = None

        reply_to_id = _normalize_reply_target_for_platform(platform, reply_to_id)
        
        # Execute actions for this message
        await _execute_reaction(platform, msg, single_parsed.reaction)
        await _execute_sticker(platform, msg, single_parsed.sticker_id, reply_to_id)
        await _execute_text(platform, msg, single_parsed.text_content, reply_to_id)
        
        # Collect moderation errors
        mod_error = await _execute_mod_action(platform, msg, single_parsed)
        if mod_error:
            mod_errors.append(mod_error)
            
    return mod_errors


def _human_inter_message_delay(parsed: ParsedResponse) -> float:
    """Return a realistic inter-message delay based on content type and length.

    - Reactions: 1–4s (scroll through reactions, pick one).
    - Stickers without text: 3–10s (browse sticker packs, pick one).
    - Text: scaled to character count at ~2-3 chars/sec (phone typing),
      with Gaussian jitter so the cadence isn't perfectly linear.
    """
    # Reaction-only
    if parsed.reaction and not parsed.text_content and not parsed.sticker_id:
        return random.uniform(1.0, 4.0)

    # Sticker-only — browsing sticker packs
    if parsed.sticker_id and not parsed.text_content:
        return random.uniform(3.0, 10.0)

    # Text message — simulate typing time
    text = parsed.text_content or ""
    chars = len(text)
    if chars == 0:
        return random.uniform(0.5, 1.0)

    # ~2-3 chars/sec for phone typing
    base = chars / random.uniform(2, 3)
    jitter = random.gauss(0, 0.3)
    return max(0.5, min(base + jitter, 25.0))


async def _execute_reaction(platform: PlatformAdapter, msg: UnifiedMessage, reaction: str | None) -> None:
    if not reaction:
        return

    try:
        await platform.react(msg.chat.id, msg.id, reaction)
    except Exception as e:
        logger.error(f"Reaction failed on {platform.platform_name}: {e}")


async def _execute_sticker(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    sticker_id: str | None,
    reply_to_id: int | str | None,
) -> None:
    if not sticker_id:
        return

    if not platform.supports_stickers:
        logger.info(f"Platform {platform.platform_name} doesn't support stickers natively. Dropping.")
        return

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


async def _execute_text(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    text_content: str,
    reply_to_id: int | str | None,
) -> int | str | None:
    if not text_content:
        return None

    try:
        sent_id = await platform.send_message(msg.chat.id, text_content, reply_to_id)
        if sent_id:
            save_reply(msg.chat.id, sent_id, platform.platform_name)
            await _record_outgoing_context(
                platform=platform,
                msg=msg,
                sent_id=sent_id,
                text_content=text_content,
                reply_to_id=reply_to_id,
            )
        return sent_id
    except Exception as e:
        logger.error(f"Text reply failed: {e}")
        return None


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


def _resolve_name_to_username(name: str, platform_name: str = "") -> str | None:
    """Resolve a display name / preferred name to the correct platform username."""
    from shin_ai.services.social import resolve_username_to_key, get_platform_username_for_member
    
    name_clean = name.lower().strip().replace("@", "")
    
    # Try resolving directly via platform-aware lookup
    member_key = resolve_username_to_key(name_clean, platform_name)
    if member_key:
        # Return the username for the current platform
        platform_uname = get_platform_username_for_member(member_key, platform_name)
        if platform_uname:
            return platform_uname
        # Fallback: try the dict key itself
        if not member_key.startswith("!") and not member_key[0].isdigit():
            return member_key
    
    # Legacy fallback: search preferred_name
    for key, data in MEMBERS.items():
        for pname in data.get("preferred_name", "").split(","):
            if pname.strip().lower() == name_clean:
                platform_uname = get_platform_username_for_member(key, platform_name)
                if platform_uname:
                    return platform_uname
                if not key.startswith("!") and not key[0].isdigit():
                    return key
    return None


async def _execute_mod_action(platform: PlatformAdapter, msg: UnifiedMessage, parsed: ParsedResponse) -> str | None:
    if not parsed.mod_action:
        return None

    action = parsed.mod_action
    
    if action in ("unban", "add"):
        target = await _resolve_mod_target(platform, msg, parsed.mod_target_username, None)
        if not target:
            return f"{action.upper()} FAILED: Could not find the user."

        try:
            if action == "unban":
                await platform.unban_chat_member(msg.chat.id, target.id)
            else:
                link = await platform.create_chat_invite_link(msg.chat.id)
                if link:
                    await platform.send_message(target.id, f"You've been invited: {link}")
        except Exception as e:
            return f"{action.upper()} FAILED: {e}"

        return None

    target = await _resolve_mod_target(platform, msg, parsed.mod_target_username, parsed.target_id)
    if not target:
        return f"{action.upper()} FAILED: Could not determine who to {action}."
        
    try:
        status = await platform.get_chat_member_status(msg.chat.id, target.id)
        if status in ["ADMINISTRATOR", "OWNER"]:
            return f"{action.upper()} FAILED: Target is an admin/owner."
            
        if action == "kick":
            await platform.kick_chat_member(msg.chat.id, target.id)
        elif action == "ban":
            await platform.ban_chat_member(msg.chat.id, target.id)
        elif action == "mute":
            if not getattr(platform, "supports_member_restrictions", True):
                return f"{action.upper()} FAILED: Platform {platform.platform_name} does not support per-user mute/unmute."
            await platform.restrict_chat_member(msg.chat.id, target.id, False)
        elif action == "unmute":
            if not getattr(platform, "supports_member_restrictions", True):
                return f"{action.upper()} FAILED: Platform {platform.platform_name} does not support per-user mute/unmute."
            await platform.restrict_chat_member(msg.chat.id, target.id, True)

        return None
    except Exception as e:
        return f"{action.upper()} FAILED: {e}"


async def _resolve_mod_target(platform: PlatformAdapter, msg: UnifiedMessage, ai_specified_username: str | None, target_id: int | str | None):
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

    if target_id:
        try:
            t_msg = await platform.get_message(msg.chat.id, target_id)
            if t_msg and t_msg.from_user and not t_msg.from_user.is_self:
                return t_msg.from_user
        except Exception:
            pass

    for ent in msg.entities + msg.caption_entities:
        if ent.type == "MENTION" or ent.type == "TEXT_MENTION":
            if ent.user and not ent.user.is_self:
                return ent.user
                
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if not msg.reply_to_message.from_user.is_self:
            return msg.reply_to_message.from_user
            
    if msg.from_user:
        return msg.from_user
    return None

async def _save_interaction_memory(platform: str, msg: UnifiedMessage, parsed_list: list[ParsedResponse], original_prompt: str, raw_answer: str, reply_text: str) -> None:
    if not raw_answer:
        return
        
    try:
        short_context = ""
        if reply_text:
            if "reply to a conversation chain" in reply_text:
                short_context = reply_text.split(":\n", 1)[-1].replace("\n- ", " > ").replace("\n", " ").strip()
            else:
                short_context = reply_text.strip()
                
        mem_parts = []
        for parsed in parsed_list:
            if parsed.text_content:
                mem_parts.append(parsed.text_content)
            if parsed.reaction:
                mem_parts.append(f"[Reacted: {parsed.reaction}]")
            if parsed.sticker_id:
                if platform == "whatsapp":
                    sticker_desc = WHATSAPP_STICKER_TO_DESCRIPTION.get(parsed.sticker_id, "Unknown Sticker")
                else:
                    sticker_desc = TELEGRAM_STICKER_TO_DESCRIPTION.get(parsed.sticker_id, "Unknown Sticker")
                mem_parts.append(f"[Sent Sticker: {sticker_desc}]")
            if parsed.mod_action:
                target = parsed.mod_target_username or "the reply target"
                mem_parts.append(f"[Action: {parsed.mod_action} on {target}]")
                
        final_memory = " ".join(mem_parts) if mem_parts else raw_answer

        await save_memory(
            platform=platform,
            user_id=msg.from_user.id, 
            username=msg.from_user.username, 
            prompt=original_prompt, 
            response=final_memory, 
            context=short_context,
            chat_id=msg.chat.id,
            chat_title=msg.chat.title or "Private Chat"
        )
    except Exception as e:
        logger.error(f"Failed to save long-term memory: {e}")
