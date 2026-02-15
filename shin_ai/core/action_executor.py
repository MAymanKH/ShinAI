"""
Action Executor Module

Executes parsed AI response actions (reactions, stickers, text, moderation).
"""
import asyncio
from pyrogram import Client, enums
from pyrogram.types import Message

from shin_ai.core.response_parser import ParsedResponse
from shin_ai.data.loader import STICKER_TO_DESCRIPTION
from shin_ai.services.replies import save_reply
from shin_ai.utils.context_manager import add_message_to_context
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory import save_memory


async def execute_response(
    client: Client,
    msg: Message,
    parsed: ParsedResponse | list[ParsedResponse],
    default_target_id: int,
    original_prompt: str,
    raw_answer: str,
    reply_text: str = "",
) -> None:
    """
    Execute all actions from a parsed AI response.
    
    Args:
        client: Pyrogram client instance
        msg: Original message that triggered the response
        parsed: Single ParsedResponse or list of ParsedResponse objects
        default_target_id: The message ID to reply to by default (the sender's message)
        original_prompt: The original user prompt
        raw_answer: The raw AI response (for memory saving)
        reply_text: Reply chain context for memory
    """
    # Normalize to list
    parsed_list = [parsed] if isinstance(parsed, ParsedResponse) else parsed
    
    # Save interaction to memory first (using the first message)
    await _save_interaction_memory(
        msg=msg,
        parsed=parsed_list[0] if parsed_list else ParsedResponse(),
        original_prompt=original_prompt,
        raw_answer=raw_answer,
        reply_text=reply_text,
    )
    
    # Execute each message in sequence
    for idx, single_parsed in enumerate(parsed_list):
        # Add delay between messages (not before the first one)
        if idx > 0:
            await asyncio.sleep(1.5)  # 1.5 second delay between messages
        
        # Resolve target: use AI-specified ID if present, otherwise default to sender
        if single_parsed.target_id:
            reply_to_id = single_parsed.target_id
        elif idx == 0:
            # First message defaults to replying to the sender
            reply_to_id = default_target_id
        else:
            # Subsequent messages without explicit target are sent without reply
            reply_to_id = None
        
        # Execute actions for this message
        await _execute_reaction(msg, single_parsed.reaction)
        await _execute_sticker(client, msg, single_parsed.sticker_id, reply_to_id)
        await _execute_text(client, msg, single_parsed.text_content, reply_to_id)
        await _execute_mod_action(client, msg, single_parsed)


async def _execute_reaction(msg: Message, reaction: str | None) -> None:
    """Send a reaction emoji to the message."""
    if not reaction:
        return
    
    try:
        # Reactions always go to the trigger message
        await msg.react(reaction)
    except Exception as e:
        logger.error(f"Reaction failed: {e}")


async def _execute_sticker(
    client: Client, 
    msg: Message, 
    sticker_id: str | None, 
    reply_to_id: int | None
) -> None:
    """Send a sticker as a reply."""
    if not sticker_id:
        return
    
    try:
        if reply_to_id:
            sent_sticker = await client.send_sticker(
                msg.chat.id, 
                sticker_id, 
                reply_to_message_id=reply_to_id
            )
        else:
            sent_sticker = await client.send_sticker(
                msg.chat.id, 
                sticker_id
            )
        save_reply(msg.chat.id, sent_sticker.id)
        add_message_to_context(sent_sticker)
    except Exception as e:
        logger.error(f"Sticker failed: {e}")


async def _execute_text(
    client: Client, 
    msg: Message, 
    text_content: str, 
    reply_to_id: int | None
) -> int | None:
    """
    Send a text message as a reply.
    
    Returns:
        The message ID of the sent message, or None if failed
    """
    if not text_content:
        return None
    
    try:
        if reply_to_id:
            sent_message = await client.send_message(
                msg.chat.id, 
                text_content, 
                reply_to_message_id=reply_to_id
            )
        else:
            sent_message = await client.send_message(
                msg.chat.id, 
                text_content
            )
        save_reply(msg.chat.id, sent_message.id)
        add_message_to_context(sent_message)
        return sent_message.id
    except Exception as e:
        logger.error(f"Text reply failed: {e}")
        return None


# ===========================================
# Moderation Actions
# ===========================================

_MOD_ACTION_LABELS = {
    "kick": ("نطرت", "أنطر"),
    "ban": ("بانيت", "أبان"),
    "unban": ("شلت البان عن", "أشيل البان عن"),
    "mute": ("سكّتت", "أسكّت"),
    "unmute": ("فكيت الميوت عن", "أفك الميوت عن"),
}


async def _execute_mod_action(
    client: Client, 
    msg: Message, 
    parsed: ParsedResponse,
) -> None:
    """Execute a moderation action (kick, ban, unban, mute, unmute) if requested."""
    if not parsed.mod_action:
        return
    
    action = parsed.mod_action
    success_label, fail_label = _MOD_ACTION_LABELS.get(action, (action, action))
    
    # Unban doesn't need target resolution from context - it needs a username
    if action == "unban":
        target = await _resolve_mod_target_simple(client, msg, parsed.mod_target_username)
        if not target:
            logger.warning(f"Unban action requested but no target could be resolved")
            await msg.reply_text(f"مين أشيل البان عنه؟")
            return
        try:
            await client.unban_chat_member(msg.chat.id, target.id)
            await msg.reply_text(f"{success_label} {target.first_name}")
        except Exception as e:
            logger.error(f"Unban failed: {e}")
            await msg.reply_text(f"ما عرفت {fail_label} {target.first_name}")
        return
    
    # For kick/ban/mute/unmute - resolve the target from context
    target = await _resolve_mod_target(
        client, msg, parsed.mod_target_username, parsed.target_id
    )
    
    if not target:
        logger.warning(f"{action} action requested but no target could be resolved")
        return
    
    try:
        # Check status before acting - can't act on admins/owners
        chat_member = await client.get_chat_member(msg.chat.id, target.id)
        if chat_member.status in [enums.ChatMemberStatus.ADMINISTRATOR, enums.ChatMemberStatus.OWNER]:
            await msg.reply_text(f"ما أقدر أسوي كذا على أدمن")
            return
        
        if action == "kick":
            await client.ban_chat_member(msg.chat.id, target.id)
            await client.unban_chat_member(msg.chat.id, target.id)
        elif action == "ban":
            await client.ban_chat_member(msg.chat.id, target.id)
        elif action == "mute":
            from pyrogram.types import ChatPermissions
            await client.restrict_chat_member(
                msg.chat.id, target.id,
                ChatPermissions()  # All permissions False = fully muted
            )
        elif action == "unmute":
            from pyrogram.types import ChatPermissions
            await client.restrict_chat_member(
                msg.chat.id, target.id,
                ChatPermissions(
                    can_send_messages=True,
                    can_send_media_messages=True,
                    can_send_other_messages=True,
                    can_add_web_page_previews=True,
                )
            )
        
        await msg.reply_text(f"{success_label} {target.first_name}")
        
    except Exception as e:
        logger.error(f"{action} failed: {e}")
        await msg.reply_text(f"ما عرفت {fail_label} {target.first_name}")


async def _resolve_mod_target_simple(
    client: Client,
    msg: Message, 
    ai_specified_username: str | None,
):
    """Resolve target from username only (used for unban where user isn't in chat)."""
    if ai_specified_username:
        try:
            clean_username = ai_specified_username.replace("@", "")
            return await client.get_users(clean_username)
        except Exception as e:
            logger.error(f"Failed to resolve username {ai_specified_username}: {e}")
    
    # Check mentions in the user's message
    entities_to_check = []
    if msg.entities:
        entities_to_check.extend([(e, msg.text) for e in msg.entities])
    if msg.caption_entities:
        entities_to_check.extend([(e, msg.caption) for e in msg.caption_entities])

    for entity, source_text in entities_to_check:
        try:
            candidate = None
            if entity.type == enums.MessageEntityType.MENTION:
                username = source_text[entity.offset:entity.offset + entity.length]
                candidate = await client.get_users(username)
            elif entity.type == enums.MessageEntityType.TEXT_MENTION:
                candidate = entity.user
            if candidate and not candidate.is_self:
                return candidate
        except Exception as e:
            logger.error(f"Error resolving mention: {e}")
    
    return None


async def _resolve_mod_target(
    client: Client, 
    msg: Message, 
    ai_specified_username: str | None,
    target_id: int | None,
):
    """
    Resolve the target user for a moderation action.
    
    Priority:
    1. AI-specified username (action:kick:@user)
    2. AI-specified target message ID (target:<id>) - fetch message, get author
    3. Mentions in the user's message
    4. Reply target of the trigger message (skip if it's the bot itself)
    5. Fallback to the sender (self-defense)
    """
    # 1. AI Specified Username
    if ai_specified_username:
        try:
            clean_username = ai_specified_username.replace("@", "")
            target = await client.get_users(clean_username)
            if target:
                return target
        except Exception as e:
            logger.error(f"Failed to resolve mod target {ai_specified_username}: {e}")

    # 2. AI Specified Target Message ID - fetch the message and get its author
    if target_id:
        try:
            t_msg = await client.get_messages(msg.chat.id, target_id)
            if t_msg and t_msg.from_user and not t_msg.from_user.is_self:
                return t_msg.from_user
        except Exception as e:
            logger.error(f"Failed to fetch target message {target_id} for mod action: {e}")

    # 3. Check mentions in the user's message
    entities_to_check = []
    if msg.entities:
        entities_to_check.extend([(e, msg.text) for e in msg.entities])
    if msg.caption_entities:
        entities_to_check.extend([(e, msg.caption) for e in msg.caption_entities])

    for entity, source_text in entities_to_check:
        try:
            candidate = None
            if entity.type == enums.MessageEntityType.MENTION:
                username = source_text[entity.offset:entity.offset + entity.length]
                candidate = await client.get_users(username)
            elif entity.type == enums.MessageEntityType.TEXT_MENTION:
                candidate = entity.user
            
            if candidate and not candidate.is_self:
                return candidate
        except Exception as e:
            logger.error(f"Error resolving mention: {e}")

    # 4. Fallback to reply target in trigger message (skip bot itself)
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if not msg.reply_to_message.from_user.is_self:
            return msg.reply_to_message.from_user
    
    # 5. Last resort: target the sender (self-defense)
    if msg.from_user:
        return msg.from_user
    
    return None


async def _save_interaction_memory(
    msg: Message,
    parsed: ParsedResponse,
    original_prompt: str,
    raw_answer: str,
    reply_text: str,
) -> None:
    """Save the interaction to long-term memory."""
    if not raw_answer:
        return
    
    try:
        # Clean up reply_text for storage
        short_context = ""
        if reply_text:
            if "reply to a conversation chain" in reply_text:
                short_context = reply_text.split(":\n", 1)[-1].replace("\n- ", " > ").replace("\n", " ").strip()
            else:
                short_context = reply_text.strip()

        # Build memory string
        mem_parts = []
        if parsed.text_content:
            mem_parts.append(parsed.text_content)
        if parsed.reaction:
            mem_parts.append(f"[Reacted: {parsed.reaction}]")
        if parsed.sticker_id:
            sticker_desc = STICKER_TO_DESCRIPTION.get(parsed.sticker_id, "Unknown Sticker")
            mem_parts.append(f"[Sent Sticker: {sticker_desc}]")
        
        final_memory = " ".join(mem_parts) if mem_parts else raw_answer

        save_memory(
            msg.from_user.id, 
            msg.from_user.username, 
            original_prompt, 
            final_memory, 
            context=short_context
        )
    except Exception as e:
        logger.error(f"Failed to save long-term memory: {e}")
