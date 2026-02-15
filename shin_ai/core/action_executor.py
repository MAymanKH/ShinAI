"""
Action Executor Module

Executes parsed AI response actions (reactions, stickers, text, moderation).
"""
import asyncio
from pyrogram import Client, enums
from pyrogram.types import Message

from shin_ai.core.response_parser import ParsedResponse
from shin_ai.data.loader import STICKER_TO_DESCRIPTION, MEMBERS
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
) -> list[str]:
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
        
    Returns:
        List of moderation action error strings (empty if all succeeded)
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
    mod_errors = []
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
        
        # Collect moderation errors
        mod_error = await _execute_mod_action(client, msg, single_parsed)
        if mod_error:
            mod_errors.append(mod_error)
    
    return mod_errors


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


def _resolve_name_to_username(name: str) -> str | None:
    """
    Look up a name/nickname in the MEMBERS social context and return the @username.
    
    Searches through all members' names, preferred_name, and dict keys
    to find a match, then returns the @username if found.
    """
    name_lower = name.lower().strip().replace("@", "")
    
    for key, data in MEMBERS.items():
        # Check all names (usernames + nicknames)
        for n in data.get("names", []):
            if n.lower().replace("@", "") == name_lower:
                # Find the @username from the names list
                for candidate in data.get("names", []):
                    if candidate.startswith("@"):
                        return candidate.replace("@", "")
                # Fallback to the dict key if it looks like a username
                if not key.startswith("!") and not key[0].isdigit():
                    return key
        
        # Check preferred_name (could be comma-separated)
        for pname in data.get("preferred_name", "").split(","):
            if pname.strip().lower() == name_lower:
                for candidate in data.get("names", []):
                    if candidate.startswith("@"):
                        return candidate.replace("@", "")
                if not key.startswith("!") and not key[0].isdigit():
                    return key
        
        # Check the dict key itself
        if key.lower() == name_lower:
            for candidate in data.get("names", []):
                if candidate.startswith("@"):
                    return candidate.replace("@", "")
            return key
    
    return None


async def _execute_mod_action(
    client: Client, 
    msg: Message, 
    parsed: ParsedResponse,
) -> str | None:
    """Execute a moderation action (kick, ban, unban, mute, unmute, add) if requested.
    
    Returns:
        Error description string if the action failed, None if succeeded or no action.
    """
    if not parsed.mod_action:
        return None
    
    action = parsed.mod_action
    
    # Unban and Invite don't need target resolution from context - they need a username
    if action in ("unban", "add"):
        target = await _resolve_mod_target_simple(client, msg, parsed.mod_target_username)
        if not target:
            logger.warning(f"{action} action requested but no target could be resolved")
            return f"{action.upper()} FAILED: Could not find the user. No username was specified. Use action:{action}:@username to specify who to {action}."
        try:
            if action == "unban":
                await client.unban_chat_member(msg.chat.id, target.id)
            else:  # add - generate invite link and DM it to the user
                invite_link = await client.create_chat_invite_link(
                    msg.chat.id, member_limit=1
                )
                await client.send_message(
                    target.id,
                    f"You've been invited to join the group:\n{invite_link.invite_link}"
                )
            return None
        except Exception as e:
            logger.error(f"{action} failed: {e}")
            return f"{action.upper()} FAILED on {target.first_name} (@{target.username or 'N/A'}): {e}"
    
    # For kick/ban/mute/unmute - resolve the target from context
    target = await _resolve_mod_target(
        client, msg, parsed.mod_target_username, parsed.target_id
    )
    
    if not target:
        logger.warning(f"{action} action requested but no target could be resolved")
        return f"{action.upper()} FAILED: Could not determine who to {action}. No target message or username was found."
    
    target_display = f"{target.first_name} (@{target.username or 'N/A'}, id:{target.id})"
    
    try:
        # Check status before acting - can't act on admins/owners
        chat_member = await client.get_chat_member(msg.chat.id, target.id)
        if chat_member.status in [enums.ChatMemberStatus.ADMINISTRATOR, enums.ChatMemberStatus.OWNER]:
            return f"{action.upper()} FAILED on {target_display}: Target is an admin/owner. You cannot {action} admins or the group owner."
        
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
                    can_add_web_page_previews=True,
                )
            )
        
        return None  # Success
        
    except Exception as e:
        logger.error(f"{action} failed: {e}")
        return f"{action.upper()} FAILED on {target_display}: {e}"


async def _resolve_mod_target_simple(
    client: Client,
    msg: Message, 
    ai_specified_username: str | None,
):
    """Resolve target from username only (used for unban where user isn't in chat)."""
    if ai_specified_username:
        clean = ai_specified_username.replace("@", "")
        
        # First try direct Telegram lookup
        try:
            result = await client.get_users(clean)
            if result:
                return result
        except Exception:
            pass
        
        # If direct lookup failed, try resolving via MEMBERS social context
        resolved_username = _resolve_name_to_username(clean)
        if resolved_username:
            try:
                result = await client.get_users(resolved_username)
                if result:
                    logger.info(f"Resolved name '{clean}' → @{resolved_username} via social context")
                    return result
            except Exception as e:
                logger.error(f"Failed to resolve name '{clean}' → @{resolved_username}: {e}")
    
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
    # 1. AI Specified Username (or name/nickname → resolve via social context)
    if ai_specified_username:
        clean = ai_specified_username.replace("@", "")
        
        # First try direct Telegram lookup
        try:
            target = await client.get_users(clean)
            if target:
                return target
        except Exception:
            pass
        
        # If direct lookup failed, try resolving via MEMBERS social context
        resolved_username = _resolve_name_to_username(clean)
        if resolved_username:
            try:
                target = await client.get_users(resolved_username)
                if target:
                    logger.info(f"Resolved name '{clean}' → @{resolved_username} via social context")
                    return target
            except Exception as e:
                logger.error(f"Failed to resolve name '{clean}' → @{resolved_username}: {e}")

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
