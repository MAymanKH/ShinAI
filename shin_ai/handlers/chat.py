"""
Chat Handler Module

Main message handler for the bot's conversational AI functionality.
"""
import random
from pyrogram import Client, filters, enums
from pyrogram.types import Message

from shin_ai.core.client import app
from shin_ai.core import state
from shin_ai.core.prompt_builder import (
    build_system_prompt,
    build_runtime_context,
    build_target_instructions,
)
from shin_ai.core.response_parser import parse_ai_response, is_ai_response_valid
from shin_ai.core.action_executor import execute_response
from shin_ai.config import AI_CHOICE
from shin_ai.utils.logger_config import logger
from shin_ai.utils.rate_limit import check_rate_limit
from shin_ai.utils.memory import retrieve_memories
from shin_ai.utils.context_manager import add_message_to_context, get_recent_context_string
from shin_ai.providers import *
from shin_ai.providers.local_llm import local_llm
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.cerebras import cerebras_api
from shin_ai.providers.groq import groq_api
from shin_ai.providers.openrouter import openrouter_api
from shin_ai.stylers.style_retriever import get_style_examples
from shin_ai.services.social import get_social_context
from shin_ai.services.replies import check_reply_chain, get_reply_chain, save_reply


# ===========================================
# Context Recorder
# ===========================================

@app.on_message(filters.group, group=-1)
async def context_recorder(client: Client, msg: Message):
    """
    Records messages in the short-term rolling buffer.
    Runs in group -1 to execute before the main handler.
    """
    try:
        add_message_to_context(msg)
    except Exception as e:
        logger.error(f"Context recorder failed: {e}")


# ===========================================
# Message Filter
# ===========================================

async def yalbot_filter_func(_, client: Client, msg: Message) -> bool:
    """
    Filter to determine if the bot should respond to a message.
    
    Triggers on:
    - Direct mentions ("يالبوت")
    - Replies to bot messages
    - Random 1% chance
    """
    text = msg.text or msg.caption
    
    # No text/caption - check for supported media in reply chain
    if not text:
        if not (msg.photo or msg.sticker):
            return False
        if await check_reply_chain(msg):
            return True
        return False
    
    # Direct mention (prioritize "يالبوت" over "يالبوتة" which is for wife bot)
    if "يالبوت" in text and text.count("يالبوت") > text.count("يالبوتة"):
        return True
    
    # Reply chain to bot's previous messages
    if await check_reply_chain(msg):
        return True
    
    # 1% random interjection chance
    if random.random() < 0.01:
        return True
    
    return False


yalbot_filter = filters.create(yalbot_filter_func)


# ===========================================
# Main Handler
# ===========================================

@app.on_message(yalbot_filter)
async def yalbot(client: Client, msg: Message):
    """Main message handler for AI-powered responses."""
    if state.IS_CHECKING_KEYS:
        return

    # Rate limit check
    if not check_rate_limit(msg.from_user.id) and AI_CHOICE != "manual":
        return await msg.react("😴")

    # Extract prompt from message
    prompt = _extract_prompt(msg)
    
    # Download any attached media
    image_bytes, mime_type = await _download_media(client, msg)
    
    # Gather context
    style_examples = _get_style_examples(prompt)
    reply_text = await _get_reply_chain_text(client, msg)
    is_direct = await _is_direct_interaction(msg)
    user_status, reply_target_status = await _get_member_statuses(client, msg)
    
    # Build runtime context
    interaction_type = (
        "DIRECT INTERACTION (User is talking to YOU)" 
        if is_direct 
        else "RANDOM INTERJECTION (User is NOT talking to you, you are engaging proactively)"
    )
    
    runtime_context = build_runtime_context(
        username=msg.from_user.username,
        full_name=msg.from_user.full_name,
        user_id=msg.from_user.id,
        user_status=user_status,
        reply_target_status=reply_target_status,
        chat_type=str(msg.chat.type),
        chat_title=msg.chat.title,
        chat_id=msg.chat.id,
        interaction_type=interaction_type,
    )
    
    # Get memory and context sections
    memory_section = _get_memory_section(prompt)
    recent_context_section = _get_recent_context(msg)
    social_context_section = get_social_context(msg, reply_text)
    
    print(social_context_section)  # Debug output
    
    # Build target instructions
    sender_name = msg.from_user.first_name if msg.from_user else "User"
    valid_targets, target_instructions = build_target_instructions(
        msg_id=msg.id,
        sender_name=sender_name,
        reply_msg=msg.reply_to_message,
    )
    
    # Build the system prompt
    system_prompt = build_system_prompt(
        style_examples=style_examples,
        social_context_section=social_context_section,
        memory_section=memory_section,
        recent_context_section=recent_context_section,
        runtime_context=runtime_context,
        reply_text=reply_text,
        target_instructions=target_instructions,
    )

    logger.info(f"style_examples: {style_examples}")
    logger.info(f"runtime context: {runtime_context}")
    logger.info(f"reply_text: {reply_text}")

    # Call AI provider
    answer = await _call_ai_provider(
        client=client,
        msg=msg,
        system_prompt=system_prompt,
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )

    logger.info(f"Answer: {answer}")

    # Fallback to manual if AI failed
    if not is_ai_response_valid(answer):
        logger.warning("AI failed, falling back to manual response")
        await msg.react("😢")
        from shin_ai.providers.manual import manual_response
        answer = await manual_response(prompt, msg.from_user)

    if not answer:
        return await msg.react("👎")

    # Parse and execute response
    parsed = parse_ai_response(answer)
    
    await execute_response(
        client=client,
        msg=msg,
        parsed=parsed,
        valid_targets=valid_targets,
        original_prompt=prompt,
        raw_answer=answer,
        reply_text=reply_text,
    )


# ===========================================
# Helper Functions
# ===========================================

def _extract_prompt(msg: Message) -> str:
    """Extract the text prompt from a message."""
    prompt = msg.text or msg.caption
    
    if prompt:
        return prompt
    
    # Generate description for media messages
    if msg.sticker:
        emoji = msg.sticker.emoji or ""
        return f"[User sent a sticker {emoji}]"
    elif msg.photo:
        return "[User sent a photo]"
    elif msg.animation:
        return "[User sent a GIF/Animation]"
    elif msg.video:
        return "[User sent a Video]"
    elif msg.voice:
        return "[User sent a Voice Message]"
    elif msg.audio:
        return "[User sent an Audio file]"
    elif msg.document:
        return "[User sent a Document]"
    
    return " "


async def _download_media(client: Client, msg: Message) -> tuple[bytes | None, str | None]:
    """Download photo or sticker from message for AI processing."""
    image_bytes = None
    mime_type = None

    try:
        target_msg = msg
        # If current message has no media, check reply
        if not (msg.photo or msg.sticker) and msg.reply_to_message:
            target_msg = msg.reply_to_message

        if target_msg.photo:
            logger.info("Downloading photo...")
            file_stream = await client.download_media(target_msg.photo, in_memory=True)
            image_bytes = file_stream.getvalue()
            mime_type = "image/jpeg"
            logger.info("Photo downloaded.")
        elif target_msg.sticker:
            if not target_msg.sticker.is_animated and not target_msg.sticker.is_video:
                logger.info("Downloading sticker...")
                file_stream = await client.download_media(target_msg.sticker, in_memory=True)
                image_bytes = file_stream.getvalue()
                mime_type = "image/webp"
                logger.info("Sticker downloaded.")
    except Exception as e:
        logger.error(f"Error downloading media: {e}")

    return image_bytes, mime_type


def _get_style_examples(prompt: str) -> str:
    """Retrieve style examples for the prompt."""
    try:
        examples = get_style_examples(prompt)
        style_str = "\n".join(examples)
        logger.info("Retrieved style examples")
        return style_str
    except Exception as e:
        logger.warning(f"No style examples retrieved: {e}")
        return ""


async def _get_reply_chain_text(client: Client, msg: Message) -> str:
    """Build the reply chain context string."""
    try:
        reply_chain = await get_reply_chain(client, msg)
        if reply_chain:
            return (
                "\n\nThe user's message is a reply to a conversation chain (most recent first):\n" 
                + "\n".join([f"- {part}" for part in reply_chain])
            )
    except Exception as e:
        logger.error(f"Error building reply chain: {e}")
        # Fallback to simple reply info
        if msg.reply_to_message:
            try:
                reply_msg = msg.reply_to_message
                if reply_msg.from_user:
                    return (
                        f"\n\nThe user's message is a reply to a previous message from "
                        f"{reply_msg.from_user.username}/{reply_msg.from_user.full_name} "
                        f"that said: {reply_msg.text}"
                    )
            except:
                pass
    
    return ""


async def _is_direct_interaction(msg: Message) -> bool:
    """Check if this is a direct interaction with the bot."""
    text_content = msg.text or msg.caption or ""
    if "يالبوت" in text_content:
        return True
    if await check_reply_chain(msg):
        return True
    return False


async def _get_member_statuses(client: Client, msg: Message) -> tuple[str, str]:
    """Get the chat member statuses for sender and reply target."""
    user_status = "Unknown"
    reply_target_status = "N/A"

    if msg.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:
        return user_status, reply_target_status

    try:
        mem = await client.get_chat_member(msg.chat.id, msg.from_user.id)
        user_status = str(mem.status).replace("ChatMemberStatus.", "")
        
        if msg.reply_to_message and msg.reply_to_message.from_user:
            t_mem = await client.get_chat_member(msg.chat.id, msg.reply_to_message.from_user.id)
            reply_target_status = str(t_mem.status).replace("ChatMemberStatus.", "")
    except Exception as e:
        logger.warning(f"Failed to fetch member status: {e}")

    return user_status, reply_target_status


def _get_memory_section(prompt: str) -> str:
    """Retrieve relevant memories for the prompt."""
    try:
        retrieved_mems = retrieve_memories(prompt)
        if retrieved_mems:
            return "PAST RELEVANT MEMORIES:\n" + "\n".join([f"- {m}" for m in retrieved_mems])
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
    return ""


def _get_recent_context(msg: Message) -> str:
    """Get recent chat context."""
    try:
        recent_context_str = get_recent_context_string(msg.chat.id, current_msg_id=msg.id)
        if recent_context_str:
            return f"RECENT GROUP ACTIVITY (Last 50 messages):\n{recent_context_str}"
        return "RECENT GROUP ACTIVITY: None recorded yet."
    except Exception as e:
        logger.error(f"Error getting recent context: {e}")
        return ""


async def _call_ai_provider(
    client: Client,
    msg: Message,
    system_prompt: str,
    prompt: str,
    image_bytes: bytes | None,
    mime_type: str | None,
) -> str | None:
    """Call the configured AI provider."""
    try:
        await client.send_chat_action(msg.chat.id, enums.ChatAction.TYPING)
        
        if AI_CHOICE == "local":
            answer = await local_llm(system_prompt, prompt)
        elif AI_CHOICE == "gemini":
            answer = await gemini_api(system_prompt, prompt, image_bytes=image_bytes, mime_type=mime_type)
        elif AI_CHOICE == "cerebras":
            answer = await cerebras_api(system_prompt, prompt)
        elif AI_CHOICE == "groq":
            answer = await groq_api(system_prompt, prompt)
        elif AI_CHOICE == "openrouter":
            answer = await openrouter_api(system_prompt, prompt)
        elif AI_CHOICE == "manual":
            from shin_ai.providers.manual import manual_response
            answer = await manual_response(prompt, msg.from_user)
        else:
            logger.error(f"Unknown AI_CHOICE: {AI_CHOICE}")
            answer = None
            
        await client.send_chat_action(msg.chat.id, enums.ChatAction.CANCEL)
        return answer
    except Exception as e:
        logger.error(f"AI error: {e}")
        return None
