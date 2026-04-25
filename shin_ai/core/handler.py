"""
Core Handler Module

Universal message handler logic for ShinAI, agnostic of platform.
"""
import asyncio
import time
from typing import List

from shin_ai.platforms.models import UnifiedMessage, UnifiedMedia
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.core import state
from shin_ai.core.prompt_builder import (
    build_system_prompt,
    build_runtime_context,
    build_target_instructions,
)
from shin_ai.core.response_parser import parse_ai_response, is_ai_response_valid
from shin_ai.core.action_executor import execute_response
from shin_ai.config import AI_CHOICE, AI_PROVIDER_TIMEOUT_SECONDS, AI_PROVIDER_MAX_RETRIES
from shin_ai.utils.logger_config import logger
from shin_ai.utils.rate_limit import check_rate_limit
from shin_ai.utils.memory import retrieve_memories
from shin_ai.utils.context_manager import get_recent_context_string, get_recent_media_messages
from shin_ai.providers import *
from shin_ai.providers.local_llm import local_llm
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.cerebras import cerebras_api
from shin_ai.providers.groq import groq_api
from shin_ai.providers.openrouter import openrouter_api
from shin_ai.stylers.style_retriever import get_style_examples
from shin_ai.services.social import get_social_context
from shin_ai.services.replies import get_reply_chain
from shin_ai.services.audio_transcriber import transcribe_audio
from shin_ai.data.loader import TELEGRAM_STICKER_MAPPINGS, WHATSAPP_STICKER_MAPPINGS, PERSONALITY


async def process_message(platform: PlatformAdapter, msg: UnifiedMessage):
    """Main message handler for AI-powered responses across any platform."""
    if state.IS_CHECKING_KEYS:
        return

    # Rate limit check
    if msg.from_user and not check_rate_limit(msg.from_user.id) and AI_CHOICE != "manual":
        return

    prompt = _extract_prompt(msg)
    media_list = await _download_media(platform, msg)

    # Audio transcription
    if msg.voice or msg.audio:
        transcription = await _transcribe_audio_message(platform, msg)
        if transcription:
            media_type = "Voice message" if msg.voice else "Audio file"
            audio_disclaimer = (
                f"[{media_type} from user - Transcription]: \"{transcription}\"\n"
                "[TRANSCRIPTION NOTE: The above was transcribed from audio. "
                "It may contain phonetic spelling errors, hallucinated artifacts, "
                "or illogical words due to dialect variations (especially Egyptian Arabic). "
                "Before responding, intelligently interpret any illogical words based on "
                "the surrounding context to find the nearest logical meaning.]"
            )
            prompt = (
                f"{audio_disclaimer}\n\n{prompt}"
                if prompt.strip()
                else audio_disclaimer
            )

    if not media_list:
        prompt_lower = prompt.lower()
        image_keywords = ["image", "photo", "picture", "pic", "sticker", "صورة", "الصورة", "صوره"]
        
        if any(keyword in prompt_lower for keyword in image_keywords):
            logger.info("User mentioned media but no reply chain - checking recent context")
            recent_media = get_recent_media_messages(platform.platform_name, msg.chat.id, max_count=10)
            
            if recent_media:
                media_ids = [m["msg_id"] for m in recent_media[:5]]
                context_media = await _download_media_from_context(platform, msg.chat.id, media_ids)
                media_list.extend(context_media)

    recent_context_section = _get_recent_context(platform.platform_name, msg)

    if getattr(msg, "is_speculative_reply", False):
        bot_identity = PERSONALITY.get("identity", "You are an AI assistant.")
        eval_system = (
            "You are a strict boolean evaluator. "
            "Your task is to determine if the user's message is addressed to you (the AI assistant) or clearly continuing a conversation with you.\n"
            "You recently sent a message, and this is the very next message in the group.\n\n"
            f"--- BOT CONTEXT ---\n"
            f"{bot_identity}\n"
            f"--- RECENT CHAT HISTORY ---\n"
            f"{recent_context_section}\n\n"
            "Rules:\n"
            "1. Output 'YES' if the user explicitly addresses you (using your name from the Bot Context), asks you a question, or says something like 'thanks' or 'haha' in clear direct response to what you just said.\n"
            "2. Output 'NO' if the user addresses someone else by name, responds to another user, or says something completely unrelated to your recent message.\n"
            "3. If in doubt, output 'NO'.\n"
            "You MUST output exactly 'YES' or 'NO' and nothing else."
        )
        eval_prompt = f"User's message: \"{prompt}\""
        try:
            logger.info("Running speculative reply pre-flight evaluation...")
            eval_ans = await _call_ai_provider(msg=msg, system_prompt=eval_system, prompt=eval_prompt, media_list=[])
            if not eval_ans or "YES" not in eval_ans.strip().upper():
                logger.info(f"Pre-flight eval rejected speculative message. Eval: {eval_ans}")
                return
            logger.info("Pre-flight evaluation passed.")
        except Exception as e:
            logger.error(f"Pre-flight evaluation failed: {e}")
            return

    style_examples = _get_style_examples(prompt)
    reply_text = await _get_reply_chain_text(platform, msg)
    is_direct = _is_direct_interaction(msg)
    user_status, reply_target_status = await _get_member_statuses(platform, msg)

    interaction_type = (
        "DIRECT INTERACTION (User is talking to YOU)" 
        if is_direct 
        else "RANDOM INTERJECTION (User is NOT talking to you, you are engaging proactively)"
    )

    if getattr(msg, "is_speculative_reply", False):
        interaction_type = "SPECULATIVE INTERACTION (You just sent a message. This is the first user message following yours. Respond naturally if it's continuing the convo with you, otherwise ignore completely.)"


    runtime_context = build_runtime_context(
        username=msg.from_user.username if msg.from_user else None,
        full_name=msg.from_user.first_name if msg.from_user else "Unknown",
        user_id=msg.from_user.id if msg.from_user else 0,
        user_status=user_status,
        reply_target_status=reply_target_status,
        chat_type=msg.chat.type,
        chat_title=msg.chat.title,
        chat_id=msg.chat.id,
        interaction_type=interaction_type,
    )
    
    # Add platform-specific capability instructions
    if not platform.supports_stickers:
        sticker_warning = (
            "\nCRITICAL: This platform ("
            + platform.platform_name
            + ") DOES NOT support sending stickers. DO NOT use 'sticker:' actions under any circumstances!"
        )
    elif platform.platform_name == "whatsapp":
        sticker_warning = (
            "\nSTICKER NOTE: WhatsApp sticker sends require a media source. "
            "Use 'sticker:wa:<https-url-or-local-path>'. "
            "DO NOT use Telegram file IDs on WhatsApp.\n"
            "CRITICAL REACTION RULE: DO NOT USE `react:<emoji>` on WhatsApp. NEVER SEND REACTIONS ON WHATSAPP. IT IS HARD BLOCKED."
        )
    else:
        sticker_warning = ""

    moderation_warning = ""
    if not getattr(platform, "supports_member_restrictions", True):
        moderation_warning = (
            "\nMODERATION NOTE: This platform does not support per-user mute/unmute. "
            "Do not use action:mute or action:unmute."
        )

    runtime_context += (
        f"\nPLATFORM: You are currently operating on {platform.platform_name.upper()}."
        f"{sticker_warning}{moderation_warning}"
    )

    logger.info(f"[{platform.platform_name}] Built Runtime Metadata:\n{runtime_context}")

    memory_section = await _get_memory_section(prompt)
    social_context_section = get_social_context(msg, reply_text)

    sender_name = msg.from_user.first_name if msg.from_user else "User"
    target_instructions = build_target_instructions(
        msg_id=msg.id,
        sender_name=sender_name,
        reply_msg=msg.reply_to_message,
    )

    if platform.platform_name == "whatsapp":
        sticker_mappings = WHATSAPP_STICKER_MAPPINGS
    else:
        sticker_mappings = TELEGRAM_STICKER_MAPPINGS

    system_prompt = build_system_prompt(
        style_examples=style_examples,
        social_context_section=social_context_section,
        memory_section=memory_section,
        recent_context_section=recent_context_section,
        runtime_context=runtime_context,
        reply_text=reply_text,
        target_instructions=target_instructions,
        sticker_mappings=sticker_mappings,
    )

    typing_task = _start_typing(platform, msg.chat.id)

    try:
        answer = await _call_ai_provider(
            msg=msg,
            system_prompt=system_prompt,
            prompt=prompt,
            media_list=media_list,
        )

        if not is_ai_response_valid(answer):
            logger.warning("AI failed, falling back to manual response")
            try:
                await platform.react(msg.chat.id, msg.id, "😢")
            except Exception as react_err:
                logger.error(f"Fallback react failed: {react_err}")
            try:
                from shin_ai.providers.manual import manual_response
                answer = await manual_response(prompt, msg.from_user)
            except Exception as manual_err:
                logger.error(f"Manual response fallback failed: {manual_err}")
                answer = None

        if not answer:
            try:
                await platform.react(msg.chat.id, msg.id, "👎")
            except Exception as react_err:
                logger.error(f"Final fallback react failed: {react_err}")
            return

        parsed = parse_ai_response(answer)
        
        mod_errors = await execute_response(
            platform=platform,
            msg=msg,
            parsed=parsed,
            default_target_id=msg.id,
            original_prompt=prompt,
            raw_answer=answer,
            reply_text=reply_text,
        )

        if mod_errors:
            error_context = "\n".join(f"- {err}" for err in mod_errors)
            error_prompt = (
                "[INTERNAL SYSTEM ERROR - NOT A USER MESSAGE]\n"
                "The following moderation action(s) you attempted have FAILED:\n"
                f"{error_context}\n\n"
                "Respond naturally to the user about this failure. "
                "Do NOT use any action: commands in your response. "
                "Just send a text message reacting to the failure in your usual style."
            )

            error_answer = await _call_ai_provider(
                msg=msg,
                system_prompt=system_prompt,
                prompt=error_prompt,
                media_list=[],
            )

            if error_answer and is_ai_response_valid(error_answer):
                error_parsed = parse_ai_response(error_answer)
                for p in error_parsed:
                    p.mod_action = None
                    p.mod_target_username = None
                await execute_response(
                    platform=platform,
                    msg=msg,
                    parsed=error_parsed,
                    default_target_id=msg.id,
                    original_prompt=prompt,
                    raw_answer=error_answer,
                    reply_text=reply_text,
                )
    finally:
        await _stop_typing(platform, msg.chat.id, typing_task)


# ===========================================
# Helper Functions
# ===========================================

def _extract_prompt(msg: UnifiedMessage) -> str:
    prompt = msg.text or msg.caption
    if prompt: return prompt
    
    if msg.sticker: return f"[User sent a sticker {msg.sticker.emoji or ''}]"
    elif msg.photo: return "[User sent a photo]"
    elif msg.animation: return "[User sent a GIF/Animation]"
    elif msg.video: return "[User sent a Video]"
    elif msg.voice: return "[User sent a Voice Message]"
    elif msg.audio: return "[User sent an Audio file]"
    elif msg.document: return "[User sent a Document]"
    
    return " "


async def _download_media(platform: PlatformAdapter, msg: UnifiedMessage) -> list[dict]:
    media_list = []
    
    async def process(target_msg: UnifiedMessage, position: str):
        sender_name = target_msg.from_user.username or target_msg.from_user.first_name if target_msg.from_user else "Unknown"
        if target_msg.photo:
            bts = await platform.download_media(target_msg.photo)
            mime = target_msg.photo.mime_type or "image/jpeg"
            return bts, mime, "photo", sender_name
        elif target_msg.sticker and not target_msg.sticker.is_animated and not target_msg.sticker.is_video:
            bts = await platform.download_media(target_msg.sticker)
            mime = target_msg.sticker.mime_type or "image/webp"
            return bts, mime, f"sticker {target_msg.sticker.emoji or ''}".strip(), sender_name
        return None, None, None, None
        
    res = await process(msg, "current")
    if res[0]:
        media_list.append({'bytes': res[0], 'mime_type': res[1], 'sender': res[3], 'position': 'Current message', 'media_type': res[2]})
        
    curr = msg
    depth = 0
    while curr.reply_to_message and depth < 10:
        reply = curr.reply_to_message
        depth += 1
        res = await process(reply, f"reply_{depth}")
        if res[0]:
            media_list.append({'bytes': res[0], 'mime_type': res[1], 'sender': res[3], 'position': f"{depth} messages back", 'media_type': res[2]})
        curr = reply

    return media_list


async def _transcribe_audio_message(platform: PlatformAdapter, msg: UnifiedMessage) -> str:
    """Download and transcribe a voice/audio message using Whisper."""
    media_handle = msg.voice or msg.audio
    if not media_handle:
        return ""

    try:
        audio_bytes = await platform.download_media(media_handle)
        if not audio_bytes:
            logger.warning("Audio download returned empty bytes")
            return ""

        mime_type = media_handle.mime_type or "audio/ogg"
        logger.info(f"Transcribing audio: {len(audio_bytes)} bytes, mime={mime_type}")
        transcription = await transcribe_audio(audio_bytes, mime_type)
        if transcription:
            logger.info(f"Audio transcription result ({len(transcription)} chars): {transcription[:100]}...")
        else:
            logger.warning("Whisper returned empty transcription")
        return transcription
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""


async def _download_media_from_context(platform: PlatformAdapter, chat_id: int | str, media_msg_ids: list[int | str]) -> list[dict]:
    media_list = []
    for idx, msg_id in enumerate(media_msg_ids):
        msg = await platform.get_message(chat_id, msg_id)
        if not msg: continue
        sender_name = msg.from_user.username or msg.from_user.first_name if msg.from_user else "Unknown"
        
        if msg.photo:
            bts = await platform.download_media(msg.photo)
            if bts:
                mime = msg.photo.mime_type or "image/jpeg"
                media_list.append({'bytes': bts, 'mime_type': mime, 'sender': sender_name, 'position': f"From context msg {idx+1}", 'media_type': 'photo'})
        elif msg.sticker and not msg.sticker.is_animated and not msg.sticker.is_video:
            bts = await platform.download_media(msg.sticker)
            if bts:
                mime = msg.sticker.mime_type or "image/webp"
                media_list.append({'bytes': bts, 'mime_type': mime, 'sender': sender_name, 'position': f"From context msg {idx+1}", 'media_type': f"sticker {msg.sticker.emoji or ''}"})
    return media_list


def _get_style_examples(prompt: str) -> str:
    try:
        return "\n".join(get_style_examples(prompt))
    except Exception as e:
        logger.warning(f"No style examples retrieved: {e}")
        return ""


async def _get_reply_chain_text(platform: PlatformAdapter, msg: UnifiedMessage) -> str:
    try:
        reply_chain = await get_reply_chain(msg, platform)
        if reply_chain:
            return "\n\nThe user's message is a reply to a conversation chain (most recent first):\n" + "\n".join([f"- {part}" for part in reply_chain])
    except Exception as e:
        logger.error(f"Error building reply chain: {e}")
        if msg.reply_to_message and msg.reply_to_message.from_user:
            return (f"\n\nThe user's message is a reply to a previous message from "
                    f"{msg.reply_to_message.from_user.username}/{msg.reply_to_message.from_user.first_name} "
                    f"that said: {msg.reply_to_message.text}")
    return ""


def _is_direct_interaction(msg: UnifiedMessage) -> bool:
    if msg.chat.type == "PRIVATE": return True
    text = msg.text or msg.caption or ""
    if "يالبوت" in text: return True
    if msg.mentioned: return True
    # If replying to bot
    if msg.reply_to_message and msg.reply_to_message.from_user and msg.reply_to_message.from_user.is_self: return True
    return False


async def _get_member_statuses(platform: PlatformAdapter, msg: UnifiedMessage) -> tuple[str, str]:
    user_status = "Unknown"
    reply_target_status = "N/A"

    if msg.chat.type in ["GROUP", "SUPERGROUP"] and msg.from_user:
        user_status = await platform.get_chat_member_status(msg.chat.id, msg.from_user.id)
        if msg.reply_to_message and msg.reply_to_message.from_user:
            reply_target_status = await platform.get_chat_member_status(msg.chat.id, msg.reply_to_message.from_user.id)
            
    return user_status, reply_target_status


async def _get_memory_section(prompt: str) -> str:
    try:
        retrieved_mems = await retrieve_memories(prompt)
        if retrieved_mems:
            return "PAST RELEVANT MEMORIES:\n" + "\n".join([f"- {m}" for m in retrieved_mems])
    except Exception:
        pass
    return ""


def _get_recent_context(platform_name: str, msg: UnifiedMessage) -> str:
    try:
        context_str = get_recent_context_string(platform_name, msg.chat.id, msg.id)
        if context_str: return f"RECENT CHAT ACTIVITY:\n{context_str}"
    except Exception:
        pass
    return "RECENT CHAT ACTIVITY: None recorded yet."


def _start_typing(platform: PlatformAdapter, chat_id: int | str) -> asyncio.Task:
    async def _loop():
        try:
            while True:
                await platform.send_chat_action(chat_id, "typing")
                await asyncio.sleep(4.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing loop ended due to error: {e}")
    return asyncio.create_task(_loop())


async def _stop_typing(platform: PlatformAdapter, chat_id: int | str, task: asyncio.Task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    try:
        await platform.send_chat_action(chat_id, "cancel")
    except Exception:
        pass


async def _call_ai_provider(msg: UnifiedMessage, system_prompt: str, prompt: str, media_list: list[dict]) -> str | None:
    max_attempts = max(1, AI_PROVIDER_MAX_RETRIES)
    base_prompt = prompt
    retry_prompt = base_prompt

    for attempt in range(1, max_attempts + 1):
        start_time = time.monotonic()
        try:
            answer = await asyncio.wait_for(
                _execute_ai_provider_once(msg, system_prompt, retry_prompt, media_list),
                timeout=AI_PROVIDER_TIMEOUT_SECONDS,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("AI provider returned empty response")
            return answer
        except asyncio.TimeoutError:
            if attempt >= max_attempts: break
            retry_prompt = base_prompt
        except Exception as e:
            if attempt >= max_attempts: break
            error_text = str(e).strip()[:400]
            retry_prompt = f"{base_prompt}\n\n[INTERNAL RETRY CONTEXT - NOT A USER MESSAGE]\nPrevious attempt failed with {type(e).__name__}: {error_text}\nIf needed, adapt your response approach to avoid the same failure."

    return None


async def _execute_ai_provider_once(msg: UnifiedMessage, system_prompt: str, prompt: str, media_list: list[dict]) -> str | None:
    if AI_CHOICE == "local": return await local_llm(system_prompt, prompt)
    if AI_CHOICE == "gemini": return await gemini_api(system_prompt, prompt, media_list=media_list)
    if AI_CHOICE == "cerebras": return await cerebras_api(system_prompt, prompt)
    if AI_CHOICE == "groq": return await groq_api(system_prompt, prompt)
    if AI_CHOICE == "openrouter": return await openrouter_api(system_prompt, prompt)
    if AI_CHOICE == "manual":
        from shin_ai.providers.manual import manual_response
        # Hack for manual response which takes from_user string or something
        return await manual_response(prompt, msg.from_user)

    logger.error(f"Unknown AI_CHOICE: {AI_CHOICE}")
    return None
