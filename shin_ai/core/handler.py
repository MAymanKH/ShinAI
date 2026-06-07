"""
Core Handler Module

Universal message handler logic for ShinAI, agnostic of platform.
"""
import asyncio
import random
import re as _re
import unicodedata as _ud

from shin_ai.platforms.models import UnifiedMessage
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.core import state
from shin_ai.core.prompt_builder import (
    get_static_system_prompt,
    build_user_prompt,
    build_runtime_context,
    build_target_instructions,
)
from shin_ai.core.response_parser import parse_ai_response, is_ai_response_valid
from shin_ai.core.action_executor import execute_response
from shin_ai.config import (
    AI_CHOICE,
    AI_FALLBACK_PROVIDERS,
    AI_PROVIDER_MAX_RETRIES,
    AI_PROVIDER_TIMEOUT_SECONDS,
    MAX_REPLY_DELAY_SECONDS,
    MIN_REPLY_DELAY_SECONDS,
)
from shin_ai.utils.logger_config import logger
from shin_ai.utils.rate_limit import check_rate_limit
from shin_ai.utils.memory import retrieve_memories
from shin_ai.utils.context_manager import get_recent_context_string, get_recent_media_messages
from shin_ai.providers.local_llm import local_llm
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.cerebras import cerebras_api
from shin_ai.providers.groq import groq_api
from shin_ai.providers.openrouter import openrouter_api
from shin_ai.providers.openai_compatible import openai_compatible_api
from shin_ai.stylers.style_retriever import get_style_examples
from shin_ai.services.social import get_social_context
from shin_ai.services.replies import get_reply_chain
from shin_ai.services.audio_transcriber import transcribe_audio
from shin_ai.data.loader import TELEGRAM_STICKER_MAPPINGS, WHATSAPP_STICKER_MAPPINGS, PERSONALITY


_chat_queues = {}
_chat_tasks = {}

async def process_message(platform: PlatformAdapter, msg: UnifiedMessage):
    """Main message handler for AI-powered responses across any platform."""
    if state.IS_CHECKING_KEYS:
        return

    # Rate limit check
    if msg.from_user and not check_rate_limit(msg.from_user.id) and AI_CHOICE != "manual":
        return

    prompt, media_list = await _prepare_prompt_and_media(platform, msg)
    recent_context_section = _get_recent_context(platform.platform_name, msg)

    if not await _passes_speculative_preflight(msg, prompt, recent_context_section):
        return

    style_examples = _get_style_examples(prompt)
    reply_text = await _get_reply_chain_text(platform, msg)
    runtime_context = await _build_runtime_context(platform, msg)
    memory_section = await _get_memory_section(prompt)
    social_context_section = get_social_context(msg, reply_text)

    _enqueue_frozen_message(
        platform=platform,
        msg=msg,
        prompt=prompt,
        media_list=media_list,
        reply_text=reply_text,
        style_examples=style_examples,
        social_context_section=social_context_section,
        memory_section=memory_section,
        runtime_context=runtime_context,
        target_instructions=_build_target_instructions(msg),
        sticker_mappings=_select_sticker_mappings(platform),
    )


async def _prepare_prompt_and_media(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
) -> tuple[str, list[dict]]:
    prompt = _extract_prompt(msg)
    media_list = await _download_media(platform, msg)
    prompt = await _attach_audio_transcription(platform, msg, prompt)

    if not media_list:
        media_list.extend(await _download_mentioned_recent_media(platform, msg, prompt))

    return prompt, media_list


async def _attach_audio_transcription(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
) -> str:
    # Check current message first, then walk the reply chain
    audio_msg = msg
    if not (audio_msg.voice or audio_msg.audio):
        audio_msg = _find_audio_in_reply_chain(msg)
    if not audio_msg:
        return prompt

    transcription = await _transcribe_audio_message(platform, audio_msg)
    if not transcription:
        return prompt

    sender_name = (
        audio_msg.from_user.first_name if audio_msg.from_user else "Unknown"
    )
    media_type = "Voice message" if audio_msg.voice else "Audio file"
    from_label = "from user" if audio_msg is msg else f"from {sender_name} (replied-to message)"
    audio_disclaimer = (
        f"[{media_type} {from_label} - Transcription]: \"{transcription}\"\n"
        "[TRANSCRIPTION NOTE: The above was transcribed from audio. "
        "It may contain phonetic spelling errors, hallucinated artifacts, "
        "or illogical words due to dialect variations (especially Egyptian Arabic). "
        "Before responding, intelligently interpret any illogical words based on "
        "the surrounding context to find the nearest logical meaning.]"
    )
    return f"{audio_disclaimer}\n\n{prompt}" if prompt.strip() else audio_disclaimer


def _find_audio_in_reply_chain(msg: UnifiedMessage) -> UnifiedMessage | None:
    """Walk the reply chain to find a voice/audio message."""
    curr = msg
    depth = 0
    while curr.reply_to_message and depth < 10:
        reply = curr.reply_to_message
        depth += 1
        if reply.voice or reply.audio:
            return reply
        curr = reply
    return None


async def _download_mentioned_recent_media(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
) -> list[dict]:
    prompt_lower = prompt.lower()
    image_keywords = ["image", "photo", "picture", "pic", "sticker", "صورة", "الصورة", "صوره"]

    if not any(keyword in prompt_lower for keyword in image_keywords):
        return []

    logger.info("User mentioned media but no reply chain - checking recent context")
    recent_media = get_recent_media_messages(platform.platform_name, msg.chat.id, max_count=10)
    if not recent_media:
        return []

    media_ids = [m["msg_id"] for m in recent_media[:5]]
    return await _download_media_from_context(platform, msg.chat.id, media_ids)


async def _passes_speculative_preflight(
    msg: UnifiedMessage,
    prompt: str,
    recent_context_section: str,
) -> bool:
    if not _should_use_speculative_reply(msg):
        return True

    bot_identity = PERSONALITY.get("identity", "You are an AI assistant.")
    eval_system = (
        "You are a strict boolean evaluator. "
        "Your task is to determine if the user's message is addressed to you (the AI assistant) or clearly continuing a conversation with you.\n"
        "You recently sent a message, and this is the very next message in the group.\n\n"
        "Rules:\n"
        "1. Output 'YES' if the user explicitly addresses you (using your name from the Bot Context), asks you a question, or says something like 'thanks' or 'haha' in clear direct response to what you just said.\n"
        "2. Output 'NO' if the user addresses someone else by name, responds to another user, or says something completely unrelated to your recent message.\n"
        "3. If in doubt, output 'NO'.\n"
        "You MUST output exactly 'YES' or 'NO' and nothing else.\n\n"
        f"--- BOT CONTEXT ---\n"
        f"{bot_identity}\n"
        f"--- RECENT CHAT HISTORY ---\n"
        f"{recent_context_section}\n"
    )
    eval_prompt = f"User's message: \"{prompt}\""
    try:
        logger.info("Running speculative reply pre-flight evaluation...")
        eval_ans = await _call_ai_provider(msg=msg, system_prompt=eval_system, prompt=eval_prompt, media_list=[])
        if not eval_ans or "YES" not in eval_ans.strip().upper():
            logger.info(f"Pre-flight eval rejected speculative message. Eval: {eval_ans}")
            return False
        logger.info("Pre-flight evaluation passed.")
        return True
    except Exception as e:
        logger.error(f"Pre-flight evaluation failed: {e}")
        return False


async def _build_runtime_context(platform: PlatformAdapter, msg: UnifiedMessage) -> str:
    user_status, reply_target_status = await _get_member_statuses(platform, msg)
    runtime_context = build_runtime_context(
        username=msg.from_user.username if msg.from_user else None,
        full_name=msg.from_user.first_name if msg.from_user else "Unknown",
        user_id=msg.from_user.id if msg.from_user else 0,
        user_status=user_status,
        reply_target_status=reply_target_status,
        chat_type=msg.chat.type,
        chat_title=msg.chat.title,
        chat_id=msg.chat.id,
        interaction_type=_get_interaction_type(msg),
    )

    runtime_context += (
        f"\nPLATFORM: You are currently operating on {platform.platform_name.upper()}."
        f"{_get_sticker_warning(platform)}{_get_moderation_warning(platform)}"
    )

    logger.info(f"[{platform.platform_name}] Built Runtime Metadata:\n{runtime_context}")
    return runtime_context


def _get_interaction_type(msg: UnifiedMessage) -> str:
    if _is_direct_interaction(msg):
        return "DIRECT INTERACTION (User is talking to YOU)"

    if _should_use_speculative_reply(msg):
        return "SPECULATIVE INTERACTION (You just sent a message. This is the first user message following yours. Respond naturally if it's continuing the convo with you, otherwise ignore completely.)"

    return "RANDOM INTERJECTION (User is NOT talking to you, you are engaging proactively)"


def _get_sticker_warning(platform: PlatformAdapter) -> str:
    if not platform.supports_stickers:
        return (
            "\nCRITICAL: This platform ("
            + platform.platform_name
            + ") DOES NOT support sending stickers. DO NOT use 'sticker:' actions under any circumstances!"
        )

    if platform.platform_name == "whatsapp":
        return (
            "\nSTICKER NOTE: WhatsApp sticker sends require a media source. "
            "Use 'sticker:wa:<https-url-or-local-path>'. "
            "DO NOT use Telegram file IDs on WhatsApp.\n"
        )

    return ""


def _get_moderation_warning(platform: PlatformAdapter) -> str:
    if getattr(platform, "supports_member_restrictions", True):
        return ""

    return (
        "\nMODERATION NOTE: This platform does not support per-user mute/unmute. "
        "Do not use action:mute or action:unmute."
    )


def _build_target_instructions(msg: UnifiedMessage) -> str:
    sender_name = msg.from_user.first_name if msg.from_user else "User"
    return build_target_instructions(
        msg_id=msg.id,
        sender_name=sender_name,
        reply_msg=msg.reply_to_message,
    )


def _select_sticker_mappings(platform: PlatformAdapter) -> dict:
    if platform.platform_name == "whatsapp":
        return WHATSAPP_STICKER_MAPPINGS
    return TELEGRAM_STICKER_MAPPINGS


def _enqueue_frozen_message(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
    media_list: list[dict],
    reply_text: str,
    style_examples: str,
    social_context_section: str,
    memory_section: str,
    runtime_context: str,
    target_instructions: str,
    sticker_mappings: dict,
) -> None:
    key = (platform.platform_name, msg.chat.id)
    if key not in _chat_queues:
        _chat_queues[key] = []

    _chat_queues[key].append({
        "platform": platform,
        "msg": msg,
        "prompt": prompt,
        "media_list": media_list,
        "reply_text": reply_text,
        "style_examples": style_examples,
        "social_context_section": social_context_section,
        "memory_section": memory_section,
        "runtime_context": runtime_context,
        "target_instructions": target_instructions,
        "sticker_mappings": sticker_mappings,
    })

    if key not in _chat_tasks or _chat_tasks[key].done():
        delay = random.uniform(MIN_REPLY_DELAY_SECONDS, MAX_REPLY_DELAY_SECONDS)
        logger.info(f"[{platform.platform_name}] Delaying reply in chat {msg.chat.id} by {delay:.2f}s")
        _chat_tasks[key] = asyncio.create_task(_delayed_queue_processor(key, delay))


async def _delayed_queue_processor(key, delay: float):
    await asyncio.sleep(delay)
    while True:
        queue = _chat_queues.get(key)
        if not queue:
            await asyncio.sleep(0)
            queue = _chat_queues.get(key)
            if not queue:
                _chat_queues.pop(key, None)
                _chat_tasks.pop(key, None)
                return

        task_args = queue.pop(0)
        try:
            await _execute_frozen_message(**task_args)
        except Exception as e:
            logger.error(f"Failed to execute frozen message in queue: {e}")


async def _execute_frozen_message(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    prompt: str,
    media_list: list,
    reply_text: str,
    style_examples: str,
    social_context_section: str,
    memory_section: str,
    runtime_context: str,
    target_instructions: str,
    sticker_mappings: dict,
):
    typing_task = None
    recent_context_section = _get_recent_context(platform.platform_name, msg)

    # Static system prompt — 100% cacheable, never changes
    system_prompt = get_static_system_prompt()

    # Dynamic context packed into user message
    enriched_prompt = build_user_prompt(
        user_message=prompt,
        style_examples=style_examples,
        social_context_section=social_context_section,
        memory_section=memory_section,
        recent_context_section=recent_context_section,
        runtime_context=runtime_context,
        reply_text=reply_text,
        target_instructions=target_instructions,
    )

    if await _should_skip_queued_reply(msg, prompt, recent_context_section):
        logger.info(
            "Skipping queued reply for chat %s (message %s already answered)",
            msg.chat.id,
            msg.id,
        )
        return


    typing_task = _start_typing(platform, msg.chat.id)

    try:
        answer = await _call_ai_provider(
            msg=msg,
            system_prompt=system_prompt,
            prompt=enriched_prompt,
            media_list=media_list,
        )

        if not is_ai_response_valid(answer):
            logger.warning("AI failed, skipping response")
            return

        if not answer:
            logger.warning("AI returned empty response, skipping response")
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
        if typing_task:
            await _stop_typing(platform, msg.chat.id, typing_task)



# ===========================================
# Helper Functions
# ===========================================

def _extract_prompt(msg: UnifiedMessage) -> str:
    prompt = msg.text or msg.caption
    if prompt:
        return prompt

    if msg.sticker:
        return f"[User sent a sticker {msg.sticker.emoji or ''}]"
    if msg.photo:
        return "[User sent a photo]"
    if msg.animation:
        return "[User sent a GIF/Animation]"
    if msg.video:
        return "[User sent a Video]"
    if msg.voice:
        return "[User sent a Voice Message]"
    if msg.audio:
        return "[User sent an Audio file]"
    if msg.document:
        return "[User sent a Document]"
    
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
    if msg.chat.type == "PRIVATE":
        return True

    text = msg.text or msg.caption or ""
    if "يالبوت" in text:
        return True
    if msg.mentioned:
        return True

    return bool(
        msg.reply_to_message
        and msg.reply_to_message.from_user
        and msg.reply_to_message.from_user.is_self
    )


def _should_use_speculative_reply(msg: UnifiedMessage) -> bool:
    return bool(getattr(msg, "is_speculative_reply", False) and not _is_direct_interaction(msg))


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
        if context_str:
            return f"RECENT CHAT ACTIVITY:\n{context_str}"
    except Exception:
        pass
    return "RECENT CHAT ACTIVITY: None recorded yet."


def _build_duplicate_classifier_prompt(recent_context_section: str) -> str:
    """Classifier for speculative replies: skip only true duplicates."""
    return (
        "You are a strict duplicate-detection classifier. "
        "Output ONLY 'REPLY' or 'SKIP'.\n\n"
        "SKIP ONLY if ALL of the following are true:\n"
        "1. The assistant has ALREADY sent a direct reply to THIS EXACT user message "
        "(not just a similar topic — the same question/statement).\n"
        "2. That reply fully and completely answered the user's message.\n"
        "3. No new information or follow-up was added by the user after the assistant's reply.\n\n"
        "REPLY in ALL other cases, including:\n"
        "- The user asked a question, even if a related topic was discussed before.\n"
        "- The user's message is a new thought, joke, or interjection.\n"
        "- The assistant's previous replies only partially or tangentially addressed the topic.\n"
        "- You are unsure whether it was already answered.\n\n"
        "When in doubt, ALWAYS output 'REPLY'.\n\n"
        "--- CONTEXT ---\n"
        f"{recent_context_section}\n"
    )


def _build_relevance_classifier_prompt(
    recent_context_section: str,
    is_direct: bool,
) -> str:
    """Classifier for deciding if the bot can naturally contribute."""
    bot_identity = PERSONALITY.get("identity", "You are an AI assistant.")
    if is_direct:
        context_line = (
            "The user is talking DIRECTLY to the bot (replied to it, mentioned it, etc.). "
            "The bot should usually respond — but not always."
        )
        unsure_guidance = (
            "When genuinely unsure, lean toward 'REPLY' — "
            "the user is talking to the bot, so silence is rude unless truly pointless."
        )
    else:
        context_line = (
            "The bot randomly decided to jump into this conversation. "
            "Decide whether the bot can NATURALLY and MEANINGFULLY contribute."
        )
        unsure_guidance = (
            "When genuinely unsure, lean slightly toward 'SKIP' — "
            "it's better to stay quiet than to force an awkward interjection."
        )

    return (
        "You are a relevance classifier for a group chat bot. "
        "Output ONLY 'REPLY' or 'SKIP'.\n\n"
        f"{context_line}\n\n"
        "Output 'REPLY' if:\n"
        "- The user asked a question or made a request.\n"
        "- The topic is something the bot can joke about, react to, or comment on naturally.\n"
        "- The message is funny, emotional, or invites casual engagement.\n"
        "- The bot has relevant knowledge or a natural opinion on the subject.\n"
        "- The conversation is general/social and the bot can fit in.\n\n"
        "Output 'SKIP' if:\n"
        "- The message is a low-content reaction (laughing emojis, 'lol', 'haha', a sticker, "
        "thumbs up, etc.) that doesn't need a response.\n"
        "- The conversation is deeply personal between specific people and the bot would be intruding.\n"
        "- The topic is so niche or technical that the bot has nothing natural to add.\n"
        "- The message is mundane logistics between other people (e.g. 'meet me at 5', 'okay omw').\n"
        "- The bot's point is already conveyed in recent chat — someone (including the bot) "
        "already said essentially what the bot would say.\n"
        "- Jumping in would feel forced or awkward.\n\n"
        f"{unsure_guidance}\n\n"
        "--- BOT IDENTITY ---\n"
        f"{bot_identity}\n\n"
        "--- RECENT CHAT HISTORY ---\n"
        f"{recent_context_section}\n"
    )



_TRIVIAL_LAUGH_PATTERN = _re.compile(
    r"^[هح\s]+$"          # Arabic laughing (ههههه / ححح)
    r"|^h[ha]+$"           # English laughing (haha, hahaha)
    r"|^lo+l+$"            # lol, looool
    r"|^lma+o+$"           # lmao
    r"|^x+d+$"             # xD, xxdd
    r"|^😂+$|^🤣+$|^😭+$"  # Pure laughing/crying emoji strings
    r"|^ك+$",              # ككككك (Arabic laughing)
    _re.IGNORECASE,
)


def _is_trivial_message(msg: UnifiedMessage) -> bool:
    """Fast-path check for messages that are meaningless to respond to."""
    text = (msg.text or msg.caption or "").strip()

    # Sticker with no meaningful text
    if msg.sticker and not text:
        return True

    if not text:
        return False

    # Pure emoji strings (no letters/digits)
    if all(
        _ud.category(ch).startswith(("So", "Sk", "Sm"))  # Symbol categories
        or _ud.category(ch) == "Zs"                       # Spaces
        or ch in "\ufe0f\u200d"                           # Variation selectors / ZWJ
        for ch in text
    ):
        return True

    # Common laughing / low-content patterns
    cleaned = _re.sub(r"[\s.,!?]+", "", text)
    if cleaned and _TRIVIAL_LAUGH_PATTERN.match(cleaned):
        return True

    return False


async def _should_skip_queued_reply(
    msg: UnifiedMessage,
    prompt: str,
    recent_context_section: str,
) -> bool:
    # Fast-path: trivial messages (stickers, emoji, laughing) never need a reply
    if _is_trivial_message(msg):
        logger.debug("Skip classifier: trivial message, skipping")
        return True

    # Pick the right classifier based on interaction type
    if _should_use_speculative_reply(msg):
        # Speculative reply: only skip true duplicates
        eval_system = _build_duplicate_classifier_prompt(recent_context_section)
    else:
        # Direct interaction or random interjection: check if the bot
        # can naturally and meaningfully contribute
        is_direct = _is_direct_interaction(msg)
        eval_system = _build_relevance_classifier_prompt(
            recent_context_section, is_direct=is_direct,
        )

    eval_prompt = f"User message: \"{prompt}\""

    try:
        eval_ans = await _call_ai_provider(
            msg=msg,
            system_prompt=eval_system,
            prompt=eval_prompt,
            media_list=[],
        )
    except Exception as e:
        logger.warning(f"Queued skip classifier failed: {e}")
        return False

    if not eval_ans:
        return False

    verdict = eval_ans.strip().upper()
    return "SKIP" in verdict


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
    base_prompt = prompt
    provider_chain = _get_provider_chain()
    last_error = None

    for provider in provider_chain:
        retry_prompt = base_prompt
        max_attempts = max(1, AI_PROVIDER_MAX_RETRIES)

        for attempt in range(1, max_attempts + 1):
            try:
                answer = await asyncio.wait_for(
                    _execute_ai_provider_once(provider, msg, system_prompt, retry_prompt, media_list),
                    timeout=AI_PROVIDER_TIMEOUT_SECONDS,
                )
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("AI provider returned empty response")
                logger.info("AI provider '%s' succeeded on attempt %s.", provider, attempt)
                return answer
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt >= max_attempts:
                    break
                retry_prompt = base_prompt
            except Exception as e:
                last_error = e
                if attempt >= max_attempts:
                    break
                error_text = str(e).strip()[:400]
                retry_prompt = (
                    f"{base_prompt}\n\n[INTERNAL RETRY CONTEXT - NOT A USER MESSAGE]\n"
                    f"Previous attempt failed with {type(e).__name__}: {error_text}\n"
                    "If needed, adapt your response approach to avoid the same failure."
                )

        if len(provider_chain) > 1:
            logger.warning("Provider '%s' failed after %s attempts. Falling back.", provider, max_attempts)

    if last_error:
        logger.error("All providers failed. Last error: %s", last_error)
    return None


def _get_provider_chain() -> list[str]:
    primary = AI_CHOICE
    fallbacks = [p for p in AI_FALLBACK_PROVIDERS if p and p != primary]
    return [primary] + fallbacks


async def _execute_ai_provider_once(
    provider: str,
    msg: UnifiedMessage,
    system_prompt: str,
    prompt: str,
    media_list: list[dict],
) -> str | None:
    if provider == "local":
        return await local_llm(system_prompt, prompt)
    if provider == "gemini":
        return await gemini_api(system_prompt, prompt, media_list=media_list)
    if provider == "cerebras":
        return await cerebras_api(system_prompt, prompt)
    if provider == "groq":
        return await groq_api(system_prompt, prompt)
    if provider == "openrouter":
        return await openrouter_api(system_prompt, prompt)
    if provider == "openai-compat":
        return await openai_compatible_api(system_prompt, prompt)
    if provider == "manual":
        from shin_ai.providers.manual import manual_response
        return await manual_response(prompt, msg.from_user)

    logger.error("Unknown AI provider: %s", provider)
    return None
