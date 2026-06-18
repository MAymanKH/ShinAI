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
from shin_ai.core.action_executor import execute_text_messages, execute_pending_actions
from shin_ai.config import (
    AI_PROVIDER_MAX_RETRIES,
    AI_PROVIDER_TIMEOUT_SECONDS,
    MAX_REPLY_DELAY_SECONDS,
    MIN_REPLY_DELAY_SECONDS,
)
from shin_ai.utils.logger_config import logger
from shin_ai.utils.rate_limit import check_rate_limit
from shin_ai.utils.memory import retrieve_memories
from shin_ai.utils.context_manager import get_recent_context_string, get_recent_media_messages
from shin_ai.providers.gemini import gemini_api
from shin_ai.providers.openai_compatible import openai_provider
from shin_ai.providers.registry import get_provider_chain, get_first_gemini_provider
from shin_ai.stylers.style_retriever import get_style_examples
from shin_ai.services.social import get_social_context
from shin_ai.services.replies import get_reply_chain
from shin_ai.services.audio_transcriber import transcribe_audio
from shin_ai.data.loader import PERSONALITY


_chat_queues = {}
_chat_tasks = {}

async def process_message(platform: PlatformAdapter, msg: UnifiedMessage):
    """Main message handler for AI-powered responses across any platform."""
    if state.IS_CHECKING_KEYS:
        return

    # Rate limit check
    if msg.from_user and not check_rate_limit(msg.from_user.id):
        return

    prompt, media_list = await _prepare_prompt_and_media(platform, msg)
    recent_context_section = _get_recent_context(platform.platform_name, msg)

    if not await _passes_speculative_preflight(msg, prompt, recent_context_section):
        return

    style_examples = _get_style_examples(prompt)
    reply_text = await _get_reply_chain_text(platform, msg)
    runtime_context = await _build_runtime_context(platform, msg)
    memory_section = await _get_memory_section(prompt, msg)
    social_context_section = get_social_context(msg, reply_text)

    # Trigger log always visible in production
    user_name = (
        (msg.from_user.username or msg.from_user.first_name)
        if msg.from_user else "unknown"
    )
    user_id = msg.from_user.id if msg.from_user else "?"
    text_preview = (msg.text or msg.caption or "").replace("\n", " ")[:100]
    media_hint = ""
    if msg.photo:      media_hint = " [photo]"
    elif msg.voice:    media_hint = " [voice]"
    elif msg.audio:    media_hint = " [audio]"
    elif msg.video:    media_hint = " [video]"
    elif msg.sticker:  media_hint = " [sticker]"
    elif msg.document: media_hint = " [document]"
    interaction = _get_interaction_type(msg)
    logger.info(
        "[%s] Triggered — chat=%s (%s) | user=%s (%s) | %s%s | text=\"%s%s\"",
        platform.platform_name,
        msg.chat.id,
        msg.chat.title or msg.chat.type,
        user_name,
        user_id,
        interaction.split("(")[0].strip(),
        media_hint,
        text_preview,
        "..." if len(msg.text or msg.caption or "") > 100 else "",
    )
    # ─────────────────────────────────────────────────────────────────────────

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

    logger.debug("User mentioned media but no reply chain — checking recent context for images")
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
        logger.debug("Running speculative reply pre-flight evaluation...")
        eval_ans, _ = await _call_ai_provider(msg=msg, system_prompt=eval_system, prompt=eval_prompt, media_list=[])
        if not eval_ans or "YES" not in eval_ans.strip().upper():
            logger.debug(f"Pre-flight eval rejected speculative message. Eval: {eval_ans!r}")
            return False
        logger.debug("Pre-flight evaluation passed.")
        return True
    except Exception as e:
        logger.error(f"Pre-flight evaluation failed: {e}", exc_info=True)
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

    runtime_context += f"\nPLATFORM: You are currently operating on {platform.platform_name.upper()}."

    logger.debug(
        "[%s] Runtime context built — chat=%s user=%s type=%s",
        platform.platform_name,
        msg.chat.id,
        msg.from_user.id if msg.from_user else "?",
        msg.chat.type,
    )
    return runtime_context


def _get_interaction_type(msg: UnifiedMessage) -> str:
    if _is_direct_interaction(msg):
        return "DIRECT INTERACTION (User is talking to YOU)"

    if _should_use_speculative_reply(msg):
        return "SPECULATIVE INTERACTION (You just sent a message. This is the first user message following yours. Respond naturally if it's continuing the convo with you, otherwise ignore completely.)"

    return "RANDOM INTERJECTION (User is NOT talking to you, you are engaging proactively)"


def _build_target_instructions(msg: UnifiedMessage) -> str:
    sender_name = msg.from_user.first_name if msg.from_user else "User"
    return build_target_instructions(
        msg_id=msg.id,
        sender_name=sender_name,
        reply_msg=msg.reply_to_message,
    )



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
    })

    if key not in _chat_tasks or _chat_tasks[key].done():
        delay = random.uniform(MIN_REPLY_DELAY_SECONDS, MAX_REPLY_DELAY_SECONDS)
        if delay > 0.1:
            logger.info(
                "[%s] Reply queued for chat %s — waiting %.2fs before sending",
                platform.platform_name, msg.chat.id, delay,
            )
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
            logger.error("Failed to execute frozen message in queue: %s", e, exc_info=True)


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
            "[%s] Skipped reply — chat=%s msg=%s (trivial/laugh/sticker) | text=\"%s\"",
            msg.chat.type,
            msg.chat.id,
            msg.id,
            (msg.text or msg.caption or "").replace("\n", " ")[:60],
        )
        return


    typing_task = _start_typing(platform, msg.chat.id)

    try:
        answer, pending_actions = await _call_ai_provider(
            msg=msg,
            system_prompt=system_prompt,
            prompt=enriched_prompt,
            media_list=media_list,
            original_prompt=prompt,
        )

        if not answer and not pending_actions:
            logger.warning(
                "[%s] AI returned empty response for chat=%s user=%s — skipping",
                platform.platform_name,
                msg.chat.id,
                msg.from_user.id if msg.from_user else "?",
            )
            return

        # Execute tool-called actions (reactions, stickers, moderation)
        # FIRST — before any text/skip logic so they always run even if
        # the AI chose [SKIP] or produced no text alongside the tool call.
        mod_errors = await execute_pending_actions(
            platform=platform,
            msg=msg,
            pending_actions=pending_actions,
            default_reply_to_id=msg.id,
            original_prompt=prompt,
            raw_answer=answer or "",
            reply_text=reply_text,
        )

        # Determine whether the AI chose to skip text output
        skip_text = False
        if answer:
            clean_ans = answer.strip().strip("`").strip().upper()
            if clean_ans in ("[SKIP]", "SKIP"):
                logger.info(
                    "[%s] AI chose to skip — chat=%s user=%s | trigger=\"%s\"",
                    platform.platform_name,
                    msg.chat.id,
                    msg.from_user.id if msg.from_user else "?",
                    (prompt or "").replace("\n", " ")[:80],
                )
                skip_text = True

        if not skip_text:
            # Split text on '---' for multi-message support
            text_messages = [
                m.strip() for m in (answer or "").split("---") if m.strip()
            ]

            # When tool actions were executed, filter out meta-commentary
            # that the AI sometimes produces alongside tool calls, e.g.
            # "(No further action needed as the sticker was sent)."
            if pending_actions and text_messages:
                text_messages = _filter_action_meta_commentary(text_messages)

            # Send text messages
            if text_messages:
                await execute_text_messages(
                    platform=platform,
                    msg=msg,
                    messages=text_messages,
                    default_reply_to_id=msg.id,
                    original_prompt=prompt,
                    raw_answer=answer or "",
                    reply_text=reply_text,
                )

        if mod_errors:
            error_context = "\n".join(f"- {err}" for err in mod_errors)
            error_prompt = (
                "[INTERNAL SYSTEM ERROR - NOT A USER MESSAGE]\n"
                "The following moderation action(s) you attempted have FAILED:\n"
                f"{error_context}\n\n"
                "Respond naturally to the user about this failure. "
                "Do NOT call any moderation tools in your response. "
                "Just send a text message reacting to the failure in your usual style."
            )

            error_answer, _ = await _call_ai_provider(
                msg=msg,
                system_prompt=system_prompt,
                prompt=error_prompt,
                media_list=[],
            )

            if error_answer and error_answer.strip():
                error_messages = [
                    m.strip() for m in error_answer.split("---") if m.strip()
                ]
                await execute_text_messages(
                    platform=platform,
                    msg=msg,
                    messages=error_messages,
                    default_reply_to_id=msg.id,
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
            logger.warning("[Audio] Download returned empty bytes — skipping transcription")
            return ""

        mime_type = media_handle.mime_type or "audio/ogg"
        transcription = await transcribe_audio(audio_bytes, mime_type)
        if transcription:
            logger.info(
                "[Audio] Transcribed %d bytes (%s) → %d chars: \"%s%s\"",
                len(audio_bytes),
                mime_type,
                len(transcription),
                transcription[:80],
                "..." if len(transcription) > 80 else "",
            )
        else:
            logger.warning("[Audio] Whisper returned empty transcription for %d bytes (%s)", len(audio_bytes), mime_type)
        return transcription
    except Exception as e:
        logger.error("Audio transcription failed: %s", e, exc_info=True)
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
        logger.debug("No style examples retrieved: %s", e)
        return ""


async def _get_reply_chain_text(platform: PlatformAdapter, msg: UnifiedMessage) -> str:
    try:
        reply_chain = await get_reply_chain(msg, platform)
        if reply_chain:
            return "\n\nThe user's message is a reply to a conversation chain (most recent first):\n" + "\n".join([f"- {part}" for part in reply_chain])
    except Exception as e:
        logger.error("Error building reply chain: %s", e, exc_info=True)
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


_MEMORY_RECALL_PATTERN = _re.compile(
    r"\b("
    r"remember|recall|memory|memories|forgot|forget|"
    r"previous|previously|earlier|before|last|yesterday|ago|"
    r"history|past|old|what did|when did|where did|who said|"
    r"did i|did we|have i|have we|my name|who am i|know me"
    r")\b",
    _re.IGNORECASE,
)

_ARABIC_MEMORY_RECALL_TERMS = (
    "فاكر",
    "فكرك",
    "تفتكر",
    "افتكر",
    "نسيت",
    "ذاكرة",
    "اتقال",
    "قلت",
    "قولت",
    "قال",
    "قالت",
    "قولنا",
    "اتكلمنا",
    "كلمنا",
    "قبل",
    "زمان",
    "امبارح",
    "مبارح",
    "النهارده",
    "انهارده",
    "امتى",
    "فين",
    "مين انا",
    "اسمي",
    "تعرفني",
)


def _should_retrieve_memory(prompt: str, msg: UnifiedMessage) -> bool:
    """Cheaply decide whether automatic long-term memory is likely useful."""
    text = (prompt or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if _MEMORY_RECALL_PATTERN.search(lowered):
        return True

    if any(term in text for term in _ARABIC_MEMORY_RECALL_TERMS):
        return True

    if msg.reply_to_message and any(marker in lowered for marker in ("this", "that", "ده", "دا", "دي")):
        return True

    return False


async def _get_memory_section(prompt: str, msg: UnifiedMessage) -> str:
    if not _should_retrieve_memory(prompt, msg):
        logger.debug("Skipping automatic memory retrieval for non-recall prompt")
        return ""

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

    return False

# Patterns that catch AI meta-commentary about tool actions just executed.
# These are messages like "(No further action needed as the sticker was sent)."
# that the model sometimes emits after calling send_sticker / send_reaction.
_ACTION_META_COMMENTARY_PATTERN = _re.compile(
    r"^\s*\(?("
    r"no\s+(further|additional)\s+(action|response|message|reply)"
    r"|sticker\s+(was|has been)\s+sent"
    r"|reaction\s+(was|has been)\s+(sent|added|applied)"
    r"|already\s+(sent|reacted|responded)"
    r"|nothing\s+(else|more)\s+(to|needed)"
    r"|action\s+(completed|done|taken)"
    r"|that'?s?\s+(all|it)\b"
    r")\b",
    _re.IGNORECASE,
)


def _filter_action_meta_commentary(messages: list[str]) -> list[str]:
    """Remove AI meta-commentary about tool actions from the text messages.

    When the AI calls send_sticker or send_reaction, it sometimes also
    produces a text message like "(No further action needed as the sticker
    was sent)." — these should never be sent to the user.
    """
    filtered = []
    for text in messages:
        if _ACTION_META_COMMENTARY_PATTERN.search(text):
            logger.debug("Filtered action meta-commentary: %r", text[:100])
            continue
        filtered.append(text)
    return filtered


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


async def _call_ai_provider(
    msg: UnifiedMessage,
    system_prompt: str,
    prompt: str,
    media_list: list[dict],
    original_prompt: str | None = None,
) -> tuple[str | None, list[dict]]:
    base_prompt = prompt
    provider_chain = get_provider_chain()
    last_error = None
    media_context = None

    for provider_cfg in provider_chain:
        provider_prompt = base_prompt
        provider_media = media_list if media_list else []

        if media_list and provider_cfg.type != "gemini":
            if media_context is None:
                media_context = await _describe_media_with_gemini(original_prompt or base_prompt, media_list)

            if media_context:
                provider_prompt = _append_media_context(base_prompt, media_context)
            else:
                logger.warning(
                    "Provider '%s' does not receive raw media and Gemini media description failed.",
                    provider_cfg.name,
                )

        retry_prompt = provider_prompt
        max_attempts = max(1, AI_PROVIDER_MAX_RETRIES)

        for attempt in range(1, max_attempts + 1):
            try:
                answer, pending_actions = await asyncio.wait_for(
                    _execute_ai_provider_once(provider_cfg, msg, system_prompt, retry_prompt, provider_media),
                    timeout=AI_PROVIDER_TIMEOUT_SECONDS,
                )
                if not isinstance(answer, str) or (not answer.strip() and not pending_actions):
                    raise RuntimeError("AI provider returned empty response")
                logger.info("AI provider '%s' succeeded on attempt %s.", provider_cfg.name, attempt)
                return answer, pending_actions
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt >= max_attempts:
                    break
                retry_prompt = provider_prompt
            except Exception as e:
                last_error = e
                if attempt >= max_attempts:
                    break
                error_text = str(e).strip()[:400]
                retry_prompt = (
                    f"{provider_prompt}\n\n[INTERNAL RETRY CONTEXT - NOT A USER MESSAGE]\n"
                    f"Previous attempt failed with {type(e).__name__}: {error_text}\n"
                    "If needed, adapt your response approach to avoid the same failure."
                )

        if len(provider_chain) > 1:
            logger.warning("Provider '%s' failed after %s attempts. Falling back.", provider_cfg.name, max_attempts)

    if last_error:
        logger.error("All providers failed. Last error: %s", last_error)
    return None, []


async def _describe_media_with_gemini(user_message: str, media_list: list[dict]) -> str:
    """Use the configured Gemini provider to describe media for non-vision providers."""
    gemini_cfg = get_first_gemini_provider()
    if gemini_cfg is None:
        logger.warning("No Gemini provider configured; cannot describe media for non-vision provider.")
        return ""

    media_description_system = (
        "You describe attached media for another AI model. Return a concise, "
        "factual description of what is visible and any readable text. "
        "IMPORTANT: If the user is asking about something specific in the media, you MUST "
        "identify it and answer it directly, accurately, and in detail so the other AI model "
        "can answer the user's question correctly."
    )
    summary_prompt = (
        "Describe the attached media for another AI provider that cannot see images. "
        "Be concise but include all visually relevant details, text/OCR, objects, people, "
        "actions, layout, and anything that may matter for answering the user's message.\n\n"
        "User message/context:\n"
        f"{user_message}"
    )

    try:
        media_context, _ = await gemini_api(media_description_system, summary_prompt, media_list=media_list)
    except Exception as e:
        logger.error(f"Gemini media fallback failed: {e}")
        return ""

    media_context = media_context.strip() if isinstance(media_context, str) else ""
    if media_context:
        logger.info("Gemini media fallback produced %s chars of context.", len(media_context))
    return media_context


def _append_media_context(prompt: str, media_context: str) -> str:
    return (
        f"{prompt}\n\n"
        "[INTERNAL MEDIA CONTEXT - generated by Gemini from attached media, not a user message]\n"
        f"{media_context}"
    )


async def _execute_ai_provider_once(
    provider_cfg,
    msg: UnifiedMessage,
    system_prompt: str,
    prompt: str,
    media_list: list[dict],
) -> tuple[str, list[dict]]:
    if provider_cfg.type == "gemini":
        return await gemini_api(system_prompt, prompt, media_list=media_list)

    if provider_cfg.type == "openai":
        return await openai_provider(provider_cfg, system_prompt, prompt, media_list=media_list)

    if provider_cfg.name == "manual":
        from shin_ai.providers.manual import manual_response
        text = await manual_response(prompt, msg.from_user)
        return text or "", []

    logger.error("Unknown provider type '%s' for provider '%s'", provider_cfg.type, provider_cfg.name)
    return "", []
