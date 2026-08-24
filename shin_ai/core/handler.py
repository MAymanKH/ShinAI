"""
Core Handler Module

Universal message handler logic for ShinAI, agnostic of platform.
"""
import asyncio
import hashlib
import os
import random
import re as _re
import time
import unicodedata as _ud
from dataclasses import dataclass

from shin_ai.platforms.models import UnifiedMessage
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.core import state
from shin_ai.core.interaction_scheduler import InteractionScheduler
from shin_ai.core.provider_router import call_ai_provider
from shin_ai.core.request_context import reset_request_context
from shin_ai.core.prompt_builder import (
    get_static_system_prompt,
    build_user_prompt,
    build_runtime_context,
    build_target_instructions,
)
from shin_ai.core.action_executor import (
    execute_text_messages,
    execute_pending_actions,
    split_reply_messages,
)
from shin_ai.config import (
    GROUP_MAX_RESPONSES_PER_WINDOW,
    GROUP_RATE_LIMIT_WINDOW_SECONDS,
    INTERACTION_TTL_SECONDS,
    LOG_CONTENT_PREVIEW_CHARS,
    MAX_REPLY_DELAY_SECONDS,
    MAX_CONCURRENT_INTERACTIONS,
    MAX_PENDING_INTERACTIONS,
    MIN_REPLY_DELAY_SECONDS,
    PER_CHAT_QUEUE_SIZE,
    SHUTDOWN_GRACE_SECONDS,
    EVENT_DEDUP_TTL_SECONDS,
)
from shin_ai.coordination.runtime import get_coordination_store
from shin_ai.utils.logger_config import bind_log_context, logger
from shin_ai.utils.rate_limit import check_group_rate_limit_shared, check_rate_limit_shared
from shin_ai.utils.memory import retrieve_memories
from shin_ai.utils.context_manager import get_recent_context_string
from shin_ai.stylers.style_retriever import get_style_examples
from shin_ai.services.social import get_social_context
from shin_ai.services.replies import get_reply_chain
from shin_ai.services.media import prepare_prompt_and_media
from shin_ai.data.loader import PERSONALITY


@dataclass(frozen=True, slots=True)
class _AdmittedInteraction:
    platform: PlatformAdapter
    message: UnifiedMessage
    interaction_id: str


_interaction_scheduler: InteractionScheduler[_AdmittedInteraction] | None = None
_shutting_down = False

# Typing indicator management
_active_typing: dict = {}
_MAX_TYPING_SECONDS = 120.0

async def process_message(platform: PlatformAdapter, msg: UnifiedMessage):
    """Deduplicate and admit an interaction without retaining downloaded media."""
    if state.IS_CHECKING_KEYS or _shutting_down:
        return

    interaction_id = _interaction_id(platform, msg)
    with bind_log_context(**_message_log_context(platform, msg, interaction_id)):
        claimed, claim = await _claim_event(platform, msg)
        if not claimed:
            logger.debug(
                "Duplicate event ignored",
                extra={"event_name": "interaction.duplicate"},
            )
            return

        _log_interaction_trigger(platform, msg)
        delay = random.uniform(MIN_REPLY_DELAY_SECONDS, MAX_REPLY_DELAY_SECONDS)
        result = await _get_interaction_scheduler().submit(
            (platform.coordination_scope, str(msg.chat.id)),
            _AdmittedInteraction(platform, msg, interaction_id),
            delay_seconds=delay,
        )
        if not result.accepted:
            if claim is not None:
                key, owner = claim
                try:
                    await get_coordination_store().delete(key, expected_value=owner)
                except Exception as error:
                    logger.warning("Failed to release rejected event claim: %s", error)
            logger.warning(
                "Interaction rejected — reason=%s pending=%d",
                result.reason,
                _get_interaction_scheduler().pending_count,
                extra={"event_name": "interaction.rejected"},
            )
        elif result.delay_applied > 0.1:
            logger.info(
                "Reply queued — delay=%.2fs",
                result.delay_applied,
                extra={"event_name": "interaction.queued"},
            )


def _interaction_id(platform: PlatformAdapter, msg: UnifiedMessage) -> str:
    raw = f"{platform.coordination_scope}|{msg.chat.id}|{msg.id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _message_log_context(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
    interaction_id: str,
) -> dict:
    return {
        "interaction_id": interaction_id,
        "platform": platform.platform_name,
        "chat_id": msg.chat.id,
        "message_id": msg.id,
        "user_id": msg.from_user.id if msg.from_user else None,
    }


async def _claim_event(
    platform: PlatformAdapter,
    msg: UnifiedMessage,
) -> tuple[bool, tuple[str, str] | None]:
    raw_key = f"{platform.coordination_scope}|{msg.chat.id}|{msg.id}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    key = f"event:{digest}"
    owner = f"{os.getpid()}:{time.monotonic_ns()}"
    try:
        claimed = await get_coordination_store().claim(
            key,
            owner,
            ttl_seconds=EVENT_DEDUP_TTL_SECONDS,
        )
        return claimed, (key, owner) if claimed else None
    except Exception as error:
        logger.warning("Event deduplication unavailable; continuing: %s", error)
        return True, None


def _get_interaction_scheduler() -> InteractionScheduler[_AdmittedInteraction]:
    global _interaction_scheduler
    if _interaction_scheduler is None:
        _interaction_scheduler = InteractionScheduler(
            _process_admitted_interaction,
            max_concurrent=MAX_CONCURRENT_INTERACTIONS,
            max_pending=MAX_PENDING_INTERACTIONS,
            per_chat_limit=PER_CHAT_QUEUE_SIZE,
            job_ttl_seconds=INTERACTION_TTL_SECONDS,
            on_error=_log_interaction_error,
            on_drop=_log_interaction_drop,
        )
    return _interaction_scheduler


async def shutdown_interaction_scheduler() -> None:
    global _interaction_scheduler, _shutting_down
    _shutting_down = True
    scheduler = _interaction_scheduler
    if scheduler is not None:
        await scheduler.close(grace_seconds=SHUTDOWN_GRACE_SECONDS)
    _interaction_scheduler = None


def _log_interaction_error(payload: _AdmittedInteraction, error: BaseException) -> None:
    msg = payload.message
    with bind_log_context(**_message_log_context(payload.platform, msg, payload.interaction_id)):
        logger.error(
            "Interaction failed: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
            extra={"event_name": "interaction.failed"},
        )


def _log_interaction_drop(payload: _AdmittedInteraction, reason: str) -> None:
    msg = payload.message
    with bind_log_context(**_message_log_context(payload.platform, msg, payload.interaction_id)):
        logger.warning(
            "Interaction dropped — reason=%s",
            reason,
            extra={"event_name": "interaction.dropped"},
        )


def _log_interaction_trigger(platform: PlatformAdapter, msg: UnifiedMessage) -> None:
    user_name = (
        (msg.from_user.username or msg.from_user.first_name)
        if msg.from_user else "unknown"
    )
    full_text = msg.text or msg.caption or ""
    text_preview = full_text.replace("\n", " ")[:LOG_CONTENT_PREVIEW_CHARS]
    media_hint = ""
    if msg.photo:
        media_hint = " [photo]"
    elif msg.voice:
        media_hint = " [voice]"
    elif msg.audio:
        media_hint = " [audio]"
    elif msg.video:
        media_hint = " [video]"
    elif msg.sticker:
        media_hint = " [sticker]"
    elif msg.document:
        media_hint = " [document]"
    logger.info(
        "Triggered — chat_name=%s user_name=%s type=%s%s text=\"%s%s\"",
        msg.chat.title or msg.chat.type,
        user_name,
        _get_interaction_type(msg).split("(")[0].strip(),
        media_hint,
        text_preview if LOG_CONTENT_PREVIEW_CHARS else "<hidden>",
        "..." if LOG_CONTENT_PREVIEW_CHARS and len(full_text) > LOG_CONTENT_PREVIEW_CHARS else "",
        extra={"event_name": "interaction.triggered"},
    )


async def _process_admitted_interaction(payload: _AdmittedInteraction) -> None:
    platform = payload.platform
    msg = payload.message
    with bind_log_context(**_message_log_context(platform, msg, payload.interaction_id)):
        reset_request_context()
        try:
            if state.IS_CHECKING_KEYS:
                return
            if msg.from_user and not await check_rate_limit_shared(
                platform.platform_name,
                msg.from_user.id,
                coordination_scope=platform.coordination_scope,
            ):
                logger.debug(
                    "User rate limit hit",
                    extra={"event_name": "interaction.rate_limited"},
                )
                return
            if msg.chat.type != "PRIVATE" and not await check_group_rate_limit_shared(
                platform.platform_name,
                msg.chat.id,
                coordination_scope=platform.coordination_scope,
            ):
                logger.debug(
                    "Group rate limit hit — allowed=%d window=%.0fs",
                    GROUP_MAX_RESPONSES_PER_WINDOW,
                    GROUP_RATE_LIMIT_WINDOW_SECONDS,
                    extra={"event_name": "interaction.rate_limited"},
                )
                return

            prompt, media_list = await prepare_prompt_and_media(platform, msg)
            recent_context_section = _get_recent_context(platform.platform_name, msg)

            if not await _passes_speculative_preflight(msg, prompt, recent_context_section):
                return

            reply_text = await _get_reply_chain_text(platform, msg)
            await _execute_frozen_message(
                platform=platform,
                msg=msg,
                prompt=prompt,
                media_list=media_list,
                reply_text=reply_text,
                style_examples=await _get_style_examples(prompt),
                social_context_section=await get_social_context(msg, reply_text),
                memory_section=await _get_memory_section(prompt, msg),
                runtime_context=await _build_runtime_context(platform, msg),
                target_instructions=_build_target_instructions(msg),
            )
        finally:
            reset_request_context()


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
        eval_ans, _ = await call_ai_provider(
            msg=msg,
            system_prompt=eval_system,
            prompt=eval_prompt,
            media_list=[],
        )
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
        answer, pending_actions = await call_ai_provider(
            msg=msg,
            system_prompt=system_prompt,
            prompt=enriched_prompt,
            media_list=media_list,
            original_prompt=prompt,
            platform=platform,
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

        # Determine whether the AI chose to skip text output.
        # Robust detection: match a standalone [SKIP] token at the very start
        # or very end of the answer, ignoring stray backticks, brackets,
        # whitespace and trailing punctuation. This catches variants like
        # "[SKIP]", "[SKIP].", "`[SKIP]`", "[skip]", "SKIP", etc. that the
        # old exact-match check silently let through and sent to the user.
        skip_text = False
        if answer:
            detected, remaining = _extract_skip_token(answer)
            if detected:
                if pending_actions:
                    logger.info(
                        "[%s] AI chose to skip text after tool action(s) — chat=%s user=%s | trigger=\"%s\"",
                        platform.platform_name,
                        msg.chat.id,
                        msg.from_user.id if msg.from_user else "?",
                        (prompt or "").replace("\n", " ")[:80],
                    )
                else:
                    logger.info(
                        "[%s] AI chose to skip — chat=%s user=%s | trigger=\"%s\"",
                        platform.platform_name,
                        msg.chat.id,
                        msg.from_user.id if msg.from_user else "?",
                        (prompt or "").replace("\n", " ")[:80],
                    )
                skip_text = True
                answer = remaining

        if not skip_text:
            # Split text on '---' for multi-message support and extract any
            # per-message [REPLY_TO:id] tags so each part can reply to a
            # different message in the chat (including an older message).
            text_messages = split_reply_messages(answer)

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

            error_answer, _ = await call_ai_provider(
                msg=msg,
                system_prompt=system_prompt,
                prompt=error_prompt,
                media_list=[],
                platform=platform,
            )

            if error_answer and error_answer.strip():
                error_messages = split_reply_messages(error_answer)
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

async def _get_style_examples(prompt: str) -> str:
    try:
        return "\n".join(await get_style_examples(prompt))
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


# Matches a standalone [SKIP] token at the very start or end of the response,
# tolerating backticks, optional brackets and surrounding punctuation
# (e.g. "[SKIP]", "[SKIP].", "`[SKIP]`", "[skip]", "SKIP").
# A trailing word boundary / leading lookbehind prevent matching embedded
# words like "skipping" or "don't skip" in the middle of a sentence.
_SKIP_TOKEN_AFTER = _re.compile(
    r"(?<![A-Za-z])`?\[?SKIP\]?,?`?[.,!?\s]*$", _re.IGNORECASE
)
_SKIP_TOKEN_BEFORE = _re.compile(
    r"^[.,!?\s]*`?\[?SKIP\b\]?,?`?", _re.IGNORECASE
)


def _extract_skip_token(answer: str) -> tuple[bool, str]:
    """Detect a [SKIP] decision in the model output.

    Returns (skip_detected, remaining_text). The token is matched at the
    very start or very end of the trimmed answer; any surrounding text is
    returned as the remainder so a combined `[SKIP] + comment` response
    still sends its real message.
    """
    text = (answer or "").strip()
    if not text:
        return False, ""

    match = _SKIP_TOKEN_AFTER.search(text)
    if match:
        remainder = text[: match.start()].strip()
    else:
        match = _SKIP_TOKEN_BEFORE.match(text)
        if not match:
            return False, text
        remainder = text[match.end():].strip()

    # If stripping the token leaves only separators/punctuation, treat the
    # whole answer as a skip decision.
    if not remainder or all(ch in "-–—`[].,!? \n" for ch in remainder):
        return True, ""
    return True, remainder


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
    for text, tag_target in messages:
        if _ACTION_META_COMMENTARY_PATTERN.search(text):
            logger.debug("Filtered action meta-commentary: %r", text[:100])
            continue
        filtered.append((text, tag_target))
    return filtered


def _start_typing(platform: PlatformAdapter, chat_id: int | str) -> asyncio.Task:
    key = (platform.platform_name, str(chat_id))

    # Cancel any existing typing task for this chat before starting a new one
    existing = _active_typing.pop(key, None)
    if existing and not existing.done():
        existing.cancel()

    start_time = time.monotonic()

    async def _loop():
        try:
            while True:
                if time.monotonic() - start_time > _MAX_TYPING_SECONDS:
                    logger.debug("Typing indicator timed out for chat=%s after %.0fs", chat_id, _MAX_TYPING_SECONDS)
                    break
                await platform.send_chat_action(chat_id, "typing")
                await asyncio.sleep(4.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing loop ended due to error: {e}")

    task = asyncio.create_task(_loop())
    _active_typing[key] = task
    return task


async def _stop_typing(platform: PlatformAdapter, chat_id: int | str, task: asyncio.Task):
    key = (platform.platform_name, str(chat_id))

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    # Remove from active tracking
    if key in _active_typing and _active_typing[key] is task:
        del _active_typing[key]

    try:
        await platform.send_chat_action(chat_id, "cancel")
    except Exception:
        pass
