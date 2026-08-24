import asyncio
import base64
import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from shin_ai.utils.action_tools import (
    ACTION_TOOL_HANDLERS,
    ACTION_TOOL_SCHEMAS,
    POST_ACTION_TOOL_REMINDER,
)
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory_lookup import MEMORY_LOOKUP_TOOL_SCHEMA, memory_lookup_tool
from shin_ai.utils.web_search import WEB_SEARCH_TOOL_SCHEMA, search_web_tool

TRANSCRIBE_AUDIO_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe_audio",
        "description": (
            "Transcribe a voice message or audio file from the chat using the configured local "
            "Whisper model (model, language='auto'-detect, and CPU thread limits come from the "
            "bot configuration).\n"
            "Voice messages and audio files are NOT automatically heard — they only appear in the "
            "conversation as '[Voice Message]' or '[Audio]' placeholders UNTIL you call this tool "
            "to transcribe them. (Images work exactly the same way via ask_gemini_about_image.)\n"
            "Call this tool on-demand, at any time — not just when explicitly asked — whenever "
            "hearing what was said would help you understand or reply to the conversation: "
            "when people discuss or react to a voice message you cannot hear, when a reply "
            "references 'what he said', when you want to join a conversation about an audio, etc.\n"
            "If message_id is omitted, transcribes the most recent voice/audio in this chat "
            "(including the message the user replied to)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": (
                        "Optional. The ID of the message containing the voice message or audio "
                        "file to transcribe. Use the (id:XXXXX) value shown next to messages "
                        "in the chat history. If omitted, the most recent voice/audio message "
                        "in this chat is transcribed."
                    ),
                }
            },
            "required": [],
        },
    },
}

# Offered only when media is attached. Gemini sees images natively but the
# system prompt advertises this tool to every provider, so it must exist for
# every provider.
ASK_GEMINI_ABOUT_IMAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_gemini_about_image",
        "description": (
            "Ask the Gemini vision model a specific question about the attached image(s): "
            "this is how you actually SEE/inspect the image content on-demand (the same way "
            "transcribe_audio lets you hear voice/audio messages). "
            "Use it to get detailed visual information, read text, identify objects/people, "
            "or verify specific details whenever the initial media description is insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to ask Gemini about the image(s).",
                }
            },
            "required": ["question"],
        },
    },
}

TOOLS = [
    WEB_SEARCH_TOOL_SCHEMA,
    MEMORY_LOOKUP_TOOL_SCHEMA,
    TRANSCRIBE_AUDIO_TOOL_SCHEMA,
    *ACTION_TOOL_SCHEMAS,
]


def tools_for_platform(tools: list[dict], platform: Any) -> list[dict]:
    """Drop tools the adapter cannot honour, and narrow the ones it partly can.

    A tool result tells the model "[ACTION EXECUTED]: the requested side-effect
    has been performed", and the system prompt then invites it to answer with
    [SKIP]. If the adapter silently drops that action the bot goes completely
    quiet, so an unsupported tool must never be offered in the first place.
    """
    if platform is None:
        return list(tools)

    allowed_actions = platform.supported_moderation_actions
    filtered: list[dict] = []
    for tool in tools:
        name = tool["function"]["name"]

        if name == "send_sticker" and not platform.supports_stickers:
            continue

        if name == "moderate_user":
            actions = [
                action
                for action in tool["function"]["parameters"]["properties"]["action"]["enum"]
                if action in allowed_actions
                and (action not in {"mute", "unmute"} or platform.supports_member_restrictions)
            ]
            if not actions:
                continue
            if actions != tool["function"]["parameters"]["properties"]["action"]["enum"]:
                tool = deepcopy(tool)
                tool["function"]["parameters"]["properties"]["action"]["enum"] = actions
            filtered.append(tool)
            continue

        filtered.append(tool)
    return filtered


async def run_tool_calling_chat(
    *,
    provider_name: str,
    create_completion: Callable[..., Any],
    system_prompt: str,
    prompt: str,
    model: str | None,
    media_list: list[dict] | None = None,
    include_raw_images: bool = False,
    tool_context: Any = None,
    max_turns: int = 3,
    turn_timeout: float = 60.0,
    **completion_kwargs: Any,
) -> tuple[str, list[dict]]:
    """Run an OpenAI-compatible chat completion loop with supported tools.

    Returns:
        (response_text, pending_actions) where pending_actions is a list of
        action dicts queued by send_reaction / send_sticker / moderate_user
        tool calls during the generation loop.
    """
    if media_list and include_raw_images:
        content: Any = [{"type": "text", "text": prompt}]
        for idx, media_info in enumerate(media_list, 1):
            image_bytes = media_info["bytes"]
            mime_type = media_info["mime_type"]
            sender = media_info["sender"]
            position = media_info["position"]
            media_type = media_info["media_type"]

            label = f"\n[Image {idx}/{len(media_list)}: {media_type} from {sender}, {position}]"
            content.append({"type": "text", "text": label})

            b64_str = base64.b64encode(image_bytes).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_str}"}})
    else:
        content = prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    platform = tool_context[0] if tool_context else None
    active_tools = tools_for_platform(TOOLS, platform)
    if media_list:
        active_tools.append(ASK_GEMINI_ABOUT_IMAGE_TOOL_SCHEMA)

    response = None
    pending_actions: list[dict] = []

    for _ in range(max_turns):
        if inspect.iscoroutinefunction(create_completion):
            response = await asyncio.wait_for(
                create_completion(
                    messages=messages,
                    model=model,
                    tools=active_tools,
                    tool_choice="auto",
                    **completion_kwargs,
                ),
                timeout=turn_timeout,
            )
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    create_completion,
                    messages=messages,
                    model=model,
                    tools=active_tools,
                    tool_choice="auto",
                    **completion_kwargs,
                ),
                timeout=turn_timeout,
            )

        if inspect.isawaitable(response):
            response = await response

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            return response_message.content or "", pending_actions

        messages.append(response_message.model_dump(exclude_unset=True))
        for tool_call in tool_calls:
            tool_result, pending_action = await _execute_tool_call(
                provider_name, tool_call, media_list, tool_context
            )
            if pending_action is not None:
                pending_actions.append(pending_action)
                tool_result += POST_ACTION_TOOL_REMINDER
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

    if response is None:
        return "", pending_actions
    return response.choices[0].message.content or "", pending_actions


def _coerce_message_id(raw_id: Any) -> int | str:
    """Convert a tool-provided message id to the platform-native type when possible."""
    raw_str = str(raw_id).strip()
    try:
        return int(raw_str)
    except (TypeError, ValueError):
        return raw_str


def _has_audio(msg: Any) -> bool:
    return bool(getattr(msg, "voice", None) or getattr(msg, "audio", None))


def _find_audio_on_reply_chain(msg: Any) -> Any | None:
    """Walk up the reply chain looking for a voice/audio message."""
    curr = msg
    for _ in range(10):
        if curr is None:
            break
        if _has_audio(curr):
            return curr
        curr = getattr(curr, "reply_to_message", None)
    return None


async def _transcribe_audio_target(
    platform: Any,
    requested_msg: Any,
    target_msg: Any,
    request_desc: str,
) -> str:
    """Download + transcribe the voice/audio on target_msg via the Whisper service."""
    media_handle = target_msg.voice or target_msg.audio
    from shin_ai.services.audio_transcriber import transcribe_audio_source

    transcription = await transcribe_audio_source(
        lambda: platform.download_media(media_handle),
        media_handle.mime_type or "audio/ogg",
    )
    if not transcription:
        return "Transcription produced no text (inaudible or transcription failed)."

    media_type = "Voice message" if target_msg.voice else "Audio file"
    sender_name = "Unknown"
    if getattr(target_msg, "from_user", None):
        sender_name = target_msg.from_user.username or target_msg.from_user.first_name or "Unknown"

    source_note = ""
    if target_msg is not requested_msg:
        source_note = (
            f" (audio source: message id '{getattr(target_msg, 'id', '?')}', found via the reply chain)"
        )

    logger.info(
        "[Audio Tool] Transcribed %s from %s (%s) → %d chars",
        media_type.lower(),
        sender_name,
        request_desc,
        len(transcription),
    )
    return (
        f"[{media_type} from {sender_name}{source_note} — Whisper transcription]:\n"
        f'"{transcription}"\n\n'
        "[TRANSCRIPTION NOTE: Transcribed with the configured local Whisper model. "
        "It may contain phonetic spelling errors, hallucinated artifacts, or illogical "
        "words due to dialect variations (especially Egyptian Arabic). Intelligently "
        "interpret any illogical words based on the surrounding context to find the "
        "nearest logical meaning.]"
    )


async def _transcribe_chat_audio(
    platform: Any,
    msg: Any,
    provider_name: str,
    message_id: Any = None,
) -> str:
    """Resolve which message to transcribe and run Whisper on it.

    Priority:
      1. The message identified by message_id (or audio found in its reply chain).
      2. The triggering message / its reply chain.
      3. The most recent voice/audio message in the chat's short-term context.
    """
    fallback_msg = _find_audio_on_reply_chain(msg)

    if message_id is not None:
        fetched = await platform.get_message(msg.chat.id, _coerce_message_id(message_id))
        if fetched is None:
            return f"No message found with ID '{message_id}'."
        target = fetched if _has_audio(fetched) else _find_audio_on_reply_chain(fetched)
        if target is None:
            return f"Message '{message_id}' (and its reply chain) contains no voice message or audio file."
        return await _transcribe_audio_target(platform, fetched, target, f"message_id={message_id}")

    if fallback_msg is not None:
        return await _transcribe_audio_target(platform, msg, fallback_msg, "reply chain / current message")

    from shin_ai.utils.context_manager import get_recent_audio_messages

    recent_audio = get_recent_audio_messages(platform.platform_name, msg.chat.id, max_count=5)
    for entry in recent_audio:
        if str(entry["msg_id"]) == str(getattr(msg, "id", "")):
            continue
        fetched = await platform.get_message(msg.chat.id, entry["msg_id"])
        if fetched is not None and _has_audio(fetched):
            return await _transcribe_audio_target(
                platform, fetched, fetched, f"recent context msg {entry['msg_id']}"
            )

    return "No voice message or audio file found in this conversation or recent chat history."


async def ask_gemini_about_image(question: str, media_list: list[dict] | None) -> str:
    """Answer a targeted question about the attached image(s) via Gemini."""
    logger.debug("Gemini image question: %r", question)
    if not media_list:
        return "No image is currently attached to this conversation context."

    from shin_ai.providers.gemini import gemini_api

    try:
        answer, _ = await gemini_api(
            "You are an assistant that answers specific questions about attached media/images. "
            "Answer the user's question accurately, concisely, and factually based on the visual content.",
            question,
            media_list=media_list,
            # This call *is* the image tool. Re-declaring it here would let the
            # nested model call the tool that invoked it.
            allow_image_tool=False,
        )
        return answer
    except Exception as e:
        logger.error("Failed to ask Gemini about the image: %s", e)
        return f"Error querying Gemini about the image: {e!s}"


async def _execute_tool_call(
    provider_name: str,
    tool_call: Any,
    media_list: list[dict] | None = None,
    tool_context: Any = None,
) -> tuple[str, dict | None]:
    """Execute a tool call and return (result_str, pending_action_or_None)."""
    tool_name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        args = {}

    logger.info(
        "Tool requested — provider=%s tool=%s",
        provider_name,
        tool_name,
        extra={"event_name": "tool.requested"},
    )
    logger.debug("Tool arguments — provider=%s tool=%s args=%r", provider_name, tool_name, args)

    if tool_name == "search_web_tool":
        query = args.get("query", "")
        result = await search_web_tool(query)
        return result, None

    if tool_name == "memory_lookup_tool":
        result = await memory_lookup_tool(**args)
        return result, None

    if tool_name == "transcribe_audio":
        message_id = args.get("message_id")
        suffix = f" for message_id='{message_id}'" if message_id is not None else " (latest audio in chat)"
        logger.debug("Audio transcription target%s", suffix)
        if tool_context is None:
            return "Audio transcription is unavailable in this context (no chat/platform bound).", None
        platform, msg = tool_context
        try:
            result = await _transcribe_chat_audio(platform, msg, provider_name, message_id)
            return result, None
        except Exception as e:
            logger.error("Tool transcribe_audio failed: %s", e, exc_info=True)
            return f"Error transcribing audio: {e!s}", None

    if tool_name == "ask_gemini_about_image":
        return await ask_gemini_about_image(args.get("question", ""), media_list), None

    handler = ACTION_TOOL_HANDLERS.get(tool_name)
    if handler:
        result_str, pending_action = await handler(args)
        return result_str, pending_action

    logger.warning("%s requested unknown tool: %s", provider_name, tool_name)
    return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False), None
