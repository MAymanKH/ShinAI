"""
Gemini AI Provider

Handles API calls to Google's Gemini models with key rotation and statistics.
"""
from google import genai

from shin_ai.providers.gemini_keys import (
    API_KEYS_MAP,
    MODELS_LIST,
    get_gemini_stats_message,
)
from shin_ai.coordination.runtime import get_coordination_store
from shin_ai.providers.gemini_errors import (
    GeminiFailure,
    GeminiFailureKind,
    classify_gemini_error,
)
from shin_ai.providers.gemini_scheduler import GeminiScheduler
from shin_ai.utils.logger_config import logger
from shin_ai.utils.web_search import search_web_tool
from shin_ai.utils.memory_lookup import memory_lookup_tool
from shin_ai.utils.action_tools import ACTION_TOOL_HANDLERS
import asyncio


# Injected into tool results for side-effect actions (reaction / sticker /
# moderation) so the model never forgets it may answer with text OR [SKIP].
_POST_ACTION_TOOL_REMINDER = (
    "\n\n[ACTION EXECUTED]: The requested side-effect has been performed. "
    "Now decide the text channel:\n"
    "  • Output [SKIP] if the action itself (reaction / sticker / moderation) IS the complete response.\n"
    "  • Write your normal text reply if words are also needed.\n"
    "  • Output [SKIP] and then write text if you want both.\n"
    "  • (For stickers) You may also call send_sticker again to send another sticker."
)

# Cache genai.Client instances per API key to avoid recreating connections
_genai_client_cache: dict[str, genai.Client] = {}
_gemini_scheduler: GeminiScheduler | None = None


def _get_genai_client(api_key: str) -> genai.Client:
    """Return a cached genai.Client for the given API key."""
    if api_key not in _genai_client_cache:
        _genai_client_cache[api_key] = genai.Client(api_key=api_key)
    return _genai_client_cache[api_key]


def get_gemini_scheduler() -> GeminiScheduler:
    global _gemini_scheduler
    if _gemini_scheduler is None:
        _gemini_scheduler = GeminiScheduler(
            API_KEYS_MAP,
            MODELS_LIST,
            get_coordination_store(),
        )
    return _gemini_scheduler


def _extract_gemini_text(response) -> str:
    """Extract text from Gemini response, including candidate parts fallback."""
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    candidates = getattr(response, "candidates", None) or []
    collected_parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue

        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                collected_parts.append(part_text.strip())

    return "\n".join(collected_parts).strip()


async def gemini_api(
    system_prompt,
    prompt,
    media_list=None,
    tool_context=None,
    *,
    scheduler: GeminiScheduler | None = None,
) -> tuple[str, list[dict]]:
    """Call the Gemini API with tool support.

    Args:
        system_prompt: The static system prompt.
        prompt:        The user prompt (may include dynamic context).
        media_list:    Optional list of media dicts (images) to attach.
        tool_context:  Optional (platform, triggering_msg) tuple that gives
                       context-bound tools (e.g. transcribe_audio) access to
                       the current chat.

    Returns:
        (response_text, pending_actions) where pending_actions is a list of
        action dicts queued by send_reaction / send_sticker / moderate_user
        tool calls during the generation loop.
    """
    active_scheduler = scheduler or get_gemini_scheduler()
    models_to_try = list(active_scheduler.models)
    last_exception = None

    for model in models_to_try:
        tried_keys: set[str] = set()
        while len(tried_keys) < len(active_scheduler.keys):
            reservation = await active_scheduler.reserve(model, excluded_keys=tried_keys)
            if reservation is None:
                break
            tried_keys.add(reservation.key_name)
            try:
                genai_client = _get_genai_client(reservation.api_key)
                contents = _build_gemini_contents(prompt, media_list)
                config = _build_gemini_config(system_prompt, model)
                response, pending_actions = await _run_gemini_generation_loop(
                    genai_client,
                    model,
                    contents,
                    config,
                    tool_context,
                )

                response_text = _extract_gemini_text(response)
                if not response_text and not pending_actions:
                    raise RuntimeError(
                        "Gemini response contained neither text nor pending actions"
                    )

                # Log cache hit stats — only when there is an actual cache hit
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    cached = getattr(usage, "cached_content_token_count", 0) or 0
                    total_in = getattr(usage, "prompt_token_count", 0) or 0
                    if cached > 0 and total_in:
                        pct = cached / total_in * 100
                        logger.info(
                            "Gemini cache hit: %d/%d input tokens cached (%.0f%%) — model=%s",
                            cached, total_in, pct, model,
                        )

                await reservation.succeeded()
                logger.info(
                    "Gemini pair succeeded — model=%s key=%s",
                    model,
                    reservation.key_name,
                )
                return response_text, pending_actions
            except asyncio.CancelledError:
                await reservation.failed(
                    GeminiFailure(
                        kind=GeminiFailureKind.TIMEOUT,
                        status_code=None,
                        retry_after_seconds=None,
                        message="request cancelled or outer deadline exceeded",
                    )
                )
                raise
            except Exception as e:
                last_exception = e
                failure = classify_gemini_error(e)
                await reservation.failed(failure)
                logger.warning(
                    "Gemini pair failed — model=%s key=%s kind=%s status=%s error=%s",
                    model,
                    reservation.key_name,
                    failure.kind.value,
                    failure.status_code,
                    failure.message,
                )
                continue

    if last_exception:
        raise last_exception
    raise RuntimeError("No eligible Gemini key/model pair is currently available")


def _build_gemini_contents(prompt: str, media_list=None) -> list:
    contents = [prompt]

    if not media_list:
        return contents

    for idx, media_info in enumerate(media_list, 1):
        image_bytes = media_info['bytes']
        mime_type = media_info['mime_type']
        sender = media_info['sender']
        position = media_info['position']
        media_type = media_info['media_type']

        label = f"\n[Image {idx}/{len(media_list)}: {media_type} from {sender}, {position}]"
        contents.append(label)
        contents.append(
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )

    logger.debug("Added %d media item(s) to Gemini request", len(media_list))
    return contents


def _build_gemini_config(system_prompt: str, model: str):
    from shin_ai.utils.action_tools import (
        SEND_REACTION_TOOL_SCHEMA,
        SEND_STICKER_TOOL_SCHEMA,
        MODERATE_USER_TOOL_SCHEMA,
    )
    from shin_ai.providers.tool_loop import TRANSCRIBE_AUDIO_TOOL_SCHEMA

    # Build Gemini-native tool declarations from the OpenAI schemas
    gemini_tools = [
        search_web_tool,
        memory_lookup_tool,
        _openai_schema_to_gemini_function(TRANSCRIBE_AUDIO_TOOL_SCHEMA),
        _openai_schema_to_gemini_function(SEND_REACTION_TOOL_SCHEMA),
        _openai_schema_to_gemini_function(SEND_STICKER_TOOL_SCHEMA),
        _openai_schema_to_gemini_function(MODERATE_USER_TOOL_SCHEMA),
    ]

    thinking_config = genai.types.ThinkingConfig(thinking_level="high") if "gemini-3" in model else None
    return genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=gemini_tools,
        thinking_config=thinking_config,
    )


def _openai_schema_to_gemini_function(schema: dict):
    """Convert an OpenAI function-calling schema to a Gemini FunctionDeclaration."""
    fn = schema["function"]
    params = fn.get("parameters", {})

    return genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        k: _param_to_gemini_schema(v)
                        for k, v in params.get("properties", {}).items()
                    },
                    required=params.get("required", []),
                ),
            )
        ]
    )


def _param_to_gemini_schema(param: dict):
    """Convert a single OpenAI parameter dict to a Gemini Schema."""
    type_map = {
        "string": genai.types.Type.STRING,
        "integer": genai.types.Type.INTEGER,
        "number": genai.types.Type.NUMBER,
        "boolean": genai.types.Type.BOOLEAN,
        "array": genai.types.Type.ARRAY,
        "object": genai.types.Type.OBJECT,
    }
    t = type_map.get(param.get("type", "string"), genai.types.Type.STRING)
    kwargs = {
        "type": t,
        "description": param.get("description", ""),
    }
    if "enum" in param:
        kwargs["enum"] = param["enum"]
    if t == genai.types.Type.ARRAY and "items" in param:
        kwargs["items"] = _param_to_gemini_schema(param["items"])
    return genai.types.Schema(**kwargs)


async def _run_gemini_generation_loop(
    genai_client, model: str, contents: list, config, tool_context=None
) -> tuple[object, list[dict]]:
    max_turns = 3
    current_turn = 0
    response = None
    pending_actions: list[dict] = []

    while current_turn < max_turns:
        response = await genai_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        if not response.function_calls:
            break

        contents.append(response.candidates[0].content)
        for fn_call in response.function_calls:
            tool_result_str, pending_action = await _dispatch_gemini_tool(fn_call, tool_context)
            if pending_action is not None:
                pending_actions.append(pending_action)
                tool_result_str += _POST_ACTION_TOOL_REMINDER
            tool_part = genai.types.Part.from_function_response(
                name=fn_call.name,
                response={"result": tool_result_str},
            )
            contents.append(genai.types.Content(role="user", parts=[tool_part]))

        current_turn += 1

    return response, pending_actions


async def _dispatch_gemini_tool(fn_call, tool_context=None) -> tuple[str, dict | None]:
    """Dispatch a Gemini function call to the appropriate handler.

    Returns:
        (tool_result_str, pending_action_or_None)
    """
    args = dict(fn_call.args) if fn_call.args else {}

    if fn_call.name == "search_web_tool":
        query = args.get("query", "")
        logger.info("Gemini → web search: %r", query)
        return await search_web_tool(query), None

    if fn_call.name == "memory_lookup_tool":
        logger.info("Gemini → memory lookup: %s", args)
        return await memory_lookup_tool(**args), None

    if fn_call.name == "transcribe_audio":
        message_id = args.get("message_id")
        suffix = f" for message_id='{message_id}'" if message_id is not None else " (latest audio in chat)"
        logger.info("Gemini → audio transcription%s", suffix)
        if tool_context is None:
            return "Audio transcription is unavailable in this context (no chat/platform bound).", None
        platform, msg = tool_context
        from shin_ai.providers.tool_loop import _transcribe_chat_audio
        try:
            return await _transcribe_chat_audio(platform, msg, "Gemini", message_id), None
        except Exception as e:
            logger.error("Tool transcribe_audio failed: %s", e, exc_info=True)
            return f"Error transcribing audio: {str(e)}", None

    handler = ACTION_TOOL_HANDLERS.get(fn_call.name)
    if handler:
        logger.info("Gemini → action tool %r: %s", fn_call.name, args)
        return await handler(args)

    logger.warning("Gemini → unknown tool requested: %r", fn_call.name)
    return f"Unknown tool: {fn_call.name}", None
