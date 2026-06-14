"""
Gemini AI Provider

Handles API calls to Google's Gemini models with key rotation and statistics.
"""
import time

from google import genai

from shin_ai.providers.gemini_keys import (
    API_KEYS_MAP,
    MODELS_LIST,
    get_gemini_stats_message,
    save_keys,
    update_key_status,
)
from shin_ai.utils.logger_config import logger
from shin_ai.utils.web_search import search_web_tool
from shin_ai.utils.memory_lookup import memory_lookup_tool
from shin_ai.utils.action_tools import ACTION_TOOL_HANDLERS
import asyncio


MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


def _is_model_on_cooldown(model: str) -> bool:
    return time.time() < MODEL_COOLDOWN_UNTIL.get(model, 0.0)


def _set_model_cooldown(model: str, seconds: int = 900) -> None:
    MODEL_COOLDOWN_UNTIL[model] = time.time() + seconds


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


async def gemini_api(system_prompt, prompt, media_list=None) -> tuple[str, list[dict]]:
    """Call the Gemini API with tool support.

    Returns:
        (response_text, pending_actions) where pending_actions is a list of
        action dicts queued by send_reaction / send_sticker / moderate_user
        tool calls during the generation loop.
    """
    models_to_try = list(MODELS_LIST)
    last_exception = None

    for model in models_to_try:
        failed_keys_count = 0
        if _is_model_on_cooldown(model):
            logger.warning(f"Model {model} is on cooldown. Skipping.")
            continue
        # Create a list of items to iterate over, preserving the current order
        for key_name, api_key in list(API_KEYS_MAP.items()):
            if not api_key:
                continue

            try:
                genai_client = genai.Client(api_key=api_key)
                contents = _build_gemini_contents(prompt, media_list)
                config = _build_gemini_config(system_prompt, model)
                response, pending_actions = await _run_gemini_generation_loop(genai_client, model, contents, config)

                response_text = _extract_gemini_text(response)
                if not response_text:
                    logger.warning(
                        f"Gemini response had no text content (model: {model}, Key: {key_name})"
                    )
                    continue

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

                update_key_status(key_name, "active", model)
                
                _rotate_key_to_back(key_name)
                    
                return response_text, pending_actions
            except asyncio.CancelledError:
                if _rotate_key_to_back(key_name):
                    logger.warning(f"Gemini timed out/cancelled (model: {model}, Key: {key_name}). Rotating key.")
                raise
            except Exception as e:
                last_exception = e
                failed_keys_count += 1
                _rotate_key_to_back(key_name)

                logger.warning(
                    f"Gemini API key failed (model: {model}, Key: {key_name}, Failed Count: {failed_keys_count})"
                )

                if "you exceeded your current quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Gemini API key quota exceeded for model {model} (Key: {key_name}, Failed Count: {failed_keys_count})")
                    update_key_status(key_name, "exhausted", model, e)
                elif "503" in str(e):
                    logger.warning(f"Gemini API model {model} is temporarily unavailable (503). Switching model.")
                    update_key_status(key_name, "unavailable", model, e)
                    break
                else:
                    logger.error("Error with Gemini API (model: %s, Key: %s): %s", model, key_name, e, exc_info=True)
                    update_key_status(key_name, "error", model, e)
                continue

        logger.warning(
            f"Model {model} failed for all keys. Failed keys: {failed_keys_count}. "
            "Trying next available model."
        )
        available_models = [m for m in models_to_try if not _is_model_on_cooldown(m)]
        if len(available_models) > 1:
            _set_model_cooldown(model)

    if last_exception:
        raise last_exception
    return "", []


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

    # Build Gemini-native tool declarations from the OpenAI schemas
    gemini_tools = [
        search_web_tool,
        memory_lookup_tool,
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
    genai_client, model: str, contents: list, config
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
            tool_result_str, pending_action = await _dispatch_gemini_tool(fn_call)
            if pending_action is not None:
                pending_actions.append(pending_action)
            tool_part = genai.types.Part.from_function_response(
                name=fn_call.name,
                response={"result": tool_result_str},
            )
            contents.append(genai.types.Content(role="user", parts=[tool_part]))

        current_turn += 1

    return response, pending_actions


async def _dispatch_gemini_tool(fn_call) -> tuple[str, dict | None]:
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

    handler = ACTION_TOOL_HANDLERS.get(fn_call.name)
    if handler:
        logger.info("Gemini → action tool %r: %s", fn_call.name, args)
        return await handler(args)

    logger.warning("Gemini → unknown tool requested: %r", fn_call.name)
    return f"Unknown tool: {fn_call.name}", None


def _rotate_key_to_back(key_name: str) -> bool:
    if key_name not in API_KEYS_MAP:
        return False

    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
    save_keys(API_KEYS_MAP)
    return True
