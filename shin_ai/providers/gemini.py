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
import asyncio


MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


def _is_model_on_cooldown(model: str) -> bool:
    return time.time() < MODEL_COOLDOWN_UNTIL.get(model, 0.0)


def _set_model_cooldown(model: str, seconds: int = 3600) -> None:
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


async def gemini_api(system_prompt, prompt, media_list=None)  -> str:
    models_to_try = list(MODELS_LIST)

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
                response = await _run_gemini_generation_loop(genai_client, model, contents, config)

                response_text = _extract_gemini_text(response)
                if not response_text:
                    logger.warning(
                        f"Gemini response had no text content (model: {model}, Key: {key_name})"
                    )
                    continue

                logger.info(f"Gemini API call successful (model: {model}, Key: {key_name})")
                update_key_status(key_name, "active", model)
                
                _rotate_key_to_back(key_name)
                    
                return response_text
            except asyncio.CancelledError:
                if _rotate_key_to_back(key_name):
                    logger.warning(f"Gemini timed out/cancelled (model: {model}, Key: {key_name}). Rotating key.")
                raise
            except Exception as e:
                failed_keys_count += 1
                _rotate_key_to_back(key_name)

                logger.warning(
                    f"Gemini API key failed (model: {model}, Key: {key_name}, Failed Count: {failed_keys_count}): {e}"
                )

                if "you exceeded your current quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Gemini API key quota exceeded for model {model} (Key: {key_name}, Failed Count: {failed_keys_count})")
                    update_key_status(key_name, "exhausted", model, e)
                elif "503" in str(e):
                    logger.warning(f"Gemini API model {model} is temporarily unavailable (503). Switching model.")
                    update_key_status(key_name, "unavailable", model, e)
                    break
                else:
                    logger.error(f"Error with Gemini API (model: {model}, Key: {key_name}): {e}")
                    update_key_status(key_name, "error", model, e)
                continue

        logger.warning(
            f"Model {model} failed for all keys. Failed keys: {failed_keys_count}. "
            "Trying next available model."
        )
        if len(models_to_try) > 1:
            _set_model_cooldown(model)

    return ""


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

    logger.info(f"Added {len(media_list)} media items to Gemini request")
    return contents


def _build_gemini_config(system_prompt: str, model: str):
    thinking_config = genai.types.ThinkingConfig(thinking_level="high") if "gemini-3" in model else None
    return genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[search_web_tool, memory_lookup_tool],
        thinking_config=thinking_config,
    )


async def _run_gemini_generation_loop(genai_client, model: str, contents: list, config) -> object:
    max_turns = 3
    current_turn = 0
    response = None

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
            await _append_gemini_tool_response(contents, fn_call)

        current_turn += 1

    return response


async def _append_gemini_tool_response(contents: list, fn_call) -> None:
    if fn_call.name == "search_web_tool":
        query = fn_call.args.get("query", "")
        logger.info(f"Gemini requested web search for: '{query}'")
        tool_result_str = await search_web_tool(query)
    elif fn_call.name == "memory_lookup_tool":
        args = dict(fn_call.args) if fn_call.args else {}
        logger.info(f"Gemini requested memory lookup with args: {args}")
        tool_result_str = await memory_lookup_tool(**args)
    else:
        logger.warning(f"Gemini requested unknown tool: {fn_call.name}")
        tool_result_str = f"Unknown tool: {fn_call.name}"

    tool_part = genai.types.Part.from_function_response(
        name=fn_call.name,
        response={"result": tool_result_str},
    )
    contents.append(genai.types.Content(role="user", parts=[tool_part]))


def _rotate_key_to_back(key_name: str) -> bool:
    if key_name not in API_KEYS_MAP:
        return False

    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
    save_keys(API_KEYS_MAP)
    return True
