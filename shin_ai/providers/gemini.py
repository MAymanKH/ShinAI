"""
Gemini AI Provider

Handles API calls to Google's Gemini models with key rotation and statistics.
"""
from google import genai
from shin_ai.config import GEMINI_MODEL, DATA_DIR
from shin_ai.utils.logger_config import logger
from shin_ai.utils.web_search import search_web_tool
from shin_ai.utils.memory_lookup import memory_lookup_tool
import json
import os
import time
import asyncio
import re
from datetime import datetime

# File paths for key management
GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"
STATS_FILE = DATA_DIR / "gemini_stats.json"


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


def load_keys() -> dict[str, str]:
    """Load API keys from JSON file or environment variables."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # If the JSON file exists, load keys from it (including order)
    if GEMINI_KEYS_FILE.exists():
        try:
            with open(GEMINI_KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load keys from {GEMINI_KEYS_FILE}: {e}")
    
    # Fallback/Migration: Load from environment variables if JSON doesn't exist
    logger.info("Initializing gemini_keys.json from environment variables...")
    keys_map = {}
    for i in range(1, 40):  # Check up to 40 just in case
        key_name = f"GEMINI_API_KEY{i}"
        api_key = os.getenv(key_name)
        if api_key:
            keys_map[key_name] = api_key
    
    # Save the initialized keys to the file
    if keys_map:
        save_keys(keys_map)
        
    return keys_map


def save_keys(current_map: dict[str, str]) -> None:
    """Save API keys to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(GEMINI_KEYS_FILE, "w") as f:
            json.dump(current_map, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save keys to {GEMINI_KEYS_FILE}: {e}")


def load_stats() -> dict:
    """Load key statistics from JSON file."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_stats(stats: dict) -> None:
    """Save key statistics to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save stats to {STATS_FILE}: {e}")

def update_key_status(key_name, status, model=None, error_msg=None):
    if not model: return
    stats = load_stats()
    
    # Initialize key dict if missing
    if key_name not in stats:
        stats[key_name] = {}
        
    # Check for legacy format (flat dict with "status") and wipe/migrate if found
    if "status" in stats[key_name]:
        # Migration: preserve old data under its model if it exists
        old_data = stats[key_name]
        stats[key_name] = {}
        if old_data.get("model"):
            stats[key_name][old_data["model"]] = old_data

    stats[key_name][model] = {
        "status": status,
        "last_updated": time.time(),
        "last_updated_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": str(error_msg) if error_msg else None
    }
    save_stats(stats)

async def check_single_key_status(key_name, api_key, model):
    try:
        client = genai.Client(api_key=api_key)
        # Use a minimal generation prompt to test real availability (models.get doesn't catch all quota issues)
        await client.aio.models.generate_content(
            model=model, 
            contents="a", # Minimal prompt
            config=genai.types.GenerateContentConfig(max_output_tokens=1)
        )
        logger.info(f"Key status check [{model}]: {key_name} is ACTIVE")
        return {"key": key_name, "model": model, "status": "active", "error": None}
    except Exception as e:
        error_msg = str(e)
        status = "error"
        if "quota" in error_msg.lower() or "429" in error_msg:
            status = "exhausted"
        logger.warning(f"Key status check [{model}]: {key_name} is {status.upper()} - {error_msg}")
        return {"key": key_name, "model": model, "status": status, "error": error_msg}

async def get_gemini_stats_message(detailed=False):
    keys = API_KEYS_MAP # Current keys
    total_keys = len(keys)
    current_models = list(MODELS_LIST)
    
    # Run checks in parallel
    tasks = []
    for model in current_models:
        for key_name in sorted(keys.keys()):
            api_key = keys.get(key_name)
            if api_key:
                tasks.append(check_single_key_status(key_name, api_key, model))
    
    # Wait for all checks to complete
    results = await asyncio.gather(*tasks)
    
    # Group results by model
    model_results = {model: [] for model in current_models}
    for res in results:
        if res["model"] in model_results:
            model_results[res["model"]].append(res)
    
    report_lines = ["**Gemini Key Statistics (Live Check)**"]
    
    for model in current_models:
        active = 0
        exhausted = 0
        error = 0
        details = []
        
        # Iterating over the results we just collected
        for res in model_results[model]:
            key_name = res["key"]
            status = res["status"]
            
            if status == "active":
                active += 1
            elif status == "exhausted":
                exhausted += 1
                details.append(f"• {key_name}: ❌ Exhausted")
            else:
                error += 1
                err_msg = res.get("error", "Unknown error")
                details.append(f"• {key_name}: ⚠️ Error: {err_msg[:20]}...")

        available_count = active
        percentage_left = (available_count / total_keys) * 100 if total_keys > 0 else 0.0
        
        section = f"""
**Model: {model}**
Health: {percentage_left:.1f}% Available
✅ Active: {active}
❌ Exhausted: {exhausted} | ⚠️ Errors: {error}"""
        
        if detailed and details:
            section += "\nIssues:\n" + "\n".join(details)
            
        report_lines.append(section)

    return "\n".join(report_lines)

API_KEYS_MAP = load_keys()


MODELS_LIST = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash"
]

async def gemini_api(system_prompt, prompt, media_list=None)  -> str:
    failed_keys_count = 0
    models_to_try = list(MODELS_LIST)

    for model in models_to_try:
        # Create a list of items to iterate over, preserving the current order
        for key_name, api_key in list(API_KEYS_MAP.items()):
            if not api_key:
                continue

            try:
                genai_client = genai.Client(api_key=api_key)
                contents = [prompt]
                
                if media_list:
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

                thinking_config = genai.types.ThinkingConfig(thinking_level="high") if "gemini-3" in model else None
                config = genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[search_web_tool, memory_lookup_tool],
                    thinking_config=thinking_config
                )

                max_turns = 3
                current_turn = 0
                response = None
                
                while current_turn < max_turns:
                    response = await genai_client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                    
                    if response.function_calls:
                        contents.append(response.candidates[0].content)
                        
                        for fn_call in response.function_calls:
                            if fn_call.name == "search_web_tool":
                                query = fn_call.args.get("query", "")
                                logger.info(f"Gemini requested web search for: '{query}'")
                                tool_result_str = await search_web_tool(query)
                                
                                tool_part = genai.types.Part.from_function_response(
                                    name="search_web_tool",
                                    response={"result": tool_result_str}
                                )
                                contents.append(genai.types.Content(role="user", parts=[tool_part]))
                            elif fn_call.name == "memory_lookup_tool":
                                args = dict(fn_call.args) if fn_call.args else {}
                                logger.info(f"Gemini requested memory lookup with args: {args}")
                                tool_result_str = await memory_lookup_tool(**args)
                                
                                tool_part = genai.types.Part.from_function_response(
                                    name="memory_lookup_tool",
                                    response={"result": tool_result_str}
                                )
                                contents.append(genai.types.Content(role="user", parts=[tool_part]))
                        current_turn += 1
                        continue
                    else:
                        break

                response_text = _extract_gemini_text(response)
                if not response_text:
                    logger.warning(
                        f"Gemini response had no text content (model: {model}, Key: {key_name})"
                    )
                    continue

                logger.info(f"Gemini API call successful (model: {model}, Key: {key_name})")
                update_key_status(key_name, "active", model)
                
                if key_name in API_KEYS_MAP:
                    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
                    save_keys(API_KEYS_MAP)
                    
                return response_text
            except asyncio.CancelledError:
                if key_name in API_KEYS_MAP:
                    logger.warning(f"Gemini timed out/cancelled (model: {model}, Key: {key_name}). Rotating key.")
                    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
                    save_keys(API_KEYS_MAP)
                raise
            except Exception as e:
                failed_keys_count += 1
                if key_name in API_KEYS_MAP:
                    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
                    save_keys(API_KEYS_MAP)

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

        logger.warning(f"Model {model} failed for all keys. Trying next available model.")

    return ""