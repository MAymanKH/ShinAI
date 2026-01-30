"""
Gemini AI Provider

Handles API calls to Google's Gemini models with key rotation and statistics.
"""
from google import genai
from shin_ai.config import GEMINI_MODEL, DATA_DIR
from shin_ai.utils.logger_config import logger
import json
import os
import time
import asyncio
from datetime import datetime

# File paths for key management
GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"
STATS_FILE = DATA_DIR / "gemini_stats.json"


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

async def gemini_api(system_prompt, prompt, image_bytes=None, mime_type=None)  -> str:
    failed_keys_count = 0

    for model in list(MODELS_LIST):
        # Create a list of items to iterate over, preserving the current order
        for key_name, api_key in list(API_KEYS_MAP.items()):
            if not api_key:
                continue

            try:
                genai_client = genai.Client(api_key=api_key)
                # Prepare contents
                contents = [prompt]
                if image_bytes and mime_type:
                    contents.append(
                        genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    )

                config = genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())]
                )

                response = await genai_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                logger.info(f"Gemini API call successful (model: {model}, Key: {key_name})")
                update_key_status(key_name, "active", model)
                return response.text
            except Exception as e:
                failed_keys_count += 1
                # Move the failed key to the end of the dictionary
                if key_name in API_KEYS_MAP:
                    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
                    save_keys(API_KEYS_MAP)

                if "you exceeded your current quota, please check your plan and billing details" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Gemini API key quota exceeded for model {model} (Key: {key_name}, Failed Count: {failed_keys_count})")
                    update_key_status(key_name, "exhausted", model, e)
                else:
                    logger.error(f"Error with Gemini API (model: {model}, Key: {key_name}, Failed Count: {failed_keys_count}): {e}")
                    update_key_status(key_name, "error", model, e)
                continue

        logger.warning(f"Model {model} failed for all keys. Rotating to end of list.")
        if model in MODELS_LIST:
            MODELS_LIST.append(MODELS_LIST.pop(MODELS_LIST.index(model)))

    return ""