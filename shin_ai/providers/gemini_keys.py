import asyncio
import json
import time
from datetime import datetime

from google import genai

from shin_ai.config import DATA_DIR, GEMINI_MODELS
from shin_ai.utils.logger_config import logger


GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"
STATS_FILE = DATA_DIR / "gemini_stats.json"

MODELS_LIST = GEMINI_MODELS


def load_keys() -> dict[str, str]:
    """Load API keys from data/gemini_keys.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if GEMINI_KEYS_FILE.exists():
        try:
            with open(GEMINI_KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load keys from %s: %s", GEMINI_KEYS_FILE, e, exc_info=True)
            return {}

    # No keys file found — create an empty one and instruct the user
    logger.warning(
        f"{GEMINI_KEYS_FILE} not found. "
        "Please add your Gemini API keys to data/gemini_keys.json in the format: "
        '{"GEMINI_API_KEY1": "AIza...", "GEMINI_API_KEY2": "AIza..."}'
    )
    save_keys({})
    return {}


def save_keys(current_map: dict[str, str]) -> None:
    """Save API keys to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(GEMINI_KEYS_FILE, "w") as f:
            json.dump(current_map, f, indent=4)
    except Exception as e:
        logger.error("Failed to save keys to %s: %s", GEMINI_KEYS_FILE, e, exc_info=True)


def load_stats() -> dict:
    """Load key statistics from JSON file."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_stats(stats: dict) -> None:
    """Save key statistics to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        logger.error("Failed to save stats to %s: %s", STATS_FILE, e, exc_info=True)


def update_key_status(key_name, status, model=None, error_msg=None):
    if not model:
        return

    stats = load_stats()

    if key_name not in stats:
        stats[key_name] = {}

    if "status" in stats[key_name]:
        old_data = stats[key_name]
        stats[key_name] = {}
        if old_data.get("model"):
            stats[key_name][old_data["model"]] = old_data

    stats[key_name][model] = {
        "status": status,
        "last_updated": time.time(),
        "last_updated_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": str(error_msg) if error_msg else None,
    }
    save_stats(stats)


async def check_single_key_status(key_name, api_key, model):
    try:
        client = genai.Client(api_key=api_key)
        await client.aio.models.generate_content(
            model=model,
            contents="a",
            config=genai.types.GenerateContentConfig(max_output_tokens=1),
        )
        logger.debug("Key status check [%s]: %s is ACTIVE", model, key_name)
        return {"key": key_name, "model": model, "status": "active", "error": None}
    except Exception as e:
        error_msg = str(e)
        status = "error"
        if "quota" in error_msg.lower() or "429" in error_msg:
            status = "exhausted"
        logger.warning(f"Key status check [{model}]: {key_name} is {status.upper()} - {error_msg}")
        return {"key": key_name, "model": model, "status": status, "error": error_msg}


async def get_gemini_stats_message(detailed=False):
    keys = API_KEYS_MAP
    total_keys = len(keys)
    current_models = list(MODELS_LIST)

    tasks = []
    for model in current_models:
        for key_name in sorted(keys.keys()):
            api_key = keys.get(key_name)
            if api_key:
                tasks.append(check_single_key_status(key_name, api_key, model))

    results = await asyncio.gather(*tasks)

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
